"""Lodstone streaming adapter for ChimeraX volume rendering."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence
from threading import Lock

import numpy as np
from chimerax.core.models import Model
from chimerax.map import volume_from_grid_data
from chimerax.map_data import ArrayGridData
from lodstone import (
    Layout,
    Planner,
    ResidentArrays,
    ResidentLease,
    ResidentWindow,
    Stream,
    TileKey,
    Update,
    View,
)
from lodstone.sources import ArrayPyramidSource

from .map_data.constants import UNITFACTOR
from .map_data.ome_metadata import OMEZarrFormatError, parse_ome_zarr_metadata, spatial_transform_angstrom
from .map_data.zarr_grid import _apply_omero_display

DEFAULT_GPU_BUDGET = 256 * 1024**2
CAMERA_DEBOUNCE_MS = 180
LOD_HYSTERESIS = 0.2
MAX_INITIAL_VOXEL_FOOTPRINT = 4.0


def _homogeneous_place(place) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = place.matrix
    return matrix


def _level_transform(multiscales, dataset) -> np.ndarray:
    """Build a full-axis Lodstone transform with spatial values in Angstrom."""

    ndim = len(multiscales.axes)
    transform = np.eye(ndim + 1, dtype=np.float64)
    for axis_index, axis in enumerate(multiscales.axes):
        factor = 1.0
        if axis.type == "space":
            try:
                factor = UNITFACTOR[axis.unit or "angstrom"]
            except KeyError as error:
                raise OMEZarrFormatError(
                    f"Unsupported spatial unit '{axis.unit}' on axis '{axis.name}'.",
                ) from error
        transform[axis_index, axis_index] = dataset.scale[axis_index] * factor
        transform[axis_index, ndim] = dataset.translation[axis_index] * factor
    return transform


def source_from_group(group, metadata=None) -> ArrayPyramidSource:
    """Adapt a normalized ChimeraX OME-Zarr group to a Lodstone source."""

    metadata = metadata or parse_ome_zarr_metadata(group)
    multiscales = metadata.multiscales
    arrays = [group[dataset.path] for dataset in multiscales.datasets]
    transforms = [_level_transform(multiscales, dataset) for dataset in multiscales.datasets]
    return ArrayPyramidSource(
        arrays,
        axes=tuple(axis.name for axis in multiscales.axes),
        transforms=transforms,
        chunks=[tuple(int(value) for value in array.chunks) for array in arrays],
    )


class ChimeraXDispatcher:
    """Drain Lodstone target calls on ChimeraX's graphics thread."""

    def __init__(self, session) -> None:
        self._callbacks = deque()
        self._lock = Lock()
        self._handler = session.triggers.add_handler("graphics update", self._graphics_update)

    def __call__(self, callback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def _graphics_update(self, *_args) -> None:
        while True:
            with self._lock:
                if not self._callbacks:
                    return
                callback = self._callbacks.popleft()
            callback()

    def close(self) -> None:
        if self._handler is not None:
            self._handler.remove()
            self._handler = None


class ChimeraXVolumeTarget:
    """Render bounded Lodstone resident windows as ChimeraX volumes."""

    def __init__(
        self,
        owner,
        source,
        metadata,
        displayed_axes,
        *,
        name: str,
        channel_index: int,
        time_index: int,
        gpu_budget: int,
    ) -> None:
        self.owner = owner
        self.session = owner.session
        self.source = source
        self.metadata = metadata
        self.displayed_axes = tuple(displayed_axes)
        self.name = name
        self.channel_index = channel_index
        self.time_index = time_index
        self.gpu_budget = gpu_budget
        self.max_texture_extent = self._maximum_texture_extent()
        dtypes = [
            np.float32 if level.dtype in (np.dtype(np.float16), np.dtype(np.uint64)) else level.dtype
            for level in source.pyramid.levels
        ]
        self.resident = ResidentArrays(source.pyramid, dtypes=dtypes, compose=True)
        self.resources = {}
        self.current_window = None
        self._front_resource = None
        self._back_resources = {}
        self._published_windows = set()
        self._deferred_retired = set()
        self._value_ranges = {}

        # Establish bounds before the first camera-driven plan. This lets the
        # normal ChimeraX open/view logic position the camera around the data.
        coarsest = len(source.pyramid.levels) - 1
        self._bounds_resource = self._create_bounds_volume(coarsest)
        self._bounds_resource[2].display = self._channel_active()

    def layout(self, view, pyramid) -> Layout:
        return Layout(
            kind="dense",
            # Matching native chunks avoids repeatedly assembling overlapping
            # remote slabs while ChimeraX progressively fills a dense texture.
            block_shape=None,
            mixed_lod=False,
            memory_limit=self.gpu_budget,
            squeeze_hidden=False,
            max_axis_extent=self.max_texture_extent,
        )

    def prepare(self, view, plan):
        transition = self.resident.prepare(plan)
        for window in transition.retired:
            if window is self.current_window:
                self._deferred_retired.add(window)
            else:
                self._retire_window(window)
        for window in transition.prepared:
            self._ensure_window(window)
        if self.current_window is None:
            fallback = self.resident.active.get(plan.target_level)
            if fallback is not None:
                self._show_window(fallback)
            elif self._bounds_resource is not None:
                self._bounds_resource[2].display = self._channel_active()
        desired = plan.desired or plan.wanted
        return ResidentLease(self.resident, frozenset(tile.key for tile in desired))

    def apply(self, updates: Sequence[Update]) -> None:
        # ResidentArrays is mutable target state. Keep its writes behind
        # Stream's stale-generation check on the ChimeraX thread.
        changes = self.resident.apply(updates)
        for change in changes:
            buffer, _grid, _volume = self._ensure_window(change.window)
            for region in change.regions:
                self._observe_values(
                    change.window,
                    self._buffer_region(buffer, change.window, region),
                )

    def phase_complete(self, view, plan, phase: int) -> None:
        levels = {tile.level for tile in plan.desired if tile.phase == phase}
        for level in sorted(levels):
            window = self.resident.windows.get(level)
            if window is not None and window.key_regions:
                buffer, _grid, old_volume = self._ensure_window(window)
                _snapshot, grid, volume = self._create_volume(
                    buffer.copy(),
                    window.level,
                    tuple(window.region.start[axis] for axis in self.displayed_axes),
                )
                # Keep observing Lodstone's mutable resident array, but expose
                # only the completed immutable snapshot to ChimeraX.
                self.resources[window] = (buffer, grid, volume)
                if not self._is_front_volume(old_volume):
                    self._delete_volume(old_volume)
                self._link_channel_volumes(window.level)
                self._update_thresholds(window)
                self._schedule_snapshot_publish(window, volume)

    def _maximum_texture_extent(self):
        render = getattr(getattr(self.session, "main_view", None), "render", None)
        if render is None:
            return None
        try:
            render.make_current()
            return int(render.max_3d_texture_size())
        except (AttributeError, RuntimeError, ValueError):
            return None

    def _buffer_region(self, buffer, window: ResidentWindow, region):
        slices = tuple(
            slice(
                region.start[axis] - window.region.start[axis],
                region.stop[axis] - window.region.start[axis],
            )
            for axis in self.displayed_axes
        )
        return buffer[slices]

    def discard(self, keys: Collection[TileKey]) -> None:
        self.resident.discard(keys)

    def complete(self, view, plan) -> None:
        transition = self.resident.complete(plan)
        for window in transition.retired:
            if window is self.current_window:
                self._deferred_retired.add(window)
            else:
                self._retire_window(window)
        target = self.resident.active.get(plan.target_level)
        if target is not None:
            presenter = getattr(self.owner, "present_lodstone_level", None)
            if presenter is None:
                self._show_window(target)
                self._flush_deferred_retired()
            else:
                presenter(target.level)
        if self._bounds_resource is not None:
            self._delete_volume(self._bounds_resource[2])
            self._bounds_resource = None

    def redraw(self) -> None:
        self.session.main_view.redraw_needed = True

    def _ensure_window(self, window: ResidentWindow):
        if window in self.resources:
            return self.resources[window]
        selection = tuple(slice(None) if axis in self.displayed_axes else 0 for axis in range(window.region.ndim))
        buffer = window.data[selection]
        spatial_start = tuple(window.region.start[axis] for axis in self.displayed_axes)
        resource = self._create_volume(buffer, window.level, spatial_start)
        self.resources[window] = resource
        return resource

    def _create_bounds_volume(self, level: int):
        info = self.source.pyramid.levels[level]
        full_shape = tuple(info.shape[axis] for axis in self.displayed_axes)
        shape = tuple(min(size, 2) for size in full_shape)
        step_scale = tuple(
            (full_size - 1) / (size - 1) if size > 1 else 1.0 for full_size, size in zip(full_shape, shape, strict=True)
        )
        dtype = self.resident.dtypes[level]
        resource = self._create_volume(
            np.zeros(shape, dtype=dtype),
            level,
            (0,) * len(shape),
            step_scale=step_scale,
        )
        resource[2].update_drawings()
        return resource

    def _create_volume(
        self,
        buffer,
        level: int,
        spatial_start,
        *,
        step_scale=None,
    ):
        dataset = self.metadata.multiscales.datasets[level]
        step, origin = spatial_transform_angstrom(self.metadata.multiscales, dataset)
        if step_scale is not None:
            step = tuple(value * factor for value, factor in zip(step, step_scale, strict=True))
        origin = tuple(value + spatial_start[axis] * step[axis] for axis, value in enumerate(origin))
        if len(self.displayed_axes) == 2:
            buffer = np.expand_dims(buffer, axis=0)
            step_xyz = (step[1], step[0], 1.0)
            origin_xyz = (origin[1], origin[0], 0.0)
        else:
            step_xyz = tuple(reversed(step))
            origin_xyz = tuple(reversed(origin))

        grid = ArrayGridData(buffer, origin=origin_xyz, step=step_xyz, name=f"{self.name} L{level}")
        grid.channel = self.channel_index
        grid.time = self.time_index
        volume = volume_from_grid_data(
            grid,
            self.session,
            style="image",
            open_model=False,
            show_dialog=False,
        )
        volume.name = f"{self.name} L{level}"
        _apply_omero_display(
            volume,
            self.metadata.omero,
            self.channel_index,
            self.time_index,
            "channel" in [axis.type for axis in self.metadata.multiscales.axes],
            "time" in [axis.type for axis in self.metadata.multiscales.axes],
            self.name,
        )
        volume.display = False
        self.owner.add([volume])
        return buffer, grid, volume

    def _link_channel_volumes(self, level: int) -> None:
        volumes = []
        for controller in getattr(self.owner, "controllers", ()):
            for window, (_buffer, _grid, volume) in controller.target.resources.items():
                if window.level == level and not volume.deleted:
                    volumes.append(volume)
                    break
        if len(volumes) > 1:
            from chimerax.map.volume import MapChannels

            MapChannels(sorted(volumes, key=lambda volume: volume.data.channel))

    def _schedule_snapshot_publish(self, window: ResidentWindow, volume) -> None:
        def publish() -> None:
            resource = self.resources.get(window)
            if volume.deleted or resource is None or resource[2] is not volume:
                return
            # Texture creation during a graphics-update trigger can disturb
            # that frame on older ChimeraX Dailies. Run between frames instead.
            volume.update_drawings()
            previous = self._back_resources.get(window.level)
            if previous is not None and previous[1][2] is not volume:
                self._delete_volume(previous[1][2])
            self._back_resources[window.level] = (window, resource)
            self._published_windows.add(window)
            presenter = getattr(self.owner, "present_lodstone_level", None)
            if presenter is None:
                retired = self._activate_back(window.level)
                if retired is not None:
                    self._delete_volume(retired[2])
            else:
                presenter(window.level)
            self.session.main_view.redraw_needed = True

        ui = getattr(self.session, "ui", None)
        if ui is None or not ui.is_gui:
            publish()
            return
        timers = getattr(self.session, "_lodstone_publish_timers", [])
        timers.append(ui.timer(0, publish))
        self.session._lodstone_publish_timers = timers

    def _show_window(self, window: ResidentWindow) -> None:
        retired = self._activate_back(window.level)
        if retired is not None:
            self._delete_volume(retired[2])
            return
        resource = self.resources.get(window)
        if resource is None or self._front_resource == (window, resource):
            return
        active = self._channel_active()
        if self._bounds_resource is not None:
            # The proxy has done its job once a completed resident volume can
            # supply real bounds. Removing it also makes ChimeraX recompute
            # clipping/drawings for the newly visible child on older Dailies.
            self._delete_volume(self._bounds_resource[2])
            self._bounds_resource = None
        for existing, (_buffer, _grid, volume) in self.resources.items():
            volume.display = active and existing is window
        self.current_window = window
        self._front_resource = (window, resource)

    def _activate_back(self, level: int):
        """Atomically select an initialized back buffer, returning the old front."""

        candidate = self._back_resources.pop(level, None)
        if candidate is None:
            return None
        window, resource = candidate
        old = None if self._front_resource is None else self._front_resource[1]
        active = self._channel_active()
        resource[2].display = active
        if old is not None and old is not resource:
            old[2].display = False
        if self._bounds_resource is not None:
            self._delete_volume(self._bounds_resource[2])
            self._bounds_resource = None
        self.current_window = window
        self._front_resource = candidate
        return old if old is not resource else None

    def _is_front_volume(self, volume) -> bool:
        return self._front_resource is not None and self._front_resource[1][2] is volume

    def _retire_window(self, window: ResidentWindow) -> None:
        resource = self.resources.pop(window, None)
        for level, candidate in tuple(self._back_resources.items()):
            if candidate[0] is window:
                self._back_resources.pop(level, None)
        self._published_windows.discard(window)
        self._value_ranges.pop(window, None)
        if resource is not None and not self._is_front_volume(resource[2]):
            self._delete_volume(resource[2])
        if self.current_window is window:
            self.current_window = None
            self._front_resource = None

    def _flush_deferred_retired(self) -> None:
        for window in tuple(self._deferred_retired):
            if window is not self.current_window:
                self._deferred_retired.discard(window)
                self._retire_window(window)

    @staticmethod
    def _delete_volume(volume) -> None:
        volume.display = False
        manager = getattr(volume.session, "_volume_update_manager", None)
        if manager is not None:
            # ChimeraX Daily builds before 2026-08-17 can retain a hidden or
            # deleted volume in only one of these two companion sets, then
            # raise KeyError on the next graphics update.
            manager._volumes_to_update.discard(volume)
            manager._displayed_volumes_to_update.discard(volume)
        volume.delete()

    def _channel_active(self) -> bool:
        omero = self.metadata.omero
        if omero and self.channel_index < len(omero.channels):
            return omero.channels[self.channel_index].active
        return self.time_index == 0

    def _observe_values(self, window: ResidentWindow, values: np.ndarray) -> None:
        if values.size == 0:
            return
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        previous = self._value_ranges.get(window)
        self._value_ranges[window] = (
            minimum if previous is None else min(previous[0], minimum),
            maximum if previous is None else max(previous[1], maximum),
        )

    def _update_thresholds(self, window: ResidentWindow) -> None:
        volume = self.resources[window][2]
        omero = self.metadata.omero
        if omero and self.channel_index < len(omero.channels):
            display_window = omero.channels[self.channel_index].window
            if display_window is not None and display_window.end > display_window.start:
                volume.set_parameters(
                    image_levels=[
                        (display_window.start, 0.0),
                        (display_window.end, 1.0),
                    ],
                )
                return
        value_range = self._value_ranges.get(window)
        if value_range is not None and value_range[1] > value_range[0]:
            volume.set_parameters(image_levels=[(value_range[0], 0.0), (value_range[1], 1.0)])


class LodstoneVolumeController:
    """Translate the active ChimeraX camera into Lodstone view snapshots."""

    def __init__(
        self,
        owner,
        source,
        target,
        index,
        displayed_axes,
        dispatcher,
    ) -> None:
        self.owner = owner
        self.session = owner.session
        self.source = source
        self.target = target
        self.index = tuple(index)
        self.displayed_axes = tuple(displayed_axes)
        self.dispatcher = dispatcher
        self._signature = None
        self._target_level = None
        self._active_coverage = None
        self._pending_view = None
        self._debounce_timer = None
        self.stream = Stream(
            source,
            target,
            planner=Planner(
                progressive=True,
                max_intermediate_levels=0,
                max_initial_voxel_footprint=MAX_INITIAL_VOXEL_FOOTPRINT,
            ),
            dispatch=dispatcher,
            workers=8,
            cpu_cache=max(2 * target.gpu_budget, 256 * 1024**2),
            inflight=min(target.gpu_budget, 128 * 1024**2),
            batch_size=8,
        )
        self._disconnect_status = self.stream.on_status_changed(self._status_changed)
        self._handler = self.session.triggers.add_handler("graphics update", self._graphics_update)

    def _graphics_update(self, *_args) -> None:
        # The open command constructs the model before adding it to the
        # session and fitting the camera to its bounds. Planning against that
        # transient camera can request an unnecessarily fine full volume and
        # delay creation of the main window.
        if self.owner.id is None:
            return
        view, signature = self._view()
        if self._signature is not None and np.allclose(signature, self._signature, rtol=1e-7, atol=1e-7):
            return
        self._signature = signature
        if self._target_level is None:
            self._submit_view(view)
            return

        self._pending_view = view
        timer = self._debounce_timer
        if timer is not None:
            timer.stop()
        ui = getattr(self.session, "ui", None)
        if ui is None or not ui.is_gui:
            self._submit_pending_view()
        else:
            self._debounce_timer = ui.timer(CAMERA_DEBOUNCE_MS, self._submit_pending_view)

    def _submit_pending_view(self) -> None:
        view = self._pending_view
        self._pending_view = None
        self._debounce_timer = None
        if view is not None and not self.owner.deleted:
            self._submit_view(view)

    def _submit_view(self, view: View) -> None:
        plan = self.stream.plan(
            view,
            previous_target_level=self._target_level,
            lod_hysteresis=LOD_HYSTERESIS,
        )
        if not plan.desired:
            self.session.logger.status(
                f"Lodstone {self.target.name}: waiting for a valid fitted view",
                blank_after=3,
            )
            return
        if plan.coverage == self._active_coverage:
            return
        self._target_level = plan.target_level
        self.stream.submit(view, plan)
        self._active_coverage = plan.coverage
        message = f"Lodstone {self.target.name}: loading {len(plan.wanted)} blocks toward level {plan.target_level}"
        self.session.logger.status(message, blank_after=3)

    def _status_changed(self, status) -> None:
        if status.state == "failed":
            self.session.logger.warning(
                f"Lodstone {self.target.name} failed: {status.error}",  # noqa: G004
            )
        elif status.state == "complete":
            mib = status.bytes_read / 1024**2
            message = f"Lodstone {self.target.name}: level ready ({status.resident} blocks, {mib:.1f} MiB read)"
            self.session.logger.info(message)

    def _view(self):
        main_view = self.session.main_view
        camera = main_view.camera
        window_size = tuple(int(value) for value in main_view.window_size)
        near_far = main_view.near_far_distances(camera, 0)
        projection = np.asarray(camera.projection_matrix(near_far, 0, window_size), dtype=np.float64)
        scene_to_camera = np.asarray(camera.position.inverse().opengl_matrix(), dtype=np.float64).T
        clip_from_scene = projection.T @ scene_to_camera

        # Lodstone coordinates are ordered like displayed OME axes (normally
        # Z,Y,X), whereas ChimeraX scene coordinates are X,Y,Z.
        xyz_from_local = np.eye(4, dtype=np.float64)
        xyz_from_local[:3, :3] = 0
        for local_axis in range(len(self.displayed_axes)):
            xyz_axis = len(self.displayed_axes) - 1 - local_axis
            xyz_from_local[xyz_axis, local_axis] = 1

        scene_from_xyz = _homogeneous_place(self.owner.scene_position)
        world_to_clip = clip_from_scene @ scene_from_xyz @ xyz_from_local

        eye_scene = np.asarray(camera.position.origin(), dtype=np.float64)
        eye_xyz = np.linalg.solve(scene_from_xyz, np.append(eye_scene, 1.0))[:3]
        eye_local = tuple(reversed(eye_xyz))
        view = View(
            displayed_axes=self.displayed_axes,
            index=self.index,
            viewport=window_size,
            world_to_clip=world_to_clip,
            eye=eye_local,
        )
        # ChimeraX recomputes near/far clipping distances as streamed child
        # volumes are shown. Those values affect world_to_clip but are not a
        # user view change; including them here caused an endless sequence of
        # canceled generations. Track camera/model transforms and projection
        # intrinsics instead.
        intrinsics = [
            float(getattr(camera, name)) for name in ("field_of_view", "field_width") if hasattr(camera, name)
        ]
        signature = np.concatenate(
            [
                np.asarray(camera.position.matrix, dtype=np.float64).ravel(),
                np.asarray(self.owner.scene_position.matrix, dtype=np.float64).ravel(),
                np.asarray(window_size, dtype=np.float64),
                np.asarray(intrinsics, dtype=np.float64),
            ],
        )
        return view, signature

    def close(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None
        if self._handler is not None:
            self._handler.remove()
            self._handler = None
        self._disconnect_status()
        self.stream.close()


class LodstoneZarrModel(Model):
    """Camera-driven progressive OME-Zarr model backed by Lodstone."""

    def __init__(
        self,
        name: str,
        session,
        group,
        *,
        gpu_budget: int = DEFAULT_GPU_BUDGET,
    ) -> None:
        super().__init__(name, session)
        self.dispatcher = None
        self.controllers = []
        self.group = group
        self.ome_zarr_metadata = metadata = parse_ome_zarr_metadata(group)
        self.source = source_from_group(group, metadata)
        multiscales = metadata.multiscales
        displayed_axes = multiscales.spatial_indices
        axes_types = [axis.type for axis in multiscales.axes]
        time_axis = axes_types.index("time") if "time" in axes_types else None
        channel_axis = axes_types.index("channel") if "channel" in axes_types else None
        finest_shape = self.source.pyramid.levels[0].shape
        time_count = finest_shape[time_axis] if time_axis is not None else 1
        channel_count = finest_shape[channel_axis] if channel_axis is not None else 1
        if time_count > 1:
            raise OMEZarrFormatError(
                "Lodstone streaming does not yet support switching timepoints; "
                "open this time series without 'streaming true'.",
            )

        self.dispatcher = ChimeraXDispatcher(session)
        for time_index in range(time_count):
            for channel_index in range(channel_count):
                index = [None] * len(multiscales.axes)
                if time_axis is not None:
                    index[time_axis] = time_index
                if channel_axis is not None:
                    index[channel_axis] = channel_index
                target = ChimeraXVolumeTarget(
                    self,
                    self.source,
                    metadata,
                    displayed_axes,
                    name=name,
                    channel_index=channel_index,
                    time_index=time_index,
                    gpu_budget=gpu_budget,
                )
                self.controllers.append(
                    LodstoneVolumeController(
                        self,
                        self.source,
                        target,
                        index,
                        displayed_axes,
                        self.dispatcher,
                    ),
                )

    @property
    def scales(self):
        return [dataset.path for dataset in self.ome_zarr_metadata.multiscales.datasets]

    def present_lodstone_level(self, level: int) -> None:
        candidates = []
        for controller in self.controllers:
            candidate = controller.target._back_resources.get(level)
            if candidate is None:
                return
            candidates.append((controller.target, candidate[0]))
        retired = []
        for target, _window in candidates:
            old = target._activate_back(level)
            if old is not None:
                retired.append((target, old))
        # Every new channel is selected before any old front is destroyed.
        for target, resource in retired:
            target._delete_volume(resource[2])
        for target, _window in candidates:
            target._flush_deferred_retired()
        self.session.main_view.redraw_needed = True

    def delete(self) -> None:
        for controller in self.controllers:
            controller.close()
        if self.dispatcher is not None:
            self.dispatcher.close()
        super().delete()
