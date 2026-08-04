"""Lodstone streaming adapter for ChimeraX volume rendering."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Sequence
from threading import Lock

import numpy as np
from chimerax.core.models import Model
from chimerax.map import volume_from_grid_data
from chimerax.map_data import ArrayGridData
from lodstone import Layout, Planner, Stream, TileKey, Update, View
from lodstone.sources import ArrayPyramidSource

from .map_data.constants import UNITFACTOR
from .map_data.ome_metadata import OMEZarrFormatError, parse_ome_zarr_metadata, spatial_transform_angstrom
from .map_data.zarr_grid import _apply_omero_display, _supported_matrix_type

DEFAULT_GPU_BUDGET = 512 * 1024**2


def _update_volume_now(volume) -> None:
    """Update one volume without leaving it in ChimeraX's global update sets."""

    volume.update_drawings()
    manager = getattr(volume.session, "_volume_update_manager", None)
    if manager is not None:
        manager._volumes_to_update.discard(volume)
        manager._displayed_volumes_to_update.discard(volume)


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
    """Assemble Lodstone updates into mutable ChimeraX array-backed volumes."""

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
        self.buffers = {}
        self.grids = {}
        self.volumes = {}
        self.level_keys = defaultdict(set)
        self.current_level = None
        self._value_ranges = {}

        # Establish bounds before the first camera-driven plan. This lets the
        # normal ChimeraX open/view logic position the camera around the data.
        coarsest = len(source.pyramid.levels) - 1
        self._ensure_level(coarsest)
        self._show_level(coarsest)

    def layout(self, view, pyramid) -> Layout:
        return Layout(
            kind="dense",
            # Matching native chunks avoids repeatedly assembling overlapping
            # remote slabs while ChimeraX progressively fills a dense texture.
            block_shape=None,
            mixed_lod=False,
            memory_limit=self.gpu_budget,
        )

    def apply(self, updates: Sequence[Update]) -> None:
        changed_levels = set()
        for update in updates:
            buffer, _grid, _volume = self._ensure_level(update.level)
            slices = tuple(
                slice(update.region.start[axis], update.region.stop[axis])
                for axis in self.displayed_axes
            )
            values = _supported_matrix_type(update.data)
            if len(self.displayed_axes) == 2:
                slices = (slice(0, 1), *slices)
                values = np.expand_dims(values, axis=0)
            buffer[slices] = values
            self.level_keys[update.level].add(update.key)
            changed_levels.add(update.level)
            self._observe_values(update.level, values)

        for level in changed_levels:
            self.grids[level].values_changed()
            self._update_thresholds(level)
            _update_volume_now(self.volumes[level])
        if updates:
            self._show_level(updates[-1].level)

    def discard(self, keys: Collection[TileKey]) -> None:
        discarded_by_level = defaultdict(set)
        for key in keys:
            discarded_by_level[key.level].add(key)
        for level, discarded in discarded_by_level.items():
            self.level_keys[level].difference_update(discarded)
            if not self.level_keys[level] and level in self.volumes:
                self.volumes[level].display = False

    def redraw(self) -> None:
        self.session.main_view.redraw_needed = True

    def _ensure_level(self, level: int):
        if level in self.buffers:
            return self.buffers[level], self.grids[level], self.volumes[level]

        level_info = self.source.pyramid.levels[level]
        spatial_shape = tuple(level_info.shape[axis] for axis in self.displayed_axes)
        dtype = np.float32 if level_info.dtype in (np.dtype(np.float16), np.dtype(np.uint64)) else level_info.dtype
        buffer = np.zeros(spatial_shape, dtype=dtype)

        dataset = self.metadata.multiscales.datasets[level]
        step, origin = spatial_transform_angstrom(self.metadata.multiscales, dataset)
        if len(spatial_shape) == 2:
            buffer = np.expand_dims(buffer, axis=0)
            step_xyz = (step[1], step[0], 1.0)
            origin_xyz = (origin[1], origin[0], 0.0)
        else:
            step_xyz = tuple(reversed(step))
            origin_xyz = tuple(reversed(origin))

        grid = ArrayGridData(buffer, origin=origin_xyz, step=step_xyz, name=f"{self.name} L{level}")
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

        self.buffers[level] = buffer
        self.grids[level] = grid
        self.volumes[level] = volume
        return buffer, grid, volume

    def _show_level(self, level: int) -> None:
        if self.current_level == level:
            return
        active = self._channel_active()
        for existing_level, volume in self.volumes.items():
            volume.display = active and existing_level == level
            if volume.display:
                _update_volume_now(volume)
        self.current_level = level

    def _channel_active(self) -> bool:
        omero = self.metadata.omero
        if omero and self.channel_index < len(omero.channels):
            return omero.channels[self.channel_index].active
        return self.time_index == 0

    def _observe_values(self, level: int, values: np.ndarray) -> None:
        if values.size == 0:
            return
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        previous = self._value_ranges.get(level)
        self._value_ranges[level] = (
            minimum if previous is None else min(previous[0], minimum),
            maximum if previous is None else max(previous[1], maximum),
        )

    def _update_thresholds(self, level: int) -> None:
        volume = self.volumes[level]
        omero = self.metadata.omero
        if omero and self.channel_index < len(omero.channels):
            window = omero.channels[self.channel_index].window
            if window is not None and window.end > window.start:
                volume.set_parameters(image_levels=[(window.start, 0.0), (window.end, 1.0)])
                return
        value_range = self._value_ranges.get(level)
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
        self.stream = Stream(
            source,
            target,
            planner=Planner(progressive=True),
            dispatch=dispatcher,
            workers=8,
            cpu_cache=2 * 1024**3,
            inflight=256 * 1024**2,
            batch_size=8,
        )
        self._disconnect_status = self.stream.on_status_changed(self._status_changed)
        self._handler = self.session.triggers.add_handler("graphics update", self._graphics_update)

    def _graphics_update(self, *_args) -> None:
        view, signature = self._view()
        if self._signature is not None and np.allclose(signature, self._signature, rtol=1e-7, atol=1e-7):
            return
        self._signature = signature
        plan = self.stream.update(view)
        message = (
            f"Lodstone {self.target.name}: loading {len(plan.wanted)} blocks "
            f"toward level {plan.target_level}"
        )
        self.session.logger.info(message)

    def _status_changed(self, status) -> None:
        if status.state == "failed":
            self.session.logger.warning(
                f"Lodstone {self.target.name} failed: {status.error}",  # noqa: G004
            )
        elif status.state == "complete":
            mib = status.bytes_read / 1024**2
            message = (
                f"Lodstone {self.target.name}: level ready "
                f"({status.resident} blocks, {mib:.1f} MiB read)"
            )
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
        signature = np.concatenate([world_to_clip.ravel(), np.asarray(window_size)]).astype(np.float64)
        return view, signature

    def close(self) -> None:
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

        self.dispatcher = ChimeraXDispatcher(session)
        self.controllers = []
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

    def delete(self) -> None:
        for controller in self.controllers:
            controller.close()
        self.dispatcher.close()
        super().delete()
