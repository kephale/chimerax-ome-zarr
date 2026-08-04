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

DEFAULT_GPU_BUDGET = 512 * 1024**2


def _update_volume_now(volume) -> None:
    """Update one volume without leaving it in ChimeraX's global update sets."""

    volume.update_drawings()
    manager = getattr(volume.session, "_volume_update_manager", None)
    if manager is not None:
        manager._volumes_to_update.discard(volume)
        manager._displayed_volumes_to_update.discard(volume)


def _upload_texture_3d(texture, data: np.ndarray, offset_zyx) -> None:
    """Upload one scalar ZYX subarray into an initialized ChimeraX texture."""

    from chimerax.graphics import opengl

    data = np.ascontiguousarray(data)
    texture_format, _internal_format, texture_dtype, _components = texture.texture_format(data)
    z_offset, y_offset, x_offset = offset_zyx
    depth, height, width = data.shape
    gl = opengl.GL
    target = texture.gl_target
    gl.glBindTexture(target, texture.id)
    try:
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage3D(
            target,
            0,
            x_offset,
            y_offset,
            z_offset,
            width,
            height,
            depth,
            texture_format,
            texture_dtype,
            data,
        )
    finally:
        gl.glBindTexture(target, 0)


def _patch_volume_texture(volume, buffer, window, updates, displayed_axes) -> bool:
    """Patch an existing scalar 3D texture, returning false when unsafe."""

    if buffer.ndim != 3:
        return False
    image = getattr(volume, "_image", None)
    if image is None or getattr(image, "deleted", False):
        return False
    options = getattr(image, "_rendering_options", None)
    if options is None or not options.colormap_on_gpu or getattr(image, "_blend_image", None) is not None:
        return False

    if getattr(image, "_p_mode", None) == "rays":
        drawing = getattr(image, "_volume_raycast_drawing", None)
    elif getattr(image, "_use_3d_texture", False):
        drawing = getattr(image, "_planes_3d", None)
    else:
        return False
    texture = getattr(drawing, "texture", None)
    if (
        texture is None
        or texture.id is None
        or texture.dimension != 3
        or texture.data is not None
        or texture._array_shape != tuple(buffer.shape)
        or texture._numpy_dtype != buffer.dtype
    ):
        return False

    patches = []
    for update in updates:
        starts = tuple(update.region.start[axis] - window.region.start[axis] for axis in displayed_axes)
        stops = tuple(update.region.stop[axis] - window.region.start[axis] for axis in displayed_axes)
        if any(
            start < 0 or stop > size or stop <= start
            for start, stop, size in zip(starts, stops, buffer.shape, strict=True)
        ):
            return False
        patch = buffer[tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))]
        patches.append((patch, starts))

    volume.session.main_view.render.make_current()
    for patch, offset in patches:
        _upload_texture_3d(texture, patch, offset)
    return True


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
        dtypes = [
            np.float32 if level.dtype in (np.dtype(np.float16), np.dtype(np.uint64)) else level.dtype
            for level in source.pyramid.levels
        ]
        self.resident = ResidentArrays(source.pyramid, dtypes=dtypes)
        self.resources = {}
        self.current_window = None
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
        )

    def prepare(self, view, plan) -> None:
        transition = self.resident.prepare(plan)
        for window in transition.retired:
            self._retire_window(window)
        for window in transition.prepared:
            self._ensure_window(window)
        if self.current_window is None:
            fallback = self.resident.active.get(plan.target_level)
            if fallback is not None:
                self._show_window(fallback)
            elif self._bounds_resource is not None:
                self._bounds_resource[2].display = self._channel_active()

    def apply(self, updates: Sequence[Update]) -> None:
        changes = self.resident.apply(updates)
        for change in changes:
            buffer, grid, volume = self._ensure_window(change.window)
            for update in change.updates:
                self._observe_values(change.window, update.data)
            self._update_thresholds(change.window)
            if not _patch_volume_texture(
                volume,
                buffer,
                change.window,
                change.updates,
                self.displayed_axes,
            ):
                grid.values_changed()
            _update_volume_now(volume)
            self._show_window(change.window)

    def discard(self, keys: Collection[TileKey]) -> None:
        self.resident.discard(keys)

    def complete(self, view, plan) -> None:
        transition = self.resident.complete(plan)
        target = self.resident.active.get(plan.target_level)
        if target is not None:
            self._show_window(target)
        for window in transition.retired:
            self._retire_window(window)
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
        return self._create_volume(
            np.zeros(shape, dtype=dtype),
            level,
            (0,) * len(shape),
            step_scale=step_scale,
        )

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
        volume.set_parameters(colormap_on_gpu=True)
        volume.display = False
        self.owner.add([volume])
        return buffer, grid, volume

    def _show_window(self, window: ResidentWindow) -> None:
        if self.current_window is window:
            return
        active = self._channel_active()
        if self._bounds_resource is not None:
            self._bounds_resource[2].display = False
        for existing, (_buffer, _grid, volume) in self.resources.items():
            volume.display = active and existing is window
            if volume.display:
                _update_volume_now(volume)
        self.current_window = window

    def _retire_window(self, window: ResidentWindow) -> None:
        resource = self.resources.pop(window, None)
        self._value_ranges.pop(window, None)
        if resource is not None:
            self._delete_volume(resource[2])
        if self.current_window is window:
            self.current_window = None

    @staticmethod
    def _delete_volume(volume) -> None:
        volume.display = False
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
        message = f"Lodstone {self.target.name}: loading {len(plan.wanted)} blocks toward level {plan.target_level}"
        self.session.logger.info(message)

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
