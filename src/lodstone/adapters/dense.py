"""Renderer-neutral dense host contract and progressive controller."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from itertools import product
from threading import Condition, Thread
from time import monotonic
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np

from ..model import Layout, Plan, Pyramid, Region, TileKey, Update, View
from ..planner import Planner
from ..resident import ResidentArrays, ResidentLease
from ..runtime import Runtime
from ..source import Source
from ..stream import Stream

__all__ = [
    "CameraDenseCanvas",
    "CameraSignal",
    "DenseCanvas",
    "DenseController",
    "DenseHandle",
    "DensePreparation",
    "DensePublication",
    "DenseTarget",
]


@runtime_checkable
class DenseHandle(Protocol):
    """A renderer-owned image or volume populated by a dense target."""

    def set_data(self, data: np.ndarray) -> None: ...

    def set_world_transform(
        self,
        scales: tuple[float, ...],
        origins: tuple[float, ...],
    ) -> None: ...

    def set_clims(self, clims: tuple[float, float]) -> None: ...

    def set_visible(self, visible: bool) -> None: ...

    def remove(self) -> None: ...


@runtime_checkable
class DenseCanvas(Protocol):
    """Minimum host surface required for dense 2-D and 3-D publication."""

    def set_ndim(self, ndim: int) -> None: ...

    def add_image(
        self, data: np.ndarray, *, reset_range: bool = False
    ) -> DenseHandle: ...

    def add_volume(
        self, data: np.ndarray, *, reset_range: bool = False
    ) -> DenseHandle: ...

    def set_range(self) -> None: ...

    def refresh(self) -> None: ...


@runtime_checkable
class CameraSignal(Protocol):
    """Connectable no-argument camera-change signal."""

    def connect(self, callback: Callable[[], None]) -> None: ...

    def disconnect(self, callback: Callable[[], None]) -> None: ...


@runtime_checkable
class CameraDenseCanvas(DenseCanvas, Protocol):
    """Optional dense canvas extension supporting automatic replanning."""

    cameraChanged: CameraSignal

    def camera_state(self) -> tuple[tuple[int, int], np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class DensePublication:
    """Immutable dense phase snapshot ready for a dense host image handle."""

    level: int
    region: Region
    data: np.ndarray
    data_axes: tuple[int, ...]
    scales: tuple[float, ...]
    origins: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DensePreparation:
    desired_keys: frozenset[TileKey]


class DenseTarget:
    """Present complete dense phases through a renderer-neutral ``DenseCanvas``.

    The target only relies on the dense canvas and image-handle protocols, so the
    same adapter works across renderer backends. Each pyramid level is a separate
    image handle so a coarse context can remain visible around the camera-focused
    high-resolution window.
    """

    def __init__(
        self,
        canvas: DenseCanvas,
        pyramid: Pyramid,
        *,
        memory_limit: int = 64 * 1024**2,
        block_shape: tuple[int, ...] = (64, 128, 128),
        focus_depth_weight: float = 0.5,
        focus_depth_target: float | None = None,
        on_presented: Callable[[DensePublication], None] | None = None,
    ) -> None:
        self.canvas = canvas
        self.pyramid = pyramid
        self.memory_limit = int(memory_limit)
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        if not np.isfinite(focus_depth_weight) or focus_depth_weight < 0:
            raise ValueError("focus_depth_weight must be finite and nonnegative")
        if focus_depth_target is not None and (
            not np.isfinite(focus_depth_target) or not 0 <= focus_depth_target <= 1
        ):
            raise ValueError("focus_depth_target must be between zero and one")
        self.block_shape = tuple(int(value) for value in block_shape)
        self.focus_depth_weight = float(focus_depth_weight)
        self.focus_depth_target = focus_depth_target
        self.resident = ResidentArrays(pyramid, compose=True)
        self.handles: dict[int, DenseHandle] = {}
        self.publications: dict[int, DensePublication] = {}
        self.handle: DenseHandle | None = None
        self._context_level = len(pyramid.levels) - 1
        self._masked_focus: tuple[int, Region] | None = None
        self.on_presented = on_presented

    def layout(self, view: View, pyramid: Pyramid) -> Layout:
        return Layout(
            kind="dense",
            block_shape=self.block_shape[-len(view.displayed_axes) :],
            mixed_lod=True,
            memory_limit=self.memory_limit,
            squeeze_hidden=False,
            max_axis_extent=512,
            memory_policy="crop",
            focus_depth_weight=self.focus_depth_weight,
            focus_depth_target=self.focus_depth_target,
        )

    def stage_prepare(self, view: View, plan: Plan) -> DensePreparation:
        self.resident.prepare(plan)
        desired = plan.desired or plan.wanted
        return DensePreparation(frozenset(tile.key for tile in desired))

    def prepare(
        self,
        view: View,
        plan: Plan,
        prepared: DensePreparation,
    ) -> ResidentLease:
        return ResidentLease(self.resident, prepared.desired_keys)

    def stage(self, updates: Sequence[Update]):
        return self.resident.apply(updates)

    def apply(self, updates: Sequence[Update]) -> None:
        # Resident writes occur in ``stage``. The host receives one immutable
        # array only when the corresponding phase is complete.
        return None

    def stage_phase(
        self,
        view: View,
        plan: Plan,
        phase: int,
    ) -> DensePublication:
        desired = plan.desired or plan.wanted
        levels = {tile.level for tile in desired if tile.phase == phase}
        if len(levels) != 1:
            raise RuntimeError("a dense phase must describe exactly one pyramid level")
        level = levels.pop()
        window = self.resident.windows[level]
        data_axes = tuple(
            axis for axis in range(window.region.ndim) if axis in view.displayed_axes
        )
        hidden = tuple(
            axis
            for axis in range(window.region.ndim)
            if axis not in view.displayed_axes
        )
        if any(window.data.shape[axis] != 1 for axis in hidden):
            raise RuntimeError("hidden axes must be singleton resident selections")
        data = np.squeeze(window.data, axis=hidden).copy()
        data.setflags(write=False)
        transform = self.pyramid.levels[level].voxel_to_world
        scales = tuple(
            float(np.linalg.norm(transform[:-1, axis])) for axis in data_axes
        )
        world_origin = transform[:-1, :-1] @ np.asarray(window.region.start)
        world_origin += transform[:-1, -1]
        origins = tuple(float(world_origin[axis]) for axis in data_axes)
        return DensePublication(level, window.region, data, data_axes, scales, origins)

    def phase_complete(
        self,
        view: View,
        plan: Plan,
        phase: int,
        publication: DensePublication,
    ) -> None:
        ndim = len(view.displayed_axes)
        self.canvas.set_ndim(ndim)
        wants_level = any(tile.level == publication.level for tile in plan.wanted)
        previous = self.publications.get(publication.level)
        if (
            publication.level == self._context_level
            and not wants_level
            and previous is not None
            and previous.region == publication.region
        ):
            # Stream phases also complete when every context tile was cached.
            # Keep the existing visual instead of uploading the same array.
            return
        handle = self.handles.get(publication.level)
        first_publication = not self.handles
        if handle is None:
            factory = self.canvas.add_image if ndim == 2 else self.canvas.add_volume
            handle = factory(publication.data, reset_range=False)
            self.handles[publication.level] = handle
        else:
            handle.set_data(publication.data)
        handle.set_clims(_data_clims(publication.data))
        handle.set_world_transform(publication.scales, publication.origins)
        handle.set_visible(True)
        self.publications[publication.level] = publication
        if publication.level == self._context_level:
            self._masked_focus = None
        self.handle = handle
        if first_publication:
            self.canvas.set_range()
        if publication.level != self._context_level:
            self._mask_context(publication)
        self.canvas.refresh()
        if self.on_presented is not None:
            self.on_presented(publication)

    def discard(self, keys: Collection[TileKey]) -> None:
        self.resident.discard(keys)

    def complete(self, view: View, plan: Plan) -> None:
        self.resident.complete(plan, retain_levels=True)
        keep = {self._context_level, plan.target_level}
        for level in tuple(self.handles):
            if level not in keep:
                self.handles.pop(level).remove()
                self.publications.pop(level, None)
        context = self.publications.get(self._context_level)
        if context is not None:
            if plan.target_level == self._context_level:
                self.handles[self._context_level].set_data(context.data)
                self._masked_focus = None
            elif (focus := self.publications.get(plan.target_level)) is not None:
                self._mask_context(focus)

    def _mask_context(self, focus: DensePublication) -> None:
        """Punch the fine focus footprint out of the additive coarse visual."""
        context = self.publications.get(self._context_level)
        handle = self.handles.get(self._context_level)
        identity = (focus.level, focus.region)
        if context is None or handle is None or self._masked_focus == identity:
            return
        overlap = _region_in_level(
            focus.region,
            self.pyramid.levels[focus.level].voxel_to_world,
            self.pyramid.levels[self._context_level].voxel_to_world,
        ).intersection(context.region)
        masked = context.data.copy()
        if overlap is not None:
            if np.count_nonzero(focus.data) == focus.data.size:
                slices = tuple(
                    slice(
                        overlap.start[axis] - context.region.start[axis],
                        overlap.stop[axis] - context.region.start[axis],
                    )
                    for axis in context.data_axes
                )
                masked[slices] = 0
            else:
                _mask_occupied_context(
                    masked,
                    context,
                    focus,
                    self.pyramid.levels[focus.level].voxel_to_world,
                    self.pyramid.levels[self._context_level].voxel_to_world,
                )
        handle.set_data(masked)
        handle.set_world_transform(context.scales, context.origins)
        self._masked_focus = identity

    def redraw(self) -> None:
        self.canvas.refresh()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()
        self.publications.clear()
        self.handle = None
        self.resident.clear()


def _region_in_level(
    region: Region,
    source_to_world: np.ndarray,
    destination_to_world: np.ndarray,
) -> Region:
    """Return a conservative destination-level box for ``region``."""
    ndim = region.ndim
    destination_from_source = np.linalg.solve(destination_to_world, source_to_world)
    corners = np.asarray(
        [(*corner, 1.0) for corner in product(*zip(region.start, region.stop))],
        dtype=np.float64,
    )
    mapped = (destination_from_source @ corners.T).T[:, :ndim]
    start = tuple(max(0, int(np.floor(value))) for value in mapped.min(axis=0))
    stop = tuple(max(0, int(np.ceil(value))) for value in mapped.max(axis=0))
    return Region(start, stop)


def _data_clims(data: np.ndarray) -> tuple[float, float]:
    """Return finite nondegenerate limits independent of renderer defaults."""
    if np.issubdtype(data.dtype, np.inexact):
        finite = data[np.isfinite(data)]
        if not finite.size:
            return (0.0, 1.0)
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    else:
        lower = float(np.min(data))
        upper = float(np.max(data))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def _mask_occupied_context(
    masked: np.ndarray,
    context: DensePublication,
    focus: DensePublication,
    focus_to_world: np.ndarray,
    context_to_world: np.ndarray,
) -> None:
    """Mask context samples covered by nonzero fine data.

    A dense focus publication can contain a great deal of empty space. Masking
    its complete box removes useful coarse landmarks in sparse volumes, while
    masking only occupied fine samples still prevents additive brightening for
    dense data. Work one leading-axis plane at a time to bound temporary memory.
    """
    context_from_focus = np.linalg.solve(context_to_world, focus_to_world)
    ndim = focus.region.ndim
    base = np.asarray(focus.region.start, dtype=np.float64) + 0.5
    data = focus.data
    for leading in range(data.shape[0]):
        occupied = np.argwhere(data[leading] != 0)
        if not len(occupied):
            continue
        local = np.column_stack(
            (np.full(len(occupied), leading, dtype=np.int64), occupied)
        )
        points = np.broadcast_to(base, (len(local), ndim)).copy()
        for local_axis, data_axis in enumerate(focus.data_axes):
            points[:, data_axis] += local[:, local_axis]
        homogeneous = np.column_stack((points, np.ones(len(points))))
        mapped = (context_from_focus @ homogeneous.T).T[:, :ndim]
        indices = np.floor(mapped).astype(np.int64)
        context_indices = np.column_stack(
            [
                indices[:, axis] - context.region.start[axis]
                for axis in context.data_axes
            ]
        )
        valid = np.ones(len(context_indices), dtype=bool)
        for axis, size in enumerate(masked.shape):
            valid &= (context_indices[:, axis] >= 0) & (context_indices[:, axis] < size)
        if np.any(valid):
            masked[tuple(context_indices[valid].T)] = 0


class DenseController:
    """Connect a Lodstone source to a dense viewer or canvas.

    The initial :class:`View` snapshot establishes displayed and selected axes.
    When the canvas exposes the dense camera API, later camera changes are captured,
    debounced, and replanned automatically without coupling the target to either
    the VisPy or pygfx backend.
    """

    def __init__(
        self,
        viewer_or_canvas: Any,
        source: Source,
        *,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        runtime: Runtime | None = None,
        memory_limit: int = 64 * 1024**2,
        block_shape: tuple[int, ...] = (64, 128, 128),
        focus_depth_weight: float = 0.5,
        focus_depth_target: float | None = None,
        on_presented: Callable[[DensePublication], None] | None = None,
        on_targeted: Callable[[Plan], None] | None = None,
        camera_debounce_ms: int = 180,
        **stream_options: Any,
    ) -> None:
        self.viewer = viewer_or_canvas if hasattr(viewer_or_canvas, "canvas") else None
        self.canvas = (
            viewer_or_canvas.canvas if self.viewer is not None else viewer_or_canvas
        )
        if dispatch is None:
            dispatch = getattr(viewer_or_canvas, "dispatch", None)
        self._dispatch = dispatch or (lambda callback: callback())
        self._on_targeted = on_targeted
        self._camera_debounce_seconds = camera_debounce_ms / 1000
        if self._camera_debounce_seconds < 0:
            raise ValueError("camera_debounce_ms must be nonnegative")
        self._camera_condition = Condition()
        self._pending_camera_view: View | None = None
        self._camera_deadline = 0.0
        self._camera_generation = 0
        self._closed = False
        self._presenting = False
        self.runtime = runtime or Runtime()
        self._owns_runtime = runtime is None
        self.target = self._make_target(
            self.canvas,
            source.pyramid,
            memory_limit=memory_limit,
            block_shape=block_shape,
            focus_depth_weight=focus_depth_weight,
            focus_depth_target=focus_depth_target,
            on_presented=on_presented,
        )
        stream_options.setdefault(
            "planner",
            Planner(
                lod_bias=2.0,
                progressive=True,
                max_intermediate_levels=0,
                max_initial_voxel_footprint=4.0,
            ),
        )
        self.stream = Stream(
            source,
            self.target,
            dispatch=self._dispatch_presentation,
            runtime=self.runtime,
            **stream_options,
        )
        self._last_view: View | None = None
        self._camera_signal = getattr(self.canvas, "cameraChanged", None)
        if self._camera_signal is not None:
            self._camera_signal.connect(self._camera_changed)
        self._camera_thread = Thread(
            target=self._camera_worker,
            name="lodstone-dense-camera",
            daemon=True,
        )
        self._camera_thread.start()

    def _make_target(
        self,
        canvas: DenseCanvas,
        pyramid: Pyramid,
        **options: Any,
    ) -> DenseTarget:
        """Build the dense target, allowing a host to specialize publication."""
        return DenseTarget(canvas, pyramid, **options)

    def update(self, view: View) -> Plan:
        self._last_view = view
        plan = self.stream.update(view)
        self._notify_targeted(plan)
        return plan

    def set_focus_policy(
        self,
        *,
        memory_limit: int | None = None,
        focus_depth_weight: float | None = None,
        focus_depth_target: float | None = None,
        lod_bias: float | None = None,
        replan: bool = True,
    ) -> Plan | None:
        """Update interactive focus controls and optionally replan the view."""
        if memory_limit is not None:
            if memory_limit <= 0:
                raise ValueError("memory_limit must be positive")
            self.target.memory_limit = int(memory_limit)
        if focus_depth_weight is not None:
            if not np.isfinite(focus_depth_weight) or focus_depth_weight < 0:
                raise ValueError("focus_depth_weight must be finite and nonnegative")
            self.target.focus_depth_weight = float(focus_depth_weight)
        if focus_depth_target is not None:
            if not np.isfinite(focus_depth_target) or not 0 <= focus_depth_target <= 1:
                raise ValueError("focus_depth_target must be between zero and one")
            self.target.focus_depth_target = float(focus_depth_target)
        if lod_bias is not None:
            planner = self.stream.planner
            self.stream.planner = Planner(
                lod_bias=lod_bias,
                progressive=planner.progressive,
                max_intermediate_levels=planner.max_intermediate_levels,
                max_initial_voxel_footprint=planner.max_initial_voxel_footprint,
            )
        if replan and self._last_view is not None:
            return self.update(self._last_view)
        return None

    def _camera_changed(self) -> None:
        if self._last_view is None or self._presenting:
            return
        viewport, world_to_clip = self.canvas.camera_state()
        # Scene-bound changes can alter only the depth/near-far transform when
        # a replacement volume is published. They are not camera interaction
        # and must not replay the same plan. The first two clip rows fully
        # describe screen-space pan, zoom, and rotation for LOD selection.
        if viewport == self._last_view.viewport and np.allclose(
            world_to_clip[:2],
            self._last_view.world_to_clip[:2],
            rtol=1e-7,
            atol=1e-7,
        ):
            return
        view = replace(
            self._last_view,
            viewport=viewport,
            world_to_clip=world_to_clip,
        )
        self._last_view = view
        with self._camera_condition:
            if self._closed:
                return
            self._camera_generation += 1
            self._pending_camera_view = view
            self._camera_deadline = monotonic() + self._camera_debounce_seconds
            self._camera_condition.notify()

    def _camera_worker(self) -> None:
        while True:
            with self._camera_condition:
                while self._pending_camera_view is None and not self._closed:
                    self._camera_condition.wait()
                if self._closed:
                    return
                remaining = self._camera_deadline - monotonic()
                if remaining > 0:
                    self._camera_condition.wait(remaining)
                    continue
                view = self._pending_camera_view
                generation = self._camera_generation
                self._pending_camera_view = None
            assert view is not None
            plan = self.stream.plan(view)
            with self._camera_condition:
                if self._closed or generation != self._camera_generation:
                    continue
            self.stream.submit(view, plan)
            self._notify_targeted(plan)

    def _dispatch_presentation(self, callback: Callable[[], None]) -> None:
        def guarded() -> None:
            self._presenting = True
            try:
                callback()
            finally:
                self._presenting = False

        self._dispatch(guarded)

    def _notify_targeted(self, plan: Plan) -> None:
        callback = self._on_targeted
        if callback is not None:
            self._dispatch(lambda: callback(plan))

    def close(self) -> None:
        if self._camera_signal is not None:
            self._camera_signal.disconnect(self._camera_changed)
        with self._camera_condition:
            self._closed = True
            self._camera_condition.notify()
        self._camera_thread.join(timeout=1)
        self.stream.close()
        self.target.close()
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
