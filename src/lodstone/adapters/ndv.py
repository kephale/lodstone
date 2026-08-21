"""Dense Lodstone target for ndv's renderer-independent canvas API."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np

from ..model import Layout, Plan, Pyramid, Region, TileKey, Update, View
from ..resident import ResidentArrays, ResidentLease
from ..runtime import Runtime
from ..source import Source
from ..stream import Stream


@dataclass(frozen=True, slots=True)
class NDVPublication:
    """Immutable dense phase snapshot ready for an ndv image handle."""

    level: int
    region: Region
    data: np.ndarray
    scales: tuple[float, ...]
    origins: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NDVPreparation:
    desired_keys: frozenset[TileKey]


class NDVTarget:
    """Present complete dense phases through an ndv ``ArrayCanvas``.

    The target only relies on ndv's canvas and image-handle abstraction, so the
    same adapter works with its VisPy and pygfx backends.  ndv currently exposes
    scale but not translation on that abstraction; displayed regions must
    therefore begin at the level origin.
    """

    def __init__(
        self,
        canvas: Any,
        pyramid: Pyramid,
        *,
        memory_limit: int = 512 * 1024**2,
        on_presented: Callable[[NDVPublication], None] | None = None,
    ) -> None:
        self.canvas = canvas
        self.pyramid = pyramid
        self.memory_limit = int(memory_limit)
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        self.resident = ResidentArrays(pyramid, compose=True)
        self.handle: Any | None = None
        self.on_presented = on_presented

    def layout(self, view: View, pyramid: Pyramid) -> Layout:
        return Layout(
            kind="dense",
            memory_limit=self.memory_limit,
            squeeze_hidden=False,
        )

    def stage_prepare(self, view: View, plan: Plan) -> NDVPreparation:
        self.resident.prepare(plan)
        desired = plan.desired or plan.wanted
        return NDVPreparation(frozenset(tile.key for tile in desired))

    def prepare(
        self,
        view: View,
        plan: Plan,
        prepared: NDVPreparation,
    ) -> ResidentLease:
        return ResidentLease(self.resident, prepared.desired_keys)

    def stage(self, updates: Sequence[Update]):
        return self.resident.apply(updates)

    def apply(self, updates: Sequence[Update]) -> None:
        # Resident writes occur in ``stage``. ndv receives one immutable array
        # only when the corresponding phase is complete.
        return None

    def stage_phase(
        self,
        view: View,
        plan: Plan,
        phase: int,
    ) -> NDVPublication:
        desired = plan.desired or plan.wanted
        levels = {tile.level for tile in desired if tile.phase == phase}
        if len(levels) != 1:
            raise RuntimeError("an ndv phase must describe exactly one pyramid level")
        level = levels.pop()
        window = self.resident.windows[level]
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
            float(np.linalg.norm(transform[:-1, axis])) for axis in view.displayed_axes
        )
        world_origin = transform[:-1, :-1] @ np.asarray(window.region.start)
        world_origin += transform[:-1, -1]
        origins = tuple(float(world_origin[axis]) for axis in view.displayed_axes)
        return NDVPublication(level, window.region, data, scales, origins)

    def phase_complete(
        self,
        view: View,
        plan: Plan,
        phase: int,
        publication: NDVPublication,
    ) -> None:
        ndim = len(view.displayed_axes)
        self.canvas.set_ndim(ndim)
        first_publication = self.handle is None
        if first_publication:
            factory = self.canvas.add_image if ndim == 2 else self.canvas.add_volume
            self.handle = factory(publication.data)
        else:
            self.handle.set_data(publication.data)
        self.canvas.set_scales(publication.scales, reset_range=False)
        self.canvas.set_origins(publication.origins)
        if first_publication:
            self.canvas.set_range()
        self.canvas.refresh()
        if self.on_presented is not None:
            self.on_presented(publication)

    def discard(self, keys: Collection[TileKey]) -> None:
        self.resident.discard(keys)

    def complete(self, view: View, plan: Plan) -> None:
        self.resident.complete(plan)

    def redraw(self) -> None:
        self.canvas.refresh()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.resident.clear()


class NDVController:
    """Connect a Lodstone source to an ndv ``ArrayViewer`` or canvas.

    The viewer integration intentionally accepts explicit :class:`View`
    snapshots while ndv's public canvas API does not yet expose a camera matrix.
    This is enough for full-view 2-D/3-D use and keeps camera-event policy out of
    the renderer target.
    """

    def __init__(
        self,
        viewer_or_canvas: Any,
        source: Source,
        *,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        runtime: Runtime | None = None,
        memory_limit: int = 512 * 1024**2,
        on_presented: Callable[[NDVPublication], None] | None = None,
        **stream_options: Any,
    ) -> None:
        self.viewer = viewer_or_canvas if hasattr(viewer_or_canvas, "canvas") else None
        self.canvas = (
            viewer_or_canvas.canvas if self.viewer is not None else viewer_or_canvas
        )
        if dispatch is None:
            dispatch = getattr(viewer_or_canvas, "dispatch", None)
        self.runtime = runtime or Runtime()
        self._owns_runtime = runtime is None
        self.target = NDVTarget(
            self.canvas,
            source.pyramid,
            memory_limit=memory_limit,
            on_presented=on_presented,
        )
        self.stream = Stream(
            source,
            self.target,
            dispatch=dispatch or (lambda callback: callback()),
            runtime=self.runtime,
            **stream_options,
        )
        self._last_view: View | None = None
        self._camera_signal = getattr(self.canvas, "cameraChanged", None)
        if self._camera_signal is not None:
            self._camera_signal.connect(self._camera_changed)

    def update(self, view: View) -> Plan:
        self._last_view = view
        return self.stream.update(view)

    def _camera_changed(self) -> None:
        if self._last_view is None:
            return
        viewport, world_to_clip = self.canvas.camera_state()
        self.update(
            replace(
                self._last_view,
                viewport=viewport,
                world_to_clip=world_to_clip,
            )
        )

    def close(self) -> None:
        if self._camera_signal is not None:
            self._camera_signal.disconnect(self._camera_changed)
        self.stream.close()
        self.target.close()
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
