"""Experimental dense Lodstone adapter for SceneX models.

This module deliberately imports SceneX only when it must create a visual or
select its application dispatcher. Lodstone's core package therefore remains
usable without SceneX installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..model import Plan, Pyramid, View
from ..source import Source
from .ndv import NDVController, NDVPublication, NDVTarget


def scenex_view_snapshot(
    view: Any,
    ndim: int,
    *,
    displayed_axes: tuple[int, ...] | None = None,
    index: tuple[int | None, ...] | None = None,
) -> View:
    """Capture a SceneX view as an immutable Lodstone camera snapshot.

    SceneX stores transforms for row-vector multiplication in ``(x, y, z)``
    order. Lodstone uses column vectors whose local coordinates follow array
    order, so both the matrix convention and spatial-axis order are converted.
    """
    if ndim < 2:
        raise ValueError("SceneX streaming requires at least two array dimensions")
    if displayed_axes is None:
        displayed_axes = tuple(range(ndim - min(ndim, 3), ndim))
    if len(displayed_axes) not in (2, 3):
        raise ValueError("displayed_axes must contain two or three axes")
    if index is None:
        shown = set(displayed_axes)
        index = tuple(None if axis in shown else 0 for axis in range(ndim))
    if len(index) != ndim:
        raise ValueError("index dimensionality must match ndim")
    rect = view.content_rect
    if rect is None:
        raise ValueError("the SceneX view must be attached to a canvas")
    _, _, width, height = rect
    if width <= 0 or height <= 0:
        raise ValueError("the SceneX view content rectangle must be positive")

    camera = view.camera
    row_world_to_clip = (camera.transform.inv() @ camera.projection).root
    scene_world_to_clip = np.asarray(row_world_to_clip, dtype=np.float64).T
    scene_from_array = np.zeros((4, 4), dtype=np.float64)
    for scene_axis, array_axis in enumerate(reversed(range(len(displayed_axes)))):
        scene_from_array[scene_axis, array_axis] = 1.0
    scene_from_array[3, 3] = 1.0
    world_to_clip = scene_world_to_clip @ scene_from_array

    eye = None
    if len(displayed_axes) == 3:
        scene_eye = np.asarray(camera.transform.map((0, 0, 0)), dtype=np.float64)[:3]
        eye = (
            float(scene_eye[2]),
            float(scene_eye[1]),
            float(scene_eye[0]),
        )
    return View(displayed_axes, index, (width, height), world_to_clip, eye=eye)


class _CameraSignal:
    def __init__(self, view: Any) -> None:
        self._signals = (
            view.camera.events.transform,
            view.camera.events.projection,
            view.layout.events.x_start,
            view.layout.events.x_end,
            view.layout.events.y_start,
            view.layout.events.y_end,
            view.canvas.events.width,
            view.canvas.events.height,
        )
        self._wrappers: dict[Callable[[], None], Callable[..., None]] = {}

    def connect(self, callback: Callable[[], None]) -> None:
        def wrapped(*_args: object) -> None:
            callback()

        self._wrappers[callback] = wrapped
        for signal in self._signals:
            signal.connect(wrapped)

    def disconnect(self, callback: Callable[[], None]) -> None:
        wrapped = self._wrappers.pop(callback, None)
        if wrapped is not None:
            for signal in self._signals:
                signal.disconnect(wrapped)


class _SceneXHandle:
    def __init__(self, view: Any, node: Any) -> None:
        self._view = view
        self._node = node

    def set_data(self, data: np.ndarray) -> None:
        self._node.data = data

    def set_world_transform(
        self, scales: tuple[float, ...], origins: tuple[float, ...]
    ) -> None:
        from scenex import Transform  # pyright: ignore[reportMissingImports]

        scene_scales = (*reversed(scales), *(1.0,) * (3 - len(scales)))
        scene_origins = (*reversed(origins), *(0.0,) * (3 - len(origins)))
        self._node.transform = (
            Transform().scaled(scene_scales).translated(scene_origins)
        )

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._node.clims = clims

    def set_visible(self, visible: bool) -> None:
        self._node.visible = visible

    def set_order(self, order: int) -> None:
        self._node.order = order

    def remove(self) -> None:
        self._view.scene.remove_child(self._node)


class _SceneXCanvas:
    """Present SceneX's model API as the dense canvas contract used by Lodstone."""

    def __init__(self, view: Any) -> None:
        if view.canvas is None:
            raise ValueError("the SceneX view must be attached to a canvas")
        self.view = view
        self.cameraChanged = _CameraSignal(view)
        self._ndim = 2

    def camera_state(self) -> tuple[tuple[int, int], np.ndarray]:
        snapshot = scenex_view_snapshot(self.view, self._ndim)
        return snapshot.viewport, snapshot.world_to_clip

    def set_ndim(self, ndim: int) -> None:
        if ndim not in (2, 3):
            raise ValueError("SceneX visuals require two or three displayed axes")
        self._ndim = ndim

    def add_image(
        self, data: np.ndarray, *, reset_range: bool = False
    ) -> _SceneXHandle:
        return self._add_visual(data, volume=False)

    def add_volume(
        self, data: np.ndarray, *, reset_range: bool = False
    ) -> _SceneXHandle:
        return self._add_visual(data, volume=True)

    def _add_visual(self, data: np.ndarray, *, volume: bool) -> _SceneXHandle:
        from scenex import Image, Volume  # pyright: ignore[reportMissingImports]

        node_type = Volume if volume else Image
        node = node_type(data=data, order=0)
        self.view.scene.add_child(node)
        return _SceneXHandle(self.view, node)

    def set_range(self) -> None:
        from scenex.utils.projections import (  # pyright: ignore[reportMissingImports]
            zoom_to_fit,
        )

        zoom_to_fit(self.view, type="orthographic", letterbox=True)

    def refresh(self) -> None:
        # SceneX adaptors react directly to model events.
        return None


class SceneXTarget(NDVTarget):
    """Dense phase target backed by a SceneX ``View`` scene graph."""

    def __init__(self, view_or_canvas: Any, pyramid: Pyramid, **options: Any) -> None:
        canvas = (
            view_or_canvas
            if isinstance(view_or_canvas, _SceneXCanvas)
            else _SceneXCanvas(view_or_canvas)
        )
        super().__init__(canvas, pyramid, **options)

    def phase_complete(
        self,
        view: View,
        plan: Plan,
        phase: int,
        publication: NDVPublication,
    ) -> None:
        super().phase_complete(view, plan, phase, publication)
        # Fine levels render after their coarse context. The context is masked
        # by NDVTarget's shared dense publication logic before either is shown.
        handle = self.handles.get(publication.level)
        if handle is not None:
            handle.set_order(self._context_level - publication.level)


class SceneXController(NDVController):
    """Connect a Lodstone source to a renderer-neutral SceneX ``View``."""

    def __init__(
        self,
        view: Any,
        source: Source,
        *,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        **options: Any,
    ) -> None:
        self.scene_view = view
        if dispatch is None:
            from scenex.app import app  # pyright: ignore[reportMissingImports]

            dispatch = lambda callback: app().call_in_main_thread(callback)
        super().__init__(_SceneXCanvas(view), source, dispatch=dispatch, **options)

    def _make_target(
        self, canvas: Any, pyramid: Pyramid, **options: Any
    ) -> SceneXTarget:
        return SceneXTarget(canvas, pyramid, **options)

    def snapshot(
        self,
        *,
        displayed_axes: tuple[int, ...] | None = None,
        index: tuple[int | None, ...] | None = None,
    ) -> View:
        """Return the current SceneX camera in Lodstone coordinates."""
        return scenex_view_snapshot(
            self.scene_view,
            self.stream.source.pyramid.ndim,
            displayed_axes=displayed_axes,
            index=index,
        )

    def update_from_scene(
        self,
        *,
        displayed_axes: tuple[int, ...] | None = None,
        index: tuple[int | None, ...] | None = None,
    ):
        """Snapshot the SceneX camera and submit a streaming update."""
        return self.update(self.snapshot(displayed_axes=displayed_axes, index=index))
