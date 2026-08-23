"""Qt overlay for inspecting Lodstone's projected dense block plans."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import QEvent, QObject, QPointF, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from lodstone import Plan, Pyramid, Region


class BlockOverlay(QWidget):
    """Draw planned blocks and their enclosing dense focus box over a canvas."""

    _EDGES = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )

    def __init__(self, canvas: Any, pyramid: Pyramid) -> None:
        parent = canvas.frontend_widget()
        super().__init__(parent)
        self._canvas = canvas
        self._pyramid = pyramid
        self._plan: Plan | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)
        self.show()
        self.raise_()

    def set_plan(self, plan: Plan) -> None:
        self._plan = plan
        self.update()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(watched, event)

    def paintEvent(self, event: Any) -> None:
        del event
        if self._plan is None:
            return
        try:
            viewport, world_to_clip = self._canvas.camera_state()
        except RuntimeError:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        context_level = len(self._pyramid.levels) - 1
        target_level = self._plan.target_level
        by_level: dict[int, list[Any]] = {}
        for tile in self._plan.desired:
            by_level.setdefault(tile.level, []).append(tile)

        # Context is cool blue; the camera-selected focus is warm orange.
        for level, tiles in by_level.items():
            color = QColor(55, 205, 255, 105)
            if level == target_level and level != context_level:
                color = QColor(255, 165, 45, 155)
            painter.setPen(QPen(color, 1.0))
            transform = self._pyramid.levels[level].voxel_to_world
            for tile in tiles:
                self._draw_region(
                    painter, tile.region, transform, world_to_clip, viewport
                )

        focus = by_level.get(target_level, [])
        if target_level != context_level and focus:
            region = _enclosing_region(focus)
            pen = QPen(QColor(255, 245, 170, 230), 2.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            self._draw_region(
                painter,
                region,
                self._pyramid.levels[target_level].voxel_to_world,
                world_to_clip,
                viewport,
            )
        painter.end()

    def _draw_region(
        self,
        painter: QPainter,
        region: Region,
        voxel_to_world: np.ndarray,
        world_to_clip: np.ndarray,
        viewport: tuple[int, int],
    ) -> None:
        corners = np.asarray(
            [(*corner, 1.0) for corner in product(*zip(region.start, region.stop))],
            dtype=np.float64,
        )
        clip = (world_to_clip @ voxel_to_world @ corners.T).T
        if np.any(np.abs(clip[:, 3]) < 1e-12):
            return
        ndc = clip[:, :3] / clip[:, 3, None]
        if not np.all(np.isfinite(ndc)) or np.all(clip[:, 3] < 0):
            return
        width, height = viewport
        points = [
            QPointF((point[0] + 1.0) * width / 2, (1.0 - point[1]) * height / 2)
            for point in ndc
        ]
        for start, stop in self._EDGES:
            painter.drawLine(points[start], points[stop])


def _enclosing_region(tiles: list[Any]) -> Region:
    from lodstone import Region

    ndim = tiles[0].region.ndim
    return Region(
        tuple(min(tile.region.start[axis] for tile in tiles) for axis in range(ndim)),
        tuple(max(tile.region.stop[axis] for tile in tiles) for axis in range(ndim)),
    )
