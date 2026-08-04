"""Deterministic camera-aware multiscale tile planning."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from .model import Layout, Plan, Pyramid, Region, Tile, TileKey, View


class Planner:
    """Choose visible tiles and their level-of-detail for a view."""

    def __init__(
        self,
        *,
        lod_bias: float = 1.0,
        progressive: bool = True,
    ) -> None:
        if lod_bias <= 0:
            raise ValueError("lod_bias must be positive")
        self.lod_bias = float(lod_bias)
        self.progressive = bool(progressive)

    def plan(
        self,
        pyramid: Pyramid,
        view: View,
        layout: Layout,
        *,
        available: frozenset[TileKey] = frozenset(),
    ) -> Plan:
        """Return an ordered, cache-aware plan for ``view``."""

        self._validate(pyramid, view, layout)
        target_level = self._select_level(pyramid, view)
        while target_level < len(pyramid.levels) - 1:
            target_tiles = self._tiles_for_level(
                pyramid, view, layout, target_level, phase=0
            )
            if _tiles_nbytes(target_tiles, pyramid) <= layout.memory_limit:
                break
            target_level += 1
        levels = [target_level]
        if self.progressive:
            levels = list(range(len(pyramid.levels) - 1, target_level - 1, -1))

        wanted: list[Tile] = []
        desired: list[Tile] = []
        retain: set[TileKey] = set()
        for phase, level_index in enumerate(levels):
            tiles = self._tiles_for_level(pyramid, view, layout, level_index, phase)
            desired.extend(tiles)
            if level_index == target_level:
                retain.update(tile.key for tile in tiles)
            wanted.extend(tile for tile in tiles if tile.key not in available)

        wanted.sort(key=lambda tile: (tile.phase, tile.priority))
        desired.sort(key=lambda tile: (tile.phase, tile.priority))
        return Plan(
            tuple(wanted),
            frozenset(retain),
            target_level,
            tuple(desired),
        )

    def _validate(self, pyramid: Pyramid, view: View, layout: Layout) -> None:
        if len(view.index) != pyramid.ndim:
            raise ValueError("view index dimensionality does not match the pyramid")
        for axis, value in enumerate(view.index):
            if value is not None and not 0 <= value < pyramid.levels[0].shape[axis]:
                raise ValueError(f"index {value} is outside axis {axis}")
        if layout.block_shape is not None and len(layout.block_shape) not in (
            len(view.displayed_axes),
            pyramid.ndim,
        ):
            raise ValueError(
                "block_shape must match displayed or complete data dimensionality"
            )

    def _select_level(self, pyramid: Pyramid, view: View) -> int:
        threshold = self.lod_bias
        selected = 0
        for index, level in enumerate(pyramid.levels):
            footprint = _voxel_footprint_px(level.voxel_to_world, level.shape, view)
            if footprint <= threshold:
                selected = index
            else:
                break
        return selected

    def _tiles_for_level(
        self,
        pyramid: Pyramid,
        view: View,
        layout: Layout,
        level_index: int,
        phase: int,
    ) -> list[Tile]:
        level = pyramid.levels[level_index]
        block = _display_block_shape(layout, level.chunks, view.displayed_axes)
        grids = [
            range(math.ceil(level.shape[axis] / block[i]))
            for i, axis in enumerate(view.displayed_axes)
        ]
        selection = tuple(-1 if value is None else int(value) for value in view.index)
        level_selection = _selection_at_level(pyramid, view, level_index)
        result: list[Tile] = []
        for grid_index in product(*grids):
            start = []
            stop = []
            display_i = 0
            for axis, selected in enumerate(level_selection):
                if selected is None:
                    value = grid_index[display_i] * block[display_i]
                    start.append(value)
                    stop.append(min(value + block[display_i], level.shape[axis]))
                    display_i += 1
                else:
                    start.append(selected)
                    stop.append(selected + 1)

            region = Region(tuple(start), tuple(stop))
            projected = _project_region(level.voxel_to_world, region, view)
            if not projected.visible:
                continue
            key = TileKey(level_index, tuple(grid_index), selection)
            result.append(Tile(key, region, projected.priority, phase))

        return result


def _display_block_shape(
    layout: Layout,
    native_chunks: tuple[int, ...],
    displayed_axes: tuple[int, ...],
) -> tuple[int, ...]:
    if layout.block_shape is None:
        return tuple(native_chunks[axis] for axis in displayed_axes)
    if len(layout.block_shape) == len(displayed_axes):
        return layout.block_shape
    return tuple(layout.block_shape[axis] for axis in displayed_axes)


def _tiles_nbytes(tiles: list[Tile], pyramid: Pyramid) -> int:
    return sum(
        tile.region.size * pyramid.levels[tile.level].dtype.itemsize for tile in tiles
    )


def _selection_at_level(
    pyramid: Pyramid, view: View, level_index: int
) -> tuple[int | None, ...]:
    """Map finest-level hidden-axis indices through the level transforms."""

    if level_index == 0:
        return view.index
    finest = pyramid.levels[0]
    level = pyramid.levels[level_index]
    point = np.asarray(
        [0.0 if value is None else float(value) for value in view.index] + [1.0]
    )
    world = finest.voxel_to_world @ point
    level_point = np.linalg.solve(level.voxel_to_world, world)[:-1]
    result: list[int | None] = []
    for axis, value in enumerate(view.index):
        if value is None:
            result.append(None)
        else:
            mapped = math.floor(float(level_point[axis]) + 0.5)
            result.append(min(max(mapped, 0), level.shape[axis] - 1))
    return tuple(result)


def _local_world(
    matrix: np.ndarray, data_points: np.ndarray, displayed_axes: tuple[int, ...]
) -> np.ndarray:
    homogeneous = np.concatenate(
        [data_points, np.ones((len(data_points), 1), dtype=np.float64)], axis=1
    )
    world = (matrix @ homogeneous.T).T[:, :-1]
    local = np.zeros((len(data_points), 3), dtype=np.float64)
    for local_axis, data_axis in enumerate(displayed_axes):
        local[:, local_axis] = world[:, data_axis]
    return local


def _clip_points(points: np.ndarray, view: View) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    clip = (view.world_to_clip @ homogeneous.T).T
    w = clip[:, 3]
    safe = np.where(np.abs(w) < 1e-12, np.nan, w)
    return clip[:, :3] / safe[:, None]


def _voxel_footprint_px(
    matrix: np.ndarray, shape: tuple[int, ...], view: View
) -> float:
    center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    points = [center]
    for axis in view.displayed_axes:
        point = center.copy()
        point[axis] += 1.0
        points.append(point)
    local = _local_world(matrix, np.asarray(points), view.displayed_axes)
    clip = _clip_points(local, view)
    if not np.all(np.isfinite(clip)):
        return math.inf
    scale = np.asarray(view.viewport, dtype=np.float64) / 2.0
    origin = clip[0, :2] * scale
    lengths = [np.linalg.norm(point[:2] * scale - origin) for point in clip[1:]]
    return float(max(lengths, default=math.inf))


class _Projection:
    __slots__ = ("priority", "visible")

    def __init__(self, visible: bool, priority: float) -> None:
        self.visible = visible
        self.priority = priority


def _project_region(matrix: np.ndarray, region: Region, view: View) -> _Projection:
    bounds = [(region.start[axis], region.stop[axis]) for axis in view.displayed_axes]
    data_center = np.asarray(
        [(a + b) / 2.0 for a, b in zip(region.start, region.stop, strict=True)]
    )
    points = []
    for corner in product(*bounds):
        point = data_center.copy()
        for local_axis, data_axis in enumerate(view.displayed_axes):
            point[data_axis] = corner[local_axis]
        points.append(point)
    local = _local_world(matrix, np.asarray(points), view.displayed_axes)
    clip = _clip_points(local, view)
    finite = np.all(np.isfinite(clip), axis=1)
    if not np.any(finite):
        return _Projection(False, math.inf)
    clip = clip[finite]
    visible = bool(
        np.all(np.max(clip, axis=0) >= -1.0) and np.all(np.min(clip, axis=0) <= 1.0)
    )
    center = np.mean(clip[:, :2], axis=0)
    center_distance = float(np.dot(center, center))
    if len(view.displayed_axes) == 3:
        # OpenGL NDC depth runs near-to-far from -1 to +1. Depth dominates
        # the center-distance tie breaker, yielding front-to-back delivery
        # while keeping chunks on a similar plane center-first.
        depth = float(np.min(clip[:, 2]))
        priority = depth * 1_000_000.0 + center_distance
    else:
        priority = center_distance
    return _Projection(visible, priority)
