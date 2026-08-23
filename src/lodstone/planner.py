"""Deterministic camera-aware multiscale tile planning."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from itertools import product

import numpy as np

from .geometry import clamp_region_to_budget
from .model import Layout, Level, Plan, Pyramid, Region, Tile, TileKey, View


class Planner:
    """Choose visible tiles and their level-of-detail for a view."""

    def __init__(
        self,
        *,
        lod_bias: float = 1.0,
        progressive: bool = True,
        max_intermediate_levels: int | None = None,
        max_initial_voxel_footprint: float | None = None,
    ) -> None:
        if lod_bias <= 0:
            raise ValueError("lod_bias must be positive")
        if max_intermediate_levels is not None and (
            not isinstance(max_intermediate_levels, int)
            or isinstance(max_intermediate_levels, bool)
            or max_intermediate_levels < 0
        ):
            raise ValueError(
                "max_intermediate_levels must be a nonnegative integer or None"
            )
        if max_initial_voxel_footprint is not None and (
            not math.isfinite(max_initial_voxel_footprint)
            or max_initial_voxel_footprint <= 0
        ):
            raise ValueError(
                "max_initial_voxel_footprint must be positive and finite or None"
            )
        self.lod_bias = float(lod_bias)
        self.progressive = bool(progressive)
        self.max_intermediate_levels = max_intermediate_levels
        self.max_initial_voxel_footprint = max_initial_voxel_footprint

    def plan(
        self,
        pyramid: Pyramid,
        view: View,
        layout: Layout,
        *,
        available: frozenset[TileKey] = frozenset(),
        previous_target_level: int | None = None,
        lod_hysteresis: float = 0.0,
    ) -> Plan:
        """Return an ordered, cache-aware plan for ``view``."""

        self._validate(pyramid, view, layout)
        if not 0 <= lod_hysteresis < 1:
            raise ValueError("lod_hysteresis must be in [0, 1)")
        if previous_target_level is not None and not 0 <= previous_target_level < len(
            pyramid.levels
        ):
            raise ValueError("previous_target_level is outside the pyramid")
        target_level = self._select_level(
            pyramid,
            view,
            previous_target_level=previous_target_level,
            lod_hysteresis=lod_hysteresis,
        )
        context_level = len(pyramid.levels) - 1
        target_depth_weight = (
            None
            if layout.mixed_lod and target_level == context_level
            else layout.focus_depth_weight
        )
        while target_level < len(pyramid.levels) - 1:
            target_tiles = self._tiles_for_level(
                pyramid,
                view,
                layout,
                target_level,
                phase=0,
                focus_depth_weight=target_depth_weight,
            )
            if layout.memory_policy == "crop":
                if target_tiles:
                    break
            elif _tiles_nbytes(target_tiles, pyramid) <= layout.memory_limit:
                break
            target_level += 1
        target_depth_weight = (
            None
            if layout.mixed_lod and target_level == context_level
            else layout.focus_depth_weight
        )
        levels = self._levels(pyramid, view, target_level)
        if layout.mixed_lod and context_level not in levels:
            levels.insert(0, context_level)

        wanted: list[Tile] = []
        desired: list[Tile] = []
        retain: set[TileKey] = set()
        for phase, level_index in enumerate(levels):
            tiles = self._tiles_for_level(
                pyramid,
                view,
                layout,
                level_index,
                phase,
                focus_depth_weight=(
                    target_depth_weight if level_index == target_level else None
                ),
            )
            desired.extend(tiles)
            if level_index == target_level or (
                layout.mixed_lod and level_index == context_level
            ):
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

    def plan_region(
        self,
        pyramid: Pyramid,
        view: View,
        layout: Layout,
        *,
        target_level: int,
        target_region: Region,
        available: frozenset[TileKey] = frozenset(),
        fetch_intermediate: bool = True,
    ) -> Plan:
        """Plan a memory-bounded cuboid chosen by a renderer integration.

        This mode preserves a target's established viewport-region policy
        while moving multilevel mapping, native-grid enumeration, priorities,
        and cache filtering into Lodstone.
        """

        self._validate(pyramid, view, layout)
        if not 0 <= target_level < len(pyramid.levels):
            raise ValueError("target_level is outside the pyramid")
        if target_region.ndim != pyramid.ndim:
            raise ValueError("target region dimensionality does not match pyramid")

        levels = self._levels(pyramid, view, target_level)
        context_level = len(pyramid.levels) - 1
        if layout.mixed_lod and context_level not in levels:
            levels.insert(0, context_level)
        desired: list[Tile] = []
        wanted: list[Tile] = []
        retain: set[TileKey] = set()
        selection = tuple(-1 if value is None else int(value) for value in view.index)

        for phase, level_index in enumerate(levels):
            region = _map_region(
                target_region,
                pyramid.levels[target_level],
                pyramid.levels[level_index],
            )
            level = pyramid.levels[level_index]
            region = clamp_region_to_budget(
                region,
                level.shape,
                itemsize=level.dtype.itemsize,
                max_bytes=layout.memory_limit,
                max_axis_extent=layout.max_axis_extent,
            )
            tiles = _tiles_in_region(
                level,
                level_index,
                region,
                view,
                selection,
                phase,
            )
            desired.extend(tiles)
            if level_index == target_level or (
                layout.mixed_lod and level_index == context_level
            ):
                retain.update(tile.key for tile in tiles)
            if fetch_intermediate or level_index == target_level:
                wanted.extend(tile for tile in tiles if tile.key not in available)

        wanted.sort(key=lambda tile: (tile.phase, tile.priority))
        desired.sort(key=lambda tile: (tile.phase, tile.priority))
        return Plan(tuple(wanted), frozenset(retain), target_level, tuple(desired))

    def plan_overview(
        self,
        pyramid: Pyramid,
        view: View,
        *,
        level_index: int | None = None,
        memory_limit: int,
        available: frozenset[TileKey] = frozenset(),
    ) -> Plan:
        """Plan persistent whole-volume context at one coarse level.

        All displayed axes cover their complete extent. Hidden axes are
        restricted to the current transformed selection, keeping overview
        residency bounded for time series and channel stacks. An empty plan
        is returned when even that displayed volume exceeds ``memory_limit``.
        """

        if memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        level_index = len(pyramid.levels) - 1 if level_index is None else level_index
        if not 0 <= level_index < len(pyramid.levels):
            raise ValueError("level_index is outside the pyramid")
        self._validate(pyramid, view, Layout(memory_limit=memory_limit))
        level = pyramid.levels[level_index]
        level_selection = _selection_at_level(pyramid, view, level_index)
        start = [0] * level.ndim
        stop = list(level.shape)
        for axis, selected in enumerate(level_selection):
            if selected is not None:
                start[axis] = selected
                stop[axis] = selected + 1
        region = Region(tuple(start), tuple(stop))
        if region.size * level.dtype.itemsize > memory_limit:
            return Plan((), frozenset(), level_index, ())

        selection = tuple(-1 if value is None else int(value) for value in view.index)
        tiles = _tiles_in_region(
            level,
            level_index,
            region,
            view,
            selection,
            phase=0,
        )
        tiles.sort(key=lambda tile: tile.priority)
        wanted = tuple(tile for tile in tiles if tile.key not in available)
        return Plan(
            wanted, frozenset(tile.key for tile in tiles), level_index, tuple(tiles)
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

    def _select_level(
        self,
        pyramid: Pyramid,
        view: View,
        *,
        previous_target_level: int | None = None,
        lod_hysteresis: float = 0.0,
    ) -> int:
        selected = self._select_level_at_threshold(pyramid, view, self.lod_bias)
        if (
            previous_target_level is None
            or selected == previous_target_level
            or lod_hysteresis == 0
        ):
            return selected
        factor = (
            1 + lod_hysteresis
            if selected < previous_target_level
            else 1 - lod_hysteresis
        )
        return self._select_level_at_threshold(pyramid, view, self.lod_bias * factor)

    @staticmethod
    def _select_level_at_threshold(
        pyramid: Pyramid, view: View, threshold: float
    ) -> int:
        selected = 0
        for index, level in enumerate(pyramid.levels):
            footprint = _voxel_footprint_px(level.voxel_to_world, level.shape, view)
            if footprint <= threshold:
                selected = index
            else:
                break
        return selected

    def _levels(self, pyramid: Pyramid, view: View, target_level: int) -> list[int]:
        if not self.progressive:
            return [target_level]
        coarsest = len(pyramid.levels) - 1
        footprint_limit = self.max_initial_voxel_footprint
        if footprint_limit is not None:
            coarsest = target_level
            for index in range(target_level + 1, len(pyramid.levels)):
                footprint = _voxel_footprint_px(
                    pyramid.levels[index].voxel_to_world,
                    pyramid.levels[index].shape,
                    view,
                )
                if footprint <= footprint_limit:
                    coarsest = index
        levels = list(range(coarsest, target_level - 1, -1))
        limit = self.max_intermediate_levels
        if limit is None or len(levels) <= limit + 2:
            return levels
        return [coarsest, *levels[-(limit + 1) :]]

    def _tiles_for_level(
        self,
        pyramid: Pyramid,
        view: View,
        layout: Layout,
        level_index: int,
        phase: int,
        *,
        focus_depth_weight: float | None = None,
    ) -> list[Tile]:
        level = pyramid.levels[level_index]
        grids = _display_grid(layout, level, view.displayed_axes)
        selection = tuple(-1 if value is None else int(value) for value in view.index)
        level_selection = _selection_at_level(pyramid, view, level_index)
        result: list[Tile] = []
        for grid_cell in product(*grids):
            grid_index = tuple(cell[0] for cell in grid_cell)
            start = []
            stop = []
            display_i = 0
            for axis, selected in enumerate(level_selection):
                if selected is None:
                    _index, cell_start, cell_stop = grid_cell[display_i]
                    start.append(cell_start)
                    stop.append(cell_stop)
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

        if layout.memory_policy == "crop":
            return _crop_dense_tiles(
                result,
                level,
                layout.memory_limit,
                view=view,
                focus_depth_weight=focus_depth_weight,
            )
        return result


def merge_plans(primary: Plan, *prefixes: Plan) -> Plan:
    """Merge persistent prerequisite plans ahead of a primary view plan.

    Tile identity is de-duplicated while preserving first occurrence order.
    The primary target level remains authoritative and all retained keys are
    carried into the combined plan.
    """

    def unique(tiles):
        seen: set[TileKey] = set()
        result = []
        for tile in tiles:
            if tile.key not in seen:
                seen.add(tile.key)
                result.append(tile)
        return tuple(result)

    ordered = (*prefixes, primary)
    wanted = unique(tile for plan in ordered for tile in plan.wanted)
    desired = unique(tile for plan in ordered for tile in plan.desired)
    retain = frozenset(key for plan in ordered for key in plan.retain)
    return Plan(wanted, retain, primary.target_level, desired)


def available_tile_keys(
    pyramid: Pyramid,
    view: View,
    chunk_bounds: Mapping[int, Collection[tuple[tuple[int, int], ...]]],
) -> frozenset[TileKey]:
    """Translate resident native chunk bounds into logical display keys."""

    selection = tuple(-1 if value is None else int(value) for value in view.index)
    keys = set()
    for level_index, chunks in chunk_bounds.items():
        level = pyramid.levels[level_index]
        for chunk in chunks:
            starts = tuple(start for start, _stop in chunk)
            grid_index = tuple(
                level.chunk_index(axis, starts[axis]) for axis in view.displayed_axes
            )
            keys.add(TileKey(level_index, grid_index, selection))
    return frozenset(keys)


def plan_from_slices(
    pyramid: Pyramid,
    view: View,
    stages: Sequence[tuple[int, Sequence[tuple[slice, ...]]]],
    *,
    target_level: int,
    available: frozenset[TileKey] = frozenset(),
    fetch_levels: Collection[int] | None = None,
) -> Plan:
    """Build an exact compatibility plan from renderer-supplied slice queues."""

    selection = tuple(-1 if value is None else int(value) for value in view.index)
    desired = []
    wanted = []
    retain = set()
    for phase, (level_index, queue) in enumerate(stages):
        level = pyramid.levels[level_index]
        for priority, key in enumerate(queue):
            region = Region(
                tuple(int(item.start) for item in key),
                tuple(int(item.stop) for item in key),
            )
            grid_index = tuple(
                level.chunk_index(axis, region.start[axis])
                for axis in view.displayed_axes
            )
            tile = Tile(
                TileKey(level_index, grid_index, selection),
                region,
                float(priority),
                phase,
            )
            desired.append(tile)
            if level_index == target_level:
                retain.add(tile.key)
            if tile.key not in available and (
                fetch_levels is None or level_index in fetch_levels
            ):
                wanted.append(tile)
    return Plan(tuple(wanted), frozenset(retain), target_level, tuple(desired))


def _display_grid(
    layout: Layout,
    level: Level,
    displayed_axes: tuple[int, ...],
) -> list[list[tuple[int, int, int]]]:
    """Return ``(index, start, stop)`` cells for each displayed axis."""

    if layout.block_shape is None:
        result = []
        for axis in displayed_axes:
            cells = []
            start = 0
            for index, size in enumerate(level.chunk_sizes(axis)):
                cells.append((index, start, start + size))
                start += size
            result.append(cells)
        return result

    block = _display_block_shape(layout, level.chunks, displayed_axes)
    return [
        [
            (index, index * block[i], min((index + 1) * block[i], level.shape[axis]))
            for index in range(math.ceil(level.shape[axis] / block[i]))
        ]
        for i, axis in enumerate(displayed_axes)
    ]


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


def _map_region(region: Region, source: Level, destination: Level) -> Region:
    """Map a half-open source-level region into destination coordinates."""

    corners = np.asarray(list(product(*zip(region.start, region.stop, strict=True))))
    homogeneous = np.concatenate(
        [corners, np.ones((len(corners), 1), dtype=np.float64)],
        axis=1,
    )
    world = (source.voxel_to_world @ homogeneous.T).T
    mapped = np.linalg.solve(destination.voxel_to_world, world.T).T[:, :-1]
    start = np.floor(np.min(mapped, axis=0)).astype(np.int64)
    stop = np.ceil(np.max(mapped, axis=0)).astype(np.int64)
    start = np.clip(start, 0, destination.shape)
    stop = np.clip(stop, start, destination.shape)
    return Region(
        tuple(int(value) for value in start),
        tuple(int(value) for value in stop),
    )


def _tiles_in_region(
    level: Level,
    level_index: int,
    region: Region,
    view: View,
    selection: tuple[int, ...],
    phase: int,
) -> list[Tile]:
    per_axis = []
    for axis in range(level.ndim):
        if region.stop[axis] <= region.start[axis]:
            return []
        first = level.chunk_index(axis, region.start[axis])
        last = level.chunk_index(axis, region.stop[axis] - 1)
        per_axis.append(range(first, last + 1))

    tiles = []
    for grid_index in product(*per_axis):
        bounds = tuple(
            level.chunk_bounds(axis, index) for axis, index in enumerate(grid_index)
        )
        tile_region = Region(
            tuple(start for start, _stop in bounds),
            tuple(stop for _start, stop in bounds),
        )
        displayed_grid = tuple(grid_index[axis] for axis in view.displayed_axes)
        key = TileKey(level_index, displayed_grid, selection)
        projection = _project_region(level.voxel_to_world, tile_region, view)
        tiles.append(Tile(key, tile_region, projection.priority, phase))
    return tiles


def _crop_dense_tiles(
    tiles: Sequence[Tile],
    level: Level,
    memory_limit: int,
    *,
    view: View,
    focus_depth_weight: float | None,
) -> list[Tile]:
    """Select a priority-ordered focus window whose dense bounds fit memory."""

    selected: list[Tile] = []
    start: tuple[int, ...] | None = None
    stop: tuple[int, ...] | None = None
    if focus_depth_weight is None or len(view.displayed_axes) != 3:
        priority = lambda item: (item.priority, item.key.grid_index)
    else:

        def priority(item: Tile) -> tuple[float, tuple[int, ...]]:
            projection = _project_region(level.voxel_to_world, item.region, view)
            # Work in perceptual NDC units.  Euclidean screen distance grows
            # linearly from the visual center, while depth is normalized from
            # near=0 to far=1.  A large depth weight therefore fills the canvas
            # on near planes first; a small nonzero weight builds a central
            # column front-to-back without falling back to data-axis ordering.
            screen_distance = math.sqrt(max(0.0, projection.center_distance))
            normalized_depth = min(1.0, max(0.0, (projection.depth + 1.0) / 2.0))
            score = screen_distance + focus_depth_weight * normalized_depth
            return score, item.key.grid_index

    for tile in sorted(tiles, key=priority):
        candidate_start = (
            tile.region.start
            if start is None
            else tuple(min(a, b) for a, b in zip(start, tile.region.start, strict=True))
        )
        candidate_stop = (
            tile.region.stop
            if stop is None
            else tuple(max(a, b) for a, b in zip(stop, tile.region.stop, strict=True))
        )
        size = math.prod(
            upper - lower
            for lower, upper in zip(candidate_start, candidate_stop, strict=True)
        )
        if size * level.dtype.itemsize > memory_limit:
            continue
        selected.append(tile)
        start, stop = candidate_start, candidate_stop
    return selected


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
    __slots__ = ("center_distance", "depth", "priority", "visible")

    def __init__(
        self,
        visible: bool,
        priority: float,
        *,
        center_distance: float = math.inf,
        depth: float = math.inf,
    ) -> None:
        self.visible = visible
        self.priority = priority
        self.center_distance = center_distance
        self.depth = depth


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
        depth = 0.0
        priority = center_distance
    return _Projection(
        visible,
        priority,
        center_distance=center_distance,
        depth=depth,
    )
