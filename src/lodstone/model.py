"""Small immutable values shared by Lodstone sources, planners, and targets."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Region:
    """A half-open integer region in data-axis order."""

    start: tuple[int, ...]
    stop: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.start) != len(self.stop):
            raise ValueError("region start and stop must have equal dimensionality")
        if any(a < 0 or b < a for a, b in zip(self.start, self.stop, strict=True)):
            raise ValueError("regions require 0 <= start <= stop on every axis")

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.start, self.stop, strict=True))

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop, strict=True))

    def intersection(self, other: Region) -> Region | None:
        if self.ndim != other.ndim:
            raise ValueError("cannot intersect regions of different dimensionality")
        start = tuple(max(a, b) for a, b in zip(self.start, other.start, strict=True))
        stop = tuple(min(a, b) for a, b in zip(self.stop, other.stop, strict=True))
        if any(a >= b for a, b in zip(start, stop, strict=True)):
            return None
        return Region(start, stop)


@dataclass(frozen=True, slots=True)
class Level:
    """Metadata for one pyramid level, ordered finest to coarsest."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunks: tuple[int, ...]
    voxel_to_world: npt.NDArray[np.float64]
    chunk_grid: tuple[tuple[int, ...], ...] | None = None
    fill_value: Any = 0

    def __post_init__(self) -> None:
        ndim = len(self.shape)
        if ndim == 0 or len(self.chunks) != ndim:
            raise ValueError(
                "shape and chunks must have equal, non-zero dimensionality"
            )
        if any(value <= 0 for value in (*self.shape, *self.chunks)):
            raise ValueError("shape and chunk dimensions must be positive")
        grid = self.chunk_grid
        if grid is not None:
            if len(grid) != ndim:
                raise ValueError(
                    "chunk grid must have one sequence per array dimension"
                )
            normalized_grid = tuple(
                tuple(int(value) for value in axis) for axis in grid
            )
            if any(
                not axis or any(value <= 0 for value in axis)
                for axis in normalized_grid
            ):
                raise ValueError("chunk grid dimensions must be non-empty and positive")
            if any(
                sum(axis) != size
                for axis, size in zip(normalized_grid, self.shape, strict=True)
            ):
                raise ValueError("chunk grid sizes must exactly cover the level shape")
            object.__setattr__(self, "chunk_grid", normalized_grid)
        matrix = np.asarray(self.voxel_to_world, dtype=np.float64)
        if matrix.shape != (ndim + 1, ndim + 1):
            raise ValueError(
                f"voxel_to_world must have shape {(ndim + 1, ndim + 1)}, "
                f"not {matrix.shape}"
            )
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "voxel_to_world", matrix)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def chunk_sizes(self, axis: int) -> tuple[int, ...]:
        """Return every native chunk size along ``axis``."""

        if self.chunk_grid is not None:
            return self.chunk_grid[axis]
        size = self.chunks[axis]
        count, remainder = divmod(self.shape[axis], size)
        return (size,) * count + ((remainder,) if remainder else ())

    def chunk_bounds(self, axis: int, index: int) -> tuple[int, int]:
        """Return the half-open bounds of one native chunk."""

        sizes = self.chunk_sizes(axis)
        if not 0 <= index < len(sizes):
            raise IndexError("chunk index is outside the native chunk grid")
        start = sum(sizes[:index])
        return start, start + sizes[index]

    def chunk_index(self, axis: int, coordinate: int) -> int:
        """Return the native chunk containing a data coordinate."""

        if not 0 <= coordinate < self.shape[axis]:
            raise IndexError("coordinate is outside the level")
        boundaries = np.cumsum(self.chunk_sizes(axis)).tolist()
        return bisect.bisect_right(boundaries, coordinate)


@dataclass(frozen=True, slots=True)
class Pyramid:
    """Axes and levels describing a multiscale array."""

    axes: tuple[str, ...]
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        if not self.axes or not self.levels:
            raise ValueError("a pyramid requires at least one axis and one level")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("pyramid axis names must be unique")
        ndim = len(self.axes)
        if any(level.ndim != ndim for level in self.levels):
            raise ValueError("all pyramid levels must match the number of axes")

    @property
    def ndim(self) -> int:
        return len(self.axes)


@dataclass(frozen=True, slots=True)
class InteractionState:
    """Optional motion metadata supplied by an interactive viewer."""

    moving: bool = False
    angular_velocity: float = 0.0
    translation_velocity: float = 0.0
    zoom_velocity: float = 0.0

    def __post_init__(self) -> None:
        velocities = (
            self.angular_velocity,
            self.translation_velocity,
            self.zoom_velocity,
        )
        if any(not np.isfinite(value) for value in velocities):
            raise ValueError("interaction velocities must be finite")


@dataclass(frozen=True, slots=True)
class View:
    """A host-neutral snapshot of a 2-D or 3-D viewer camera and selection.

    ``world_to_clip`` consumes coordinates ordered like ``displayed_axes``.
    For a 2-D view the third coordinate is zero. ``index`` contains ``None``
    for displayed axes and an integer selection for every other axis.
    """

    displayed_axes: tuple[int, ...]
    index: tuple[int | None, ...]
    viewport: tuple[int, int]
    world_to_clip: npt.NDArray[np.float64]
    eye: tuple[float, float, float] | None = None
    interaction: InteractionState | None = None

    def __post_init__(self) -> None:
        if len(self.displayed_axes) not in (2, 3):
            raise ValueError("displayed_axes must contain two or three axes")
        if len(set(self.displayed_axes)) != len(self.displayed_axes):
            raise ValueError("displayed_axes must be unique")
        if any(size <= 0 for size in self.viewport):
            raise ValueError("viewport dimensions must be positive")
        if any(axis < 0 or axis >= len(self.index) for axis in self.displayed_axes):
            raise ValueError("displayed axis is outside index dimensionality")
        displayed = set(self.displayed_axes)
        if any(
            (axis in displayed) != (value is None)
            for axis, value in enumerate(self.index)
        ):
            raise ValueError("index must be None exactly on the displayed axes")
        matrix = np.asarray(self.world_to_clip, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("world_to_clip must be a 4 by 4 matrix")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "world_to_clip", matrix)


@dataclass(frozen=True, slots=True)
class Layout:
    """The storage layout a rendering target can accept."""

    kind: Literal["dense", "tiled", "bricked"] = "dense"
    block_shape: tuple[int, ...] | None = None
    mixed_lod: bool = False
    memory_limit: int = 1 << 30
    squeeze_hidden: bool = True
    max_axis_extent: int | None = None
    memory_policy: Literal["coarsen", "crop"] = "coarsen"
    focus_depth_weight: float | None = None

    def __post_init__(self) -> None:
        if self.block_shape is not None and any(
            value <= 0 for value in self.block_shape
        ):
            raise ValueError("block_shape dimensions must be positive")
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        if self.max_axis_extent is not None and self.max_axis_extent <= 0:
            raise ValueError("max_axis_extent must be positive")
        if self.memory_policy not in {"coarsen", "crop"}:
            raise ValueError("memory_policy must be 'coarsen' or 'crop'")
        if self.focus_depth_weight is not None and (
            not np.isfinite(self.focus_depth_weight) or self.focus_depth_weight < 0
        ):
            raise ValueError("focus_depth_weight must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TileKey:
    """Stable identity for a display tile and its hidden-axis selection."""

    level: int
    grid_index: tuple[int, ...]
    selection: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Tile:
    """A target tile requested by a plan."""

    key: TileKey
    region: Region
    priority: float
    phase: int = 0

    @property
    def level(self) -> int:
        return self.key.level


@dataclass(frozen=True, slots=True)
class PlanCoverage:
    """Order-independent identity of the data covered by a plan.

    Tile priority and progressive phase are intentionally excluded.  Hidden-axis
    selections are included explicitly as well as through :class:`TileKey`, so
    integrations can inspect them without decoding tile identities.
    """

    target_level: int
    tile_regions: frozenset[tuple[TileKey, Region]]
    retained_keys: frozenset[TileKey]
    hidden_axis_selections: frozenset[tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class PlanDelta:
    """Stable coverage changes between two successive plans."""

    retained: frozenset[TileKey]
    requested: tuple[Tile, ...]
    reprioritized: tuple[TileKey, ...]
    released: frozenset[TileKey]


@dataclass(frozen=True, slots=True)
class Plan:
    """A complete desired tile set and the reads needed to reach it.

    ``desired`` includes every progressive ladder level for the current
    pass, whether already available or newly requested. ``wanted`` is the
    ordered subset that still needs to be read. ``retain`` describes target
    residency after the pass completes.
    """

    wanted: tuple[Tile, ...]
    retain: frozenset[TileKey]
    target_level: int
    desired: tuple[Tile, ...] = ()

    @property
    def coverage(self) -> PlanCoverage:
        """Return the requested coverage, ignoring delivery order and priority."""

        tiles = self.desired or self.wanted
        keys = (*[tile.key for tile in tiles], *self.retain)
        return PlanCoverage(
            target_level=self.target_level,
            tile_regions=frozenset((tile.key, tile.region) for tile in tiles),
            retained_keys=self.retain,
            hidden_axis_selections=frozenset(key.selection for key in keys),
        )

    def delta(self, previous: Plan | None) -> PlanDelta:
        """Compare stable coverage while preserving current request order."""

        current_tiles = self.desired or self.wanted
        current = {tile.key: tile for tile in current_tiles}
        if previous is None:
            return PlanDelta(
                retained=frozenset(),
                requested=tuple(self.wanted),
                reprioritized=(),
                released=frozenset(),
            )
        previous_tiles = previous.desired or previous.wanted
        old = {tile.key: tile for tile in previous_tiles}
        retained = frozenset(
            key
            for key, tile in current.items()
            if key in old and old[key].region == tile.region
        )
        requested = tuple(tile for tile in self.wanted if tile.key not in retained)
        reprioritized = tuple(
            tile.key
            for tile in current_tiles
            if tile.key in retained
            and (
                old[tile.key].priority != tile.priority
                or old[tile.key].phase != tile.phase
            )
        )
        released = frozenset(
            key
            for key, tile in old.items()
            if key not in retained or key not in self.retain
        )
        return PlanDelta(retained, requested, reprioritized, released)


@dataclass(frozen=True, slots=True)
class Update:
    """One display-ready array region delivered to a target."""

    key: TileKey
    region: Region
    data: np.ndarray
    transform: npt.NDArray[np.float64]

    @property
    def level(self) -> int:
        return self.key.level


@dataclass(frozen=True, slots=True)
class Status:
    """A compact observable snapshot of a stream."""

    generation: int = 0
    state: Literal["idle", "loading", "complete", "failed", "closed"] = "idle"
    wanted: int = 0
    resident: int = 0
    inflight: int = 0
    bytes_read: int = 0
    progress: float = 0.0
    error: BaseException | None = None


class ChunkState(StrEnum):
    """Observable lifecycle state for one decoded native chunk."""

    NEW = "new"
    QUEUED = "queued"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class ChunkEvent:
    """One native-chunk cache transition."""

    generation: int
    key: tuple[int, tuple[int, ...]]
    previous: ChunkState
    current: ChunkState
    reason: str


@dataclass(frozen=True, slots=True)
class StreamDiagnostics:
    """Plan and native-read counters for one stream generation."""

    generation: int = 0
    desired_tiles: int = 0
    wanted_tiles: int = 0
    unique_native_chunks: int = 0
    cache_hits: int = 0
    joined_reads: int = 0
    source_reads: int = 0
    evictions: int = 0
    cache_chunks: int = 0
    cache_bytes: int = 0
    prepare_stage_seconds: float = 0.0
    update_stage_seconds: float = 0.0
    phase_stage_seconds: float = 0.0


def identity_transform(ndim: int) -> npt.NDArray[np.float64]:
    """Return an identity homogeneous transform for ``ndim`` data axes."""

    return np.eye(ndim + 1, dtype=np.float64)
