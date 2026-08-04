"""Small immutable values shared by Lodstone sources, planners, and targets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Literal

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

    def __post_init__(self) -> None:
        ndim = len(self.shape)
        if ndim == 0 or len(self.chunks) != ndim:
            raise ValueError(
                "shape and chunks must have equal, non-zero dimensionality"
            )
        if any(value <= 0 for value in (*self.shape, *self.chunks)):
            raise ValueError("shape and chunk dimensions must be positive")
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

    def __post_init__(self) -> None:
        if self.block_shape is not None and any(
            value <= 0 for value in self.block_shape
        ):
            raise ValueError("block_shape dimensions must be positive")
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")


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
class Plan:
    """Ordered desired tiles and the target level chosen for the view."""

    wanted: tuple[Tile, ...]
    retain: frozenset[TileKey]
    target_level: int


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


def identity_transform(ndim: int) -> npt.NDArray[np.float64]:
    """Return an identity homogeneous transform for ``ndim`` data axes."""

    return np.eye(ndim + 1, dtype=np.float64)
