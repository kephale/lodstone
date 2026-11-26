"""Regular chunk-grid axes, materialized or compact.

A regular axis of ``extent`` voxels in chunks of ``step`` is fully described
by those two numbers, yet the grid API hands out each chunk's size (a tuple)
and each boundary (an array). That is cheap for typical datasets, with tens
to thousands of chunks per axis, and is what they keep getting. A deep
pyramid's finest level can have millions or billions of chunks per axis
(2**26 at 26 levels of 32-voxel tiles: 512 MiB per tuple and per array), so
axes with more than :func:`compact_grid_min_chunks` chunks are represented
compactly instead, by :class:`RegularChunkSizes` and
:class:`RegularChunkBoundaries`, which compute entries on demand.

Code that needs sizes or boundaries of any axis should use the helpers here
(:func:`max_chunk_size`, :func:`searchsorted`, :func:`chunk_index_range`)
rather than iterating, slicing or converting with NumPy, which works for
compact axes too but costs as much as materializing them.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterator, Sequence
from typing import Any, overload

import numpy as np

#: Axes with more chunks than this are compact; set to 0 to make every
#: regular axis compact, or to a huge value to never use them. Read at each
#: grid construction, so tests and applications may change it at runtime.
COMPACT_GRID_MIN_CHUNKS = int(
    os.environ.get("LODSTONE_COMPACT_GRID_MIN_CHUNKS", str(1 << 16))
)


def compact_grid_min_chunks() -> int:
    """Chunk count along an axis above which regular axes are compact."""

    return COMPACT_GRID_MIN_CHUNKS


def _count(extent: int, step: int) -> int:
    return -(-extent // step)


class RegularChunkSizes(Sequence[int]):
    """The chunk sizes along a regular axis, computed on demand.

    Equal to the tuple it stands for, element by element. Iterating it, or
    comparing it with anything but another regular axis, takes time
    proportional to its length; use the module helpers instead.
    """

    __slots__ = ("extent", "step")

    def __init__(self, extent: int, step: int) -> None:
        if extent <= 0 or step <= 0:
            raise ValueError("shape and chunk dimensions must be positive")
        self.extent = int(extent)
        self.step = int(step)

    def __len__(self) -> int:
        return _count(self.extent, self.step)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        count = len(self)
        index = int(index)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError("chunk index out of range")
        return min(self.step, self.extent - index * self.step)

    def __iter__(self) -> Iterator[int]:
        full, remainder = divmod(self.extent, self.step)
        for _ in range(full):
            yield self.step
        if remainder:
            yield remainder

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RegularChunkSizes):
            return (self.extent, self.step) == (other.extent, other.step)
        if isinstance(other, Sequence) and not isinstance(other, str):
            return len(other) == len(self) and all(
                a == b for a, b in zip(self, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((RegularChunkSizes, self.extent, self.step))

    def __repr__(self) -> str:
        return f"RegularChunkSizes(extent={self.extent}, step={self.step})"


class RegularChunkBoundaries:
    """The chunk boundaries along a regular axis, zero through the extent.

    Indexes like the boundary array it stands for. ``np.asarray`` works but
    materializes it; use :func:`searchsorted` and integer indexing.
    """

    __slots__ = ("extent", "step")

    ndim = 1
    dtype = np.dtype(np.int64)

    def __init__(self, extent: int, step: int) -> None:
        if extent <= 0 or step <= 0:
            raise ValueError("shape and chunk dimensions must be positive")
        self.extent = int(extent)
        self.step = int(step)

    def __len__(self) -> int:
        return _count(self.extent, self.step) + 1

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            positions = np.arange(len(self), dtype=np.int64)[index]
            return np.minimum(positions * self.step, self.extent)
        index = int(index)
        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError("boundary index out of range")
        return min(index * self.step, self.extent)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        bounds = np.minimum(
            np.arange(len(self), dtype=np.int64) * self.step, self.extent
        )
        return bounds if dtype is None else bounds.astype(dtype)

    def searchsorted(self, value: float, side: str = "left") -> int:
        """Like ``np.searchsorted(boundaries, value, side)``."""

        if value <= 0:
            return 0 if value < 0 or side == "left" else 1
        if value > self.extent:
            return len(self)
        if value == self.extent:
            return len(self) - 1 if side == "left" else len(self)
        quotient, remainder = divmod(value, self.step)
        index = int(quotient)
        if remainder == 0 and side == "left":
            return index
        return index + 1

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RegularChunkBoundaries):
            return (self.extent, self.step) == (other.extent, other.step)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((RegularChunkBoundaries, self.extent, self.step))

    def __repr__(self) -> str:
        return f"RegularChunkBoundaries(extent={self.extent}, step={self.step})"


# Materialized regular axes are remembered by identity, so code handed one
# back can tell it is regular without reading every size.
_REGULAR_AXES: dict[int, tuple[tuple[int, ...], int, int]] = {}


@functools.lru_cache(maxsize=256)
def _materialized_axis(extent: int, step: int) -> tuple[int, ...]:
    full, remainder = divmod(extent, step)
    sizes = (step,) * full + ((remainder,) if remainder else ())
    # the registry holds the tuple, so its id is not reused
    _REGULAR_AXES[id(sizes)] = (sizes, extent, step)
    return sizes


@functools.lru_cache(maxsize=256)
def _materialized_boundaries(extent: int, step: int) -> np.ndarray:
    bounds = np.minimum(
        np.arange(_count(extent, step) + 1, dtype=np.int64) * step, extent
    )
    bounds.setflags(write=False)
    return bounds


def regular_axis_sizes(extent: int, step: int) -> Sequence[int]:
    """The chunk sizes along a regular axis: a cached tuple, or compact."""

    extent, step = int(extent), int(step)
    if extent <= 0 or step <= 0:
        raise ValueError("shape and chunk dimensions must be positive")
    if _count(extent, step) > compact_grid_min_chunks():
        return RegularChunkSizes(extent, step)
    return _materialized_axis(extent, step)


def regular_axis_boundaries(extent: int, step: int) -> Any:
    """The chunk boundaries along a regular axis: a cached array, or compact."""

    extent, step = int(extent), int(step)
    if extent <= 0 or step <= 0:
        raise ValueError("shape and chunk dimensions must be positive")
    if _count(extent, step) > compact_grid_min_chunks():
        return RegularChunkBoundaries(extent, step)
    return _materialized_boundaries(extent, step)


def known_regular_axis(sizes: Any) -> tuple[int, int] | None:
    """(extent, chunk size) of a regular axis built here, else None."""

    if isinstance(sizes, RegularChunkSizes):
        return sizes.extent, sizes.step
    entry = _REGULAR_AXES.get(id(sizes))
    if entry is None or entry[0] is not sizes:
        return None
    return entry[1], entry[2]


def max_chunk_size(sizes: Sequence[int]) -> int:
    """The largest chunk along an axis."""

    regular = known_regular_axis(sizes)
    if regular is not None:
        extent, step = regular
        return min(extent, step)
    return max(sizes)


def searchsorted(boundaries: Any, value: float, side: str = "left") -> int:
    """``np.searchsorted`` for materialized or compact boundaries."""

    if isinstance(boundaries, RegularChunkBoundaries):
        return boundaries.searchsorted(value, side)
    return int(np.searchsorted(boundaries, value, side=side))  # type: ignore[call-overload]


def chunk_index_range(boundaries: Any, start: float, stop: float) -> range:
    """Indices of the chunks intersecting ``[start, stop)``."""

    count = len(boundaries) - 1
    # the first chunk whose stop exceeds start, the first whose start
    # reaches stop
    first = min(max(searchsorted(boundaries, start, "right") - 1, 0), count)
    last = min(max(searchsorted(boundaries, stop, "left"), 0), count)
    return range(first, last)
