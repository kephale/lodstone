"""Renderer-neutral native chunk-grid normalization and queries."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np

from .model import Region

ChunkGrid = tuple[tuple[int, ...], ...]


def chunk_key_id(key: Sequence[slice]) -> tuple[tuple[int, int], ...]:
    """Return stable absolute bounds for a concrete chunk slice key."""

    return tuple((int(item.start), int(item.stop)) for item in key)


def regular_chunk_sizes(
    shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> ChunkGrid:
    """Expand a regular chunk shape into clipped sizes along every axis."""

    if len(shape) != len(chunk_shape):
        raise ValueError("shape and chunk shape must have equal dimensionality")
    return tuple(
        _regular_axis_sizes(int(size), int(step))
        for size, step in zip(shape, chunk_shape, strict=True)
    )


# A deep pyramid's finest level has millions of chunks per axis, and every
# view of a level asks for its grid: build each regular axis once.
@functools.lru_cache(maxsize=256)
def _regular_axis_sizes(size: int, step: int) -> tuple[int, ...]:
    if size <= 0 or step <= 0:
        raise ValueError("shape and chunk dimensions must be positive")
    count, remainder = divmod(size, step)
    return (step,) * count + ((remainder,) if remainder else ())


@functools.lru_cache(maxsize=256)
def _regular_axis_boundaries(size: int, step: int) -> np.ndarray:
    if size <= 0 or step <= 0:
        raise ValueError("shape and chunk dimensions must be positive")
    bounds = np.minimum(np.arange(0, size + step, step, dtype=np.int64), size)
    if len(bounds) > 1 and bounds[-1] == bounds[-2]:
        bounds = bounds[:-1]
    bounds.setflags(write=False)
    return bounds


def regular_chunk_shape(array: Any) -> tuple[int, ...] | None:
    """Return the chunk shape of an array with a regular chunk grid, or None.

    A regular grid is described by one chunk shape, so its per-axis sizes
    need not be read chunk by chunk: Zarr's ``read_chunk_sizes`` lists every
    chunk, millions per axis for a deep pyramid's finest level. An array's
    ``read_chunk_sizes`` stays authoritative unless its Zarr metadata
    declares a regular grid (Zarr v2 grids always are); ``chunks`` is then
    the read chunk shape (the inner one when sharded).
    """

    if hasattr(type(array), "read_chunk_sizes") or "read_chunk_sizes" in getattr(
        array, "__dict__", {}
    ):
        metadata = getattr(array, "metadata", None)
        grid = getattr(metadata, "chunk_grid", None)
        # RegularChunkGrid before Zarr 3.4, RegularChunkGridMetadata since
        regular_v3 = grid is not None and type(grid).__name__.startswith(
            "RegularChunkGrid"
        )
        if not (regular_v3 or type(metadata).__name__ == "ArrayV2Metadata"):
            return None
    try:
        chunks = getattr(array, "chunks", None)
    except NotImplementedError:  # Zarr: unsharded rectilinear grid
        return None
    if chunks is None or len(chunks) != len(array.shape):
        return None
    if not all(isinstance(value, (int, np.integer)) for value in chunks):
        return None
    shape = tuple(int(value) for value in chunks)
    if any(value <= 0 for value in shape):
        return None
    return shape


def normalize_chunk_sizes(
    shape: Sequence[int],
    chunks: Sequence[int] | Sequence[Sequence[int]],
) -> ChunkGrid:
    """Normalize regular or rectilinear chunk metadata to axis sizes."""

    if len(shape) != len(chunks):
        raise ValueError("shape and chunks must have equal dimensionality")
    if all(isinstance(value, (int, np.integer)) for value in chunks):
        return regular_chunk_sizes(shape, chunks)  # type: ignore[arg-type]
    grid = tuple(
        tuple(int(value) for value in axis)
        for axis in cast(Sequence[Sequence[int]], chunks)
    )
    if any(not axis or any(value <= 0 for value in axis) for axis in grid):
        raise ValueError("chunk grid dimensions must be non-empty and positive")
    if any(sum(axis) != int(size) for axis, size in zip(grid, shape, strict=True)):
        raise ValueError("chunk grid sizes must exactly cover the array shape")
    return grid


def chunk_sizes_for(array: Any, *, fallback: int = 256) -> ChunkGrid:
    """Return exact clipped native chunk sizes for an array-like object.

    Dask-style rectilinear grids, regular Zarr grids, and Zarr's
    ``read_chunk_sizes`` extension are supported. Plain arrays receive a
    bounded regular fallback grid.
    """

    regular = regular_chunk_shape(array)
    if regular is not None:
        return regular_chunk_sizes(array.shape, regular)
    sizes = getattr(array, "read_chunk_sizes", None)
    if sizes is not None:
        return normalize_chunk_sizes(array.shape, sizes)
    chunks = getattr(array, "chunks", None)
    if chunks is not None:
        return normalize_chunk_sizes(array.shape, chunks)
    return regular_chunk_sizes(
        array.shape,
        tuple(min(int(size), fallback) for size in array.shape),
    )


def chunk_shape_for(array: Any, *, fallback: int = 256) -> tuple[int, ...]:
    """Return the largest native chunk size along each array axis."""

    chunksize = getattr(array, "chunksize", None)
    if chunksize is not None:
        return tuple(int(value) for value in chunksize)
    regular = regular_chunk_shape(array)
    if regular is not None:
        return tuple(
            min(int(step), int(size))
            for step, size in zip(regular, array.shape, strict=True)
        )
    return tuple(
        max(axis) if axis else 1 for axis in chunk_sizes_for(array, fallback=fallback)
    )


def chunk_boundaries(array: Any, *, fallback: int = 256) -> tuple[np.ndarray, ...]:
    """Return integer boundary positions from zero through each axis extent."""

    regular = regular_chunk_shape(array)
    if regular is not None:
        return tuple(
            _regular_axis_boundaries(int(size), int(step))
            for size, step in zip(array.shape, regular, strict=True)
        )
    return tuple(
        np.concatenate(([0], np.cumsum(axis, dtype=np.int64)))
        for axis in chunk_sizes_for(array, fallback=fallback)
    )


def chunk_ids_in_region(
    boundaries: Sequence[np.ndarray],
    start: Sequence[int],
    stop: Sequence[int],
) -> Iterable[tuple[tuple[int, int], ...]]:
    """Iterate absolute native chunk bounds intersecting ``[start, stop)``."""

    if not (len(boundaries) == len(start) == len(stop)):
        raise ValueError("boundaries and region must have equal dimensionality")
    per_axis = []
    for axis, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        first = int(np.searchsorted(stops, int(start[axis]), side="right"))
        last = int(np.searchsorted(starts, int(stop[axis]), side="left"))
        per_axis.append(
            tuple((int(starts[i]), int(stops[i])) for i in range(first, last))
        )
    return itertools.product(*per_axis)


def chunk_slices_for(
    array: Any,
    region: Region | tuple[Sequence[int], Sequence[int]] | None = None,
) -> tuple[tuple[slice, ...], ...]:
    """Return per-axis native chunk slices, optionally intersecting a region."""

    boundaries = chunk_boundaries(array)
    if region is None:
        start = (0,) * len(boundaries)
        stop = tuple(int(bounds[-1]) for bounds in boundaries)
    elif isinstance(region, Region):
        start, stop = region.start, region.stop
    else:
        start, stop = region
    per_axis = []
    for axis, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        first = int(np.searchsorted(stops, int(start[axis]), side="right"))
        last = int(np.searchsorted(starts, int(stop[axis]), side="left"))
        per_axis.append(
            tuple(
                slice(int(starts[index]), int(stops[index]))
                for index in range(first, last)
            )
        )
    return tuple(per_axis)
