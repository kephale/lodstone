"""Transform-aware composition of coarse data into resident detail windows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from itertools import product

import numpy as np

from .chunks import ChunkGrid
from .model import Region


def nearest_resample_region(
    source: np.ndarray,
    source_region: Region,
    source_transform: np.ndarray,
    destination_region: Region,
    destination_transform: np.ndarray,
) -> np.ndarray:
    """Sample an axis-aligned source window into a destination region.

    Transforms map voxel coordinates to the same world coordinate system.
    Destination voxel centers are mapped through world space and sampled
    nearest-neighbor from ``source``. Coordinates beyond the resident source
    window clamp to its edge, matching persistent coarse-backdrop behavior.
    """

    if source_region.ndim != destination_region.ndim:
        raise ValueError("source and destination regions must have equal dimensions")
    ndim = source_region.ndim
    if tuple(source.shape) != source_region.shape:
        raise ValueError("source array shape must equal its resident region")
    source_transform = _axis_aligned_transform(source_transform, ndim, "source")
    destination_transform = _axis_aligned_transform(
        destination_transform, ndim, "destination"
    )
    if source.size == 0 or destination_region.size == 0:
        return np.empty(destination_region.shape, dtype=source.dtype)

    source_scale = np.diag(source_transform)[:ndim]
    source_offset = source_transform[:ndim, ndim]
    destination_scale = np.diag(destination_transform)[:ndim]
    destination_offset = destination_transform[:ndim, ndim]
    window = []
    repeats = []
    for axis in range(ndim):
        coordinates = np.arange(
            destination_region.start[axis],
            destination_region.stop[axis],
            dtype=np.float64,
        )
        world = (coordinates + 0.5) * destination_scale[axis] + destination_offset[axis]
        source_coordinates = np.floor(
            (world - source_offset[axis]) / source_scale[axis]
        ).astype(np.int64)
        relative = source_coordinates - source_region.start[axis]
        index = np.clip(relative, 0, source.shape[axis] - 1)
        # Positive scales make each axis's indices non-decreasing, so the
        # gather is a contiguous source window with every sample repeated
        # (or skipped, when downsampling) a whole number of times. Slicing
        # and repeating per axis is several times faster than an
        # ``np.ix_`` fancy-index gather over the whole destination.
        low, high = int(index[0]), int(index[-1]) + 1
        window.append(slice(low, high))
        repeats.append(np.bincount(index - low, minlength=high - low))
    result = source[tuple(window)]
    copied = False
    for axis, counts in enumerate(repeats):
        if not np.all(counts == 1):
            result = np.repeat(result, counts, axis=axis)
            copied = True
    # Never hand back a view: callers write the backdrop into resident
    # buffers, and a view would alias the source level.
    return result if copied else result.copy()


def fill_unloaded_chunks(
    destination: np.ndarray,
    destination_region: Region,
    content: np.ndarray,
    content_region: Region,
    chunk_grid: ChunkGrid,
    loaded: Collection[Region],
    *,
    boundaries: Sequence[np.ndarray] | None = None,
) -> tuple[Region, ...]:
    """Copy backdrop content into native chunks not present in ``loaded``.

    Both arrays are addressed in absolute level coordinates by their regions.
    Partial chunks at the edge of ``content_region`` are clipped consistently.
    The returned regions identify native chunks whose unloaded portion changed.

    ``boundaries``, when given, are the grid's chunk boundary positions per
    axis (as from ``chunk_boundaries``), saving their derivation from
    ``chunk_grid``: for a deep pyramid's finest level that is millions of
    sizes per axis, on every call.
    """

    ndim = destination_region.ndim
    if not (
        content_region.ndim == ndim
        and len(chunk_grid) == ndim
        and tuple(destination.shape) == destination_region.shape
        and tuple(content.shape) == content_region.shape
    ):
        raise ValueError("arrays, regions, and chunk grid must have equal dimensions")
    overlap = destination_region.intersection(content_region)
    if overlap is None:
        return ()

    if boundaries is None:
        boundaries = [np.concatenate(([0], np.cumsum(axis))) for axis in chunk_grid]
    elif len(boundaries) != ndim:
        raise ValueError("boundaries must have one array per dimension")
    loaded_set = set(loaded)

    def chunk_range(axis: int, start: int, stop: int) -> range:
        starts, stops = boundaries[axis][:-1], boundaries[axis][1:]
        first = int(np.searchsorted(stops, start, side="right"))
        last = int(np.searchsorted(starts, stop, side="left"))
        return range(first, last)

    # Index loaded regions by the native chunks they touch, so each chunk
    # tests only its own neighbours. Scanning every loaded region for every
    # chunk is quadratic: a 3D tile with hundreds of loaded chunks took
    # seconds, all of it under the caller's lock on the resident level.
    touching: dict[tuple[int, ...], list[Region]] = defaultdict(list)
    for region in loaded_set:
        if region.ndim != ndim:
            continue
        for index in product(
            *(
                chunk_range(axis, region.start[axis], region.stop[axis])
                for axis in range(ndim)
            )
        ):
            touching[index].append(region)

    filled = []
    for index in product(
        *(
            chunk_range(axis, overlap.start[axis], overlap.stop[axis])
            for axis in range(ndim)
        )
    ):
        chunk = Region(
            tuple(int(boundaries[axis][i]) for axis, i in enumerate(index)),
            tuple(int(boundaries[axis][i + 1]) for axis, i in enumerate(index)),
        )
        nearby = touching.get(index, ())
        if chunk in loaded_set or any(
            region.intersection(chunk) == chunk for region in nearby
        ):
            continue
        write = chunk.intersection(overlap)
        if write is None:
            continue
        destination_key = _relative_slices(write, destination_region)
        content_key = _relative_slices(write, content_region)
        # Loaded tiles may be smaller than native chunks; several of them can
        # cover one chunk jointly or partially. Fill only uncovered voxels so
        # recomposing a reused window never overwrites loaded detail.
        covered = np.zeros(write.shape, dtype=bool)
        for region in nearby:
            part = region.intersection(write)
            if part is not None:
                covered[_relative_slices(part, write)] = True
        if covered.all():
            continue
        if covered.any():
            uncovered = ~covered
            destination[destination_key][uncovered] = content[content_key][uncovered]
        else:
            destination[destination_key] = content[content_key]
        filled.append(chunk)
    return tuple(filled)


def _relative_slices(region: Region, container: Region) -> tuple[slice, ...]:
    return tuple(
        slice(
            region.start[axis] - container.start[axis],
            region.stop[axis] - container.start[axis],
        )
        for axis in range(region.ndim)
    )


def _axis_aligned_transform(
    transform: np.ndarray,
    ndim: int,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError(f"{name} transform has the wrong shape")
    linear = matrix[:ndim, :ndim]
    if not np.allclose(linear, np.diag(np.diag(linear))):
        raise ValueError(f"{name} transform must be axis aligned")
    if np.any(np.diag(linear) <= 0):
        raise ValueError(f"{name} transform scales must be positive")
    return matrix
