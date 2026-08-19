"""Transform-aware composition of coarse data into resident detail windows."""

from __future__ import annotations

from collections.abc import Collection
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
    indices = []
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
        indices.append(np.clip(relative, 0, source.shape[axis] - 1))
    return source[np.ix_(*indices)]


def fill_unloaded_chunks(
    destination: np.ndarray,
    destination_region: Region,
    content: np.ndarray,
    content_region: Region,
    chunk_grid: ChunkGrid,
    loaded: Collection[Region],
) -> tuple[Region, ...]:
    """Copy backdrop content into native chunks not present in ``loaded``.

    Both arrays are addressed in absolute level coordinates by their regions.
    Partial chunks at the edge of ``content_region`` are clipped consistently.
    The returned regions identify native chunks whose unloaded portion changed.
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

    boundaries = [np.concatenate(([0], np.cumsum(axis))) for axis in chunk_grid]
    loaded_set = set(loaded)
    per_axis = []
    for axis, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        first = int(np.searchsorted(stops, overlap.start[axis], side="right"))
        last = int(np.searchsorted(starts, overlap.stop[axis], side="left"))
        per_axis.append(
            tuple((int(starts[i]), int(stops[i])) for i in range(first, last))
        )

    filled = []
    for bounds in product(*per_axis):
        chunk = Region(
            tuple(start for start, _stop in bounds),
            tuple(stop for _start, stop in bounds),
        )
        if chunk in loaded_set:
            continue
        write = chunk.intersection(overlap)
        if write is None:
            continue
        destination_key = _relative_slices(write, destination_region)
        content_key = _relative_slices(write, content_region)
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
