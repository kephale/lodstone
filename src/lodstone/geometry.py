"""Memory- and target-constrained dense region geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import Region


def isotropic_extent_for_bytes(
    dtype: np.dtype | str | type,
    max_bytes: int,
    *,
    ndim: int = 3,
    max_axis_extent: int | None = None,
    minimum: int = 1,
) -> int:
    """Largest isotropic integer extent fitting a dense byte budget."""

    if max_bytes <= 0 or ndim <= 0 or minimum <= 0:
        raise ValueError("byte budget, dimensionality, and minimum must be positive")
    itemsize = max(np.dtype(dtype).itemsize, 1)
    max_elements = max_bytes // itemsize
    extent = round(max_elements ** (1.0 / ndim))
    while (extent + 1) ** ndim <= max_elements:
        extent += 1
    while extent**ndim > max_elements:
        extent -= 1
    if max_axis_extent is not None:
        if max_axis_extent <= 0:
            raise ValueError("max_axis_extent must be positive")
        extent = min(extent, int(max_axis_extent))
    return max(extent, minimum)


def anisotropic_extent_for_bytes(
    shape: Sequence[int],
    max_bytes: int,
    itemsize: int,
    *,
    max_axis_extent: int | None = None,
) -> tuple[int, ...]:
    """Fit a dense extent to a budget while preserving short axes."""

    if max_bytes <= 0 or itemsize <= 0:
        raise ValueError("byte budget and itemsize must be positive")
    extent = np.asarray(shape, dtype=np.int64)
    if extent.size == 0 or np.any(extent <= 0):
        raise ValueError("shape dimensions must be positive")
    if max_axis_extent is not None:
        if max_axis_extent <= 0:
            raise ValueError("max_axis_extent must be positive")
        extent = np.minimum(extent, int(max_axis_extent))
    result = extent.copy()
    max_elements = max(max_bytes // itemsize, 1)
    for _ in range(len(result)):
        volume = int(np.prod(result, dtype=np.int64))
        if volume <= max_elements:
            break
        shrinkable = np.where(result > 1)[0]
        if len(shrinkable) == 0:
            break
        ratio = (max_elements / volume) ** (1.0 / len(shrinkable))
        for axis in shrinkable:
            result[axis] = max(int(result[axis] * ratio), 1)
    return tuple(int(value) for value in np.maximum(result, 1))


def clamp_region_to_budget(
    region: Region,
    shape: Sequence[int],
    *,
    itemsize: int,
    max_bytes: int,
    max_axis_extent: int | None = None,
) -> Region:
    """Shrink a region around its center to satisfy dense target limits."""

    if region.ndim != len(shape):
        raise ValueError("region and shape must have equal dimensionality")
    if itemsize <= 0 or max_bytes <= 0:
        raise ValueError("itemsize and byte budget must be positive")
    start = np.asarray(region.start, dtype=np.int64)
    stop = np.asarray(region.stop, dtype=np.int64)
    shape_array = np.asarray(shape, dtype=np.int64)
    extent = np.maximum(stop - start, 1)

    if max_axis_extent is not None:
        if max_axis_extent <= 0:
            raise ValueError("max_axis_extent must be positive")
        for axis in range(len(extent)):
            if extent[axis] <= max_axis_extent:
                continue
            center = (start[axis] + stop[axis]) // 2
            half = max_axis_extent // 2
            start[axis] = max(center - half, 0)
            stop[axis] = min(start[axis] + max_axis_extent, shape_array[axis])
            start[axis] = max(stop[axis] - max_axis_extent, 0)
            extent[axis] = stop[axis] - start[axis]

    max_elements = max(max_bytes // itemsize, 1)
    while int(np.prod(extent, dtype=np.int64)) > max_elements:
        axis = int(np.argmax(extent))
        center = (start[axis] + stop[axis]) // 2
        half = max(extent[axis] // 4, 1)
        new_start = max(center - half, start[axis])
        new_stop = min(center + half, stop[axis])
        new_extent = max(new_stop - new_start, 1)
        if new_extent == extent[axis]:
            break
        start[axis] = new_start
        stop[axis] = new_stop
        extent[axis] = new_extent
    return Region(tuple(int(value) for value in start), tuple(int(value) for value in stop))
