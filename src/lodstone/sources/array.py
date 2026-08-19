"""Source adapter for NumPy-like pyramid levels."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..chunks import chunk_shape_for, chunk_sizes_for, normalize_chunk_sizes
from ..model import Level, Pyramid, Region, identity_transform


class FixedAxisArray:
    """Lazy fixed-axis view over an indexable array-like object."""

    def __init__(self, array: Any, fixed_index: Mapping[int, int]) -> None:
        self._array = array
        self._fixed = {int(axis): int(index) for axis, index in fixed_index.items()}
        ndim = len(array.shape)
        if any(axis < 0 or axis >= ndim for axis in self._fixed):
            raise ValueError("fixed axis is outside the source dimensionality")
        if any(
            index < 0 or index >= int(array.shape[axis])
            for axis, index in self._fixed.items()
        ):
            raise ValueError("fixed index is outside the source shape")
        self.source_axes = tuple(
            axis for axis in range(ndim) if axis not in self._fixed
        )
        if not self.source_axes:
            raise ValueError("at least one non-fixed axis is required")
        self.shape = tuple(int(array.shape[axis]) for axis in self.source_axes)
        self.dtype = np.dtype(array.dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.fill_value = getattr(array, "fill_value", 0)

        source_grid = chunk_sizes_for(array)
        self.read_chunk_sizes = tuple(source_grid[axis] for axis in self.source_axes)
        self.chunksize = tuple(max(axis) for axis in self.read_chunk_sizes)

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        if any(item is Ellipsis for item in key):
            position = key.index(Ellipsis)
            missing = self.ndim - (len(key) - 1)
            key = (*key[:position], *(slice(None),) * missing, *key[position + 1 :])
        if len(key) > self.ndim:
            raise IndexError("too many indices for fixed-axis array view")
        key = (*key, *(slice(None),) * (self.ndim - len(key)))
        visible = iter(key)
        source_key = tuple(
            self._fixed[axis] if axis in self._fixed else next(visible)
            for axis in range(len(self._array.shape))
        )
        return self._array[source_key]


class ArrayPyramidSource:
    """Expose a sequence of indexable arrays as a Lodstone source."""

    def __init__(
        self,
        levels: Sequence[Any],
        *,
        axes: Sequence[str] | None = None,
        transforms: Sequence[np.ndarray] | None = None,
        chunks: Sequence[tuple[int, ...] | tuple[tuple[int, ...], ...]] | None = None,
    ) -> None:
        if not levels:
            raise ValueError("at least one array level is required")
        self._arrays = tuple(levels)
        ndim = len(self._arrays[0].shape)
        axis_names = tuple(axes or (f"axis_{i}" for i in range(ndim)))
        matrices = tuple(transforms or (identity_transform(ndim) for _ in levels))
        if len(matrices) != len(levels):
            raise ValueError("one transform is required per level")
        if chunks is not None and len(chunks) != len(levels):
            raise ValueError("one chunk shape is required per level")

        metadata = []
        for index, (array, matrix) in enumerate(
            zip(self._arrays, matrices, strict=True)
        ):
            shape = tuple(int(value) for value in array.shape)
            if len(shape) != ndim:
                raise ValueError("all arrays must have equal dimensionality")
            native = None if chunks is None else chunks[index]
            if native is None:
                chunk_grid = chunk_sizes_for(array)
                chunk_shape = chunk_shape_for(array)
            else:
                chunk_grid = normalize_chunk_sizes(shape, native)
                chunk_shape = tuple(max(axis) for axis in chunk_grid)
            fill_value = getattr(array, "fill_value", 0)
            if fill_value is None:
                fill_value = 0
            metadata.append(
                Level(
                    shape,
                    np.dtype(array.dtype),
                    chunk_shape,
                    np.asarray(matrix),
                    chunk_grid,
                    fill_value,
                )
            )
        self._pyramid = Pyramid(axis_names, tuple(metadata))

    @property
    def pyramid(self) -> Pyramid:
        return self._pyramid

    @property
    def arrays(self) -> tuple[Any, ...]:
        """Lazy array levels, ordered finest to coarsest.

        Viewer adapters that can consume the storage arrays directly should
        use this property instead of routing reads through ``read``.  In
        particular, napari's progressive renderer needs the array metadata
        and indexing protocol in order to manage its bounded resident
        intervals and partial texture uploads.
        """

        return self._arrays

    async def read(self, level: int, region: Region) -> np.ndarray:
        array = self._arrays[level]

        def _read() -> np.ndarray:
            result = array[region.slices()]
            if hasattr(result, "read"):
                result = result.read().result()
            if hasattr(result, "compute"):
                result = result.compute()
            return np.asarray(result)

        return await asyncio.to_thread(_read)
