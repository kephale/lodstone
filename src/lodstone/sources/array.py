"""Source adapter for NumPy-like pyramid levels."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..model import Level, Pyramid, Region, identity_transform


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
                native = getattr(array, "chunks", None)
            if native is None:
                native = shape
            chunk_grid = None
            if native and isinstance(native[0], tuple):
                chunk_grid = tuple(
                    tuple(int(value) for value in axis_chunks)
                    for axis_chunks in cast(Sequence[Sequence[int]], native)
                )
                chunk_shape = tuple(axis_chunks[0] for axis_chunks in chunk_grid)
            else:
                chunk_shape = tuple(int(value) for value in cast(Sequence[int], native))
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
