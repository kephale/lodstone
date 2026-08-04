"""Source adapter for NumPy-like pyramid levels."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

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
        chunks: Sequence[tuple[int, ...]] | None = None,
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
                if native and native and isinstance(native[0], tuple):
                    native = tuple(int(axis_chunks[0]) for axis_chunks in native)
            if native is None:
                native = shape
            metadata.append(
                Level(shape, np.dtype(array.dtype), tuple(native), np.asarray(matrix))
            )
        self._pyramid = Pyramid(axis_names, tuple(metadata))

    @property
    def pyramid(self) -> Pyramid:
        return self._pyramid

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
