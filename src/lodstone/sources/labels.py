"""Sparse single-writer label storage for Lodstone."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np

from ..model import Pyramid, Region
from .array import ArrayPyramidSource


class ZarrLabels(ArrayPyramidSource):
    """A local integer label layer with zero as the background ID."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._storage_lock = RLock()

    @classmethod
    def create(
        cls, path: str | Path, reference: Pyramid, *, dtype: str = "uint32"
    ) -> ZarrLabels:
        import zarr

        if np.dtype(dtype).kind != "u":
            raise ValueError("labels require an unsigned integer dtype")
        level = reference.levels[0]
        group = zarr.open_group(str(path), mode="w-")
        group.create_array(
            "0",
            shape=level.shape,
            chunks=level.chunks,
            dtype=dtype,
            fill_value=0,
            config={"write_empty_chunks": False},
        )
        group.attrs.update(
            {
                "axes": list(reference.axes),
                "transforms": [level.voxel_to_world.tolist()],
            }
        )
        return cls.open(path)

    @classmethod
    def open(cls, path: str | Path) -> ZarrLabels:
        import zarr

        group = zarr.open_group(str(path), mode="r+")
        return cls(
            [group["0"]],
            axes=cast(list[str], group.attrs["axes"]),
            transforms=[
                np.asarray(matrix) for matrix in cast(list, group.attrs["transforms"])
            ],
        )

    def chunk_region(self, key: tuple[int, ...]) -> Region:
        """Return the clipped bounds of a finest-level storage chunk."""
        level = self.pyramid.levels[0]
        if len(key) != level.ndim:
            raise ValueError("expected one chunk index per axis")
        start = tuple(i * size for i, size in zip(key, level.chunks, strict=True))
        if any(a < 0 or a >= n for a, n in zip(start, level.shape, strict=True)):
            raise IndexError("chunk outside the label array")
        stop = tuple(
            min(a + size, n)
            for a, size, n in zip(start, level.chunks, level.shape, strict=True)
        )
        return Region(start, stop)

    def write_chunk(self, key: tuple[int, ...], data: np.ndarray) -> None:
        """Replace one storage chunk, including a clipped boundary chunk."""
        region = self.chunk_region(key)
        if data.shape != region.shape or data.dtype != self.pyramid.levels[0].dtype:
            raise ValueError("data must match the shape and dtype of the label chunk")
        with self._storage_lock:
            self.arrays[0][region.slices()] = data

    def read_region(self, level: int, region: Region) -> np.ndarray:
        with self._storage_lock:
            return np.asarray(self.arrays[level][region.slices()])

    async def read(self, level: int, region: Region) -> np.ndarray:
        return await asyncio.to_thread(self.read_region, level, region)
