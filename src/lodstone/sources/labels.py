"""Sparse single-writer label storage for Lodstone."""

from __future__ import annotations

import asyncio
import json
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import cast

import numpy as np

from ..model import Pyramid, Region
from .array import ArrayPyramidSource


class ZarrLabels(ArrayPyramidSource):
    """A local integer label layer with zero as the background ID."""

    def __init__(self, path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._storage_lock = RLock()
        self._journal = path / ".lodstone-pending-chunk"
        self._factors = _factors(self.pyramid)
        self._repair()

    @classmethod
    def create(
        cls, path: str | Path, reference: Pyramid, *, dtype: str = "uint32"
    ) -> ZarrLabels:
        import zarr

        if np.dtype(dtype).kind != "u":
            raise ValueError("labels require an unsigned integer dtype")
        _factors(reference)
        group = zarr.open_group(str(path), mode="w-")
        for index, level in enumerate(reference.levels):
            group.create_array(
                str(index),
                shape=level.shape,
                chunks=level.chunks,
                dtype=dtype,
                fill_value=0,
                config={"write_empty_chunks": False},
            )
        group.attrs.update(
            {
                "axes": list(reference.axes),
                "transforms": [
                    level.voxel_to_world.tolist() for level in reference.levels
                ],
            }
        )
        return cls.open(path)

    @classmethod
    def open(cls, path: str | Path) -> ZarrLabels:
        import zarr

        group = zarr.open_group(str(path), mode="r+")
        transforms = [
            np.asarray(matrix) for matrix in cast(list, group.attrs["transforms"])
        ]
        return cls(
            Path(path),
            [group[str(index)] for index in range(len(transforms))],
            axes=cast(list[str], group.attrs["axes"]),
            transforms=transforms,
        )

    def chunk_region(self, key: tuple[int, ...], index: int = 0) -> Region:
        """Return the clipped bounds of a storage chunk."""
        level = self.pyramid.levels[index]
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
            self._repair()
            if len(self._factors) > 1:
                with NamedTemporaryFile(
                    mode="w", dir=self._journal.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    try:
                        json.dump(key, handle)
                        handle.close()
                        temporary.replace(self._journal)
                    finally:
                        temporary.unlink(missing_ok=True)
            self.arrays[0][region.slices()] = data
            if len(self._factors) > 1:
                self._repair()

    def _repair(self) -> None:
        if not self._journal.exists():
            return
        key = tuple(json.loads(self._journal.read_text()))
        fine = self.chunk_region(key)
        for index, level in enumerate(self.pyramid.levels[1:], 1):
            affected = self.coarse_overlap(index, fine)
            if affected is None:
                continue
            coarse_keys = _chunk_keys(affected, level.chunks)
            for coarse_key in coarse_keys:
                chunk = self.chunk_region(coarse_key, index)
                self.arrays[index][chunk.slices()] = self.arrays[0].oindex[
                    self.finest_indices(index, chunk)
                ]
        self._journal.unlink()

    def finest_indices(self, index: int, region: Region) -> tuple[np.ndarray, ...]:
        """Map coarse voxel centers to finest-level voxel indices."""
        return tuple(
            np.minimum(np.arange(a, b) * step + step // 2, size - 1)
            for a, b, step, size in zip(
                region.start,
                region.stop,
                self._factors[index],
                self.pyramid.levels[0].shape,
                strict=True,
            )
        )

    def coarse_overlap(self, index: int, fine: Region) -> Region | None:
        """Find the coarse voxels that sample the supplied finest region."""
        start, stop = [], []
        for a, b, step, n, m in zip(
            fine.start,
            fine.stop,
            self._factors[index],
            self.pyramid.levels[0].shape,
            self.pyramid.levels[index].shape,
            strict=True,
        ):
            start.append(min(m, max(0, -((step // 2 - a) // step))))
            stop.append(m if b == n else min(m, max(0, -((step // 2 - b) // step))))
        if any(a >= b for a, b in zip(start, stop, strict=True)):
            return None
        return Region(tuple(start), tuple(stop))

    def read_region(self, level: int, region: Region) -> np.ndarray:
        with self._storage_lock:
            self._repair()
            return np.asarray(self.arrays[level][region.slices()])

    async def read(self, level: int, region: Region) -> np.ndarray:
        return await asyncio.to_thread(self.read_region, level, region)


def _chunk_keys(region: Region, chunks: tuple[int, ...]):
    return product(
        *(
            range(a // step, (b - 1) // step + 1)
            for a, b, step in zip(region.start, region.stop, chunks, strict=True)
        )
    )


def _factors(pyramid: Pyramid) -> tuple[tuple[int, ...], ...]:
    fine = pyramid.levels[0]
    factors = [(1,) * fine.ndim]
    for level in pyramid.levels[1:]:
        transform = np.linalg.solve(fine.voxel_to_world, level.voxel_to_world)
        rounded = np.rint(np.diag(transform)[:-1])
        if not np.all(rounded >= 1) or not np.allclose(
            transform, np.diag([*rounded, 1]), atol=1e-9, rtol=0
        ):
            raise ValueError("label levels need aligned positive integer scale factors")
        factor = tuple(int(step) for step in rounded)
        if level.shape != tuple(
            (n + step - 1) // step for n, step in zip(fine.shape, factor, strict=True)
        ):
            raise ValueError("label level shapes must cover the finest grid")
        factors.append(factor)
    return tuple(factors)
