"""Optional ownership-based Zarr Vectors raw triangle mesh source."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import numpy as np

from ..mesh import MeshChunk, MeshData, MeshLevel
from ..mesh_diagnostics import StoreReadCounter, counted_store


class ZarrVectorsMeshSource:
    """One or more stored surface levels, ordered finest to coarsest.

    Errors must be supplied in world units; storage bin sizes and object
    sparsity are not geometric error estimates. Coordinates are used as stored,
    with an optional explicit affine transform into the viewer's XYZ frame.
    Requires the companion MeshReader branch until it is accepted upstream.
    """

    def __init__(
        self,
        source: Any,
        *,
        level_errors: Sequence[float],
        stored_levels: Sequence[int] | None = None,
        transform: np.ndarray | None = None,
        max_read_bytes: int = 128 << 20,
        track_io: bool = False,
    ) -> None:
        try:
            MeshReader = import_module("zarr_vectors.building").MeshReader
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "ZarrVectorsMeshSource requires zarr-vectors with building.MeshReader; "
                "see docs/mesh-streaming.md for the companion branch"
            ) from error
        if max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be positive")
        self.max_read_bytes = max_read_bytes
        self.io = StoreReadCounter()
        if track_io:
            if isinstance(source, (str, Path)):
                if str(source).startswith(("http://", "https://", "s3://", "gs://")):
                    raise ValueError(
                        "For remote I/O tracking, supply an explicit Zarr Store"
                    )
                source = import_module("zarr.storage").LocalStore(
                    source, read_only=True
                )
            source = counted_store(source, self.io)
        indices = (
            tuple(range(len(level_errors)))
            if stored_levels is None
            else tuple(stored_levels)
        )
        if (
            not indices
            or len(indices) != len(level_errors)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("provide distinct stored levels and one error per level")
        if any(b < a for a, b in pairwise(level_errors)):
            raise ValueError("level errors must be ordered finest to coarsest")
        self._transform = np.array(
            np.eye(4) if transform is None else transform, dtype=float, copy=True
        )
        if self._transform.shape != (4, 4) or not np.all(np.isfinite(self._transform)):
            raise ValueError("transform must be a finite 4 by 4 matrix")
        if not np.array_equal(self._transform[3], [0, 0, 0, 1]):
            raise ValueError("mesh transform must be affine")
        self._lock = threading.Lock()
        self._readers = tuple(MeshReader(source, level=index) for index in indices)
        if any(reader.width != 3 for reader in self._readers):
            raise NotImplementedError("only triangle meshes are currently supported")
        levels = []
        for reader, error in zip(self._readers, level_errors, strict=True):
            chunks = []
            for chunk in reader.chunks:
                corners = np.asarray(list(product(*zip(*chunk.bounds))), dtype=float)
                moved = corners @ self._transform[:3, :3].T + self._transform[:3, 3]
                chunks.append(
                    MeshChunk(
                        chunk.key, (tuple(moved.min(axis=0)), tuple(moved.max(axis=0)))
                    )
                )
            levels.append(MeshLevel(tuple(chunks), float(error)))
        self.levels = tuple(levels)

    async def read(self, level: int, key: tuple[int, ...]) -> MeshData:
        return await asyncio.to_thread(partial(self._read, level, key))

    def _read(self, level: int, key: tuple[int, ...]) -> MeshData:
        with self._lock:
            result = self._readers[level].read([key], max_bytes=self.max_read_bytes)
        vertices = result["vertices"]
        moved = vertices @ self._transform[:3, :3].T + self._transform[:3, 3]
        return MeshData(moved, result["faces"])
