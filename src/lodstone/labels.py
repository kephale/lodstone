"""Chunk-local editing of integer label arrays."""

from __future__ import annotations

import asyncio
from itertools import pairwise, product
from threading import Lock

import numpy as np

from .model import Pyramid, Region
from .sources.labels import ZarrLabels

ChunkKey = tuple[int, ...]
Changes = dict[ChunkKey, tuple[np.ndarray, np.ndarray]]


class LabelEditor:
    """Overlay unsaved world-space slice strokes on a writable label layer."""

    def __init__(self, source: ZarrLabels) -> None:
        self.source = source
        self._dirty: dict[ChunkKey, np.ndarray] = {}
        self._undo: list[Changes] = []
        self._redo: list[Changes] = []
        self._lock = Lock()
        self._save_lock = Lock()

    @property
    def pyramid(self) -> Pyramid:
        return self.source.pyramid

    @property
    def dirty_chunks(self) -> int:
        with self._lock:
            return len(self._dirty)

    def _keys(self, region: Region):
        return product(
            *(
                range(start // step, (stop - 1) // step + 1)
                for start, stop, step in zip(
                    region.start,
                    region.stop,
                    self.pyramid.levels[0].chunks,
                    strict=True,
                )
            )
        )

    def _read(self, level: int, region: Region) -> np.ndarray:
        with self._lock:
            result = self.source.read_region(level, region)
            for key in self._keys(region) if level == 0 else ():
                if key not in self._dirty:
                    continue
                chunk = self.source.chunk_region(key)
                overlap = chunk.intersection(region)
                if overlap is None:
                    continue
                dest = _relative_slices(overlap, region)
                result[dest] = self._dirty[key][_relative_slices(overlap, chunk)]
            return result

    async def read(self, level: int, region: Region) -> np.ndarray:
        return await asyncio.to_thread(self._read, level, region)

    def stroke(
        self,
        points: np.ndarray,
        radius: float,
        label: int,
        *,
        fixed_indices: tuple[int, ...] = (),
    ) -> int:
        """Paint a physical-radius polyline on a finest-level slice.

        Points are in world units for the last two data axes. Fixed indices
        pick one voxel along each preceding axis. Zero erases. The return
        value counts chunks with changed labels.
        """
        level = self.pyramid.levels[0]
        transform = level.voxel_to_world
        spacing = np.diag(transform)[:-1]
        offset = transform[:-1, -1]
        if not np.array_equal(transform[:-1, :-1], np.diag(spacing)) or not np.all(
            spacing > 0
        ):
            raise ValueError("slice painting requires positive axis-aligned spacing")
        if len(fixed_indices) != level.ndim - 2 or any(
            not 0 <= index < size for index, size in zip(fixed_indices, level.shape)
        ):
            raise ValueError("choose one valid index per axis before the paint plane")
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or not np.isfinite(points).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("expected finite world points and a positive radius")
        if (
            not isinstance(label, (int, np.integer))
            or not 0 <= label <= np.iinfo(level.dtype).max
        ):
            raise ValueError("label ID is outside the storage dtype")
        changes: Changes = {}
        with self._lock:
            segments = pairwise(points) if len(points) > 1 else [(points[0], points[0])]
            for start, end in segments:
                lower = (np.minimum(start, end) - radius - offset[-2:]) / spacing[-2:]
                upper = (np.maximum(start, end) + radius - offset[-2:]) / spacing[-2:]
                lo = tuple(
                    int(v) for v in np.clip(np.floor(lower), 0, level.shape[-2:])
                )
                hi = tuple(int(v) for v in np.clip(np.ceil(upper), 0, level.shape[-2:]))
                if any(a >= b for a, b in zip(lo, hi, strict=True)):
                    continue
                bounds = Region(
                    fixed_indices + lo, tuple(i + 1 for i in fixed_indices) + hi
                )
                for key in self._keys(bounds):
                    chunk = self.source.chunk_region(key)
                    overlap = chunk.intersection(bounds)
                    assert overlap is not None
                    y, x = np.ogrid[
                        overlap.start[-2] : overlap.stop[-2],
                        overlap.start[-1] : overlap.stop[-1],
                    ]
                    dy = (y + 0.5) * spacing[-2] + offset[-2] - start[0]
                    dx = (x + 0.5) * spacing[-1] + offset[-1] - start[1]
                    vector = end - start
                    length_squared = vector @ vector
                    if length_squared:
                        t = np.clip(
                            (dy * vector[0] + dx * vector[1]) / length_squared, 0, 1
                        )
                        dy = dy - t * vector[0]
                        dx = dx - t * vector[1]
                    mask = dy**2 + dx**2 <= radius**2
                    if not np.any(mask):
                        continue
                    before = self._dirty.get(key)
                    if before is None:
                        before = self.source.read_region(0, chunk)
                    relative = _relative_slices(overlap, chunk)
                    plane = before[relative].reshape(mask.shape)
                    if np.all(plane[mask] == label):
                        continue
                    after = before.copy()
                    selection = after[relative]
                    selection[...] = np.where(mask, label, plane).reshape(
                        selection.shape
                    )
                    changes[key] = (changes.get(key, (before, before))[0], after)
                    self._dirty[key] = after
            if changes:
                self._undo.append(changes)
                self._redo.clear()
        return len(changes)

    def undo(self) -> None:
        with self._lock:
            if self._undo:
                changes = self._undo.pop()
                self._dirty.update(
                    {key: before for key, (before, _) in changes.items()}
                )
                self._redo.append(changes)

    def redo(self) -> None:
        with self._lock:
            if self._redo:
                changes = self._redo.pop()
                self._dirty.update({key: after for key, (_, after) in changes.items()})
                self._undo.append(changes)

    def save(self) -> None:
        """Persist pending edits, retaining any chunks changed during a write."""
        with self._save_lock:
            with self._lock:
                pending = dict(self._dirty)
            for key, data in pending.items():
                self.source.write_chunk(key, data)
                with self._lock:
                    if self._dirty.get(key) is data:
                        del self._dirty[key]


def _relative_slices(inner: Region, outer: Region) -> tuple[slice, ...]:
    return tuple(
        slice(start - base, stop - base)
        for start, stop, base in zip(inner.start, inner.stop, outer.start, strict=True)
    )
