"""A generated multiscale Mandelbulb surface, for stress testing mesh streams.

The surface is the level set ``DE = offset`` of the usual Mandelbulb
distance estimate: the fractal thickened by about a finest voxel, which
gives every level the same well-defined surface to approximate. Level ``k``
(0 finest) splits the cube ``[-radius, radius]^3`` into ``2**(levels-1-k)``
chunks per side; each chunk is marching-cubed from a ``(tile + 1)^3`` grid
of samples on its level's lattice, so neighbouring chunks share their
boundary samples exactly and meet without cracks.

Only chunks the surface may reach are listed. Candidates are the children
of the chunks kept one level up; each is probed on a small grid, and kept
when the probe crosses the surface or the distance estimate says the
surface may pass between probe points. About 30% of listed chunks turn out
empty, which costs a read but no geometry.

Building the chunk lists costs a few seconds at 8 levels (about 175,000
finest chunks, 366 million triangles in total at level 0); chunks are
meshed when read. ``numba`` speeds up sampling when installed; reading
needs ``scikit-image``.
"""

from __future__ import annotations

import asyncio
import math
from importlib import import_module

import numpy as np

from ..mesh import MeshChunk, MeshData, MeshLevel

try:
    from numba import njit, prange  # pyright: ignore[reportMissingImports]

    HAS_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised without numba
    HAS_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in; without numba, sampling uses NumPy instead."""

        if args and callable(args[0]):
            return args[0]
        return lambda func: func


_CHILDREN = np.array(list(np.ndindex(2, 2, 2)), dtype=np.int64)


def _de_numpy(points: np.ndarray, maxiter: int, power: int) -> np.ndarray:
    """Distance estimates for ``(n, 3)`` points; 0 inside the set."""

    c = np.asarray(points, dtype=np.float64)
    z = c.copy()
    dr = np.ones(len(c))
    out = np.zeros(len(c))
    active = np.arange(len(c))
    for _ in range(maxiter):
        r = np.sqrt(np.einsum("ij,ij->i", z, z))
        escaped = r > 2.0
        if escaped.any():
            done = active[escaped]
            out[done] = 0.5 * np.log(r[escaped]) * r[escaped] / dr[escaped]
            keep = ~escaped
            active, z, dr, r = active[keep], z[keep], dr[keep], r[keep]
        if not len(active):
            break
        safe = np.maximum(r, 1e-300)
        theta = np.arccos(np.clip(z[:, 2] / safe, -1.0, 1.0)) * power
        phi = np.arctan2(z[:, 1], z[:, 0]) * power
        rp = r ** (power - 1)
        dr = rp * power * dr + 1.0
        zr = rp * r
        st = np.sin(theta)
        z = (
            np.stack(
                [zr * st * np.cos(phi), zr * st * np.sin(phi), zr * np.cos(theta)],
                axis=1,
            )
            + c[active]
        )
    return out


if HAS_NUMBA:

    @njit(nogil=True, fastmath=True, cache=True)
    def _de_one(px, py, pz, maxiter, power):  # pragma: no cover - jitted
        x, y, z = px, py, pz
        dr = 1.0
        for _ in range(maxiter):
            r = math.sqrt(x * x + y * y + z * z)
            if r > 2.0:
                return 0.5 * math.log(r) * r / dr
            ratio = z / max(r, 1e-300)
            ratio = min(max(ratio, -1.0), 1.0)
            theta = math.acos(ratio) * power
            phi = math.atan2(y, x) * power
            rp = r ** (power - 1)
            dr = rp * power * dr + 1.0
            zr = rp * r
            st = math.sin(theta)
            x = zr * st * math.cos(phi) + px
            y = zr * st * math.sin(phi) + py
            z = zr * math.cos(theta) + pz
        return 0.0

    @njit(parallel=True, nogil=True, fastmath=True, cache=True)
    def _de_points_numba(points, maxiter, power):  # pragma: no cover
        out = np.empty(points.shape[0])
        for i in prange(points.shape[0]):
            out[i] = _de_one(points[i, 0], points[i, 1], points[i, 2], maxiter, power)
        return out

    @njit(parallel=True, nogil=True, fastmath=True, cache=True)
    def _de_lattice_numba(
        origin, n, step, low, maxiter, power
    ):  # pragma: no cover - jitted
        out = np.empty((n, n, n))
        for i in prange(n):
            x = low + (origin[0] + i) * step
            for j in range(n):
                y = low + (origin[1] + j) * step
                for k in range(n):
                    z = low + (origin[2] + k) * step
                    out[i, j, k] = _de_one(x, y, z, maxiter, power)
        return out


def distance_estimate(
    points: np.ndarray, maxiter: int = 12, power: int = 8
) -> np.ndarray:
    """Mandelbulb distance estimates for ``(n, 3)`` XYZ points.

    Zero for points that do not escape within ``maxiter`` iterations.
    """

    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    if HAS_NUMBA:
        return _de_points_numba(points, int(maxiter), int(power))
    return _de_numpy(points, int(maxiter), int(power))


class MandelbulbMeshSource:
    """A ``lodstone.mesh.MeshSource`` of the Mandelbulb surface.

    Parameters
    ----------
    levels : int
        Number of levels; the finest has ``tile * 2**(levels-1)`` cells per
        side.
    tile : int
        Marching-cubes cells per chunk side.
    maxiter : int
        Iterations of the distance estimate. More iterations add detail
        that only the finer levels resolve.
    power : int
        Mandelbulb power.
    radius : float
        Half the side of the cube holding the surface, centered on 0.
    offset : float, optional
        Distance estimate of the surface; defaults to two finest cells.
    probe : int
        Probe points per side when deciding which chunks to list.

    Notes
    -----
    Chunk keys are integer XYZ chunk indices; level ``k``'s ``error`` is its
    cell size.
    """

    def __init__(
        self,
        levels: int = 8,
        *,
        tile: int = 32,
        maxiter: int = 12,
        power: int = 8,
        radius: float = 1.25,
        offset: float | None = None,
        probe: int = 5,
    ) -> None:
        if levels < 1 or tile < 2 or probe < 2:
            raise ValueError("levels, tile and probe are too small")
        self.tile = int(tile)
        self.maxiter = int(maxiter)
        self.power = int(power)
        self.radius = float(radius)
        finest = 2 * self.radius / (self.tile * 2 ** (levels - 1))
        self.offset = 2.0 * finest if offset is None else float(offset)
        keys = self._list_chunks(int(levels), int(probe))
        built = []
        for level in range(levels):
            span = self._span(level, levels)
            built.append(
                MeshLevel(
                    tuple(
                        MeshChunk(
                            tuple(int(v) for v in key),
                            (
                                tuple(float(-self.radius + v * span) for v in key),
                                tuple(
                                    float(-self.radius + (v + 1) * span) for v in key
                                ),
                            ),
                        )
                        for key in keys[level]
                    ),
                    error=span / self.tile,
                )
            )
        self.levels: tuple[MeshLevel, ...] = tuple(built)

    def _span(self, level: int, levels: int | None = None) -> float:
        levels = len(self.levels) if levels is None else levels
        return 2 * self.radius / 2 ** (levels - 1 - level)

    def _list_chunks(self, levels: int, probe: int) -> dict[int, np.ndarray]:
        ticks = np.linspace(0.0, 1.0, probe)
        offsets = np.stack(np.meshgrid(ticks, ticks, ticks, indexing="ij"), -1).reshape(
            -1, 3
        )
        keys = np.zeros((1, 3), dtype=np.int64)
        listed = {levels - 1: keys}
        for level in range(levels - 2, -1, -1):
            candidates = (keys[:, None, :] * 2 + _CHILDREN[None]).reshape(-1, 3)
            span = self._span(level, levels)
            points = -self.radius + (candidates[:, None, :] + offsets) * span
            field = (
                distance_estimate(points.reshape(-1, 3), self.maxiter, self.power)
                - self.offset
            ).reshape(len(candidates), -1)
            # the surface may pass between probe points up to half a probe
            # diagonal from the nearest one; the estimate is approximate,
            # so allow half again
            reach = 1.5 * span / (probe - 1) * math.sqrt(3) / 2
            keep = (field.min(axis=1) <= reach) & (field.max(axis=1) >= 0)
            keys = candidates[keep]
            listed[level] = keys
        return listed

    def _samples(self, level: int, key: tuple[int, ...]) -> np.ndarray:
        n = self.tile + 1
        step = self._span(level) / self.tile
        origin = np.asarray(key, dtype=np.int64) * self.tile
        if HAS_NUMBA:
            return _de_lattice_numba(
                origin, n, step, -self.radius, self.maxiter, self.power
            )
        index = np.arange(n)
        grid = np.stack(np.meshgrid(index, index, index, indexing="ij"), -1).reshape(
            -1, 3
        )
        points = -self.radius + (origin + grid) * step
        return _de_numpy(points, self.maxiter, self.power).reshape(n, n, n)

    def chunk_mesh(self, level: int, key: tuple[int, ...]) -> MeshData:
        """Mesh one chunk (synchronously)."""

        field = self._samples(level, key) - self.offset
        if field.min() > 0 or field.max() < 0:
            return MeshData(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        measure = import_module("skimage.measure")
        # lattice-index vertices, so a shared face interpolates identically
        # from both sides
        vertices, faces, _, _ = measure.marching_cubes(field, 0.0)
        step = self._span(level) / self.tile
        origin = np.asarray(key, dtype=np.float64) * self.tile
        vertices = -self.radius + (vertices + origin) * step
        return MeshData(vertices.astype(np.float32), faces.astype(np.int32))

    async def read(self, level: int, key: tuple[int, ...]) -> MeshData:
        return await asyncio.to_thread(self.chunk_mesh, level, key)
