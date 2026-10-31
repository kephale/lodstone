"""Experimental geometry streaming, separate from dense array contracts."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np

from .model import View
from .runtime import Runtime

MeshKey = tuple[int, tuple[int, ...]]


@dataclass(frozen=True)
class MeshChunk:
    """One face ownership cell, with conservative world-space bounds."""

    key: tuple[int, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class MeshLevel:
    """One surface resolution; error is an explicit world-space estimate."""

    chunks: tuple[MeshChunk, ...]
    error: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.error) or self.error < 0:
            raise ValueError("mesh level error must be finite and nonnegative")
        if len({chunk.key for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("mesh chunk keys must be unique within each level")
        for chunk in self.chunks:
            lo, hi = np.asarray(chunk.bounds, dtype=float)
            if lo.shape != (3,) or hi.shape != (3,):
                raise ValueError("mesh bounds must be three dimensional")
            if not np.all(np.isfinite([lo, hi])) or np.any(hi < lo):
                raise ValueError("mesh bounds must be finite and ordered")


@dataclass(frozen=True)
class MeshData:
    """A self-contained triangle fragment; indices address its own vertices."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must contain integer indices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face index is outside the vertex array")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def nbytes(self) -> int:
        return self.vertices.nbytes + self.faces.nbytes


class MeshSource(Protocol):
    levels: tuple[MeshLevel, ...]

    async def read(self, level: int, key: tuple[int, ...]) -> MeshData: ...


class MeshTarget(Protocol):
    def replace(self, level: int, chunks: Sequence[tuple[MeshKey, MeshData]]) -> None:
        """Replace the complete displayed phase, releasing previous geometry."""
        ...


@dataclass(frozen=True)
class MeshStreamSnapshot:
    generation: int
    state: str
    requested_level: int | None
    displayed_level: int | None
    reads: int
    cache_hits: int
    stale_reads: int
    evictions: int
    cache_bytes: int
    staged_bytes: int
    phase_chunks_done: int
    phase_chunks_total: int
    displayed_chunks: int
    limited: bool
    error: str | None


def visible_chunks(level: MeshLevel, view: View) -> tuple[MeshChunk, ...]:
    """Conservative homogeneous frustum culling, nearest screen center first."""
    if view.displayed_axes != (0, 1, 2):
        raise ValueError("mesh views must use displayed_axes=(0, 1, 2) in XYZ order")
    selected = []
    for chunk in level.chunks:
        points = np.asarray(list(product(*zip(*chunk.bounds))), dtype=float)
        clip = np.column_stack([points, np.ones(len(points))]) @ view.world_to_clip.T
        # Reject only when every corner is outside the same clip plane. This
        # remains conservative for boxes crossing the eye or near plane.
        w = clip[:, 3:4]
        if np.any(np.all(clip[:, :3] < -w, axis=0)) or np.any(
            np.all(clip[:, :3] > w, axis=0)
        ):
            continue
        center = np.mean(points, axis=0)
        projected = view.world_to_clip @ np.append(center, 1)
        priority = (
            float(np.sum((projected[:2] / projected[3]) ** 2))
            if projected[3] > 0
            else 0.0
        )
        selected.append((priority, chunk.key, chunk))
    return tuple(item[2] for item in sorted(selected))


def select_mesh_level(levels: tuple[MeshLevel, ...], view: View, pixels: float) -> int:
    """Choose the coarsest level whose supplied error fits a pixel threshold.

    Evaluate projected error at visible bound corners, not only the center:
    nearby perspective geometry must not inherit a distant object's LOD.
    """
    if not levels or not np.isfinite(pixels) or pixels <= 0:
        raise ValueError("levels and a finite positive pixel threshold are required")
    for index in reversed(range(len(levels))):
        level = levels[index]
        worst = 0.0
        for chunk in visible_chunks(level, view):
            for corner in product(*zip(*chunk.bounds)):
                p = np.append(corner, 1.0)
                a = view.world_to_clip @ p
                if a[3] <= 0:
                    worst = float("inf")
                    break
                # Project a cube enclosing the error ball; checking only axes
                # can underestimate diagonal screen-space displacement.
                for displacement in product((-level.error, level.error), repeat=3):
                    q = p.copy()
                    q[:3] += displacement
                    b = view.world_to_clip @ q
                    if b[3] <= 0:
                        worst = float("inf")
                        break
                    delta = (
                        (b[:2] / b[3] - a[:2] / a[3]) * np.asarray(view.viewport) / 2
                    )
                    worst = max(worst, float(np.linalg.norm(delta)))
        if worst <= pixels:
            return index
    return 0


class MeshStream:
    """Serial chunk reads, bounded retained geometry, atomic LOD publication.

    Each phase uses one resolution throughout; mixed-LOD seams are excluded.
    ``memory_limit`` bounds cached/staged decoded arrays, and each publication
    separately. Decoder scratch and one incoming chunk are additional; GPU
    allocations belong to the target. An oversized refinement preserves the
    completed coarser phase and sets ``limited`` rather than silently cropping.
    """

    def __init__(
        self,
        source: MeshSource,
        target: MeshTarget,
        *,
        dispatch: Callable[[Callable[[], None]], None] = lambda callback: callback(),
        runtime: Runtime | None = None,
        memory_limit: int = 128 << 20,
        pixel_error: float = 2.0,
    ) -> None:
        if memory_limit <= 0 or not np.isfinite(pixel_error) or pixel_error <= 0:
            raise ValueError("memory limit and pixel error must be positive")
        if not source.levels:
            raise ValueError("a mesh source needs at least one level")
        self.source, self.target, self.dispatch = source, target, dispatch
        self.memory_limit, self.pixel_error = memory_limit, pixel_error
        self.runtime = runtime or Runtime()
        self._owns_runtime = runtime is None
        self._lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._cache: OrderedDict[MeshKey, MeshData] = OrderedDict()
        self._future: Future | None = None
        self._pending: set[Future] = set()
        self.limited = False
        self.displayed_level: int | None = None
        self.error: BaseException | None = None
        self.reads = 0
        self.cache_bytes = 0
        self.state = "idle"
        self.requested_level: int | None = None
        self.cache_hits = self.stale_reads = self.evictions = 0
        self.staged_bytes = self.phase_chunks_done = self.phase_chunks_total = 0
        self.displayed_keys: frozenset[MeshKey] = frozenset()

    def update(self, view: View, *, level: int | None = None) -> None:
        selected = (
            select_mesh_level(self.source.levels, view, self.pixel_error)
            if level is None
            else level
        )
        if not 0 <= selected < len(self.source.levels):
            raise ValueError("mesh level is outside the source")
        visible_chunks(self.source.levels[selected], view)  # validate before scheduling
        with self._lock:
            if self._closed:
                raise RuntimeError("mesh stream is closed")
            self._generation += 1
            generation = self._generation
            self.state = "loading"
            self.requested_level = selected
            # Do not cancel an executor read: cancelling the await does not
            # stop its worker and could overlap reads on a non-threadsafe source.
            previous = self._future
            self._future = asyncio.run_coroutine_threadsafe(
                self._run_after(previous, generation, view, selected),
                self.runtime.loop,
            )
            self._pending.add(self._future)
            self._future.add_done_callback(self._forget_future)

    def _forget_future(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)

    async def _run_after(
        self, previous: Future | None, generation: int, view: View, selected: int
    ) -> None:
        if previous is not None:
            # A failed prior view must not prevent retrying a new one. Its
            # exception remains observable through wait()/error for that view.
            await asyncio.gather(asyncio.wrap_future(previous), return_exceptions=True)
        if not self._current(generation):
            return
        self.limited, self.error = False, None
        completed_phase = False
        try:
            for level in reversed(range(selected, len(self.source.levels))):
                staged: list[tuple[MeshKey, MeshData]] = []
                pinned: set[MeshKey] = set()
                phase_bytes = 0
                chunks = visible_chunks(self.source.levels[level], view)
                with self._lock:
                    if not self._current(generation):
                        return
                    self.phase_chunks_done, self.phase_chunks_total = 0, len(chunks)
                    self.staged_bytes = 0
                for chunk in chunks:
                    if not self._current(generation):
                        return
                    key = (level, chunk.key)
                    with self._lock:
                        data = self._cache.get(key)
                    if data is None:
                        try:
                            data = await self.source.read(level, chunk.key)
                        except MemoryError:
                            self.limited = True
                            if completed_phase:
                                return
                            raise
                        self.reads += 1
                        if not self._current(generation):
                            self.stale_reads += 1
                            return
                    else:
                        self.cache_hits += 1
                    phase_bytes += data.nbytes
                    if phase_bytes > self.memory_limit:
                        self.limited = True
                        if not completed_phase:
                            raise MemoryError(
                                "visible coarse mesh exceeds memory_limit"
                            )
                        return
                    if not self._remember(generation, key, data, pinned):
                        return
                    pinned.add(key)
                    staged.append((key, data))
                    with self._lock:
                        if not self._current(generation):
                            return
                        self.staged_bytes = phase_bytes
                        self.phase_chunks_done += 1
                if not self._current(generation):
                    return
                try:
                    await self._publish(generation, level, staged)
                except MemoryError:
                    self.limited = True
                    if completed_phase:
                        return
                    raise
                completed_phase = True
        except Exception as error:
            if self._current(generation):
                self.error = error
            raise
        finally:
            with self._lock:
                if self._current(generation):
                    self.staged_bytes = 0
                    self.state = (
                        "failed"
                        if self.error
                        else "limited"
                        if self.limited
                        else "complete"
                    )

    def _remember(
        self, generation: int, key: MeshKey, data: MeshData, pinned: set[MeshKey]
    ) -> bool:
        with self._lock:
            if not self._current(generation):
                return False
            # The lock also prevents close() from clearing the cache mid-eviction.
            for old in list(self._cache):
                added = 0 if key in self._cache else data.nbytes
                if self.cache_bytes + added <= self.memory_limit:
                    break
                if old not in pinned and old != key:
                    self.cache_bytes -= self._cache.pop(old).nbytes
                    self.evictions += 1
            if key not in self._cache:
                self._cache[key] = data
                self.cache_bytes += data.nbytes
            self._cache.move_to_end(key)
            return True

    async def _publish(
        self, generation: int, level: int, staged: list[tuple[MeshKey, MeshData]]
    ) -> None:
        completed: Future = Future()

        def apply() -> None:
            try:
                with self._lock:
                    if self._current(generation):
                        self.target.replace(level, staged)
                        self.displayed_level = level
                        self.displayed_keys = frozenset(key for key, _data in staged)
                completed.set_result(None)
            except Exception as error:  # noqa: BLE001 - relay renderer errors to wait()
                if not completed.done():
                    completed.set_exception(error)

        self.dispatch(apply)
        await asyncio.wrap_future(completed)

    def _current(self, generation: int) -> bool:
        with self._lock:
            return not self._closed and generation == self._generation

    def wait(self, timeout: float = 30.0) -> None:
        """Wait for reads and host publication; never call on the GUI thread."""
        future = self._future
        if future is not None:
            future.result(timeout)

    def snapshot(self) -> MeshStreamSnapshot:
        """Return the current observable status, including failures and limits."""
        with self._lock:
            return MeshStreamSnapshot(
                self._generation,
                self.state,
                self.requested_level,
                self.displayed_level,
                self.reads,
                self.cache_hits,
                self.stale_reads,
                self.evictions,
                self.cache_bytes,
                self.staged_bytes,
                self.phase_chunks_done,
                self.phase_chunks_total,
                len(self.displayed_keys),
                self.limited,
                str(self.error) if self.error else None,
            )

    def close(self) -> None:
        """Cancel stream-owned work without stopping a borrowed runtime.

        Cancellation cannot interrupt a backend's already running worker thread.
        The backend must serialize its own close with those reads, as the Zarr
        Vectors source does. Closed generations can never publish their result.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.state = "closed"
            self._generation += 1
            pending = tuple(self._pending)
            self._cache.clear()
            self.cache_bytes = self.staged_bytes = 0
        # Cancel every submitted generation: cancelling only the latest one
        # misses predecessors if that coroutine has not started awaiting them.
        for future in pending:
            future.cancel()
        if self._owns_runtime:
            self.runtime.close()
