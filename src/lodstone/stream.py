"""Progressive, cancellable execution of Lodstone plans."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import Future
from typing import Any, Self

import numpy as np

from .model import Plan, Region, Status, Tile, TileKey, Update, View
from .planner import Planner
from .source import Source
from .target import Target

Dispatch = Callable[[Callable[[], None]], None]
StatusCallback = Callable[[Status], None]


def _direct_dispatch(callback: Callable[[], None]) -> None:
    callback()


class Stream:
    """Plan and progressively deliver multiscale data to a rendering target.

    The stream owns a private asyncio loop on a background thread. Viewer code
    only needs to provide a dispatcher that schedules callbacks on its required
    UI/render thread.
    """

    def __init__(
        self,
        source: Source,
        target: Target,
        *,
        planner: Planner | None = None,
        dispatch: Dispatch = _direct_dispatch,
        workers: int = 8,
        cpu_cache: int = 2 << 30,
        inflight: int = 256 << 20,
        batch_size: int = 8,
    ) -> None:
        if workers <= 0 or cpu_cache <= 0 or inflight <= 0 or batch_size <= 0:
            raise ValueError(
                "workers, cache, inflight, and batch size must be positive"
            )
        self.source = source
        self.target = target
        self.planner = planner or Planner()
        self.dispatch = dispatch
        self.workers = workers
        self.cpu_cache_limit = cpu_cache
        self.inflight_limit = inflight
        self.batch_size = batch_size

        self._state_lock = threading.RLock()
        self._generation = 0
        self._status = Status()
        self._available: set[TileKey] = set()
        self._status_callbacks: list[StatusCallback] = []
        self._closed = False

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="lodstone-stream", daemon=True
        )
        self._thread.start()
        self._active: Future[Any] | None = None
        self._read_semaphore: asyncio.Semaphore | None = None
        self._chunk_cache: OrderedDict[tuple[int, tuple[int, ...]], np.ndarray] = (
            OrderedDict()
        )
        self._chunk_cache_bytes = 0
        self._chunk_tasks: dict[
            tuple[int, tuple[int, ...]], asyncio.Task[np.ndarray]
        ] = {}

    @property
    def status(self) -> Status:
        with self._state_lock:
            return self._status

    @property
    def available(self) -> frozenset[TileKey]:
        with self._state_lock:
            return frozenset(self._available)

    def on_status_changed(self, callback: StatusCallback) -> Callable[[], None]:
        with self._state_lock:
            self._status_callbacks.append(callback)

        def disconnect() -> None:
            with self._state_lock:
                if callback in self._status_callbacks:
                    self._status_callbacks.remove(callback)

        return disconnect

    def update(self, view: View) -> Plan:
        """Plan and start streaming the newest view, returning its plan."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            self._generation += 1
            generation = self._generation
            available = frozenset(self._available)
        layout = self.target.layout(view, self.source.pyramid)
        plan = self.planner.plan(self.source.pyramid, view, layout, available=available)
        if self._active is not None:
            self._active.cancel()
        self._set_status(
            Status(
                generation=generation,
                state="loading" if plan.wanted else "complete",
                wanted=len(plan.wanted),
                resident=len(available),
                progress=0.0 if plan.wanted else 1.0,
            )
        )
        if plan.wanted:
            self._active = asyncio.run_coroutine_threadsafe(
                self._execute(generation, view, plan), self._loop
            )
        else:
            self._dispatch_redraw(generation)
        return plan

    def refresh(self) -> None:
        """Drop logical target residency so the next update reloads it."""

        with self._state_lock:
            keys = frozenset(self._available)
            self._available.clear()
        if keys:
            self.dispatch(lambda: self.target.discard(keys))

    def cancel(self) -> None:
        """Cancel delivery for the active view."""

        with self._state_lock:
            self._generation += 1
            generation = self._generation
        if self._active is not None:
            self._active.cancel()
        self._set_status(Status(generation=generation, state="idle"))

    def close(self) -> None:
        """Cancel work and stop the private runtime thread."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            generation = self._generation
        if self._active is not None:
            self._active.cancel()
        self._set_status(Status(generation=generation, state="closed"))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._read_semaphore = asyncio.Semaphore(self.workers)
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    async def _execute(self, generation: int, view: View, plan: Plan) -> None:
        completed = 0
        bytes_read = 0
        try:
            phases = sorted({tile.phase for tile in plan.wanted})
            for phase in phases:
                phase_tiles = [tile for tile in plan.wanted if tile.phase == phase]
                for window in self._tile_windows(phase_tiles):
                    if not self._is_current(generation):
                        return
                    results = await asyncio.gather(
                        *(self._tile_update(tile, view) for tile in window)
                    )
                    if not self._is_current(generation):
                        return
                    bytes_read += sum(update.data.nbytes for update in results)
                    completed += len(results)
                    await self._deliver_updates(generation, results)
                    self._set_status(
                        Status(
                            generation=generation,
                            state="loading",
                            wanted=len(plan.wanted),
                            resident=len(self.available),
                            inflight=max(0, len(plan.wanted) - completed),
                            bytes_read=bytes_read,
                            progress=completed / len(plan.wanted),
                        )
                    )

            if not self._is_current(generation):
                return
            stale = self.available - plan.retain
            if stale:
                await self._deliver_discard(generation, stale)
            await self._deliver_redraw(generation)
            self._set_status(
                Status(
                    generation=generation,
                    state="complete",
                    wanted=len(plan.wanted),
                    resident=len(self.available),
                    bytes_read=bytes_read,
                    progress=1.0,
                )
            )
        except asyncio.CancelledError:
            raise
        # Source and target adapters may raise arbitrary operational errors;
        # turn them into observable stream state instead of killing the loop.
        except Exception as error:  # noqa: BLE001
            if self._is_current(generation):
                self._set_status(
                    Status(
                        generation=generation,
                        state="failed",
                        wanted=len(plan.wanted),
                        resident=len(self.available),
                        bytes_read=bytes_read,
                        progress=completed / max(1, len(plan.wanted)),
                        error=error,
                    )
                )

    def _tile_windows(self, tiles: Sequence[Tile]) -> list[list[Tile]]:
        """Batch tiles without exceeding count or decoded-byte backpressure."""

        windows: list[list[Tile]] = []
        current: list[Tile] = []
        current_bytes = 0
        levels = self.source.pyramid.levels
        for tile in tiles:
            tile_bytes = tile.region.size * levels[tile.level].dtype.itemsize
            if current and (
                len(current) >= self.batch_size
                or current_bytes + tile_bytes > self.inflight_limit
            ):
                windows.append(current)
                current = []
                current_bytes = 0
            current.append(tile)
            current_bytes += tile_bytes
        if current:
            windows.append(current)
        return windows

    async def _tile_update(self, tile: Tile, view: View) -> Update:
        data = await self._read_region(tile.level, tile.region)
        hidden = tuple(
            axis for axis, value in enumerate(view.index) if value is not None
        )
        if hidden:
            data = np.squeeze(data, axis=hidden)
        transform = self.source.pyramid.levels[tile.level].voxel_to_world
        return Update(tile.key, tile.region, data, transform)

    async def _read_region(self, level_index: int, region: Region) -> np.ndarray:
        level = self.source.pyramid.levels[level_index]
        chunk_ranges = [
            range(
                region.start[axis] // level.chunks[axis],
                (region.stop[axis] - 1) // level.chunks[axis] + 1,
            )
            for axis in range(region.ndim)
        ]
        output = np.empty(region.shape, dtype=level.dtype)
        from itertools import product

        for chunk_index in product(*chunk_ranges):
            chunk = await self._get_chunk(level_index, tuple(chunk_index))
            chunk_start = tuple(
                chunk_index[axis] * level.chunks[axis] for axis in range(region.ndim)
            )
            chunk_stop = tuple(
                min(chunk_start[axis] + level.chunks[axis], level.shape[axis])
                for axis in range(region.ndim)
            )
            overlap = region.intersection(Region(chunk_start, chunk_stop))
            if (
                overlap is None
            ):  # pragma: no cover - grid construction guarantees overlap
                continue
            source_slice = tuple(
                slice(
                    overlap.start[axis] - chunk_start[axis],
                    overlap.stop[axis] - chunk_start[axis],
                )
                for axis in range(region.ndim)
            )
            destination_slice = tuple(
                slice(
                    overlap.start[axis] - region.start[axis],
                    overlap.stop[axis] - region.start[axis],
                )
                for axis in range(region.ndim)
            )
            output[destination_slice] = chunk[source_slice]
        return output

    async def _get_chunk(
        self, level_index: int, chunk_index: tuple[int, ...]
    ) -> np.ndarray:
        key = (level_index, chunk_index)
        if key in self._chunk_cache:
            array = self._chunk_cache.pop(key)
            self._chunk_cache[key] = array
            return array
        task = self._chunk_tasks.get(key)
        if task is None:
            task = self._loop.create_task(self._fetch_chunk(level_index, chunk_index))
            self._chunk_tasks[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and not task.cancelled():
                self._chunk_tasks.pop(key, None)

    async def _fetch_chunk(
        self, level_index: int, chunk_index: tuple[int, ...]
    ) -> np.ndarray:
        level = self.source.pyramid.levels[level_index]
        start = tuple(
            chunk_index[axis] * level.chunks[axis] for axis in range(level.ndim)
        )
        stop = tuple(
            min(start[axis] + level.chunks[axis], level.shape[axis])
            for axis in range(level.ndim)
        )
        # ``update()`` may be called immediately after construction, before the
        # runtime thread has entered ``run_forever``.  Initialise lazily on the
        # runtime loop as well as eagerly in ``_run_loop`` to make that race
        # harmless.
        if self._read_semaphore is None:
            self._read_semaphore = asyncio.Semaphore(self.workers)
        async with self._read_semaphore:
            array = np.asarray(await self.source.read(level_index, Region(start, stop)))
        expected = tuple(b - a for a, b in zip(start, stop, strict=True))
        if array.shape != expected:
            raise ValueError(
                f"source returned shape {array.shape} for region with shape {expected}"
            )
        key = (level_index, chunk_index)
        self._chunk_cache[key] = array
        self._chunk_cache_bytes += array.nbytes
        while (
            self._chunk_cache_bytes > self.cpu_cache_limit
            and len(self._chunk_cache) > 1
        ):
            _, evicted = self._chunk_cache.popitem(last=False)
            self._chunk_cache_bytes -= evicted.nbytes
        return array

    async def _deliver_updates(
        self, generation: int, updates: Sequence[Update]
    ) -> None:
        def apply() -> None:
            if not self._is_current(generation):
                return
            self.target.apply(updates)
            with self._state_lock:
                self._available.update(update.key for update in updates)
            self.target.redraw()

        await self._run_on_target(apply)

    async def _deliver_discard(
        self, generation: int, keys: Collection[TileKey]
    ) -> None:
        def discard() -> None:
            if not self._is_current(generation):
                return
            self.target.discard(keys)
            with self._state_lock:
                self._available.difference_update(keys)

        await self._run_on_target(discard)

    async def _deliver_redraw(self, generation: int) -> None:
        def redraw() -> None:
            if self._is_current(generation):
                self.target.redraw()

        await self._run_on_target(redraw)

    async def _run_on_target(self, callback: Callable[[], None]) -> None:
        """Dispatch a target call and wait until the host has applied it."""

        completed: asyncio.Future[None] = self._loop.create_future()

        def resolve(error: Exception | None = None) -> None:
            if completed.done():
                return
            if error is None:
                completed.set_result(None)
            else:
                completed.set_exception(error)

        def run() -> None:
            try:
                callback()
            except Exception as error:  # noqa: BLE001 - adapter boundary
                self._loop.call_soon_threadsafe(resolve, error)
            else:
                self._loop.call_soon_threadsafe(resolve)

        self.dispatch(run)
        await completed

    def _dispatch_redraw(self, generation: int) -> None:
        def redraw() -> None:
            if self._is_current(generation):
                self.target.redraw()

        self.dispatch(redraw)

    def _set_status(self, status: Status) -> None:
        with self._state_lock:
            if status.generation < self._generation:
                return
            self._status = status
            callbacks = tuple(self._status_callbacks)
        for callback in callbacks:
            self.dispatch(lambda callback=callback, status=status: callback(status))

    def _is_current(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._generation and not self._closed
