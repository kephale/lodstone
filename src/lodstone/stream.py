"""Progressive, cancellable execution of Lodstone plans."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import Future
from dataclasses import replace
from itertools import product
from typing import Any, Self

import numpy as np

from .model import (
    ChunkEvent,
    ChunkState,
    Plan,
    PlanDelta,
    Region,
    Status,
    StreamDiagnostics,
    Tile,
    TileKey,
    Update,
    View,
)
from .planner import Planner
from .source import Source
from .target import ResidencyLease, Target

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
        bytes_per_second: float | None = None,
    ) -> None:
        if (
            workers <= 0
            or cpu_cache <= 0
            or inflight <= 0
            or batch_size <= 0
            or (bytes_per_second is not None and bytes_per_second <= 0)
        ):
            raise ValueError(
                "workers, cache, inflight, batch size, and rate must be positive"
            )
        self.source = source
        self.target = target
        self.planner = planner or Planner()
        self.dispatch = dispatch
        self.workers = workers
        self.cpu_cache_limit = cpu_cache
        self.inflight_limit = inflight
        self.batch_size = batch_size
        self.bytes_per_second = bytes_per_second

        self._state_lock = threading.RLock()
        self._generation = 0
        self._status = Status()
        self._available: set[TileKey] = set()
        self._lease: Any | None = None
        self._plan: Plan | None = None
        self._delta = PlanDelta(frozenset(), (), (), frozenset())
        self._status_callbacks: list[StatusCallback] = []
        self._closed = False
        self._paused = False

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="lodstone-stream", daemon=True
        )
        self._thread.start()
        self._active: Future[Any] | None = None
        self._read_semaphore: asyncio.Semaphore | None = None
        self._resume_event: asyncio.Event | None = None
        self._rate_lock: asyncio.Lock | None = None
        self._next_read_time = 0.0
        self._chunk_cache: OrderedDict[tuple[int, tuple[int, ...]], np.ndarray] = (
            OrderedDict()
        )
        self._chunk_cache_bytes = 0
        self._chunk_pins: dict[tuple[int, tuple[int, ...]], int] = {}
        self._chunk_states: dict[tuple[int, tuple[int, ...]], ChunkState] = {}
        self._cache_events: deque[ChunkEvent] = deque(maxlen=512)
        self._diagnostics = StreamDiagnostics()
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
            if self._lease is not None:
                return frozenset(self._lease.available_keys)
            return frozenset(self._available)

    @property
    def delta(self) -> PlanDelta:
        """Coverage changes applied by the current or most recent request."""

        with self._state_lock:
            return self._delta

    @property
    def diagnostics(self) -> StreamDiagnostics:
        """Native-read counters for the current or most recent pass."""
        with self._state_lock:
            return self._diagnostics

    @property
    def cache_events(self) -> tuple[ChunkEvent, ...]:
        """Recent native-chunk state transitions, oldest first."""
        with self._state_lock:
            return tuple(self._cache_events)

    @property
    def chunk_states(self) -> dict[tuple[int, tuple[int, ...]], ChunkState]:
        """Snapshot of the latest state recorded for each native chunk."""
        with self._state_lock:
            return dict(self._chunk_states)

    def on_status_changed(self, callback: StatusCallback) -> Callable[[], None]:
        with self._state_lock:
            self._status_callbacks.append(callback)

        def disconnect() -> None:
            with self._state_lock:
                if callback in self._status_callbacks:
                    self._status_callbacks.remove(callback)

        return disconnect

    def plan(
        self,
        view: View,
        *,
        previous_target_level: int | None = None,
        lod_hysteresis: float = 0.0,
    ) -> Plan:
        """Plan a view without changing the active generation."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            # Lease-aware targets explicitly confirm storage that survives a
            # replan. Legacy targets retain conservative generation behavior.
            if self._lease is not None:
                available = frozenset(self._lease.available_keys)
            elif self._status.state == "complete":
                available = frozenset(self._available)
            else:
                available = frozenset()
        layout = self.target.layout(view, self.source.pyramid)
        return self.planner.plan(
            self.source.pyramid,
            view,
            layout,
            available=available,
            previous_target_level=previous_target_level,
            lod_hysteresis=lod_hysteresis,
        )

    def update(self, view: View) -> Plan:
        """Plan and start streaming the newest view, returning its plan."""

        return self.submit(view, self.plan(view))

    def submit(self, view: View, plan: Plan) -> Plan:
        """Execute an adapter-supplied plan for the newest view.

        Viewer integrations may already own renderer-specific region selection
        or need to preserve an established loading policy exactly. ``submit``
        bypasses :attr:`planner` but retains the stream's cancellation,
        native-chunk caching, batching, pacing, and stale-generation rejection.
        """

        with self._state_lock:
            if self._closed:
                raise RuntimeError("stream is closed")
        layout = self.target.layout(view, self.source.pyramid)
        self._validate_submitted_plan(plan)
        return self._start(view, plan, layout)

    def _start(self, view: View, plan: Plan, layout: Any) -> Plan:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("stream is closed")
            self._generation += 1
            generation = self._generation
            self._delta = plan.delta(self._plan)
            self._plan = plan
            available = frozenset(self._available)
            cache_chunks = self._diagnostics.cache_chunks
            cache_bytes = self._diagnostics.cache_bytes
            self._diagnostics = StreamDiagnostics(
                generation=generation,
                desired_tiles=len(plan.desired or plan.wanted),
                wanted_tiles=len(plan.wanted),
                unique_native_chunks=len(self._native_chunk_keys(plan.wanted)),
                cache_chunks=cache_chunks,
                cache_bytes=cache_bytes,
            )
        if self._active is not None:
            self._active.cancel()
        self._set_status(
            Status(
                generation=generation,
                state="loading",
                wanted=len(plan.wanted),
                resident=len(available),
                progress=0.0 if plan.wanted else 1.0,
            )
        )
        self._active = asyncio.run_coroutine_threadsafe(
            self._execute(generation, view, plan, layout), self._loop
        )
        return plan

    def _validate_submitted_plan(self, plan: Plan) -> None:
        levels = self.source.pyramid.levels
        if not 0 <= plan.target_level < len(levels):
            raise ValueError("plan target level is outside the source pyramid")
        for tile in (*plan.wanted, *plan.desired):
            if not 0 <= tile.level < len(levels):
                raise ValueError("plan tile level is outside the source pyramid")
            level = levels[tile.level]
            if tile.region.ndim != level.ndim or any(
                start >= stop or stop > size
                for start, stop, size in zip(
                    tile.region.start,
                    tile.region.stop,
                    level.shape,
                    strict=True,
                )
            ):
                raise ValueError("plan tile region is outside its source level")

    def pause(self) -> None:
        """Pause new reads and target delivery without cancelling the pass."""

        with self._state_lock:
            if self._closed:
                return
            self._paused = True
        self._loop.call_soon_threadsafe(self._set_resume_state, False)

    def resume(self) -> None:
        """Resume a pass paused during viewer interaction."""

        with self._state_lock:
            if self._closed:
                return
            self._paused = False
        self._loop.call_soon_threadsafe(self._set_resume_state, True)

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
        self._resume_event = asyncio.Event()
        self._rate_lock = asyncio.Lock()
        if not self._paused:
            self._resume_event.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    async def _execute(
        self, generation: int, view: View, plan: Plan, layout: Any
    ) -> None:
        completed = 0
        bytes_read = 0
        try:
            await self._reconcile_chunk_tasks(self._native_chunk_keys(plan.wanted))
            await self._wait_until_resumed()
            lease = await self._deliver_prepare(generation, view, plan)
            available = (
                frozenset(lease.available_keys) if lease is not None else frozenset()
            )
            wanted = tuple(tile for tile in plan.wanted if tile.key not in available)
            phases = sorted({tile.phase for tile in wanted})
            for phase in phases:
                phase_tiles = [tile for tile in wanted if tile.phase == phase]
                for window in self._tile_windows(phase_tiles):
                    await self._wait_until_resumed()
                    if not self._is_current(generation):
                        return
                    pinned = self._native_chunk_keys(window)
                    self._pin_chunks(pinned)
                    try:
                        results = await asyncio.gather(
                            *(
                                self._tile_update(tile, view, layout, generation)
                                for tile in window
                            )
                        )
                    finally:
                        self._unpin_chunks(pinned, generation)
                    if not self._is_current(generation):
                        return
                    bytes_read += sum(update.data.nbytes for update in results)
                    completed += len(results)
                    await self._deliver_updates(generation, results)
                    self._set_status(
                        Status(
                            generation=generation,
                            state="loading",
                            wanted=len(wanted),
                            resident=len(self.available),
                            inflight=max(0, len(wanted) - completed),
                            bytes_read=bytes_read,
                            progress=completed / len(wanted),
                        )
                    )
                await self._deliver_phase_complete(generation, view, plan, phase)

            if not self._is_current(generation):
                return
            stale = self.available - plan.retain
            if stale:
                await self._deliver_discard(generation, stale)
            await self._deliver_complete(generation, view, plan)
            await self._deliver_redraw(generation)
            self._set_status(
                Status(
                    generation=generation,
                    state="complete",
                    wanted=len(wanted),
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

    async def _reconcile_chunk_tasks(
        self, desired: frozenset[tuple[int, tuple[int, ...]]]
    ) -> None:
        """Keep loading overlap and rebuild queued work in newest priority order."""

        for key, task in tuple(self._chunk_tasks.items()):
            state = self._chunk_states.get(key, ChunkState.NEW)
            if key not in desired or state is ChunkState.QUEUED:
                if self._chunk_tasks.get(key) is task:
                    self._chunk_tasks.pop(key, None)
                task.cancel()
        # Let cancellation release semaphore waiters before requesting the
        # newest ordered tile windows.
        await asyncio.sleep(0)

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

    def _native_chunk_keys(
        self, tiles: Sequence[Tile]
    ) -> frozenset[tuple[int, tuple[int, ...]]]:
        """Return the unique native chunks intersected by ``tiles``."""
        keys = set()
        levels = self.source.pyramid.levels
        for tile in tiles:
            level = levels[tile.level]
            ranges = (
                range(
                    level.chunk_index(axis, tile.region.start[axis]),
                    level.chunk_index(axis, tile.region.stop[axis] - 1) + 1,
                )
                for axis in range(tile.region.ndim)
            )
            keys.update((tile.level, tuple(index)) for index in product(*ranges))
        return frozenset(keys)

    async def _tile_update(
        self, tile: Tile, view: View, layout: Any, generation: int
    ) -> Update:
        await self._wait_until_resumed()
        data = await self._read_region(tile.level, tile.region, generation)
        if layout.squeeze_hidden:
            hidden = tuple(
                axis for axis, value in enumerate(view.index) if value is not None
            )
            if hidden:
                data = np.squeeze(data, axis=hidden)
        transform = self.source.pyramid.levels[tile.level].voxel_to_world
        return Update(tile.key, tile.region, data, transform)

    async def _read_region(
        self, level_index: int, region: Region, generation: int
    ) -> np.ndarray:
        level = self.source.pyramid.levels[level_index]
        chunk_ranges = [
            range(
                level.chunk_index(axis, region.start[axis]),
                level.chunk_index(axis, region.stop[axis] - 1) + 1,
            )
            for axis in range(region.ndim)
        ]
        output = np.empty(region.shape, dtype=level.dtype)
        for chunk_index in product(*chunk_ranges):
            chunk = await self._get_chunk(level_index, tuple(chunk_index), generation)
            bounds = tuple(
                level.chunk_bounds(axis, chunk_index[axis])
                for axis in range(region.ndim)
            )
            chunk_start = tuple(start for start, _stop in bounds)
            chunk_stop = tuple(stop for _start, stop in bounds)
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
        self,
        level_index: int,
        chunk_index: tuple[int, ...],
        generation: int,
    ) -> np.ndarray:
        key = (level_index, chunk_index)
        if key in self._chunk_cache:
            array = self._chunk_cache.pop(key)
            self._chunk_cache[key] = array
            self._increment_diagnostics(generation, cache_hits=1)
            return array
        task = self._chunk_tasks.get(key)
        if task is None:
            previous = self._chunk_states.get(key, ChunkState.NEW)
            reason = "retry requested" if previous is ChunkState.FAILED else "requested"
            self._transition_chunk(generation, key, ChunkState.QUEUED, reason)
            task = self._loop.create_task(
                self._fetch_chunk(level_index, chunk_index, generation)
            )
            self._chunk_tasks[key] = task
            task.add_done_callback(
                lambda completed, key=key: self._forget_chunk_task(key, completed)
            )
        else:
            self._increment_diagnostics(generation, joined_reads=1)
        return await asyncio.shield(task)

    def _forget_chunk_task(
        self,
        key: tuple[int, tuple[int, ...]],
        task: asyncio.Task[np.ndarray],
    ) -> None:
        if self._chunk_tasks.get(key) is task:
            self._chunk_tasks.pop(key, None)

    async def _fetch_chunk(
        self,
        level_index: int,
        chunk_index: tuple[int, ...],
        generation: int,
    ) -> np.ndarray:
        level = self.source.pyramid.levels[level_index]
        bounds = tuple(
            level.chunk_bounds(axis, chunk_index[axis]) for axis in range(level.ndim)
        )
        start = tuple(value for value, _stop in bounds)
        stop = tuple(value for _start, value in bounds)
        # ``update()`` may be called immediately after construction, before the
        # runtime thread has entered ``run_forever``.  Initialise lazily on the
        # runtime loop as well as eagerly in ``_run_loop`` to make that race
        # harmless.
        if self._read_semaphore is None:
            self._read_semaphore = asyncio.Semaphore(self.workers)
        key = (level_index, chunk_index)
        try:
            async with self._read_semaphore:
                self._transition_chunk(
                    generation, key, ChunkState.LOADING, "worker acquired"
                )
                await self._pace_read(Region(start, stop).size * level.dtype.itemsize)
                await self._wait_until_resumed()
                self._increment_diagnostics(generation, source_reads=1)
                array = np.asarray(
                    await self.source.read(level_index, Region(start, stop))
                )
        except asyncio.CancelledError:
            self._transition_chunk(
                generation, key, ChunkState.EVICTED, "read cancelled"
            )
            raise
        except Exception:
            self._transition_chunk(
                generation, key, ChunkState.FAILED, "source read failed"
            )
            raise
        expected = tuple(b - a for a, b in zip(start, stop, strict=True))
        if array.shape != expected:
            self._transition_chunk(
                generation, key, ChunkState.FAILED, "invalid source shape"
            )
            raise ValueError(
                f"source returned shape {array.shape} for region with shape {expected}"
            )
        self._chunk_cache[key] = array
        self._chunk_cache_bytes += array.nbytes
        self._transition_chunk(
            generation, key, ChunkState.READY, "source read completed"
        )
        self._evict_cache(generation, "cache limit")
        self._update_cache_size()
        return array

    def _pin_chunks(self, keys: Collection[tuple[int, tuple[int, ...]]]) -> None:
        for key in keys:
            self._chunk_pins[key] = self._chunk_pins.get(key, 0) + 1

    def _unpin_chunks(
        self,
        keys: Collection[tuple[int, tuple[int, ...]]],
        generation: int,
    ) -> None:
        for key in keys:
            count = self._chunk_pins[key] - 1
            if count:
                self._chunk_pins[key] = count
            else:
                del self._chunk_pins[key]
        self._evict_cache(generation, "request complete")
        self._update_cache_size()

    def _evict_cache(self, generation: int, reason: str) -> None:
        while (
            self._chunk_cache_bytes > self.cpu_cache_limit
            and len(self._chunk_cache) > 1
        ):
            key = next(
                (
                    candidate
                    for candidate in self._chunk_cache
                    if candidate not in self._chunk_pins
                ),
                None,
            )
            if key is None:
                return
            evicted = self._chunk_cache.pop(key)
            self._chunk_cache_bytes -= evicted.nbytes
            self._transition_chunk(generation, key, ChunkState.EVICTED, reason)
            self._increment_diagnostics(generation, evictions=1)

    def _transition_chunk(
        self,
        generation: int,
        key: tuple[int, tuple[int, ...]],
        current: ChunkState,
        reason: str,
    ) -> None:
        with self._state_lock:
            previous = self._chunk_states.get(key, ChunkState.NEW)
            self._chunk_states[key] = current
            self._cache_events.append(
                ChunkEvent(generation, key, previous, current, reason)
            )

    def _increment_diagnostics(self, generation: int, **changes: float) -> None:
        with self._state_lock:
            if self._diagnostics.generation != generation:
                return
            self._diagnostics = replace(
                self._diagnostics,
                **{
                    name: getattr(self._diagnostics, name) + value
                    for name, value in changes.items()
                },
            )

    def _update_cache_size(self) -> None:
        with self._state_lock:
            self._diagnostics = replace(
                self._diagnostics,
                cache_chunks=len(self._chunk_cache),
                cache_bytes=self._chunk_cache_bytes,
            )

    async def _deliver_updates(
        self, generation: int, updates: Sequence[Update]
    ) -> None:
        # Resident-array writes, dtype conversion, and upload-block packing
        # can be substantial. Targets may stage that CPU work here on the
        # stream thread so the dispatched host/UI callback only submits the
        # prepared rendering update.
        stage = getattr(self.target, "stage", None)
        stage_started = time.perf_counter()
        prepared = stage(updates) if stage is not None else updates
        if stage is not None:
            self._increment_diagnostics(
                generation,
                update_stage_seconds=time.perf_counter() - stage_started,
            )

        def apply() -> None:
            if not self._is_current(generation):
                return
            self.target.apply(prepared)
            with self._state_lock:
                self._available.update(update.key for update in updates)
            self.target.redraw()

        await self._run_on_target(apply)

    async def _deliver_prepare(
        self, generation: int, view: View, plan: Plan
    ) -> Any | None:
        prepare = getattr(self.target, "prepare", None)
        if prepare is None:
            return None

        stage_prepare = getattr(self.target, "stage_prepare", None)
        prepared = None
        if stage_prepare is not None and self._is_current(generation):
            stage_started = time.perf_counter()
            prepared = stage_prepare(view, plan)
            self._increment_diagnostics(
                generation,
                prepare_stage_seconds=time.perf_counter() - stage_started,
            )

        result: Any | None = None

        def run() -> None:
            nonlocal result
            if self._is_current(generation):
                if stage_prepare is None:
                    result = prepare(view, plan)
                else:
                    result = prepare(view, plan, prepared)

        await self._run_on_target(run)
        lease = result if isinstance(result, ResidencyLease) else None
        if lease is not None and self._is_current(generation):
            with self._state_lock:
                self._lease = lease
                self._available = set(lease.available_keys)
        return lease

    async def _deliver_complete(self, generation: int, view: View, plan: Plan) -> None:
        complete = getattr(self.target, "complete", None)
        if complete is None:
            return

        def run() -> None:
            if self._is_current(generation):
                complete(view, plan)

        await self._run_on_target(run)

    async def _deliver_phase_complete(
        self, generation: int, view: View, plan: Plan, phase: int
    ) -> None:
        phase_complete = getattr(self.target, "phase_complete", None)
        if phase_complete is None:
            return

        stage_phase = getattr(self.target, "stage_phase", None)
        prepared = None
        if stage_phase is not None and self._is_current(generation):
            stage_started = time.perf_counter()
            prepared = stage_phase(view, plan, phase)
            self._increment_diagnostics(
                generation,
                phase_stage_seconds=time.perf_counter() - stage_started,
            )

        def run() -> None:
            if self._is_current(generation):
                if stage_phase is None:
                    phase_complete(view, plan, phase)
                else:
                    phase_complete(view, plan, phase, prepared)

        await self._run_on_target(run)

    async def _deliver_discard(
        self, generation: int, keys: Collection[TileKey]
    ) -> None:
        def discard() -> None:
            if not self._is_current(generation):
                return
            if self._lease is None:
                self.target.discard(keys)
            else:
                self._lease.release(keys)
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
            if self._loop.is_closed():
                return
            try:
                callback()
            except Exception as error:  # noqa: BLE001 - adapter boundary
                if not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(resolve, error)
            else:
                if not self._loop.is_closed():
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

    async def _wait_until_resumed(self) -> None:
        if self._resume_event is None:
            self._resume_event = asyncio.Event()
            if not self._paused:
                self._resume_event.set()
        await self._resume_event.wait()

    def _set_resume_state(self, resumed: bool) -> None:
        if self._resume_event is None:
            return
        if resumed:
            self._resume_event.set()
        else:
            self._resume_event.clear()

    async def _pace_read(self, nbytes: int) -> None:
        if self.bytes_per_second is None:
            return
        if self._rate_lock is None:
            self._rate_lock = asyncio.Lock()
        async with self._rate_lock:
            now = self._loop.time()
            scheduled = max(now, self._next_read_time)
            self._next_read_time = scheduled + nbytes / self.bytes_per_second
        if scheduled > now:
            await asyncio.sleep(scheduled - now)
