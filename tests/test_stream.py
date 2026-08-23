from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue

import numpy as np

from lodstone import (
    ChunkState,
    Layout,
    Plan,
    Planner,
    Region,
    ResidentArrays,
    ResidentLease,
    Runtime,
    Stream,
    Tile,
    TileKey,
)
from lodstone.testing import RecordingTarget, SimulatedSource


def test_stream_reuses_native_chunks_for_smaller_display_tiles(
    ortho_view, wait
) -> None:
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(2, 2)))
    stream = Stream(source, target, planner=Planner(progressive=False), batch_size=4)
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert len(target.updates) == 16
        assert len(source.reads) == 4
        assert stream.status.progress == 1
    finally:
        stream.close()


def test_plan_does_not_start_or_replace_a_generation(ortho_view) -> None:
    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        plan = stream.plan(ortho_view((8, 8), viewport=(64, 64)))

        assert plan.wanted
        assert stream.status.generation == 0
        assert stream.status.state == "idle"
        assert target.updates == []
    finally:
        stream.close()


def test_lease_confirmed_tiles_are_reused_while_request_is_loading(
    ortho_view, wait
) -> None:
    class LeaseTarget(RecordingTarget):
        def __init__(self, source) -> None:
            super().__init__(Layout(block_shape=(4, 4), squeeze_hidden=False))
            self.resident = ResidentArrays(source.pyramid)

        def prepare(self, _view, plan):
            self.resident.prepare(plan)
            desired = plan.desired or plan.wanted
            return ResidentLease(self.resident, frozenset(tile.key for tile in desired))

        def apply(self, updates) -> None:
            self.resident.apply(updates)
            super().apply(updates)

        def discard(self, keys) -> None:
            self.resident.discard(keys)
            super().discard(keys)

        def complete(self, _view, plan) -> None:
            self.resident.complete(plan)

    source = SimulatedSource(
        [np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)], latency=0.04
    )
    target = LeaseTarget(source)
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        workers=1,
        batch_size=1,
    )
    view = ortho_view((8, 8), viewport=(64, 64))
    try:
        stream.update(view)
        wait(lambda: stream.status.state == "loading" and len(target.updates) == 1)

        replacement = stream.plan(view)

        assert len(replacement.wanted) == 3
        assert target.updates[0].key not in {tile.key for tile in replacement.wanted}
    finally:
        stream.close()


def test_replan_keeps_loading_overlap_and_cancels_obsolete_queue(
    ortho_view, wait
) -> None:
    source = SimulatedSource(
        [np.zeros((4, 12), dtype=np.uint8)], chunks=[(4, 4)], latency=0.08
    )
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        workers=1,
        batch_size=3,
    )
    view = ortho_view((4, 12), viewport=(64, 64))
    tiles = tuple(
        Tile(
            TileKey(0, (0, index), ()),
            Region((0, index * 4), (4, (index + 1) * 4)),
            float(index),
        )
        for index in range(3)
    )
    first = Plan(tiles, frozenset(tile.key for tile in tiles), 0, tiles)
    replacement_tiles = (tiles[0], tiles[2])
    replacement = Plan(
        replacement_tiles,
        frozenset(tile.key for tile in replacement_tiles),
        0,
        replacement_tiles,
    )
    try:
        stream.submit(view, first)
        wait(lambda: stream.chunk_states.get((0, (0, 0))) is ChunkState.LOADING)
        stream.submit(view, replacement)
        wait(lambda: stream.status.state == "complete")

        assert source.reads.count((0, tiles[0].region)) == 1
        assert (0, tiles[1].region) not in source.reads
        assert (0, tiles[2].region) in source.reads
    finally:
        stream.close()


def test_stream_diagnostics_separate_tiles_from_native_reads(ortho_view, wait) -> None:
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(2, 2)))
    stream = Stream(source, target, planner=Planner(progressive=False), batch_size=4)
    view = ortho_view((8, 8), viewport=(64, 64))
    try:
        plan = stream.update(view)
        wait(lambda: stream.status.state == "complete")

        first = stream.diagnostics
        assert first.desired_tiles == 16
        assert first.wanted_tiles == 16
        assert first.unique_native_chunks == 4
        assert first.source_reads == 4
        assert first.cache_chunks == 4
        assert all(state is ChunkState.READY for state in stream.chunk_states.values())

        stream.submit(view, plan)
        wait(
            lambda: (
                stream.status.state == "complete"
                and stream.status.generation > first.generation
            )
        )

        second = stream.diagnostics
        assert second.source_reads == 0
        assert second.cache_hits == 16
        assert second.unique_native_chunks == 4
    finally:
        stream.close()


def test_stream_pins_active_batch_until_region_assembly(ortho_view, wait) -> None:
    data = np.arange(32, dtype=np.uint8).reshape(4, 8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        batch_size=2,
        cpu_cache=16,
    )
    try:
        stream.update(ortho_view(data.shape, viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")

        evictions = [
            event
            for event in stream.cache_events
            if event.current is ChunkState.EVICTED
        ]
        assert len(source.reads) == 2
        assert len(evictions) == 1
        assert evictions[0].reason == "request complete"
        assert stream.diagnostics.evictions == 1
        assert stream.diagnostics.cache_chunks == 1
    finally:
        stream.close()


def test_failed_chunk_is_observable_and_retried_on_next_request(
    ortho_view, wait
) -> None:
    class FailOnceSource(SimulatedSource):
        failed = False

        async def read(self, level, region):
            if not self.failed:
                self.failed = True
                raise OSError("temporary source failure")
            return await super().read(level, region)

    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    source = FailOnceSource([data], chunks=[(4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False))
    view = ortho_view(data.shape, viewport=(64, 64))
    try:
        plan = stream.update(view)
        wait(lambda: stream.status.state == "failed")
        assert set(stream.chunk_states.values()) == {ChunkState.FAILED}

        stream.submit(view, plan)
        wait(lambda: stream.status.state == "complete")

        transitions = [event.current for event in stream.cache_events]
        assert ChunkState.FAILED in transitions
        assert any(
            event.current is ChunkState.QUEUED and event.reason == "retry requested"
            for event in stream.cache_events
        )
        assert set(stream.chunk_states.values()) == {ChunkState.READY}
    finally:
        stream.close()


def test_stream_submits_adapter_plan_without_replanning(ortho_view, wait) -> None:
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False))
    key = TileKey(0, (1, 1), ())
    tile = Tile(key, Region((4, 4), (8, 8)), 0.0)
    plan = Plan((tile,), frozenset({key}), 0, (tile,))
    try:
        returned = stream.submit(
            ortho_view((8, 8), viewport=(64, 64)),
            plan,
        )
        wait(lambda: stream.status.state == "complete")

        assert returned is plan
        assert [update.region for update in target.updates] == [tile.region]
        np.testing.assert_array_equal(target.updates[0].data, data[4:8, 4:8])
    finally:
        stream.close()


def test_target_stages_updates_before_host_dispatch(ortho_view, wait) -> None:
    calls: Queue[Callable[[], None]] = Queue()

    class StagingRecordingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(kind="tiled", block_shape=(4, 4)))
            self.stage_thread: int | None = None
            self.apply_thread: int | None = None

        def stage(self, updates):
            self.stage_thread = threading.get_ident()
            return ("staged", tuple(updates))

        def apply(self, prepared) -> None:
            marker, updates = prepared
            assert marker == "staged"
            self.apply_thread = threading.get_ident()
            super().apply(updates)

    target = StagingRecordingTarget()
    source = SimulatedSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)])
    stream = Stream(source, target, dispatch=calls.put)
    main_thread = threading.get_ident()
    try:
        stream.update(ortho_view((4, 4), viewport=(64, 64)))
        deadline = time.monotonic() + 5
        while stream.status.state != "complete":
            assert time.monotonic() < deadline
            try:
                callback = calls.get(timeout=0.05)
            except Empty:
                continue
            callback()
        assert target.stage_thread is not None
        assert target.stage_thread != main_thread
        assert target.apply_thread == main_thread
    finally:
        stream.close()


def test_target_stages_phase_before_host_dispatch(ortho_view, wait) -> None:
    calls: Queue[Callable[[], None]] = Queue()

    class PhaseStagingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(kind="tiled", block_shape=(4, 4)))
            self.stage_thread: int | None = None
            self.publish_thread: int | None = None
            self.prepared = None

        def stage_phase(self, view, plan, phase):
            self.stage_thread = threading.get_ident()
            return (plan.target_level, phase)

        def phase_complete(self, view, plan, phase, prepared) -> None:
            self.publish_thread = threading.get_ident()
            self.prepared = prepared

    target = PhaseStagingTarget()
    source = SimulatedSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)])
    stream = Stream(source, target, dispatch=calls.put)
    main_thread = threading.get_ident()
    try:
        stream.update(ortho_view((4, 4), viewport=(64, 64)))
        deadline = time.monotonic() + 5
        while stream.status.state != "complete":
            assert time.monotonic() < deadline
            try:
                callback = calls.get(timeout=0.05)
            except Empty:
                continue
            callback()
        assert target.stage_thread is not None
        assert target.stage_thread != main_thread
        assert target.publish_thread == main_thread
        assert target.prepared == (0, 0)
        assert stream.diagnostics.phase_stage_seconds > 0
    finally:
        stream.close()


def test_target_stages_prepare_before_host_dispatch(ortho_view) -> None:
    calls: Queue[Callable[[], None]] = Queue()

    class PrepareStagingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(kind="tiled", block_shape=(4, 4)))
            self.stage_thread: int | None = None
            self.prepare_thread: int | None = None

        def stage_prepare(self, view, plan):
            self.stage_thread = threading.get_ident()
            return plan.target_level

        def prepare(self, view, plan, prepared) -> None:
            self.prepare_thread = threading.get_ident()
            assert prepared == plan.target_level

    target = PrepareStagingTarget()
    source = SimulatedSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)])
    stream = Stream(source, target, dispatch=calls.put)
    main_thread = threading.get_ident()
    try:
        stream.update(ortho_view((4, 4), viewport=(64, 64)))
        deadline = time.monotonic() + 5
        while stream.status.state != "complete":
            assert time.monotonic() < deadline
            try:
                callback = calls.get(timeout=0.05)
            except Empty:
                continue
            callback()
        assert target.stage_thread is not None
        assert target.stage_thread != main_thread
        assert target.prepare_thread == main_thread
        assert stream.diagnostics.prepare_stage_seconds > 0
    finally:
        stream.close()


def test_stream_reads_rectilinear_native_chunks(ortho_view, wait) -> None:
    data = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)
    source = SimulatedSource(
        [data],
        chunks=[((2, 3, 2), (4, 1, 4))],
    )
    target = RecordingTarget(Layout(kind="tiled"))
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        stream.update(ortho_view(data.shape, viewport=(128, 128)))
        wait(lambda: stream.status.state == "complete")

        assert len(target.updates) == 9
        assert len(source.reads) == 9
        assert {read.shape for _level, read in source.reads} == {
            (2, 4),
            (2, 1),
            (3, 4),
            (3, 1),
        }
        for update in target.updates:
            np.testing.assert_array_equal(
                update.data,
                data[update.region.slices()],
            )
    finally:
        stream.close()


def test_stream_squeezes_hidden_axes(ortho_view, wait) -> None:
    data = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    source = SimulatedSource([data], chunks=[(1, 4, 4)])
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    view = ortho_view(
        data.shape, displayed_axes=(1, 2), index=(1, None, None), viewport=(64, 64)
    )
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        stream.update(view)
        wait(lambda: stream.status.state == "complete")
        assert target.updates
        assert all(update.data.ndim == 2 for update in target.updates)
        assert min(update.data.min() for update in target.updates) >= 64
    finally:
        stream.close()


def test_target_can_keep_hidden_axes(ortho_view, wait) -> None:
    data = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    source = SimulatedSource([data], chunks=[(1, 4, 4)])
    target = RecordingTarget(
        Layout(kind="bricked", block_shape=(1, 4, 4), squeeze_hidden=False)
    )
    view = ortho_view(
        data.shape,
        displayed_axes=(1, 2),
        index=(1, None, None),
        viewport=(64, 64),
    )
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        stream.update(view)
        wait(lambda: stream.status.state == "complete")
        assert target.updates
        assert all(update.data.ndim == 3 for update in target.updates)
        assert all(update.data.shape[0] == 1 for update in target.updates)
    finally:
        stream.close()


def test_new_generation_rejects_stale_delivery(ortho_view, wait) -> None:
    data = np.stack([np.zeros((8, 8), dtype=np.uint8), np.ones((8, 8), dtype=np.uint8)])
    source = SimulatedSource([data], chunks=[(1, 4, 4)], latency=0.05)
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False), batch_size=1)
    try:
        first = ortho_view(
            data.shape,
            displayed_axes=(1, 2),
            index=(0, None, None),
            viewport=(64, 64),
        )
        second = ortho_view(
            data.shape,
            displayed_axes=(1, 2),
            index=(1, None, None),
            viewport=(64, 64),
        )
        stream.update(first)
        stream.update(second)
        wait(lambda: stream.status.state == "complete")
        assert target.updates
        assert all(np.all(update.data == 1) for update in target.updates)
        assert all(update.key.selection[0] == 1 for update in target.updates)
    finally:
        stream.close()


def test_loading_replan_redelivers_tiles_from_superseded_pass(ortho_view, wait) -> None:
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    source = SimulatedSource([data], chunks=[(4, 4)], latency=0.05)
    target = RecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False), batch_size=1)
    view = ortho_view(data.shape, viewport=(128, 128))
    try:
        first = stream.update(view)
        wait(lambda: bool(stream.available) and stream.status.state == "loading")

        second = stream.update(view)

        assert len(stream.available) < len(first.desired)
        assert len(second.wanted) == len(second.desired)
        wait(lambda: stream.status.state == "complete")
        assert len(target.updates) >= len(second.desired)
    finally:
        stream.close()


def test_status_callbacks_observe_completion(ortho_view, wait) -> None:
    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    states = []
    stream = Stream(source, target, planner=Planner(progressive=False))
    stream.on_status_changed(lambda status: states.append(status.state))
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert states[0] == "loading"
        assert states[-1] == "complete"
    finally:
        stream.close()


def test_stream_waits_for_host_dispatch_before_completion(ortho_view) -> None:
    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    callbacks: Queue[Callable[[], None]] = Queue()
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        dispatch=callbacks.put,
    )
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        callback = callbacks.get(timeout=2)
        assert stream.status.state == "loading"
        assert not target.updates
        callback()
        deadline = time.monotonic() + 2
        while stream.status.state == "loading" and time.monotonic() < deadline:
            try:
                callbacks.get(timeout=0.01)()
            except Empty:
                pass
        assert stream.status.state == "complete"
        assert target.updates
    finally:
        stream.close()


def test_queued_host_delivery_is_harmless_after_close(ortho_view) -> None:
    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    callbacks: Queue[Callable[[], None]] = Queue()
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        dispatch=callbacks.put,
    )
    stream.update(ortho_view((8, 8), viewport=(64, 64)))
    callback = callbacks.get(timeout=2)

    stream.close()
    callback()

    assert stream.status.state == "closed"


def test_inflight_limit_bounds_delivered_batches(ortho_view, wait) -> None:
    class BatchRecordingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(block_shape=(2, 2)))
            self.batch_bytes = []

        def apply(self, updates) -> None:
            self.batch_bytes.append(sum(update.data.nbytes for update in updates))
            super().apply(updates)

    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = BatchRecordingTarget()
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        batch_size=16,
        inflight=8,
    )
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert target.batch_bytes
        assert max(target.batch_bytes) <= 8
    finally:
        stream.close()


def test_progressive_coarse_tiles_are_discarded_after_refinement(
    ortho_view, wait
) -> None:
    source = SimulatedSource(
        [
            np.zeros((16, 16), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.uint8),
        ],
        transforms=[np.eye(3), np.diag([2.0, 2.0, 1.0])],
        chunks=[(4, 4), (4, 4)],
    )
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=True))
    try:
        stream.update(ortho_view((16, 16), viewport=(128, 128)))
        wait(lambda: stream.status.state == "complete")
        assert {key.level for key in stream.available} == {0}
        assert target.discarded
        assert {key.level for key in target.discarded} == {1}
    finally:
        stream.close()


def test_pass_lifecycle_wraps_delivery(ortho_view, wait) -> None:
    class ResidentTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(block_shape=(4, 4)))
            self.events = []

        def prepare(self, view, plan) -> None:
            self.events.append(("prepare", plan.target_level, len(plan.desired)))

        def apply(self, updates) -> None:
            self.events.append(("apply", len(updates)))
            super().apply(updates)

        def complete(self, view, plan) -> None:
            self.events.append(("complete", plan.target_level))

    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = ResidentTarget()
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert target.events[0][0] == "prepare"
        assert target.events[-1][0] == "complete"
        assert any(event[0] == "apply" for event in target.events)
    finally:
        stream.close()


def test_phase_lifecycle_follows_each_progressive_phase(ortho_view, wait) -> None:
    class PhaseRecordingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(block_shape=(4, 4)))
            self.phases = []

        def phase_complete(self, view, plan, phase) -> None:
            self.phases.append(phase)

    source = SimulatedSource(
        [
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        ],
        transforms=[np.eye(3), np.diag([2.0, 2.0, 1.0])],
        chunks=[(4, 4), (4, 4)],
    )
    target = PhaseRecordingTarget()
    stream = Stream(source, target, planner=Planner(progressive=True))
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert target.phases == [0, 1]
    finally:
        stream.close()


def test_phase_lifecycle_presents_already_resident_phases(ortho_view, wait) -> None:
    class PhaseRecordingTarget(RecordingTarget):
        def __init__(self) -> None:
            super().__init__(Layout(block_shape=(4, 4)))
            self.phases: list[int] = []

        def phase_complete(self, view, plan, phase) -> None:
            self.phases.append(phase)

    source = SimulatedSource(
        [
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        ],
        transforms=[np.eye(3), np.diag([2.0, 2.0, 1.0])],
        chunks=[(4, 4), (4, 4)],
    )
    target = PhaseRecordingTarget()
    stream = Stream(source, target, planner=Planner(progressive=True))
    view = ortho_view((8, 8), viewport=(64, 64))
    try:
        stream.update(view)
        wait(lambda: stream.status.state == "complete")
        first_reads = tuple(source.reads)

        cached = stream.plan(view)
        assert {tile.phase for tile in cached.wanted} == {0}
        assert any(tile.phase == 1 for tile in cached.desired)
        assert not any(tile.phase == 1 for tile in cached.wanted)
        stream.submit(view, cached)
        wait(
            lambda: stream.status.state == "complete" and stream.status.generation == 2
        )

        assert target.phases == [0, 1, 0, 1]
        assert tuple(source.reads) == first_reads
    finally:
        stream.close()


def test_shared_runtime_stages_without_blocking_other_streams(ortho_view, wait) -> None:
    stage_started = threading.Event()
    release_stage = threading.Event()

    class BlockingTarget(RecordingTarget):
        def stage(self, updates):
            stage_started.set()
            assert release_stage.wait(timeout=5)
            return updates

    runtime = Runtime(compute_workers=2)
    slow = Stream(
        SimulatedSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)]),
        BlockingTarget(Layout(block_shape=(4, 4))),
        runtime=runtime,
    )
    fast = Stream(
        SimulatedSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)]),
        RecordingTarget(Layout(block_shape=(4, 4))),
        runtime=runtime,
    )
    view = ortho_view((4, 4), viewport=(64, 64))
    try:
        slow.update(view)
        assert stage_started.wait(timeout=5)
        fast.update(view)
        wait(lambda: fast.status.state == "complete")
        assert slow.status.state == "loading"

        release_stage.set()
        wait(lambda: slow.status.state == "complete")
        slow.close()
        assert not runtime.closed
    finally:
        release_stage.set()
        slow.close()
        fast.close()
        runtime.close()

    assert runtime.closed


def test_closing_stream_cancels_its_reads_on_shared_runtime(ortho_view, wait) -> None:
    started = threading.Event()
    cancelled = threading.Event()

    class BlockingSource(SimulatedSource):
        async def read(self, level, region):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    runtime = Runtime()
    stream = Stream(
        BlockingSource([np.zeros((4, 4), dtype=np.uint8)], chunks=[(4, 4)]),
        RecordingTarget(Layout(block_shape=(4, 4))),
        runtime=runtime,
    )
    try:
        stream.update(ortho_view((4, 4), viewport=(64, 64)))
        assert started.wait(timeout=5)
        stream.close()
        wait(cancelled.is_set)
        assert not runtime.closed
    finally:
        stream.close()
        runtime.close()


def test_pause_holds_reads_until_resume(ortho_view, wait) -> None:
    source = SimulatedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False))
    try:
        stream.pause()
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        time.sleep(0.05)
        assert source.reads == []
        assert target.updates == []

        stream.resume()
        wait(lambda: stream.status.state == "complete")
        assert source.reads
        assert target.updates
    finally:
        stream.close()


def test_stream_paces_native_reads(ortho_view, wait) -> None:
    class TimedSource(SimulatedSource):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.read_times: list[float] = []

        async def read(self, level, region):
            self.read_times.append(time.monotonic())
            return await super().read(level, region)

    source = TimedSource([np.zeros((8, 8), dtype=np.uint8)], chunks=[(4, 4)])
    target = RecordingTarget(Layout(block_shape=(4, 4)))
    stream = Stream(
        source,
        target,
        planner=Planner(progressive=False),
        workers=4,
        bytes_per_second=320,
    )
    try:
        stream.update(ortho_view((8, 8), viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        assert len(source.read_times) == 4
        assert source.read_times[-1] - source.read_times[0] >= 0.1
    finally:
        stream.close()
