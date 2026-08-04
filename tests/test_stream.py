from __future__ import annotations

import time
from collections.abc import Callable
from queue import Empty, Queue

import numpy as np

from lodstone import Layout, Planner, Stream
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
