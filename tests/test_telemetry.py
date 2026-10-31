from __future__ import annotations

import numpy as np

from lodstone import (
    Layout,
    PerformanceRecorder,
    Planner,
    PlanSummary,
    Stream,
    View,
)
from lodstone.testing import RecordingTarget, SimulatedSource


class PhasedRecordingTarget(RecordingTarget):
    def __init__(self, layout: Layout | None = None) -> None:
        super().__init__(layout or Layout(kind="tiled", block_shape=(4, 4)))

    def phase_complete(self, view, plan, phase) -> None:
        return None


def test_stream_and_target_performance_are_comparable(ortho_view, wait) -> None:
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    target = PhasedRecordingTarget(Layout(kind="tiled", block_shape=(4, 4)))
    stream = Stream(source, target, planner=Planner(progressive=False))
    recorder = PerformanceRecorder(stream, host="test-viewer", backend="test-gpu")
    try:
        stream.update(ortho_view(data.shape, viewport=(64, 64)))
        wait(lambda: stream.status.state == "complete")
        sample = recorder.sample()

        assert sample.stream.planned_bytes == data.nbytes
        assert sample.stream.source_bytes == data.nbytes
        assert sample.stream.delivered_bytes == data.nbytes
        assert sample.stream.updates_delivered == 4
        assert sample.stream.phases_presented == 1
        assert sample.stream.time_to_first_phase_seconds is not None
        assert sample.stream.elapsed_seconds > 0
        assert sample.stream.target_wait_seconds > 0
        assert sample.stream.max_target_wait_seconds > 0
        assert sample.target.submitted_bytes == data.nbytes
        assert sample.target.presentations >= 1

        record = recorder.records()[-1]
        assert record["host"] == "test-viewer"
        assert record["backend"] == "test-gpu"
        assert record["status"]["state"] == "complete"
        assert record["stream"]["source_bytes"] == data.nbytes
        assert record["target"]["submitted_bytes"] == data.nbytes
        assert record["view"]["viewport"] == [64, 64]
        assert record["view"]["displayed_axes"] == [0, 1]
        assert record["view"]["index"] == [None, None]
        assert record["view"]["eye"] is None
        assert len(record["view"]["world_to_clip"]) == 4
        assert record["plan"] == {
            "target_level": 0,
            "desired_tiles": 4,
            "wanted_tiles": 4,
        }
    finally:
        recorder.close()
        stream.close()


def test_recorder_can_disconnect_without_losing_samples() -> None:
    source = SimulatedSource([np.zeros((2, 2), dtype=np.uint8)])
    stream = Stream(source, RecordingTarget())
    recorder = PerformanceRecorder(stream, max_samples=2)
    try:
        recorder.sample()
        recorder.close()
        assert len(recorder.samples) == 2
        assert recorder.records()[-1]["view"] is None
        assert recorder.records()[-1]["plan"] is None
        assert recorder.latest.status.state == "idle"
    finally:
        stream.close()


def test_records_keep_the_hidden_axis_selection(wait) -> None:
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    source = SimulatedSource([data], chunks=[(1, 4, 4)])
    stream = Stream(source, PhasedRecordingTarget(), planner=Planner(progressive=False))
    recorder = PerformanceRecorder(stream)
    try:
        view = View((1, 2), (3, None, None), (64, 64), np.eye(4))
        stream.update(view)
        wait(lambda: stream.status.state == "complete")
        assert recorder.records()[-1]["view"]["index"] == [3, None, None]
    finally:
        recorder.close()
        stream.close()


def test_snapshots_do_not_retain_plan_tiles(ortho_view, wait) -> None:
    data = np.zeros((8, 8), dtype=np.uint8)
    source = SimulatedSource([data], chunks=[(4, 4)])
    stream = Stream(source, PhasedRecordingTarget(), planner=Planner(progressive=False))
    try:
        stream.update(ortho_view(data.shape, viewport=(64, 64)))
        assert stream.performance.plan == PlanSummary(0, 4, 4)
    finally:
        stream.close()
