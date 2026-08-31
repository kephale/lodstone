from __future__ import annotations

import numpy as np

from lodstone import Layout, PerformanceRecorder, Planner, Stream
from lodstone.testing import RecordingTarget, SimulatedSource


class PhasedRecordingTarget(RecordingTarget):
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
        assert recorder.latest.status.state == "idle"
    finally:
        stream.close()
