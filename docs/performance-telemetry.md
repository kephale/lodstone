# Performance telemetry

`Stream.performance` returns one immutable `PerformanceSnapshot` with these
parts:

- `status`: generation, progress, residency, and failure state;
- `stream`: planned, source-read, cache, delivered, staging, phase, and
  host-dispatch measurements;
- `target`: optional renderer-reported upload and presentation measurements;
- `view`: the camera and selection most recently submitted to the stream, not a
  live reading from the host; and
- `plan`: a `PlanSummary` of the most recently submitted plan.

The stream measurements have the same meaning in every viewer. `planned_bytes`
is logical tile demand, `source_bytes` is decoded native-chunk data read from the
source, and `delivered_bytes` is array data handed to the target. They are
deliberately separate: caching, chunk overlap, dtype conversion, double
buffering, and GPU formats can make each value different.

## Recording a run

```python
from lodstone import PerformanceRecorder

with PerformanceRecorder(
    controller.stream,
    host="ndv",
    backend="pygfx",
) as recorder:
    controller.update(view)
    # wait for completion or interact with the viewer

records = recorder.records()  # JSON-compatible dictionaries
```

The recorder samples status boundaries automatically. A frame timer may call
`sample()` to capture renderer backlog between stream updates. Storage is
bounded by `max_samples`.

## Renderer metrics

A target opts in by implementing `performance_metrics()` and returning
`TargetDiagnostics`. Values cover bytes submitted and uploaded, current upload
backlog, presentation count, aggregate upload time, and the worst observed
upload stall. Counters must be monotonic within the target lifetime and snapshots
must be safe to read from the host or sampling thread.

Targets should report what the renderer actually does. For example, a staged
float conversion or two writes into a double buffer counts the converted or
duplicated GPU bytes, while Lodstone's `delivered_bytes` remains the original
array payload. Unsupported values stay zero rather than being estimated.
