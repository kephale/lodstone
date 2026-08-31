# Dense host integration

`DenseController` connects Lodstone's planner and stream to viewers that publish
ordinary dense image or volume arrays. The contract deliberately describes host
behavior rather than a GUI toolkit or renderer.

## Required canvas surface

A `DenseCanvas` creates image and volume handles, selects 2-D or 3-D display,
resets the initial camera range, and requests redraws. Each returned
`DenseHandle` accepts immutable array snapshots, contrast limits, a data-order
scale and origin, visibility changes, and removal.

```python
from lodstone.adapters import DenseCanvas, DenseController

assert isinstance(canvas, DenseCanvas)
controller = DenseController(canvas, source, dispatch=run_on_host_thread)
controller.update(initial_view)
```

All storage reads, composition, and array copying occur away from the host
thread. The dispatch callback only publishes completed phases and requests a
redraw. A coarse handle remains visible around the finer camera-focused handle.

## Camera extension

Automatic replanning is enabled when the canvas also satisfies
`CameraDenseCanvas`: it exposes a connectable `cameraChanged` signal and returns
the current `(viewport, world_to_clip)` snapshot from `camera_state()`. Matrices
consume coordinates in displayed data-axis order and preserve homogeneous
coordinates for perspective cameras.

Camera events are trailing-edge debounced. Events caused by Lodstone's own
publication are ignored, and planning happens on the controller's camera worker
rather than the interaction thread.

## Lifecycle

The host owns native visuals; `DenseTarget` owns the handles it creates. Closing
the controller disconnects camera signals, stops pending streaming work, removes
all handles, and closes an internally created runtime. A shared runtime remains
owned by its caller.

Host-specific behavior should be implemented by a thin canvas wrapper or by
overriding `DenseController._make_target`. Renderer APIs, shader code, and GUI
objects do not belong in Lodstone's planner or source contracts.
