# Lodstone

Lodstone is a renderer-neutral engine for view-dependent streaming of
multiscale chunked arrays. It turns a multiscale `Source` and a camera `View`
into progressive array `Update`s accepted by a viewer-specific `Target`.

It is intended to be shared by clients such as ChimeraX, napari, and ndv.
Lodstone does not create windows, textures, shaders, layers, or viewer models.

```text
Source + View + Target
          │
        Planner
          │
        Stream
```

## Installation

```bash
pip install lodstone
pip install "lodstone[ome-zarr]"  # OME-Zarr and remote stores
```

Lodstone supports Python 3.11 through 3.14. NumPy is its only required runtime
dependency.

## Example

```python
import numpy as np

from lodstone import Layout, Stream, View
from lodstone.sources import ArrayPyramidSource
from lodstone.testing import RecordingTarget

fine = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
coarse = fine[::4, ::4]

fine_transform = np.eye(3)
coarse_transform = np.diag([4.0, 4.0, 1.0])
source = ArrayPyramidSource(
    [fine, coarse],
    axes=("y", "x"),
    transforms=[fine_transform, coarse_transform],
    chunks=[(64, 64), (32, 32)],
)

# A real integration implements this protocol to upload updates into its
# renderer. RecordingTarget is useful for tests and examples.
target = RecordingTarget(Layout(kind="tiled", block_shape=(64, 64)))

# Map the 256 by 256 world extent into clip coordinates [-1, 1].
world_to_clip = np.array(
    [
        [2 / 256, 0, 0, -1],
        [0, 2 / 256, 0, -1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)
view = View(
    displayed_axes=(0, 1),
    index=(None, None),
    viewport=(800, 800),
    world_to_clip=world_to_clip,
)

with Stream(source, target) as stream:
    stream.update(view)
```

Adapters that already have a renderer-specific region policy can call
``stream.submit(view, plan)``. This executes the supplied regions exactly while
retaining Lodstone's cancellation, native-chunk cache, batching, pacing, and
stale-generation rejection. This is how the napari adapter preserves PR
#9067's camera-bounded 3-D loading behavior.

``Planner.plan_region(...)`` provides an incremental migration path for those
adapters: the viewer may continue choosing the target level and bounded region
while Lodstone owns transform-aware ladder mapping, native-grid enumeration,
memory and axis limits, cache filtering, and delivery priority. Integrations
can compare its stable ``PlanTrace`` against an established planner and retain
their fallback whenever geometry differs.

Viewer integrations normally provide three small pieces:

1. Camera and dimension state converted into `View`.
2. A `Target` that applies array updates to the renderer.
3. A dispatcher that runs target methods on the viewer's render/UI thread.

```python
stream = Stream(
    source,
    target,
    dispatch=run_on_viewer_thread,
)
```

## napari adapter

The optional adapter uses the rendering architecture from napari PR #9067: a
single multiscale layer backed by bounded resident intervals, camera-selected
2-D or 3-D tiles, and partial GPU texture uploads. It passes the source's
original lazy arrays through, so Zarr data is not materialized into dense
NumPy levels:

```python
import napari
from lodstone.adapters.napari import NapariController
from lodstone.sources import OMEZarrSource

source = OMEZarrSource.open("https://example.org/image.zarr")
viewer = napari.Viewer()
controller = NapariController(viewer, source, fixed_index={0: 0})
napari.run()
controller.close()
```

Pass ``layer_type="labels"`` for an integer segmentation pyramid. Image and
Labels layers use the same camera-driven Lodstone planner, cancellation,
caching, and delivery pipeline. Regular and rectilinear native chunk grids
are preserved rather than approximated by a single nominal chunk size.

This currently requires the `lodstone-integration` napari branch based on PR
#9067. Run
`examples/napari_ome_zarr.py` for a two-channel remote example. The core
package still has no napari or Qt dependency.

`examples/napari_zebrahub.py` opens one lazy timepoint from the public
ZSNS001 Zebrahub light-sheet series in 3-D. Its approximately 32 MiB native
chunks make it a useful stress test for cancellation, interaction holds, and
GPU upload pacing. The example exposes `--tile-mib`, `--interval-mib`, and
`--rate-mib` for tuning those constraints. Pass `--trace-chunks` to report the
exact desired/wanted tile counts alongside unique native chunks, cache hits,
joined in-flight reads, actual source reads, and evictions for every pass.
Pass `--diagnostic-levels` to perform the same real source reads while replacing
the returned pixels with solid categorical labels: magenta is missing content,
green is L0, yellow is L1, orange is L2, and deeper levels use additional
stable colors. This makes incomplete viewport coverage visible independently of
the image's contrast or texture values.

## Public concepts

- **Source** — pyramid metadata and asynchronous regional reads.
- **View** — displayed axes, hidden-axis selections, viewport, and camera matrix.
- **Target** — desired dense/tiled/bricked layout and update delivery.
- **Planner** — deterministic visible-tile and LOD selection.
- **Stream** — cancellation, priorities, native-chunk reuse, CPU caching,
  batching, progressive delivery, and stale-generation rejection.
- **Composition** — transform-aware nearest-neighbor backdrop sampling and
  unloaded-chunk filling for bounded dense targets.
- **VirtualData** — a lazy full-shape array view backed by one movable,
  chunk-aligned in-memory interval; `MultiScaleVirtualData` coordinates these
  intervals and coarse-to-fine backdrop composition across pyramid levels.

Storage chunks and display tiles are deliberately distinct. A target may ask
for 32-cubed bricks while the Zarr source stores 16 by 64 by 64 chunks.
Lodstone reads each overlapping native chunk once and assembles the requested
display updates from its decoded cache.

`stream.diagnostics` separates renderer tiles from native storage activity for
the current or most recent generation. `stream.cache_events` records recent
`queued`, `loading`, `ready`, `failed`, and `evicted` transitions, while
`stream.chunk_states` exposes the latest state per native chunk. Native chunks
required by a delivery batch remain pinned until all its display regions have
been assembled, preventing mid-request eviction and avoidable rereads.

## Source adapters

`ArrayPyramidSource` accepts NumPy, Dask, Zarr, or other indexable array-like
levels. `ZarrPyramidSource` opens explicitly named arrays in a Zarr group.
`OMEZarrSource` discovers nested pyramid levels, axes, and per-level scale and
translation transforms from OME-Zarr v0.1-v0.5 metadata. It also supports bare
array pyramids, bounded caches, remote storage options, level limits, and lazy
fixed-axis or singleton-axis selection.

The public chunk-grid utilities normalize NumPy, Dask, regular Zarr, and
rectilinear Zarr metadata into one exact grid used consistently by sources,
planners, resident buffers, and viewer adapters.

Zarr remains a lazy storage source. NumPy arrays are only the concrete buffers
delivered for requested regions.

## Target contract

```python
class Target:
    def layout(self, view, pyramid) -> Layout: ...
    def apply(self, updates) -> None: ...
    def discard(self, keys) -> None: ...
    def redraw(self) -> None: ...
```

Targets with bounded resident windows may additionally implement
`prepare(view, plan)` and `complete(view, plan)`. Preparation runs on the
viewer thread before any updates for a pass; completion runs after refinement
and stale-residency retirement. A plan exposes both its complete `desired`
tile ladder and the cache-filtered `wanted` reads. Interactive viewers can
call `stream.pause()` and `stream.resume()` without discarding the active pass.
`bytes_per_second` can pace aggregate source reads when decoding or remote I/O
would otherwise compete with interaction and rendering.

Dense targets can use `ResidentArrays` to avoid allocating complete pyramid
levels. It stages one full-ND bounding window per desired level, preserves
overlapping content when the camera moves, translates absolute updates into
window-relative writes, and retires coarse/replaced storage on completion.
The viewer still owns the corresponding grid, texture, or volume objects:

```python
from lodstone import Layout, ResidentArrays

resident = ResidentArrays(source.pyramid)


def layout(view, pyramid):
    return Layout(kind="dense", memory_limit=512 * 1024**2, squeeze_hidden=False)


def prepare(view, plan):
    transition = resident.prepare(plan)
    # Create renderer resources for transition.prepared and remove
    # transition.retired resources.


def apply(updates):
    for change in resident.apply(updates):
        # Patch or invalidate the renderer resource for change.window.
        pass


def complete(view, plan):
    transition = resident.complete(plan)
    # Present resident.active[plan.target_level] and retire old resources.
```

The initial expected layouts are:

| Client | Typical layout |
| --- | --- |
| ChimeraX | dense, uniform LOD |
| napari | tiled |
| ndv | dense initially |

Lodstone deliberately stops before physical GPU allocation. The target owns
textures, double buffering, shader indirection, and renderer invalidation.

## Development

```bash
uv run --extra test pytest
uv run --group dev ruff check .
uv run --group dev pyright src
```

The test suite is network-independent. Remote opening and reading has also
been checked against the EBI IDR OME-Zarr v0.4 store used by
`chimerax-ome-zarr`.
