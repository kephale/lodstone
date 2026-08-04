# Lodstone

Lodstone is a renderer-neutral engine for view-dependent streaming of
multiscale chunked arrays. It turns a multiscale `Source` and a camera `View`
into progressive array `Update`s accepted by a viewer-specific `Target`.

It is intended to be shared by clients such as ChimeraX, Blender, napari, and
ndv. Lodstone does not create windows, textures, shaders, layers, or viewer
models.

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

This currently requires the `lodstone-integration` napari branch based on PR
#9067. Run
`examples/napari_ome_zarr.py` for a two-channel remote example. The core
package still has no napari or Qt dependency.

`examples/napari_zebrahub.py` opens one lazy timepoint from the public
ZSNS001 Zebrahub light-sheet series in 3-D. Its approximately 32 MiB native
chunks make it a useful stress test for cancellation, interaction holds, and
GPU upload pacing. The example exposes `--tile-mib`, `--interval-mib`, and
`--rate-mib` for tuning those constraints.

## Public concepts

- **Source** — pyramid metadata and asynchronous regional reads.
- **View** — displayed axes, hidden-axis selections, viewport, and camera matrix.
- **Target** — desired dense/tiled/bricked layout and update delivery.
- **Planner** — deterministic visible-tile and LOD selection.
- **Stream** — cancellation, priorities, native-chunk reuse, CPU caching,
  batching, progressive delivery, and stale-generation rejection.

Storage chunks and display tiles are deliberately distinct. A target may ask
for 32-cubed bricks while the Zarr source stores 16 by 64 by 64 chunks.
Lodstone reads each overlapping native chunk once and assembles the requested
display updates from its decoded cache.

## Source adapters

`ArrayPyramidSource` accepts NumPy, Dask, Zarr, or other indexable array-like
levels. `ZarrPyramidSource` opens explicitly named arrays in a Zarr group.
`OMEZarrSource` discovers pyramid levels, axes, and per-level scale and
translation transforms from OME-Zarr multiscales metadata.

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

The initial expected layouts are:

| Client | Typical layout |
| --- | --- |
| ChimeraX | dense, uniform LOD |
| Blender/OpenVDB | dense, uniform LOD |
| napari | tiled |
| ndv | dense initially |
| Atlas renderer | bricked, optionally mixed LOD |

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
