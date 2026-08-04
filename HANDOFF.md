# Development handoff

Last updated: 2026-08-04

## Repository state

### Lodstone

- Repository: `https://github.com/kephale/lodstone`
- Local checkout: `/Users/kharrington/git/uermel/lodstone`
- Branch: `main`, clean and synchronized with `origin/main`
- Current commit: `38ac773` (`feat: add Zebrahub napari stress test`)
- Important preceding commits:
  - `cf917d0`: aggregate source-read pacing
  - `f8bb9b6`: resident pass lifecycle, pause/resume, complete LOD ladders,
    and 3-D front-to-back priority
  - `ffe387c`: PR-style napari adapter
  - `e2b9c18`: initial multiscale streaming core

### napari

- Repository: `https://github.com/kephale/napari`
- Local checkout: `/Users/kharrington/git/kephale/napari`
- Original PR branch: `progressive-loading-rebase`; it was not modified.
- Integration branch: `lodstone-integration`, clean and pushed to
  `origin/lodstone-integration`
- Integration commit: `965a30a2`
- The integration branch is based directly on the current PR branch head,
  `07e4f3fc`.

### chimerax-ome-zarr

- Local checkout: `/Users/kharrington/git/uermel/chimerax-ome-zarr`
- Branch: `main`, clean and one commit ahead of `origin/main`
- Local commit: `8e0bd70` (`feat: add Lodstone streaming adapter`)
- This commit has not been pushed.
- Its streaming extra pins Lodstone commit `cf917d0`; updating the pin to a
  later Lodstone commit is safe when the adapter next changes.

## Current architecture

Lodstone owns renderer-neutral work:

- pyramid metadata and transforms;
- camera/view representation;
- visible tile selection and LOD selection;
- complete coarse-to-fine ladders;
- 3-D front-to-back tile priority;
- asynchronous source reads and decoded native-chunk caching;
- batching and decoded-byte backpressure;
- aggregate read-rate pacing;
- pass cancellation and stale-generation rejection;
- interaction `pause()` and `resume()`;
- target-thread dispatch;
- optional `prepare(view, plan)` and `complete(view, plan)` lifecycle.

Viewer targets own renderer-specific work:

- physical CPU/GPU allocation;
- resident window representation;
- texture updates and upload queues;
- double buffering and presentation;
- renderer invalidation and UI events.

`Plan.desired` is the complete ordered ladder, including already available
tiles. `Plan.wanted` is the cache-filtered read set. `Plan.retain` is logical
target residency after completion. `Layout.squeeze_hidden=False` lets a
resident-volume target receive full-ND singleton slabs instead of display-only
arrays.

## napari integration boundary

The working napari implementation intentionally preserves the rendering path
from napari PR #9067:

- one multiscale Image layer;
- `MultiScaleVirtualData` and bounded resident intervals;
- coarsest backdrops and intermediate-level repair;
- camera-driven 2-D and 3-D subvolume selection;
- texture patching;
- double-buffered image and volume textures;
- GLIR upload metering;
- interaction quality changes and renderer event handling.

The `lodstone-integration` branch replaces the PR's pass execution with
Lodstone. Lodstone now performs chunk reads, decoded caching, batching,
cancellation, stale-pass rejection, rate limiting, and interaction pause.
Arriving updates are written into `VirtualData` and then passed through the
PR's existing `_on_chunks` texture path.

The PR still constructs the pass. Its level selection, viewport interval,
chunk queue, coarse ladder, and napari-specific priority are converted into a
Lodstone `Plan` by `_PassPlanner`. Therefore the next major convergence task is
to make napari produce a real Lodstone `View`, compare Lodstone plans against
the PR plans, and then transfer planning authority without changing rendering.

The public Lodstone adapter entry point is `NapariController`; it calls
`napari.experimental._lodstone_loading.add_lodstone_loading_image`. It requires
the `lodstone-integration` napari branch. Stock napari does not contain this
module.

## ChimeraX integration boundary

The ChimeraX adapter is selected with the `streaming true` open option. It
captures the ChimeraX camera on graphics updates, lets Lodstone choose levels
and chunks, dispatches target work back to the graphics thread, and updates
multichannel ChimeraX volume models.

Current limitation: each visited level receives a dense CPU `ArrayGridData`
buffer. Only requested source chunks are read, but CPU allocation is still for
the complete spatial level and `values_changed()` may cause broad texture
work. The next ChimeraX task is a bounded resident grid/volume target followed
by investigation of the smallest monkey-patch seam for partial texture
updates. Do not redesign the shared planner for this; the pass lifecycle now
has the required resident-window hooks.

## Validated data and commands

### Zebrahub

Default store:

`https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr`

Metadata observed on 2026-08-04:

- axes: `(t, c, z, y, x)`;
- 791 timepoints and one channel;
- finest per-timepoint shape: `448 x 2174 x 2423`, uint16;
- finest chunks after fixing time/channel: `128 x 362 x 362`, about 32 MiB;
- three pyramid levels.

Run from the napari checkout:

```bash
git switch lodstone-integration
.venv/bin/pip install -e "/Users/kharrington/git/uermel/lodstone[ome-zarr]"
.venv/bin/python \
  /Users/kharrington/git/uermel/lodstone/examples/napari_zebrahub.py \
  --time 400
```

Useful options:

```bash
# Aggregate source pacing
.../napari_zebrahub.py --time 400 --rate-mib 64

# Reduce the GPU tile budget
.../napari_zebrahub.py --time 400 --tile-mib 32

# Automated visual capture
.../napari_zebrahub.py --time 400 \
  --screenshot /tmp/lodstone-zebrahub.png --screenshot-delay 90
```

Timepoint 400 was validated remotely in 3-D with the default 64 MiB tile and
512 MiB interval budgets. The `/tmp` screenshots are ephemeral and are not
part of either repository.

### EBI IDR

`examples/napari_ome_zarr.py` uses:

`https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr`

Both 2-D and 3-D, two-channel rendering were validated through Lodstone.

## Last successful checks

Lodstone:

```bash
uv run pytest -q                 # 29 passed
uv run ruff check src tests examples
uv run pyright src
uv build
```

napari integration branch, with Lodstone on `PYTHONPATH`:

```bash
PYTHONPATH="/Users/kharrington/git/uermel/lodstone/src:$PWD" \
  .venv/bin/python -m pytest -q \
  src/napari/experimental/_tests/test_lodstone_loading.py \
  src/napari/experimental/_tests/test_progressive_loading.py
# 79 passed
```

ChimeraX portable:

```bash
PYTHONPATH="/Users/kharrington/git/uermel/lodstone/src:tests/stubs:$PWD" \
  .venv/bin/python -m pytest \
  -m "not remote and not upstream" tests/portable -q
# 14 passed
```

ChimeraX native:

```bash
PYTHONPATH="/Users/kharrington/git/uermel/lodstone/src:$PWD" \
  /Applications/ChimeraX_Daily.app/Contents/bin/python3.14 \
  -m pytest -q tests/chimerax
# 66 passed; existing ChimeraX teardown warnings remain
```

## Known limitations

- Napari planning is not yet shared; only execution is shared.
- The napari Lodstone factory currently supports Image, not Labels.
- Fixed axes are chosen when a napari layer is constructed. Zebrahub time is
  not yet a live napari slider backed by pass cancellation.
- Lodstone `Level.chunks` currently models regular chunks. The napari PR's
  native `VirtualData` supports rectilinear chunks, but converting such a pass
  through `ArrayPyramidSource` needs a richer shared chunk-grid model.
- Lodstone rate limiting governs source reads, not the renderer's independently
  metered GPU upload queue.
- ChimeraX still allocates dense per-level CPU buffers and does not issue
  explicit per-chunk GL texture updates.
- Blender/MicroscopyNodes and ndv adapters have not yet been implemented.

## Recommended next work

1. Capture a real 2-D/3-D napari camera as a Lodstone `View` and build a plan
   trace comparison harness against the PR loader.
2. Move napari level/interval/chunk planning to Lodstone once traces agree;
   retain every PR renderer and interaction feature.
3. Add a reusable bounded-resident target helper based on `prepare`, `apply`,
   and `complete`; use it for ChimeraX and later Blender/ndv adapters.
4. Explore ChimeraX's `Texture3d`/volume drawing update path to determine
   whether partial uploads can be monkey-patched entirely in Python.
5. Implement the Blender/MicroscopyNodes adapter, then ndv, using the same plan
   trace fixtures.
6. Extend the shared chunk-grid metadata for rectilinear chunks and add mutable
   fixed-axis selection for large time series such as Zebrahub.
