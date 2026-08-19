# Development handoff

Last updated: 2026-08-19

## Repository state

### Lodstone

- Repository: `https://github.com/kephale/lodstone`
- Local checkout: `/Users/kharrington/git/napari/lodstone`
- Branch: `main`, synchronized with `origin/main` before this handoff update
- Latest implementation adds adapter-supplied exact plan execution,
  rectilinear native chunk grids, source fill values, and progressive napari
  Labels support on top of the bounded resident array work.
- Important preceding commits:
  - `06de99c`: development handoff
  - `38ac773`: Zebrahub napari stress test
  - `cf917d0`: aggregate source-read pacing
  - `f8bb9b6`: resident pass lifecycle, pause/resume, complete LOD ladders,
    and 3-D front-to-back priority
  - `ffe387c`: PR-style napari adapter
  - `e2b9c18`: initial multiscale streaming core

### napari

- Repository: `https://github.com/kephale/napari`
- Local integration checkout:
  `/Users/kharrington/git/napari/napari-lodstone-integration`
- Original PR branch: `progressive-loading-rebase`; it was not modified.
- Integration branch: `lodstone-integration`, clean and pushed to
  `origin/lodstone-integration`
- Integration commit: `c8385ae1` (`feat: move progressive planning to Lodstone`)
- The integration branch includes the current PR branch head, `aab611402`,
  including its latest teardown fixes, and canonical napari `main` through
  `f76b6d74b`.

### chimerax-ome-zarr

- Local checkout: `/Users/kharrington/git/uermel/chimerax-ome-zarr`
- Branch: `main`, clean and three commits ahead of `origin/main`
- Latest local commit: `f6b096a` (`feat: patch resident ChimeraX textures`)
- Preceding local commits:
  - `f5934f0` (`feat: bound Lodstone volume residency`)
  - `8e0bd70` (`feat: add Lodstone streaming adapter`)
- These commits have not been pushed.
- Its streaming extra pins Lodstone commit `f001e3f`.

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

The `lodstone-integration` branch delegates pass execution to Lodstone while
preserving the PR's renderer-specific plan exactly. A real 2-D/3-D camera,
viewport, hidden-axis selection, and per-level napari world transforms are
captured as a Lodstone `View` and `Pyramid`. Napari chooses the target level,
chunk-aligned bounding volume, coarse ladder, and delivery order; Lodstone's
`Stream.submit()` performs those exact reads with decoded caching, batching,
cancellation, stale-pass rejection, rate limiting, and interaction pause.
Arriving updates are written into `VirtualData` and then passed through the
PR's existing `_on_chunks` texture path.

The PR still owns bounded resident intervals, coarsest resident-worker reads,
backdrops, texture patching, double buffering, and presentation. Lodstone's
generic frustum planner remains available as a comparison trace, but it is not
authoritative for napari. A real Zebrahub audit showed why: at the same target
level the PR requested 22 desired / 18 missing tiles while strict frustum
intersection requested 16 / 12, omitting six chunks from napari's intentional
camera-bounded cuboid. The submitted Lodstone plan now equals the PR plan.

The public Lodstone adapter entry point is `NapariController`; it calls the
Image or Labels factory in `napari.experimental._lodstone_loading`. The
integration branch also routes PR #9067's automatic Image/Labels replacement
and derived Labels creation through Lodstone. It requires the
`lodstone-integration` napari branch. Stock napari does not contain this module.

## PR #9067 review audit

The review and issue comments were rechecked on 2026-08-19. Their actionable
behavioral points map to the integration as follows:

- 3-D camera direction, extent selection, FOV, texture-size clamps, and
  pathological chunk shapes stay on the proven PR planning path; Lodstone
  executes its exact regions instead of substituting strict frustum culling.
- Rectilinear Zarr/Dask chunk grids are retained end to end in Lodstone rather
  than collapsed to the first chunk size.
- Source fill values initialize Lodstone resident windows and napari virtual
  data instead of being hard-coded to zero.
- Full napari transforms and affine matrices are preserved, and the adapter
  uses `viewer.scene.camera` rather than the deprecated camera alias.
- Half-voxel alignment, backdrops, RGB handling, time-step presentation,
  double-buffered textures, GLIR upload metering, and teardown safety remain
  napari-owned and therefore follow the PR code directly.
- Lodstone core does not hold a viewer reference. `Stream.submit()` is the
  adapter boundary requested in the architectural review: host-specific
  slicing policy produces a plan and the renderer-neutral stream executes it.
- PyQt6 remains the validated backend for this experimental path; the PR's
  PySide6 skips and teardown protections are retained.

## ChimeraX integration boundary

The ChimeraX adapter is selected with the `streaming true` open option. It
captures the ChimeraX camera on graphics updates, lets Lodstone choose levels
and chunks, dispatches target work back to the graphics thread, and updates
multichannel ChimeraX volume models.

ChimeraX now uses Lodstone's reusable `ResidentArrays` helper. `prepare()`
stages full-ND bounding windows for the desired ladder, overlap is preserved
across camera moves, `apply()` performs window-relative writes, and
`complete()` keeps the target window while retiring coarse or replaced
volumes. Grid origins include each window offset. The pre-plan camera-bounds
placeholder is only two samples per spatial axis rather than a dense coarsest
level.

Streaming volumes use ChimeraX's GPU colormap path. Once a scalar 3-D texture
has been initialized, each Lodstone batch is uploaded with `glTexSubImage3D`
at its resident-window-relative offset without calling
`ArrayGridData.values_changed()` or destroying the drawing. Single-plane,
blended, uninitialized, and shape/dtype-mismatched textures conservatively use
the existing full-refresh path.

The texture patch is intentionally isolated in the ChimeraX adapter. Lodstone
still owns renderer-neutral planning and residency while ChimeraX owns OpenGL
context selection, texture validation, and upload.

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
.venv/bin/pip install -e "/Users/kharrington/git/napari/lodstone[ome-zarr]"
.venv/bin/python \
  /Users/kharrington/git/napari/lodstone/examples/napari_zebrahub.py \
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
part of either repository. It was revalidated with Lodstone planning authority
after the current napari changes using a 90-second automated capture.

### EBI IDR

`examples/napari_ome_zarr.py` uses:

`https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr`

Both 2-D and 3-D, two-channel rendering were validated through Lodstone. They
were revalidated after transferring planning authority on 2026-08-04 with
30-second remote screenshot runs; the resulting `/tmp` images are ephemeral.

## Last successful checks

Lodstone:

```bash
uv run --extra test pytest -q    # 40 passed
uv run ruff check src tests examples
uv run pyright src
uv build
```

napari integration branch, with Lodstone on `PYTHONPATH`:

```bash
PYTHONPATH="/Users/kharrington/git/napari/lodstone/src:$PWD" \
  .venv/bin/python -m pytest -q \
  src/napari/experimental/_tests/test_auto_progressive.py \
  src/napari/experimental/_tests/test_lodstone_loading.py \
  src/napari/experimental/_tests/test_progressive_loading.py
# 100 passed with PyQt6
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
# 68 passed; existing ChimeraX teardown warnings remain
```

The development bundle was also installed into ChimeraX Daily and rendered
the two-channel EBI IDR volume above through Lodstone for 30 seconds using the
normal macOS graphics backend. The resulting screenshot was visually valid;
it is ephemeral and not part of the repository. ChimeraX's macOS
`--offscreen` mode could not be used because that distribution has no OSMesa
library.

## Known limitations

- Axes removed with the adapter's `fixed_index` option are chosen when a layer
  is constructed. Axes retained in the source remain live napari sliders and
  changes are handled as cancellable Lodstone generations.
- Lodstone rate limiting governs source reads, not the renderer's independently
  metered GPU upload queue.
- ChimeraX's Lodstone texture uploads do not yet have a renderer-side byte
  meter or queue comparable to napari PR #9067's GLIR upload metering.
- Blender/MicroscopyNodes and ndv adapters have not yet been implemented.

## Recommended next work

1. Implement the Blender/MicroscopyNodes adapter, then ndv, using the same plan
   trace fixtures.
2. Add mutable adapter-level fixed-axis selection for clients that deliberately
   remove very large time or channel axes from the napari layer.
3. Add renderer-side upload byte metering if ChimeraX needs stronger frame-time
   control for very large resident windows.
