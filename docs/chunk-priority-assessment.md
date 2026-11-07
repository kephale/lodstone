# Curved fluorescence samples: assessment and implementation

This assessment concerns the local 0.1.0a2 source, not a verified installed
version in every viewer. Kyle reports a central fine chunk despite nearer
sample regions and apparently unused capacity. That observation is consistent
with the code; an interactive Zebrahub reproduction has not been performed.

## What the implementation establishes

`Planner` selects one target level from the maximum sampled voxel footprint.
It does not select a different level independently for each brick. Dense hosts
keep a coarse context and compose one enclosing fine cuboid. A curved sample
therefore pays for background between its arms. More RAM can enlarge that box,
but cannot make it follow the sample surface.

The shared DenseTarget and DenseViewerStream previously defaulted to a depth
weight of 0.5: screen distance plus weighted normalized clip depth. Perspective
compresses distant depth differences, making screen center even more dominant.
A focus depth target can deliberately favor an interior plane. Neither is a
nearest-sample-surface metric. The default is now None, using the existing
front-to-back geometric order; explicit numeric weights retain the focus policy.
Viewer integrations that explicitly supply a weight retain their own policy.

The dense default is 64 MiB per planned phase. This is not a total application
or GPU memory cap: coarse residency, intermediate phases, decoded cache,
composition copies and renderer textures also consume memory. Raising this
constant without accounting for those allocations is not a calibrated solution.
`plan_region` and renderer-supplied slice queues also bypass parts of ordinary
planning; their hosts must be inspected independently.

## Changes implemented

- Dense adapter defaults to front-to-back priority. For orthographic views,
  proximity alone does not imply greater projected voxel size; equal resolution
  may be appropriate even when a nearest-first loading policy is requested.
- Crop planning for tiled/bricked layouts sums selected tile bytes. Dense
  layouts retain enclosing-box accounting. This fixes wasted sparse capacity;
  it does not turn a dense host into a sparse renderer.
- `Planner.plan(invisible_tiles=...)` excludes explicitly identified tiles
  before allocation and ordering. Keys include level and hidden-axis selection.
  Callers must invalidate hints when channels, transfer function, clipping,
  time or source revision change. No automatic content index is claimed.

## Content and rendering policy

A bounding box nearest the eye may contain only dark background. Conservative
native-resolution min/max statistics can exclude a chunk when all enabled
channels provably contribute zero under the current transfer function. For
non-monotonic transfer functions, inspect the entire value interval, not just
its endpoints. Coarse averaged intensity cannot prove the absence of fine bright
features. Unknown statistics must not suppress requests.

MIP needs potential maxima along the whole ray; near tissue does not occlude
far tissue. Additive rendering likewise needs far contributions. Opacity
compositing can use conservative transmittance feedback, invalidated on camera
or transfer-function changes. Brightness alone is not scientific importance:
keep coarse coverage and a refinement opportunity for dim structures.

The next renderer capability is independently resident fine bricks with coarse
fallback per spatial region. Rank refinements by projected error reduction and
possible visible contribution per incremental resident byte. Use projected
voxel size rather than raw distance for resolution; use proximity for delivery
order where desired. Include anisotropic transforms, clipping planes, active
channels, opacity/mixing mode and viewport pixel dimensions. Charge actual GPU
format, halos, fallback, in-flight decode and upload staging separately.

## Acceptance experiment

Before further policy changes, compare identical recorded camera poses across
napari, ndv VisPy, ndv pygfx and ChimeraX. Capture effective settings and source
revision, not just package version. Use a curved synthetic fluorescent shell
with bright far branches, dim near branches and empty foreground; then use the
Zebrahub timepoint already used by the demos. Rotate, zoom and switch rendering
modes. Include perspective, orthographic, near-plane intersections and unequal
voxel scales.

Compare old focus, front-to-back and future content-aware brick refinement at
64, 128 and 256 MiB. Record target versus displayed levels, useful fine-covered
pixels by depth, requested/decoded/resident/uploaded bytes, peak CPU/GPU memory,
time to coarse coverage and stable detail, and churn after motion. Require
budget compliance and preserved far MIP contributions. More selected chunks
alone is not evidence of a better image. Reject a content heuristic if it hides
dim or fine structures; reject a budget increase if peak residency exceeds the
configured device allowance. No cross-viewer performance improvement is yet
measured.
