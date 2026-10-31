# Biological image and multiscale surface demo

This demo uses image **6001240**, study **idr0062 / Nessys**, from the Image Data
Resource. The source provides LaminB1 and Dapi fluorescence channels plus
instance labels in the same physical coordinate frame.

- Blin et al. (2019), *Nessys: A new set of tools for the automated detection of
  nuclei within intact tissues and dense 3D cultures*, PLOS Biology 17(8):e3000388.
  <https://doi.org/10.1371/journal.pbio.3000388>
- Dataset catalog: <https://idr.github.io/ome-ngff-samples/>
- Source: <https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr>
- License: **CC BY 4.0**, copyright Blin et al.

## Reference and physical coordinates

The bounded level-0 crop is Z `[48,176)`, Y `[32,224)`, X `[32,224)`. Native
image and label chunks are cached; downloads fail explicitly if a required
chunk is unavailable. SHA-256 checks verify the decoded crop against the pinned
image and label bytes. The cache retains source metadata and each chunk's URL,
checksum and transfer size.

Six complete nuclei are selected by size, excluding labels touching the crop
boundary. IDs are **2, 6, 36, 1, 22, 8**. Meshes are derived from the supplied
labels, not independently established biological ground truth. Unselected
objects remain visible in the image volume.

Voxel spacing is Z **0.5002025531914894 µm**, Y/X **0.3603981534640209 µm**.
The crop origin is retained. Images use ZYX array storage; meshes use physical
XYZ coordinates. Reversing axes also reverses face winding so the orientation
is preserved. An asymmetric anisotropic fixture checks the mapping independently.

The original marching-cubes geometry is saved in `reference.npz`. The fine
streamed surface has **81,566 triangles**. Reading its 88 ownership cells gives
exactly the same multiset of oriented triangles after the declared float32
storage conversion: no missing or duplicated faces.

The coarse surface has **24,466 triangles**, approximately 70% fewer. Each
object retains its component count, Euler characteristic, and closed manifold
edge incidence. A separate one-voxel bridge control passes those checks; that
control is not evidence about arbitrary biological thin structures.

LOD errors include float32 quantization and use a conservative continuous-surface
distance bound relative to the retained fine mesh. The bound uses triangle
edge covers plus nearest-vertex distances; it is deliberately looser than the
sampled vertex distance and is not an RMS error or an accuracy claim about tissue.
The coarse bound is about **5.02 µm**. Automatic LOD therefore stays conservative;
the `L` key also permits explicit coarse/fine comparison.

Lewiner marching cubes can place interior vertices in ambiguous cells away
from the trilinear 0.5 contour (one selected object has residual 0.125). This
is recorded as extraction behavior, not treated as a coordinate-alignment test.

Measured per-object checks and reference digest:
[idr0062-reference-validation.json](../examples/data/idr0062-reference-validation.json).

## Local development

After installing the companion mesh-reader branch and scientific dependencies:

```bash
.venv/bin/python -m lodstone.adapters.biological_mesh_demo
.venv/bin/python -m lodstone.adapters.biological_mesh_demo --offscreen /tmp/biological-mesh
```

`V` toggles the paired image volume, `M` the streamed surfaces, `B` conservative
chunk bounds, `R` the retained **unstreamed** reference, and `L` cycles
automatic/coarse/fine. Drag to orbit and scroll to zoom. The default cache is
`~/.cache/lodstone/idr0062-mesh-v1` (or under `XDG_CACHE_HOME`); use `--cache` to
choose another location. Initial preparation downloads only intersecting
native chunks, then builds the two raw Zarr Vectors surface levels.

## Diagnostics and checks

The on-screen overlay distinguishes loading, complete, limited and failed states.
It shows the displayed LOD, phase progress, decoded cache bytes, active submitted
mesh-buffer bytes, encoded-store bytes, metadata bytes, logical store reads,
cache hits/evictions, rejected stale reads, process RSS and sampled peak RSS.

Store bytes are measured at Zarr's Store interface, including range reads and
metadata. They are not decoded-geometry sizes or an HTTP wire-byte estimate.
The biological demo streams a locally prepared Zarr Vectors store; source-image
HTTP download sizes are recorded separately in `input-manifest.json`.

RSS sampling starts after fixture preparation and includes volume arrays,
reference geometry and renderer state. Actual GPU driver allocations are not
available. Submitted position/index/normal buffer accounting is labeled
separately; it excludes texture/driver overhead and is not a total GPU-memory
claim. A small mesh budget does not cap the whole process or image textures.

```bash
.venv/bin/python -m lodstone.adapters.biological_mesh_demo --offscreen /tmp/mesh-check --diagnostic-replay
.venv/bin/python -m lodstone.adapters.biological_mesh_demo --offscreen /tmp/mesh-limited --memory-mib 1
.venv/bin/python -m lodstone.adapters.biological_mesh_demo --offscreen /tmp/mesh-failure --inject-failure
```

The final command deliberately exits nonzero after saving `failure.png` and
`failure.json`. It verifies that a read fault stays visible rather than being
reported as a successful empty surface. Ordinary GUI read failures remain in
the red overlay and can be retried by changing LOD.

The recorded replay shows zero additional store bytes for a warm decoded-cache
repeat, fewer displayed chunks after panning, a coarse automatic LOD at a far
view, and rejection of a superseded read during rapid cold-cache navigation.
At a 1 MiB cache budget the coarse surface remains visible, refinement is marked
limited, and cache eviction keeps retained arrays within that separate limit.

Baseline measurements, including a large repeated-metadata cost discovered by
these diagnostics: [idr0062-diagnostics-baseline.json](../examples/data/idr0062-diagnostics-baseline.json).
