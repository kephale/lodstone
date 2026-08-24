# Camera-aware multiscale rendering

Lodstone separates storage and request orchestration from renderer residency.
The planner describes *what* visual coverage is useful; a host target decides
how that coverage becomes dense arrays, texture bricks, or virtual pages. No
storage read, decode, or allocation should occur in a draw callback.

## Visual demand

For level `l`, compose the level-to-world and world-to-clip transforms:

```text
C_l = world_to_clip @ voxel_to_world_l
```

Visibility and priority must use homogeneous clip coordinates. Dividing basis
points by `w` and reconstructing an affine transform loses perspective and can
select the wrong central ray. Level selection estimates the largest visible
voxel footprint, in pixels, at center and corner rays. A level is adequate when
that footprint is below the configured LOD threshold. Sampling the near visible
depth is important: a dataset-center estimate can undersample magnified
foreground voxels.

Requests are staged in this order:

1. Establish a bounded, persistent coarse overview.
2. Cull blocks outside the frustum and blocks known to contribute no opacity.
3. Choose the target level from screen-space voxel footprint.
4. Fill a coherent fine focus box within the renderer memory limit.
5. Order blocks within that box by visual priority, reusing resident blocks
   before issuing reads.

Screen distance is measured from the focus point to the projected block hull,
not merely between centroids. In 3-D, `focus_depth_weight` controls the tradeoff
between canvas coverage and depth. The default depth order is front-to-back,
which suits opaque transfer functions. `focus_depth_target=0.5` instead ranks
distance from the visual depth center, which suits additive fluorescence and
makes the most central fine blocks arrive first.

The next planner refinement should rank marginal visual benefit per byte:

```text
benefit(block) = newly_covered_pixels * lod_error_reduction * opacity_likelihood
score(block)   = benefit(block) / (fetch_bytes + upload_bytes)
```

Dense targets add a constraint: selected blocks must fit in one enclosing
cuboid. Bricked or virtual-texture targets do not, so they can spend the same
budget on a closer approximation to screen-space demand.

## Storage contract

Zarr v3 sharding and a chunk statistics index belong below Lodstone's planner.
Shards amortize object-store request latency, but logical inner chunks remain
the unit of visibility, caching, and cancellation. Range coalescing should
combine queued inner-chunk reads from the same shard without exposing shard
layout to renderers.

An optional per-level chunk index should provide at least finite `min`, `max`,
and occupancy. It must be versioned with the array metadata. Planning can then
avoid a fetch when:

- fluorescence `max` is below the current black level; or
- a transfer function has zero opacity over the chunk's `[min, max]` interval.

Min/max is conservative, not a complete opacity summary for a non-monotonic
transfer function. Histograms or value-range bitsets can be added later without
changing the renderer contract.

## Residency contract

Dense snapshots remain the compatibility path for napari, ChimeraX, and ndv.
They should use pooled CPU buffers and partial texture updates where host APIs
permit it. Fetch and decode run on Lodstone's runtime, never the Qt or render
thread. Process isolation or a native decoder is warranted only when profiling
shows codec GIL contention; network I/O and codecs that release the GIL do not
automatically require another process.

A higher-performance target can expose virtual residency:

- one preallocated 3-D atlas with padded brick slots;
- a 3-D integer page-table texture for OpenGL 3.3 compatibility;
- fallback-level and validity data in each page entry;
- an LRU or clock allocator with leases for pages used by the current frame;
- `glTexSubImage3D` updates into existing textures, never texture allocation in
  the render loop; and
- a small ring of persistently allocated PBOs when asynchronous upload is
  measurable. PBOs remove a synchronous client copy/stall, but the GPU still
  performs a transfer; fences are required before reusing a buffer.

Page-table lookup and atlas sampling are renderer work. Lodstone should supply
page updates and fallback relationships, not inject renderer-specific GLSL.
VisPy can use a modular shader function and texture page table; ChimeraX can use
its native texture/render hooks; WebGPU targets may use storage buffers.

## Modality and channels

Rendering policy should be explicit:

- fluorescence: MIP/additive accumulation, center-out focus by default, and no
  opacity-based early ray termination;
- EM or opaque transfer functions: front-to-back compositing, conservative
  opacity culling, and early termination near full opacity.

Arbitrary channels should not be silently compressed with semantic embeddings:
that is lossy, data-dependent, and cannot preserve quantitative intensities.
Prefer an explicit channel mixing matrix, batching into the renderer's fixed
texture limit, or a precomputed scientifically declared composite. Channel
descriptors may help suggest a mix, but should not define the data path.

Renderer visibility feedback is a later optimization. Asynchronous termination
depth or visible-page IDs can lower the priority of occluded EM blocks. Feedback
must be delayed and conservative: it may cancel pending work, but must not evict
the coarse fallback or make correctness depend on a stale frame.

## Delivery sequence

1. Validate homogeneous camera matrices and center-ray priorities in every host.
2. Add the optional chunk-statistics source contract and zero-fetch culling.
3. Add a renderer-neutral page-residency target API and a VisPy prototype.
4. Add pooled/PBO uploads after measuring upload stalls.
5. Add shard-aware range coalescing at the source/runtime boundary.
6. Add modality profiles and, for opaque EM, conservative visibility feedback.

This sequence improves current dense integrations first while keeping the atlas
and shader work additive rather than making it a prerequisite for correctness.
