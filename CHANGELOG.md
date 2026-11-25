# Changelog

Lodstone follows semantic versioning. Viewer adapters remain provisional during
the 0.1 alpha series and may change between prereleases.

## Unreleased

### Performance

- Describe regular chunk grids by their chunk shape instead of listing every
  chunk. A deep pyramid's finest level has millions of chunks per axis
  (8.4 million at 23 levels of 32-voxel tiles), and opening one built and
  validated those lists several times over: about 26 s at 23 levels, now
  well under a second. Zarr arrays whose metadata declares a regular grid
  (Zarr v2, or a v3 regular grid, sharded or not) skip `read_chunk_sizes`;
  `chunk_boundaries` and regular grids are computed once per extent and
  chunk size; `ArrayPyramidSource` keeps regular levels without a per-chunk
  grid, including when given one equal to the array's own; and `Level`
  locates chunks of a regular grid arithmetically. `read_chunk_sizes` stays
  authoritative for everything else.
- Stream deep pyramids without re-reading their grids per chunk. Regular axes
  built by `chunk_sizes_for` are recognized when handed back (for example as
  `VirtualData.read_chunk_sizes`), so `normalize_chunk_sizes`,
  `native_chunks_in_region` and `expand_region_to_chunk_grid` no longer walk
  millions of sizes per call; new `axis_chunk_boundaries` gives one axis's
  boundaries (cached for other grids). `fill_unloaded_chunks` takes optional
  precomputed `boundaries`, which `VirtualData` and resident repair pass. At
  23 levels the finest level streamed about 3 chunks/s; it now completes 893
  chunks in about 11 s.

## 0.1.0a5 - 2026-11-24

### Datasets

- Compute the generated Mandelbulb's power without trigonometry, about 5x
  faster per chunk with numba. Escape counts match the trigonometric form
  except at rare chaotic boundary points. `order` must be a positive integer.

## 0.1.0a4 - 2026-11-24

### Meshes

- Select mesh LOD by the projected error ball rather than its bounding cube,
  which overstated screen error by up to sqrt(3) and loaded one level denser
  than the pixel threshold needs.
- Select levels without sorting and projecting every visible chunk corner:
  one vectorized visibility test, and a constant screen Jacobian for
  orthographic views (about 9x faster for 20k chunks).
- Mesh targets may define `prepare` and `replace_prepared`; the stream then
  merges and builds buffers off the host thread and leaves the host only the
  replacement.

### Performance

- Resample coarse backdrops by slicing and repeating instead of a fancy-index
  gather (about 8x faster for a 3D tile).
- Fill unloaded chunks by testing each chunk only against the loaded regions
  it touches; the fill was quadratic and ran under the resident level's lock.
- Project all of a plan region's tiles with one matrix product.
- Cache each level axis's chunk boundaries for chunk lookups.

## 0.1.0a3 - 2026-11-24

### Meshes

- Add renderer-neutral streaming for Zarr Vectors triangle meshes: reads use
  stored face ownership so triangles crossing chunk boundaries stay complete,
  with conservative frustum culling, uniform level-of-detail switching, bounded
  decoded caches and atomic geometry replacement.
- Publish mesh support as the `mesh` extra, built on the released
  `zarr-vectors>=0.9.4,<0.10`.
- Keep slow geometry replacement outside the mesh stream state lock, so
  snapshots, camera changes and close no longer wait on the host (#24).
- Add PyGFX mesh demos, including biological surfaces and two procedural
  examples: a connected recursive crystal and a continuous spiral zoom.

### Dense images

- Refine dense regions front to back and prefer tissue over background. Fine
  detail no longer reverts to coarse detail after zooming, rotating or
  changing priority settings.
- Add `fit_on_first_publication` so hosts that already placed their camera can
  skip the initial range fit.
- Scope the camera guard to Lodstone's own publication, so camera changes made
  from presentation or status callbacks replan instead of being dropped.
- Add fixed contrast limits (`clims`) to dense targets and controllers.

### Cameras and telemetry

- Add `orthographic_world_to_clip` and a shared `vispy_world_to_clip` camera
  readback for hosts that draw through a VisPy view box.
- Record the submitted view and a compact plan summary in performance samples.

- Add experimental `MeshStream` for view-dependent triangle-mesh LOD with
  frustum culling, bounded decoded caches and atomic level replacement.
- Add `ZarrVectorsMeshSource` (optional `mesh` extra) and a PyGFX mesh target.
- Add neuron and Nessys biological mesh demos.
- Add `fit_on_first_publication` so a dense host can keep its own camera.
- Ignore camera events only during Lodstone's own publication, so a camera moved
  from a host callback replans instead of being dropped.
- Add `clims` for fixed contrast limits on dense targets.
- Add `orthographic_world_to_clip` for building reproducible camera views.
- Record the submitted view and a plan summary in performance snapshots and
  records.
- Add `vispy_world_to_clip` for hosts that draw through a VisPy view box.

## 0.1.0a2 - 2026-08-30

- Add a SceneX dense target and camera-aware controller based on the generic
  dense-host contract.
- Report renderer submissions and presentations from every dense-host target.
- Extract the ndv dense adapter into renderer-neutral `DenseCanvas`,
  `DenseHandle`, and `CameraDenseCanvas` protocols.
- Add reusable `DenseTarget` and `DenseController` implementations while
  preserving the existing ndv import names.
- Expose depth-centered focus policy through the generalized dense controller.
- Add standardized stream, source, host-dispatch, and renderer performance
  snapshots with a bounded cross-viewer recorder.

## 0.1.0a1 - 2026-08-30

- Improve perspective camera-aware level selection by sampling visible depth.
- Prioritize focus blocks by distance to their projected hull instead of only
  their centroid.
- Add configurable depth-centered focus ordering for additive and fluorescence
  volume rendering.
- Document the renderer residency architecture and practical napari, ndv, and
  ChimeraX integration workflows.

## 0.1.0a0 - 2026-08-23

- Add renderer-neutral multiscale chunk planning and progressive streaming.
- Add bounded caches, cancellation, request reprioritization, and diagnostics.
- Add dense resident windows with coarse-to-fine composition.
- Add NumPy, Zarr, and OME-Zarr sources.
- Add experimental napari and ndv adapters.
- Support shared runtimes across layers and channels.
