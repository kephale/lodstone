# Changelog

Lodstone follows semantic versioning. Viewer adapters remain provisional during
the 0.1 alpha series and may change between prereleases.

## Unreleased

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
