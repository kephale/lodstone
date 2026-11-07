# Stream triangle meshes

Lodstone loads complete triangles for the current view, shows coarse detail
first, and replaces it with finer detail as it becomes available.
[See the neuron demo](biological-mesh.md).

## Install

From a Lodstone checkout:

```bash
pip install -e ".[mesh]"                   # headless mesh support
uv sync --frozen --extra biological-demo   # include the neuron viewer
```

The mesh extras pin [this Zarr Vectors fork commit](https://github.com/kephale/zarr-vectors-py/commit/40bd38045e0c23019c5c88a54824b89f89243e4e).
No sibling checkout is needed. Replace the Git pin with a released reader before
[publishing to PyPI](https://github.com/pypi/warehouse/blob/main/warehouse/forklift/metadata.py).

## Open your own mesh

```bash
uv run --frozen --extra mesh-demo python examples/zarr_vectors_mesh.py mesh.zarrvectors \
  --levels 0 1 --errors 0 0.5 --memory-mib 128
```

Supply a justified surface-error bound for each level, in the mesh's physical
coordinate units. The values above are examples. Omit the path to try a sphere.

## Connect a viewer

Your viewer supplies a target, a 3-D `View`, and a callback dispatcher:

```python
from lodstone.mesh import MeshStream
from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

source = ZarrVectorsMeshSource("mesh.zarrvectors", level_errors=[0, 0.5])
stream = MeshStream(source, target, dispatch=run_on_viewer_thread)
stream.update(view)
```

Vertices use XYZ coordinates; set `displayed_axes=(0, 1, 2)` in the view.
The source also accepts an affine transform. Complete triangles include any
vertices they need from neighboring storage chunks.

Close the stream, then the source, when the viewer closes. A shared `Runtime`
stays open for other streams; close it last. Source cleanup waits for an active
blocking read to finish. Use `wait()` only off the GUI thread.

[Stream and target contracts](../src/lodstone/mesh.py) ·
[Zarr Vectors source](../src/lodstone/sources/zarr_vectors.py) ·
[PyGFX target](../src/lodstone/adapters/pygfx_mesh.py)

## Limits

- Raw 3-D triangles only. Draco, object filtering and mixed-LOD stitching are unsupported.
- `memory_limit` bounds decoded cache and staging data. If finer detail does not
  fit, the completed coarse surface stays visible. The whole process can use more memory.
- `max_read_bytes` checks dependency sizes before loading vertex payloads.
  Metadata and decoder scratch use additional memory.
- Chunk-local normals can show shading seams. Large coordinates may need a
  camera-relative origin to preserve fine detail on the GPU.
- Actual GPU allocations and remote mesh latency remain unmeasured. The demos
  stream locally prepared stores.

## Checks

```bash
uv run --extra test pytest
```

Synthetic lifecycle and geometry tests run offline in CI. Public-data reference
checks need a prepared cache. [Neuron checks](../examples/data/hemibrain-reference-validation.json) ·
[Nessys checks](../examples/data/idr0062-reference-validation.json) ·
[Independent reader reproducer](https://github.com/kephale/zarr-vectors-py/blob/feat/bounded-mesh-reads/docs/upstream/mesh-streaming-review.md)
