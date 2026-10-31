# Neuron mesh demo

Explore five branching Drosophila neurons at two levels of detail (LOD).
These close-ups use the same camera and scale.

| Fine: 183,763 triangles | Coarse: 96,951 triangles (47% fewer) |
| --- | --- |
| ![Fine neuron close-up](../img/hemibrain-neurons-smoothed.png) | ![Coarse neuron close-up](../img/hemibrain-neurons-coarse.png) |

[Full-neuron overview](../img/hemibrain-neurons-overview.png)

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git:

```bash
uv run https://raw.githubusercontent.com/kephale/lodstone/refs/heads/feat/zarr-vectors-mesh-streaming/examples/biological_mesh.py
```

The first run downloads about 3.5 MB and builds the surfaces, which can take a
minute or two. Later runs use the cache. No project environment is needed.
[Install from a checkout](mesh-streaming.md#install)

| Control | Action |
| --- | --- |
| Drag / scroll | Orbit / zoom |
| **L** | Auto → coarse → fine → Auto |
| **S** / **R** | Compare the original / smoothed reference with the stream |
| **B** / **M** | Toggle chunk bounds / surfaces |

Auto may start at fine detail; zoom out or press **L** to see coarse detail.
Reference views are unstreamed. Use **R** and **S** for a smoothing comparison.

To save the matched close-ups, append these options to the command above:

```text
--focus 40 182 125 --radius 10 --compact-overlay --offscreen neuron-closeups
```

[Camera settings](../examples/data/hemibrain-closeup-camera.json) ·
[Viewer options and code](../src/lodstone/adapters/biological_mesh_demo.py)

## Data and checks

The five meshes come from the [NAVis hemibrain examples](https://github.com/navis-org/navis/blob/4fe56c87fb8c2b0c9f89ceb3a4beb6dbf8f30839/navis/data/readme.md),
from [Scheffer et al. (2020)](https://doi.org/10.7554/eLife.57443).
Coordinates are in micrometers. Source assets retain their [original terms](https://github.com/navis-org/navis/blob/4fe56c87fb8c2b0c9f89ceb3a4beb6dbf8f30839/LICENSE).

Fine detail retains every supplied face after smoothing. Coarse detail stays
within 0.5 µm of that smoothed surface and passes the recorded topology checks.
These are geometry checks, not a measure of biological accuracy.

[Measured checks](../examples/data/hemibrain-reference-validation.json) ·
[Smoothing](../src/lodstone/neuron_mesh.py) · [Coarse construction](../src/lodstone/mesh_simplification.py) ·
[Limits and memory](mesh-streaming.md#limits) · [Nessys image/surface demo](nessys-mesh.md)
