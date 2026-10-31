# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "lodstone @ git+https://github.com/kephale/lodstone@559ff95371aa1784fb1472b0d6843bee73c786b7",
#   "zarr-vectors @ git+https://github.com/kephale/zarr-vectors-py@40bd38045e0c23019c5c88a54824b89f89243e4e",
#   "numpy==2.5.1",
#   "zarr==3.3.0",
#   "pygfx==0.17.0",
#   "rendercanvas==2.7.2",
#   "glfw==2.11.0",
#   "imageio==2.38.1",
#   "pillow==13.0.0",
#   "scipy==1.18.1",
#   "scikit-image==0.26.0",
#   "fast-simplification==0.2.0",
#   "psutil==7.2.2",
# ]
# ///
"""Run the public neuron-surface demo (or --sample nessys) with its own pinned environment."""

from lodstone.adapters.biological_mesh_demo import main

if __name__ == "__main__":
    main()
