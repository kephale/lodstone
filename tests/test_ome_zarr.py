from __future__ import annotations

import numpy as np

from lodstone.sources.ome_zarr import OMEZarrSource


def test_open_ome_zarr_reads_axes_levels_and_transforms(tmp_path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "image.zarr", mode="w")
    group.create_array("0", data=np.zeros((8, 16), dtype=np.uint16), chunks=(4, 4))
    group.create_array("1", data=np.zeros((4, 8), dtype=np.uint16), chunks=(2, 2))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [2.0, 2.0]},
                        {"type": "translation", "translation": [0.5, 1.5]},
                    ],
                },
            ],
        }
    ]

    source = OMEZarrSource.open(tmp_path / "image.zarr")
    assert source.pyramid.axes == ("y", "x")
    assert [level.shape for level in source.pyramid.levels] == [(8, 16), (4, 8)]
    transform = source.pyramid.levels[1].voxel_to_world
    np.testing.assert_allclose(np.diag(transform)[:2], [2.0, 2.0])
    np.testing.assert_allclose(transform[:2, 2], [0.5, 1.5])
