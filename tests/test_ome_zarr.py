from __future__ import annotations

import numpy as np
import pytest

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


def test_nested_v05_metadata_and_level_limit(tmp_path) -> None:
    import zarr

    root = zarr.open_group(tmp_path / "nested.zarr", mode="w")
    group = root.create_group("recon").create_group("em")
    group.create_array("0", data=np.zeros((8, 16), dtype=np.uint8), chunks=(4, 4))
    group.create_array("1", data=np.zeros((4, 8), dtype=np.uint8), chunks=(2, 2))
    group.attrs["ome"] = {
        "multiscales": [
            {
                "axes": ["y", "x"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1, 1]}
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [2, 2]}
                        ],
                    },
                ],
            }
        ]
    }

    source = OMEZarrSource.open(tmp_path / "nested.zarr", num_levels=1)
    assert source.pyramid.axes == ("y", "x")
    assert len(source.pyramid.levels) == 1
    assert source.arrays[0].shape == (8, 16)


def test_bare_pyramid_infers_level_transform(tmp_path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "bare.zarr", mode="w")
    group.create_array("0", data=np.zeros((12, 20), dtype=np.uint8), chunks=(4, 5))
    group.create_array("1", data=np.zeros((4, 10), dtype=np.uint8), chunks=(2, 5))

    source = OMEZarrSource.open(tmp_path / "bare.zarr")
    np.testing.assert_allclose(
        np.diag(source.pyramid.levels[1].voxel_to_world)[:2],
        [3, 2],
    )


def test_squeeze_leading_singletons_and_reduce_transform(tmp_path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "squeeze.zarr", mode="w")
    group.create_array(
        "0",
        data=np.zeros((1, 1, 6, 8), dtype=np.uint8),
        chunks=(1, 1, 3, 4),
    )
    group.attrs["multiscales"] = [
        {
            "axes": ["t", "c", "y", "x"],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 2, 3]},
                        {"type": "translation", "translation": [0, 0, 5, 7]},
                    ],
                }
            ],
        }
    ]

    source = OMEZarrSource.open(tmp_path / "squeeze.zarr", squeeze=True)
    assert source.pyramid.axes == ("y", "x")
    assert source.arrays[0].shape == (6, 8)
    assert source.pyramid.levels[0].chunks == (3, 4)
    transform = source.pyramid.levels[0].voxel_to_world
    np.testing.assert_allclose(np.diag(transform)[:2], [2, 3])
    np.testing.assert_allclose(transform[:2, 2], [5, 7])


def test_fixed_axis_by_name(tmp_path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "fixed.zarr", mode="w")
    data = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6)
    group.create_array("0", data=data, chunks=(1, 2, 3))
    group.attrs["multiscales"] = [
        {"axes": ["c", "y", "x"], "datasets": [{"path": "0"}]}
    ]

    source = OMEZarrSource.open(tmp_path / "fixed.zarr", fixed_index={"c": 1})
    assert source.pyramid.axes == ("y", "x")
    np.testing.assert_array_equal(source.arrays[0][:], data[1])


def test_num_levels_must_be_positive(tmp_path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "bad-levels.zarr", mode="w")
    group.create_array("0", data=np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        OMEZarrSource.open(tmp_path / "bad-levels.zarr", num_levels=0)
