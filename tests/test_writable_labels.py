import asyncio

import numpy as np
import pytest

from lodstone import Level, Pyramid, Region
from lodstone.sources.labels import ZarrLabels


def test_sparse_writes_reopen_with_exact_geometry(tmp_path):
    ref = Pyramid(
        ("y", "x"),
        (
            Level(
                (80_001, 100_001),
                np.dtype("uint16"),
                (64, 64),
                np.array([[2, 0, -38], [0, 0.75, 15], [0, 0, 1]]),
            ),
        ),
    )
    path = tmp_path / "labels.zarr"
    labels = ZarrLabels.create(path, ref)
    assert list(path.glob("0/c/**/*")) == []
    for key, value in [((0, 0), 17), ((0, 1), 34)]:
        chunk = labels.chunk_region(key)
        labels.write_chunk(key, np.full(chunk.shape, value, dtype="uint32"))
    assert len([p for p in (path / "0/c").rglob("*") if p.is_file()]) == 2
    reopened = ZarrLabels.open(path)
    assert reopened.pyramid.axes == ref.axes
    np.testing.assert_array_equal(
        reopened.pyramid.levels[0].voxel_to_world, ref.levels[0].voxel_to_world
    )
    assert asyncio.run(reopened.read(0, Region((0, 63), (1, 65)))).tolist() == [
        [17, 34]
    ]
    assert (
        asyncio.run(reopened.read(0, Region((80_000, 100_000), (80_001, 100_001))))[
            0, 0
        ]
        == 0
    )
    with pytest.raises(ValueError, match="shape and dtype"):
        labels.write_chunk((0, 0), np.zeros((64, 64), dtype="int32"))
