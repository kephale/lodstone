import asyncio

import numpy as np
import pytest

from lodstone import Level, Pyramid, Region
from lodstone.labels import LabelEditor
from lodstone.sources import ZarrLabels


def test_coarse_overlay_matches_rebuild_and_recovers_interrupted_save(
    tmp_path, monkeypatch
):
    transform = np.array([[2, 0, -12], [0, 0.75, 2], [0, 0, 1]])
    pyramid = Pyramid(
        ("y", "x"),
        (
            Level((11, 17), np.dtype("uint32"), (4, 5), transform),
            Level((4, 6), np.dtype("uint32"), (2, 4), transform @ np.diag([3, 3, 1])),
        ),
    )
    path = tmp_path / "labels.zarr"
    store = ZarrLabels.create(path, pyramid)
    editor = LabelEditor(store)
    fine = Region((0, 0), (11, 17))
    coarse = Region((0, 0), (4, 6))

    def expected():
        labels = asyncio.run(editor.read(0, fine))
        return labels[np.ix_([1, 4, 7, 10], [1, 4, 7, 10, 13, 16])]

    def check():
        np.testing.assert_array_equal(asyncio.run(editor.read(1, coarse)), expected())

    points = np.array([[-8.5, 4.1], [9, 15]])
    editor.stroke(points, 2.5, 23)
    assert expected().any()
    check()
    editor.stroke(points[:1], 2.5, 0)
    check()
    editor.undo()
    check()
    editor.save()
    np.testing.assert_array_equal(
        asyncio.run(ZarrLabels.open(path).read(1, coarse)), expected()
    )

    region = store.chunk_region((2, 3))
    changed = np.full(region.shape, 49, dtype=np.uint32)
    repair = store._repair

    def interrupt():
        if (path / ".lodstone-pending-chunk").exists():
            raise OSError("interrupted parent update")
        repair()

    with monkeypatch.context() as patch:
        patch.setattr(store, "_repair", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            store.write_chunk((2, 3), changed)
    reopened = ZarrLabels.open(path)
    assert asyncio.run(reopened.read(1, coarse))[-1, -1] == 49
    assert not (path / ".lodstone-pending-chunk").exists()
    np.testing.assert_array_equal(asyncio.run(reopened.read(1, coarse)), expected())
