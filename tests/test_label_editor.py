import asyncio
import time

import numpy as np
import pytest

from lodstone import Level, Pyramid, Region
from lodstone.labels import LabelEditor
from lodstone.sources import ZarrLabels


def test_brush_undo_erase_and_failed_save(tmp_path, monkeypatch, record_property):
    geometry = Pyramid(
        ("z", "y", "x"),
        (Level((3, 9, 13), np.dtype("uint8"), (1, 4, 5), np.diag([3, 2, 1, 1])),),
    )
    path = tmp_path / "labels.zarr"
    store = ZarrLabels.create(path, geometry)
    editor = LabelEditor(store)
    full = Region((0, 0, 0), geometry.levels[0].shape)
    points = np.array([[7.0, 3.5], [7.0, 8.5]])
    assert editor.stroke(points, 1.1, 9, fixed_indices=(1,)) == 2
    expected = np.zeros(full.shape, dtype=np.uint32)
    expected[1, 3, 2:10] = 9
    np.testing.assert_array_equal(asyncio.run(editor.read(0, full)), expected)

    editor.stroke(points[:1], 0.6, 0, fixed_indices=(1,))
    assert asyncio.run(editor.read(0, full))[1, 3, 3] == 0
    editor.undo()
    np.testing.assert_array_equal(asyncio.run(editor.read(0, full)), expected)
    editor.redo()
    assert asyncio.run(editor.read(0, full))[1, 3, 3] == 0
    editor.undo()

    write = store.write_chunk

    def fail(key, data):
        if key[-1] == 1:
            raise OSError("injected")
        write(key, data)

    with monkeypatch.context() as patch:
        patch.setattr(store, "write_chunk", fail)
        with pytest.raises(OSError, match="injected"):
            editor.save()
        assert editor.dirty_chunks == 1
    editor.save()
    assert editor.dirty_chunks == 0
    np.testing.assert_array_equal(
        asyncio.run(ZarrLabels.open(path).read(0, full)), expected
    )

    start = time.perf_counter()
    for label in range(10, 60):
        editor.stroke(points, 1.1, label, fixed_indices=(1,))
    record_property("50_strokes_ms", 1000 * (time.perf_counter() - start))
    assert editor.dirty_chunks == 2
