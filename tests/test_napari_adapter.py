from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lodstone.adapters import napari as adapter
from lodstone.adapters.napari import NapariController, _SlicedArray, add_lodstone_image
from lodstone.sources import ArrayPyramidSource


def source_4d() -> ArrayPyramidSource:
    fine = np.arange(2 * 4 * 6 * 8, dtype=np.uint16).reshape(2, 4, 6, 8)
    coarse = fine[:, ::2, ::2, ::2]
    fine_transform = np.diag([1.0, 5.0, 2.0, 3.0, 1.0])
    coarse_transform = np.diag([1.0, 10.0, 4.0, 6.0, 1.0])
    return ArrayPyramidSource(
        [fine, coarse],
        axes=("c", "z", "y", "x"),
        transforms=[fine_transform, coarse_transform],
        chunks=[(1, 2, 3, 4), (1, 1, 3, 4)],
    )


def test_array_source_exposes_original_lazy_levels() -> None:
    source = source_4d()

    assert len(source.arrays) == 2
    assert source.arrays[0].shape == (2, 4, 6, 8)


def test_sliced_array_removes_fixed_axes_without_materializing() -> None:
    source = source_4d()
    view = _SlicedArray(
        source.arrays[0],
        {0: 1, 2: 3},
        source.pyramid.levels[0].chunks,
    )

    assert view.shape == (4, 8)
    assert view.chunks == (2, 4)
    np.testing.assert_array_equal(
        view[1:3, 2:6],
        source.arrays[0][1, 1:3, 3, 2:6],
    )


def test_add_image_delegates_to_progressive_napari_factory(monkeypatch) -> None:
    source = source_4d()
    loader = SimpleNamespace(close=lambda: None)
    layer = SimpleNamespace(metadata={"progressive_loader": loader})
    captured = {}

    def factory(arrays, **kwargs):
        captured["arrays"] = arrays
        captured["kwargs"] = kwargs
        return layer

    monkeypatch.setattr(adapter, "_progressive_image_factory", lambda: factory)
    viewer = object()
    result = add_lodstone_image(
        source,
        viewer,
        fixed_index={0: 1},
        name="DAPI",
        colormap="blue",
    )

    assert result is layer
    assert [array.shape for array in captured["arrays"]] == [
        (4, 6, 8),
        (2, 3, 4),
    ]
    assert captured["kwargs"]["viewer"] is viewer
    assert captured["kwargs"]["name"] == "DAPI"
    np.testing.assert_allclose(
        captured["kwargs"]["affine"],
        np.diag([5.0, 2.0, 3.0, 1.0]),
    )


def test_controller_owns_progressive_loader_lifecycle(monkeypatch) -> None:
    source = source_4d()
    closed = []
    loader = SimpleNamespace(close=lambda: closed.append(True))
    layer = SimpleNamespace(metadata={"progressive_loader": loader})
    monkeypatch.setattr(adapter, "add_lodstone_image", lambda *args, **kwargs: layer)

    controller = NapariController(object(), source, fixed_index={0: 0})
    controller.close()

    assert controller.layer is layer
    assert closed == [True]


def test_generic_source_without_arrays_is_rejected() -> None:
    source = SimpleNamespace(pyramid=source_4d().pyramid, read=lambda *_args: None)

    with pytest.raises(TypeError, match="lazy array levels"):
        add_lodstone_image(source)
