from __future__ import annotations

import numpy as np

from lodstone.adapters.ndv import NDVController
from lodstone.testing import SimulatedSource


class Handle:
    def __init__(self, data) -> None:
        self.history = [np.asarray(data)]
        self.removed = False

    def set_data(self, data) -> None:
        self.history.append(np.asarray(data))

    def remove(self) -> None:
        self.removed = True


class Canvas:
    def __init__(self) -> None:
        self.handle = None
        self.kinds = []
        self.ndim = None
        self.scales = None
        self.refreshes = 0

    def set_ndim(self, ndim) -> None:
        self.ndim = ndim

    def add_image(self, data):
        self.kinds.append("image")
        self.handle = Handle(data)
        return self.handle

    def add_volume(self, data):
        self.kinds.append("volume")
        self.handle = Handle(data)
        return self.handle

    def set_scales(self, scales) -> None:
        self.scales = scales

    def refresh(self) -> None:
        self.refreshes += 1


def test_ndv_target_presents_dense_progressive_phases(ortho_view, wait) -> None:
    fine = np.arange(64, dtype=np.uint16).reshape(8, 8)
    coarse = fine[::2, ::2]
    source = SimulatedSource(
        [fine, coarse],
        transforms=[np.eye(3), np.diag([2.0, 2.0, 1.0])],
        chunks=[(4, 4), (4, 4)],
    )
    canvas = Canvas()
    controller = NDVController(canvas, source)
    try:
        controller.update(ortho_view(fine.shape, viewport=(64, 64)))
        wait(lambda: controller.stream.status.state == "complete")

        assert canvas.kinds == ["image"]
        assert canvas.ndim == 2
        assert len(canvas.handle.history) == 2
        np.testing.assert_array_equal(canvas.handle.history[0], coarse)
        np.testing.assert_array_equal(canvas.handle.history[1], fine)
        assert canvas.scales == (1.0, 1.0)
    finally:
        handle = canvas.handle
        controller.close()

    assert handle.removed


def test_ndv_target_squeezes_hidden_axes_for_volume(ortho_view, wait) -> None:
    data = np.arange(2 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 4, 5, 6)
    transform = np.diag([1.0, 2.0, 3.0, 4.0, 1.0])
    source = SimulatedSource(
        [data],
        transforms=[transform],
        chunks=[(1, 4, 5, 6)],
    )
    canvas = Canvas()
    controller = NDVController(canvas, source)
    view = ortho_view(
        data.shape,
        displayed_axes=(1, 2, 3),
        index=(1, None, None, None),
        viewport=(64, 64),
    )
    try:
        controller.update(view)
        wait(lambda: controller.stream.status.state == "complete")

        assert canvas.kinds == ["volume"]
        assert canvas.ndim == 3
        np.testing.assert_array_equal(canvas.handle.history[-1], data[1])
        assert canvas.scales == (2.0, 3.0, 4.0)
    finally:
        controller.close()
