from __future__ import annotations

import time

import numpy as np

from lodstone import View
from lodstone.adapters.ndv import NDVController
from lodstone.testing import SimulatedSource


class Handle:
    def __init__(self) -> None:
        self.history = []
        self.removed = False
        self.visible = True
        self.scales = None
        self.origins = None

    def set_data(self, data) -> None:
        self.history.append(np.asarray(data))

    def set_world_transform(self, scales, origins) -> None:
        self.scales = scales
        self.origins = origins

    def set_visible(self, visible) -> None:
        self.visible = visible

    def remove(self) -> None:
        self.removed = True


class Canvas:
    def __init__(self) -> None:
        self.handle = None
        self.handles = []
        self.kinds = []
        self.ndim = None
        self.scales = None
        self.origins = None
        self.range_resets = 0
        self.refreshes = 0

    def set_ndim(self, ndim) -> None:
        self.ndim = ndim

    def add_image(self, data=None, *, reset_range=True):
        self.kinds.append("image")
        self.handle = Handle()
        self.handles.append(self.handle)
        if data is not None:
            self.handle.set_data(data)
        return self.handle

    def add_volume(self, data=None, *, reset_range=True):
        self.kinds.append("volume")
        self.handle = Handle()
        self.handles.append(self.handle)
        if data is not None:
            self.handle.set_data(data)
        return self.handle

    def set_scales(self, scales, *, reset_range=True) -> None:
        self.scales = scales

    def set_origins(self, origins) -> None:
        self.origins = origins

    def set_range(self) -> None:
        self.range_resets += 1

    def refresh(self) -> None:
        self.refreshes += 1


class Event:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback

    def disconnect(self, callback) -> None:
        if self.callback == callback:
            self.callback = None

    def emit(self) -> None:
        if self.callback is not None:
            self.callback()


class CameraCanvas(Canvas):
    def __init__(self) -> None:
        super().__init__()
        self.cameraChanged = Event()
        self.world_to_clip = np.eye(4)

    def camera_state(self):
        return (64, 64), self.world_to_clip.copy()


def test_ndv_target_presents_dense_camera_phase(ortho_view, wait) -> None:
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

        assert canvas.kinds == ["image", "image"]
        assert canvas.ndim == 2
        assert len(canvas.handle.history) == 1
        np.testing.assert_array_equal(canvas.handle.history[0], fine)
        assert canvas.handle.scales == (1.0, 1.0)
        assert canvas.handle.origins == (0.0, 0.0)
        assert canvas.range_resets == 1
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
        assert canvas.handle.scales == (2.0, 3.0, 4.0)
        assert canvas.handle.origins == (0.0, 0.0, 0.0)
    finally:
        controller.close()


def test_ndv_target_places_translated_dense_window(wait) -> None:
    data = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    source = SimulatedSource([data], chunks=[(8, 8)])
    canvas = Canvas()
    presented = []
    controller = NDVController(
        canvas,
        source,
        memory_limit=512,
        block_shape=(8, 8),
        on_presented=presented.append,
    )
    world_to_clip = np.eye(4)
    world_to_clip[:2, -1] = -15
    view = View(
        displayed_axes=(0, 1),
        index=(None, None),
        viewport=(64, 64),
        world_to_clip=world_to_clip,
    )
    try:
        controller.update(view)
        wait(lambda: controller.stream.status.state == "complete")

        assert canvas.handle.origins == (8.0, 8.0)
        assert canvas.handle.history[-1].shape == (16, 16)
        assert presented[-1].region.start == (8, 8)
    finally:
        controller.close()


def test_ndv_target_keeps_masked_coarse_context_around_fine_focus(
    ortho_view, wait
) -> None:
    fine = np.full((32, 32), 7, dtype=np.uint16)
    coarse = np.full((8, 8), 3, dtype=np.uint16)
    source = SimulatedSource(
        [fine, coarse],
        transforms=[np.eye(3), np.diag([4.0, 4.0, 1.0])],
        chunks=[(8, 8), (4, 4)],
    )
    canvas = Canvas()
    controller = NDVController(
        canvas,
        source,
        memory_limit=512,
        block_shape=(8, 8),
    )
    try:
        controller.update(ortho_view(fine.shape, viewport=(128, 128)))
        wait(lambda: controller.stream.status.state == "complete")

        assert canvas.kinds == ["image", "image"]
        context, focus = canvas.handles
        assert context.scales == (4.0, 4.0)
        assert focus.scales == (1.0, 1.0)
        assert np.count_nonzero(context.history[-1]) < context.history[-1].size
        assert np.count_nonzero(context.history[-1]) > 0
        assert np.all(focus.history[-1] == 7)
        assert canvas.range_resets == 1
    finally:
        controller.close()


def test_ndv_camera_updates_are_debounced_off_interaction_thread(ortho_view, wait):
    data = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    source = SimulatedSource([data], chunks=[(8, 8)])
    canvas = CameraCanvas()
    targets = []
    controller = NDVController(
        canvas,
        source,
        memory_limit=512,
        block_shape=(8, 8),
        camera_debounce_ms=30,
        on_targeted=lambda plan: targets.append(plan.target_level),
    )
    try:
        controller.update(ortho_view(data.shape, viewport=(64, 64)))
        wait(lambda: controller.stream.status.state == "complete")
        targets.clear()

        for offset in range(10):
            canvas.world_to_clip[:2, -1] = -offset
            canvas.cameraChanged.emit()

        # Camera callbacks only capture the newest view; planning happens after
        # the trailing debounce on the dedicated camera worker.
        assert targets == []
        wait(lambda: len(targets) == 1)
        assert targets == [0]
        canvas.cameraChanged.emit()
        time.sleep(0.06)
        assert targets == [0]
    finally:
        controller.close()
