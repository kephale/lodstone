from __future__ import annotations

import numpy as np
import pytest

scenex = pytest.importorskip("scenex")

from lodstone.adapters.scenex import SceneXController, scenex_view_snapshot
from lodstone.testing import SimulatedSource


def _view(width: int = 80, height: int = 40):
    canvas = scenex.Canvas(width=width, height=height)
    view = scenex.View()
    view.canvas = canvas
    return view


def test_scenex_snapshot_converts_matrix_and_array_axis_order() -> None:
    view = _view()
    view.camera.transform = scenex.Transform().translated((5, 10, 0))
    from scenex.utils.projections import (  # pyright: ignore[reportMissingImports]
        orthographic,
    )

    view.camera.projection = orthographic(20, 40, 2)

    snapshot = scenex_view_snapshot(view, 2)

    assert snapshot.displayed_axes == (0, 1)
    assert snapshot.index == (None, None)
    assert snapshot.viewport == (80, 40)
    assert snapshot.eye is None
    # Lodstone coordinates are array ordered (y, x), while SceneX is (x, y).
    np.testing.assert_allclose(
        snapshot.world_to_clip @ np.array([10, 5, 0, 1]),
        np.array([0, 0, 0, 1]),
        atol=1e-15,
    )


def test_scenex_snapshot_converts_three_dimensional_eye() -> None:
    from scenex.utils.projections import (  # pyright: ignore[reportMissingImports]
        orthographic,
    )

    view = _view()
    view.camera.transform = scenex.Transform().translated((5, 10, 15))
    view.camera.projection = orthographic(20, 40, 60)

    snapshot = scenex_view_snapshot(view, 3)

    assert snapshot.eye == (15.0, 10.0, 5.0)
    np.testing.assert_allclose(
        snapshot.world_to_clip @ np.array([15, 10, 5, 1]),
        np.array([0, 0, 0, 1]),
        atol=1e-15,
    )


def test_scenex_controller_publishes_and_tracks_camera(wait) -> None:
    fine = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    source = SimulatedSource([fine], chunks=[(8, 8)])
    view = _view(64, 64)
    targets = []
    controller = SceneXController(
        view,
        source,
        dispatch=lambda callback: callback(),
        camera_debounce_ms=10,
        on_targeted=lambda plan: targets.append(plan.target_level),
    )
    try:
        controller.update_from_scene()
        wait(lambda: controller.stream.status.state == "complete")

        visuals = [child for child in view.scene.children if child is not view.camera]
        assert len(visuals) == 1
        np.testing.assert_array_equal(visuals[0].data, fine)
        assert visuals[0].clims == (0.0, float(fine.max()))
        np.testing.assert_allclose(visuals[0].transform.map((0, 0, 0))[:2], (0, 0))

        targets.clear()
        view.camera.transform = view.camera.transform.translated((2, 0, 0))
        wait(lambda: targets == [0])
    finally:
        controller.close()

    assert not [child for child in view.scene.children if child is not view.camera]
