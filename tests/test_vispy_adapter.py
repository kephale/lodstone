from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lodstone.adapters import vispy_world_to_clip


def _view(matrix: np.ndarray, size: tuple[int, int]) -> SimpleNamespace:
    def map_points(points: np.ndarray) -> np.ndarray:
        return points @ matrix.T

    scene = SimpleNamespace(transform=SimpleNamespace(map=map_points))
    return SimpleNamespace(scene=scene, size=size)


def test_reads_a_panzoom_scene_in_data_order() -> None:
    # Scene XY maps to pixels with Y down; the view box is 200 wide, 100 tall.
    pixels = np.eye(4)
    pixels[:2, :2] = np.diag((2.0, -2.0))
    pixels[:2, 3] = (100.0, 50.0)

    matrix = vispy_world_to_clip(_view(pixels, (200, 100)), 2)

    # Data point (y=5, x=10) is scene (10, 5).
    np.testing.assert_allclose(
        matrix @ np.array((5.0, 10.0, 0.0, 1.0)), (0.2, 0.2, 0.0, 1.0)
    )


def test_keeps_homogeneous_coordinates() -> None:
    pixels = np.eye(4)
    pixels[3, 2] = 0.5

    matrix = vispy_world_to_clip(_view(pixels, (2, 2)), 3)

    # Data Z is scene Z, which drives w.
    assert matrix[3, 0] == pytest.approx(0.5)


def test_rejects_unsupported_dimensionality() -> None:
    with pytest.raises(ValueError, match="two or three"):
        vispy_world_to_clip(_view(np.eye(4), (2, 2)), 4)


def test_rejects_an_empty_view_box() -> None:
    with pytest.raises(ValueError, match="positive"):
        vispy_world_to_clip(_view(np.eye(4), (0, 2)), 2)
