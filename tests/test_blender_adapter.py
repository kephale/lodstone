"""Blender adapter checks that do not need ``bpy``.

The adapter's Blender-side behavior (exact grid values, physical placement,
streaming and rendering) is checked inside Blender by
``examples/blender_native_grid_check.py``.
"""

from __future__ import annotations

import numpy as np

from lodstone.adapters.blender import MainThreadQueue, blender_viewport_snapshot


class _Matrix:
    """Minimal stand-in for ``mathutils.Matrix`` (row iteration + inverse)."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype=np.float64)

    def __iter__(self):
        return iter(self.array.tolist())

    def inverted(self) -> _Matrix:
        return _Matrix(np.linalg.inv(self.array))


class _Region:
    width, height = 640, 480


class _Region3D:
    def __init__(self, view: np.ndarray, projection: np.ndarray, persp: bool):
        self.view_matrix = _Matrix(view)
        self.perspective_matrix = _Matrix(projection @ view)
        self.is_perspective = persp


class _Root:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix_world = _Matrix(matrix)


def _perspective(near: float = 0.1, far: float = 100.0) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def test_viewport_snapshot_maps_data_zyx_through_root_to_clip() -> None:
    root = np.diag([0.5, 0.5, 0.5, 1.0])
    root[:3, 3] = (1.0, 2.0, 3.0)  # Blender world location of the data origin
    view = np.eye(4)
    view[:3, 3] = (0.0, 0.0, -10.0)  # camera at +10 Z looking down -Z
    projection = _perspective()
    snapshot = blender_viewport_snapshot(
        _Region(),
        _Region3D(view, projection, persp=True),
        _Root(root),
        displayed_axes=(2, 3, 4),
        index=(7, 0, None, None, None),
    )

    data_zyx = np.array([6.0, 4.0, 2.0, 1.0])  # z, y, x in physical units
    local_xyz = np.array([2.0, 4.0, 6.0, 1.0])
    expected = projection @ view @ root @ local_xyz
    np.testing.assert_allclose(snapshot.world_to_clip @ data_zyx, expected)
    assert snapshot.viewport == (640, 480)
    assert snapshot.index == (7, 0, None, None, None)
    # Camera at world (0, 0, 10) -> root-local ((0-1)/0.5, (0-2)/0.5, (10-3)/0.5).
    assert snapshot.eye is not None
    np.testing.assert_allclose(snapshot.eye, (14.0, -4.0, -2.0))


def test_viewport_snapshot_has_no_eye_for_orthographic_views() -> None:
    snapshot = blender_viewport_snapshot(
        _Region(),
        _Region3D(np.eye(4), np.eye(4), persp=False),
        _Root(np.eye(4)),
        displayed_axes=(0, 1, 2),
        index=(None, None, None),
    )
    assert snapshot.eye is None


def test_main_thread_queue_runs_in_order_and_keeps_errors() -> None:
    queue = MainThreadQueue(budget_seconds=1.0)
    seen: list[int] = []
    queue(lambda: seen.append(1))

    def fail() -> None:
        raise ZeroDivisionError

    queue(fail)
    queue(lambda: seen.append(2))
    assert queue.drain() == 3
    assert seen == [1, 2]
    assert len(queue.errors) == 1
    assert isinstance(queue.errors[0], ZeroDivisionError)
    assert queue.drain() == 0
