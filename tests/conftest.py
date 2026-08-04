from __future__ import annotations

import time

import numpy as np
import pytest

from lodstone import View


def orthographic_view(
    shape: tuple[int, ...],
    *,
    displayed_axes: tuple[int, ...] | None = None,
    index: tuple[int | None, ...] | None = None,
    viewport: tuple[int, int] = (256, 256),
    extent_scale: float = 1.0,
) -> View:
    displayed = displayed_axes or tuple(range(len(shape)))
    if index is None:
        index = tuple(None if axis in displayed else 0 for axis in range(len(shape)))
    matrix = np.eye(4)
    for local_axis, data_axis in enumerate(displayed):
        matrix[local_axis, local_axis] = 2.0 / (shape[data_axis] * extent_scale)
        matrix[local_axis, 3] = -1.0
    return View(displayed, index, viewport, matrix)


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


@pytest.fixture
def ortho_view():
    return orthographic_view


@pytest.fixture
def wait():
    return wait_until
