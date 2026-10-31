"""Camera readback shared by hosts that render through a VisPy scene."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["vispy_world_to_clip"]


def vispy_world_to_clip(view: Any, ndim: int) -> np.ndarray:
    """Return a data-order world-to-clip matrix from a VisPy ``ViewBox``.

    The scene transform maps XYZ scene coordinates to logical pixels local to
    the view box, so clip space spans that box even when it shares its canvas
    with other views. The result consumes YX or ZYX coordinates and keeps
    homogeneous coordinates intact, so perspective cameras stay correct away
    from the data origin.
    """
    if ndim not in (2, 3):
        raise ValueError("VisPy scenes display two or three axes")
    width, height = view.size
    if width <= 0 or height <= 0:
        raise ValueError("the view box size must be positive")
    points = np.zeros((ndim + 1, 4), dtype=np.float64)
    points[:, 3] = 1.0
    for axis in range(ndim):
        points[axis + 1, ndim - axis - 1] = 1.0
    mapped = np.asarray(view.scene.transform.map(points), dtype=np.float64)

    local = np.eye(4, dtype=np.float64)
    local[:, -1] = mapped[0]
    for axis in range(ndim):
        local[:, axis] = mapped[axis + 1] - mapped[0]

    local_to_clip = np.eye(4, dtype=np.float64)
    local_to_clip[0] = (2.0 / width, 0.0, 0.0, -1.0)
    local_to_clip[1] = (0.0, -2.0 / height, 0.0, 1.0)
    return local_to_clip @ local
