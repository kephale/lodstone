"""Pixel-level quality and Pareto evaluation for matched camera replays."""

from __future__ import annotations

import numpy as np


def image_error(
    reference: np.ndarray, image: np.ndarray, *, foreground: float = 0.02
) -> dict[str, float]:
    """Compare fixed-range RGB frames; never normalize individual images."""
    if reference.shape != image.shape or reference.ndim != 3 or reference.shape[2] < 3:
        raise ValueError("frames must have identical HxWxRGB(A) shape")

    def rgb(array):
        if array.dtype == np.uint8:
            return array[..., :3].astype(float) / 255
        result = array[..., :3].astype(float)
        if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
            raise ValueError("float frames must be finite and in [0, 1]")
        return result

    ref, actual = rgb(reference), rgb(image)
    delta = actual - ref
    mask = np.max(ref, axis=-1) > foreground
    edges = []
    for axis in (0, 1):
        if ref.shape[axis] > 1:
            edges.append(
                np.mean((np.diff(actual, axis=axis) - np.diff(ref, axis=axis)) ** 2)
            )
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "foreground_rmse": float(np.sqrt(np.mean(delta[mask] ** 2)))
        if mask.any()
        else 0.0,
        "edge_rmse": float(np.sqrt(np.mean(edges))) if edges else 0.0,
        "brightness_bias": float(np.mean(delta)),
        "foreground_fraction": float(mask.mean()),
    }


def pareto_front(rows: list[dict], objectives: tuple[str, ...]) -> list[str]:
    """Return nondominated IDs, minimizing every named objective (ties retained)."""
    values = np.asarray([[row[key] for key in objectives] for row in rows], dtype=float)
    if not objectives or (values.size and not np.all(np.isfinite(values))):
        raise ValueError("objectives must be nonempty and finite")
    return [
        row["id"]
        for i, row in enumerate(rows)
        if not any(
            np.all(other <= values[i]) and np.any(other < values[i])
            for j, other in enumerate(values)
            if i != j
        )
    ]
