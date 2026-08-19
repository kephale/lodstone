from __future__ import annotations

import numpy as np

from lodstone import (
    Region,
    anisotropic_extent_for_bytes,
    clamp_region_to_budget,
    isotropic_extent_for_bytes,
)


def test_isotropic_extent_obeys_byte_and_axis_limits() -> None:
    assert isotropic_extent_for_bytes(np.uint8, 64**3, ndim=3) == 64
    assert (
        isotropic_extent_for_bytes(
            np.uint8,
            64**3,
            ndim=3,
            max_axis_extent=32,
        )
        == 32
    )


def test_anisotropic_extent_preserves_short_axis() -> None:
    extent = anisotropic_extent_for_bytes(
        (42, 304, 657),
        8 * 1024**2,
        1,
    )
    assert extent[0] == 42
    assert np.prod(extent) <= 8 * 1024**2


def test_clamp_region_respects_axis_and_total_limits() -> None:
    result = clamp_region_to_budget(
        Region((0, 0, 0), (64, 2174, 512)),
        (64, 2174, 512),
        itemsize=1,
        max_bytes=32 * 1024**2,
        max_axis_extent=2048,
    )
    assert all(size <= 2048 for size in result.shape)
    assert np.prod(result.shape) <= 32 * 1024**2
