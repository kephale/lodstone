from __future__ import annotations

import numpy as np
import pytest

from lodstone import (
    Region,
    anisotropic_extent_for_bytes,
    clamp_region_to_budget,
    expand_region_to_chunk_grid,
    isotropic_extent_for_bytes,
    native_chunks_in_region,
    orthographic_world_to_clip,
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


def test_rectilinear_grid_expansion_rebalances_at_dataset_edge() -> None:
    chunks = ((128, 128, 77), (362, 362, 362, 362, 362, 362, 258), (362,) * 14 + (186,))
    region = Region((250, 2050, 5000), (330, 2420, 5250))

    expanded = expand_region_to_chunk_grid(
        region,
        (333, 2430, 5254),
        chunks,
        itemsize=2,
        max_bytes=512 * 1024**2,
        max_axis_extent=2048,
    )

    assert expanded == Region((128, 1810, 4706), (333, 2430, 5254))
    assert all(
        a <= b < c <= limit
        for a, b, c, limit in zip(
            expanded.start, region.start, region.stop, expanded.stop, strict=True
        )
    )


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (Region((0, 0, 0), (128, 724, 724)), 4),
        (Region((0, 0, 0), (256, 724, 724)), 8),
    ],
)
def test_exact_large_native_chunk_count(region: Region, expected: int) -> None:
    chunks = ((128, 128, 77), (362,) * 6 + (258,), (362,) * 14 + (186,))
    assert native_chunks_in_region(region, (333, 2430, 5254), chunks) == expected


def test_grid_expansion_preserves_budget_when_native_union_is_too_large() -> None:
    region = Region((32, 32, 32), (96, 96, 96))
    assert (
        expand_region_to_chunk_grid(
            region,
            (256, 256, 256),
            ((128, 128),) * 3,
            itemsize=2,
            max_bytes=64**3 * 2,
        )
        == region
    )


def test_orthographic_world_to_clip_maps_frame_to_clip_edges() -> None:
    matrix = orthographic_world_to_clip(
        (5, 10, 20),
        (0, 0, 2),
        (0, -3, 0),
        (-4, 0, 0),
        height=100,
        viewport=(800, 400),
        depth=40,
    )

    def clip(*point: float) -> np.ndarray:
        return (matrix @ np.array((*point, 1.0)))[:3]

    np.testing.assert_allclose(clip(5, 10, 20), (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(clip(5, 10, 120), (1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(clip(5, -40, 20), (0, 1, 0), atol=1e-12)
    # Along forward is farther from the viewer, which is clip Z +1.
    np.testing.assert_allclose(clip(-15, 10, 20), (0, 0, 1), atol=1e-12)
    np.testing.assert_allclose(clip(25, 10, 20), (0, 0, -1), atol=1e-12)


@pytest.mark.parametrize(
    ("right", "up", "forward", "message"),
    [
        ((1, 0, 0), (1, 1, 0), (0, 0, 1), "perpendicular"),
        ((1, 0, 0), (0, 1, 0), (1, 0, 1), "perpendicular"),
        ((0, 0, 0), (0, 1, 0), (0, 0, 1), "nonzero"),
        ((1, 0, 0), (0, 1, 0), (0, 0, float("nan")), "nonzero"),
    ],
)
def test_orthographic_world_to_clip_rejects_a_bad_frame(
    right, up, forward, message
) -> None:
    with pytest.raises(ValueError, match=message):
        orthographic_world_to_clip(
            (0, 0, 0), right, up, forward, height=1, viewport=(1, 1), depth=1
        )
