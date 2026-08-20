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
