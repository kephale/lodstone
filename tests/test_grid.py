from __future__ import annotations

import itertools

import numpy as np
import pytest

from lodstone import chunk_boundaries, chunk_ids_in_region, chunk_sizes_for, grid
from lodstone.grid import (
    RegularChunkBoundaries,
    RegularChunkSizes,
    chunk_index_range,
    max_chunk_size,
    regular_axis_boundaries,
    regular_axis_sizes,
    searchsorted,
)

CASES = [(10, 4), (12, 4), (1, 4), (4, 4), (5, 1), (33, 32)]


@pytest.mark.parametrize(("extent", "step"), CASES)
def test_compact_axes_match_the_materialized_ones(extent, step) -> None:
    sizes = RegularChunkSizes(extent, step)
    bounds = RegularChunkBoundaries(extent, step)
    full, remainder = divmod(extent, step)
    expected_sizes = (step,) * full + ((remainder,) if remainder else ())
    expected_bounds = np.concatenate(([0], np.cumsum(expected_sizes)))

    assert tuple(sizes) == expected_sizes
    assert sizes == expected_sizes and sizes == RegularChunkSizes(extent, step)
    assert len(sizes) == len(expected_sizes)
    assert [sizes[i] for i in range(-len(sizes), len(sizes))] == list(
        expected_sizes * 2
    )
    assert sizes[1:3] == expected_sizes[1:3]
    assert max_chunk_size(sizes) == max(expected_sizes)
    np.testing.assert_array_equal(np.asarray(bounds), expected_bounds)
    np.testing.assert_array_equal(bounds[1:-1], expected_bounds[1:-1])
    assert len(bounds) == len(expected_bounds) and bounds[-1] == extent
    for value, side in itertools.product(
        np.arange(-2, extent + 3, 0.5), ("left", "right")
    ):
        assert searchsorted(bounds, value, side) == np.searchsorted(
            expected_bounds, value, side=side
        ), (value, side)
    for start, stop in itertools.product(range(-1, extent + 2), repeat=2):
        assert chunk_index_range(bounds, start, stop) == chunk_index_range(
            expected_bounds, start, stop
        )


def test_out_of_range_entries_raise() -> None:
    sizes = RegularChunkSizes(10, 4)
    bounds = RegularChunkBoundaries(10, 4)
    with pytest.raises(IndexError):
        sizes[3]
    with pytest.raises(IndexError):
        bounds[4]
    with pytest.raises(ValueError, match="positive"):
        RegularChunkSizes(0, 4)


def test_huge_axes_stay_compact() -> None:
    # 2**40 chunks: materialized, 8 TiB per axis
    extent = 2**45
    sizes = regular_axis_sizes(extent, 32)
    bounds = regular_axis_boundaries(extent, 32)
    assert isinstance(sizes, RegularChunkSizes)
    assert isinstance(bounds, RegularChunkBoundaries)
    assert len(sizes) == 2**40 and sizes[-1] == 32
    assert chunk_index_range(bounds, 2**44 + 5, 2**44 + 70) == range(2**39, 2**39 + 3)


def test_typical_axes_stay_materialized(monkeypatch) -> None:
    monkeypatch.setattr(grid, "COMPACT_GRID_MIN_CHUNKS", 1 << 16)
    assert isinstance(regular_axis_sizes(4096, 32), tuple)
    assert isinstance(regular_axis_boundaries(4096, 32), np.ndarray)
    # the threshold is read per call
    monkeypatch.setattr(grid, "COMPACT_GRID_MIN_CHUNKS", 0)
    assert isinstance(regular_axis_sizes(4096, 32), RegularChunkSizes)


def test_compact_grids_answer_chunk_queries(monkeypatch) -> None:
    monkeypatch.setattr(grid, "COMPACT_GRID_MIN_CHUNKS", 2)

    class _Regular:
        shape = (10, 3)
        chunks = (4, 3)

    sizes = chunk_sizes_for(_Regular())
    bounds = chunk_boundaries(_Regular())
    assert isinstance(sizes[0], RegularChunkSizes)
    assert isinstance(sizes[1], tuple)  # one chunk: below the threshold
    assert list(chunk_ids_in_region(bounds, (3, 0), (9, 3))) == [
        ((0, 4), (0, 3)),
        ((4, 8), (0, 3)),
        ((8, 10), (0, 3)),
    ]


def test_levels_locate_chunks_of_compact_explicit_axes() -> None:
    from lodstone import Level, identity_transform

    axis = RegularChunkSizes(2**40 + 3, 32)
    level = Level((2**40 + 3,), np.uint8, (32,), identity_transform(1), (axis,))
    assert level.chunk_grid == (axis,)
    assert level.chunk_sizes(0) is axis
    assert level.chunk_index(0, 2**40 + 2) == 2**35
    assert level.chunk_bounds(0, 2**35) == (2**40, 2**40 + 3)
    with pytest.raises(ValueError, match="exactly cover"):
        Level((10,), np.uint8, (4,), identity_transform(1), (RegularChunkSizes(12, 4),))
