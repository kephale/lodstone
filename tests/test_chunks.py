from __future__ import annotations

import numpy as np
import pytest

from lodstone import (
    chunk_boundaries,
    chunk_ids_in_region,
    chunk_key_id,
    chunk_shape_for,
    chunk_sizes_for,
    chunk_slices_for,
    normalize_chunk_sizes,
)


class _RectilinearArray:
    shape = (10, 20)
    dtype = np.dtype("u1")
    read_chunk_sizes = ((2, 3, 5), (10, 10))

    @property
    def chunks(self):
        raise AssertionError("read_chunk_sizes must take precedence")


def test_regular_chunk_sizes_are_clipped() -> None:
    assert normalize_chunk_sizes((10, 9), (4, 5)) == (
        (4, 4, 2),
        (5, 4),
    )


def test_rectilinear_array_metadata_and_boundaries() -> None:
    array = _RectilinearArray()
    assert chunk_sizes_for(array) == ((2, 3, 5), (10, 10))
    assert chunk_shape_for(array) == (5, 10)
    boundaries = chunk_boundaries(array)
    np.testing.assert_array_equal(boundaries[0], [0, 2, 5, 10])
    np.testing.assert_array_equal(boundaries[1], [0, 10, 20])
    assert list(chunk_ids_in_region(boundaries, (1, 12), (7, 18))) == [
        ((0, 2), (10, 20)),
        ((2, 5), (10, 20)),
        ((5, 10), (10, 20)),
    ]


def test_invalid_rectilinear_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        normalize_chunk_sizes((10, 20), ((2, 3), (10, 10)))


def test_chunk_slice_keys_and_ids() -> None:
    array = _RectilinearArray()

    slices = chunk_slices_for(array, ((1, 12), (7, 18)))

    assert slices == (
        (slice(0, 2), slice(2, 5), slice(5, 10)),
        (slice(10, 20),),
    )
    assert chunk_key_id((slices[0][1], slices[1][0])) == (
        (2, 5),
        (10, 20),
    )


def _deep_zarr_level(shards=None):
    zarr = pytest.importorskip("zarr")
    # 2**22 chunks per axis: listing them is what made deep pyramids slow
    return zarr.create_array(
        {},
        shape=(2**27, 2**27, 2**27),
        chunks=(32, 32, 32),
        shards=shards,
        dtype="u1",
    )


def test_regular_zarr_grid_is_described_without_listing_chunks(
    monkeypatch,
) -> None:
    array = _deep_zarr_level()
    zarr_array = type(array)

    def listed(self):
        raise AssertionError("a regular grid must not be listed chunk by chunk")

    # older Zarr has no read_chunk_sizes; add the trap either way
    monkeypatch.setattr(zarr_array, "read_chunk_sizes", property(listed), raising=False)

    from lodstone.chunks import regular_chunk_shape

    assert regular_chunk_shape(array) == (32, 32, 32)
    assert chunk_shape_for(array) == (32, 32, 32)
    sizes = chunk_sizes_for(array)
    assert len(sizes[0]) == 2**22 and set(sizes[0]) == {32}
    boundaries = chunk_boundaries(array)
    assert boundaries[0][1] == 32 and boundaries[0][-1] == 2**27


def test_sharded_regular_zarr_grid_uses_the_inner_chunks() -> None:
    array = _deep_zarr_level(shards=(64, 64, 64))
    assert chunk_shape_for(array) == (32, 32, 32)


def test_regular_boundaries_clip_the_last_chunk() -> None:
    class _Regular:
        shape = (10, 8)
        chunks = (4, 8)

    np.testing.assert_array_equal(chunk_boundaries(_Regular())[0], [0, 4, 8, 10])
    np.testing.assert_array_equal(chunk_boundaries(_Regular())[1], [0, 8])
    assert chunk_shape_for(_Regular()) == (4, 8)


def test_array_source_keeps_regular_levels_compact() -> None:
    from lodstone.sources import ArrayPyramidSource

    array = _deep_zarr_level()
    implicit = ArrayPyramidSource([array])
    # an explicit grid equal to the array's own is recognized as regular
    explicit = ArrayPyramidSource([array], chunks=[chunk_sizes_for(array)])
    shape_only = ArrayPyramidSource([array], chunks=[(32, 32, 32)])

    for source in (implicit, explicit, shape_only):
        level = source.pyramid.levels[0]
        assert level.chunk_grid is None
        assert level.chunks == (32, 32, 32)
        assert level.chunk_bounds(0, 2**22 - 1) == (2**27 - 32, 2**27)


def test_array_source_keeps_a_different_explicit_grid() -> None:
    from lodstone.sources import ArrayPyramidSource

    zarr = pytest.importorskip("zarr")
    array = zarr.create_array({}, shape=(10,), chunks=(5,), dtype="u1")
    source = ArrayPyramidSource([array], chunks=[((2, 3, 5),)])

    assert source.pyramid.levels[0].chunk_grid == ((2, 3, 5),)


def test_regular_axes_are_recognized_without_reading_them() -> None:
    from lodstone.chunks import axis_chunk_boundaries
    from lodstone.geometry import expand_region_to_chunk_grid, native_chunks_in_region
    from lodstone.model import Region

    class _Deep:
        shape = (2**27, 2**27, 2**27)
        chunks = (32, 32, 32)

    grid = chunk_sizes_for(_Deep())
    # handing the grid back (as VirtualData.read_chunk_sizes does) is O(1)
    assert normalize_chunk_sizes(_Deep.shape, grid) is not None
    assert normalize_chunk_sizes(_Deep.shape, list(grid))[0] is grid[0]
    bounds = axis_chunk_boundaries(grid[0])
    assert bounds[1] == 32 and bounds[-1] == 2**27

    region = Region((40, 70, 2**27 - 5), (100, 71, 2**27))
    assert native_chunks_in_region(region, _Deep.shape, grid) == 3 * 1 * 1
    expanded = expand_region_to_chunk_grid(
        region, _Deep.shape, grid, itemsize=1, max_bytes=2**20
    )
    assert expanded == Region((32, 64, 2**27 - 32), (128, 96, 2**27))


def test_explicit_axes_get_cached_boundaries() -> None:
    from lodstone.chunks import axis_chunk_boundaries

    sizes = (2, 3, 5)
    first = axis_chunk_boundaries(sizes)
    np.testing.assert_array_equal(first, [0, 2, 5, 10])
    assert axis_chunk_boundaries(sizes) is first
    with pytest.raises(ValueError, match="positive"):
        axis_chunk_boundaries((2, 0))


def test_fill_unloaded_chunks_accepts_precomputed_boundaries() -> None:
    from lodstone.composition import fill_unloaded_chunks
    from lodstone.model import Region

    region = Region((0, 0), (8, 8))
    grid = ((4, 4), (4, 4))
    loaded = [Region((0, 0), (4, 4))]
    content = np.full((8, 8), 7, dtype=np.uint8)

    plain = np.zeros((8, 8), dtype=np.uint8)
    fast = np.zeros((8, 8), dtype=np.uint8)
    expected = fill_unloaded_chunks(plain, region, content, region, grid, loaded)
    got = fill_unloaded_chunks(
        fast,
        region,
        content,
        region,
        grid,
        loaded,
        boundaries=(np.array([0, 4, 8]), np.array([0, 4, 8])),
    )

    assert got == expected
    np.testing.assert_array_equal(fast, plain)
    assert np.all(fast[:4, :4] == 0) and np.all(fast[4:, :] == 7)
