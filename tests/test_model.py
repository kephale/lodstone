from __future__ import annotations

import numpy as np
import pytest

import lodstone
from lodstone import Level, Pyramid, Region, View, identity_transform


def test_package_exposes_version() -> None:
    assert lodstone.__version__


def test_region_intersection_and_slices() -> None:
    region = Region((2, 3), (8, 9))
    overlap = region.intersection(Region((0, 7), (5, 12)))
    assert overlap == Region((2, 7), (5, 9))
    assert overlap.shape == (3, 2)
    assert overlap.slices() == (slice(2, 5), slice(7, 9))


def test_level_transform_is_copied_and_read_only() -> None:
    transform = identity_transform(2)
    level = Level((10, 12), np.dtype("u2"), (4, 4), transform)
    transform[0, 0] = 99
    assert level.voxel_to_world[0, 0] == 1
    with pytest.raises(ValueError):
        level.voxel_to_world[0, 0] = 3


def test_level_rectilinear_chunk_grid_bounds() -> None:
    level = Level(
        (7, 9),
        np.dtype("u1"),
        (2, 4),
        identity_transform(2),
        ((2, 3, 2), (4, 1, 4)),
    )

    assert level.chunk_bounds(0, 1) == (2, 5)
    assert level.chunk_bounds(1, 2) == (5, 9)
    assert [level.chunk_index(0, value) for value in range(7)] == [
        0,
        0,
        1,
        1,
        1,
        2,
        2,
    ]


def test_pyramid_rejects_mismatched_dimensions() -> None:
    level = Level((10, 12), np.dtype("u1"), (4, 4), identity_transform(2))
    with pytest.raises(ValueError, match="number of axes"):
        Pyramid(("z", "y", "x"), (level,))


def test_view_requires_exact_hidden_axis_selections() -> None:
    with pytest.raises(ValueError, match="exactly"):
        View((1, 2), (None, None, None), (100, 100), np.eye(4))


def test_regular_level_locates_chunks_arithmetically() -> None:
    # 2**30 chunks per axis: a list of them would not fit in memory
    level = Level((2**35 + 5,), np.uint8, (32,), identity_transform(1))
    last = 2**30
    assert level.chunk_index(0, 2**35 + 4) == last
    assert level.chunk_bounds(0, last) == (2**35, 2**35 + 5)
    assert level.chunk_bounds(0, 3) == (96, 128)
    with pytest.raises(IndexError):
        level.chunk_bounds(0, last + 1)


def test_regular_and_explicit_grids_locate_chunks_alike() -> None:
    regular = Level((10,), np.uint8, (4,), identity_transform(1))
    explicit = Level((10,), np.uint8, (4,), identity_transform(1), ((4, 4, 2),))
    for coordinate in range(10):
        assert regular.chunk_index(0, coordinate) == explicit.chunk_index(0, coordinate)
    for index in range(3):
        assert regular.chunk_bounds(0, index) == explicit.chunk_bounds(0, index)
