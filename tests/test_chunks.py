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
