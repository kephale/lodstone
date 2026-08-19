from __future__ import annotations

import numpy as np
import pytest

from lodstone import Region, fill_unloaded_chunks, nearest_resample_region


def _transform(scale, translate=(0, 0)):
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0], matrix[1, 1] = scale
    matrix[:2, 2] = translate
    return matrix


def test_nearest_resample_uses_world_scale_and_translation() -> None:
    source = np.arange(16, dtype=np.uint8).reshape(4, 4)
    result = nearest_resample_region(
        source,
        Region((10, 20), (14, 24)),
        _transform((2, 2), (5, 7)),
        Region((20, 40), (28, 48)),
        _transform((1, 1), (5, 7)),
    )

    assert result.shape == (8, 8)
    np.testing.assert_array_equal(result[::2, ::2], source)
    np.testing.assert_array_equal(result[1::2, 1::2], source)


def test_translation_changes_sampling_without_changing_extent() -> None:
    source = np.arange(6, dtype=np.uint8).reshape(2, 3)
    result = nearest_resample_region(
        source,
        Region((0, 0), (2, 3)),
        _transform((1, 1), (0, 1)),
        Region((0, 0), (2, 3)),
        _transform((1, 1)),
    )
    np.testing.assert_array_equal(result[:, 1:], source[:, :-1])


def test_fill_unloaded_chunks_preserves_loaded_detail() -> None:
    destination = np.zeros((6, 8), dtype=np.uint8)
    content = np.full((4, 6), 3, dtype=np.uint8)
    loaded = Region((2, 4), (4, 8))
    destination[2:4, 4:8] = 9

    filled = fill_unloaded_chunks(
        destination,
        Region((0, 0), (6, 8)),
        content,
        Region((1, 1), (5, 7)),
        ((2, 2, 2), (4, 4)),
        {loaded},
    )

    assert loaded not in filled
    assert np.all(destination[2:4, 4:8] == 9)
    assert np.all(destination[1:2, 1:7] == 3)
    assert np.all(destination[4:5, 1:7] == 3)


def test_resampling_rejects_rotated_transform() -> None:
    transform = _transform((1, 1))
    transform[0, 1] = 0.5
    with pytest.raises(ValueError, match="axis aligned"):
        nearest_resample_region(
            np.zeros((2, 2)),
            Region((0, 0), (2, 2)),
            transform,
            Region((0, 0), (2, 2)),
            _transform((1, 1)),
        )
