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


def test_fill_unloaded_chunks_preserves_tiles_smaller_than_native_chunks() -> None:
    # Dense blocks are often smaller than native chunks (Zebrahub: 64x128x128
    # blocks in 128x362x362 chunks). Recomposing a reused window must not
    # overwrite loaded blocks merely because no single block covers a chunk.
    destination = np.zeros((4, 8), dtype=np.uint8)
    destination[:, :4] = 9  # two loaded blocks inside the first 4x8 chunk
    loaded = {Region((0, 0), (2, 4)), Region((2, 0), (4, 4))}
    content = np.full((4, 8), 3, dtype=np.uint8)

    filled = fill_unloaded_chunks(
        destination,
        Region((0, 0), (4, 8)),
        content,
        Region((0, 0), (4, 8)),
        ((4,), (8,)),
        loaded,
    )

    assert np.all(destination[:, :4] == 9)
    assert np.all(destination[:, 4:] == 3)
    assert filled == (Region((0, 0), (4, 8)),)


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


def _reference_nearest_resample(
    source, source_region, source_transform, region, transform
):
    """The direct fancy-index gather the fast path must reproduce."""
    ndim = source_region.ndim
    indices = []
    for axis in range(ndim):
        coordinates = np.arange(region.start[axis], region.stop[axis], dtype=np.float64)
        world = (coordinates + 0.5) * transform[axis, axis] + transform[axis, ndim]
        source_coordinates = np.floor(
            (world - source_transform[axis, ndim]) / source_transform[axis, axis]
        ).astype(np.int64)
        relative = source_coordinates - source_region.start[axis]
        indices.append(np.clip(relative, 0, source.shape[axis] - 1))
    return source[np.ix_(*indices)]


def _random_axis_aligned(rng, ndim):
    matrix = np.eye(ndim + 1)
    matrix[:ndim, :ndim] = np.diag(rng.uniform(0.3, 5.0, ndim))
    matrix[:ndim, ndim] = rng.uniform(-3.0, 3.0, ndim)
    return matrix


def test_nearest_resample_matches_direct_gather() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        ndim = int(rng.integers(1, 5))
        shape = tuple(int(n) for n in rng.integers(1, 20, ndim))
        source = rng.integers(0, 255, shape).astype(np.uint16)

        source_transform, transform_ = (
            _random_axis_aligned(rng, ndim),
            _random_axis_aligned(rng, ndim),
        )
        source_start = tuple(int(n) for n in rng.integers(0, 5, ndim))
        source_region = Region(source_start, tuple(np.add(source_start, shape)))
        start = tuple(int(n) for n in rng.integers(0, 10, ndim))
        region = Region(start, tuple(np.add(start, rng.integers(1, 30, ndim))))

        args = (source, source_region, source_transform, region, transform_)
        np.testing.assert_array_equal(
            nearest_resample_region(*args), _reference_nearest_resample(*args)
        )


def test_nearest_resample_never_aliases_the_source() -> None:
    source = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    identity = np.eye(4)
    result = nearest_resample_region(
        source,
        Region((0, 0, 0), (4, 4, 4)),
        identity,
        Region((1, 0, 0), (3, 4, 4)),
        identity,
    )
    np.testing.assert_array_equal(result, source[1:3])
    result[...] = 0
    assert source[1:3].any()
