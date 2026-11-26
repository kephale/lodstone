import asyncio

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("zarr.abc", reason="lodstone.datasets requires zarr v3")

from lodstone.datasets import mandelbulb_mesh
from lodstone.datasets.mandelbulb_mesh import (
    MandelbulbMeshSource,
    _de_numpy,
    distance_estimate,
)


@pytest.fixture(scope="module")
def source():
    return MandelbulbMeshSource(4, tile=8)


def test_levels_nest_and_tile_the_cube(source):
    levels = source.levels
    assert len(levels) == 4
    assert [chunk.key for chunk in levels[-1].chunks] == [(0, 0, 0)]
    for level in range(3):
        span = 2 * source.radius / 2 ** (3 - level)
        assert levels[level].error == pytest.approx(span / 8)
        parents = {chunk.key for chunk in levels[level + 1].chunks}
        assert levels[level].chunks
        for chunk in levels[level].chunks:
            assert tuple(v // 2 for v in chunk.key) in parents
            low, high = np.asarray(chunk.bounds)
            np.testing.assert_allclose(high - low, span)
            np.testing.assert_allclose(
                low, -source.radius + np.asarray(chunk.key) * span
            )
    # the shell, not the whole cube
    assert len(levels[0].chunks) < 8**3


def test_chunks_lie_in_their_bounds_and_cover_the_surface(source):
    faces = 0
    for chunk in source.levels[0].chunks:
        data = asyncio.run(source.read(0, chunk.key))
        if not len(data.faces):
            continue
        faces += len(data.faces)
        low, high = np.asarray(chunk.bounds)
        assert np.all(data.vertices >= low - 1e-6)
        assert np.all(data.vertices <= high + 1e-6)
        assert data.faces.max() < len(data.vertices)
        # vertices lie on the DE = offset level set, to within a cell
        sample = data.vertices[:: max(1, len(data.vertices) // 50)]
        estimate = distance_estimate(sample, source.maxiter, source.power)
        assert np.all(np.abs(estimate - source.offset) < source.levels[0].error * 2)
    assert faces > 1000


def test_neighbours_share_their_boundary_vertices(source):
    meshes = {
        chunk.key: source.chunk_mesh(0, chunk.key) for chunk in source.levels[0].chunks
    }
    step = source.levels[0].error
    pairs = 0
    for key, data in meshes.items():
        right = (key[0] + 1, key[1], key[2])
        if right not in meshes or not len(data.faces) or not len(meshes[right].faces):
            continue
        # computed as the source places lattice points
        plane = np.float32(-source.radius + (key[0] + 1) * source.tile * step)
        mine = data.vertices[data.vertices[:, 0] == plane]
        theirs = meshes[right].vertices[meshes[right].vertices[:, 0] == plane]
        if not len(mine):
            continue
        assert {tuple(v) for v in mine.tolist()} == {tuple(v) for v in theirs.tolist()}
        pairs += 1
    assert pairs > 0


def test_numpy_fallback_matches(monkeypatch):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.25, 1.25, (2000, 3))
    expected = _de_numpy(points, 12, 8)
    assert (expected == 0).any() and (expected > 0).any()
    np.testing.assert_allclose(
        distance_estimate(points), expected, rtol=1e-6, atol=1e-9
    )

    small = MandelbulbMeshSource(3, tile=6)
    key = small.levels[0].chunks[len(small.levels[0].chunks) // 2].key
    fast = small.chunk_mesh(0, key)
    monkeypatch.setattr(mandelbulb_mesh, "HAS_NUMBA", False)
    slow = small.chunk_mesh(0, key)
    np.testing.assert_array_equal(fast.faces, slow.faces)
    np.testing.assert_allclose(fast.vertices, slow.vertices, atol=1e-5)


def test_rejects_tiny_parameters():
    with pytest.raises(ValueError):
        MandelbulbMeshSource(0)
