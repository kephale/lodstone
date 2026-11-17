"""Consumer checks across a curved surface split among sixteen face owners."""

import asyncio
from collections import Counter

import numpy as np
import pytest


def oriented_triangles(vertices, faces):
    # Ignore cyclic starting corner, but retain winding and multiplicity.
    result = Counter()
    for triangle in vertices[faces]:
        corners = tuple(map(tuple, triangle.tolist()))
        result[min(corners[i:] + corners[:i] for i in range(3))] += 1
    return result


def test_sixteen_owner_tiles_preserve_surface_and_shared_edges(tmp_path):
    building = pytest.importorskip("zarr_vectors.building")
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    vertices = np.array(
        [
            (1 + 10 * x, 1 + 10 * y, 1 + (x * x + y * y) / 16)
            for x in range(5)
            for y in range(5)
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            triangle
            for x in range(4)
            for y in range(4)
            for triangle in (
                (5 * x + y, 5 * (x + 1) + y, 5 * (x + 1) + y + 1),
                (5 * x + y, 5 * (x + 1) + y + 1, 5 * x + y + 1),
            )
        ]
    )
    path = tmp_path / "curved-grid.zv"
    building.write_mesh(path, vertices, faces, chunk_shape=(10.0, 10.0, 10.0))
    source = ZarrVectorsMeshSource(path, level_errors=[0.0])
    try:
        keys = [chunk.key for chunk in source.levels[0].chunks]
        assert set(keys) == {(x, y, 0) for x in range(4) for y in range(4)}
        observed = Counter()
        edges = Counter()
        for key in keys:
            data = asyncio.run(source.read(0, key))
            assert len(data.faces) == 2
            observed.update(oriented_triangles(data.vertices, data.faces))
            for triangle in data.vertices[data.faces]:
                corners = list(map(tuple, triangle.tolist()))
                for i in range(3):
                    edges[tuple(sorted((corners[i], corners[(i + 1) % 3])))] += 1
        assert observed == oriented_triangles(vertices, faces)
        # A 4x4 triangulated sheet has 16 outer edges; every internal edge
        # (including all tile seams) must have exactly two incident faces.
        assert Counter(edges.values()) == {1: 16, 2: 40}
    finally:
        source.close()
