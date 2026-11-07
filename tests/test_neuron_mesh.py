"""Meaningful geometry controls for the public neuron example."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from lodstone.neuron_mesh import read_example_ply, smooth_surface

pytest.importorskip("scipy")


def ply_triangle():
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"comment https://github.com/mikedh/trimesh\n"
        b"element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="<f4")
    face = np.array([(3, [2, 0, 1])], dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    return header + vertices.tobytes() + face.tobytes(), vertices


def test_source_ply_preserves_coordinates_and_face_winding():
    payload, expected = ply_triangle()
    vertices, faces = read_example_ply(payload)
    np.testing.assert_array_equal(vertices, expected)
    np.testing.assert_array_equal(faces, [[2, 0, 1]])


def test_source_ply_rejects_a_different_schema_and_truncation():
    payload, _ = ply_triangle()
    for broken in (
        payload[:-1],
        payload.replace(b"property float x", b"property double x"),
    ):
        with pytest.raises(ValueError):
            read_example_ply(broken)


def test_smoothing_preserves_tiny_component_instead_of_collapsing_it():
    pytest.importorskip("skimage")
    from lodstone.biological_mesh import label_surface, topology

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    v, f = label_surface(mask, np.ones(3), np.zeros(3))
    faces_before = f.copy()
    smoothed, change = smooth_surface(v, f, max_displacement_um=0.8)
    assert abs(change["relative_volume_change"]) <= 0.05 + 1e-12
    assert change["component_volume_backoffs"] > 0
    assert np.linalg.norm(smoothed - v, axis=1).max() <= 0.8 + 1e-12
    assert topology(smoothed, f) == topology(v, f)
    np.testing.assert_array_equal(f, faces_before)


def test_smoothing_reduces_voxel_stair_steps_with_bounded_movement():
    pytest.importorskip("skimage")
    from lodstone.biological_mesh import label_surface

    z, y, x = np.indices((28, 28, 28))
    mask = ((x - 14) ** 2 + (y - 14) ** 2 + (z - 14) ** 2) < 100
    v, f = label_surface(mask, np.ones(3), np.zeros(3))
    result, change = smooth_surface(v, f, max_displacement_um=0.8)
    # For a voxelized sphere, radial variation measures the stair-step artifact.
    before = np.linalg.norm(v - [14, 14, 14], axis=1).std()
    after = np.linalg.norm(result - [14, 14, 14], axis=1).std()
    assert after < before * 0.8
    assert change["max_vertex_displacement_um"] <= 0.8 + 1e-12
    assert change["max_component_relative_volume_change"] <= 0.05 + 1e-12


def test_smoothing_keeps_a_one_voxel_bridge_without_local_face_reversal():
    pytest.importorskip("skimage")
    from lodstone.biological_mesh import label_surface, topology

    mask = np.zeros((24, 18, 34), dtype=bool)
    mask[5:18, 4:14, 3:13] = True
    mask[5:18, 4:14, 21:31] = True
    mask[11:12, 8:9, 12:22] = True
    v, f = label_surface(mask, np.array([0.8, 0.4, 0.3]), np.zeros(3))
    smoothed, change = smooth_surface(v, f, max_displacement_um=0.24)
    assert topology(smoothed, f) == topology(v, f)
    raw_t, new_t = v[f], smoothed[f]
    raw_n = np.cross(raw_t[:, 1] - raw_t[:, 0], raw_t[:, 2] - raw_t[:, 0])
    new_n = np.cross(new_t[:, 1] - new_t[:, 0], new_t[:, 2] - new_t[:, 0])
    assert np.all(np.einsum("ij,ij->i", raw_n, new_n) > 0)
    assert change["max_vertex_displacement_um"] <= 0.24 + 1e-12


def test_public_neuron_fixture_streams_every_supplied_face():
    cache = os.environ.get("LODSTONE_NEURON_TEST_CACHE")
    if not cache:
        pytest.skip(
            "set LODSTONE_NEURON_TEST_CACHE to opt into a prepared public fixture"
        )
    from lodstone.biological_mesh import verify_reference
    from lodstone.neuron_mesh import load_neuron_meshes

    report = json.loads((Path(cache) / "surfaces-v4-smooth20/report.json").read_text())
    meshes, _ = load_neuron_meshes(Path(cache))
    assert {i for i, _, _ in meshes} == {o["label_id"] for o in report["objects"]}
    assert report["coverage"]["omitted_labels"] == []
    assert report["coverage"]["meshed_labels"] == 5
    result = verify_reference(report)
    assert result["streamed_faces"] == sum(len(f) for _, _, f in meshes)
    assert result["declared_smoothing_regenerated"]
    coarse = verify_reference(report, level=1)
    assert coarse["streamed_faces"] == sum(
        o["coarse"]["faces"] for o in report["objects"]
    )
    assert coarse["streamed_faces"] < result["streamed_faces"] * 0.7
    for item in report["objects"]:
        assert item["coarse_construction"]["protected_faces_exact"]
        assert item["coarse_construction"]["source_face_map_certified"]
        for key in ("components", "euler", "boundary_edges", "nonmanifold_edges"):
            assert item["fine"][key] == item["coarse"][key]


@pytest.mark.parametrize("level_count", [1, 2])
def test_surface_writer_can_be_consumed_with_one_or_two_levels(tmp_path, level_count):
    building = pytest.importorskip("zarr_vectors.building")
    from lodstone.biological_mesh import write_surface_levels
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    vertices = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    levels = [[(vertices, faces, 42)]] * level_count
    root = write_surface_levels(tmp_path, levels)
    source = ZarrVectorsMeshSource(root, level_errors=[0.0] * level_count)
    try:
        assert len(source.levels) == level_count
        for level in range(level_count):
            with building.MeshReader(root, level=level) as reader:
                result = reader.read([c.key for c in reader.chunks])
                assert len(result["faces"]) == len(faces)
    finally:
        source.close()
