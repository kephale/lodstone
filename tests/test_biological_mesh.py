"""Physical alignment, topology and explicit approximation limits."""

from pathlib import Path

import numpy as np
import pytest

from lodstone.biological_mesh import label_surface, surface_error_bound, topology

pytest.importorskip("skimage")
pytest.importorskip("scipy")


def test_asymmetric_label_has_correct_xyz_physical_bounds():
    mask = np.zeros((8, 10, 12), dtype=bool)
    mask[2:5, 3:7, 4:9] = True
    spacing = np.array([2.0, 3.0, 5.0])
    origin = np.array([100.0, 200.0, 300.0])
    vertices, faces = label_surface(mask, spacing, origin)
    np.testing.assert_allclose(vertices.min(axis=0), [317.5, 207.5, 103.0])
    np.testing.assert_allclose(vertices.max(axis=0), [342.5, 219.5, 109.0])
    summary = topology(vertices, faces)
    assert summary["components"] == 1 and summary["euler"] == 2
    assert summary["boundary_edges"] == summary["nonmanifold_edges"] == 0


def test_topology_detects_a_missing_face():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    v, f = label_surface(mask, np.ones(3), np.zeros(3))
    assert topology(v, f[1:])["boundary_edges"] == 3


def test_thin_bridge_control_survives_simplification():
    simplifier = pytest.importorskip("fast_simplification")
    mask = np.zeros((24, 18, 34), dtype=bool)
    mask[5:18, 4:14, 3:13] = True
    mask[5:18, 4:14, 21:31] = True
    mask[11:12, 8:9, 12:22] = True  # one-voxel bridge, not nuclei evidence
    v, f = label_surface(mask, np.array([0.8, 0.4, 0.3]), np.zeros(3))
    cv, cf = simplifier.simplify(v, f, target_reduction=0.7)
    before, after = topology(v, f), topology(cv, cf)
    for key in ["components", "euler", "boundary_edges", "nonmanifold_edges"]:
        assert before[key] == after[key]


def test_surface_bound_is_not_a_sampled_error_claim():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    shifted = vertices + [0.0, 0.0, 3.0]
    result = surface_error_bound(vertices, faces, shifted, faces)
    assert result["sampled_vertex_distance_max_um"] == 3.0
    assert result["surface_distance_upper_bound_um"] > 3.0


def test_cached_public_fixture_matches_retained_reference():
    # No network in tests. An explicit opt-in cache path enables this check.
    import json
    import os

    cache = os.environ.get("LODSTONE_BIOLOGICAL_TEST_CACHE")
    if not cache:
        pytest.skip(
            "set LODSTONE_BIOLOGICAL_TEST_CACHE after preparing the public fixture"
        )
    from lodstone.biological_mesh import verify_reference

    report = json.loads((Path(cache) / "surfaces-v2-6/report.json").read_text())
    result = verify_reference(report)
    assert result["streamed_faces"] == result["reference_faces"] == 81566
