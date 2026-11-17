"""Source singularities and a surface error certificate survive coarse LOD."""

import numpy as np
import pytest

from lodstone.mesh_simplification import certify_surface_map, constrained_coarse_mesh

pytest.importorskip("scipy")


def test_certificate_covers_removed_triangles_as_well_as_retained_faces():
    fine = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0]])
    faces = np.array([[0, 3, 2], [3, 1, 2]])
    coarse, triangle = fine[:3], np.array([[0, 1, 2]])
    mapping = np.array([0, 1, 2, 0])
    assert certify_surface_map(fine, faces, coarse, triangle, mapping) == 0.5
    with pytest.raises(ValueError, match="surviving source-face map"):
        certify_surface_map(fine, faces, coarse, triangle[:, ::-1], mapping)


def test_singular_features_stay_fixed_while_regular_geometry_gets_smaller():
    pytest.importorskip("skimage")
    from lodstone.biological_mesh import label_surface, topology

    z, y, x = np.indices((22, 22, 22))
    mask = (z - 11) ** 2 + (y - 11) ** 2 + (x - 11) ** 2 < 64
    v, f = label_surface(mask, np.ones(3) * 0.1, np.zeros(3))
    # Source duplicate face, open triangle, and two tetrahedra meeting at one
    # bow-tie vertex. None should be repaired or swallowed by simplification.
    f = np.concatenate([f, f[:1]])
    start = len(v)
    extra = np.array(
        [
            [5.0, 0, 0],
            [6, 0, 0],
            [5, 1, 0],
            [8, 0, 0],
            [9, 0, 0],
            [8, 1, 0],
            [8, 0, 1],
            [7, 0, 0],
            [8, -1, 0],
            [8, 0, -1],
        ]
    )
    extra_faces = (
        np.array(
            [
                [0, 1, 2],
                [3, 5, 4],
                [3, 4, 6],
                [3, 6, 5],
                [4, 5, 6],
                [3, 8, 7],
                [3, 7, 9],
                [3, 9, 8],
                [7, 8, 9],
            ]
        )
        + start
    )
    v, f = np.concatenate([v, extra]), np.concatenate([f, extra_faces])
    cv, cf, report = constrained_coarse_mesh(v, f, max_error=0.12)
    assert len(cf) < len(f) * 0.8
    assert report["protected_faces_exact"] and report["source_face_map_certified"]
    assert report["surface_distance_upper_bound_um"] <= 0.12 + 1e-12
    before, after = topology(v, f), topology(cv, cf)
    for k in ("components", "euler", "boundary_edges", "nonmanifold_edges"):
        assert before[k] == after[k]
    actual = {tuple(t.ravel()) for t in cv[cf]}
    assert all(tuple(t.ravel()) in actual for t in v[extra_faces])


def test_thin_bridge_stays_connected_and_bound_is_respected():
    pytest.importorskip("skimage")
    from lodstone.biological_mesh import label_surface, topology

    mask = np.zeros((24, 18, 34), dtype=bool)
    mask[5:18, 4:14, 3:13] = True
    mask[5:18, 4:14, 21:31] = True
    mask[11:12, 8:9, 12:22] = True
    v, f = label_surface(mask, np.array([0.8, 0.4, 0.3]), np.zeros(3))
    cv, cf, report = constrained_coarse_mesh(v, f, max_error=0.24)
    assert len(cf) < len(f)
    assert topology(cv, cf)["components"] == 1
    assert topology(cv, cf)["boundary_edges"] == 0
    assert report["surface_distance_upper_bound_um"] <= 0.24 + 1e-12
    triangles = cv[cf]
    assert np.all(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        > 0
    )
