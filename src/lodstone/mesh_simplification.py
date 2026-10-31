"""Conservative edge collapses for example surfaces with source singularities.

This deliberately favors a bounded, inspectable coarse mesh over a requested
triangle count. It is preprocessing, independent of the runtime mesh streamer.
"""

from __future__ import annotations

import heapq

import numpy as np

from .biological_mesh import topology


def constrained_coarse_mesh(vertices, faces, *, max_error=0.08, target_reduction=0.6):
    """Collapse regular interior edges, keeping source singularities fixed.

    Boundary/nonmanifold edges, duplicate faces and disconnected vertex links
    are protected. The link condition, duplicate-face and local normal checks
    apply to every collapse. A cluster-radius bound limits vertex movement.
    The final original-to-coarse map independently certifies a continuous
    surface-distance bound, including triangles collapsed to edges or points.
    """
    if not np.isfinite(max_error) or max_error <= 0 or not 0 < target_reduction < 1:
        raise ValueError("require positive finite error and reduction in (0, 1)")
    original = np.asarray(vertices, dtype=float)
    original_faces = np.asarray(faces, dtype=np.int64)
    positions, triangles = original.copy(), original_faces.copy()
    nv, nf = len(positions), len(triangles)
    incident: list[set[int]] = [set() for _ in range(nv)]
    neighbors: list[set[int]] = [set() for _ in range(nv)]
    for fi, (a, b, c) in enumerate(triangles.tolist()):
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            incident[u].add(fi)
            neighbors[u].update((v, w))
    edges = np.sort(
        np.concatenate(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    locked = np.zeros(nv, dtype=bool)
    locked[np.unique(edges[counts != 2])] = True
    _, inverse, counts = np.unique(
        np.sort(triangles, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    locked[np.unique(triangles[counts[inverse] > 1])] = True
    # A bow-tie vertex can have only two-face edges but multiple link cycles.
    for u in np.flatnonzero(~locked):
        if not incident[u]:
            continue
        link: dict[int, set[int]] = {}
        for fi in incident[u]:
            a, b = [int(v) for v in triangles[fi] if v != u]
            link.setdefault(a, set()).add(b)
            link.setdefault(b, set()).add(a)
        seen, pending = set(), [next(iter(link))]
        while pending:
            v = pending.pop()
            if v not in seen:
                seen.add(v)
                pending.extend(link[v] - seen)
        if len(seen) != len(link) or any(len(n) != 2 for n in link.values()):
            locked[u] = True
    protected_faces = np.any(locked[triangles], axis=1)
    locked[np.unique(triangles[protected_faces])] = True
    active_v = np.ones(nv, dtype=bool)
    active_f = np.ones(nf, dtype=bool)
    parent = np.arange(nv)
    radius = np.zeros(nv)
    versions = np.zeros(nv, dtype=np.int64)
    heap: list = []

    def enqueue(u, v):
        u, v = min(u, v), max(u, v)
        if u == v or locked[u] or locked[v] or not active_v[u] or not active_v[v]:
            return
        distance = float(np.linalg.norm(positions[u] - positions[v]))
        if distance / 2 + max(radius[u], radius[v]) <= max_error:
            heapq.heappush(heap, (distance, u, v, int(versions[u]), int(versions[v])))

    for u, v in edges.tolist():
        enqueue(u, v)
    remaining, collapses = nf, 0
    while heap and remaining > nf * (1 - target_reduction):
        _, u, v, vu, vv = heapq.heappop(heap)
        if not active_v[u] or not active_v[v] or versions[u] != vu or versions[v] != vv:
            continue
        shared = incident[u] & incident[v]
        if len(shared) != 2:
            continue
        opposite = set(triangles[list(shared)].ravel().tolist()) - {u, v}
        if len(opposite) != 2 or (neighbors[u] & neighbors[v]) != opposite:
            continue
        point = (positions[u] + positions[v]) / 2
        new_radius = (
            max(radius[u], radius[v])
            + float(np.linalg.norm(positions[u] - positions[v])) / 2
        )
        if new_radius > max_error:
            continue
        surviving = sorted((incident[u] | incident[v]) - shared)
        old_faces = triangles[surviving]
        new_faces = np.where(old_faces == v, u, old_faces)
        # Includes the tetrahedron exception to the neighbor-only link check.
        if len(np.unique(np.sort(new_faces, axis=1), axis=0)) != len(new_faces):
            continue
        old_t = positions[old_faces]
        new_t = positions[new_faces].copy()
        new_t[new_faces == u] = point
        old_norm = np.cross(old_t[:, 1] - old_t[:, 0], old_t[:, 2] - old_t[:, 0])
        new_norm = np.cross(new_t[:, 1] - new_t[:, 0], new_t[:, 2] - new_t[:, 0])
        old_area, new_area = (
            np.linalg.norm(old_norm, axis=1),
            np.linalg.norm(new_norm, axis=1),
        )
        if np.any(new_area < old_area * 0.1) or np.any(
            np.einsum("ij,ij->i", old_norm, new_norm) <= 0.5 * old_area * new_area
        ):
            continue
        affected = incident[u] | incident[v]
        local_vertices = set(triangles[list(affected)].ravel().tolist())
        for fi in affected:
            for w in triangles[fi]:
                incident[w].discard(fi)
        active_f[list(shared)] = False
        triangles[surviving] = new_faces
        for fi in surviving:
            for w in triangles[fi]:
                incident[w].add(fi)
        positions[u], radius[u], parent[v], active_v[v] = point, new_radius, u, False
        for w in local_vertices:
            versions[w] += 1
            neighbors[w] = set(triangles[list(incident[w])].ravel().tolist()) - {w}
        # All edges touching changed topology get a fresh candidate version.
        for w in local_vertices:
            for q in neighbors[w]:
                enqueue(w, q)
        remaining -= 2
        collapses += 1
    mapping = np.arange(nv)
    while not np.array_equal(mapping, parent[mapping]):
        mapping = parent[mapping]
    compact = np.full(nv, -1, dtype=np.int64)
    compact[active_v] = np.arange(np.count_nonzero(active_v))
    coarse_v, coarse_f = positions[active_v], compact[triangles[active_f]]
    source_to_coarse = compact[mapping]
    movement = certify_surface_map(
        original, original_faces, coarse_v, coarse_f, source_to_coarse
    )
    if not np.array_equal(
        coarse_v[source_to_coarse[original_faces[protected_faces]]],
        original[original_faces[protected_faces]],
    ):
        raise ValueError("protected source geometry moved")
    before, after = topology(original, original_faces), topology(coarse_v, coarse_f)
    for key in ("components", "euler", "boundary_edges", "nonmanifold_edges"):
        if before[key] != after[key]:
            raise ValueError(f"constrained collapse changed {key}")
    if movement > max_error + 1e-12:
        raise ValueError("coarse displacement exceeds declared bound")
    return (
        coarse_v,
        coarse_f,
        {
            "method": "regular-edge midpoint collapses with protected source singularities",
            "collapses": collapses,
            "locked_vertices": int(locked.sum()),
            "protected_faces": int(protected_faces.sum()),
            "protected_faces_exact": True,
            "requested_reduction": target_reduction,
            "actual_reduction": 1 - len(coarse_f) / nf,
            "maximum_error_um": max_error,
            "surface_distance_upper_bound_um": movement,
            "source_face_map_certified": True,
            "topology_before": before,
            "topology_after": after,
        },
    )


def certify_surface_map(original, original_faces, coarse_v, coarse_f, source_to_coarse):
    """Bound both surface directions using a surjective piecewise-linear map.

    Every source triangle maps by the same barycentric coordinates to either
    a retained triangle or one of its edges/vertices. Every coarse triangle is
    the image of a source triangle. Thus the maximum vertex movement bounds
    the symmetric continuous-surface distance, rather than just point samples.
    """
    # Certify that the piecewise-linear source map covers exactly the coarse
    # surface. Deleted source faces must map to existing coarse edges/vertices.
    mapped = source_to_coarse[original_faces]
    nondegenerate = (
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 1] != mapped[:, 2])
        & (mapped[:, 0] != mapped[:, 2])
    )
    if not np.array_equal(mapped[nondegenerate], coarse_f):
        raise ValueError("coarse faces do not match the surviving source-face map")
    edge_set = {
        tuple(sorted(e))
        for f in coarse_f.tolist()
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
    }
    used_vertices = set(coarse_f.ravel().tolist())
    for face in mapped[~nondegenerate]:
        key = tuple(sorted(set(face.tolist())))
        if (len(key) == 2 and key not in edge_set) or (
            len(key) == 1 and key[0] not in used_vertices
        ):
            raise ValueError("collapsed source face maps outside the coarse surface")
    movement = float(
        np.linalg.norm(coarse_v[source_to_coarse] - original, axis=1).max(initial=0)
    )
    return movement
