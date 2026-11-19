"""Recursive crystal rosettes on the twenty facets of an icosahedron."""

import numpy as np
from scipy.spatial import ConvexHull

from lodstone.biological_mesh import write_surface_levels


def crystal(depth):
    phi = (1 + np.sqrt(5)) / 2
    points = [
        np.roll([0, a, b * phi], axis) * 7
        for axis in range(3)
        for a in (-1, 1)
        for b in (-1, 1)
    ]
    hull = ConvexHull(points)
    faces = []

    def facet(a, b, c, remaining):
        if remaining == 0:
            faces.append((a, b, c))
            return
        triangle = np.array([points[i] for i in (a, b, c)])
        center = triangle.mean(axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        # An inset keeps a visible terrace around every daughter spire.
        inner = len(points)
        points.extend(center + 0.77 * (triangle - center))
        apex = len(points)
        points.append(
            center
            + normal / np.linalg.norm(normal) * np.sqrt(np.linalg.norm(normal)) * 0.58
        )
        for i, outer in enumerate((a, b, c)):
            j = (i + 1) % 3
            following = (a, b, c)[j]
            faces.extend([(outer, following, inner + j), (outer, inner + j, inner + i)])
            facet(inner + i, inner + j, apex, remaining - 1)

    for a, b, c in hull.simplices:
        if np.dot(points[a], np.cross(points[b], points[c])) < 0:
            b, c = c, b
        facet(a, b, c, depth)
    return np.asarray(points, dtype=np.float32) + 32, np.asarray(faces, dtype=np.int32)


def crystal_store(directory):
    return write_surface_levels(directory, [[(*crystal(d), 1)] for d in (5, 3, 1)])
