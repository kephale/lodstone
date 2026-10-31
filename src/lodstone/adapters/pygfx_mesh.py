"""Experimental PyGFX consumer for complete uniform-resolution mesh phases."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np

from ..mesh import MeshData, MeshKey


class PyGFXMeshTarget:
    """Replace a group on the render thread; old geometry leaves the scene.

    The byte limit covers submitted position/index/normal buffers. Actual driver
    allocation, padding and deferred GPU resource destruction are not measured.
    """

    def __init__(self, scene: Any, *, memory_limit: int = 128 << 20) -> None:
        gfx = import_module("pygfx")

        if memory_limit <= 0:
            raise ValueError("memory limit must be positive")
        self.group = gfx.Group()
        scene.add(self.group)
        self.memory_limit = memory_limit
        self.submitted_bytes = 0
        self.level: int | None = None

    def replace(self, level: int, chunks: Sequence[tuple[MeshKey, MeshData]]) -> None:
        gfx = import_module("pygfx")

        buffers = []
        for key, data in chunks:
            if len(data.vertices) > np.iinfo(np.uint32).max:
                raise ValueError("PyGFX mesh fragment exceeds uint32 index capacity")
            buffers.append(
                (
                    key,
                    np.asarray(data.vertices, dtype=np.float32),
                    np.asarray(data.faces, dtype=np.uint32),
                )
            )
        size = sum(2 * vertices.nbytes + faces.nbytes for _, vertices, faces in buffers)
        if size > self.memory_limit:
            raise MemoryError("mesh publication exceeds target buffer limit")
        objects = []
        for key, vertices, faces in buffers:
            if not len(faces):
                continue
            normals = np.zeros_like(vertices)
            triangles = vertices[faces]
            face_normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            for corner in range(3):
                np.add.at(normals, faces[:, corner], face_normals)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            np.divide(normals, lengths, out=normals, where=lengths > 0)
            geometry = gfx.Geometry(positions=vertices, indices=faces, normals=normals)
            material = gfx.MeshNormalMaterial(side="both")
            mesh = gfx.Mesh(geometry, material)
            mesh.name = f"L{level}:{key[1]}"
            objects.append(mesh)
        self.group.clear()
        if objects:
            self.group.add(*objects)
        self.level, self.submitted_bytes = level, size

    def close(self) -> None:
        self.group.clear()
        self.submitted_bytes = 0
