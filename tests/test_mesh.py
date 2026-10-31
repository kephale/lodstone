from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from lodstone import View
from lodstone.mesh import (
    MeshChunk,
    MeshData,
    MeshLevel,
    MeshStream,
    select_mesh_level,
    visible_chunks,
)


def view(scale=1.0):
    matrix = np.diag([scale, scale, scale, 1.0])
    return View((0, 1, 2), (None, None, None), (100, 100), matrix)


def triangle():
    return MeshData(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        np.array([[2, 1, 0]], dtype=np.int64),
    )


class Source:
    levels = (
        MeshLevel((MeshChunk((0, 0, 0), ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))),), 0.0),
        MeshLevel((MeshChunk((0, 0, 0), ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))),), 0.1),
    )

    def __init__(self):
        self.calls = []

    async def read(self, level, key):
        self.calls.append((level, key))
        return triangle()


class Target:
    def __init__(self):
        self.phases = []

    def replace(self, level, chunks):
        self.phases.append((level, tuple(chunks)))


def test_culling_uses_geometry_bounds_and_lod_uses_explicit_error():
    near = MeshChunk((0,), ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    far = MeshChunk((1,), ((10.0, 10.0, 10.0), (11.0, 11.0, 11.0)))
    assert visible_chunks(MeshLevel((near, far), 0.0), view()) == (near,)
    assert select_mesh_level(Source.levels, view(), 2.0) == 0
    assert select_mesh_level(Source.levels, view(0.1), 2.0) == 1
    # A box straddling the near plane is conservatively retained.
    assert visible_chunks(
        MeshLevel((MeshChunk((2,), ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),), 0.0),
        view(),
    )


def test_coarse_to_fine_and_cached_revisit():
    source, target = Source(), Target()
    stream = MeshStream(source, target)
    try:
        stream.update(view(), level=0)
        stream.wait()
        assert [level for level, _ in target.phases] == [1, 0]
        assert stream.reads == 2
        stream.update(view(), level=0)
        stream.wait()
        assert stream.reads == 2
        assert stream.cache_bytes <= stream.memory_limit
        stream.update(view(0.01), level=1)
        stream.wait()
        assert stream.displayed_level == 1
    finally:
        stream.close()


def test_budget_preserves_completed_coarse_phase():
    class LargeFine(Source):
        async def read(self, level, key):
            if level:
                return triangle()
            return MeshData(np.tile(triangle().vertices, (10, 1)), triangle().faces)

    target = Target()
    stream = MeshStream(LargeFine(), target, memory_limit=100)
    try:
        stream.update(view(), level=0)
        stream.wait()
        assert stream.limited
        assert stream.displayed_level == 1
        assert [level for level, _ in target.phases] == [1]
        assert stream.cache_bytes <= 100
        assert stream.snapshot().state == "limited"
    finally:
        stream.close()


def test_oversized_coarse_fails_instead_of_silently_cropping():
    stream = MeshStream(Source(), Target(), memory_limit=1)
    try:
        stream.update(view())
        with pytest.raises(MemoryError):
            stream.wait()
        assert isinstance(stream.error, MemoryError)
        assert stream.snapshot().state == "failed"
        assert stream.snapshot().error is not None
    finally:
        stream.close()


def test_stale_blocking_read_does_not_publish_or_overlap():
    entered, release = threading.Event(), threading.Event()

    class Slow(Source):
        async def read(self, level, key):
            self.calls.append(level)
            if len(self.calls) == 1:
                entered.set()
                await asyncio.to_thread(release.wait)
            return triangle()

    source, target = Slow(), Target()
    stream = MeshStream(source, target)
    try:
        stream.update(view(), level=0)
        assert entered.wait(3)
        stream.update(view(), level=1)
        assert len(source.calls) == 1
        release.set()
        stream.wait()
        assert [level for level, _ in target.phases] == [1]
        assert stream.snapshot().stale_reads == 1
    finally:
        release.set()
        stream.close()


def test_delayed_host_callback_rechecks_generation():
    import queue

    callbacks = queue.Queue()
    target = Target()
    stream = MeshStream(Source(), target, dispatch=callbacks.put)
    try:
        stream.update(view(), level=0)
        stale = callbacks.get(timeout=3)
        stream.update(view(), level=1)
        stale()
        callbacks.get(timeout=3)()
        stream.wait()
        assert [level for level, _ in target.phases] == [1]
    finally:
        stream.close()


def test_zarr_vectors_boundary_triangle_and_affine(tmp_path):
    pytest.importorskip("zarr_vectors")
    from zarr_vectors import building

    if not hasattr(building, "MeshReader"):
        pytest.skip("requires companion mesh reader branch")
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    vertices = np.array(
        [[1.0, 1.0, 1.0], [11.0, 1.0, 1.0], [1.0, 11.0, 1.0]], dtype=np.float32
    )
    building.write_mesh(
        tmp_path / "mesh.zv",
        vertices,
        np.array([[2, 1, 0]]),
        chunk_shape=(10.0, 10.0, 10.0),
    )
    transform = np.diag([2.0, 3.0, 4.0, 1.0])
    transform[:3, 3] = [5.0, -7.0, 9.0]
    source = ZarrVectorsMeshSource(
        tmp_path / "mesh.zv", level_errors=[0.0], transform=transform
    )
    result = asyncio.run(source.read(0, (0, 0, 0)))
    expected = vertices @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_array_equal(result.vertices[result.faces], expected[[[2, 1, 0]]])


def test_source_budget_failure_keeps_coarse_phase():
    class Limited(Source):
        async def read(self, level, key):
            if level == 0:
                raise MemoryError("decoded dependency budget")
            return triangle()

    target = Target()
    stream = MeshStream(Limited(), target)
    try:
        stream.update(view(), level=0)
        stream.wait()
        assert stream.limited and stream.displayed_level == 1
        assert stream.error is None
    finally:
        stream.close()


def test_pygfx_target_releases_previous_objects_and_caps_buffers():
    gfx = pytest.importorskip("pygfx")
    from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget

    target = PyGFXMeshTarget(gfx.Scene(), memory_limit=100)
    target.replace(1, [((1, (0,)), triangle())])
    assert len(target.group.children) == 1
    assert target.submitted_bytes == 84  # positions + normals + uint32 faces
    with pytest.raises(MemoryError):
        target.replace(0, [((0, (0,)), triangle()), ((0, (1,)), triangle())])
    assert target.level == 1 and len(target.group.children) == 1
    target.replace(0, [])
    assert not target.group.children and target.submitted_bytes == 0


def test_target_budget_failure_keeps_coarse_phase():
    class LimitedTarget(Target):
        def replace(self, level, chunks):
            if level == 0:
                raise MemoryError("render buffer budget")
            super().replace(level, chunks)

    stream = MeshStream(Source(), LimitedTarget())
    try:
        stream.update(view(), level=0)
        stream.wait()
        assert stream.limited and stream.displayed_level == 1
        assert stream.error is None
    finally:
        stream.close()


def test_phase_replacement_evicts_cache_without_losing_the_surface():
    target = Target()
    stream = MeshStream(Source(), target, memory_limit=100)
    try:
        stream.update(view(), level=0)
        stream.wait()
        state = stream.snapshot()
        assert state.evictions == 1 and state.cache_bytes == 60
        assert state.state == "complete" and state.displayed_level == 0
        assert len(target.phases[-1][1]) == 1
    finally:
        stream.close()
