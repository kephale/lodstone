"""Mesh resources are released independently of a shared viewer runtime."""

import asyncio
import queue
import threading

import numpy as np
import pytest

from lodstone.mesh import MeshChunk, MeshData, MeshLevel, MeshStream
from lodstone.model import View
from lodstone.runtime import Runtime


class Source:
    levels = (MeshLevel((MeshChunk((0,), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),), 0.0),)

    async def read(self, level, key):
        return MeshData(
            np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            np.array([[0, 1, 2]]),
        )


class Target:
    def __init__(self):
        self.phases = []

    def replace(self, level, chunks):
        self.phases.append(level)


def view():
    return View((0, 1, 2), (None, None, None), (100, 100), np.eye(4))


def test_close_cancels_every_generation_without_closing_shared_runtime():
    entered, cancelled = threading.Event(), threading.Event()

    class WaitingSource(Source):
        async def read(self, level, key):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    runtime = Runtime()
    target = Target()
    stream = MeshStream(WaitingSource(), target, runtime=runtime)
    other = MeshStream(Source(), Target(), runtime=runtime)
    try:
        stream.update(view())
        assert entered.wait(3)
        futures = [stream._future]
        for _ in range(12):
            stream.update(view())
            futures.append(stream._future)
        stream.close()
        assert cancelled.wait(3)
        assert all(future.done() for future in futures)
        assert not runtime.closed and stream.snapshot().state == "closed"
        assert not target.phases
        other.update(view())
        other.wait()
        assert other.snapshot().state == "complete"
    finally:
        stream.close()
        other.close()
        runtime.close()


def test_queued_publication_is_harmless_after_shared_stream_close():
    callbacks = queue.Queue()
    runtime, target = Runtime(), Target()
    stream = MeshStream(Source(), target, runtime=runtime, dispatch=callbacks.put)
    try:
        stream.update(view())
        apply = callbacks.get(timeout=3)
        stream.close()
        assert stream._future.done()
        apply()
        assert not target.phases and not runtime.closed
        assert stream.snapshot().cache_bytes == stream.snapshot().staged_bytes == 0
    finally:
        stream.close()
        runtime.close()


def test_close_during_staging_does_not_restore_closed_stream_progress(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    runtime, target = Runtime(), Target()
    stream = MeshStream(Source(), target, runtime=runtime)
    remember = stream._remember

    def blocked_remember(*args):
        result = remember(*args)
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(stream, "_remember", blocked_remember)
    try:
        stream.update(view())
        assert entered.wait(3)
        stream.close()
        release.set()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), runtime.loop).result(3)
        state = stream.snapshot()
        assert state.state == "closed" and state.cache_bytes == state.staged_bytes == 0
        assert not target.phases
    finally:
        release.set()
        stream.close()
        runtime.close()


@pytest.fixture
def mesh_group(tmp_path):
    building = pytest.importorskip("zarr_vectors.building")
    from zarr_vectors.core.store import open_store

    path = tmp_path / "mesh"
    building.write_mesh(
        path,
        np.array([[0.1, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        np.array([[0, 1, 2]]),
        chunk_shape=(1.0, 1.0, 1.0),
    )
    return open_store(path)


def test_blocking_backend_read_can_finish_safely_after_stream_close(
    mesh_group, monkeypatch
):
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    entered, release, finished, closing, closed = (threading.Event() for _ in range(5))
    source = ZarrVectorsMeshSource(mesh_group, level_errors=[0.0])
    original_read = source._readers[0].read

    def blocked_read(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        result = original_read(*args, **kwargs)
        finished.set()
        return result

    monkeypatch.setattr(source._readers[0], "read", blocked_read)
    runtime, target = Runtime(), Target()
    stream = MeshStream(source, target, runtime=runtime)

    def close_source():
        closing.set()
        source.close()
        closed.set()

    closer = threading.Thread(target=close_source)
    try:
        stream.update(view())
        assert entered.wait(3)
        stream.close()
        assert stream._future.done() and not finished.is_set()
        closer.start()
        assert closing.wait(3) and not closed.is_set()
        release.set()
        assert closed.wait(3) and finished.is_set()
        assert not target.phases and not runtime.closed
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=3)
        stream.close()
        source.close()
        runtime.close()


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -1.0])
def test_invalid_errors_open_no_metadata_snapshot(mesh_group, monkeypatch, error):
    from zarr_vectors import building

    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    created = []

    class Reader(building.MeshReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(building, "MeshReader", Reader)
    with pytest.raises(ValueError) as caught:
        ZarrVectorsMeshSource(mesh_group, level_errors=[error])
    assert caught.value is not None  # retain the traceback during the assertions
    assert not created and mesh_group._node_cache is None


@pytest.mark.parametrize("failure", ["second_reader", "unsupported_width"])
def test_failed_constructor_closes_acquired_readers_even_with_retained_traceback(
    mesh_group, monkeypatch, failure
):
    from zarr_vectors import building

    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    created = []

    class Reader(building.MeshReader):
        def __init__(self, *args, **kwargs):
            if failure == "second_reader" and created:
                raise OSError("second level unavailable")
            super().__init__(*args, **kwargs)
            created.append(self)
            if failure == "unsupported_width":
                self.width = 4

    monkeypatch.setattr(building, "MeshReader", Reader)
    errors = [0.0, 0.1] if failure == "second_reader" else [0.0]
    with pytest.raises((OSError, NotImplementedError)) as caught:
        ZarrVectorsMeshSource(mesh_group, level_errors=errors)
    assert caught.value is not None
    assert created and all(reader._closed for reader in created)
    assert mesh_group._node_cache is None
