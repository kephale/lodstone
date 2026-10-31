import asyncio

import numpy as np
import pytest

from lodstone.mesh_diagnostics import StoreReadCounter, counted_store


def test_counter_tracks_returned_ranges_missing_values_and_metadata():
    from zarr.abc.store import RangeByteRequest
    from zarr.core.buffer import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def scenario():
        prototype = default_buffer_prototype()
        original = MemoryStore()
        await original.set("data", prototype.buffer.from_bytes(b"abcdef"))
        await original.set("zarr.json", prototype.buffer.from_bytes(b"{}"))
        counter = StoreReadCounter()
        store = counted_store(original, counter)
        await store.get("data", prototype, RangeByteRequest(1, 4))
        await store.get_partial_values(
            prototype, [("zarr.json", None), ("missing", None)]
        )
        snapshot = counter.snapshot()
        assert snapshot.requests == 3
        assert snapshot.bytes_returned == 5
        assert snapshot.metadata_bytes == 2 and snapshot.payload_bytes == 3
        assert snapshot.misses == 1
        counter.reset()
        assert counter.snapshot().bytes_returned == 0

    asyncio.run(scenario())


def test_counted_mesh_source_reads_only_required_payload(tmp_path):
    pytest.importorskip("zarr_vectors")
    from zarr_vectors import building

    if not hasattr(building, "MeshReader"):
        pytest.skip("requires companion reader branch")
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    path = tmp_path / "mesh.zv"
    building.write_mesh(
        path,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [11.0, 1.0, 1.0],
                [1.0, 11.0, 1.0],
                [101.0, 1.0, 1.0],
                [102.0, 1.0, 1.0],
                [101.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [3, 4, 5]]),
        chunk_shape=(10.0, 10.0, 10.0),
    )
    source = ZarrVectorsMeshSource(path, level_errors=[0.0], track_io=True)
    source.io.reset()
    result = asyncio.run(source.read(0, (0, 0, 0)))
    assert len(result.faces) == 1
    assert source.io.snapshot().payload_bytes > 0
    assert not any("/c/10/0/0" in key for key in source.io.key_reads())
