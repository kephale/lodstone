"""Optional store I/O and process memory measurements for geometry demos."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class StoreReadSnapshot:
    requests: int
    bytes_returned: int
    metadata_bytes: int
    payload_bytes: int
    misses: int


class StoreReadCounter:
    """Count actual buffers returned by the underlying Zarr store.

    These are encoded-store bytes, including metadata and range reads, not
    decoded geometry bytes or HTTP headers. OS/backend caching is not inferred.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, tuple[int, int]] = {}
        self._requests = self._bytes = self._metadata = self._misses = 0

    def record(self, key: str, buffer: Any) -> None:
        size = len(buffer) if buffer is not None else 0
        with self._lock:
            self._requests += 1
            self._bytes += size
            self._misses += int(buffer is None)
            if key.rsplit("/", 1)[-1] in {
                "zarr.json",
                ".zattrs",
                ".zarray",
                ".zgroup",
                ".zmetadata",
            }:
                self._metadata += size
            count, total = self._keys.get(key, (0, 0))
            self._keys[key] = (count + 1, total + size)

    def snapshot(self) -> StoreReadSnapshot:
        with self._lock:
            return StoreReadSnapshot(
                self._requests,
                self._bytes,
                self._metadata,
                self._bytes - self._metadata,
                self._misses,
            )

    def key_reads(self) -> dict[str, tuple[int, int]]:
        with self._lock:
            return dict(self._keys)

    def reset(self) -> None:
        with self._lock:
            self._keys.clear()
            self._requests = self._bytes = self._metadata = self._misses = 0


def counted_store(store: Any, counter: StoreReadCounter) -> Any:
    """Wrap a supplied Store, keeping get and batched-range reads observable."""
    WrapperStore = import_module("zarr.storage").WrapperStore

    class CountedStore(WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            result = await self._store.get(key, prototype, byte_range)
            counter.record(key, result)
            return result

        async def get_partial_values(self, prototype, key_ranges):
            requests = list(key_ranges)
            results = await self._store.get_partial_values(prototype, requests)
            for (key, _byte_range), result in zip(requests, results, strict=True):
                counter.record(key, result)
            return results

    return CountedStore(store)


class ProcessMemoryMonitor:
    """Sample process RSS in bytes. This is not an actual GPU allocation meter."""

    def __init__(self, interval: float = 0.02) -> None:
        self._process = import_module("psutil").Process()
        self._stop = threading.Event()
        self.current = self.peak = self._process.memory_info().rss
        self.started = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._thread.start()

    def _run(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.current = self._process.memory_info().rss
            self.peak = max(self.peak, self.current)

    def snapshot(self) -> dict:
        self.current = self._process.memory_info().rss
        self.peak = max(self.peak, self.current)
        return {
            "rss_bytes": self.current,
            "sampled_peak_rss_bytes": self.peak,
            "elapsed_seconds": time.monotonic() - self.started,
            "actual_gpu_memory_bytes": None,
            "gpu_memory_note": "Submitted buffer accounting is separate; actual driver allocations unavailable.",
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def diagnostic_snapshot(
    stream: Any, source: Any, target: Any, memory: ProcessMemoryMonitor
) -> dict:
    result = asdict(stream.snapshot())
    result["store_io"] = asdict(source.io.snapshot())
    result["submitted_render_buffer_bytes"] = target.submitted_bytes
    result["process_memory"] = memory.snapshot()
    return result
