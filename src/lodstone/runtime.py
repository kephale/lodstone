"""Shared execution resources for one or more Lodstone streams."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import Any, Self, TypeVar

T = TypeVar("T")


class Runtime:
    """Own the asynchronous scheduler and bounded CPU staging pool.

    A runtime may be shared by multiple streams in one viewer.  Sharing avoids
    creating one scheduler thread and one staging pool per layer or channel and
    establishes the coordination boundary for future cross-stream cache and
    priority policies.  Streams created without an explicit runtime retain the
    original self-contained lifecycle by creating and closing their own runtime.

    Callers that supply an executor retain ownership of it.  Callers that share
    a runtime must close it after all of its streams have been closed.
    """

    def __init__(
        self,
        *,
        compute_workers: int = 1,
        executor: Executor | None = None,
    ) -> None:
        if compute_workers <= 0:
            raise ValueError("compute_workers must be positive")
        self._lock = threading.RLock()
        self._closed = False
        self._executor = executor or ThreadPoolExecutor(
            max_workers=compute_workers,
            thread_name_prefix="lodstone-stage",
        )
        self._owns_executor = executor is None
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="lodstone-runtime",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the runtime event loop while it is open."""

        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            return self._loop

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    async def run_cpu(self, function: Callable[..., T], /, *args: Any) -> T:
        """Run bounded CPU staging without blocking scheduling or I/O."""

        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            executor = self._executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, partial(function, *args))

    def close(self) -> None:
        """Stop scheduling and release runtime-owned compute threads."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()
