"""Small reusable source and target implementations for adapter tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Collection, Sequence

import numpy as np

from .model import Layout, Pyramid, Region, TargetDiagnostics, TileKey, Update, View
from .sources.array import ArrayPyramidSource


class SimulatedSource(ArrayPyramidSource):
    """An array source with controllable latency and observable reads."""

    def __init__(self, *args, latency: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latency = latency
        self.reads: list[tuple[int, Region]] = []
        self._reads_lock = threading.Lock()

    async def read(self, level: int, region: Region) -> np.ndarray:
        if self.latency:
            await asyncio.sleep(self.latency)
        with self._reads_lock:
            self.reads.append((level, region))
        return await super().read(level, region)


class RecordingTarget:
    """A target that records updates without requiring a renderer."""

    def __init__(self, layout: Layout | None = None) -> None:
        self._layout = layout or Layout()
        self.updates: list[Update] = []
        self.discarded: list[TileKey] = []
        self.redraws = 0
        self._lock = threading.Lock()

    def layout(self, view: View, pyramid: Pyramid) -> Layout:
        return self._layout

    def apply(self, updates: Sequence[Update]) -> None:
        with self._lock:
            self.updates.extend(updates)

    def discard(self, keys: Collection[TileKey]) -> None:
        with self._lock:
            self.discarded.extend(keys)

    def redraw(self) -> None:
        with self._lock:
            self.redraws += 1

    def performance_metrics(self) -> TargetDiagnostics:
        """Report renderer-submission values available to this test target."""

        with self._lock:
            return TargetDiagnostics(
                submitted_bytes=sum(update.data.nbytes for update in self.updates),
                presentations=self.redraws,
            )
