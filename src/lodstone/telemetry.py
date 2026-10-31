"""Portable recording and serialization of stream performance samples."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from threading import Lock
from typing import TYPE_CHECKING, Any, Self

from .model import PerformanceSnapshot, Status

if TYPE_CHECKING:
    from collections.abc import Callable

    from .stream import Stream


def _view_record(sample: PerformanceSnapshot) -> dict[str, Any] | None:
    view = sample.view
    if view is None:
        return None
    return {
        "displayed_axes": list(view.displayed_axes),
        "index": list(view.index),
        "viewport": list(view.viewport),
        "world_to_clip": view.world_to_clip.tolist(),
        "eye": None if view.eye is None else list(view.eye),
    }


class PerformanceRecorder:
    """Record comparable status-boundary samples from one stream.

    A renderer may provide extra GPU/upload values through the optional
    ``InstrumentedTarget`` protocol. ``sample`` can also be called by a viewer's
    frame timer to capture upload backlog changes between stream status events.
    """

    def __init__(
        self,
        stream: Stream,
        *,
        host: str = "",
        backend: str = "",
        max_samples: int = 1024,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.stream = stream
        self.host = host
        self.backend = backend
        self._samples: deque[PerformanceSnapshot] = deque(maxlen=max_samples)
        self._lock = Lock()
        self._disconnect: Callable[[], None] | None = stream.on_status_changed(
            self._on_status
        )
        self.sample()

    def _on_status(self, status: Status) -> None:
        del status
        self.sample()

    def sample(self) -> PerformanceSnapshot:
        """Capture and return the current standardized snapshot."""

        snapshot = self.stream.performance
        with self._lock:
            self._samples.append(snapshot)
        return snapshot

    @property
    def samples(self) -> tuple[PerformanceSnapshot, ...]:
        """Return recorded samples in capture order."""

        with self._lock:
            return tuple(self._samples)

    @property
    def latest(self) -> PerformanceSnapshot:
        """Return the most recently captured sample."""

        with self._lock:
            return self._samples[-1]

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-compatible nested records for cross-viewer comparison."""

        records = []
        for sample in self.samples:
            error = sample.status.error
            status = asdict(replace(sample.status, error=None))
            status["error"] = None if error is None else repr(error)
            records.append(
                {
                    "host": self.host,
                    "backend": self.backend,
                    "status": status,
                    "stream": asdict(sample.stream),
                    "target": asdict(sample.target),
                    "view": _view_record(sample),
                    "plan": None if sample.plan is None else asdict(sample.plan),
                }
            )
        return tuple(records)

    def close(self) -> None:
        """Disconnect status sampling; existing samples remain available."""

        if self._disconnect is not None:
            self._disconnect()
            self._disconnect = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
