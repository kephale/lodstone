"""Data-source protocol for multiscale arrays."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .model import Pyramid, Region


@runtime_checkable
class Source(Protocol):
    """A multiscale data source capable of asynchronous regional reads."""

    @property
    def pyramid(self) -> Pyramid: ...

    async def read(self, level: int, region: Region) -> np.ndarray:
        """Return ``region`` at ``level`` in data-axis order."""
        ...
