"""Rendering-target protocol."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol, runtime_checkable

from .model import Layout, Pyramid, TileKey, Update, View


@runtime_checkable
class Target(Protocol):
    """A viewer-specific consumer of progressive array updates."""

    def layout(self, view: View, pyramid: Pyramid) -> Layout: ...

    def apply(self, updates: Sequence[Update]) -> None: ...

    def discard(self, keys: Collection[TileKey]) -> None: ...

    def redraw(self) -> None: ...
