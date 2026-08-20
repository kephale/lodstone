"""Rendering-target protocol."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Protocol, runtime_checkable

from .model import Layout, Plan, Pyramid, TileKey, Update, View


@runtime_checkable
class Target(Protocol):
    """A viewer-specific consumer of progressive array updates."""

    def layout(self, view: View, pyramid: Pyramid) -> Layout: ...

    def apply(self, updates: Sequence[Update]) -> None: ...

    def discard(self, keys: Collection[TileKey]) -> None: ...

    def redraw(self) -> None: ...


@runtime_checkable
class ResidencyLease(Protocol):
    """Target-confirmed storage that survives request replanning."""

    @property
    def available_keys(self) -> frozenset[TileKey]: ...

    @property
    def pending_keys(self) -> frozenset[TileKey]: ...

    def release(self, keys: Collection[TileKey]) -> None: ...


@runtime_checkable
class PassTarget(Protocol):
    """Optional lifecycle implemented by resident-window render targets."""

    def prepare(self, view: View, plan: Plan) -> ResidencyLease | None:
        """Prepare target residency before the pass starts."""
        ...

    def complete(self, view: View, plan: Plan) -> None:
        """Reconcile or present the completed pass."""
        ...


@runtime_checkable
class PhaseTarget(Protocol):
    """Optional target hook invoked after one progressive phase is delivered."""

    def phase_complete(self, view: View, plan: Plan, phase: int) -> None:
        """Present or reconcile a completed coarse-to-fine phase."""
        ...


@runtime_checkable
class StagingTarget(Protocol):
    """Optional target hook for CPU preparation before host dispatch."""

    def stage(self, updates: Sequence[Update]) -> Any:
        """Prepare updates on the stream thread for cheap host application."""
        ...
