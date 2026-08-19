"""Renderer-neutral plan comparison and source-provenance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import Plan, Tile, View


@dataclass(frozen=True, slots=True)
class PlanTrace:
    """Planner-independent summary of the geometry and order in a plan."""

    target_level: int
    tiles: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]
    wanted: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]

    @staticmethod
    def _tiles(
        tiles: tuple[Tile, ...],
    ) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]:
        return tuple(
            (tile.level, tile.region.start, tile.region.stop, tile.phase)
            for tile in tiles
        )

    @classmethod
    def from_plan(cls, plan: Plan) -> PlanTrace:
        """Create a stable trace while ignoring planner-specific tile keys."""

        tiles = plan.desired or plan.wanted
        return cls(
            target_level=plan.target_level,
            tiles=cls._tiles(tiles),
            wanted=cls._tiles(plan.wanted),
        )


@dataclass(frozen=True, slots=True)
class PlanComparison:
    """Two planning traces for the same captured view."""

    view: View
    reference: PlanTrace
    candidate: PlanTrace

    @property
    def matches(self) -> bool:
        return self.reference == self.candidate

    @property
    def geometry_matches(self) -> bool:
        """Compare levels and regions while ignoring delivery order."""

        return (
            self.reference.target_level == self.candidate.target_level
            and frozenset(self.reference.tiles) == frozenset(self.candidate.tiles)
            and frozenset(self.reference.wanted)
            == frozenset(self.candidate.wanted)
        )


class LevelDiagnosticArray:
    """Read an array normally but return a categorical source-level value."""

    def __init__(
        self,
        array: Any,
        level: int,
        *,
        missing_value: int = 1,
        level_offset: int = 2,
        dtype: np.dtype | str | type = np.uint8,
    ) -> None:
        self._array = array
        self.level = int(level)
        self.level_offset = int(level_offset)
        self.shape = tuple(int(value) for value in array.shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.dtype = np.dtype(dtype)
        self.fill_value = self.dtype.type(missing_value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._array, name)

    def __getitem__(self, key: Any) -> np.ndarray:
        result = self._array[key]
        if hasattr(result, "read"):
            result = result.read().result()
        if hasattr(result, "compute"):
            result = result.compute()
        return np.full(
            np.asarray(result).shape,
            self.level + self.level_offset,
            dtype=self.dtype,
        )
