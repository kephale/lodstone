"""Bounded NumPy residency for dense viewer targets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np

from .model import Plan, Pyramid, Region, Tile, TileKey, Update


@dataclass(eq=False, slots=True)
class ResidentWindow:
    """One level's bounded full-ND CPU buffer."""

    level: int
    region: Region
    data: np.ndarray
    transform: np.ndarray
    key_regions: dict[TileKey, Region] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def write(self, update: Update) -> None:
        """Write a full-ND update using coordinates relative to the window."""
        overlap = self.region.intersection(update.region)
        if overlap != update.region:
            raise ValueError("update lies outside its prepared resident window")
        if update.data.shape != update.region.shape:
            raise ValueError(
                "resident windows require unsqueezed full-dimensional updates"
            )
        destination = tuple(
            slice(
                update.region.start[axis] - self.region.start[axis],
                update.region.stop[axis] - self.region.start[axis],
            )
            for axis in range(self.region.ndim)
        )
        self.data[destination] = update.data
        self.key_regions[update.key] = update.region


@dataclass(frozen=True, slots=True)
class ResidentChange:
    """Updates written to one resident window."""

    window: ResidentWindow
    updates: tuple[Update, ...]


@dataclass(frozen=True, slots=True)
class ResidentTransition:
    """Windows created or retired by a pass lifecycle operation."""

    prepared: tuple[ResidentWindow, ...] = ()
    retired: tuple[ResidentWindow, ...] = ()


class ResidentArrays:
    """Manage bounded dense CPU windows for a renderer-specific target.

    ``prepare`` stages the complete desired ladder, ``apply`` writes updates
    into those staged windows, and ``complete`` promotes the target-level
    window while retiring coarse and obsolete buffers. Window arrays retain
    all pyramid dimensions; targets using this helper should request
    ``Layout(squeeze_hidden=False)``.
    """

    def __init__(
        self,
        pyramid: Pyramid,
        *,
        dtypes: Sequence[np.dtype | str | type] | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.dtypes = tuple(
            np.dtype(value)
            for value in (dtypes or [level.dtype for level in pyramid.levels])
        )
        if len(self.dtypes) != len(pyramid.levels):
            raise ValueError("dtypes must contain one value per pyramid level")
        self.active: dict[int, ResidentWindow] = {}
        self.pending: dict[int, ResidentWindow] | None = None

    @property
    def windows(self) -> dict[int, ResidentWindow]:
        """Windows receiving updates for the current pass."""
        return self.pending if self.pending is not None else self.active

    @property
    def nbytes(self) -> int:
        """Physical bytes held across active and staged windows."""
        unique = {id(window): window for window in self.active.values()}
        if self.pending is not None:
            unique.update({id(window): window for window in self.pending.values()})
        return sum(window.nbytes for window in unique.values())

    def prepare(self, plan: Plan) -> ResidentTransition:
        """Stage bounded windows covering every desired level."""
        regions = _level_regions(plan.desired or plan.wanted)
        previous_pending = self.pending or {}
        candidates = {**self.active, **previous_pending}
        pending: dict[int, ResidentWindow] = {}
        prepared = []

        for level, region in regions.items():
            previous = candidates.get(level)
            if previous is not None and previous.region == region:
                pending[level] = previous
                continue
            window = self._allocate(level, region)
            if previous is not None:
                _copy_overlap(previous, window)
            pending[level] = window
            prepared.append(window)

        reused = set(pending.values())
        retired = tuple(
            window
            for window in previous_pending.values()
            if window not in reused and window not in self.active.values()
        )
        self.pending = pending
        return ResidentTransition(tuple(prepared), retired)

    def apply(self, updates: Sequence[Update]) -> tuple[ResidentChange, ...]:
        """Write a batch and group renderer notifications by window."""
        grouped: dict[ResidentWindow, list[Update]] = defaultdict(list)
        windows = self.windows
        for update in updates:
            try:
                window = windows[update.level]
            except KeyError as error:
                raise RuntimeError(
                    f"level {update.level} was not prepared for this pass"
                ) from error
            window.write(update)
            grouped[window].append(update)
        return tuple(
            ResidentChange(window, tuple(batch)) for window, batch in grouped.items()
        )

    def discard(self, keys: Collection[TileKey]) -> None:
        """Forget logical tile ownership without changing window storage."""
        for window in _unique_windows(self.active, self.pending or {}):
            for key in keys:
                window.key_regions.pop(key, None)

    def complete(self, plan: Plan) -> ResidentTransition:
        """Promote the target window and retire coarse or replaced storage."""
        if self.pending is None:
            return ResidentTransition()
        target = self.pending.get(plan.target_level)
        next_active = {} if target is None else {plan.target_level: target}
        retained = set(next_active.values())
        retired = tuple(
            window
            for window in _unique_windows(self.active, self.pending)
            if window not in retained
        )
        self.active = next_active
        self.pending = None
        return ResidentTransition(retired=retired)

    def clear(self) -> ResidentTransition:
        """Retire all active and staged windows."""
        retired = tuple(_unique_windows(self.active, self.pending or {}))
        self.active = {}
        self.pending = None
        return ResidentTransition(retired=retired)

    def _allocate(self, level: int, region: Region) -> ResidentWindow:
        info = self.pyramid.levels[level]
        data = np.zeros(region.shape, dtype=self.dtypes[level])
        return ResidentWindow(level, region, data, info.voxel_to_world)


def _level_regions(tiles: Sequence[Tile]) -> dict[int, Region]:
    bounds: dict[int, tuple[list[int], list[int]]] = {}
    for tile in tiles:
        if tile.level not in bounds:
            bounds[tile.level] = (
                list(tile.region.start),
                list(tile.region.stop),
            )
            continue
        start, stop = bounds[tile.level]
        for axis in range(tile.region.ndim):
            start[axis] = min(start[axis], tile.region.start[axis])
            stop[axis] = max(stop[axis], tile.region.stop[axis])
    return {
        level: Region(tuple(start), tuple(stop))
        for level, (start, stop) in bounds.items()
    }


def _copy_overlap(source: ResidentWindow, destination: ResidentWindow) -> None:
    overlap = source.region.intersection(destination.region)
    if overlap is None:
        return
    source_slice = tuple(
        slice(
            overlap.start[axis] - source.region.start[axis],
            overlap.stop[axis] - source.region.start[axis],
        )
        for axis in range(overlap.ndim)
    )
    destination_slice = tuple(
        slice(
            overlap.start[axis] - destination.region.start[axis],
            overlap.stop[axis] - destination.region.start[axis],
        )
        for axis in range(overlap.ndim)
    )
    destination.data[destination_slice] = source.data[source_slice]
    destination.key_regions.update(
        {
            key: region
            for key, region in source.key_regions.items()
            if destination.region.intersection(region) == region
        }
    )


def _unique_windows(*collections: dict[int, ResidentWindow]) -> list[ResidentWindow]:
    unique: dict[int, ResidentWindow] = {}
    for windows in collections:
        unique.update({id(window): window for window in windows.values()})
    return list(unique.values())
