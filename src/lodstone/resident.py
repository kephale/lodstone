"""Bounded NumPy residency for dense viewer targets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np

from .composition import fill_unloaded_chunks, nearest_resample_region
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
    updates: tuple[Update, ...] = ()
    repaired: tuple[Region, ...] = ()

    @property
    def regions(self) -> tuple[Region, ...]:
        """Every region changed directly or by coarse-backdrop repair."""

        return (*tuple(update.region for update in self.updates), *self.repaired)


@dataclass(frozen=True, slots=True)
class ResidentTransition:
    """Windows created or retired by a pass lifecycle operation."""

    prepared: tuple[ResidentWindow, ...] = ()
    retired: tuple[ResidentWindow, ...] = ()


@dataclass(slots=True)
class ResidentLease:
    """Dynamic lease over keys held by a :class:`ResidentArrays` instance."""

    resident: ResidentArrays
    desired_keys: frozenset[TileKey]

    @property
    def available_keys(self) -> frozenset[TileKey]:
        return self.resident.available_keys()

    @property
    def pending_keys(self) -> frozenset[TileKey]:
        return self.desired_keys - self.available_keys

    def release(self, keys: Collection[TileKey]) -> None:
        self.resident.discard(keys)


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
        compose: bool = False,
        on_timing: Callable[[str, float, int, int, Region], Any] | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.dtypes = tuple(
            np.dtype(value)
            for value in (dtypes or [level.dtype for level in pyramid.levels])
        )
        if len(self.dtypes) != len(pyramid.levels):
            raise ValueError("dtypes must contain one value per pyramid level")
        self.compose = bool(compose)
        self.on_timing = on_timing
        self.lock = RLock()
        self.active: dict[int, ResidentWindow] = {}
        self.pending: dict[int, ResidentWindow] | None = None

    @property
    def windows(self) -> dict[int, ResidentWindow]:
        """Windows receiving updates for the current pass."""
        return self.pending if self.pending is not None else self.active

    @property
    def nbytes(self) -> int:
        """Physical bytes held across active and staged windows."""
        with self.lock:
            unique = {id(window): window for window in self.active.values()}
            if self.pending is not None:
                unique.update({id(window): window for window in self.pending.values()})
            return sum(window.nbytes for window in unique.values())

    def available_keys(self) -> frozenset[TileKey]:
        """Return a synchronized snapshot of every logically resident key."""
        with self.lock:
            return frozenset(
                key
                for window in _unique_windows(self.active, self.pending or {})
                for key in window.key_regions
            )

    def prepare(self, plan: Plan) -> ResidentTransition:
        """Stage bounded windows covering every desired level."""
        with self.lock:
            return self._prepare(plan)

    def _prepare(self, plan: Plan) -> ResidentTransition:
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
                started = perf_counter()
                _copy_overlap(previous, window)
                overlap = previous.region.intersection(window.region)
                if overlap is not None:
                    self._timing(
                        "overlap_copy",
                        started,
                        int(np.prod(overlap.shape)) * window.data.dtype.itemsize,
                        level,
                        overlap,
                    )
            pending[level] = window
            prepared.append(window)

        if self.compose:
            started = perf_counter()
            _compose_windows(self.pyramid, pending)
            self._timing(
                "composition",
                started,
                sum(window.nbytes for window in pending.values()),
                min(pending) if pending else -1,
                next(iter(pending.values())).region if pending else Region((), ()),
            )

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
        with self.lock:
            return self._apply(updates)

    def _apply(self, updates: Sequence[Update]) -> tuple[ResidentChange, ...]:
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
        repaired: dict[ResidentWindow, list[Region]] = defaultdict(list)
        if self.compose:
            started = perf_counter()
            for window, regions in _compose_windows(
                self.pyramid,
                windows,
                source_levels={window.level for window in grouped},
            ).items():
                repaired[window].extend(regions)
            self._timing(
                "composition",
                started,
                sum(window.nbytes for window in windows.values()),
                min(windows) if windows else -1,
                next(iter(windows.values())).region if windows else Region((), ()),
            )
        changed = set(grouped) | set(repaired)
        return tuple(
            ResidentChange(
                window,
                tuple(grouped.get(window, ())),
                tuple(repaired.get(window, ())),
            )
            for window in sorted(changed, key=lambda item: item.level, reverse=True)
        )

    def discard(self, keys: Collection[TileKey]) -> None:
        """Forget logical tile ownership without changing window storage."""
        with self.lock:
            for window in _unique_windows(self.active, self.pending or {}):
                for key in keys:
                    window.key_regions.pop(key, None)

    def complete(self, plan: Plan) -> ResidentTransition:
        """Promote the target window and retire coarse or replaced storage."""
        with self.lock:
            return self._complete(plan)

    def _complete(self, plan: Plan) -> ResidentTransition:
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
        with self.lock:
            retired = tuple(_unique_windows(self.active, self.pending or {}))
            self.active = {}
            self.pending = None
            return ResidentTransition(retired=retired)

    def _allocate(self, level: int, region: Region) -> ResidentWindow:
        info = self.pyramid.levels[level]
        started = perf_counter()
        data = np.full(
            region.shape,
            info.fill_value,
            dtype=self.dtypes[level],
        )
        self._timing("allocation_fill", started, data.nbytes, level, region)
        return ResidentWindow(level, region, data, info.voxel_to_world)

    def _timing(
        self,
        operation: str,
        started: float,
        bytes_processed: int,
        level: int,
        region: Region,
    ) -> None:
        if self.on_timing is not None:
            self.on_timing(
                operation,
                perf_counter() - started,
                bytes_processed,
                level,
                region,
            )


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


def _compose_windows(
    pyramid: Pyramid,
    windows: dict[int, ResidentWindow],
    *,
    source_levels: Collection[int] | None = None,
) -> dict[ResidentWindow, tuple[Region, ...]]:
    """Fill unloaded fine chunks from the most detailed available backdrop."""

    changed: dict[ResidentWindow, list[Region]] = defaultdict(list)
    sources = windows if source_levels is None else source_levels
    for source_level in sorted(sources, reverse=True):
        source = windows[source_level]
        if not source.key_regions:
            continue
        for destination_level in sorted(
            (level for level in windows if level < source_level), reverse=True
        ):
            destination = windows[destination_level]
            content = nearest_resample_region(
                source.data,
                source.region,
                source.transform,
                destination.region,
                destination.transform,
            )
            repaired = fill_unloaded_chunks(
                destination.data,
                destination.region,
                content,
                destination.region,
                tuple(
                    pyramid.levels[destination_level].chunk_sizes(axis)
                    for axis in range(pyramid.ndim)
                ),
                destination.key_regions.values(),
            )
            if repaired:
                changed[destination].extend(repaired)
    return {window: tuple(regions) for window, regions in changed.items()}


def _unique_windows(*collections: dict[int, ResidentWindow]) -> list[ResidentWindow]:
    unique: dict[int, ResidentWindow] = {}
    for windows in collections:
        unique.update({id(window): window for window in windows.values()})
    return list(unique.values())
