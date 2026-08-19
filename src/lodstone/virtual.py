"""Bounded in-memory array views used by progressive loading.

This module provides array-like objects that present the full shape of a
(potentially enormous) array while only keeping a small, chunk-aligned
region of it in memory:

- :class:`VirtualData` wraps a single scale level and exposes the common lazy
  array protocol (``shape``, ``dtype``, ``ndim``, ``size``, ``__getitem__``).
  Reads outside the in-memory interval return the configured fill value.
- :class:`VirtualArrayView` is the lazy result of indexing a
  :class:`VirtualData`. It records the requested region in absolute
  coordinates and only materializes (zero-padded) data when converted with
  ``np.asarray``. This lets clients compose multiple indexing operations
  (e.g. first a displayed-axis crop, then a fixed-axis point selection)
  before any data is copied.
- :class:`MultiScaleVirtualData` coordinates one :class:`VirtualData` per
  scale level and can initialize newly exposed regions from a coarser,
  fully resident level so the canvas never shows empty space while chunks
  stream in.

All hyperslice accesses are guarded by a per-``VirtualData`` reentrant
lock so background fetch threads and the main thread can safely interleave.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from .chunks import (
    chunk_boundaries,
    chunk_ids_in_region,
    chunk_shape_for,
    chunk_sizes_for,
)
from .composition import fill_unloaded_chunks, nearest_resample_region
from .model import Region

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MultiScaleVirtualData",
    "VirtualArrayView",
    "VirtualData",
    "chunk_boundaries",
    "chunk_ids_in_region",
    "chunk_shape_for",
    "chunk_sizes_for",
]


class VirtualArrayView:
    """A lazy, zero-padded view into a :class:`VirtualData`.

    The view tracks the requested region in the *absolute* coordinates of
    the wrapped array. Indexing a view returns another view (with the keys
    composed), and ``np.asarray`` materializes the data: regions inside the
    currently resident interval are copied from the hyperslice and
    everything else is zero.
    """

    def __init__(
        self,
        data: VirtualData,
        index: tuple[int | tuple[int, int], ...],
    ):
        # index has one entry per dimension of ``data``:
        # an int collapses the dimension, a (start, stop) pair keeps it.
        self._data = data
        self._index = index

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            stop - start
            for entry in self._index
            if isinstance(entry, tuple)
            for start, stop in [entry]
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key) -> VirtualArrayView:
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            n_missing = self.ndim - sum(1 for k in key if k is not Ellipsis)
            expanded: list = []
            for k in key:
                if k is Ellipsis:
                    expanded.extend([slice(None)] * n_missing)
                else:
                    expanded.append(k)
            key = tuple(expanded)

        out: list[int | tuple[int, int]] = []
        key_pos = 0
        for entry in self._index:
            if isinstance(entry, int):
                out.append(entry)
                continue
            start, stop = entry
            n = stop - start
            k = key[key_pos] if key_pos < len(key) else slice(None)
            key_pos += 1
            if isinstance(k, slice):
                k_start, k_stop, k_step = k.indices(n)
                if k_step != 1:
                    raise IndexError(
                        f"VirtualArrayView only supports step-1 slices, got {k!r}",
                    )
                out.append((start + k_start, start + max(k_stop, k_start)))
            elif isinstance(k, (int, np.integer)):
                idx = int(k)
                if idx < 0:
                    idx += n
                if not 0 <= idx < n:
                    raise IndexError(
                        f"index {k} is out of bounds for dimension of size {n}",
                    )
                out.append(start + idx)
            else:
                raise IndexError(
                    f"unsupported index for VirtualArrayView: {k!r}",
                )
        if key_pos < len(key):
            raise IndexError(
                f"too many indices: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed",
            )
        return VirtualArrayView(self._data, tuple(out))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        data = self._data
        out = np.full(self.shape, data.fill_value, dtype=data.dtype)
        with data.lock:
            resident_min = data._min_coord
            resident_max = data._max_coord
            if resident_min is not None and resident_max is not None:
                src_key: list = []
                dst_key: list = []
                inside = True
                for dim, entry in enumerate(self._index):
                    min_c = resident_min[dim]
                    max_c = resident_max[dim]
                    if isinstance(entry, int):
                        if not min_c <= entry < max_c:
                            inside = False
                            break
                        src_key.append(entry - min_c)
                    else:
                        start, stop = entry
                        lo = max(start, min_c)
                        hi = min(stop, max_c)
                        if hi <= lo:
                            inside = False
                            break
                        src_key.append(slice(lo - min_c, hi - min_c))
                        dst_key.append(slice(lo - start, hi - start))
                if inside:
                    out[tuple(dst_key)] = data.hyperslice[tuple(src_key)]
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def transpose(self, axes=None) -> np.ndarray:
        """Materialize and transpose (numpy compatibility)."""
        return np.asarray(self).transpose(axes)

    def __repr__(self) -> str:
        return (
            f"<VirtualArrayView shape={self.shape} dtype={self.dtype} "
            f"index={self._index}>"
        )


class VirtualData:
    """Present a full-size array while holding only one region in memory.

    ``VirtualData`` wraps one scale level's array (zarr, dask, numpy, ...).
    It reports the wrapped array's full ``shape``/``dtype`` but only stores
    the data within the current *interval* (set via :meth:`set_interval`) in
    an in-memory numpy array called the *hyperslice*. The interval is always
    aligned outward to chunk boundaries of the wrapped array.

    Indexing uses the wrapped array's (absolute) coordinates and returns a
    lazy :class:`VirtualArrayView`; reads outside the interval materialize
    as zeros. Writes go through :meth:`set_offset`, which clips the value to
    the resident interval.

    Attributes
    ----------
    array : array-like
        The wrapped array (not copied; only read from).
    hyperslice : np.ndarray
        In-memory data for the current interval.
    translate : tuple of int
        Offset of the hyperslice origin from the wrapped array's origin.
    lock : threading.RLock
        Guards ``hyperslice``/interval mutation and access.

    """

    def __init__(self, array, scale_level: int = 0, fill_value=0):
        self.array = array
        self.dtype = np.dtype(array.dtype)
        self.shape = tuple(int(s) for s in array.shape)
        self.ndim = len(self.shape)
        self.scale_level = scale_level
        self.fill_value = fill_value

        self.lock = threading.RLock()
        self.translate: tuple[int, ...] = (0,) * self.ndim
        self.hyperslice = np.full((0,) * self.ndim, fill_value, dtype=self.dtype)
        self._min_coord: list[int] | None = None
        self._max_coord: list[int] | None = None
        self._chunk_grid = chunk_sizes_for(array)
        self._boundaries = chunk_boundaries(array)
        self._chunk_shape = chunk_shape_for(array)
        # Chunk keys (tuples of (start, stop) pairs) whose data is resident
        # in the hyperslice. Maintained by the progressive loader.
        self.loaded_chunks: set[tuple[tuple[int, int], ...]] = set()
        # Best-effort provenance per resident chunk: the scale level the
        # chunk's current content came from (this level = real data; a
        # coarser level = upsampled backdrop). Feeds debugging overlays;
        # absent entries mean unknown origin (typically still zeros).
        self.chunk_source: dict[tuple[tuple[int, int], ...], int] = {}

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        # cached in __init__: the wrapped array is fixed, and this is read
        # per-level on the interactive interval-sizing path
        return self._chunk_shape

    @property
    def interval(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Current resident interval as ``(min_coord, max_coord)`` or None."""
        with self.lock:
            resident_min = self._min_coord
            resident_max = self._max_coord
            if resident_min is None or resident_max is None:
                return None
            return tuple(resident_min), tuple(resident_max)

    def chunk_aligned_interval(
        self,
        min_coord,
        max_coord,
    ) -> tuple[list[int], list[int]]:
        """Expand ``[min_coord, max_coord)`` outward to chunk boundaries."""
        lo: list[int] = []
        hi: list[int] = []
        for dim in range(self.ndim):
            bounds = self._boundaries[dim]
            min_c = int(np.clip(min_coord[dim], 0, self.shape[dim]))
            max_c = int(np.clip(max_coord[dim], 0, self.shape[dim]))
            i = max(int(np.searchsorted(bounds, min_c, side="right")) - 1, 0)
            j = int(np.searchsorted(bounds, max_c, side="left"))
            j = min(max(j, i + 1), len(bounds) - 1)
            lo.append(int(bounds[i]))
            hi.append(int(bounds[j]))
        return lo, hi

    def covers(self, min_coord, max_coord) -> bool:
        """Return True if the resident interval covers the given region."""
        with self.lock:
            resident_min = self._min_coord
            resident_max = self._max_coord
            if resident_min is None or resident_max is None:
                return False
            return all(
                resident_min[d] <= int(min_coord[d])
                and int(max_coord[d]) <= resident_max[d]
                for d in range(self.ndim)
            )

    def set_interval(
        self,
        min_coord,
        max_coord,
        backdrop=None,
        backdrop_source: int | None = None,
    ) -> None:
        """Set the resident interval, preserving overlapping data.

        The interval is expanded outward to chunk boundaries. Data that
        overlaps the previous interval is carried over; newly exposed
        regions are initialized from ``backdrop`` (if given) or zeros.

        Parameters
        ----------
        min_coord, max_coord : sequence of int
            Requested interval in the wrapped array's coordinates
            (half-open, like slices).
        backdrop : callable, optional
            ``backdrop(min_coord, max_coord) -> np.ndarray | None`` returning
            initial content for the new hyperslice (e.g. upsampled data from
            a coarser scale). Called with the chunk-aligned interval.
        backdrop_source : int, optional
            Scale level the backdrop content comes from; recorded in
            ``chunk_source`` (debug provenance) for chunks the backdrop
            initialized.

        """
        new_min, new_max = self.chunk_aligned_interval(min_coord, max_coord)
        with self.lock:
            prev_min = self._min_coord
            prev_hyperslice = self.hyperslice

            if (
                prev_min is not None
                and new_min == prev_min
                and new_max == self._max_coord
            ):
                return

            new_shape = [mx - mn for mn, mx in zip(new_min, new_max, strict=True)]
            next_hyperslice = None
            if backdrop is not None:
                try:
                    content = backdrop(new_min, new_max)
                except Exception:  # pragma: no cover - backdrop is advisory
                    LOGGER.exception("backdrop initialization failed")
                    content = None
                if content is not None and tuple(content.shape) == tuple(
                    new_shape,
                ):
                    next_hyperslice = np.ascontiguousarray(
                        content,
                        dtype=self.dtype,
                    )
            used_backdrop = next_hyperslice is not None
            if next_hyperslice is None:
                next_hyperslice = np.full(new_shape, self.fill_value, dtype=self.dtype)

            # Carry over data overlapping the previous interval to avoid
            # re-fetching and visual flashing.
            if prev_min is not None and prev_hyperslice.size:
                src_key = []
                dst_key = []
                overlaps = True
                for dim in range(self.ndim):
                    lo = max(prev_min[dim], new_min[dim])
                    hi = min(
                        prev_min[dim] + prev_hyperslice.shape[dim],
                        new_max[dim],
                    )
                    if hi <= lo:
                        overlaps = False
                        break
                    src_key.append(
                        slice(lo - prev_min[dim], hi - prev_min[dim]),
                    )
                    dst_key.append(slice(lo - new_min[dim], hi - new_min[dim]))
                if overlaps:
                    next_hyperslice[tuple(dst_key)] = prev_hyperslice[tuple(src_key)]

            self._min_coord = new_min
            self._max_coord = new_max
            self.translate = tuple(new_min)
            self.hyperslice = next_hyperslice

            # Drop bookkeeping for chunks that fell outside the interval.
            self.loaded_chunks = {
                key
                for key in self.loaded_chunks
                if all(
                    new_min[d] <= start and stop <= new_max[d]
                    for d, (start, stop) in enumerate(key)
                )
            }
            self.chunk_source = {
                key: src
                for key, src in self.chunk_source.items()
                if all(
                    new_min[d] <= start and stop <= new_max[d]
                    for d, (start, stop) in enumerate(key)
                )
            }
            if backdrop_source is not None and used_backdrop:
                # debug provenance: chunks not carried over start out
                # showing the backdrop source's (upsampled) content
                for chunk_id in chunk_ids_in_region(
                    self._boundaries,
                    new_min,
                    new_max,
                ):
                    self.chunk_source.setdefault(chunk_id, backdrop_source)

    def set_offset(self, key: tuple[slice, ...], value) -> None:
        """Write ``value`` at ``key`` (absolute coordinates), clipped.

        Regions of ``key`` outside the resident interval are ignored.
        """
        value = np.asarray(value)
        with self.lock:
            resident_min = self._min_coord
            resident_max = self._max_coord
            if resident_min is None or resident_max is None:
                return
            dst_key = []
            src_key = []
            for dim, sl in enumerate(key):
                lo = max(int(sl.start), resident_min[dim])
                hi = min(int(sl.stop), resident_max[dim])
                if hi <= lo:
                    return
                dst_key.append(
                    slice(lo - resident_min[dim], hi - resident_min[dim]),
                )
                src_key.append(slice(lo - int(sl.start), hi - int(sl.start)))
            expected = tuple(s.stop - s.start for s in src_key)
            if value[tuple(src_key)].shape != expected:
                LOGGER.warning(
                    "set_offset: value shape %s does not cover key %s",
                    value.shape,
                    key,
                )
                return
            self.hyperslice[tuple(dst_key)] = value[tuple(src_key)]

    def set_chunk(
        self,
        key: tuple[slice, ...],
        value,
        source_level: int,
    ) -> bool:
        """Atomically write a resident chunk and record its provenance.

        Backdrop repair uses ``loaded_chunks`` while holding ``lock`` to
        decide which pixels it may replace.  Keeping the pixel write and
        that bookkeeping in one critical section prevents a repair from
        overwriting a newly loaded fine chunk in between those operations.

        Returns ``False`` when an interval change made the complete chunk
        stale before it could be applied.
        """
        chunk_id = tuple((int(sl.start), int(sl.stop)) for sl in key)
        with self.lock:
            resident_min = self._min_coord
            resident_max = self._max_coord
            if (
                resident_min is None
                or resident_max is None
                or any(
                    int(sl.start) < resident_min[dim]
                    or int(sl.stop) > resident_max[dim]
                    for dim, sl in enumerate(key)
                )
            ):
                return False
            self.set_offset(key, value)
            self.loaded_chunks.add(chunk_id)
            self.chunk_source[chunk_id] = int(source_level)
            return True

    def __getitem__(self, key) -> VirtualArrayView:
        full = tuple((0, s) for s in self.shape)
        return VirtualArrayView(self, full)[key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return VirtualArrayView(
            self,
            tuple((0, s) for s in self.shape),
        ).__array__(dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"<VirtualData scale_level={self.scale_level} shape={self.shape} "
            f"dtype={self.dtype} interval={self.interval}>"
        )


class MultiScaleVirtualData:
    """Coordinate a :class:`VirtualData` per level of a multiscale image.

    The list of per-level :class:`VirtualData` objects (``._data``) follows
    the common multiscale array protocol. The object also tracks the scale
    factors between levels and can initialize a level's freshly exposed
    region from a coarser level's resident data
    (:meth:`backdrop_for`), which keeps the canvas filled with low-resolution
    content while high-resolution chunks stream in.

    Parameters
    ----------
    arrays : sequence of array-like
        Multiscale levels from highest resolution (index 0) to lowest.

    """

    def __init__(self, arrays: Sequence, fill_value=0, transforms=None):
        if len(arrays) == 0:
            raise ValueError("arrays must be a non-empty sequence")
        self.arrays = list(arrays)
        self._data = [
            VirtualData(array, scale_level=level, fill_value=fill_value)
            for level, array in enumerate(self.arrays)
        ]
        highest_res = self._data[0]
        self.dtype = highest_res.dtype
        self.shape = highest_res.shape
        self.ndim = highest_res.ndim

        # Per-level, per-dimension downsampling factor relative to level 0.
        self._scale_factors = [
            [
                hr_dim / level_dim
                for hr_dim, level_dim in zip(
                    highest_res.shape,
                    vdata.shape,
                    strict=True,
                )
            ]
            for vdata in self._data
        ]
        if transforms is None:
            transforms = []
            for factors in self._scale_factors:
                transform = np.eye(self.ndim + 1, dtype=float)
                transform[:-1, :-1] = np.diag(factors)
                transforms.append(transform)
        if len(transforms) != len(self._data):
            raise ValueError("one transform is required per pyramid level")
        self._level_transforms = [
            np.asarray(transform, dtype=float) for transform in transforms
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, level: int) -> VirtualData:
        return self._data[level]

    def __iter__(self):
        return iter(self._data)

    @property
    def levels(self) -> int:
        return len(self._data)

    def backdrop_for(self, level: int, src_level: int):
        """Return a backdrop callable that upsamples from ``src_level``.

        The returned function is suitable for
        :meth:`VirtualData.set_interval`'s ``backdrop`` parameter; it samples
        the source level's resident hyperslice with nearest-neighbor
        interpolation. Returns ``None`` when no useful backdrop exists.
        """
        if src_level == level:
            return None
        src = self._data[src_level]

        def backdrop(min_coord, max_coord):
            with src.lock:
                resident_min = src._min_coord
                resident_max = src._max_coord
                if (
                    resident_min is None
                    or resident_max is None
                    or src.hyperslice.size == 0
                ):
                    return None
                return nearest_resample_region(
                    src.hyperslice,
                    Region(tuple(resident_min), tuple(resident_max)),
                    self._level_transforms[src_level],
                    Region(tuple(min_coord), tuple(max_coord)),
                    self._level_transforms[level],
                )

        return backdrop

    def set_interval(
        self,
        level: int,
        min_coord,
        max_coord,
        backdrop_level: int | None = None,
    ) -> None:
        """Set the resident interval for ``level`` (in level coordinates).

        If ``backdrop_level`` is given, newly exposed regions are initialized
        with nearest-neighbor upsampled data from that level.
        """
        backdrop = (
            self.backdrop_for(level, backdrop_level)
            if backdrop_level is not None
            else None
        )
        self._data[level].set_interval(
            min_coord,
            max_coord,
            backdrop=backdrop,
            backdrop_source=backdrop_level if backdrop is not None else None,
        )

    def fill_unloaded_from(
        self,
        level: int,
        src_level: int,
        region=None,
    ) -> bool:
        """Fill not-yet-loaded chunk regions of ``level`` from ``src_level``.

        Used to repair the backdrop when the source level finished loading
        only after the destination level's interval had been initialized
        (which would otherwise leave zeros on screen until real chunks
        arrive). Already-loaded chunks are left untouched.

        Parameters
        ----------
        region : tuple of (min_coord, max_coord), optional
            Restrict the repair to this absolute-coordinate region (e.g.
            the currently rendered tile); the upsampling gather scales
            with the region size.

        Returns True if anything was written.

        """
        backdrop = self.backdrop_for(level, src_level)
        if backdrop is None:
            return False
        dst = self._data[level]
        with dst.lock:
            resident_min = dst._min_coord
            resident_max = dst._max_coord
            if resident_min is None or resident_max is None or dst.hyperslice.size == 0:
                return False
            fill_min = list(resident_min)
            fill_max = list(resident_max)
            if region is not None:
                fill_min = [
                    min(max(int(r), lo), hi)
                    for r, lo, hi in zip(region[0], fill_min, fill_max, strict=True)
                ]
                fill_max = [
                    min(max(int(r), lo), hi)
                    for r, lo, hi in zip(region[1], resident_min, fill_max, strict=True)
                ]
                if any(mx <= mn for mn, mx in zip(fill_min, fill_max, strict=True)):
                    return False
            content = backdrop(fill_min, fill_max)
            expected = tuple(mx - mn for mn, mx in zip(fill_min, fill_max, strict=True))
            if content is None or tuple(content.shape) != expected:
                return False
            filled = fill_unloaded_chunks(
                dst.hyperslice,
                Region(tuple(resident_min), tuple(resident_max)),
                content,
                Region(tuple(fill_min), tuple(fill_max)),
                dst._chunk_grid,
                {
                    Region(
                        tuple(start for start, _stop in chunk_id),
                        tuple(stop for _start, stop in chunk_id),
                    )
                    for chunk_id in dst.loaded_chunks
                },
            )
            for chunk in filled:
                chunk_id = tuple(zip(chunk.start, chunk.stop, strict=True))
                dst.chunk_source[chunk_id] = src_level
            return bool(filled)
