"""napari adapter for its progressive multiscale rendering path.

This module deliberately does not assemble streamed chunks into dense NumPy
layers.  It supplies Lodstone-backed lazy arrays to napari's progressive
loader, which owns the viewer-specific resident intervals, single multiscale
layer, texture double buffering, and partial GPU uploads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..source import Source


@runtime_checkable
class ArraySource(Source, Protocol):
    """A Lodstone source that can expose its original lazy array levels."""

    @property
    def arrays(self) -> Sequence[Any]: ...


class _SlicedArray:
    """Lazy fixed-axis view retaining the metadata napari needs."""

    def __init__(
        self,
        array: Any,
        fixed_index: Mapping[int, int],
        chunks: Sequence[int] | None = None,
    ) -> None:
        self._array = array
        self._fixed = dict(fixed_index)
        ndim = len(array.shape)
        if any(axis < 0 or axis >= ndim for axis in self._fixed):
            raise ValueError("fixed axis is outside the source dimensionality")
        if any(
            not 0 <= index < array.shape[axis] for axis, index in self._fixed.items()
        ):
            raise ValueError("fixed index is outside the source shape")
        self._axes = tuple(axis for axis in range(ndim) if axis not in self._fixed)
        if len(self._axes) < 2:
            raise ValueError("napari images require at least two non-fixed axes")
        self.shape = tuple(int(array.shape[axis]) for axis in self._axes)
        self.dtype = np.dtype(array.dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))

        native_chunks = chunks or getattr(array, "chunks", None)
        if native_chunks is not None:
            self.chunks = tuple(native_chunks[axis] for axis in self._axes)
        chunksize = getattr(array, "chunksize", None)
        if chunksize is not None:
            self.chunksize = tuple(chunksize[axis] for axis in self._axes)

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            position = key.index(Ellipsis)
            missing = self.ndim - (len(key) - 1)
            key = (*key[:position], *(slice(None),) * missing, *key[position + 1 :])
        if len(key) > self.ndim:
            raise IndexError("too many indices for fixed-axis array view")
        key = (*key, *(slice(None),) * (self.ndim - len(key)))
        source_key: list[Any] = []
        visible = iter(key)
        for axis in range(len(self._array.shape)):
            source_key.append(
                self._fixed[axis] if axis in self._fixed else next(visible)
            )
        return self._array[tuple(source_key)]


def _progressive_image_factory():
    try:
        from napari.experimental._lodstone_loading import (  # pyright: ignore[reportMissingImports]
            add_lodstone_loading_image,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "the Lodstone napari adapter requires napari's progressive-loading "
            "implementation (napari PR #9067)"
        ) from error
    return add_lodstone_loading_image


def _layer_affine(source: Source, fixed_index: Mapping[int, int]) -> np.ndarray:
    transform = source.pyramid.levels[0].voxel_to_world
    axes = tuple(axis for axis in range(source.pyramid.ndim) if axis not in fixed_index)
    affine = np.eye(len(axes) + 1, dtype=np.float64)
    for row, source_row in enumerate(axes):
        for column, source_column in enumerate(axes):
            affine[row, column] = transform[source_row, source_column]
        affine[row, len(axes)] = transform[source_row, source.pyramid.ndim] + sum(
            transform[source_row, axis] * index for axis, index in fixed_index.items()
        )
    return affine


def add_lodstone_image(
    source: Source,
    viewer: Any = None,
    *,
    fixed_index: Mapping[int, int] | None = None,
    **layer_kwargs: Any,
) -> Any:
    """Add ``source`` as one progressively loaded multiscale napari layer.

    ``source`` must expose an ``arrays`` property containing lazy, indexable
    levels.  Lodstone's array, Zarr, and OME-Zarr sources all do. Fixed axes
    (commonly an OME channel axis) are sliced lazily and removed from the
    napari layer without materializing data.

    The returned layer stores napari's loader in
    ``layer.metadata['progressive_loader']``.  All keyword arguments other
    than ``fixed_index`` pass through to napari's progressive image factory.
    """

    if not isinstance(source, ArraySource):
        raise TypeError(
            "napari progressive rendering requires a source exposing lazy "
            "array levels through an 'arrays' property"
        )
    fixed = dict(fixed_index or {})
    arrays: Sequence[Any] = source.arrays
    if fixed:
        arrays = tuple(
            _SlicedArray(array, fixed, level.chunks)
            for array, level in zip(arrays, source.pyramid.levels, strict=True)
        )

    layer_kwargs.setdefault("affine", _layer_affine(source, fixed))
    factory = _progressive_image_factory()
    return factory(arrays, viewer=viewer, **layer_kwargs)


class NapariController:
    """Lifecycle wrapper around a PR-style progressive napari layer."""

    def __init__(
        self,
        viewer: Any,
        source: Source,
        *,
        fixed_index: Mapping[int, int] | None = None,
        **layer_kwargs: Any,
    ) -> None:
        self.viewer = viewer
        self.source = source
        self.layer = add_lodstone_image(
            source,
            viewer,
            fixed_index=fixed_index,
            **layer_kwargs,
        )
        self.loader = self.layer.metadata["progressive_loader"]

    def close(self) -> None:
        self.loader.close()
