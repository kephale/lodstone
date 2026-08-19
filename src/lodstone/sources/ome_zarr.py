"""OME-Zarr and conventional multiscale Zarr source adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

from .array import ArrayPyramidSource, FixedAxisArray


class OMEZarrSource(ArrayPyramidSource):
    """Open a nested OME-Zarr image or a conventional array pyramid."""

    @classmethod
    def open(
        cls,
        store: Any,
        *,
        multiscale: int = 0,
        num_levels: int | None = None,
        zarr_format: int | None = None,
        storage_options: Mapping[str, Any] | None = None,
        cache_bytes: int | None = None,
        search_depth: int = 4,
        fixed_index: Mapping[int | str, int] | None = None,
        squeeze: bool = False,
    ) -> OMEZarrSource:
        """Open a multiscale source without materializing its arrays.

        Metadata discovery accepts OME-Zarr v0.1-v0.4 ``multiscales`` and
        the v0.5 ``ome.multiscales`` wrapper. If metadata is absent, sorted
        child arrays are treated as a pyramid and transforms are inferred
        from their shapes.
        """

        try:
            import zarr
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError("install lodstone[ome-zarr] to open OME-Zarr") from error

        resolved = _resolve_store(
            store,
            storage_options=storage_options,
            cache_bytes=cache_bytes,
        )
        open_kwargs: dict[str, Any] = {"store": resolved, "mode": "r"}
        if zarr_format is not None:
            open_kwargs["zarr_format"] = zarr_format
        root = zarr.open_group(**open_kwargs)
        group, entries = _find_multiscales(root, max_depth=search_depth)

        metadata: dict[str, Any] | None = None
        if entries:
            try:
                metadata = cast(dict[str, Any], entries[multiscale])
            except IndexError as error:
                raise ValueError(
                    f"multiscale index {multiscale} does not exist"
                ) from error
        else:
            group = root

        datasets = [] if metadata is None else list(metadata.get("datasets", ()))
        if not datasets:
            datasets = [
                {"path": key}
                for key in sorted(group.keys(), key=lambda key: (len(key), key))
                if _is_array(group, key)
            ]
        if num_levels is not None:
            if num_levels <= 0:
                raise ValueError("num_levels must be positive")
            datasets = datasets[:num_levels]
        if not datasets:
            raise ValueError("Zarr group does not contain pyramid arrays")

        arrays: list[Any] = [cast(Any, group[item["path"]]) for item in datasets]
        ndim = len(arrays[0].shape)
        if any(len(array.shape) != ndim for array in arrays):
            raise ValueError("all pyramid arrays must have equal dimensionality")

        raw_axes = () if metadata is None else tuple(metadata.get("axes", ()))
        axes = tuple(_axis_name(axis, i) for i, axis in enumerate(raw_axes))
        if len(axes) != ndim:
            axes = tuple(f"axis_{i}" for i in range(ndim))

        common = () if metadata is None else tuple(
            metadata.get("coordinateTransformations", ())
        )
        transforms = []
        finest_shape = np.asarray(arrays[0].shape, dtype=np.float64)
        for level, (dataset, array) in enumerate(zip(datasets, arrays, strict=True)):
            items = (*dataset.get("coordinateTransformations", ()), *common)
            if items:
                transform = _coordinate_transform(items, ndim)
            else:
                transform = np.eye(ndim + 1, dtype=np.float64)
                if level:
                    transform[np.arange(ndim), np.arange(ndim)] = (
                        finest_shape / np.asarray(array.shape, dtype=np.float64)
                    )
            transforms.append(transform)

        fixed = _fixed_axes(fixed_index, axes)
        if squeeze:
            for axis, size in enumerate(arrays[0].shape):
                if len(arrays[0].shape) - len(fixed) <= 2:
                    break
                if int(size) == 1:
                    fixed.setdefault(axis, 0)
                else:
                    break
        if fixed:
            keep = tuple(axis for axis in range(ndim) if axis not in fixed)
            arrays = [FixedAxisArray(array, fixed) for array in arrays]
            transforms = [
                _reduced_transform(transform, keep, fixed) for transform in transforms
            ]
            axes = tuple(axes[axis] for axis in keep)

        return cls(arrays, axes=axes, transforms=transforms)


def _resolve_store(
    store: Any,
    *,
    storage_options: Mapping[str, Any] | None,
    cache_bytes: int | None,
) -> Any:
    if not isinstance(store, str):
        resolved = store
    else:
        scheme = urlparse(store).scheme
        if scheme not in {"http", "https", "s3", "gs"}:
            resolved = store
        else:
            try:
                from zarr.storage import FsspecStore
            except ImportError as error:  # pragma: no cover
                raise ImportError(
                    "remote OME-Zarr requires lodstone[ome-zarr]"
                ) from error
            options = dict(storage_options or {})
            if scheme == "s3":
                options.setdefault("anon", True)
            resolved = FsspecStore.from_url(store, storage_options=options)
    if cache_bytes is None:
        return resolved
    if cache_bytes <= 0:
        raise ValueError("cache_bytes must be positive")
    try:
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import MemoryStore
    except ImportError as error:  # pragma: no cover
        raise ImportError("Zarr cache support requires zarr>=3") from error
    return CacheStore(
        resolved,  # pyright: ignore[reportArgumentType]
        cache_store=MemoryStore(),
        max_size=int(cache_bytes),
    )


def _find_multiscales(group: Any, *, max_depth: int, depth: int = 0):
    attrs = dict(group.attrs)
    ome = attrs.get("ome")
    entries = attrs.get("multiscales") or (
        ome.get("multiscales") if isinstance(ome, dict) else None
    )
    if entries:
        return group, list(entries)
    if depth >= max_depth:
        return group, []
    for key in group:
        try:
            child = group[key]
        except (KeyError, OSError, ValueError):
            continue
        if not hasattr(child, "attrs") or not hasattr(child, "keys"):
            continue
        found, entries = _find_multiscales(
            child,
            max_depth=max_depth,
            depth=depth + 1,
        )
        if entries:
            return found, entries
    return group, []


def _is_array(group: Any, key: str) -> bool:
    try:
        value = group[key]
    except (KeyError, OSError, ValueError):
        return False
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _fixed_axes(
    fixed_index: Mapping[int | str, int] | None,
    axes: Sequence[str],
) -> dict[int, int]:
    result = {}
    for axis, index in (fixed_index or {}).items():
        if isinstance(axis, str):
            try:
                resolved = axes.index(axis)
            except ValueError as error:
                raise ValueError(f"unknown fixed axis {axis!r}") from error
        else:
            resolved = int(axis)
        result[resolved] = int(index)
    return result


def _axis_name(axis: Any, index: int) -> str:
    if isinstance(axis, str):
        return axis
    if isinstance(axis, dict):
        return str(axis.get("name", f"axis_{index}"))
    return f"axis_{index}"


def _coordinate_transform(items: Any, ndim: int) -> np.ndarray:
    result = np.eye(ndim + 1, dtype=np.float64)
    for item in items:
        kind = item.get("type")
        if kind == "scale":
            values = np.asarray(item["scale"], dtype=np.float64)
            transform = np.eye(ndim + 1, dtype=np.float64)
            transform[np.arange(ndim), np.arange(ndim)] = values
        elif kind == "translation":
            values = np.asarray(item["translation"], dtype=np.float64)
            transform = np.eye(ndim + 1, dtype=np.float64)
            transform[:ndim, ndim] = values
        else:
            continue
        result = transform @ result
    return result


def _reduced_transform(
    transform: np.ndarray,
    keep: Sequence[int],
    fixed: Mapping[int, int],
) -> np.ndarray:
    result = np.eye(len(keep) + 1, dtype=np.float64)
    for row, source_row in enumerate(keep):
        for column, source_column in enumerate(keep):
            result[row, column] = transform[source_row, source_column]
        result[row, -1] = transform[source_row, -1] + sum(
            transform[source_row, axis] * index for axis, index in fixed.items()
        )
    return result
