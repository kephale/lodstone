"""OME-Zarr multiscale source adapter."""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

from .array import ArrayPyramidSource


class OMEZarrSource(ArrayPyramidSource):
    """Open the first or a selected OME-Zarr multiscale image."""

    @classmethod
    def open(
        cls,
        store: Any,
        *,
        multiscale: int = 0,
    ) -> OMEZarrSource:
        try:
            import zarr
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError("install lodstone[ome-zarr] to open OME-Zarr") from error

        resolved_store = _resolve_store(store)
        group = zarr.open_group(store=resolved_store, mode="r")
        attributes: dict[str, Any] = dict(group.attrs)
        entries = cast(list[dict[str, Any]] | None, attributes.get("multiscales"))
        if not entries:
            raise ValueError("Zarr group does not contain OME multiscales metadata")
        try:
            metadata = entries[multiscale]
        except IndexError as error:
            raise ValueError(f"multiscale index {multiscale} does not exist") from error

        datasets = cast(list[dict[str, Any]], metadata.get("datasets", ()))
        if not datasets:
            raise ValueError("OME multiscale metadata does not contain datasets")
        arrays: list[Any] = [cast(Any, group[item["path"]]) for item in datasets]
        ndim = len(arrays[0].shape)
        axes = tuple(
            _axis_name(axis, i) for i, axis in enumerate(metadata.get("axes", ()))
        )
        if len(axes) != ndim:
            axes = tuple(f"axis_{i}" for i in range(ndim))
        common_transforms = tuple(metadata.get("coordinateTransformations", ()))
        transforms = [
            _coordinate_transform(
                (*dataset.get("coordinateTransformations", ()), *common_transforms),
                ndim,
            )
            for dataset in datasets
        ]
        return cls(arrays, axes=axes, transforms=transforms)


def _resolve_store(store: Any) -> Any:
    if not isinstance(store, str):
        return store
    parsed = urlparse(store)
    if parsed.scheme not in {"http", "https"}:
        return store
    try:
        import fsspec
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("remote OME-Zarr requires lodstone[ome-zarr]") from error
    return fsspec.get_mapper(store)


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
