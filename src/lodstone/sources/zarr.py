"""Plain Zarr pyramid source."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..model import identity_transform
from .array import ArrayPyramidSource


class ZarrPyramidSource(ArrayPyramidSource):
    """A pyramid backed by explicitly selected Zarr arrays."""

    @classmethod
    def open(
        cls,
        store: Any,
        *,
        paths: Sequence[str],
        axes: Sequence[str] | None = None,
        transforms: Sequence[np.ndarray] | None = None,
    ) -> ZarrPyramidSource:
        try:
            import zarr
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError("install lodstone[zarr] to open Zarr sources") from error

        group = zarr.open_group(store=store, mode="r")
        arrays: list[Any] = [cast(Any, group[path]) for path in paths]
        if transforms is None:
            ndim = len(arrays[0].shape)
            transforms = [identity_transform(ndim) for _ in arrays]
        return cls(arrays, axes=axes, transforms=transforms)
