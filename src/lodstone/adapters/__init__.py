"""Optional viewer adapters built on Lodstone's core protocols."""

from .napari import (
    ArraySource,
    NapariController,
    add_lodstone_diagnostics,
    add_lodstone_image,
    add_lodstone_labels,
)
from .ndv import NDVController, NDVPublication, NDVTarget

__all__ = [
    "ArraySource",
    "NDVController",
    "NDVPublication",
    "NDVTarget",
    "NapariController",
    "add_lodstone_diagnostics",
    "add_lodstone_image",
    "add_lodstone_labels",
]
