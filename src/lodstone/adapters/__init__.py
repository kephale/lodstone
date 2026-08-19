"""Optional viewer adapters built on Lodstone's core protocols."""

from .napari import (
    ArraySource,
    NapariController,
    add_lodstone_diagnostics,
    add_lodstone_image,
    add_lodstone_labels,
)

__all__ = [
    "ArraySource",
    "NapariController",
    "add_lodstone_diagnostics",
    "add_lodstone_image",
    "add_lodstone_labels",
]
