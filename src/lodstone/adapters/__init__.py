"""Optional viewer adapters built on Lodstone's core protocols."""

from .dense import (
    CameraDenseCanvas,
    CameraSignal,
    DenseCanvas,
    DenseController,
    DenseHandle,
    DensePreparation,
    DensePublication,
    DenseTarget,
)
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
    "CameraDenseCanvas",
    "CameraSignal",
    "DenseCanvas",
    "DenseController",
    "DenseHandle",
    "DensePreparation",
    "DensePublication",
    "DenseTarget",
    "NDVController",
    "NDVPublication",
    "NDVTarget",
    "NapariController",
    "add_lodstone_diagnostics",
    "add_lodstone_image",
    "add_lodstone_labels",
]
