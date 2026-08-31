"""Compatibility names for the renderer-neutral dense host adapter.

ndv's ``ArrayCanvas`` is the reference implementation of these protocols, but
the implementation itself is host neutral.  Existing imports keep working while
new hosts can depend on the explicit ``Dense*`` names from ``adapters.dense``.
"""

from .dense import (
    DenseController,
    DensePreparation,
    DensePublication,
    DenseTarget,
)

NDVController = DenseController
NDVPreparation = DensePreparation
NDVPublication = DensePublication
NDVTarget = DenseTarget

__all__ = [
    "NDVController",
    "NDVPreparation",
    "NDVPublication",
    "NDVTarget",
]
