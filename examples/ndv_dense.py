"""Display a progressive dense volume through ndv's VisPy or pygfx canvas.

Install ``ndv[qt]`` (or another ndv frontend extra) alongside Lodstone before
running this example.
"""

import ndv
import numpy as np

from lodstone import View
from lodstone.adapters.ndv import NDVController
from lodstone.sources import ArrayPyramidSource

z, y, x = np.indices((64, 96, 128))
fine = np.asarray((z - 32) ** 2 + (y - 48) ** 2 + (x - 64) ** 2, dtype=np.uint16)
coarse = fine[::2, ::2, ::2]
source = ArrayPyramidSource(
    [fine, coarse],
    axes=("z", "y", "x"),
    transforms=[np.eye(4), np.diag([2.0, 2.0, 2.0, 1.0])],
    chunks=[(32, 32, 32), (32, 32, 32)],
)

viewer = ndv.ArrayViewer()
controller = NDVController(viewer, source)
world_to_clip = np.diag([2 / 64, 2 / 96, 2 / 128, 1.0])
world_to_clip[:-1, -1] = -1
controller.update(
    View(
        displayed_axes=(0, 1, 2),
        index=(None, None, None),
        viewport=(800, 600),
        world_to_clip=world_to_clip,
    )
)
viewer.show()
ndv.run_app()
controller.close()
