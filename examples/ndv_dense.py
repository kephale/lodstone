"""Display a real Zebrahub timepoint through ndv's dense 3-D canvas.

The initial full-volume view selects the coarsest source level, whose roughly
74 MiB dense array fits comfortably in the default ndv target budget. Finer,
translated camera windows await a public ndv canvas translation API.
"""

from __future__ import annotations

import argparse

import ndv
import numpy as np

from lodstone import Plan, View
from lodstone.adapters.ndv import NDVController, NDVPublication
from lodstone.sources import OMEZarrSource

DEFAULT_URL = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/"
    "single-objective/ZSNS001.ome.zarr"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--time", type=int, default=400)
    arguments = parser.parse_args()

    print(f"Opening Zebrahub timepoint {arguments.time} from {arguments.url}")
    source = OMEZarrSource.open(
        arguments.url,
        fixed_index={"t": arguments.time, "c": 0},
    )
    finest = source.pyramid.levels[0]
    linear = finest.voxel_to_world[:-1, :-1]
    world_extent = np.abs(linear) @ np.asarray(finest.shape, dtype=np.float64)
    world_to_clip = np.eye(4, dtype=np.float64)
    world_to_clip[np.arange(3), np.arange(3)] = 2.0 / world_extent
    world_to_clip[:-1, -1] = -1.0

    viewer = ndv.ArrayViewer()
    widget = viewer.widget()
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel

    indicator = QLabel("Lodstone · waiting", widget)
    indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    indicator.setStyleSheet(
        "background: rgba(0, 0, 0, 190); color: white; "
        "padding: 6px 9px; border-radius: 4px; font-weight: bold;"
    )
    indicator.move(12, 12)
    indicator.adjustSize()
    indicator.show()
    indicator.raise_()
    active_level: int | None = None
    target_level: int | None = None

    def update_indicator() -> None:
        active = "waiting" if active_level is None else f"L{active_level}"
        target = "?" if target_level is None else f"L{target_level}"
        indicator.setText(f"Lodstone · active {active} · target {target}")
        indicator.adjustSize()
        indicator.raise_()

    def show_presented_level(publication: NDVPublication) -> None:
        nonlocal active_level
        active_level = publication.level
        spacing = " × ".join(f"{scale:g}" for scale in publication.scales)
        title = f"Lodstone Zebrahub — rendered L{publication.level} ({spacing})"
        print(title)
        widget.window().setWindowTitle(title)
        update_indicator()

    def show_target_level(plan: Plan) -> None:
        nonlocal target_level
        target_level = plan.target_level
        update_indicator()

    controller = NDVController(
        viewer,
        source,
        on_presented=show_presented_level,
        on_targeted=show_target_level,
    )
    plan = controller.update(
        View(
            displayed_axes=(0, 1, 2),
            index=(None, None, None),
            viewport=(800, 600),
            world_to_clip=world_to_clip,
        )
    )
    print(
        f"Loading pyramid level {plan.target_level} "
        f"from {len(plan.wanted)} native chunks"
    )
    viewer.show()
    try:
        ndv.run_app()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
