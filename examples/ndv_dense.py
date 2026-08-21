"""Display a real Zebrahub timepoint through ndv's dense 3-D canvas.

The coarsest source level remains visible as a clipmap-style context while
translated, camera-selected higher-resolution windows stream above it.
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
    # ndv's scene axes are XYZ while the source axes are ZYX. Bootstrap a
    # top-down full-volume view; camera snapshots take over after publication.
    world_to_clip = np.zeros((4, 4), dtype=np.float64)
    world_to_clip[0, 2] = 2.0 / world_extent[2]
    world_to_clip[1, 1] = 2.0 / world_extent[1]
    world_to_clip[2, 0] = 2.0 / world_extent[0]
    world_to_clip[3, 3] = 1.0
    world_to_clip[:-1, -1] = -1.0

    viewer = ndv.ArrayViewer()
    widget = viewer.widget()
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtWidgets import (
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QSlider,
        QWidget,
    )

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

    controls = QFrame(widget)
    controls.setStyleSheet(
        "QFrame { background: rgba(0, 0, 0, 190); color: white; "
        "border-radius: 4px; } QLabel { color: white; }"
    )
    form = QFormLayout(controls)
    form.setContentsMargins(9, 7, 9, 7)
    form.setSpacing(5)

    def slider_row(
        minimum: int, maximum: int, value: int
    ) -> tuple[QWidget, QSlider, QLabel]:
        row = QWidget(controls)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal, row)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setMinimumWidth(150)
        readout = QLabel(row)
        readout.setMinimumWidth(48)
        row_layout.addWidget(slider)
        row_layout.addWidget(readout)
        return row, slider, readout

    budget_row, budget_slider, budget_value = slider_row(32, 256, 64)
    depth_row, depth_slider, depth_value = slider_row(0, 100, 75)
    lod_row, lod_slider, lod_value = slider_row(50, 300, 200)
    form.addRow("Focus MiB", budget_row)
    form.addRow("Depth coverage", depth_row)
    form.addRow("LOD bias", lod_row)

    tune_timer = QTimer(controls)
    tune_timer.setSingleShot(True)
    tune_timer.setInterval(250)

    def configure_focus() -> None:
        memory_mib = budget_slider.value()
        # A larger UI value means more camera-axis coverage. Lodstone's weight
        # is the inverse: zero treats screen-center and all depths equally.
        depth_weight = 2.0 * (1.0 - depth_slider.value() / 100.0)
        lod_bias = lod_slider.value() / 100.0
        budget_value.setText(str(memory_mib))
        depth_value.setText(f"{depth_slider.value()}%")
        lod_value.setText(f"{lod_bias:.2f}")
        controller.set_focus_policy(
            memory_limit=memory_mib * 1024**2,
            focus_depth_weight=depth_weight,
            lod_bias=lod_bias,
            replan=False,
        )
        tune_timer.start()

    tune_timer.timeout.connect(lambda: controller.set_focus_policy())
    for slider in (budget_slider, depth_slider, lod_slider):
        slider.valueChanged.connect(configure_focus)
    configure_focus()
    tune_timer.stop()
    controls.move(12, 50)
    controls.adjustSize()
    controls.show()
    controls.raise_()

    plan = controller.update(
        View(
            displayed_axes=(0, 1, 2),
            index=(None, None, None),
            viewport=(600, 600),
            world_to_clip=world_to_clip,
        )
    )
    print(
        f"Loading pyramid level {plan.target_level} "
        f"from {len(plan.wanted)} logical blocks"
    )
    viewer.show()
    try:
        ndv.run_app()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
