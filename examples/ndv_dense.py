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
    parser.add_argument(
        "--backend",
        choices=("vispy", "pygfx"),
        help="ndv canvas backend (defaults to ndv's automatic selection)",
    )
    arguments = parser.parse_args()

    if arguments.backend is not None:
        ndv.set_canvas_backend(arguments.backend)

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
    from _ndv_block_overlay import BlockOverlay
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtWidgets import (
        QCheckBox,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QSlider,
        QWidget,
    )

    block_overlay = BlockOverlay(viewer.canvas, source.pyramid)
    viewer.canvas.cameraChanged.connect(lambda: block_overlay.update())

    backend_name = {
        "VispyArrayCanvas": "VisPy",
        "GfxArrayCanvas": "PyGFX",
    }.get(type(viewer.canvas).__name__, type(viewer.canvas).__name__)
    indicator = QLabel(f"Lodstone · {backend_name} · waiting", widget)
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
        indicator.setText(
            f"Lodstone · {backend_name} · active {active} · target {target}"
        )
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
        block_overlay.set_plan(plan)
        tiles = [tile for tile in plan.desired if tile.level == plan.target_level]
        if tiles:
            ndim = tiles[0].region.ndim
            start = tuple(
                min(tile.region.start[axis] for tile in tiles) for axis in range(ndim)
            )
            stop = tuple(
                max(tile.region.stop[axis] for tile in tiles) for axis in range(ndim)
            )
            shape = tuple(upper - lower for lower, upper in zip(start, stop))
            level = source.pyramid.levels[plan.target_level]
            dense_mib = np.prod(shape) * level.dtype.itemsize / 1024**2
            block_text = "×".join(str(value) for value in shape)
            geometry_value.setText(
                f"L{plan.target_level} · {len(tiles)} blk · {dense_mib:.1f} MiB\n"
                f"{block_text}"
            )
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

    budget_row, budget_slider, budget_value = slider_row(32, 512, 128)
    depth_row, depth_slider, depth_value = slider_row(0, 100, 60)
    lod_row, lod_slider, lod_value = slider_row(50, 300, 200)
    depth_slider.setToolTip(
        "0% favors high-resolution coverage across the canvas; "
        "100% favors a central column reaching front-to-back"
    )
    geometry_value = QLabel("waiting", controls)
    blocks_toggle = QCheckBox("Show planned block wireframes", controls)
    blocks_toggle.setChecked(True)
    blocks_toggle.toggled.connect(block_overlay.setVisible)
    form.addRow("Focus MiB", budget_row)
    form.addRow("Depth reach", depth_row)
    form.addRow("LOD bias", lod_row)
    form.addRow("Focus box", geometry_value)
    form.addRow("Diagnostics", blocks_toggle)

    tune_timer = QTimer(controls)
    tune_timer.setSingleShot(True)
    tune_timer.setInterval(250)

    def configure_focus() -> None:
        memory_mib = budget_slider.value()
        # Span a wide perceptual range: the canvas-heavy endpoint strongly
        # penalizes far chunks, while the depth-heavy endpoint retains a small
        # near-to-far tie breaker instead of reverting to data-axis ordering.
        depth_fraction = depth_slider.value() / 100.0
        depth_weight = 8.0 * (0.5 / 8.0) ** depth_fraction
        lod_bias = lod_slider.value() / 100.0
        budget_value.setText(str(memory_mib))
        depth_value.setText(f"{depth_slider.value()}% depth")
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
