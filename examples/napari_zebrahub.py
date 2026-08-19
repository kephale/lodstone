"""Stress-test Lodstone progressive rendering with a Zebrahub timepoint."""

from __future__ import annotations

import argparse
import logging

import napari

from lodstone.adapters.napari import NapariController
from lodstone.sources import OMEZarrSource

DEFAULT_URL = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/"
    "single-objective/ZSNS001.ome.zarr"
)
MIB = 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--time", type=int, default=400)
    parser.add_argument("--ndisplay", type=int, choices=(2, 3), default=3)
    parser.add_argument("--tile-mib", type=int, default=64)
    parser.add_argument("--interval-mib", type=int, default=512)
    parser.add_argument("--rate-mib", type=float, default=None)
    parser.add_argument("--screenshot")
    parser.add_argument("--screenshot-delay", type=float, default=90.0)
    parser.add_argument(
        "--camera-center",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="initial 3-D camera center",
    )
    parser.add_argument(
        "--camera-angles",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="initial 3-D camera Euler angles in degrees",
    )
    parser.add_argument("--zoom", type=float, help="initial camera zoom")
    parser.add_argument(
        "--trace-chunks",
        action="store_true",
        help="log planned tiles, native reads, cache hits, and evictions",
    )
    parser.add_argument(
        "--diagnostic-levels",
        action="store_true",
        help="replace image values with solid labels identifying source levels",
    )
    arguments = parser.parse_args()
    if arguments.trace_chunks:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    source = OMEZarrSource.open(arguments.url)
    shape = source.pyramid.levels[0].shape
    if not 0 <= arguments.time < shape[0]:
        parser.error(f"--time must be between 0 and {shape[0] - 1}")

    title = f"Lodstone Zebrahub t={arguments.time}"
    if arguments.diagnostic_levels:
        title += " — level coverage"
    viewer = napari.Viewer(title=title)
    layer_kwargs: dict[str, object] = {
        "name": f"ZSNS001 t={arguments.time}",
    }
    if arguments.diagnostic_levels:
        layer_kwargs["layer_type"] = "diagnostic"
    else:
        layer_kwargs.update(
            contrast_limits=(0, 1500),
            colormap="gray",
            rendering="attenuated_mip",
        )
    controller = NapariController(
        viewer,
        source,
        # ZSNS001 axes are (t, c, z, y, x). Fixing t and c produces one
        # lazy 3-D multiscale image without materializing either axis.
        fixed_index={0: arguments.time, 1: 0},
        tile_max_bytes_3d=arguments.tile_mib * MIB,
        interval_max_bytes=arguments.interval_mib * MIB,
        max_bytes_per_second=(
            None if arguments.rate_mib is None else arguments.rate_mib * MIB
        ),
        **layer_kwargs,
    )
    viewer.dims.ndisplay = arguments.ndisplay
    viewer.reset_view()
    if any(
        value is not None
        for value in (
            arguments.camera_center,
            arguments.camera_angles,
            arguments.zoom,
        )
    ):
        from qtpy.QtCore import QTimer

        def restore_camera() -> None:
            # Let the initial volume texture and GL objects be created before
            # the saved pose triggers a new progressive-loading pass.
            if arguments.camera_center is not None:
                viewer.camera.center = tuple(arguments.camera_center)
            if arguments.camera_angles is not None:
                viewer.camera.angles = tuple(arguments.camera_angles)
            if arguments.zoom is not None:
                viewer.camera.zoom = arguments.zoom

        QTimer.singleShot(1000, restore_camera)

    if arguments.screenshot:
        from qtpy.QtCore import QTimer

        def save() -> None:
            viewer.screenshot(arguments.screenshot, canvas_only=True)
            viewer.close()

        QTimer.singleShot(round(arguments.screenshot_delay * 1000), save)

    try:
        napari.run()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
