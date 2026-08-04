"""Stress-test Lodstone progressive rendering with a Zebrahub timepoint."""

from __future__ import annotations

import argparse

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
    arguments = parser.parse_args()

    source = OMEZarrSource.open(arguments.url)
    shape = source.pyramid.levels[0].shape
    if not 0 <= arguments.time < shape[0]:
        parser.error(f"--time must be between 0 and {shape[0] - 1}")

    viewer = napari.Viewer(title=f"Lodstone Zebrahub t={arguments.time}")
    controller = NapariController(
        viewer,
        source,
        # ZSNS001 axes are (t, c, z, y, x). Fixing t and c produces one
        # lazy 3-D multiscale image without materializing either axis.
        fixed_index={0: arguments.time, 1: 0},
        name=f"ZSNS001 t={arguments.time}",
        colormap="gray",
        contrast_limits=(0, 1500),
        rendering="attenuated_mip",
        tile_max_bytes_3d=arguments.tile_mib * MIB,
        interval_max_bytes=arguments.interval_mib * MIB,
        max_bytes_per_second=(
            None if arguments.rate_mib is None else arguments.rate_mib * MIB
        ),
    )
    viewer.dims.ndisplay = arguments.ndisplay
    viewer.reset_view()

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
