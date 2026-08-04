"""Visually inspect remote OME-Zarr streaming in napari."""

from __future__ import annotations

import argparse

import napari

from lodstone.adapters.napari import NapariController
from lodstone.sources import OMEZarrSource

DEFAULT_URL = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--screenshot")
    parser.add_argument("--ndisplay", type=int, choices=(2, 3), default=2)
    arguments = parser.parse_args()

    source = OMEZarrSource.open(arguments.url)
    viewer = napari.Viewer(title="Lodstone OME-Zarr")
    controllers = [
        NapariController(
            viewer,
            source,
            fixed_index={0: channel},
            name=name,
            colormap=colormap,
            blending="additive",
        )
        for channel, name, colormap in (
            (0, "LaminB1", "green"),
            (1, "DAPI", "blue"),
        )
    ]
    current = list(viewer.dims.current_step)
    current[0] = source.pyramid.levels[0].shape[1] // 2
    viewer.dims.current_step = tuple(current)
    viewer.dims.ndisplay = arguments.ndisplay
    viewer.reset_view()

    if arguments.screenshot:
        from qtpy.QtCore import QTimer

        def save() -> None:
            viewer.screenshot(arguments.screenshot, canvas_only=True)
            viewer.close()

        # The progressive renderer remains active after its first pass. Give
        # the remote example a bounded visual-inspection window.
        QTimer.singleShot(30_000, save)

    try:
        napari.run()
    finally:
        for controller in controllers:
            controller.close()


if __name__ == "__main__":
    main()
