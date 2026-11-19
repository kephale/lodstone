"""Inspect a registered image and label pyramid.

Run with uv run --with matplotlib examples/label_pyramid.py.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from lodstone import Region
from lodstone.sources import ArrayPyramidSource


def reference_pyramids() -> tuple[ArrayPyramidSource, ArrayPyramidSource]:
    """An odd-sized reference with a landmark crossing a chunk boundary."""
    y, x = np.ogrid[:129, :197]
    inside = ((y - 63) / 21) ** 2 + ((x - 63) / 28) ** 2 <= 1
    image = np.where(inside, 200, (x + y) % 80).astype(np.uint8)
    labels = np.where(inside, 1, 0).astype(np.uint32)
    fine = np.array([[2.0, 0.0, -38.0], [0.0, 0.75, 15.0], [0.0, 0.0, 1.0]])
    transforms = [fine, fine @ np.diag([3, 3, 1])]
    options = {
        "axes": ("y", "x"),
        "transforms": transforms,
        "chunks": [(32, 32), (19, 19)],
    }
    return (
        ArrayPyramidSource([image, image[1::3, 1::3]], **options),
        ArrayPyramidSource([labels, labels[1::3, 1::3]], **options),
    )


def visible_region(level, xlim, ylim) -> Region | None:
    """Map the viewport's world bounds to a clipped region in data order."""
    offset = level.voxel_to_world[:2, 2]
    spacing = np.diag(level.voxel_to_world)[:2]
    starts = np.floor((np.array([min(ylim), min(xlim)]) - offset) / spacing)
    stops = np.ceil((np.array([max(ylim), max(xlim)]) - offset) / spacing)
    start = tuple(int(v) for v in np.maximum(starts, 0))
    stop = tuple(int(v) for v in np.minimum(stops, level.shape))
    if any(a >= b for a, b in zip(start, stop, strict=True)):
        return None
    return Region(start, stop)


def render(ax, image, labels):
    """Draw only the visible region, with an integer categorical overlay."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    screen_spacing = abs(ylim[1] - ylim[0]) / max(ax.bbox.height, 1)
    index = 0
    for candidate, level in enumerate(image.pyramid.levels):
        if level.voxel_to_world[0, 0] <= screen_spacing:
            index = candidate
    label_index = min(index, len(labels.pyramid.levels) - 1)
    for artist in list(ax.images):
        artist.remove()
    for source, level_index in ((image, index), (labels, label_index)):
        level = source.pyramid.levels[level_index]
        region = visible_region(level, xlim, ylim)
        if region is None:
            continue
        pixels = asyncio.run(source.read(level_index, region))
        matrix = level.voxel_to_world
        transform = np.diag(matrix)[:2]
        low = transform * region.start + matrix[:2, 2]
        high = transform * region.stop + matrix[:2, 2]
        extent = (low[1], high[1], low[0], high[0])
        if source is labels:
            rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
            rgba[..., 0] = (pixels.astype(np.uint64) * 53 + 71) % 255
            rgba[..., 1] = (pixels.astype(np.uint64) * 97 + 32) % 255
            rgba[..., 2] = (pixels.astype(np.uint64) * 31 + 119) % 255
            rgba[..., 3] = np.where(pixels == 0, 0, 150)
            pixels = rgba
        ax.imshow(
            pixels,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            vmin=0,
            vmax=200,
        )
    ax.figure.canvas.draw_idle()


def main() -> None:
    from matplotlib import pyplot as plt

    from lodstone.labels import LabelEditor
    from lodstone.sources import ZarrLabels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, help="new or existing writable label store"
    )
    parser.add_argument(
        "--radius", type=float, default=4, help="brush radius in world units"
    )
    args = parser.parse_args()
    if not np.isfinite(args.radius) or args.radius <= 0:
        parser.error("--radius must be positive")
    image, labels = reference_pyramids()
    editor = None
    if args.labels is not None:
        store = (
            ZarrLabels.open(args.labels)
            if args.labels.exists()
            else ZarrLabels.create(args.labels, image.pyramid)
        )
        editor = LabelEditor(store)
        labels = editor
    fig, ax = plt.subplots()
    ax.set(xlim=(15, 163), ylim=(220, -38), autoscale_on=False)
    ax.callbacks.connect("xlim_changed", lambda _: render(ax, image, labels))
    ax.callbacks.connect("ylim_changed", lambda _: render(ax, image, labels))
    if editor is not None:
        points = []
        button = None

        def press(event):
            nonlocal button
            toolbar = fig.canvas.toolbar
            if (
                event.inaxes == ax
                and event.button in (1, 3)
                and not (toolbar and toolbar.mode)
            ):
                points[:] = [(event.ydata, event.xdata)]
                button = event.button

        def move(event):
            if button is not None and event.inaxes == ax:
                points.append((event.ydata, event.xdata))

        def release(event):
            nonlocal button
            if button is None:
                return
            move(event)
            editor.stroke(np.asarray(points), args.radius, 1 if button == 1 else 0)
            button = None
            render(ax, image, labels)

        def keypress(event):
            if event.key == "ctrl+s":
                try:
                    editor.save()
                except OSError as error:
                    ax.set_title(f"Save failed: {error}")
                else:
                    ax.set_title("Labels saved")
            elif event.key == "ctrl+z":
                editor.undo()
            elif event.key == "ctrl+y":
                editor.redo()
            else:
                return
            render(ax, image, labels)

        plt.rcParams["keymap.save"] = []
        ax.set_title("Left drag paints · right drag erases · Ctrl+S saves")
        for name, callback in (
            ("button_press_event", press),
            ("motion_notify_event", move),
            ("button_release_event", release),
            ("key_press_event", keypress),
        ):
            fig.canvas.mpl_connect(name, callback)
    render(ax, image, labels)
    plt.show()


if __name__ == "__main__":
    main()
