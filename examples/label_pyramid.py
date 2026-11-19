"""Inspect a registered image and label pyramid.

Run with uv run --with matplotlib examples/label_pyramid.py.
"""

from __future__ import annotations

import asyncio

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

    image, labels = reference_pyramids()
    _, ax = plt.subplots()
    ax.set(xlim=(15, 163), ylim=(220, -38), autoscale_on=False)
    ax.callbacks.connect("xlim_changed", lambda _: render(ax, image, labels))
    ax.callbacks.connect("ylim_changed", lambda _: render(ax, image, labels))
    render(ax, image, labels)
    plt.show()


if __name__ == "__main__":
    main()
