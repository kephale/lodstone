"""A spiral relief whose central bud repeats at one quarter of the scale."""

from time import perf_counter

import numpy as np
import pygfx as gfx

from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget
from lodstone.mesh import MeshData


def spiral_surface():
    # Adjacent bands share a ring of indexed vertices. The radial envelope
    # and its derivative agree at both ends of every logarithmic period.
    sectors = 192
    depth = np.arange(-64, 321)[:, None] / 32
    theta = 2 * np.pi * np.arange(sectors) / sectors
    radius = 4.0 ** -depth
    phase = 2 * np.pi * depth
    petals = (0.5 + 0.5 * np.cos(7 * theta + phase)) ** 3
    ribs = (0.5 + 0.5 * np.cos(21 * theta + 3 * phase)) ** 5
    envelope = np.sin(phase / 2) ** 2
    height = radius * envelope * (0.09 + 0.54 * petals + 0.08 * ribs)
    # Offset each bud toward the same fixed point as its parent contracts.
    x = radius * (np.cos(theta) + 0.17 * np.cos(phase))
    y = radius * (np.sin(theta) + 0.15 * np.sin(phase))
    vertices = np.stack((x, y, height), axis=-1).reshape(-1, 3)
    ring = np.arange(len(vertices)).reshape(-1, sectors)
    a, b = ring[:-1], np.roll(ring[:-1], -1, axis=1)
    c, d = ring[1:], np.roll(ring[1:], -1, axis=1)
    faces = np.concatenate([np.stack(t, axis=-1).reshape(-1, 3)
                            for t in ((a, b, c), (b, d, c))])
    return MeshData(vertices, faces)


def run(output):
    if output:
        from rendercanvas.offscreen import RenderCanvas
    else:
        from rendercanvas.auto import RenderCanvas, loop

    canvas = RenderCanvas(size=(720, 600), title="Lodstone · Spiral descent")
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    target = PyGFXMeshTarget(scene)
    target.replace(0, [((0, 0), spiral_surface())])
    camera = gfx.OrthographicCamera(2, 2)
    camera.local.position = (0, 0, 16)
    start = perf_counter()
    offset = 0.0

    @canvas.add_event_handler("wheel")
    def scroll(event):
        nonlocal offset
        offset -= event["dy"] * 0.001

    def paint(phase):
        # The surface repeats exactly under 4x scaling. Rebase while both
        # truncation rings are outside the view or smaller than one pixel.
        camera.zoom = 4 ** (phase % 1)
        renderer.render(scene, camera)

    try:
        if output:
            from PIL import Image

            output.mkdir(parents=True, exist_ok=True)
            frames = []
            for frame in range(96):
                canvas.request_draw(lambda: paint(frame / 96))
                frames.append(Image.fromarray(np.asarray(canvas.draw())).convert("RGB"))
            frames[0].save(output / "descent.webp", save_all=True,
                           append_images=frames[1:], duration=50, loop=0, quality=88)
        else:
            def draw():
                paint((perf_counter() - start) / 4.8 + offset)
                canvas.request_draw()

            canvas.request_draw(draw)
            loop.run()
    finally:
        target.close()
