"""A finite descent through eight nested tiers of a connected spiral relief."""

from math import floor, log10
from time import perf_counter

import numpy as np
import pygfx as gfx

from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget
from lodstone.mesh import MeshData


LAST_TIER = 8
SECONDS_PER_TIER = 3.0


def spiral_surface():
    # Adjacent bands share a ring of indexed vertices. The radial envelope
    # and its derivative agree at both ends of every logarithmic period.
    sectors = 192
    depth = np.arange(-64, 321)[:, None] / 32
    theta = 2 * np.pi * np.arange(sectors) / sectors
    radius = 4.0 ** -depth
    phase = 2 * np.pi * depth
    tier = np.clip(np.floor(depth).astype(int), 0, LAST_TIER)
    lobes = np.array([7, 5, 9, 4, 8, 6, 11, 5, 3])[tier]
    twist = np.where(tier % 2 == 0, 1, -1)
    angle = lobes * theta + twist * phase
    petals = (0.5 + 0.5 * np.cos(angle)) ** 3
    ribs = (0.5 + 0.5 * np.cos(3 * angle)) ** 5
    envelope = np.sin(phase / 2) ** 2
    height = radius * envelope * (0.09 + 0.54 * petals + 0.08 * ribs)
    # Flutes, crosshatching and beadwork repeat in successive tiers.
    flute = (0.5 + 0.5 * np.cos(4 * angle)) ** 3
    cross = (0.5 + 0.5 * np.cos(2 * angle + 4 * phase)) * (
        0.5 + 0.5 * np.cos(2 * angle - 4 * phase)
    )
    beads = ribs * (0.5 + 0.5 * np.cos(5 * phase)) ** 2
    grain = np.choose(tier % 3, (flute, cross, beads))
    height += 0.048 * radius * envelope * grain
    # Blend mineral colors in mesh coordinates. Crest highlights follow the relief.
    palette = np.array([
        [32, 120, 123], [48, 165, 143], [53, 94, 150], [133, 83, 154],
        [178, 94, 78], [211, 150, 67], [95, 148, 112], [48, 123, 148],
        [107, 90, 161],
    ]) / 255
    fraction = np.clip(depth - tier, 0, 1)
    blend = (fraction ** 2 * (3 - 2 * fraction))[..., None]
    base = palette[tier] * (1 - blend) + palette[np.minimum(tier + 1, LAST_TIER)] * blend
    highlight = (envelope * (0.35 * petals + 0.4 * grain))[..., None]
    colors = base * (1 - highlight) + np.array([0.98, 0.86, 0.65]) * highlight
    # Offset each bud toward the same fixed point as its parent contracts.
    x = radius * (np.cos(theta) + 0.17 * np.cos(phase))
    y = radius * (np.sin(theta) + 0.15 * np.sin(phase))
    vertices = np.stack((x, y, height), axis=-1).reshape(-1, 3)
    ring = np.arange(len(vertices)).reshape(-1, sectors)
    a, b = ring[:-1], np.roll(ring[:-1], -1, axis=1)
    c, d = ring[1:], np.roll(ring[1:], -1, axis=1)
    faces = np.concatenate([np.stack(t, axis=-1).reshape(-1, 3)
                            for t in ((a, b, c), (b, d, c))])
    # Close the innermost ring with a fan at the finite tip.
    tip = len(vertices)
    cap = np.column_stack((ring[-1], np.roll(ring[-1], -1), np.full(sectors, tip)))
    surface = MeshData(np.vstack((vertices, [0, 0, 0])), np.vstack((faces, cap)))
    colors = np.vstack((colors.reshape(-1, 3), palette[-1])).astype(np.float32)
    return surface, colors


def run(output):
    if output:
        from rendercanvas.offscreen import RenderCanvas
    else:
        from rendercanvas.auto import RenderCanvas, loop

    canvas = RenderCanvas(size=(720, 600), title="Lodstone · Spiral descent")
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    target = PyGFXMeshTarget(scene)
    surface, colors = spiral_surface()
    target.replace(0, [((0, 0), surface)])
    mesh = target.group.children[0]
    mesh.geometry.colors = gfx.Buffer(colors)
    mesh.material = gfx.MeshPhongMaterial(color_mode="vertex", shininess=22, side="both")
    scene.add(gfx.AmbientLight(intensity=0.55))
    light = gfx.DirectionalLight(intensity=2)
    light.local.position = (-4, 3, 6)
    scene.add(light)
    camera = gfx.OrthographicCamera(2, 2)
    camera.local.position = (0, 0, 16)
    overlay = gfx.Scene()
    caption = gfx.Text(
        text="", font_size=18, screen_space=True, anchor="top-left",
        material=gfx.TextMaterial(color="white", outline_color="black", outline_thickness=0.35),
    )
    overlay.add(caption)
    overlay_camera = gfx.ScreenCoordsCamera()
    def bar(color, thickness):
        line = gfx.Line(
            gfx.Geometry(positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)),
            gfx.LineMaterial(color=color, thickness=thickness, depth_test=False),
        )
        line.local.position = (17, 18, 0)
        overlay.add(line)
        return line

    outline = bar("black", 6)
    ruler = bar("white", 3)
    ruler.render_order = 1
    label = gfx.Text(text="", font_size=15, screen_space=True, anchor="bottom-left",
                     material=caption.material)
    label.local.position = (16, 24, 0)
    overlay.add(label)
    phase = 0.0
    running = True
    last_tick = perf_counter()

    @canvas.add_event_handler("key_down")
    def key(event):
        nonlocal phase, running, last_tick
        name = event["key"].lower()
        if name in ("r", "home"):
            phase, running = 0.0, True
        elif name == " ":
            running = not running
        last_tick = perf_counter()

    @canvas.add_event_handler("wheel")
    def scroll(event):
        nonlocal phase
        phase = float(np.clip(phase - event["dy"] * 0.001, 0, LAST_TIER))

    def paint(depth):
        camera.zoom = 4 ** depth
        width, height = canvas.get_logical_size()
        camera.set_view_size(width, height)
        # The orthographic projection maps two clip units to the canvas width.
        units_per_pixel = 2 / (camera.projection_matrix[0, 0] * width)
        desired = min(145, width / 3) * units_per_pixel
        decade = 10 ** floor(log10(desired))
        length = max(n * decade for n in (1, 2, 5) if n * decade <= desired)
        ruler.local.scale_x = outline.local.scale_x = length / units_per_pixel
        label.set_text(f"{length:.5g} model units")
        caption.set_text(f"{camera.zoom:,.2f}×   ·   R / Home restart   ·   Space pause")
        caption.local.position = (16, height - 16, 0)
        renderer.render(scene, camera, flush=False)
        renderer.render(overlay, overlay_camera)

    try:
        if output:
            from PIL import Image

            output.mkdir(parents=True, exist_ok=True)
            frames = []
            for frame in range(257):
                canvas.request_draw(lambda: paint(frame / 32))
                frames.append(Image.fromarray(np.asarray(canvas.draw())).convert("RGB"))
            frames[0].save(output / "descent.webp", save_all=True,
                           append_images=frames[1:], duration=[94] * 256 + [2000], loop=1, quality=88)
        else:
            def draw():
                nonlocal phase, running, last_tick
                now = perf_counter()
                if running:
                    phase = min(LAST_TIER, phase + (now - last_tick) / SECONDS_PER_TIER)
                    if phase == LAST_TIER:
                        running = False
                last_tick = now
                paint(phase)
                canvas.request_draw()

            canvas.request_draw(draw)
            loop.run()
    finally:
        target.close()
