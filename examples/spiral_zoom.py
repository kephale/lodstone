"""A finite descent through twelve nested tiers of a connected spiral relief."""

from math import floor, log10
from time import perf_counter

import numpy as np
import pygfx as gfx

from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget
from lodstone.mesh import MeshData


LAST_TIER = 12
SECONDS_PER_TIER = 3.0


def spiral_surface():
    # Adjacent bands share a ring of indexed vertices. The radial envelope
    # and its derivative agree at both ends of every logarithmic period.
    sectors = 192
    depth = np.arange(-64, 32 * (LAST_TIER + 2) + 1)[:, None] / 32
    theta = 2 * np.pi * np.arange(sectors) / sectors
    radius = 4.0 ** -depth
    phase = 2 * np.pi * depth
    tier = np.clip(np.floor(depth).astype(int), 0, LAST_TIER)
    lobes = np.array([7, 5, 9, 4, 8, 6, 11, 5, 3, 8, 6, 4, 9])[tier]
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
    # At high magnification change the relief itself: circular steps, staggered
    # domes, diamond facets, then a branching ridge. The common envelope
    # brings each profile back to the shared ring at both ends of its tier.
    terraces = 0.5 + 0.5 * np.tanh(3 * np.cos(3 * phase + 0.7 * np.cos(4 * theta)))
    domes = ((0.5 + 0.5 * np.cos(9 * theta + 1.8 * np.cos(2 * phase)))
             * (0.5 + 0.5 * np.cos(3 * phase))) ** 1.5
    diamonds = np.maximum(0, 1 - (np.abs(np.sin(5 * theta + phase))
                                  + np.abs(np.sin(2 * phase))))
    trunk = 6 * theta - phase
    split = 1.7 * (1 - np.cos(phase))
    veins = np.maximum.reduce([np.exp(-3 * (1 - np.cos(trunk + shift)))
                               for shift in (0, split, -split)])
    scallops = (0.5 + 0.5 * np.cos(2 * phase + 2 * np.cos(8 * theta))) ** 3
    braid = np.exp(-4 * (np.cos(3 * theta + phase) - 0.55 * np.cos(2 * phase)) ** 2)
    mesas = 0.5 + 0.5 * np.tanh(6 * (np.cos(4 * theta) + np.cos(2 * phase) - 0.4))
    crown = (0.5 + 0.5 * np.cos(9 * theta + 0.8 * np.sin(phase))) ** 4
    relief = np.select([tier == 5, tier == 6, tier == 7, tier == 8,
                        tier == 9, tier == 10, tier == 11, tier == 12],
                       [terraces, domes, diamonds, veins, scallops, braid, mesas, crown])
    height = np.where(tier >= 5, radius * envelope * (0.08 + 0.57 * relief), height)
    # Blend mineral colors in mesh coordinates. Crest highlights follow the relief.
    palette = np.array([
        [32, 120, 123], [48, 165, 143], [53, 94, 150], [133, 83, 154],
        [178, 94, 78], [211, 150, 67], [95, 148, 112], [48, 123, 148],
        [107, 90, 161], [49, 123, 115], [182, 109, 83], [61, 96, 145], [190, 142, 71],
    ]) / 255
    fraction = np.clip(depth - tier, 0, 1)
    blend = (fraction ** 2 * (3 - 2 * fraction))[..., None]
    base = palette[tier] * (1 - blend) + palette[np.minimum(tier + 1, LAST_TIER)] * blend
    highlight = (envelope * (0.35 * petals + 0.4 * grain))[..., None]
    highlight = np.where((tier >= 5)[..., None], (0.78 * relief * envelope)[..., None], highlight)
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
    camera = gfx.PerspectiveCamera(40, depth_range=(0.1, 15))
    distance = 1 / np.tan(np.deg2rad(camera.fov / 2))
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
        magnification = 4 ** depth
        # Express the scene in units of the current tier. This is equivalent
        # to moving the eye closer, with stable camera and clip distances.
        mesh.local.scale = (magnification,) * 3
        azimuth = 0.38 * depth
        tilt = np.deg2rad(30 + 10 * np.sin(1.35 * depth))
        camera.local.position = distance * np.array([
            np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)
        ])
        camera.look_at((0, 0, 0))
        width, height = canvas.get_logical_size()
        camera.set_view_size(width, height)
        # Horizontal ruler at the depth of the central focus point.
        units_per_pixel = 2 * distance / magnification / (camera.projection_matrix[0, 0] * width)
        desired = min(145, width / 3) * units_per_pixel
        decade = 10 ** floor(log10(desired))
        length = max(n * decade for n in (1, 2, 5) if n * decade <= desired)
        ruler.local.scale_x = outline.local.scale_x = length / units_per_pixel
        label.set_text(f"{length:.5g} model units at focus")
        caption.set_text(f"{magnification:,.2f}×   ·   R / Home restart   ·   Space pause")
        caption.local.position = (16, height - 16, 0)
        renderer.render(scene, camera, flush=False)
        renderer.render(overlay, overlay_camera)

    try:
        if output:
            from PIL import Image

            output.mkdir(parents=True, exist_ok=True)
            frames = []
            for frame in range(LAST_TIER * 32 + 1):
                canvas.request_draw(lambda: paint(frame / 32))
                frames.append(Image.fromarray(np.asarray(canvas.draw())).convert("RGB"))
            frames[0].save(output / "rendered-descent.gif", save_all=True,
                           append_images=frames[1:], duration=[94] * (LAST_TIER * 32) + [2000], loop=0)
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
