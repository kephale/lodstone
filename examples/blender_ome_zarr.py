"""Stream a remote multiscale OME-Zarr into native Blender volumes, in memory.

Run inside Blender 5.2+ with Lodstone and its ``ome-zarr`` extras importable::

    blender --python examples/blender_ome_zarr.py -- [URL] [--time T] ...

Lodstone reads only the chunks the 3-D viewport needs, on background threads,
directly from the OME-Zarr. Each level is copied into an in-memory attribute
and turned into a native volume grid by geometry nodes, so the viewport,
EEVEE and Cycles all render it. No VDB or other cache file is written, and the
full dataset is never downloaded or converted.

The default dataset is Zebrahub ZSNS001: a 791-timepoint, 3.7 TB light-sheet
series. One full-resolution timepoint is 448 x 2174 x 2423 uint16 (4.7 GB).

N sidebar > Lodstone shows the streamed level, bytes read and publish times.
"Render with finest data" loads the render camera's view at the render budget,
then renders (F12 itself renders whatever the viewport has loaded).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from lodstone.adapters.blender import (
    LEVEL_COLORS,
    BlenderController,
    blender_fixed_index,
    emission_material,
)
from lodstone.sources import OMEZarrSource

DEFAULT_URL = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/"
    "single-objective/ZSNS001.ome.zarr"
)
KEY = "lodstone_ome_zarr"
MIB = 1024**2


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="blender_ome_zarr")
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--time", type=int, default=400)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--memory-mib", type=int, default=32)
    parser.add_argument("--render-memory-mib", type=int, default=128)
    parser.add_argument("--clims", type=float, nargs=2, default=(0.0, 250.0))
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--absorption", type=float, default=8.0)
    parser.add_argument(
        "--units-per-blender-unit",
        type=float,
        default=100.0,
        help="physical units (e.g. µm) per Blender unit",
    )
    return parser.parse_args(argv)


def _box(low, high):
    corners = [
        (x, y, z)
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if sum(a != b for a, b in zip(corners[i], corners[j], strict=True)) == 1
    ]
    return corners, edges


def extent_box(
    name: str, low: np.ndarray, high: np.ndarray, parent
) -> bpy.types.Object:
    """Wireframe cube bounding ``low``..``high`` in root-local XYZ."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(*_box(low, high), [])
    obj = bpy.data.objects.new(name, mesh)
    obj.parent = parent
    obj.display_type = "WIRE"
    obj.hide_render = True
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup(arguments: argparse.Namespace) -> BlenderController:
    scene = bpy.context.scene
    started = time.perf_counter()
    source = OMEZarrSource.open(arguments.url)
    opened_s = time.perf_counter() - started
    pyramid = source.pyramid
    fixed = blender_fixed_index(pyramid, time=arguments.time, channel=arguments.channel)
    previous = bpy.app.driver_namespace.pop(KEY, None)
    if previous is not None:
        previous.close()

    # Physical extent of the finest level, data order (z, y, x) -> local XYZ.
    finest = pyramid.levels[0]
    transform = finest.voxel_to_world
    spacing = np.array([np.linalg.norm(transform[:-1, a]) for a in range(-4, -1)])
    origin = transform[-4:-1, -1]
    shape = np.asarray(finest.shape[-3:], dtype=float)
    low = (origin - 0.5 * spacing)[::-1]
    high = (origin + (shape - 0.5) * spacing)[::-1]

    scale = 1.0 / arguments.units_per_blender_unit
    root = bpy.data.objects.new("Lodstone OME-Zarr", None)
    root.empty_display_type = "PLAIN_AXES"
    scene.collection.objects.link(root)
    root.scale = (scale, scale, scale)
    root.location = -Vector((low + high) / 2) * scale
    extent_box("Full-resolution extent", low, high, root)

    material = emission_material(
        "Lodstone OME-Zarr volume",
        strength=arguments.brightness,
        density=arguments.absorption,
    )
    controller = BlenderController(
        source,
        root=root,
        material=material,
        fixed_index=fixed,
        memory_limit=arguments.memory_mib * MIB,
        render_memory_limit=arguments.render_memory_mib * MIB,
        clims=tuple(arguments.clims),
    )
    controller.info = {
        "url": arguments.url,
        "open_s": round(opened_s, 2),
        "levels": [tuple(level.shape) for level in pyramid.levels],
        "full_bytes": int(np.prod(finest.shape) * finest.dtype.itemsize),
        "timepoint_bytes": int(np.prod(finest.shape[-3:]) * finest.dtype.itemsize),
        "fixed": fixed,
    }

    # Render camera and a viewport framing the whole specimen.
    size = Vector(high - low) * scale
    camera = bpy.data.objects.new("Lodstone camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    camera.location = Vector((0.9, -1.3, 0.9)) * max(size)
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = 1000
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 960
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.0, 0.0, 0.0)
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    # EEVEE draws volumes on a screen-space froxel grid; the 8 px default
    # blurs fine data regardless of resolution. 2 px keeps nuclei distinct.
    scene.eevee.volumetric_tile_size = "2"
    scene.eevee.volumetric_samples = 96
    # Configure every workspace's 3-D view: the launcher may run from Scripting.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.shading.type = "MATERIAL"
            space.clip_end = 10000
            space.show_region_ui = True
            r3d = space.region_3d
            r3d.view_location = (0, 0, 0)
            r3d.view_distance = 1.6 * max(size)
            r3d.view_perspective = "PERSP"
            r3d.view_rotation = camera.rotation_euler.to_quaternion()

    tries = {"left": 240}

    def show_tab():
        # A sidebar accepts a tab only after it has drawn once, e.g. when the
        # Layout workspace is shown after running from Scripting. Keep trying.
        # Only displayed screens accept it; others report it read-only.
        pending = False
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                for region in area.regions if area.type == "VIEW_3D" else ():
                    if region.type != "UI":
                        continue
                    with contextlib.suppress(TypeError, AttributeError):
                        region.active_panel_category = "Lodstone"
                    if region.active_panel_category != "Lodstone":
                        pending = True
            if window.workspace.name != "Layout":
                pending = True
        tries["left"] -= 1
        return 0.5 if pending and tries["left"] > 0 else None

    bpy.app.timers.register(show_tab, first_interval=0.5)
    install_overlay()
    bpy.app.driver_namespace[KEY] = controller
    controller.start()
    return controller


def _overlay_lines(controller: BlenderController) -> list[tuple[str, tuple]]:
    lines = []
    pyramid = controller.stream.source.pyramid
    for level in sorted(controller.target.publications):
        publication = controller.target.publications[level]
        spacing = "x".join(f"{v:.2f}" for v in publication.scales[::-1])
        shape = "x".join(str(n) for n in publication.data.shape[::-1])
        if level == 0:
            name = "FULL RESOLUTION (level 0)"
        elif level == len(pyramid.levels) - 1:
            name = f"coarse context (level {level})"
        else:
            name = f"level {level}"
        tint = LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)]
        lines.append((f"{name}: {shape} voxels @ {spacing} um", tint))
    status = controller.stream.status
    summary = (
        f"streaming {status.state} - read {_human(status.bytes_read)} this pass"
        f" from {_human(controller.info['full_bytes'])} OME-Zarr"
    )
    lines.append((summary, (0.85, 0.85, 0.85)))
    return lines


OVERLAY = {"handle": None}


def install_overlay() -> None:
    import blf

    if OVERLAY["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(OVERLAY["handle"], "WINDOW")

    def draw():
        controller = bpy.app.driver_namespace.get(KEY)
        if controller is None:
            return
        size = 15 * bpy.context.preferences.system.ui_scale
        font = 0
        blf.size(font, size)
        lines = _overlay_lines(controller)
        scale = bpy.context.preferences.system.ui_scale
        y = 16 * scale + size * 1.5 * (len(lines) - 1)  # bottom-left corner
        for text, color in lines:
            blf.color(font, *color, 1.0)
            blf.position(font, 16 * scale, y, 0)
            blf.draw(font, text)
            y -= size * 1.5

    OVERLAY["handle"] = bpy.types.SpaceView3D.draw_handler_add(
        draw, (), "WINDOW", "POST_PIXEL"
    )


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


class LODSTONE_OT_render_finest(bpy.types.Operator):
    bl_idname = "lodstone.render_finest"
    bl_label = "Render with finest data"
    bl_description = "Load the render camera's view at the render budget, then render"

    def execute(self, context):
        controller = bpy.app.driver_namespace.get(KEY)
        if controller is None:
            return {"CANCELLED"}
        try:
            started = time.perf_counter()
            plan = controller.prepare_render()
            controller.info["render_prep_s"] = round(time.perf_counter() - started, 2)
            controller.info["render_level"] = plan.target_level
            result = bpy.ops.render.render("INVOKE_DEFAULT")
            if "CANCELLED" in result:
                controller.finish_render()
                return {"CANCELLED"}
        except Exception as error:  # noqa: BLE001 - report and restore the viewport
            controller.finish_render()
            controller.error = f"{type(error).__name__}: {error}"
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        def restore():
            if controller._closed:
                return None
            if controller.rendering or bpy.app.is_job_running("RENDER"):
                return 0.25
            controller.finish_render()
            return None

        bpy.app.timers.register(restore, first_interval=0.5)
        return {"FINISHED"}


def _largest_view() -> tuple[bpy.types.Area, bpy.types.RegionView3D] | None:
    best = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                size = area.width * area.height
                if best is None or size > best[0]:
                    best = (size, area, area.spaces.active.region_3d)
    return None if best is None else (best[1], best[2])


def _tissue_point(controller: BlenderController, eye: Vector) -> Vector:
    """World position of bright coarse-context tissue nearest ``eye``."""
    publication = controller.target.publications[max(controller.target.publications)]
    data = publication.data
    index = np.argwhere(data > np.percentile(data, 99.0))
    index = index[:: max(1, len(index) // 4000)]
    local = (np.asarray(publication.origins) + index * publication.scales)[:, ::-1]
    root = controller.root.matrix_world
    world = [root @ Vector(point) for point in local]
    return min(world, key=lambda point: (point - eye).length)


def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


DEMO = {"timer": None}


class LODSTONE_OT_demo_path(bpy.types.Operator):
    bl_idname = "lodstone.demo_path"
    bl_label = "Play demo path"
    bl_description = (
        "Drive the viewport along a fixed real-time path: orbit, dolly into "
        "tissue, orbit close, zoom out; optionally finish with a finest render"
    )

    render: bpy.props.BoolProperty(name="Finish with render", default=False)

    def execute(self, context):
        controller = bpy.app.driver_namespace.get(KEY)
        found = _largest_view()
        if controller is None or found is None or not controller.target.publications:
            self.report({"WARNING"}, "Wait for the first view to load")
            return {"CANCELLED"}
        if DEMO["timer"] is not None and bpy.app.timers.is_registered(DEMO["timer"]):
            bpy.app.timers.unregister(DEMO["timer"])
        area, r3d = found
        start_rot = r3d.view_rotation.copy()
        start_loc = r3d.view_location.copy()
        start_dist = r3d.view_distance
        orbit_a = Quaternion((0, 0, 1), np.radians(70)) @ start_rot
        # Pick the tissue target as seen after the first orbit.
        far_eye = start_loc + orbit_a @ Vector((0, 0, start_dist))
        target = _tissue_point(controller, far_eye)
        orbit_b = Quaternion((0, 0, 1), np.radians(40)) @ orbit_a
        close = 0.22 * start_dist
        # (duration s, rotation, location, distance, color-by-level) keyframes,
        # eased between. Level colors fade in close up: green = full resolution.
        keys = [
            (3.0, start_rot, start_loc, start_dist, 0.0),  # hold overview
            (8.0, orbit_a, start_loc, start_dist, 0.0),  # orbit
            (8.0, orbit_a, target, close, 0.0),  # dolly into tissue
            (3.0, orbit_a, target, close, 0.0),  # hold: fine chunks arrive
            (2.0, orbit_a, target, close, 1.0),  # show levels
            (8.0, orbit_b, target, close, 1.0),  # orbit close, levels shown
            (2.0, orbit_b, target, close, 0.0),  # hide levels
            (6.0, start_rot, start_loc, start_dist, 0.0),  # zoom back out
            (3.0, start_rot, start_loc, start_dist, 0.0),  # hold
        ]
        material = bpy.data.materials.get("Lodstone OME-Zarr volume")
        toggle = (
            None
            if material is None
            else next(
                (n for n in material.node_tree.nodes if n.label == "Color by level"),
                None,
            )
        )
        if toggle is not None:
            toggle.outputs[0].default_value = 0.0
        render = self.render
        state = {"t0": time.perf_counter(), "segment": 0, "from": keys[0]}

        def step():
            elapsed = time.perf_counter() - state["t0"]
            index = state["segment"]
            duration, rot, loc, dist, color = keys[index]
            _, rot0, loc0, dist0, color0 = state["from"]
            f = _ease(elapsed / duration)
            r3d.view_rotation = rot0.slerp(rot, f)
            r3d.view_location = loc0.lerp(loc, f)
            if toggle is not None:
                toggle.outputs[0].default_value = color0 + (color - color0) * f
            r3d.view_distance = float(
                np.exp(np.log(dist0) * (1 - f) + np.log(dist) * f)
            )
            area.tag_redraw()
            if elapsed < duration:
                return 1 / 60
            state["segment"] += 1
            state["from"] = keys[index]
            state["t0"] = time.perf_counter()
            if state["segment"] < len(keys):
                return 1 / 60
            DEMO["timer"] = None
            if render:
                bpy.ops.lodstone.render_finest()
            return None

        DEMO["timer"] = step
        bpy.app.timers.register(step, first_interval=0.0)
        return {"FINISHED"}


class LODSTONE_PT_ome_zarr(bpy.types.Panel):
    bl_label = "Lodstone OME-Zarr"
    bl_idname = "LODSTONE_PT_ome_zarr"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lodstone"

    def draw(self, context):
        controller = bpy.app.driver_namespace.get(KEY)
        layout = self.layout
        if controller is None:
            layout.label(text="Not running")
            return
        s = controller.summary()
        info = controller.info
        col = layout.column(align=True)
        col.label(text=f"Dataset: {_human(info['full_bytes'])} total")
        col.label(text=f"This timepoint: {_human(info['timepoint_bytes'])}")
        col.separator()
        col.label(text=f"State: {s['state']}  ({s['progress']:.0%})")
        col.label(text=f"Focus level: {s['level']}  shape {s['shape']}")
        col.label(text=f"Levels shown: {s['levels_shown']}")
        col.label(text=f"Read this pass: {_human(s['bytes_read'])}")
        col.label(text=f"Resident: {_human(s['resident_bytes'])}")
        if s["publish_ms_last"] is not None:
            col.label(
                text=f"Publish: {s['publish_ms_last']:.0f} ms "
                f"(max {s['publish_ms_max']:.0f})"
            )
        if "render_level" in info:
            col.label(
                text=f"Last render: level {info['render_level']}, "
                f"prep {info['render_prep_s']} s"
            )
        if s["error"]:
            col.label(text=s["error"], icon="ERROR")
        material = bpy.data.materials.get("Lodstone OME-Zarr volume")
        brightness = (
            None
            if material is None
            else next(
                (n for n in material.node_tree.nodes if n.label == "Brightness"), None
            )
        )
        if brightness is not None:
            layout.prop(brightness.inputs[1], "default_value", text="Brightness")
        nodes = (
            {} if material is None else {n.label: n for n in material.node_tree.nodes}
        )
        if "Absorption" in nodes:
            layout.prop(
                nodes["Absorption"].inputs[1], "default_value", text="Absorption"
            )
        if "Color by level" in nodes:
            layout.prop(
                nodes["Color by level"].outputs[0],
                "default_value",
                text="Color by level (green = full res)",
                slider=True,
            )
        layout.operator(LODSTONE_OT_render_finest.bl_idname, icon="RENDER_STILL")
        row = layout.row(align=True)
        row.operator(LODSTONE_OT_demo_path.bl_idname, icon="PLAY").render = False
        row.operator(
            LODSTONE_OT_demo_path.bl_idname, text="Play + render", icon="RENDER_STILL"
        ).render = True


def _redraw_panel():
    controller = bpy.app.driver_namespace.get(KEY)
    if controller is None:
        return None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return 0.25


def register() -> None:
    for cls in (LODSTONE_OT_render_finest, LODSTONE_OT_demo_path, LODSTONE_PT_ome_zarr):
        old = getattr(bpy.types, cls.__name__, None)
        if old is not None:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    if not bpy.app.timers.is_registered(_redraw_panel):
        bpy.app.timers.register(_redraw_panel, first_interval=0.5, persistent=True)


def main(argv: list[str] | None = None) -> BlenderController:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    arguments = parse(argv)
    register()
    return setup(arguments)


if __name__ == "__main__":
    main()
