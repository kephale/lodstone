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
from mathutils import Vector

from lodstone.adapters.blender import BlenderController, emission_material
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
    parser.add_argument("--clims", type=float, nargs=2, default=(0.0, 1500.0))
    parser.add_argument("--brightness", type=float, default=0.5)
    parser.add_argument(
        "--units-per-blender-unit",
        type=float,
        default=100.0,
        help="physical units (e.g. µm) per Blender unit",
    )
    return parser.parse_args(argv)


def extent_box(
    name: str, low: np.ndarray, high: np.ndarray, parent
) -> bpy.types.Object:
    """Wireframe cube bounding the full-resolution extent, in root-local XYZ."""
    mesh = bpy.data.meshes.new(name)
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
    mesh.from_pydata(corners, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    obj.parent = parent
    obj.display_type = "WIRE"
    obj.hide_render = True
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup(arguments: argparse.Namespace) -> BlenderController:
    previous = bpy.app.driver_namespace.pop(KEY, None)
    if previous is not None:
        previous.close()
    scene = bpy.context.scene
    started = time.perf_counter()
    source = OMEZarrSource.open(arguments.url)
    opened_s = time.perf_counter() - started
    pyramid = source.pyramid
    fixed = {axis: 0 for axis in range(pyramid.ndim - 3)}
    if pyramid.ndim >= 5:
        fixed[0], fixed[1] = arguments.time, arguments.channel
    elif pyramid.ndim == 4:
        fixed[0] = arguments.channel

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
        "Lodstone OME-Zarr volume", strength=arguments.brightness
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

    def show_tab():
        # The sidebar's tab list exists only after it has drawn once.
        for screen in bpy.data.screens:
            for area in screen.areas:
                for region in area.regions if area.type == "VIEW_3D" else ():
                    if region.type == "UI":
                        with contextlib.suppress(TypeError):
                            region.active_panel_category = "Lodstone"

    bpy.app.timers.register(show_tab, first_interval=1.0)
    bpy.app.driver_namespace[KEY] = controller
    controller.start()
    return controller


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
        started = time.perf_counter()
        plan = controller.prepare_render()
        controller.info["render_prep_s"] = round(time.perf_counter() - started, 2)
        controller.info["render_level"] = plan.target_level
        bpy.ops.render.render("INVOKE_DEFAULT")

        def restore():
            if controller.rendering or bpy.app.is_job_running("RENDER"):
                return 0.25
            controller.finish_render()
            return None

        bpy.app.timers.register(restore, first_interval=0.5)
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
        layout.operator(LODSTONE_OT_render_finest.bl_idname, icon="RENDER_STILL")


def _redraw_panel():
    if KEY not in bpy.app.driver_namespace:
        return None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
    return 0.5


def register() -> None:
    for cls in (LODSTONE_OT_render_finest, LODSTONE_PT_ome_zarr):
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
