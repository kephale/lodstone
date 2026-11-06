"""Headless checks for the in-memory Blender grid adapter.

Run with Blender 5.2+ (Lodstone must be importable, e.g. from a built wheel)::

    blender -b --factory-startup --python examples/blender_native_grid_check.py \
        -- /path/to/lodstone.whl /path/to/output-dir

Checks:
1. Every grid voxel equals the published array value (index order, exactness).
2. Grid sampling at expected physical voxel centers returns the same values
   (bounds, anisotropic spacing and origin).
3. A synthetic pyramid streams through ``BlenderController`` with no files and
   renders in Cycles and EEVEE.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if argv and argv[0].endswith(".whl"):
    sys.path.insert(0, argv.pop(0))
OUT = Path(argv[0] if argv else ".").resolve()
OUT.mkdir(parents=True, exist_ok=True)

import bpy
import numpy as np

from lodstone.adapters import blender as lb
from lodstone.sources import ArrayPyramidSource

results: dict[str, object] = {}
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def probe(target, values_shape, positions=None):
    """Read grid values back through geometry nodes on a probe mesh."""
    nz, ny, nx = values_shape
    count = nx * ny * nz if positions is None else len(positions)
    mesh = bpy.data.meshes.new("probe")
    mesh.vertices.add(count)
    if positions is not None:
        mesh.vertices.foreach_set("co", np.asarray(positions, np.float32).ravel())
    obj = bpy.data.objects.new("probe", mesh)
    scene.collection.objects.link(obj)
    group = bpy.data.node_groups.new("probe", "GeometryNodeTree")
    group.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    group.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes, link = group.nodes, group.links.new
    gi, go = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = target
    info.transform_space = "RELATIVE"
    get = nodes.new("GeometryNodeGetNamedGrid")
    get.data_type = "FLOAT"
    get.inputs["Name"].default_value = lb.GRID_NAME
    link(info.outputs["Geometry"], get.inputs["Volume"])
    if positions is None:
        index = nodes.new("GeometryNodeInputIndex")

        def imath(op, a, b):
            n = nodes.new("FunctionNodeIntegerMath")
            n.operation = op
            link(a, n.inputs[0])
            n.inputs[1].default_value = b
            return n.outputs[0]

        x = imath("MODULO", index.outputs[0], nx)
        y = imath("MODULO", imath("DIVIDE_FLOOR", index.outputs[0], nx), ny)
        z = imath("DIVIDE_FLOOR", index.outputs[0], nx * ny)
        sample = nodes.new("GeometryNodeSampleGridIndex")
        sample.data_type = "FLOAT"
        link(get.outputs["Grid"], sample.inputs["Grid"])
        link(x, sample.inputs["X"])
        link(y, sample.inputs["Y"])
        link(z, sample.inputs["Z"])
    else:
        position = nodes.new("GeometryNodeInputPosition")
        sample = nodes.new("GeometryNodeSampleGrid")
        sample.data_type = "FLOAT"
        link(get.outputs["Grid"], sample.inputs["Grid"])
        link(position.outputs[0], sample.inputs["Position"])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.inputs["Name"].default_value = "probe"
    link(gi.outputs[0], store.inputs["Geometry"])
    link(sample.outputs["Value"], store.inputs["Value"])
    link(store.outputs[0], go.inputs[0])
    obj.modifiers.new("probe", "NODES").node_group = group
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph).data
    out = np.empty(count, np.float32)
    evaluated.attributes["probe"].data.foreach_get("value", out)
    bpy.data.objects.remove(obj)
    return out


# 1 + 2: exact values and physical placement of one handle ------------------
rng = np.random.default_rng(0)
shape = (5, 7, 11)  # (z, y, x), deliberately unequal
data = rng.integers(0, 1000, size=shape).astype(np.uint16)
root = bpy.data.objects.new("root", None)
scene.collection.objects.link(root)
canvas = lb._BlenderCanvas(
    root, scene.collection, lb.emission_material(), (0, 1, 2), (None, None, None)
)
handle = canvas.add_volume(data)
scales, origins = (2.0, 0.5, 0.25), (10.0, -3.0, 4.0)  # (z, y, x)
handle.set_world_transform(scales, origins)
handle.set_clims((0.0, 1000.0))
expected = data.astype(np.float32) / 1000.0
got = probe(handle.object, shape)
results["index_order_max_abs_error"] = float(np.abs(got - expected.ravel()).max())

z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
centers = np.stack(
    [
        origins[2] + x * scales[2],
        origins[1] + y * scales[1],
        origins[0] + z * scales[0],
    ],
    axis=-1,
).reshape(-1, 3)
got = probe(handle.object, shape, centers)
results["physical_centers_max_abs_error"] = float(np.abs(got - expected.ravel()).max())

# Changing data in place republishes without new objects.
handle.set_data((data[::-1]).copy())
got = probe(handle.object, shape)
results["update_in_place_max_abs_error"] = float(
    np.abs(got - (data[::-1].astype(np.float32) / 1000).ravel()).max()
)
handle.remove()
bpy.data.objects.remove(root)

# 3: stream a synthetic pyramid and render -----------------------------------
full = (96, 192, 192)
zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n) for n in full), indexing="ij")
shell = np.exp(-((np.sqrt(xx**2 + yy**2 + zz**2) - 0.6) ** 2) / 0.004)
stripes = 0.5 + 0.5 * np.sin(14 * np.arctan2(yy, xx))
volume = (1000 * shell * stripes).astype(np.uint16)
levels, transforms = [], []
for k in range(3):
    f = 2**k
    levels.append(volume[::f, ::f, ::f].copy())
    transforms.append(np.diag([2.0 * f, 1.0 * f, 1.0 * f, 1.0]))
    transforms[-1][:3, 3] = (f - 1) / 2 * np.array([2.0, 1.0, 1.0])
source = ArrayPyramidSource(levels, axes=("z", "y", "x"), transforms=transforms)

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.resolution_x, scene.render.resolution_y = 320, 240
scene.world = bpy.data.worlds.new("world")
controller = lb.BlenderController(
    source, clims=(0, 1000), memory_limit=8 * 1024**2, render_memory_limit=64 * 1024**2
)
controller.root.scale = (0.02, 0.02, 0.02)
from mathutils import Vector

center = Vector((96 * 0.02, 96 * 0.02, 96 * 0.02))
camera.location = center + Vector((5, -5, 4))
camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
started = time.perf_counter()
plan = controller.prepare_render(timeout=60)
results["render_plan_target_level"] = plan.target_level
results["prepare_render_s"] = round(time.perf_counter() - started, 3)
results["controller"] = {
    k: (list(v) if isinstance(v, tuple) else v) for k, v in controller.summary().items()
}
for engine in ("CYCLES", "BLENDER_EEVEE"):
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = 16
    scene.render.filepath = str(OUT / f"stream-{engine.lower()}.png")
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    results[f"render_{engine.lower()}_s"] = round(time.perf_counter() - started, 2)
controller.close()
results["objects_after_close"] = [
    o.name for o in bpy.data.objects if "Lodstone volume" in o.name
]

ok = (
    results["index_order_max_abs_error"] < 1e-6
    and results["physical_centers_max_abs_error"] < 1e-6
    and results["update_in_place_max_abs_error"] < 1e-6
    and results["render_plan_target_level"] == 0
    and not results["objects_after_close"]
)
results["ok"] = bool(ok)
(OUT / "check.json").write_text(json.dumps(results, indent=2, default=str))
print("LODSTONE_CHECK", json.dumps(results, default=str))
