"""Stream multiscale arrays into native Blender volume grids, without files.

Each dense phase is copied into an in-memory float attribute on a carrier mesh.
A geometry-node group turns that attribute into a native float volume grid
(``Field to Grid`` over a ``Cube Grid Topology``), so Cycles, EEVEE and the
viewport render it like any other Blender volume. Nothing is written to disk
and no OpenVDB file is involved.

Importing this module does not import ``bpy``. Every Blender call happens on
Blender's main thread: chunk reads and planning run on Lodstone's background
threads, and their results reach Blender through :class:`MainThreadQueue`.

Coordinates: data axes ``(z, y, x)`` in the pyramid's physical units map to the
local ``(Z, Y, X)`` axes of a root empty. Move, rotate or scale the root to
place the volume in the scene; every level is parented to it.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np

from ..model import Plan, Pyramid, View
from ..source import Source
from .dense import DenseController, DensePublication, DenseTarget

__all__ = [
    "BlenderController",
    "BlenderTarget",
    "MainThreadQueue",
    "blender_camera_snapshot",
    "blender_viewport_snapshot",
    "grid_node_group",
]

GRID_NAME = "density"
NODE_GROUP_NAME = "Lodstone native grid"
_CARRIER_ATTRIBUTE = "lodstone_value"

# Maps data-order (z, y, x, 1) to Blender local (X, Y, Z, 1).
_DATA_TO_LOCAL = np.array(
    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def _bpy() -> Any:
    import bpy  # pyright: ignore[reportMissingImports]

    return bpy


def _matrix(value: Any) -> np.ndarray:
    return np.array([list(row) for row in value], dtype=np.float64)


def _snapshot(
    world_to_clip_local: np.ndarray,
    eye_local: np.ndarray | None,
    viewport: tuple[int, int],
    ndim: int,
    displayed_axes: tuple[int, ...],
    index: tuple[int | None, ...],
) -> View:
    if len(displayed_axes) != 3 or len(index) != ndim:
        raise ValueError("Blender volumes display exactly three data axes")
    eye = None
    if eye_local is not None:
        eye = (float(eye_local[2]), float(eye_local[1]), float(eye_local[0]))
    return View(
        displayed_axes,
        index,
        (max(1, int(viewport[0])), max(1, int(viewport[1]))),
        world_to_clip_local @ _DATA_TO_LOCAL,
        eye=eye,
    )


def blender_viewport_snapshot(
    region: Any,
    region_3d: Any,
    root: Any,
    displayed_axes: tuple[int, ...],
    index: tuple[int | None, ...],
) -> View:
    """Capture a 3-D viewport as a Lodstone view in the root's data frame."""
    root_matrix = _matrix(root.matrix_world)
    world_to_clip = _matrix(region_3d.perspective_matrix) @ root_matrix
    eye = None
    if region_3d.is_perspective:
        world_eye = _matrix(region_3d.view_matrix.inverted())[:3, 3]
        eye = (np.linalg.inv(root_matrix) @ np.append(world_eye, 1.0))[:3]
    return _snapshot(
        world_to_clip,
        eye,
        (region.width, region.height),
        len(index),
        displayed_axes,
        index,
    )


def blender_camera_snapshot(
    scene: Any,
    root: Any,
    displayed_axes: tuple[int, ...],
    index: tuple[int | None, ...],
) -> View:
    """Capture the scene's render camera at the final output resolution."""
    camera = scene.camera
    if camera is None:
        raise ValueError("the scene has no render camera")
    render = scene.render
    scale = render.resolution_percentage / 100
    width = max(1, round(render.resolution_x * scale))
    height = max(1, round(render.resolution_y * scale))
    depsgraph = _bpy().context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )
    root_matrix = _matrix(root.matrix_world)
    camera_matrix = _matrix(camera.matrix_world)
    world_to_clip = _matrix(projection) @ np.linalg.inv(camera_matrix) @ root_matrix
    eye = None
    if camera.data.type == "PERSP":
        eye = (np.linalg.inv(root_matrix) @ camera_matrix[:, 3])[:3]
    return _snapshot(
        world_to_clip, eye, (width, height), len(index), displayed_axes, index
    )


class MainThreadQueue:
    """Run callbacks from Lodstone threads on Blender's main thread.

    A persistent ``bpy.app.timers`` callback drains the queue. :meth:`drain`
    can also be called directly on the main thread, for example while waiting
    for a final render's data.
    """

    def __init__(self, interval: float = 0.02, budget_seconds: float = 0.03) -> None:
        self.interval = interval
        self.budget_seconds = budget_seconds
        self.paused = False
        self.errors: list[BaseException] = []
        self._items: deque[Callable[[], None]] = deque()
        self._lock = Lock()
        self._registered = False

    def __call__(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._items.append(callback)

    def drain(self, budget_seconds: float | None = None) -> int:
        budget = self.budget_seconds if budget_seconds is None else budget_seconds
        deadline = time.perf_counter() + budget
        count = 0
        while True:
            with self._lock:
                if not self._items:
                    return count
                callback = self._items.popleft()
            try:
                callback()
            except Exception as error:  # noqa: BLE001 - surface, keep the timer alive
                self.errors.append(error)
            count += 1
            if time.perf_counter() >= deadline:
                return count

    def _tick(self) -> float | None:
        if not self._registered:
            return None
        if not self.paused:
            self.drain()
        return self.interval

    def start(self) -> None:
        if not self._registered:
            self._registered = True
            _bpy().app.timers.register(self._tick, first_interval=self.interval)

    def stop(self) -> None:
        bpy = _bpy()
        self._registered = False
        if bpy.app.timers.is_registered(self._tick):
            bpy.app.timers.unregister(self._tick)


def grid_node_group() -> Any:
    """Return the shared node group building a native grid from an attribute.

    Inputs: per-axis resolution, the cell-boundary box in root-local units,
    contrast limits mapped to ``[0, 1]``, and the material. The voxel at
    ``(x, y, z)`` reads carrier vertex ``z * nx * ny + y * nx + x``, the C-order
    ravel of a ``(z, y, x)`` array; integer math keeps indices exact.
    """
    bpy = _bpy()
    existing = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if existing is not None:
        return existing
    group = bpy.data.node_groups.new(NODE_GROUP_NAME, "GeometryNodeTree")
    interface = group.interface
    interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for axis in "XYZ":
        interface.new_socket(
            f"Resolution {axis}", in_out="INPUT", socket_type="NodeSocketInt"
        )
    interface.new_socket("Bounds Min", in_out="INPUT", socket_type="NodeSocketVector")
    interface.new_socket("Bounds Max", in_out="INPUT", socket_type="NodeSocketVector")
    interface.new_socket("Low", in_out="INPUT", socket_type="NodeSocketFloat")
    high = interface.new_socket("High", in_out="INPUT", socket_type="NodeSocketFloat")
    high.default_value = 1.0
    interface.new_socket("Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, link = group.nodes, group.links.new
    source = nodes.new("NodeGroupInput")
    output = nodes.new("NodeGroupOutput")

    def integer(operation: str, *inputs: Any) -> Any:
        node = nodes.new("FunctionNodeIntegerMath")
        node.operation = operation
        for socket, value in zip(node.inputs, inputs, strict=False):
            link(value, socket)
        return node.outputs[0]

    voxel = nodes.new("GeometryNodeInputVoxelIndex")
    row = integer(
        "MULTIPLY_ADD",
        voxel.outputs["Y"],
        source.outputs["Resolution X"],
        voxel.outputs["X"],
    )
    plane = integer(
        "MULTIPLY", source.outputs["Resolution X"], source.outputs["Resolution Y"]
    )
    flat = integer("MULTIPLY_ADD", voxel.outputs["Z"], plane, row)

    attribute = nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = "FLOAT"
    attribute.inputs["Name"].default_value = _CARRIER_ATTRIBUTE
    sample = nodes.new("GeometryNodeSampleIndex")
    sample.data_type = "FLOAT"
    sample.domain = "POINT"
    link(source.outputs["Geometry"], sample.inputs["Geometry"])
    link(attribute.outputs["Attribute"], sample.inputs["Value"])
    link(flat, sample.inputs["Index"])

    normalize = nodes.new("ShaderNodeMapRange")
    normalize.clamp = True
    link(sample.outputs[0], normalize.inputs["Value"])
    link(source.outputs["Low"], normalize.inputs["From Min"])
    link(source.outputs["High"], normalize.inputs["From Max"])

    topology = nodes.new("GeometryNodeCubeGridTopology")
    link(source.outputs["Bounds Min"], topology.inputs["Bounds Min"])
    link(source.outputs["Bounds Max"], topology.inputs["Bounds Max"])
    for axis in "XYZ":
        link(
            source.outputs[f"Resolution {axis}"], topology.inputs[f"Resolution {axis}"]
        )

    to_grid = nodes.new("GeometryNodeFieldToGrid")
    to_grid.data_type = "FLOAT"
    to_grid.grid_items.new("FLOAT", GRID_NAME)
    link(topology.outputs["Topology"], to_grid.inputs["Topology"])
    link(normalize.outputs["Result"], to_grid.inputs[GRID_NAME])

    store = nodes.new("GeometryNodeStoreNamedGrid")
    store.data_type = "FLOAT"
    store.inputs["Name"].default_value = GRID_NAME
    link(to_grid.outputs[GRID_NAME], store.inputs["Grid"])

    material = nodes.new("GeometryNodeSetMaterial")
    link(store.outputs["Volume"], material.inputs["Geometry"])
    link(source.outputs["Material"], material.inputs["Material"])
    link(material.outputs["Geometry"], output.inputs["Geometry"])
    return group


LEVEL_COLORS = (
    (0.15, 1.0, 0.3),  # level 0 (full resolution): green
    (1.0, 0.85, 0.1),  # level 1: yellow
    (1.0, 0.2, 0.9),  # level 2: magenta
    (0.2, 0.7, 1.0),  # level 3+: blue
)


def emission_material(
    name: str = "Lodstone volume",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    strength: float = 8.0,
    density: float = 0.0,
) -> Any:
    """Return a fluorescence-style material reading the grid.

    Emission is ``strength * value``. ``density`` adds absorption from the same
    value (napari's "attenuated" look): near structure occludes far structure
    instead of integrating into haze. The node labelled "Color by level" mixes
    the emission color toward a per-level tint read from each volume object's
    ``lodstone_level`` property, so resolution is visible at a glance.
    """
    bpy = _bpy()
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    nodes, link = tree.nodes, tree.links.new
    nodes.clear()
    value = nodes.new("ShaderNodeAttribute")
    value.attribute_name = GRID_NAME
    emission = nodes.new("ShaderNodeMath")
    emission.operation = "MULTIPLY"
    emission.label = "Brightness"
    emission.inputs[1].default_value = strength
    absorption = nodes.new("ShaderNodeMath")
    absorption.operation = "MULTIPLY"
    absorption.label = "Absorption"
    absorption.inputs[1].default_value = density

    level = nodes.new("ShaderNodeAttribute")
    level.attribute_type = "OBJECT"
    level.attribute_name = "lodstone_level"
    to_ramp = nodes.new("ShaderNodeMath")
    to_ramp.operation = "DIVIDE"
    to_ramp.inputs[1].default_value = len(LEVEL_COLORS)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    # Constant ramp sampled at level / len(LEVEL_COLORS): stop k sits half a
    # step below level k, so each level lands on its own color.
    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[1].position = 0.5 / len(LEVEL_COLORS)
    for index in range(2, len(LEVEL_COLORS)):
        elements.new((index - 0.5) / len(LEVEL_COLORS))
    ordered = sorted(elements, key=lambda element: element.position)
    for element, tint in zip(ordered, LEVEL_COLORS, strict=True):
        element.color = (*tint, 1.0)
    toggle = nodes.new("ShaderNodeValue")
    toggle.label = "Color by level"
    toggle.outputs[0].default_value = 0.0
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*color, 1.0)

    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # absorb only
    out = nodes.new("ShaderNodeOutputMaterial")
    link(value.outputs["Fac"], emission.inputs[0])
    link(value.outputs["Fac"], absorption.inputs[0])
    link(level.outputs["Fac"], to_ramp.inputs[0])
    link(to_ramp.outputs[0], ramp.inputs["Fac"])
    link(toggle.outputs[0], mix.inputs["Factor"])
    link(ramp.outputs["Color"], mix.inputs["B"])
    link(mix.outputs["Result"], volume.inputs["Emission Color"])
    link(emission.outputs[0], volume.inputs["Emission Strength"])
    link(absorption.outputs[0], volume.inputs["Density"])
    link(volume.outputs["Volume"], out.inputs["Volume"])
    return material


def _set_input(modifier: Any, name: str, value: Any) -> None:
    item = next(
        item
        for item in modifier.node_group.interface.items_tree
        if getattr(item, "in_out", None) == "INPUT" and item.name == name
    )
    properties = getattr(modifier, "properties", None)
    if properties is not None and hasattr(properties, "inputs"):
        getattr(properties.inputs, item.identifier).value = value  # Blender 5.x
    else:
        modifier[item.identifier] = value


class _GridHandle:
    """One pyramid level: a carrier mesh whose modifier emits a native grid."""

    def __init__(self, canvas: _BlenderCanvas, data: np.ndarray) -> None:
        bpy = _bpy()
        self._canvas = canvas
        level = len(canvas.handles_created)
        canvas.handles_created.append(self)
        self.mesh = bpy.data.meshes.new(f"Lodstone carrier {level}")
        self.object = bpy.data.objects.new(f"Lodstone volume {level}", self.mesh)
        self.object.parent = canvas.root
        canvas.collection.objects.link(self.object)
        self.modifier = self.object.modifiers.new("Lodstone native grid", "NODES")
        self.modifier.node_group = canvas.node_group
        _set_input(self.modifier, "Material", canvas.material)
        self.shape: tuple[int, int, int] = (0, 0, 0)
        self._scales = (1.0, 1.0, 1.0)
        self._origins = (0.0, 0.0, 0.0)
        self.set_data(data)

    def set_data(self, data: np.ndarray) -> None:
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError("Blender grids need a 3-D array")
        values = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        mesh = self.mesh
        if len(mesh.vertices) != values.size:
            mesh.clear_geometry()
            mesh.vertices.add(values.size)
        attribute = mesh.attributes.get(_CARRIER_ATTRIBUTE)
        if attribute is None:
            attribute = mesh.attributes.new(_CARRIER_ATTRIBUTE, "FLOAT", "POINT")
        attribute.data.foreach_set("value", values)
        mesh.update()
        shape = tuple(int(n) for n in data.shape)
        if shape != self.shape:
            self.shape = shape  # type: ignore[assignment]
            nz, ny, nx = shape
            for axis, size in zip("XYZ", (nx, ny, nz), strict=True):
                _set_input(self.modifier, f"Resolution {axis}", size)
            self._apply_bounds()
        self._canvas.submitted(values.nbytes)

    def set_world_transform(
        self, scales: tuple[float, ...], origins: tuple[float, ...]
    ) -> None:
        self._scales = tuple(float(s) for s in scales[-3:])  # type: ignore[assignment]
        self._origins = tuple(float(o) for o in origins[-3:])  # type: ignore[assignment]
        self._apply_bounds()

    def _apply_bounds(self) -> None:
        # Cube Grid Topology maps voxel index i to Bounds Min + i * size, with
        # size = (Bounds Max - Bounds Min) / resolution. OpenVDB index i is a
        # voxel center, matching Lodstone origins (first-voxel centers).
        scales = np.asarray(self._scales[::-1])
        low = np.asarray(self._origins[::-1])
        high = low + scales * np.asarray(self.shape[::-1], dtype=np.float64)
        _set_input(self.modifier, "Bounds Min", tuple(low))
        _set_input(self.modifier, "Bounds Max", tuple(high))
        self.object.update_tag()

    def set_clims(self, clims: tuple[float, float]) -> None:
        _set_input(self.modifier, "Low", float(clims[0]))
        _set_input(self.modifier, "High", float(clims[1]))

    def set_visible(self, visible: bool) -> None:
        self.object.hide_viewport = not visible
        self.object.hide_render = not visible

    def remove(self) -> None:
        bpy = _bpy()
        try:
            bpy.data.objects.remove(self.object)
            bpy.data.meshes.remove(self.mesh)
        except ReferenceError:
            pass


class _CameraSignal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def connect(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback: Callable[[], None]) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback()


class _BlenderCanvas:
    """Dense canvas backed by native volume grids under one root empty."""

    def __init__(
        self,
        root: Any,
        collection: Any,
        material: Any,
        displayed_axes: tuple[int, ...],
        index: tuple[int | None, ...],
    ) -> None:
        self.root = root
        self.collection = collection
        self.material = material
        self.node_group = grid_node_group()
        self.displayed_axes = displayed_axes
        self.index = index
        self.cameraChanged = _CameraSignal()
        self.handles_created: list[_GridHandle] = []
        self.render_view: View | None = None
        self.submitted_bytes = 0

    def submitted(self, nbytes: int) -> None:
        self.submitted_bytes += nbytes

    def viewport(self) -> tuple[Any, Any] | None:
        """Return the largest 3-D viewport's ``(region, region_3d)``."""
        best = None
        for window in _bpy().context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for region in area.regions:
                    if region.type == "WINDOW" and region.width and region.height:
                        size = region.width * region.height
                        if best is None or size > best[0]:
                            best = (size, region, area.spaces.active.region_3d)
        return None if best is None else (best[1], best[2])

    def view(self) -> View | None:
        if self.render_view is not None:
            return self.render_view
        found = self.viewport()
        if found is None:
            return None
        return blender_viewport_snapshot(
            found[0], found[1], self.root, self.displayed_axes, self.index
        )

    def camera_state(self) -> tuple[tuple[int, int], np.ndarray]:
        view = self.view()
        if view is None:
            raise RuntimeError("no 3-D viewport is open")
        return view.viewport, view.world_to_clip

    def set_ndim(self, ndim: int) -> None:
        if ndim != 3:
            raise ValueError("Blender native grids display three axes")

    def add_image(self, data: np.ndarray, *, reset_range: bool = False) -> _GridHandle:
        return _GridHandle(self, data)

    def add_volume(self, data: np.ndarray, *, reset_range: bool = False) -> _GridHandle:
        return _GridHandle(self, data)

    def set_range(self) -> None:
        return None  # never move the user's view

    def refresh(self) -> None:
        for window in _bpy().context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()


class BlenderTarget(DenseTarget):
    """Dense target publishing each level as a native Blender volume grid."""


class BlenderController(DenseController):
    """Stream a Lodstone source into native Blender volumes, in memory.

    ``fixed_index`` selects non-displayed axes such as time and channel; the
    last three axes are displayed as ``(z, y, x)``. Camera changes are polled
    on a Blender timer, chunks are read on background threads, and every
    Blender data change happens on the main thread.

    Final renders: call :meth:`prepare_render` (or the bundled operator) before
    rendering. It plans the scene camera with ``render_memory_limit``, waits for
    the chunks, and Lodstone's timers stay paused while Blender renders so the
    render thread never sees data change underneath it.
    """

    def __init__(
        self,
        source: Source,
        *,
        root: Any = None,
        collection: Any = None,
        material: Any = None,
        fixed_index: dict[int, int] | None = None,
        memory_limit: int = 32 * 1024**2,
        render_memory_limit: int = 128 * 1024**2,
        poll_interval: float = 0.05,
        clims: tuple[float, float] | None = None,
        **options: Any,
    ) -> None:
        bpy = _bpy()
        ndim = source.pyramid.ndim
        if ndim < 3:
            raise ValueError("Blender streaming needs at least three axes")
        fixed = dict(fixed_index or {})
        displayed = tuple(range(ndim - 3, ndim))
        if any(axis in fixed for axis in displayed):
            raise ValueError("fixed_index may only select non-displayed axes")
        index = tuple(
            None if axis in displayed else int(fixed.get(axis, 0))
            for axis in range(ndim)
        )
        if collection is None:
            collection = bpy.context.scene.collection
        if root is None:
            root = bpy.data.objects.new("Lodstone root", None)
            root.empty_display_type = "CUBE"
            collection.objects.link(root)
        self.root = root
        self.collection = collection
        self.memory_limit = int(memory_limit)
        self.render_memory_limit = int(render_memory_limit)
        self.queue = MainThreadQueue()
        canvas = _BlenderCanvas(
            root,
            collection,
            material if material is not None else emission_material(),
            displayed,
            index,
        )
        self.blender_canvas = canvas
        options.setdefault("block_shape", (64, 128, 128))
        super().__init__(
            canvas,
            source,
            dispatch=self.queue,
            memory_limit=memory_limit,
            fit_on_first_publication=False,
            clims=clims,
            **options,
        )
        self.poll_interval = poll_interval
        self.rendering = False
        self.error = ""
        self.last_publication: DensePublication | None = None
        self.publish_ms: deque[float] = deque(maxlen=64)
        self._last_state: tuple[tuple[int, int], np.ndarray] | None = None
        self._handlers: list[tuple[Any, Any]] = []
        self._polling = False

    def _make_target(
        self, canvas: Any, pyramid: Pyramid, **options: Any
    ) -> BlenderTarget:
        presented = options.pop("on_presented", None)

        def on_presented(publication: DensePublication) -> None:
            self.last_publication = publication
            if presented is not None:
                presented(publication)

        target = BlenderTarget(canvas, pyramid, on_presented=on_presented, **options)
        publish = target._publish

        def timed(view: View, plan: Plan, publication: DensePublication) -> bool:
            started = time.perf_counter()
            try:
                return publish(view, plan, publication)
            finally:
                self.publish_ms.append((time.perf_counter() - started) * 1000)
                for level, handle in target.handles.items():
                    obj = getattr(handle, "object", None)
                    if obj is not None and obj.get("lodstone_level") != level:
                        obj["lodstone_level"] = level

        target._publish = timed  # type: ignore[method-assign]
        return target

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Plan the current viewport and begin polling camera changes."""
        bpy = _bpy()
        self.queue.start()
        view = self.blender_canvas.view()
        if view is not None:
            self._last_state = (view.viewport, view.world_to_clip)
            self.update(view)
        self._polling = True
        bpy.app.timers.register(self._poll, first_interval=self.poll_interval)

        def render_started(*_args: Any) -> None:
            # Runs on Blender's render thread: only set flags here.
            self.rendering = True
            self.queue.paused = True

        def render_finished(*_args: Any) -> None:
            self.rendering = False
            self.queue.paused = False

        for handlers, callback in (
            (bpy.app.handlers.render_init, render_started),
            (bpy.app.handlers.render_complete, render_finished),
            (bpy.app.handlers.render_cancel, render_finished),
            (bpy.app.handlers.load_pre, lambda *_: self.close()),
        ):
            handlers.append(callback)
            self._handlers.append((handlers, callback))

    def _poll(self) -> float | None:
        if not self._polling:
            return None
        if self.rendering:
            return self.poll_interval
        try:
            view = self.blender_canvas.view()
            if view is not None and self._last_view is None:
                # No viewport existed at start(): plan the first one directly.
                self._last_state = (view.viewport, view.world_to_clip)
                self.update(view)
            elif view is not None:
                state = (view.viewport, view.world_to_clip)
                last = self._last_state
                if (
                    last is None
                    or state[0] != last[0]
                    or not np.allclose(state[1], last[1], rtol=1e-7, atol=1e-9)
                ):
                    self._last_state = state
                    self.blender_canvas.cameraChanged.emit()
            if self.queue.errors:
                raise self.queue.errors.pop(0)
        except Exception as error:  # noqa: BLE001 - shown in the UI, keep polling
            self.error = f"{type(error).__name__}: {error}"
        return self.poll_interval

    # -- final render -----------------------------------------------------
    def prepare_render(self, timeout: float = 120.0) -> Plan:
        """Load the scene camera's view at the render budget; block until done."""
        bpy = _bpy()
        canvas = self.blender_canvas
        view = blender_camera_snapshot(
            bpy.context.scene, self.root, canvas.displayed_axes, canvas.index
        )
        canvas.render_view = view
        self.set_focus_policy(memory_limit=self.render_memory_limit, replan=False)
        plan = self.update(view)
        generation = self.stream.status.generation
        deadline = time.perf_counter() + timeout
        while True:
            self.queue.drain(budget_seconds=0.25)
            status = self.stream.status
            if status.generation >= generation and status.state in (
                "complete",
                "failed",
            ):
                self.queue.drain(budget_seconds=5.0)
                break
            if time.perf_counter() > deadline:
                raise TimeoutError("final-render chunks did not arrive in time")
            time.sleep(0.01)
        if status.state == "failed":
            raise RuntimeError(f"final-render load failed: {status.error!r}")
        return plan

    def finish_render(self) -> None:
        """Return to the interactive budget and the viewport's view."""
        self.blender_canvas.render_view = None
        self.set_focus_policy(memory_limit=self.memory_limit, replan=False)
        view = self.blender_canvas.view()
        if view is not None:
            self._last_state = (view.viewport, view.world_to_clip)
            self.update(view)

    # -- diagnostics ------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        status = self.stream.status
        publication = self.last_publication
        timings = list(self.publish_ms)
        return {
            "state": status.state,
            "level": None if publication is None else publication.level,
            "shape": None if publication is None else publication.data.shape,
            "region_start": None if publication is None else publication.region.start,
            "levels_shown": sorted(self.target.handles),
            "resident_bytes": self.target.resident.nbytes,
            "bytes_read": status.bytes_read,
            "progress": status.progress,
            "publish_ms_last": timings[-1] if timings else None,
            "publish_ms_max": max(timings) if timings else None,
            "error": self.error,
        }

    def close(self) -> None:
        if getattr(self, "_closing", False):
            return
        self._closing = True
        bpy = _bpy()
        self._polling = False
        if bpy.app.timers.is_registered(self._poll):
            bpy.app.timers.unregister(self._poll)
        for handlers, callback in self._handlers:
            if callback in handlers:
                handlers.remove(callback)
        self._handlers.clear()
        self.queue.stop()
        super().close()
