"""Public Nessys image/surface demo; preprocessing preserves reference geometry."""

from __future__ import annotations

import argparse
import json
import os
import queue
from importlib import import_module
from pathlib import Path

import numpy as np

from ..biological_mesh import prepare_biological_mesh
from ..mesh import MeshStream
from ..model import View
from ..sources.zarr_vectors import ZarrVectorsMeshSource
from .pygfx_mesh import PyGFXMeshTarget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "lodstone/idr0062-mesh-v1"
    )
    parser.add_argument("--cache", type=Path, default=default_cache)
    parser.add_argument("--offscreen", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--memory-mib", type=float, default=64.0)
    args = parser.parse_args()
    report = prepare_biological_mesh(args.cache)
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    gfx = import_module("pygfx")
    backend = import_module(
        "rendercanvas.offscreen" if args.offscreen else "rendercanvas.auto"
    )
    canvas = backend.RenderCanvas(
        size=(1000, 760), title="Lodstone · Nessys nuclei · micrometers"
    )
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.Background(None, gfx.BackgroundMaterial("#101722")))
    camera = gfx.PerspectiveCamera(45, 1000 / 760)
    with np.load(report["inputs_path"]) as inputs:
        image = inputs["image"]
    spacing_xyz = np.asarray(report["spacing_zyx_um"])[::-1]
    origin_xyz = np.asarray(report["origin_xyz_um"])
    extent = origin_xyz + (np.asarray(image.shape[1:])[::-1] - 1) * spacing_xyz
    center = (origin_xyz + extent) / 2
    camera.show_object(
        (*center, float(np.linalg.norm(extent - origin_xyz) / 2)),
        view_dir=(1.0, -1.0, -0.8),
        up=(0.0, 0.0, 1.0),
    )
    controller = gfx.OrbitController(camera, register_events=renderer)
    images = gfx.Group()
    scene.add(images)
    for index, color in [(0, (0.0, 0.85, 1.0)), (1, (1.0, 0.55, 0.1))]:
        grid = np.ascontiguousarray(image[index])
        clim = (float(np.percentile(grid, 65)), float(np.percentile(grid, 99.7)))
        colormap = gfx.Texture(
            np.array([[0.0, 0.0, 0.0], color], dtype=np.float32), dim=1
        )
        volume = gfx.Volume(
            gfx.Geometry(grid=grid),
            gfx.VolumeMipMaterial(clim=clim, map=colormap, opacity=0.35, gamma=0.8),
        )
        volume.local.scale = tuple(spacing_xyz)
        volume.local.position = tuple(origin_xyz)
        images.add(volume)
    source = ZarrVectorsMeshSource(
        report["mesh_path"], level_errors=report["level_errors_um"]
    )
    limit = int(args.memory_mib * (1 << 20))
    target = PyGFXMeshTarget(scene, memory_limit=limit)
    callbacks: queue.Queue = queue.Queue()
    stream = MeshStream(
        source,
        target,
        memory_limit=limit,
        dispatch=(lambda f: f()) if args.offscreen else callbacks.put,
    )
    overlay = gfx.Scene()
    text = gfx.Text(text="", font_size=15, screen_space=True, anchor="top-left")
    overlay.add(text)
    overlay_camera = gfx.ScreenCoordsCamera()
    selected_level: int | None = None

    def snapshot() -> View:
        width, height = canvas.get_logical_size()
        camera.set_view_size(width, height)
        return View(
            (0, 1, 2),
            (None, None, None),
            (int(width), int(height)),
            camera.camera_matrix.copy(),
        )

    def paint() -> None:
        _width, height = canvas.get_logical_size()
        text.local.position = (14, height - 14, 0)
        text.set_text(
            f"Nessys / IDR0062 — 6 complete labeled nuclei — micrometers\n"
            f"LOD {stream.displayed_level} | cached {stream.cache_bytes / (1 << 20):.2f} MiB | reads {stream.reads}\n"
            "V: image  M: surface  L: auto/coarse/fine  | drag: orbit, wheel: zoom\n"
            "Blin et al. 2019 · CC BY 4.0 · surfaces derived from supplied labels"
        )
        renderer.render(scene, camera, flush=False)
        renderer.render(overlay, overlay_camera)

    @renderer.add_event_handler("key_down")
    def key(event) -> None:
        nonlocal selected_level
        if event.key.lower() == "v":
            images.visible = not images.visible
        elif event.key.lower() == "m":
            target.group.visible = not target.group.visible
        elif event.key.lower() == "l":
            selected_level = {None: 1, 1: 0, 0: None}[selected_level]
            stream.update(snapshot(), level=selected_level)
        canvas.request_draw()

    try:
        if args.offscreen:
            args.offscreen.mkdir(parents=True, exist_ok=True)
            iio = import_module("imageio.v3")
            evidence = []
            for level in (1, 0):
                stream.update(snapshot(), level=level)
                stream.wait(timeout=120)
                for name, show_image, show_mesh in [
                    ("overlay", True, True),
                    ("image", True, False),
                    ("mesh", False, True),
                ]:
                    images.visible, target.group.visible = show_image, show_mesh
                    canvas.request_draw(paint)
                    pixels = np.asarray(canvas.draw())
                    iio.imwrite(args.offscreen / f"level-{level}-{name}.png", pixels)
                evidence.append(
                    {
                        "level": level,
                        "chunks": len(target.group.children),
                        "cache_bytes": stream.cache_bytes,
                        "render_buffer_bytes": target.submitted_bytes,
                        "limited": stream.limited,
                    }
                )
            (args.offscreen / "biological-report.json").write_text(
                json.dumps(report, indent=2)
            )
            (args.offscreen / "streaming.json").write_text(
                json.dumps(evidence, indent=2)
            )
            print(json.dumps(evidence, indent=2))
            return
        previous = None

        def draw() -> None:
            nonlocal previous
            while not callbacks.empty():
                callbacks.get_nowait()()
            current = snapshot()
            signature = (
                current.viewport,
                current.world_to_clip.tobytes(),
                selected_level,
            )
            if signature != previous:
                stream.update(current, level=selected_level)
                previous = signature
            paint()
            canvas.request_draw()

        canvas.request_draw(draw)
        backend.loop.run()
    finally:
        stream.close()
        target.close()
        del controller


if __name__ == "__main__":
    main()
