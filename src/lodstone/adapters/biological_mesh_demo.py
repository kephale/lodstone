"""Public Nessys image/surface demo; preprocessing preserves reference geometry."""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from importlib import import_module
from itertools import product
from pathlib import Path

import numpy as np

from ..biological_mesh import prepare_biological_mesh
from ..mesh import MeshData, MeshStream
from ..mesh_diagnostics import ProcessMemoryMonitor, diagnostic_snapshot
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
    parser.add_argument(
        "--diagnostic-replay",
        action="store_true",
        help="record cold/warm and rapid-camera checks",
    )
    parser.add_argument(
        "--inject-failure",
        action="store_true",
        help="exercise the visible read-failure state",
    )
    args = parser.parse_args()
    report = prepare_biological_mesh(args.cache)
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    gfx = import_module("pygfx")
    memory = ProcessMemoryMonitor()
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
        report["mesh_path"], level_errors=report["level_errors_um"], track_io=True
    )
    if args.inject_failure:

        async def failed_read(level: int, key: tuple[int, ...]) -> MeshData:
            raise OSError(
                f"Injected mesh read failure at L{level}:{key} (diagnostic scenario)"
            )

        source.read = failed_read
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
    bound_group = gfx.Group()
    bound_group.visible = False
    scene.add(bound_group)
    displayed_bounds = None
    reference: PyGFXMeshTarget | None = None
    chunk_bounds = {
        (level, chunk.key): chunk.bounds
        for level, data in enumerate(source.levels)
        for chunk in data.chunks
    }

    def show_reference() -> None:
        nonlocal reference
        if reference is None:
            reference = PyGFXMeshTarget(scene, memory_limit=64 << 20)
            with np.load(report["reference_path"]) as original:
                chunks = [
                    (
                        (0, (o["label_id"],)),
                        MeshData(
                            original[f"vertices_{o['label_id']}"],
                            original[f"faces_{o['label_id']}"],
                        ),
                    )
                    for o in report["objects"]
                ]
            reference.replace(0, chunks)
        else:
            reference.group.visible = not reference.group.visible
        target.group.visible = not reference.group.visible

    def update_bounds() -> None:
        nonlocal displayed_bounds
        if stream.displayed_keys == displayed_bounds:
            return
        displayed_bounds = stream.displayed_keys
        points = []
        for key in sorted(displayed_bounds):
            corners = np.asarray(
                list(product(*zip(*chunk_bounds[key]))), dtype=np.float32
            )
            for a in range(8):
                for b in range(a + 1, 8):
                    if (a ^ b).bit_count() == 1:
                        points.extend([corners[a], corners[b]])
        bound_group.clear()
        if points:
            bound_group.add(
                gfx.Line(
                    gfx.Geometry(positions=np.asarray(points, dtype=np.float32)),
                    gfx.LineSegmentMaterial(
                        color="#80f7ff", thickness=1.0, opacity=0.5
                    ),
                )
            )

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
        update_bounds()
        status = stream.snapshot()
        io = source.io.snapshot()
        usage = memory.snapshot()
        _width, height = canvas.get_logical_size()
        text.local.position = (14, height - 14, 0)
        reference_visible = reference is not None and reference.group.visible
        active_buffer_bytes = (
            reference.submitted_bytes
            if reference is not None and reference.group.visible
            else target.submitted_bytes
        )
        text.material.color = (
            "#ff6666" if status.error else "#ffd67b" if status.limited else "white"
        )
        text.set_text(
            f"Nessys / IDR0062 — 6 complete labeled nuclei — micrometers\n"
            f"{status.state.upper()} | {'REFERENCE (not streamed)' if reference_visible else f'LOD {status.displayed_level}'} | phase {status.phase_chunks_done}/{status.phase_chunks_total}\n"
            f"Mesh cache {status.cache_bytes / (1 << 20):.2f}/{args.memory_mib:g} MiB | active mesh buffers {active_buffer_bytes / (1 << 20):.2f} MiB\n"
            f"Store {io.bytes_returned / (1 << 20):.2f} MiB ({io.metadata_bytes / (1 << 20):.2f} metadata) | {io.requests} store reads\n"
            f"Cache hits {status.cache_hits} | evictions {status.evictions} | stale reads {status.stale_reads}\n"
            f"Process RSS {usage['rss_bytes'] / (1 << 20):.1f} MiB; sampled peak {usage['sampled_peak_rss_bytes'] / (1 << 20):.1f} MiB | GPU allocations: unavailable\n"
            "V image · M surface · L auto/coarse/fine · B bounds · R retained reference\n"
            f"{status.error or ('Refinement exceeded budget; coarse phase retained' if status.limited else 'Drag to orbit; scroll to zoom')}\n"
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
        elif event.key.lower() == "b":
            bound_group.visible = not bound_group.visible
        elif event.key.lower() == "r":
            show_reference()
        canvas.request_draw()

    try:
        if args.offscreen:
            args.offscreen.mkdir(parents=True, exist_ok=True)
            iio = import_module("imageio.v3")
            evidence = []
            for level in (1, 0):
                stream.update(snapshot(), level=level)
                try:
                    stream.wait(timeout=120)
                except Exception:
                    canvas.request_draw(paint)
                    iio.imwrite(
                        args.offscreen / "failure.png", np.asarray(canvas.draw())
                    )
                    (args.offscreen / "failure.json").write_text(
                        json.dumps(
                            diagnostic_snapshot(stream, source, target, memory),
                            indent=2,
                        )
                    )
                    raise
                for name, show_image, show_mesh in [
                    ("overlay", True, True),
                    ("image", True, False),
                    ("mesh", False, True),
                ]:
                    images.visible, target.group.visible = show_image, show_mesh
                    canvas.request_draw(paint)
                    pixels = np.asarray(canvas.draw())
                    iio.imwrite(args.offscreen / f"level-{level}-{name}.png", pixels)
                evidence.append(diagnostic_snapshot(stream, source, target, memory))
            images.visible = target.group.visible = True
            bound_group.visible = True
            canvas.request_draw(paint)
            iio.imwrite(args.offscreen / "chunk-bounds.png", np.asarray(canvas.draw()))
            bound_group.visible = False
            show_reference()
            canvas.request_draw(paint)
            iio.imwrite(
                args.offscreen / "retained-reference.png", np.asarray(canvas.draw())
            )
            show_reference()
            if args.diagnostic_replay:
                replay = []
                source.io.reset()
                # Warm replay must reuse decoded fragments, not merely OS cache.
                before = source.io.snapshot().bytes_returned
                stream.update(snapshot(), level=0)
                stream.wait(timeout=120)
                replay.append(
                    {
                        "case": "warm_repeat",
                        "additional_store_bytes": source.io.snapshot().bytes_returned
                        - before,
                        **diagnostic_snapshot(stream, source, target, memory),
                    }
                )
                radius = float(np.linalg.norm(extent - origin_xyz) / 2)
                for name, scale, translation in [
                    ("zoom", 0.5, 0.0),
                    ("pan", 0.5, 25.0),
                    ("far", 32.0, 0.0),
                    ("return", 1.0, 0.0),
                ]:
                    destination = center + [translation, 0.0, 0.0]
                    camera.show_object(
                        (*destination, radius * scale),
                        view_dir=(1.0, -1.0, -0.8),
                        up=(0.0, 0.0, 1.0),
                    )
                    stream.update(snapshot())
                    stream.wait(timeout=120)
                    replay.append(
                        {
                            "case": name,
                            **diagnostic_snapshot(stream, source, target, memory),
                        }
                    )
                # A new stream gives the rapid-navigation check a cold decoded cache.
                stream.close()
                stream = MeshStream(source, target, memory_limit=limit)
                for index in range(12):
                    destination = center + [float(index % 3) * 8, 0.0, 0.0]
                    camera.show_object(
                        (*destination, radius),
                        view_dir=(1.0, -1.0, -0.8),
                        up=(0.0, 0.0, 1.0),
                    )
                    stream.update(snapshot(), level=0)
                    time.sleep(0.002)
                stream.wait(timeout=120)
                replay.append(
                    {
                        "case": "rapid_navigation",
                        **diagnostic_snapshot(stream, source, target, memory),
                    }
                )
                (args.offscreen / "camera-replay.json").write_text(
                    json.dumps(replay, indent=2)
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
        source.close()
        if reference is not None:
            reference.close()
        memory.close()
        del controller


if __name__ == "__main__":
    main()
