"""Explore raw multiscale Zarr Vectors surfaces, or run a synthetic smoke demo.

See docs/mesh-streaming.md for installing the companion reader branch.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lodstone import View
from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget
from lodstone.mesh import MeshStream
from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource


def synthetic_store(directory: Path) -> Path:
    """Two explicit tessellations of one analytic surface; no decimation claim."""
    import pygfx as gfx
    from zarr_vectors.building import write_mesh

    root = directory / "surface.zarrvectors"
    for level, detail in enumerate((32, 8)):
        geometry = gfx.sphere_geometry(
            radius=1.0, width_segments=detail, height_segments=detail
        )
        destination = root if level == 0 else directory / "coarse.zarrvectors"
        write_mesh(
            destination,
            geometry.positions.data,
            geometry.indices.data.reshape(-1, 3),
            chunk_shape=(0.5, 0.5, 0.5),
        )
        if level:
            shutil.copytree(destination / "0", root / str(level))
            node = root / str(level) / "zarr.json"
            metadata = json.loads(node.read_text())
            metadata["attributes"]["zarr_vectors_level"]["level"] = level
            node.write_text(json.dumps(metadata))
    node = root / "zarr.json"
    metadata = json.loads(node.read_text())
    multiscales = metadata["attributes"]["multiscales"][0]
    coarse = dict(multiscales["datasets"][0])
    coarse["path"] = "1"
    multiscales["datasets"].append(coarse)
    node.write_text(json.dumps(metadata))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", nargs="?", help="omit to generate a two-level sphere")
    parser.add_argument(
        "--errors",
        type=float,
        nargs="+",
        default=[0.005, 0.08],
        help="world-space error estimate per stored level, finest first",
    )
    parser.add_argument("--levels", type=int, nargs="+", help="stored level numbers")
    parser.add_argument("--memory-mib", type=float, default=128.0)
    parser.add_argument(
        "--offscreen", type=Path, help="render fine/coarse PNGs and metrics here"
    )
    arguments = parser.parse_args()
    import pygfx as gfx

    temporary = tempfile.TemporaryDirectory(prefix="lodstone-mesh-")
    try:
        store = arguments.store or synthetic_store(Path(temporary.name))
        source = ZarrVectorsMeshSource(
            store, level_errors=arguments.errors, stored_levels=arguments.levels
        )
        if arguments.offscreen:
            from rendercanvas.offscreen import RenderCanvas
        else:
            from rendercanvas.auto import RenderCanvas, loop
        canvas = RenderCanvas(size=(640, 480), title="Lodstone · Zarr Vectors meshes")
        renderer = gfx.WgpuRenderer(canvas)
        scene = gfx.Scene()
        scene.add(gfx.Background(None, gfx.BackgroundMaterial("#18212b")))
        camera = gfx.PerspectiveCamera(50, 640 / 480)
        bounds = np.asarray([chunk.bounds for chunk in source.levels[0].chunks])
        extent = np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])
        center = extent.mean(axis=0)
        radius = float(np.linalg.norm(extent[1] - extent[0]) / 2)
        camera.show_object((*center, max(radius, 1e-6)))
        controller = gfx.OrbitController(camera, register_events=renderer)
        target = PyGFXMeshTarget(
            scene, memory_limit=int(arguments.memory_mib * (1 << 20))
        )
        callbacks: queue.Queue = queue.Queue()
        dispatch = (
            (lambda callback: callback()) if arguments.offscreen else callbacks.put
        )
        stream = MeshStream(
            source,
            target,
            dispatch=dispatch,
            memory_limit=int(arguments.memory_mib * (1 << 20)),
        )

        def snapshot() -> View:
            width, height = canvas.get_logical_size()
            camera.set_view_size(width, height)
            return View(
                (0, 1, 2),
                (None, None, None),
                (max(1, int(width)), max(1, int(height))),
                camera.camera_matrix.copy(),
            )

        if arguments.offscreen:
            import imageio.v3 as iio

            arguments.offscreen.mkdir(parents=True, exist_ok=True)
            phases = []
            try:
                for level in reversed(range(len(source.levels))):
                    stream.update(snapshot(), level=level)
                    stream.wait()
                    renderer.render(scene, camera)
                    pixels = np.asarray(canvas.draw())
                    iio.imwrite(arguments.offscreen / f"level-{level}.png", pixels)
                    phases.append(
                        {
                            "level": target.level,
                            "chunks": len(target.group.children),
                            "buffer_bytes": target.submitted_bytes,
                            "cache_bytes": stream.cache_bytes,
                            "reads": stream.reads,
                            "limited": stream.limited,
                        }
                    )
                (arguments.offscreen / "metrics.json").write_text(
                    json.dumps(phases, indent=2)
                )
                print(json.dumps(phases, indent=2))
            finally:
                stream.close()
                target.close()
            return

        previous = None

        def draw() -> None:
            nonlocal previous
            while not callbacks.empty():
                callbacks.get_nowait()()
            current = snapshot()
            signature = (current.viewport, current.world_to_clip.tobytes())
            if signature != previous:
                stream.update(current)
                previous = signature
            renderer.render(scene, camera)
            canvas.request_draw()

        canvas.request_draw(draw)
        try:
            loop.run()
        finally:
            stream.close()
            target.close()
        del controller
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
