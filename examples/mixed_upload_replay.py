"""Measure contention between Lodstone image tiles and mesh phases."""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
from upload_replay import CANVAS, TILE, TRACE, add_texture

from lodstone.adapters.pygfx_mesh import PyGFXMeshTarget
from lodstone.mesh import MeshData
from lodstone.sources import OMEZarrSource

RESIDENCY = 8 << 20
FRAME_SECONDS = 0.002
COARSE_RING = 12
FINE_RING = 96


def mesh_disc(width, height, count):
    """Tessellate a disk registered to the image's physical coordinates."""
    theta = np.arange(count) * (2 * np.pi / count)
    radii = np.linspace(0.01, 1, count)
    radius = min(width, height) * 0.19
    x = np.outer(radii, np.cos(theta)) * radius + 0.65 * width
    y = np.outer(radii, np.sin(theta)) * radius + 0.55 * height
    vertices = np.stack((x, y, np.zeros_like(x) + 2), axis=-1).reshape(-1, 3)
    inner = np.arange(count * (count - 1), dtype=np.uint32)
    next_column = inner + 1 - count * ((inner + 1) % count == 0)
    faces = np.concatenate(
        [
            np.stack((inner, next_column, inner + count), axis=1),
            np.stack((next_column, next_column + count, inner + count), axis=1),
        ]
    )
    return MeshData(vertices.astype(np.float32), faces.astype(np.uint32))


def trial(source, renderer, canvas, shared, reference):
    image_group, scene = gfx.Group(), gfx.Scene()
    scene.add(image_group)
    camera = gfx.OrthographicCamera()
    camera.local.z = 100
    canvas.request_draw(lambda: renderer.render(scene, camera))
    target = PyGFXMeshTarget(scene, memory_limit=RESIDENCY)
    fine = source.arrays[0]
    coarse = source.arrays[-1]
    depth, ny, nx = fine.shape
    _, dy, dx = np.diag(source.pyramid.levels[0].voxel_to_world)[:-1]
    _, sy, sx = np.diag(source.pyramid.levels[-1].voxel_to_world)[:-1]
    width, height = nx * dx, ny * dy
    images = np.asarray(fine[depth // 2], dtype=np.float32)
    backdrop = np.asarray(coarse[coarse.shape[0] // 2], dtype=np.float32)
    meshes = [
        (0, mesh_disc(width, height, FINE_RING)),
        (1, mesh_disc(width, height, COARSE_RING)),
    ]
    frame_times, waits, coverage_errors, image_errors, settled = [], [], [], [], []
    uploads = peak = 0
    try:
        for pose_index, (cx, cy, fraction) in enumerate(TRACE):
            image_group.clear()
            target.close()
            camera.width, camera.height = width * fraction, height * fraction
            camera.local.x, camera.local.y = cx * width, cy * height
            add_texture(image_group, backdrop, 0, 0, sx, sy, 0)
            target.replace(1, [((1, (0,)), meshes[1][1])])
            mesh_bytes = target.submitted_bytes
            image_bytes = backdrop.nbytes
            uploads += image_bytes + mesh_bytes
            canvas.draw()
            x0 = max(0, int((cx - fraction / 2) * nx) // TILE * TILE)
            x1 = min(nx, int(np.ceil((cx + fraction / 2) * nx)))
            y0 = max(0, int((cy - fraction / 2) * ny) // TILE * TILE)
            y1 = min(ny, int(np.ceil((cy + fraction / 2) * ny)))
            tiles = deque(
                (y, x, np.ascontiguousarray(images[y : y + TILE, x : x + TILE]))
                for y in range(y0, y1, TILE)
                for x in range(x0, x1, TILE)
            )
            ready = perf_counter()
            first_tile, mesh_presented = None, None
            refinement_pending = True
            while tiles or refinement_pending:
                start = perf_counter()
                # Independent streams each spend their own frame slice.
                # In the shared trial the mesh goes first once a tile has shown.
                show_mesh = refinement_pending and (
                    not shared or first_tile is not None
                )
                if show_mesh:
                    target.replace(0, [((0, (0,)), meshes[0][1])])
                    uploads += target.submitted_bytes
                    peak = max(peak, image_bytes + mesh_bytes + target.submitted_bytes)
                    mesh_bytes = target.submitted_bytes
                    refinement_pending = False
                image_start = start if shared else perf_counter()
                count = 0
                while tiles and count < 32:
                    if (
                        count or show_mesh
                    ) and perf_counter() - image_start >= FRAME_SECONDS:
                        break
                    y, x, pixels = tiles.popleft()
                    add_texture(image_group, pixels, x * dx, y * dy, dx, dy, 1)
                    image_bytes += pixels.nbytes
                    uploads += pixels.nbytes
                    count += 1
                peak = max(peak, image_bytes + mesh_bytes)
                if peak > RESIDENCY:
                    raise MemoryError(
                        "mixed replay exceeded its submitted buffer budget"
                    )
                rgba = np.array(canvas.draw())
                end = perf_counter()
                if first_tile is None and count:
                    first_tile = end
                if show_mesh:
                    mesh_presented = end
                frame_times.append((end - start) * 1000)
                projected_radius = min(width, height) * 0.19 / camera.height * CANVAS[1]
                sides = FINE_RING if mesh_presented is not None else COARSE_RING
                coverage_errors.append(projected_radius * (1 - np.cos(np.pi / sides)))
                if reference is not None:
                    baseline = reference[pose_index]
                    visible_image = np.ptp(baseline[..., :3], axis=-1) == 0
                    delta = rgba[..., :3].astype(float) - baseline[..., :3]
                    image_errors.append(
                        float(np.sqrt(np.mean(delta[visible_image] ** 2)))
                    )
            assert first_tile is not None and mesh_presented is not None
            waits.append(((first_tile - ready) * 1000, (mesh_presented - ready) * 1000))
            settled.append(rgba)
    finally:
        target.close()
    return {
        "frame_p95_ms": float(np.percentile(frame_times, 95)),
        "frame_max_ms": max(frame_times),
        "max_image_wait_ms": max(wait[0] for wait in waits),
        "max_mesh_wait_ms": max(wait[1] for wait in waits),
        "peak_submitted_bytes": peak,
        "submitted_bytes": uploads,
        "mesh_silhouette_error_px_per_frame": coverage_errors,
        "image_rmse_per_frame": image_errors,
    }, settled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = OMEZarrSource.open(str(args.dataset))
    if source.pyramid.ndim != 3 or len(source.arrays) < 2:
        parser.error("provide a 3D OME-Zarr pyramid")
    canvas = RenderCanvas(size=CANVAS)
    renderer = gfx.WgpuRenderer(canvas)
    reference = trial(source, renderer, canvas, False, None)[1]
    runs = []
    with contextlib.closing(canvas):
        for repeat in range(3):
            for shared in (False, True) if repeat % 2 == 0 else (True, False):
                metrics, _ = trial(source, renderer, canvas, shared, reference)
                runs.append(dict(metrics, repeat=repeat, shared=shared))
    args.output.write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
