"""Compare frame admission policies using the same OME-Zarr and pygfx target."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas

from lodstone.sources import OMEZarrSource

CANVAS = (192, 192)
TRACE = ((0.48, 0.48, 0.95), (0.57, 0.48, 0.72), (0.62, 0.57, 0.52))
TILE = 16


def prepare(array, z, y, x):
    start = perf_counter()
    data = np.asarray(array[z, y : y + TILE, x : x + TILE], dtype=np.float32)
    return (data, perf_counter() - start)


def add_texture(scene, data, x, y, sx, sy, depth):
    texture = gfx.Texture(data, dim=2)
    image = gfx.Image(
        gfx.Geometry(grid=texture),
        gfx.ImageBasicMaterial(clim=(0, 65535), interpolation="nearest"),
    )
    # Pygfx image pixels have integer centers.
    image.local.position = (x + sx / 2, y + sy / 2, depth)
    image.local.scale = (sx, sy, 1)
    scene.add(image)


def replay(
    source, renderer, canvas, budget, executor, cache, reference=None, frames_path=None
):
    scene = gfx.Scene()
    camera = gfx.OrthographicCamera()
    camera.local.z = 100
    canvas.request_draw(lambda: renderer.render(scene, camera))
    fine = source.arrays[0]
    coarse = source.arrays[-1]
    scale = np.diag(source.pyramid.levels[-1].voxel_to_world)[:-1]
    fine_scale = np.diag(source.pyramid.levels[0].voxel_to_world)[:-1]
    depth = fine.shape[0] // 2
    z_coarse = min(coarse.shape[0] - 1, int(depth * fine_scale[0] / scale[0]))
    slices = source.arrays
    timings, admissions, draws, read_times, first_frames = [], [], [], [], []
    peak_bytes, uploaded_bytes = 0, 0
    settled, transient, settle_ms = [], [], []
    recorded_frames = {}
    for pose_index, (cx, cy, fraction) in enumerate(TRACE):
        started = perf_counter()
        scene.clear()
        height, width = fine.shape[1:]
        camera.width, camera.height = (
            width * fraction * fine_scale[2],
            height * fraction * fine_scale[1],
        )
        camera.local.x, camera.local.y = (
            cx * width * fine_scale[2],
            cy * height * fine_scale[1],
        )
        lo_x = max(0, int((cx - fraction / 2) * width))
        hi_x = min(width, int(np.ceil((cx + fraction / 2) * width)))
        lo_y = max(0, int((cy - fraction / 2) * height))
        hi_y = min(height, int(np.ceil((cy + fraction / 2) * height)))
        requested = sorted(
            product(
                range(lo_y // TILE * TILE, hi_y, TILE),
                range(lo_x // TILE * TILE, hi_x, TILE),
            ),
            key=lambda pos: (
                (pos[0] + TILE / 2 - cy * height) ** 2
                + (pos[1] + TILE / 2 - cx * width) ** 2
            ),
        )
        pose_bytes = coarse.shape[1] * coarse.shape[2] * 4
        for y, x in requested:
            pose_bytes += min(TILE, height - y) * min(TILE, width - x) * 4
        if pose_bytes > (1 << 20):
            raise ValueError("camera trace exceeds the 1 MiB submitted texture budget")
        peak_bytes = max(pose_bytes, peak_bytes)

        def load(key):
            if key in cache:
                return cache[key], 0.0
            index, z, y, x = key
            if index == len(slices) - 1:
                start = perf_counter()
                return np.asarray(coarse[z], dtype=np.float32), perf_counter() - start
            return prepare(slices[index], z, y, x)

        coarse_key = (len(slices) - 1, z_coarse, 0, 0)
        expiry = perf_counter() + 30
        tasks = [
            (key, executor.submit(load, key))
            for key in [
                coarse_key,
                *((0, depth, y, x) for y, x in requested),
            ]
        ]
        frames = []
        while tasks:
            elapsed = perf_counter()
            if elapsed >= expiry:
                raise TimeoutError("upload replay did not settle within 30 seconds")
            admitted = 0
            upload_start = perf_counter()
            while tasks and tasks[0][1].done():
                if admitted and (
                    admitted >= 32
                    or (budget is not None and perf_counter() - upload_start >= budget)
                ):
                    break
                key, future = tasks.pop(0)
                data, read_seconds = future.result()
                cache[key] = data
                read_times.append(read_seconds)
                _, _, y, x = key
                if key == coarse_key:
                    add_texture(scene, data, 0, 0, scale[2], scale[1], 0)
                else:
                    add_texture(
                        scene,
                        data,
                        x * fine_scale[2],
                        y * fine_scale[1],
                        fine_scale[2],
                        fine_scale[1],
                        1,
                    )
                admitted += 1
                uploaded_bytes += data.nbytes
                # Present the coarse fallback before refining.
                if key == coarse_key:
                    break
            if admitted == 0:
                sleep(0.001)
                continue
            before_draw = perf_counter()
            rendered = np.array(canvas.draw())
            end = perf_counter()
            admissions.append(before_draw - upload_start)
            draws.append(end - before_draw)
            timings.append(end - elapsed)
            if not frames:
                first_frames.append(end - started)
            frames.append(rendered[..., :3])
        settled.append(frames[-1])
        settle_ms.append((end - started) * 1000)
        if frames_path is not None:
            recorded_frames[f"pose_{pose_index}"] = np.stack(frames)
        if reference is not None:
            delta = np.stack(frames).astype(np.float32) - reference[pose_index]
            transient.append(np.sqrt(np.mean(delta**2, axis=(1, 2, 3))).tolist())
    if frames_path is not None:
        np.savez_compressed(frames_path, **recorded_frames)
    return {
        "settle_ms": settle_ms,
        "first_coverage_ms": [t * 1000 for t in first_frames],
        "frame_p95_ms": float(np.percentile(timings, 95) * 1000),
        "frame_max_ms": max(timings) * 1000,
        "submit_ms": sum(admissions) * 1000,
        "draw_ms": sum(draws) * 1000,
        "chunk_read_ms": sum(read_times) * 1000,
        "uploaded_bytes": uploaded_bytes,
        "peak_submitted_texture_bytes": peak_bytes,
        "frame_rmse_to_settled_baseline": transient,
    }, settled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=Path, help="directory for per-pose frames")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    source = OMEZarrSource.open(str(args.dataset))
    if source.pyramid.ndim != 3 or len(source.arrays) < 2:
        parser.error("provide a 3D OME-Zarr pyramid")
    canvas = RenderCanvas(size=CANVAS)
    renderer = gfx.WgpuRenderer(canvas)
    if args.frames is not None:
        args.frames.mkdir(parents=True, exist_ok=True)
    runs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        reference = replay(source, renderer, canvas, None, executor, {})[1]
        for repeat in range(args.repeats):
            policies = (("fixed_32", None), ("budget_2ms", 0.002))
            for name, policy in policies[:: (-1 if repeat % 2 else 1)]:
                cache = {}
                for cache_mode in ("cold_app_cache", "warm_app_cache"):
                    frames_path = (
                        args.frames / f"{repeat}-{name}-{cache_mode}.npz"
                        if args.frames is not None
                        else None
                    )
                    result, _ = replay(
                        source,
                        renderer,
                        canvas,
                        policy,
                        executor,
                        cache,
                        reference,
                        frames_path,
                    )
                    runs.append(
                        dict(result, repeat=repeat, policy=name, cache=cache_mode)
                    )
    canvas.close()
    args.output.write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
