"""Replay the camera trace against demo-multiscale's TensorStore/pygfx path."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
from upload_replay import CANVAS, TRACE, replay

from lodstone.sources import OMEZarrSource


def trial(store, renderer, canvas, executor, cache, reference):
    from demo_multiscale.data_store import OMEZarrImageDataStore
    from demo_multiscale.render_visual import GFXMultiscaleImageVisual, ImageGeometry2D
    from demo_multiscale.state import AxisAlignedSelectionState, DimsState

    assert isinstance(store, OMEZarrImageDataStore)
    depth, ny, nx = store.level_shapes[0]
    scales = list(store.voxel_sizes)
    width, height = nx * scales[2], ny * scales[1]
    axes = (1, 2)
    transforms = store.level_transforms
    geometry = ImageGeometry2D(
        [tuple(shape[1:]) for shape in store.level_shapes],
        32,
        store.n_levels,
        [transform.set_slice(axes) for transform in transforms],
    )
    grayscale = gfx.TextureMap(
        gfx.Texture(np.array([[0, 0, 0], [1, 1, 1]], dtype="float32"), dim=1)
    )
    visual = GFXMultiscaleImageVisual(
        uuid4(),
        None,
        geometry,
        {"2d"},
        displayed_axes=axes,
        colormap=grayscale,
        clim=(0, 65535),
        gpu_budget_bytes_2d=1 << 20,
        voxel_scales=scales,
        full_level_shapes=store.level_shapes,
        full_level_transforms=transforms,
    )
    scene, camera = gfx.Scene(), gfx.OrthographicCamera()
    scene.add(visual.node_2d)
    camera.local.z = 100
    canvas.request_draw(lambda: renderer.render(scene, camera))
    state = DimsState(
        tuple(store.axis_names), AxisAlignedSelectionState(axes, {0: depth // 2})
    )

    def fetch(request):
        key = (request.scale_index, request.axis_selections)
        start = perf_counter()
        if key in cache:
            return cache[key], 0.0, key
        return store.get_data(request), perf_counter() - start, key

    coverage, frame_times, errors, settle, compare = [], [], [], [], []
    read_time = submit_time = draw_time = 0.0
    uploaded = 0
    for pose, (cx, cy, fraction) in enumerate(TRACE):
        started = perf_counter()
        camera.width, camera.height = width * fraction, height * fraction
        camera.local.x, camera.local.y = cx * width, cy * height
        visual.invalidate_2d_cache()
        requests = visual.build_slice_request_2d(
            np.asarray(camera.local.position),
            CANVAS[0],
            camera.width,
            None,
            None,
            state,
        )
        futures = [(request, executor.submit(fetch, request)) for request in requests]
        frames = []
        while futures or not frames:
            if futures:
                futures[0][1].result(timeout=30)
            started_frame = perf_counter()
            batch = []
            while futures and futures[0][1].done() and len(batch) < 64:
                request, future = futures.pop(0)
                data, secs, key = future.result()
                cache[key] = data
                read_time += secs
                uploaded += data.nbytes
                batch.append((request, data))
            if batch:
                visual.on_data_ready_2d(batch)
            before_draw = perf_counter()
            pixels = np.array(canvas.draw())
            end = perf_counter()
            submit_time += before_draw - started_frame
            draw_time += end - before_draw
            frame_times.append((end - started_frame) * 1000)
            if not frames:
                coverage.append((end - started) * 1000)
            frames.append(pixels[..., :3])
        settle.append((end - started) * 1000)
        delta = np.stack(frames).astype(np.float32) - frames[-1]
        errors.append(np.sqrt(np.mean(delta**2, axis=(1, 2, 3))).tolist())
        compare.append(
            float(np.sqrt(np.mean((frames[-1].astype(float) - reference[pose]) ** 2)))
        )
    return {
        "first_rendered_frame_ms": coverage,
        "settled_rmse_to_lodstone": compare,
        "settle_ms": settle,
        "frame_p95_ms": float(np.percentile(frame_times, 95)),
        "frame_max_ms": max(frame_times),
        "read_ms": read_time * 1000,
        "submit_ms": submit_time * 1000,
        "draw_ms": draw_time * 1000,
        "decoded_bytes_submitted": uploaded,
        "cache_texture_bytes": visual._block_cache_2d.cache_data.nbytes,
        "frame_rmse_to_own_settled": errors,
    }


def main():
    from demo_multiscale.data_store import OMEZarrImageDataStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = OMEZarrImageDataStore.from_path(str(args.dataset))
    canvas = RenderCanvas(size=CANVAS)
    renderer = gfx.WgpuRenderer(canvas)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        source = OMEZarrSource.open(str(args.dataset))
        reference = replay(source, renderer, canvas, None, executor, {})[1]
        trial(store, renderer, canvas, executor, {}, reference)
        for repeat in range(3):
            cache = {}
            for mode in ("cold_app_cache", "warm_app_cache"):
                metrics = trial(store, renderer, canvas, executor, cache, reference)
                results.append(dict(metrics, repeat=repeat, cache=mode))
    canvas.close()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
