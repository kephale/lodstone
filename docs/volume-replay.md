# Volume replay benchmark and optimization handoff

Status: benchmark setup, not a completed optimization. Kyle reports that fine
Zebrahub detail is fetched, appears briefly, then gives way to coarse detail;
toggling content restores it. The earlier presentation fixes did not resolve
that observation. Keep this issue open until actual frame evidence passes.

## Record the real ndv failure

From the ndv environment used for the working viewer, with this checkout first
on PYTHONPATH:

```bash
PYTHONPATH="$PWD/src" ../ndv/.venv/bin/python examples/ndv_dense.py \
  --backend vispy --time 400 --record-dir /tmp/zebrahub-live-001
```

Adjust the ndv environment path for the local checkout. Rotate and zoom, wait
for loading to finish, then toggle content off/on without moving the camera.
Close normally. The recorder saves raw canvas RGB(A) `.npy` frames, JSONL
telemetry/settings, native Arcball states and camera matrices every 250 ms and
at phase publication (up to 1200 frames). It records actual handle visibility,
shape and contrast, not just planned levels. GUI labels and wireframe overlays
are excluded by capturing the VisPy canvas. Recording itself adds rendering and
I/O overhead; do not treat instrumented latency as an unperturbed frame rate.

This opt-in GUI path has been syntax-checked but has not been exercised against
a running ndv GUI on this machine. It currently supports VisPy only and uses
private ndv canvas hooks. Preserve the exact ndv revision alongside recordings.
The standalone path below has been rendered and evaluated offscreen.

## Reproduce matched baselines and a bounded sweep

Install `lodstone[replay]` into a dedicated or existing viewer environment. On
Linux without a display, VisPy EGL can use software rendering:

```bash
EGL_PLATFORM=surfaceless python examples/volume_replay.py \
  --output /tmp/synthetic-replay --poses 6

EGL_PLATFORM=surfaceless python examples/volume_replay.py \
  --zebrahub --time 400 --roi 160,1612,1000,288,1868,1256 \
  --clim 0 800 --budgets 2 8 --poses 3 --size 128 --timeout 180 \
  --output /tmp/zebrahub-replay
```

For a desktop application backend supply `--app pyqt6` (and install Qt).
Replay a recorded camera path with `--path .../camera-path.json --stride 4`.
Use the same world coordinates, source, viewport and region for comparisons;
a recorded full-sample path may leave a small reference ROI. The runner rejects
frames with no measurable reference foreground rather than declaring blank
images optimal. Exact viewport size is an explicit `--size` setting, not inferred
from the path. The full-frame reference must fit `--reference-mib`; it never
silently coarsens. The smoke test uses a 16 MiB finest-resolution region, not a
full-resolution rendering of the entire fish. Outside that region is excluded
from every mode, so this cannot yet establish whole-sample quality.

Each configuration uses the same poses and fixed intensity range:

- `reference`: full finest-resolution region, loaded once and retained.
- `coarse`: only the coarsest level at every pose.
- `blank`: no volume; camera follows the same path.
- `stationary`: loaded coarse scene held at the first camera pose.
- `adaptive-B-W`: budget B MiB, coarse-signal weight W (zero or one).

The adaptive runner uses the core Planner defaults (LOD bias one, all
intermediate levels), not the ndv demo's bias two and shortened ladder. It uses
DenseTarget and ndv-style VisPy MIP/additive composition with a queued dispatch,
but not ndv GUI camera debounce. It intentionally keeps contrast fixed across
levels; ndv's per-publication auto contrast remains a separate variable to test.

The generated path moves across the selected region, rotates and changes zoom.
This is a sequence of settled camera poses, not yet a fixed-speed continuous
flight. Phase and periodic loading captures record the trajectory to completion;
settled and delayed frames test the endpoint. Native camera recording supports
an actual user path. Faster-than-loading replay and backend-specific UI races
remain follow-up work.

## Measurements and controls

`report.json` lists configurations and the nondominated set minimizing reference
foreground RMSE, decoded source bytes and peak sampled resident-array bytes.
`run.json` includes frame-level global/foreground/edge errors, signed brightness
bias, source-read counts, per-generation diagnostics, visible levels, settled
versus delayed error, and final-quality regressions relative to earlier frames.
All image scores use fixed RGB range; no per-frame rescaling is allowed.

Foreground masks come from the reference, not the candidate. Blank and stationary
controls are excluded from the Pareto candidates. Edge errors and brightness
bias are diagnostic; they are not yet additional Pareto objectives. A final
foreground RMSE more than 0.01 worse than the best prior frame is flagged; this
is a screening threshold, not an established perceptual or scientific tolerance.

Costs need careful interpretation:

- Fresh Lodstone stream/cache per configuration, warm across poses; filesystem,
  transport and server caches are not reset. Source object is shared.
- `source_bytes` counts decoded regional payload returned to Lodstone, not
  encoded network traffic or complete decoder scratch buffers.
- ROI chunk grids preserve intersections with native chunks. Source-read count
  measures logical adapter reads, not necessarily HTTP request count.
- Resident memory is sampled from CPU windows, including staged windows. RSS
  includes the whole process and allocator history; configurations currently
  share a process. GPU allocation/upload completion is not measured. Do not
  optimize against sequential RSS differences as if they were isolated trials.
- Full reference loading can cost less than an adaptive run that loads several
  levels of a small ROI. That is a legitimate outcome within this limited test,
  not proof that whole-volume streaming is unnecessary.

## Handoff to the optimization agent

1. Freeze the runner source hashes, dataset/time/channel/ROI, camera path,
   renderer, viewport, transfer function, contrast and cache protocol. Preserve
   the baseline outputs before changing code. Use a separate held-out path for
   selection validation; do not tune and report on the same three smoke poses.
2. Reproduce the actual GUI disappearance using `--record-dir`. Correlate pixels
   with publication phase, visible handles, target level, camera state and
   generation. Determine whether the loss is composition, contrast, texture
   replacement/downsampling, camera events, stale submission or residency.
   The content toggle is a diagnostic intervention, not a solution.
3. Correct endpoint quality regressions before expanding the search. A finer
   phase must not merely fetch more bytes while worsening the displayed image.
   Compare a single-volume finest reference with multi-volume composition;
   independent MIPs added together are not mathematically a single-volume MIP.
4. Sweep content weight, geometry policy, LOD bias, phase ladder, brick shape and
   budget on the frozen training paths. Report the Pareto set and hold-out
   performance. Preserve dim nuclei and far MIP contributions; coarse averages
   cannot justify hard skipping. Do not equate brightness with information.
5. Add bounded per-pose high-resolution reference capture across the full sample,
   continuous-motion replay, native network/GPU telemetry and isolated memory
   trials before recommending defaults for actual machines. Validate VisPy and
   PyGFX independently, then the separate napari path.

Completion means a reproducible improvement in actual settled and transient
pixels at stated costs, with no disappearing fine detail—not more planned or
fetched chunks. This handoff does not claim that the display bug is fixed.

## Saved smoke-test evidence

[Baseline report](../examples/data/volume-replay-zebrahub/report.json),
[manifest and source hashes](../examples/data/volume-replay-zebrahub/manifest.json),
and [camera path](../examples/data/volume-replay-zebrahub/camera-path.json).
Each configuration directory contains the actual PNG sequence and `run.json`.
This is a three-pose software-EGL smoke test of one selected Zebrahub ROI,
not a whole-sample or hardware performance claim.

| Configuration | Foreground RMSE | Decoded source MiB | Peak resident MiB |
| --- | ---: | ---: | ---: |
| Finest reference | 0 | 16 | 16 |
| Coarse only | 0.0412 | 0.25 | 0.25 |
| Adaptive 2 MiB, geometry | 0.0456 | 18.25 | 6.125 |
| Adaptive 2 MiB, content | 0.0389 | 18.25 | 6.164 |
| Adaptive 8 MiB, geometry | 0.0280 | 18.25 | 17.859 |
| Adaptive 8 MiB, content | 0.0287 | 18.25 | 18 |

The 2 MiB geometry configuration is worse than coarse-only in this aggregate
metric despite fetching much more data. Both 2 MiB adaptive configurations
finish worse than their best intermediate frame at all three poses. Both 8 MiB
configurations trigger that regression flag at one pose. This is measured
pixel evidence to investigate, not proof of the specific GUI disappearance.
The content heuristic improves some cases and worsens another; no universal
benefit or recommended default follows from this smoke test.
