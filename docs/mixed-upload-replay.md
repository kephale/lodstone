# Replay concurrent image and mesh uploads

The replay overlays a tessellated disk on the same OME-Zarr slice and camera trace used by the [image upload replay](upload-replay.md). Lodstone's PyGFX mesh target replaces complete phases. Both trials present a coarse image and coarse mesh before admitting fine work.

```sh
uv run --extra ome-zarr --with pygfx --with rendercanvas \
  examples/mixed_upload_replay.py /tmp/example.ome.zarr --output /tmp/mixed.json
```

The independent trial submits the fine mesh and gives ready image tiles their own 2 ms submission slice. The shared trial gives image tiles a turn, then counts the fine mesh replacement against the same frame's image submission budget. A complete mesh phase is indivisible and may overrun that slice. Both trials use the same residency ceiling and fine work. The script counts submitted image and mesh payloads including overlapping mesh buffers during replacement. Driver allocations and GPU transfer timings are outside that accounting.

The JSON records frame p95 and maximum, time from readiness to first refinement for each stream, peak bytes, and per-frame image and mesh error. Image error uses the grayscale portion of the settled frame as its reference. Mesh error is the screen-space sagitta of the projected polygon's outer ring relative to a circle. The scene uses decoded inputs to isolate contention during submission and drawing.

In the software-rendered trial shared admission gave the image its first refinement earlier and delayed the mesh refinement. Frame p95 was similar in both groups and worst-frame time varied between repeats. Settled frames matched and peak submitted bytes were identical. Keep shared scheduling in the replay until a representative mixed workload shows a consistent frame-time benefit.
