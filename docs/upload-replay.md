# Replay image upload policies

Use the deterministic synthetic volume from [demo-multiscale at 8262b85](https://github.com/kevinyamauchi/demo-multiscale/tree/8262b85f4c52e6689cb20ee4b7e36e77ac62dae9).

```sh
uv run --with zarr /path/to/demo-multiscale/make_example_data.py \
  --output /tmp/example.ome.zarr --shape 64 256 256 --voxel-scales 4 1 1 \
  --n-levels 4 --n-blobs 12 --dtype uint16 --seed 42
uv run --extra ome-zarr --with pygfx --with rendercanvas \
  examples/upload_replay.py /tmp/example.ome.zarr --output /tmp/upload.json \
  --frames /tmp/upload-frames
PYTHONPATH=/path/to/demo-multiscale uv run --extra ome-zarr --with pygfx \
  --with rendercanvas --with tensorstore --with pydantic --with 'yaozarrs[io]' \
  examples/demo_loader_replay.py /tmp/example.ome.zarr --output /tmp/demo.json
```

The Lodstone replay holds the source, coarse-first view, camera poses, renderer, and 1 MiB texture limit fixed. Compare a cap of 32 ready tiles per frame with the same cap and a 2 ms CPU submission budget. Start with an empty application cache, repeat with its decoded tiles retained, and alternate policy order across trials. Operating system file caches and shader state may stay warm. Frame time includes submitting ready tiles, drawing and offscreen readback. The draw phase includes deferred uploads. Read time sums individual worker durations and can overlap other work. Texture bytes describe submitted resources. The JSON contains a pixel error trace relative to each settled fixed-count frame. Optional NPZ files record each intermediate frame.

## Software-rendered trial

Three repeats on an offscreen software adapter gave the following medians. Settle time is the sum over the three camera poses.

| Policy and application cache | Frame p95 | Worst frame | Settle time |
| --- | ---: | ---: | ---: |
| Fixed count, cold | 69.1 ms | 71.6 ms | 842 ms |
| 2 ms pacing, cold | 18.8 ms | 25.1 ms | 1,281 ms |
| Fixed count, warm | 44.6 ms | 46.3 ms | 714 ms |
| 2 ms pacing, warm | 13.7 ms | 40.4 ms | 1,158 ms |

The two Lodstone policies settled to identical pixels across all trials. Each submitted 536,576 texture bytes across the trace with a peak live texture payload of 266,240 bytes. Pacing reduced per-frame stalls while adding about 440 ms of total settle time. The budget stays in this replay pending a representative hardware trial that can judge that tradeoff.

The external comparison exercises the demo's own TensorStore reader, tile planner, texture cache and pygfx shader, driven offscreen at the same camera poses with its 64-tile cap and a 1 MiB cache texture budget. Its cold and warm p95 medians were 9.9 and 5.7 ms. Its settled output differed from the Lodstone replay by 9.9–15.5 RGB levels RMS, and it retained renderer cache entries between poses. Treat these numbers as end-to-end observations for two different rendering strategies. The controlled pacing result applies within the Lodstone replay.
