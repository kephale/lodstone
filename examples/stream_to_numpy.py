"""Stream a two-level array pyramid into a recording target."""

from __future__ import annotations

import time

import numpy as np

from lodstone import Layout, Stream, View
from lodstone.sources import ArrayPyramidSource
from lodstone.testing import RecordingTarget


def main() -> None:
    fine = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    coarse = fine[::4, ::4]
    source = ArrayPyramidSource(
        [fine, coarse],
        axes=("y", "x"),
        transforms=[np.eye(3), np.diag([4.0, 4.0, 1.0])],
        chunks=[(64, 64), (32, 32)],
    )
    target = RecordingTarget(Layout(kind="tiled", block_shape=(64, 64)))
    world_to_clip = np.array(
        [
            [2 / 512, 0, 0, -1],
            [0, 2 / 512, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    view = View((0, 1), (None, None), (800, 800), world_to_clip)

    with Stream(source, target) as stream:
        stream.update(view)
        while stream.status.state == "loading":
            time.sleep(0.01)
        print(stream.status)
        print(f"received {len(target.updates)} updates in {target.redraws} redraws")


if __name__ == "__main__":
    main()
