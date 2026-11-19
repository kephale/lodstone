import runpy
from pathlib import Path

import numpy as np

example = runpy.run_path(str(Path(__file__).parents[1] / "examples/label_pyramid.py"))


def test_label_landmark_is_registered_across_levels():
    image, labels = example["reference_pyramids"]()
    world = np.array([-38 + 64.5 * 2, 15 + 64.5 * 0.75])
    for index, level in enumerate(labels.pyramid.levels):
        np.testing.assert_array_equal(
            level.voxel_to_world, image.pyramid.levels[index].voxel_to_world
        )
        x, y = world[::-1]
        region = example["visible_region"](
            level, (x - 0.1, x + 0.1), (y - 0.1, y + 0.1)
        )
        assert region is not None
        assert np.all(labels.arrays[index][region.slices()] == 1)
        assert np.all(image.arrays[index][region.slices()] == 200)
