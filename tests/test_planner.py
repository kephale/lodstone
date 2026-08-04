from __future__ import annotations

import numpy as np

from lodstone import Layout, Planner
from lodstone.sources import ArrayPyramidSource


def _pyramid() -> ArrayPyramidSource:
    return ArrayPyramidSource(
        [
            np.zeros((256, 256), dtype=np.uint16),
            np.zeros((128, 128), dtype=np.uint16),
            np.zeros((64, 64), dtype=np.uint16),
        ],
        axes=("y", "x"),
        transforms=[
            np.eye(3),
            np.diag([2.0, 2.0, 1.0]),
            np.diag([4.0, 4.0, 1.0]),
        ],
        chunks=[(32, 32)] * 3,
    )


def test_zoomed_out_view_selects_coarser_level(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(64, 64), extent_scale=1.0)
    plan = Planner(progressive=False).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )
    assert plan.target_level == 2
    assert {tile.level for tile in plan.wanted} == {2}


def test_zoomed_in_view_selects_finest_level(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512), extent_scale=1.0)
    plan = Planner(progressive=False).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )
    assert plan.target_level == 0


def test_progressive_plan_puts_coarse_coverage_first(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=True).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )
    levels = [tile.level for tile in plan.wanted]
    first_fine = levels.index(0)
    assert set(levels[:first_fine]) == {1, 2}
    assert levels == sorted(levels, reverse=True)
    assert set(levels[first_fine:]) == {0}
    assert {key.level for key in plan.retain} == {0}
    assert {tile.level for tile in plan.desired} == {0, 1, 2}


def test_gpu_budget_can_select_a_coarser_level(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(block_shape=(32, 32), memory_limit=128 * 128 * 2),
    )
    assert plan.target_level == 1


def test_available_tiles_are_retained_but_not_requested(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    planner = Planner(progressive=False)
    initial = planner.plan(source.pyramid, view, Layout(block_shape=(32, 32)))
    available = frozenset(tile.key for tile in initial.wanted[:3])
    updated = planner.plan(
        source.pyramid,
        view,
        Layout(block_shape=(32, 32)),
        available=available,
    )
    assert available <= updated.retain
    assert available.isdisjoint(tile.key for tile in updated.wanted)
    assert available <= {tile.key for tile in updated.desired}


def test_3d_tiles_are_prioritized_front_to_back(ortho_view) -> None:
    source = ArrayPyramidSource(
        [np.zeros((16, 16, 16), dtype=np.uint8)],
        axes=("z", "y", "x"),
        chunks=[(16, 16, 4)],
    )
    view = ortho_view(
        source.pyramid.levels[0].shape,
        displayed_axes=(0, 1, 2),
        viewport=(256, 256),
    )
    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(kind="bricked", block_shape=(16, 16, 4)),
    )

    assert [tile.region.start[2] for tile in plan.wanted] == [0, 4, 8, 12]


def test_hidden_selection_is_part_of_tile_identity(ortho_view) -> None:
    source = ArrayPyramidSource(
        [np.zeros((2, 64, 64), dtype=np.uint8)],
        axes=("c", "y", "x"),
        chunks=[(1, 32, 32)],
    )
    view0 = ortho_view((2, 64, 64), displayed_axes=(1, 2), index=(0, None, None))
    view1 = ortho_view((2, 64, 64), displayed_axes=(1, 2), index=(1, None, None))
    planner = Planner(progressive=False)
    layout = Layout(block_shape=(32, 32))
    keys0 = {tile.key for tile in planner.plan(source.pyramid, view0, layout).wanted}
    keys1 = {tile.key for tile in planner.plan(source.pyramid, view1, layout).wanted}
    assert keys0.isdisjoint(keys1)


def test_hidden_selection_uses_level_scale_and_translation(ortho_view) -> None:
    coarse_transform = np.eye(4)
    coarse_transform[0, 0] = 2
    coarse_transform[1, 1] = 2
    coarse_transform[2, 2] = 2
    coarse_transform[0, 3] = 1
    source = ArrayPyramidSource(
        [
            np.zeros((16, 32, 32), dtype=np.uint8),
            np.zeros((8, 16, 16), dtype=np.uint8),
        ],
        axes=("z", "y", "x"),
        transforms=[np.eye(4), coarse_transform],
        chunks=[(1, 8, 8), (1, 8, 8)],
    )
    view = ortho_view(
        (16, 32, 32),
        displayed_axes=(1, 2),
        index=(5, None, None),
        viewport=(8, 8),
    )
    plan = Planner(progressive=False).plan(
        source.pyramid, view, Layout(block_shape=(8, 8))
    )
    assert plan.target_level == 1
    assert {tile.region.start[0] for tile in plan.wanted} == {2}
