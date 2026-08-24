from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lodstone import (
    Layout,
    Plan,
    Planner,
    Region,
    Tile,
    TileKey,
    View,
    available_tile_keys,
    merge_plans,
    plan_from_slices,
)
from lodstone.sources import ArrayPyramidSource


def test_plan_delta_retains_coverage_and_reports_priority_changes() -> None:
    key0 = TileKey(0, (0, 0), ())
    key1 = TileKey(0, (0, 1), ())
    key2 = TileKey(0, (0, 2), ())
    tile0 = Tile(key0, Region((0, 0), (4, 4)), 1.0)
    tile1 = Tile(key1, Region((0, 4), (4, 8)), 2.0)
    first = Plan((tile0, tile1), frozenset({key0, key1}), 0, (tile0, tile1))
    moved0 = replace(tile0, priority=-1.0)
    tile2 = Tile(key2, Region((0, 8), (4, 12)), 3.0)
    second = Plan((moved0, tile2), frozenset({key0, key2}), 0, (moved0, tile2))

    delta = second.delta(first)

    assert delta.retained == frozenset({key0})
    assert delta.requested == (tile2,)
    assert delta.reprioritized == (key0,)
    assert delta.released == frozenset({key1})


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


def test_rectilinear_chunk_containing_crosshair_has_first_priority() -> None:
    source = ArrayPyramidSource(
        [np.zeros((100, 200), dtype=np.uint8)],
        chunks=[((100,), (90, 10, 100))],
    )
    matrix = np.eye(4)
    matrix[0, 0] = 2 / 100
    matrix[0, 3] = -1
    matrix[1, 1] = 2 / 100
    matrix[1, 3] = -1.6
    view = View((0, 1), (None, None), (512, 512), matrix)

    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(kind="tiled"),
    )

    assert plan.wanted[0].region == Region((0, 0), (100, 90))


def test_perspective_lod_uses_near_visible_voxel_footprint() -> None:
    projection = np.zeros((4, 4), dtype=np.float64)
    near, far = 1.0, 100.0
    projection[0, 0] = projection[1, 1] = 1.0
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -(2 * far * near) / (far - near)
    projection[3, 2] = -1.0
    fine_transform = np.diag((0.01, 0.01, 1.0, 1.0))
    fine_transform[:2, 3] = -0.5
    fine_transform[2, 3] = -10.0
    coarse_transform = np.diag((0.02, 0.02, 2.0, 1.0))
    coarse_transform[:2, 3] = -0.5
    coarse_transform[2, 3] = -10.0
    source = ArrayPyramidSource(
        [
            np.zeros((100, 100, 8), dtype=np.uint8),
            np.zeros((50, 50, 4), dtype=np.uint8),
        ],
        transforms=[fine_transform, coarse_transform],
        chunks=[(25, 25, 2), (25, 25, 2)],
    )
    view = View(
        (0, 1, 2),
        (None, None, None),
        (512, 512),
        projection,
    )

    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(kind="bricked", memory_limit=1 << 30),
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


def test_progressive_plan_can_skip_intermediate_levels(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=True, max_intermediate_levels=0).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )

    assert [tile.level for tile in plan.wanted] == sorted(
        (tile.level for tile in plan.wanted), reverse=True
    )
    assert {tile.level for tile in plan.desired} == {0, 2}


def test_progressive_plan_can_adapt_initial_level_to_voxel_footprint(
    ortho_view,
) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(
        progressive=True,
        max_intermediate_levels=0,
        max_initial_voxel_footprint=4.1,
    ).plan(source.pyramid, view, Layout(block_shape=(32, 32)))

    assert {tile.level for tile in plan.desired} == {0, 1}
    assert plan.desired[0].level == 1


def test_progressive_initial_level_remains_coarsest_by_default(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=True, max_intermediate_levels=0).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )

    assert {tile.level for tile in plan.desired} == {0, 2}


def test_mixed_lod_retains_coarsest_context_with_target(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(
        progressive=True,
        max_intermediate_levels=0,
        max_initial_voxel_footprint=1.1,
    ).plan(
        source.pyramid,
        view,
        Layout(block_shape=(32, 32), mixed_lod=True),
    )

    assert plan.target_level == 0
    assert {tile.level for tile in plan.desired} == {0, 2}
    assert {key.level for key in plan.retain} == {0, 2}


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_progressive_initial_voxel_footprint_must_be_positive_and_finite(value) -> None:
    with pytest.raises(ValueError, match="max_initial_voxel_footprint"):
        Planner(max_initial_voxel_footprint=value)


def test_lod_hysteresis_resists_small_level_boundary_crossings(ortho_view) -> None:
    source = _pyramid()
    layout = Layout(block_shape=(32, 32))
    slightly_inside_fine = ortho_view((256, 256), viewport=(512, 512), extent_scale=3.9)
    slightly_inside_coarse = ortho_view(
        (256, 256), viewport=(512, 512), extent_scale=4.1
    )

    assert (
        Planner(progressive=False)
        .plan(source.pyramid, slightly_inside_fine, layout)
        .target_level
        == 0
    )
    assert (
        Planner(progressive=False)
        .plan(
            source.pyramid,
            slightly_inside_fine,
            layout,
            previous_target_level=1,
            lod_hysteresis=0.2,
        )
        .target_level
        == 1
    )
    assert (
        Planner(progressive=False)
        .plan(source.pyramid, slightly_inside_coarse, layout)
        .target_level
        == 1
    )
    assert (
        Planner(progressive=False)
        .plan(
            source.pyramid,
            slightly_inside_coarse,
            layout,
            previous_target_level=0,
            lod_hysteresis=0.2,
        )
        .target_level
        == 0
    )


def test_gpu_budget_can_select_a_coarser_level(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(block_shape=(32, 32), memory_limit=128 * 128 * 2),
    )
    assert plan.target_level == 1


def test_dense_crop_budget_preserves_camera_selected_level(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    memory_limit = 128 * 128 * 2
    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(
            block_shape=(32, 32),
            memory_limit=memory_limit,
            memory_policy="crop",
        ),
    )

    assert plan.target_level == 0
    start = tuple(
        min(tile.region.start[axis] for tile in plan.desired) for axis in range(2)
    )
    stop = tuple(
        max(tile.region.stop[axis] for tile in plan.desired) for axis in range(2)
    )
    assert np.prod(np.subtract(stop, start)) * 2 <= memory_limit


def test_dense_crop_can_balance_focus_depth_against_screen_center(ortho_view) -> None:
    shape = (64, 64, 64)
    source = ArrayPyramidSource(
        [np.zeros(shape, dtype=np.uint8)],
        axes=("z", "y", "x"),
        chunks=[(16, 16, 16)],
    )
    view = ortho_view(shape, viewport=(256, 256))
    planner = Planner(progressive=False)
    common = {
        "block_shape": (16, 16, 16),
        "memory_limit": 8 * 16**3,
        "memory_policy": "crop",
    }

    slab = planner.plan(source.pyramid, view, Layout(**common))
    volume = planner.plan(
        source.pyramid,
        view,
        Layout(**common, focus_depth_weight=0.5),
    )

    def extent(plan, axis):
        return max(tile.region.stop[axis] for tile in plan.desired) - min(
            tile.region.start[axis] for tile in plan.desired
        )

    # This camera maps data axis 2 to clip depth. The balanced focus spends
    # the same byte budget on two depth layers instead of one front slab.
    assert extent(slab, 2) == 16
    assert extent(volume, 2) == 32
    assert sum(tile.region.size for tile in slab.desired) == sum(
        tile.region.size for tile in volume.desired
    )


def test_volume_focus_priority_delivers_center_ray_before_canvas_edges(
    ortho_view,
) -> None:
    shape = (64, 64, 64)
    source = ArrayPyramidSource(
        [np.zeros(shape, dtype=np.uint8)],
        chunks=[(16, 16, 16)],
    )
    view = ortho_view(shape, viewport=(256, 256))
    matrix = view.world_to_clip.copy()
    matrix[0, 3] = matrix[1, 3] = -0.75
    view = replace(view, world_to_clip=matrix)

    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(
            kind="dense",
            block_shape=(16, 16, 16),
            memory_limit=8 * 16**3,
            memory_policy="crop",
            focus_depth_weight=0.5,
        ),
    )

    assert [tile.key.grid_index for tile in plan.wanted[:2]] == [
        (1, 1, 0),
        (1, 1, 1),
    ]


def test_volume_focus_can_refine_outward_from_visual_depth_center(
    ortho_view,
) -> None:
    shape = (64, 64, 64)
    source = ArrayPyramidSource(
        [np.zeros(shape, dtype=np.uint8)],
        chunks=[(16, 16, 16)],
    )
    view = ortho_view(shape, viewport=(256, 256))

    plan = Planner(progressive=False).plan(
        source.pyramid,
        view,
        Layout(
            kind="dense",
            block_shape=(16, 16, 16),
            memory_limit=8 * 16**3,
            memory_policy="crop",
            focus_depth_weight=0.5,
            focus_depth_target=0.5,
        ),
    )

    assert plan.wanted[0].key.grid_index == (1, 1, 2)
    assert {tile.key.grid_index[2] for tile in plan.wanted[:4]} == {2}
    assert {tile.key.grid_index[2] for tile in plan.wanted} == {1, 2}


def test_dense_crop_tunes_canvas_coverage_against_depth_reach(ortho_view) -> None:
    shape = (64, 64, 64)
    source = ArrayPyramidSource(
        [np.zeros(shape, dtype=np.uint8)],
        axes=("z", "y", "x"),
        chunks=[(16, 16, 16)],
    )
    view = ortho_view(shape, viewport=(256, 256))
    planner = Planner(progressive=False)
    common = {
        "block_shape": (16, 16, 16),
        "memory_limit": 8 * 16**3,
        "memory_policy": "crop",
    }

    canvas = planner.plan(
        source.pyramid,
        view,
        Layout(**common, focus_depth_weight=8.0),
    )
    depth = planner.plan(
        source.pyramid,
        view,
        Layout(**common, focus_depth_weight=0.5),
    )

    def extents(plan):
        return tuple(
            max(tile.region.stop[axis] for tile in plan.desired)
            - min(tile.region.start[axis] for tile in plan.desired)
            for axis in range(3)
        )

    canvas_extents = extents(canvas)
    depth_extents = extents(depth)
    # This camera maps axis 2 to depth. The canvas-heavy policy spends more of
    # the same dense budget in axes 0/1; the depth-heavy policy reaches farther
    # along axis 2 around the screen center.
    assert canvas_extents[0] * canvas_extents[1] > (depth_extents[0] * depth_extents[1])
    assert canvas_extents[2] < depth_extents[2]


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_focus_depth_weight_must_be_finite_and_nonnegative(value) -> None:
    with pytest.raises(ValueError, match="focus_depth_weight"):
        Layout(focus_depth_weight=value)


@pytest.mark.parametrize("value", [-1, 1.1, float("inf"), float("nan")])
def test_focus_depth_target_must_be_normalized_and_finite(value) -> None:
    with pytest.raises(ValueError, match="focus_depth_target"):
        Layout(focus_depth_target=value)


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


def test_plan_coverage_ignores_tile_order_priority_and_phase(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=True).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )
    reordered = replace(
        plan,
        desired=tuple(
            replace(tile, priority=-tile.priority, phase=tile.phase + 10)
            for tile in reversed(plan.desired)
        ),
    )

    assert reordered.coverage == plan.coverage


def test_plan_coverage_detects_regions_retention_and_hidden_selection(
    ortho_view,
) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=False).plan(
        source.pyramid, view, Layout(block_shape=(32, 32))
    )
    tile = plan.desired[0]
    shifted_region = Region(
        (tile.region.start[0] + 1, *tile.region.start[1:]),
        tile.region.stop,
    )
    changed_region = replace(
        plan,
        desired=(replace(tile, region=shifted_region), *plan.desired[1:]),
    )
    changed_retention = replace(plan, retain=frozenset())
    selected_key = TileKey(tile.level, tile.key.grid_index, (4, -1))
    changed_selection = replace(
        plan,
        desired=(
            Tile(selected_key, tile.region, tile.priority, tile.phase),
            *plan.desired[1:],
        ),
    )

    assert changed_region.coverage != plan.coverage
    assert changed_retention.coverage != plan.coverage
    assert changed_selection.coverage != plan.coverage


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


def test_native_rectilinear_chunks_define_display_tiles(ortho_view) -> None:
    data = np.zeros((7, 9), dtype=np.uint8)
    source = ArrayPyramidSource(
        [data],
        chunks=[((2, 3, 2), (4, 1, 4))],
    )

    plan = Planner(progressive=False).plan(
        source.pyramid,
        ortho_view(data.shape, viewport=(128, 128)),
        Layout(kind="tiled"),
    )

    assert {tile.region for tile in plan.wanted} == {
        Region((y0, x0), (y1, x1))
        for y0, y1 in ((0, 2), (2, 5), (5, 7))
        for x0, x1 in ((0, 4), (4, 5), (5, 9))
    }


def test_region_plan_maps_chunk_aligned_ladder_through_transforms(
    ortho_view,
) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    plan = Planner(progressive=True).plan_region(
        source.pyramid,
        view,
        Layout(memory_limit=1 << 30),
        target_level=0,
        target_region=Region((32, 64), (96, 160)),
    )

    assert {tile.level for tile in plan.desired} == {0, 1, 2}
    assert {tile.region for tile in plan.desired if tile.level == 0} == {
        Region((y0, x0), (y1, x1))
        for y0, y1 in ((32, 64), (64, 96))
        for x0, x1 in ((64, 96), (96, 128), (128, 160))
    }
    assert {tile.region for tile in plan.desired if tile.level == 2} == {
        Region((0, 0), (32, 32)),
        Region((0, 32), (32, 64)),
    }


def test_region_plan_obeys_dense_target_limits(ortho_view) -> None:
    source = _pyramid()
    plan = Planner(progressive=False).plan_region(
        source.pyramid,
        ortho_view((256, 256), viewport=(512, 512)),
        Layout(memory_limit=32 * 32 * 2, max_axis_extent=40),
        target_level=0,
        target_region=Region((0, 0), (256, 256)),
    )
    starts = np.min([tile.region.start for tile in plan.desired], axis=0)
    stops = np.max([tile.region.stop for tile in plan.desired], axis=0)
    # The 32-pixel dense window straddles native boundaries and therefore
    # expands to the two intersecting 32-pixel chunks on each axis.
    assert tuple(stops - starts) == (64, 64)


def test_overview_plan_keeps_complete_displayed_volume(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))

    plan = Planner().plan_overview(
        source.pyramid,
        view,
        memory_limit=64 * 64 * 2,
    )

    assert plan.target_level == 2
    assert {tile.region for tile in plan.desired} == {
        Region((y, x), (y + 32, x + 32)) for y in (0, 32) for x in (0, 32)
    }
    assert {tile.key for tile in plan.desired} == plan.retain


def test_overview_plan_restricts_hidden_axes_and_obeys_budget(ortho_view) -> None:
    source = ArrayPyramidSource(
        [np.zeros((8, 64, 64), dtype=np.uint8)],
        chunks=[(1, 32, 32)],
    )
    view = ortho_view(
        (8, 64, 64),
        displayed_axes=(1, 2),
        index=(5, None, None),
    )
    planner = Planner()

    plan = planner.plan_overview(source.pyramid, view, memory_limit=64 * 64)
    rejected = planner.plan_overview(source.pyramid, view, memory_limit=64)

    assert {tile.region.start[0] for tile in plan.desired} == {5}
    assert {tile.region.stop[0] for tile in plan.desired} == {6}
    assert rejected.desired == ()


def test_merge_plans_prepends_overview_and_unions_retention(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    planner = Planner(progressive=False)
    primary = planner.plan_region(
        source.pyramid,
        view,
        Layout(memory_limit=1 << 30),
        target_level=0,
        target_region=Region((64, 64), (128, 128)),
    )
    overview = planner.plan_overview(
        source.pyramid,
        view,
        memory_limit=64 * 64 * 2,
    )

    merged = merge_plans(primary, overview)

    assert merged.target_level == primary.target_level
    assert merged.wanted[: len(overview.wanted)] == overview.wanted
    assert merged.retain == primary.retain | overview.retain
    assert len({tile.key for tile in merged.desired}) == len(merged.desired)


def test_renderer_slice_plan_uses_native_keys_and_availability(ortho_view) -> None:
    source = _pyramid()
    view = ortho_view((256, 256), viewport=(512, 512))
    stages = [
        (2, [(slice(0, 32), slice(0, 32))]),
        (0, [(slice(32, 64), slice(64, 96))]),
    ]
    available = available_tile_keys(
        source.pyramid,
        view,
        {2: {((0, 32), (0, 32))}},
    )

    plan = plan_from_slices(
        source.pyramid,
        view,
        stages,
        target_level=0,
        available=available,
    )

    assert [tile.level for tile in plan.desired] == [2, 0]
    assert [tile.level for tile in plan.wanted] == [0]
    assert plan.desired[0].key in available
    assert plan.retain == {plan.desired[1].key}
