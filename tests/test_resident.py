from __future__ import annotations

import numpy as np

from lodstone import (
    Plan,
    Region,
    ResidentArrays,
    ResidentLease,
    Tile,
    TileKey,
    Update,
)
from lodstone.sources import ArrayPyramidSource


def _tile(level: int, start: tuple[int, ...], stop: tuple[int, ...], phase=0):
    key = TileKey(level, start, ())
    return Tile(key, Region(start, stop), 0.0, phase)


def _plan(*tiles: Tile, target: int = 0) -> Plan:
    return Plan(tuple(tiles), frozenset(tile.key for tile in tiles), target, tiles)


def test_prepare_allocates_only_desired_bounds() -> None:
    source = ArrayPyramidSource(
        [np.zeros((100, 100), dtype=np.uint16)], chunks=[(10, 10)]
    )
    arrays = ResidentArrays(source.pyramid)
    plan = _plan(
        _tile(0, (20, 30), (30, 40)),
        _tile(0, (30, 40), (40, 50)),
    )

    transition = arrays.prepare(plan)
    window = arrays.windows[0]

    assert transition.prepared == (window,)
    assert window.region == Region((20, 30), (40, 50))
    assert window.data.shape == (20, 20)
    assert arrays.nbytes == 20 * 20 * 2


def test_prepare_uses_source_fill_value() -> None:
    class FilledArray:
        shape = (20, 20)
        dtype = np.dtype(np.uint16)
        fill_value = 17

        def __getitem__(self, key):
            return np.full(self.shape, self.fill_value, dtype=self.dtype)[key]

    source = ArrayPyramidSource([FilledArray()], chunks=[(10, 10)])
    arrays = ResidentArrays(source.pyramid)

    arrays.prepare(_plan(_tile(0, (0, 0), (10, 10))))

    assert np.all(arrays.windows[0].data == 17)


def test_apply_uses_window_relative_coordinates() -> None:
    source = ArrayPyramidSource(
        [np.zeros((100, 100), dtype=np.uint16)], chunks=[(10, 10)]
    )
    arrays = ResidentArrays(source.pyramid)
    tile = _tile(0, (20, 30), (30, 40))
    plan = _plan(tile)
    arrays.prepare(plan)
    update = Update(
        tile.key,
        tile.region,
        np.full((10, 10), 7, dtype=np.uint16),
        np.eye(3),
    )

    changes = arrays.apply([update])

    assert len(changes) == 1
    assert changes[0].updates == (update,)
    assert np.all(arrays.windows[0].data == 7)


def test_shifted_window_preserves_overlap_and_loaded_keys() -> None:
    source = ArrayPyramidSource(
        [np.zeros((100, 100), dtype=np.uint8)], chunks=[(10, 10)]
    )
    arrays = ResidentArrays(source.pyramid)
    first = _tile(0, (20, 20), (40, 40))
    first_plan = _plan(first)
    arrays.prepare(first_plan)
    arrays.apply(
        [Update(first.key, first.region, np.ones((20, 20), np.uint8), np.eye(3))]
    )
    arrays.complete(first_plan)

    second = _tile(0, (30, 30), (50, 50))
    transition = arrays.prepare(_plan(second))
    shifted = arrays.windows[0]

    assert len(transition.prepared) == 1
    assert np.all(shifted.data[:10, :10] == 1)
    assert np.all(shifted.data[10:, :] == 0)
    assert first.key not in shifted.key_regions


def test_resident_lease_tracks_confirmed_storage_and_release() -> None:
    source = ArrayPyramidSource([np.zeros((20, 20), dtype=np.uint8)], chunks=[(10, 10)])
    arrays = ResidentArrays(source.pyramid)
    first = _tile(0, (0, 0), (10, 10))
    second = _tile(0, (0, 10), (10, 20))
    plan = _plan(first, second)
    arrays.prepare(plan)
    lease = ResidentLease(arrays, frozenset({first.key, second.key}))

    assert lease.available_keys == frozenset()
    assert lease.pending_keys == frozenset({first.key, second.key})

    arrays.apply(
        [Update(first.key, first.region, np.ones((10, 10), np.uint8), np.eye(3))]
    )
    assert lease.available_keys == frozenset({first.key})
    assert lease.pending_keys == frozenset({second.key})

    lease.release({first.key})
    assert lease.available_keys == frozenset()


def test_complete_keeps_target_and_retires_coarse_ladder() -> None:
    source = ArrayPyramidSource(
        [
            np.zeros((32, 32), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.uint8),
        ],
        chunks=[(8, 8), (8, 8)],
    )
    arrays = ResidentArrays(source.pyramid)
    coarse = _tile(1, (0, 0), (16, 16), phase=0)
    fine = _tile(0, (8, 8), (24, 24), phase=1)
    plan = _plan(coarse, fine)

    arrays.prepare(plan)
    coarse_window = arrays.windows[1]
    fine_window = arrays.windows[0]
    transition = arrays.complete(plan)

    assert arrays.active == {0: fine_window}
    assert transition.retired == (coarse_window,)
    assert arrays.nbytes == fine_window.nbytes


def test_full_nd_hidden_axis_updates_are_required() -> None:
    source = ArrayPyramidSource(
        [np.zeros((4, 20, 20), dtype=np.uint8)], chunks=[(1, 10, 10)]
    )
    arrays = ResidentArrays(source.pyramid)
    tile = _tile(0, (2, 0, 0), (3, 10, 10))
    arrays.prepare(_plan(tile))

    arrays.apply(
        [
            Update(
                tile.key,
                tile.region,
                np.ones((1, 10, 10), dtype=np.uint8),
                np.eye(4),
            )
        ]
    )
    assert arrays.windows[0].data.shape == (1, 10, 10)

    with np.testing.assert_raises_regex(ValueError, "unsqueezed"):
        arrays.apply(
            [
                Update(
                    tile.key,
                    tile.region,
                    np.ones((10, 10), dtype=np.uint8),
                    np.eye(4),
                )
            ]
        )


def test_composed_residency_repairs_only_unloaded_fine_chunks() -> None:
    fine_transform = np.eye(3)
    coarse_transform = np.diag([2.0, 2.0, 1.0])
    source = ArrayPyramidSource(
        [
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        ],
        chunks=[(4, 4), (2, 2)],
        transforms=[fine_transform, coarse_transform],
    )
    arrays = ResidentArrays(source.pyramid, compose=True)
    coarse = _tile(1, (0, 0), (4, 4), phase=0)
    fine_left = _tile(0, (0, 0), (4, 4), phase=1)
    fine_right = _tile(0, (0, 4), (4, 8), phase=1)
    plan = _plan(coarse, fine_left, fine_right)
    arrays.prepare(plan)

    coarse_update = Update(
        coarse.key,
        coarse.region,
        np.full((4, 4), 3, dtype=np.uint8),
        coarse_transform,
    )
    changes = arrays.apply([coarse_update])

    fine_window = arrays.windows[0]
    fine_change = next(change for change in changes if change.window is fine_window)
    assert fine_change.updates == ()
    assert fine_change.repaired
    assert np.all(fine_window.data == 3)

    fine_update = Update(
        fine_left.key,
        fine_left.region,
        np.full((4, 4), 9, dtype=np.uint8),
        fine_transform,
    )
    arrays.apply([fine_update])
    arrays.apply(
        [
            Update(
                coarse.key,
                coarse.region,
                np.full((4, 4), 5, dtype=np.uint8),
                coarse_transform,
            )
        ]
    )

    assert np.all(fine_window.data[:, :4] == 9)
    assert np.all(fine_window.data[:, 4:] == 5)
