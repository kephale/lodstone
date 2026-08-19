from __future__ import annotations

import numpy as np

from lodstone import (
    LevelDiagnosticArray,
    Plan,
    PlanComparison,
    PlanTrace,
    Region,
    Tile,
    TileKey,
    View,
)


def test_plan_comparison_ignores_planner_specific_keys_and_priority() -> None:
    region = Region((0, 0), (8, 8))
    first_tile = Tile(TileKey(0, (0, 0), ()), region, 0.0)
    second_tile = Tile(TileKey(0, (99, 99), ()), region, 42.0)
    first = Plan((first_tile,), frozenset({first_tile.key}), 0, (first_tile,))
    second = Plan((second_tile,), frozenset({second_tile.key}), 0, (second_tile,))

    trace = PlanTrace.from_plan(first)
    view = View((0, 1), (None, None), (100, 100), np.eye(4))
    comparison = PlanComparison(view, trace, PlanTrace.from_plan(second))

    assert comparison.matches
    assert comparison.geometry_matches
    assert trace.tiles == ((0, (0, 0), (8, 8), 0),)


def test_level_diagnostic_array_reads_source_before_replacing_values() -> None:
    class RecordingArray:
        shape = (8, 8)
        dtype = np.dtype("u2")
        chunks = (4, 4)

        def __init__(self) -> None:
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return np.zeros(self.shape, dtype=self.dtype)[key]

    source = RecordingArray()
    diagnostic = LevelDiagnosticArray(source, 2)
    result = diagnostic[1:5, 2:7]

    assert source.reads == [(slice(1, 5), slice(2, 7))]
    assert diagnostic.fill_value == 1
    np.testing.assert_array_equal(result, np.full((4, 5), 4, dtype=np.uint8))
