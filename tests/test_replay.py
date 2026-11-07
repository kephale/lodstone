import numpy as np
import pytest

from lodstone.replay import image_error, pareto_front


def test_blank_and_brightness_cannot_game_foreground_error():
    reference = np.zeros((20, 20, 3), np.uint8)
    reference[10, 10] = 100
    blank = np.zeros_like(reference)
    doubled = reference * 2
    assert image_error(reference, reference)["rmse"] == 0
    assert image_error(reference, blank)["foreground_rmse"] > 0.39
    assert (
        image_error(reference, blank)["foreground_rmse"]
        > image_error(reference, blank)["rmse"]
    )
    assert image_error(reference, doubled)["brightness_bias"] > 0
    assert image_error(reference, doubled)["edge_rmse"] > 0


def test_pareto_keeps_tradeoffs_and_ties():
    rows = [
        {"id": "fine", "error": 0, "cost": 10},
        {"id": "balanced", "error": 1, "cost": 5},
        {"id": "bad", "error": 2, "cost": 6},
        {"id": "tie", "error": 1, "cost": 5},
    ]
    assert pareto_front(rows, ("error", "cost")) == ["fine", "balanced", "tie"]
    with pytest.raises(ValueError):
        pareto_front([{"id": "invalid", "error": float("nan")}], ("error",))


def test_frame_shape_and_range_must_match():
    with pytest.raises(ValueError):
        image_error(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)))
    with pytest.raises(ValueError):
        image_error(np.zeros((2, 2, 3)), np.full((2, 2, 3), 2.0))
