import math

from metrics import smape_np


def test_smape_zero_error():
    assert smape_np([100, 200], [100, 200]) == 0.0


def test_smape_all_zero_safe():
    assert smape_np([0, 0], [0, 0]) == 0.0


def test_smape_negative_values():
    val = smape_np([100, -100], [0, 0])
    assert math.isclose(val, 200.0, rel_tol=1e-6)

