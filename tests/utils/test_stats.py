import math

import pytest

from trafficrl.utils.stats import summarise, t975


def test_t_values():
    assert t975(2) == 4.303 and t975(4) == 2.776 and t975(100) == 1.96
    with pytest.raises(ValueError):
        t975(0)


def test_known_sample():
    s = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["std"] == pytest.approx(math.sqrt(2.5))
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert s["ci95_low"] == pytest.approx(3 - half) and s["ci95_high"] == pytest.approx(3 + half)
    assert s["n"] == 5 and not s["deterministic"]


def test_constant_sample_is_deterministic_and_single_value_has_no_ci():
    s = summarise([7.5] * 5)
    assert s["deterministic"] and s["std"] == 0 and s["ci95_low"] == s["ci95_high"] == 7.5
    one = summarise([3.0])
    assert math.isnan(one["std"]) and not one["deterministic"]
    with pytest.raises(ValueError):
        summarise([])
