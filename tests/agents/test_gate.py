"""Uncertainty gate helpers (plan phase6-ugmarl D5, D9)."""

import pandas as pd
import pytest
import torch

from trafficrl.agents.ugmarl.gate import (
    gate_file_for,
    load_gate,
    percentile_thresholds,
    save_gate,
    select_percentile,
    sigma_from_q,
)


def test_sigma_by_hand():
    # 1 state, K = 2 heads, A = 3: mean Q = [1, 3, 2] -> a* = 1; head values at a* = 2 and 4
    q = torch.tensor([[[0.0, 2.0, 1.0], [2.0, 4.0, 3.0]]])
    mu, a_star, sigma = sigma_from_q(q)
    assert torch.allclose(mu, torch.tensor([[1.0, 3.0, 2.0]]))
    assert a_star.tolist() == [1] and sigma.item() == pytest.approx(1.0)  # population std


def test_single_head_sigma_is_zero():
    assert sigma_from_q(torch.randn(4, 1, 8))[2].eq(0).all()


def test_percentile_thresholds():
    t = percentile_thresholds(list(range(101)), [80, 90, 95, 99])
    assert t == {80.0: 80.0, 90.0: 90.0, 95.0: 95.0, 99.0: 99.0}
    with pytest.raises(ValueError):
        percentile_thresholds([], [90])


def test_select_percentile_prefers_smaller_p_within_tolerance():
    df = pd.DataFrame({"p": [80, 90, 95, 99], "score": [310.8, 310.0, 312.0, 309.9]})
    assert select_percentile(df, 1.0) == 80  # 310.8 is within 1 s of 309.9
    assert select_percentile(df, 0.0) == 99


def test_gate_file_round_trip_and_location(tmp_path):
    f = save_gate(tmp_path / "run" / "gate.json", {"p": 90, "tau": 0.12})
    assert load_gate(f)["tau"] == 0.12
    assert gate_file_for("r/x_s0/checkpoints/best.pt").as_posix() == "r/x_s0/gate.json"
    with pytest.raises(ValueError):
        save_gate(tmp_path / "g.json", {"p": 90})
