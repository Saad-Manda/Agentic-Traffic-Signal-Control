"""Every Phase 8 figure function runs on a small fake study and writes PNG + PDF + caption."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from trafficrl.eval import analysis as A
from trafficrl.eval import figures as F

MODELS = ["fixed_time", "max_pressure", "dqn", "idqn", "ugmarl", "ugmarl-nogate"]
CONDS = [{}, {"robustness.noise_std": 0.1}, {"robustness.noise_std": 0.2},
         {"robustness.dropout_p": 0.1}, {"robustness.dropout_p": 0.2},
         {"robustness.surge": 1.5}]  # fmt: skip
NAMES = ["clean", "noise0.1", "noise0.2", "drop0.1", "drop0.2", "surge1.5"]
IIDS = [f"intersection_{x}_{y}" for x in (1, 2) for y in (1, 2)]
SEEDS = [0, 1, 2]
FLOW = [{"route": ["road_0_1_0", "road_1_1_0"][: 2 + 0], "startTime": 10 * i} for i in range(40)]


def write_job(root, model, cond, rng):
    d = root / "eval" / f"{model}__{cond}" / "run"
    d.mkdir(parents=True)
    det = model in ("fixed_time", "max_pressure")
    rows = []
    for s in SEEDS:
        noise = 0.0 if det else rng.normal(0, 2)
        rows.append({"seed": s, "avg_travel_time_s": 300 + 20 * MODELS.index(model) + noise,
                     "max_waiting_time_s": 900 + noise * 10, "avg_queue_veh": 1.0,
                     "fallback_rate": 0.2})  # fmt: skip
        times = np.arange(0, 600, 15)
        tr = pd.DataFrame(
            [{"t": t, "iid": i, "phase": 1 + (k + j) % 3, "in_yellow": False, "ready": True,
              **{f"q_{n}": int(rng.integers(0, 4)) for n in range(12)},
              **{f"n_{n}": 0 for n in range(12)}}
             for k, t in enumerate(times) for j, i in enumerate(IIDS)]
        )  # fmt: skip
        tr.to_csv(d / f"trace_s{s}.csv", index=False)
        pd.DataFrame(
            {"vehicle": [f"flow_{i}_0" for i in range(40)],
             "enter_t": [10 * i + 1 for i in range(40)],
             "leave_t": [10 * i + 100.0 + rng.integers(0, 50) for i in range(40)],
             "wait_s": rng.integers(0, 60, 40)}
        ).to_csv(d / f"vehicles_s{s}.csv", index=False)  # fmt: skip
        if model.startswith("ugmarl"):
            n = len(tr)
            w = rng.dirichlet(np.ones(3), n)
            sig = rng.uniform(0.001, 0.02, n)
            pd.DataFrame(
                {"t": tr["t"], "iid": tr["iid"], "sigma": sig, "tau": 0.01,
                 "fallback": (sig > 0.01) & (model == "ugmarl"),
                 "greedy_phase": rng.integers(1, 4, n), "mp_phase": rng.integers(1, 4, n),
                 "phase_taken": 1, "attn_self": w[:, 0], "attn_E": w[:, 1], "attn_N": w[:, 2],
                 "attn_W": 0.0, "attn_S": 0.0, "q_gap": 0.1, "in_veh_observed": 3.0}
            ).to_csv(d / f"decisions_s{s}.csv", index=False)  # fmt: skip
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    rng = np.random.default_rng(0)
    for m in MODELS:
        for c in NAMES:
            write_job(root, m, c, rng)
    spec = {"seeds": SEEDS, "models": {m: {} for m in MODELS}, "conditions": CONDS}
    return A.Study(spec, root)


def written(out, name):
    return all((out / f"{name}.{ext}").stat().st_size > 0 for ext in ("png", "pdf", "md"))


def test_result_figures(study, tmp_path):
    agents = MODELS[:5]
    F.fig_main_comparison(study, agents, tmp_path)
    F.fig_ablation(study, ["ugmarl", "ugmarl-nogate"], ["clean", "noise0.2"], tmp_path)
    F.fig_wait_tail(study, MODELS, ["clean", "noise0.2"], tmp_path)
    F.fig_generalisation(study, MODELS, tmp_path)
    F.fig_seed_spread(study, ["dqn", "ugmarl", "ugmarl-nogate"], tmp_path)
    F.fig_robustness(study, MODELS, tmp_path)
    F.fig_regret(study, MODELS, agents, tmp_path)
    F.fig_gate_firing(study, "ugmarl", NAMES, ["clean", "noise0.2"], 300, tmp_path)
    F.fig_attention_map(study, "ugmarl", "clean", tmp_path)
    for name in ("main_comparison", "ablation", "wait_tail", "generalisation", "seed_spread",
                 "robustness_degradation", "regret", "gate_firing", "attention_map"):  # fmt: skip
        assert written(tmp_path, name), name
    text = (tmp_path / "main_comparison.md").read_text()
    assert "**Caption.**" in text and "**Values.**" in text and "associations" in text
    plt.close("all")


def test_regret_frames_use_the_reference_agents(study):
    reg, summ = F.regret_frames(study, MODELS, MODELS[:5])
    assert (reg[reg["model"] == "fixed_time"]["regret"] == 0).all()  # lowest mean by design
    assert set(summ["model"]) == set(MODELS)


def test_error_figures_and_cases(study, tmp_path):
    fr = A.breakdowns(study, FLOW, ["ugmarl", "dqn", "max_pressure"], ["clean", "noise0.2"], 300)
    chk = fr["check"]
    assert np.allclose(chk["queue_logs"], chk["queue_logs"].round(6))  # frames built
    pairs = [(b, c) for c in ("clean", "noise0.2") for b in ("dqn", "max_pressure")]
    ci = {k: A.comparison(fr, "intersections", ["iid"], "wait_veh_s", "ugmarl", *k) for k in pairs}
    cw = {k: A.comparison(fr, "windows", ["window"], "travel_s", "ugmarl", *k) for k in pairs}
    cr = {k: A.comparison(fr, "length", ["length_group"], "travel_s", "ugmarl", *k) for k in pairs}
    F.fig_error_intersections(ci, "ugmarl", tmp_path)
    F.fig_error_windows(cw, "ugmarl", 300, tmp_path)
    F.fig_error_routes(cr, "ugmarl", "length_group", tmp_path)
    dec = study.log("ugmarl", "clean", "decisions", 0)
    cases = A.pick_cases(
        A.unit_table(study.log("ugmarl", "clean", "trace", 0), 300),
        A.unit_table(study.log("dqn", "clean", "trace", 0), 300),
        dec,
        300,
    )
    for kind, unit in cases.items():
        series = {m: A.case_series(study.log(m, "clean", "trace", 0), unit["iid"],
                                   unit["window"], 300) for m in ("ugmarl", "dqn")}  # fmt: skip
        F.fig_case(kind, unit, series, dec, 300, 0, "clean", tmp_path)
        assert written(tmp_path, f"case_{kind}")
    for name in ("error_intersections", "error_windows", "error_routes"):
        assert written(tmp_path, name), name
    st = A.stability_table(study, ["ugmarl"], ["clean"])
    assert 0 <= st["flip_flop_share"].iloc[0] <= 1
    assert A.seed_spread(study, ["dqn"], "clean", "avg_travel_time_s")["range"].iloc[0] >= 0
    plt.close("all")


def test_tau_and_faithfulness_figures(tmp_path):
    rows = []
    for p in ["max_pressure", 50.0, 80.0, 90.0, "ungated"]:
        for c in ("clean", "noise0.2"):
            rows.append({"p": p, "condition": c, "avg_travel_time_mean": 300.0 + len(rows),
                         "fallback_rate_mean": 0.5})  # fmt: skip
    F.fig_tau_sensitivity(pd.DataFrame(rows), 80, tmp_path)
    per = pd.DataFrame(
        [{"seed": s, "perturbation": p, "slot": k, "abs_dmu": 0.01 * (i + 1) + 0.001 * s,
          "flip": 0.02 * (i + 1)}
         for s in range(3) for p in ("replacement", "knock-out")
         for i, k in enumerate(("hi", "lo", "self"))]
    )  # fmt: skip
    F.fig_faithfulness(per, tmp_path)
    assert written(tmp_path, "tau_sensitivity") and written(tmp_path, "attention_faithfulness")
    plt.close("all")


def test_grid_position_from_the_intersection_id():
    assert F.grid_xy("intersection_4_3") == (4, 3)
    with pytest.raises(ValueError):
        F.grid_xy("node7")
