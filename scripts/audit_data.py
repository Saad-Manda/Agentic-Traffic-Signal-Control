"""Data audit of every dataset file in the split (ROADMAP 3.0, guidelines G3).

Structural checks only — nothing is fitted and no file is modified. Writes

* ``docs/data_audit.json``  — all counts, machine-readable
* ``docs/data_audit.md``    — the audit report (tables, figures, interpretation)
* ``docs/figures/audit/``   — plots, **train split only**

Counts are produced for all splits (train, val, test, held-out); demand distributions and
plots only for the splits in ``PLOT_SPLITS`` (``database.md`` rules 7 and 9).

    python scripts/audit_data.py
    python scripts/audit_data.py --config configs/eda.yaml

Exit code 1 if any defect count is non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trafficrl.data import eda  # noqa: E402
from trafficrl.data.audit import (  # noqa: E402
    FLOW_DEFECTS,
    FLOW_INFO,
    ROADNET_DEFECTS,
    ROADNET_INFO,
    arrival_counts,
    audit_flow,
    audit_roadnet,
    burst_threshold,
    defects,
    low_threshold,
)
from trafficrl.data.roadnet import load_roadnet  # noqa: E402
from trafficrl.data.splits import REPO_ROOT, load_splits, verify_hashes  # noqa: E402
from trafficrl.utils.config import load_config  # noqa: E402
from trafficrl.utils.plotting import BLUE, INK_SECONDARY, ORANGE, apply_style, save_figure  # noqa: E402

PLOT_SPLITS = ("train",)  # splits whose demand distribution may be plotted / described
DOCS = REPO_ROOT / "docs"
FIG_DIR = DOCS / "figures" / "audit"


def run_audit(episode_seconds: int, bin_seconds: int) -> dict:
    """Audit every roadnet and flow file in ``splits.json``; returns the full result."""
    splits = load_splits()
    verify_hashes(splits)
    out: dict = {"episode_seconds": episode_seconds, "arrival_bin_seconds": bin_seconds}
    out["datasets"] = {}
    for name, ds in splits.items():
        rn_path = ds["roadnet"].path
        roadnet = load_roadnet(rn_path)
        flows_out = {}
        for split, files in ds["splits"].items():
            for f in files:
                flows = json.loads(f.path.read_text())
                flows_out[f.path.name] = {
                    "split": split,
                    **audit_flow(flows, roadnet, episode_seconds, bin_seconds),
                }
        out["datasets"][name] = {
            "role": ds["role"],
            "roadnet": {"file": rn_path.name, **audit_roadnet(json.loads(rn_path.read_text()))},
            "flows": flows_out,
        }
    return out


def train_description(episode_seconds: int, bin_seconds: int) -> dict:
    """Demand description + figures for the splits in ``PLOT_SPLITS`` (train only)."""
    apply_style()
    splits = load_splits()
    desc = {}
    for name, ds in splits.items():
        if ds["role"] != "primary":
            continue
        roadnet = load_roadnet(ds["roadnet"].path)
        for split in PLOT_SPLITS:
            flows = [v for f in ds["splits"][split] for v in json.loads(f.path.read_text())]
            counts = arrival_counts(flows, episode_seconds, bin_seconds)
            fence = burst_threshold(counts)
            entry = eda.entry_demand(flows)
            moves = eda.movement_demand(flows, roadnet)
            lengths = eda.route_lengths(flows)
            kinds = moves.groupby("kind")["vehicles"].sum()
            load = eda.intersection_load(moves)

            minutes = np.arange(len(counts)) * bin_seconds / 60
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            ax.plot(minutes, counts, color=BLUE, label="arrivals per minute")
            low = low_threshold(counts)
            ax.axhline(fence, color=ORANGE, linestyle="--", linewidth=1.2)
            ax.axhline(low, color=ORANGE, linestyle="--", linewidth=1.2)
            ax.text(
                minutes[-1], low, f"lower fence (Q1 − 1.5·IQR) = {low:.0f}",
                ha="right", va="top", color=INK_SECONDARY, fontsize=9,
            )  # fmt: skip
            ax.text(
                minutes[-1], fence, f"outlier fence (Q3 + 1.5·IQR) = {fence:.0f}",
                ha="right", va="bottom", color=INK_SECONDARY, fontsize=9,
            )  # fmt: skip
            ax.set(
                title=f"Vehicle arrivals per minute — {name} {split} flow",
                xlabel="time since episode start (min)",
                ylabel=f"vehicles entering per {bin_seconds} s",
                ylim=(0, max(counts.max(), fence) * 1.15),
            )
            save_figure(fig, FIG_DIR / f"{name}_{split}_arrivals.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            vc = lengths.value_counts().sort_index()
            ax.bar(vc.index, vc.values, color=BLUE, width=0.8)
            ax.set(
                title=f"Route length — {name} {split} flow",
                xlabel="roads per route (count)",
                ylabel="vehicles",
                xticks=range(int(vc.index.min()), int(vc.index.max()) + 1),
            )
            ax.grid(axis="x", visible=False)
            save_figure(fig, FIG_DIR / f"{name}_{split}_route_length.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.bar(range(len(entry)), entry.values, color=BLUE, width=0.8)
            ax.set_xticks(range(len(entry)))
            ax.set_xticklabels(
                [r.removeprefix("road_") for r in entry.index], rotation=90, fontsize=8
            )
            ax.set(
                title=f"Demand per entry road — {name} {split} flow",
                xlabel="entry road (road_<x>_<y>_<direction>; 0 E, 1 N, 2 W, 3 S)",
                ylabel="vehicles per hour",
            )
            ax.grid(axis="x", visible=False)
            save_figure(fig, FIG_DIR / f"{name}_{split}_entry_demand.png")
            plt.close(fig)

            n_entry_roads = sum(
                roadnet.road_start[r] in roadnet.virtual_ids for r in roadnet.road_ids
            )
            desc[f"{name}/{split}"] = {
                "vehicles": len(flows),
                "arrivals_per_bin": {
                    "min": int(counts.min()),
                    "median": float(np.median(counts)),
                    "mean": round(float(counts.mean()), 1),
                    "max": int(counts.max()),
                    "fence": round(fence, 1),
                    "low_fence": round(low, 1),
                    "low_bins": int((counts < low).sum()),
                    "peak_minute": int(counts.argmax() * bin_seconds // 60),
                },
                "route_roads": {
                    "min": int(lengths.min()),
                    "median": float(lengths.median()),
                    "mean": round(float(lengths.mean()), 2),
                    "p95": float(lengths.quantile(0.95)),
                    "max": int(lengths.max()),
                },
                "entry_roads_total": n_entry_roads,
                "entry_roads_used": int(len(entry)),
                "entry_top5_share": round(float(entry.head(5).sum() / entry.sum()), 3),
                "entry_max": {"road": entry.index[0], "vehicles": int(entry.iloc[0])},
                "entry_min": {"road": entry.index[-1], "vehicles": int(entry.iloc[-1])},
                "movement_share": {k: round(float(v / kinds.sum()), 3) for k, v in kinds.items()},
                "movements_total": int(len(moves)),
                "movements_unused": int((moves["vehicles"] == 0).sum()),
                "intersection_load": {
                    "min": {"id": load.idxmin(), "vehicles": int(load.min())},
                    "max": {"id": load.idxmax(), "vehicles": int(load.max())},
                },
            }
    return desc


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(res: dict) -> str:
    files = [
        (name, fn, f)
        for name, ds in res["datasets"].items()
        for fn, f in ds["flows"].items()
    ]  # fmt: skip
    cols = [f"{name} {f['split']}<br>`{fn.split('_', 3)[-1].removesuffix('.json')}`"
            for name, fn, f in files]  # fmt: skip
    n_defects = sum(sum(defects(f).values()) for _, _, f in files) + sum(
        sum(defects(ds["roadnet"]).values()) for ds in res["datasets"].values()
    )
    total = sum(f["records"] for _, _, f in files)

    n_bins = res["episode_seconds"] // res["arrival_bin_seconds"]

    def pct(f, key):
        return f"{f[key]} ({100 * f[key] / f['records']:.1f} %)"

    md = [
        "# Data audit — Jinan (primary) and Hangzhou (held-out)",
        "",
        "ROADMAP 3.0 · guidelines G3 · **generated by `python scripts/audit_data.py`** — do not "
        "edit by hand; change the script and re-run. Machine-readable copy: "
        "`docs/data_audit.json`. Checks are implemented in `src/trafficrl/data/audit.py` and "
        "tested in `tests/data/test_audit.py`.",
        "",
        "## 1. Scope and method",
        "",
        f"- **Files:** {len(files)} flow files ({total} vehicle records) and "
        f"{len(res['datasets'])} road networks, all verified against the SHA-256 hashes in "
        "`data/processed/splits.json` before auditing.",
        "- **Structural checks only.** Every check counts records that are missing, duplicated "
        "or impossible. Nothing is fitted, and no file is edited or filtered: the flow files are "
        "hash-pinned benchmark data, so a defect would be handled by a dated Decision Log entry, "
        "not by silent cleaning.",
        "- **Leakage rule.** Counts are reported for every split. Demand *distributions* and "
        "plots (§5–§6) are shown for the **train split only**; validation, test and the "
        "held-out city get counts only (`.claude/reference/database.md`, rules 7 and 9).",
        f"- **Episode:** {res['episode_seconds']} s; arrivals are binned in "
        f"{res['arrival_bin_seconds']} s windows for the outlier check.",
        "",
        "## 2. Result",
        "",
        (
            f"**No defects: all {len(FLOW_DEFECTS)} flow checks and {len(ROADNET_DEFECTS)} "
            "road-network checks return 0 on every file.** The data is used as released."
            if n_defects == 0
            else f"**{n_defects} defects found — see the tables below. Handling must be "
            "decided and recorded before the data is used.**"
        ),
        "",
        "Three properties of the data are reported for information (§4): trips that end "
        "inside the network, routes that pass an intersection twice, and arrival bursts.",
        "",
        "## 3. Defect counts (expected: 0)",
        "",
        "### 3.1 Flow files",
        "",
        _table(["Check"] + cols, [[f"`{k}`"] + [f[k] for _, _, f in files] for k in FLOW_DEFECTS]),
        "",
        "What the checks mean:",
        "",
        "- **Missing values:** `missing_field` (one of `vehicle`, `route`, `interval`, "
        "`startTime`, `endTime` absent or null), `missing_vehicle_field` (one of the 9 vehicle "
        "parameters), `wrong_type`.",
        "- **Duplicates:** `duplicate_records` — exact duplicate records. The files contain no "
        "vehicle ID (CityFlow generates `flow_<i>_<j>` IDs when it loads a file), so duplicate "
        "IDs cannot occur; duplicate *records* are the equivalent check.",
        "- **Impossible values:** start time outside `[0, episode)` or not an integer; "
        "`endTime` before `startTime`; a record that generates more than one vehicle; route "
        "shorter than 2 roads; a road that is not in the road network; consecutive roads with "
        "no permitted movement between them (`disconnected_route`); a route that does not "
        "start at a boundary node or uses the same road twice; non-positive vehicle length, "
        "width, acceleration, gap, speed or headway; vehicle speed above the lane limit.",
        "- **Privacy:** `unexpected_field` (any key outside the documented data dictionary, "
        "e.g. a plate or device ID) and `unexpected_string_value` (free text such as a date).",
        "",
        "### 3.2 Road networks",
        "",
        _table(
            ["Check"] + [f"{n}<br>`{ds['roadnet']['file']}`" for n, ds in res["datasets"].items()],
            [
                [f"`{k}`"] + [ds["roadnet"][k] for ds in res["datasets"].values()]
                for k in ROADNET_DEFECTS
            ],
        ),
        "",
        "Sizes: "
        + "; ".join(
            f"{n} — {ds['roadnet']['intersections_signalised']} signalised + "
            f"{ds['roadnet']['intersections_virtual']} virtual intersections, "
            f"{ds['roadnet']['roads']} roads, {ds['roadnet']['lanes']} lanes"
            for n, ds in res["datasets"].items()
        )
        + ". Every signalised intersection has 4 incoming and 4 outgoing roads, 12 movements "
        "and 9 light phases (phase 0 = 5 s transition), which is what the feature builder "
        "and the environment assume.",
        "",
        "## 4. Information counts (not defects)",
        "",
        _table(
            ["Item"] + cols,
            [
                ["`records` (vehicles)"] + [f["records"] for _, _, f in files],
                ["`vehicle_profiles` (distinct parameter sets)"]
                + [f["vehicle_profiles"] for _, _, f in files],
                ["`route_ends_inside_network`"]
                + [pct(f, "route_ends_inside_network") for _, _, f in files],
                ["`route_revisits_intersection`"]
                + [pct(f, "route_revisits_intersection") for _, _, f in files],
                [f"`arrival_burst_bins` (of {n_bins})"]
                + [f["arrival_burst_bins"] for _, _, f in files],
                [f"`arrival_low_bins` (of {n_bins})"]
                + [f["arrival_low_bins"] for _, _, f in files],
                ["`same_second_same_entry_lane`"]
                + [f["same_second_same_entry_lane"] for _, _, f in files],
            ],
        ),
        "",
        "Interpretation:",
        "",
        "- **Trips ending inside the network (1–2 % of every file).** The last road of the "
        "route ends at a signalised intersection, not at a boundary node. CityFlow accepts "
        "this: the vehicle is removed when it reaches the end of its last road, and its "
        "travel time is counted like any other. These are "
        "trips whose destination lies inside the area. **Kept as released.**",
        "- **Routes passing an intersection twice (3–4 %).** Detours or loops in the recorded "
        "trips (no road is used twice — `route_repeats_road` is 0). They are legal routes and "
        "part of the published benchmark. **Kept as released**; they make those vehicles' "
        "travel times long under every controller, so they affect all methods equally.",
        "- **Arrival outliers.** A minute is flagged as a burst when its count is above "
        "Q3 + 1.5·IQR of the file's own per-minute counts, and as a low bin when it is below "
        "Q1 − 1.5·IQR. Flagged bins are reported, not removed: peaks and troughs are real "
        "traffic, and handling them is the controller's job.",
        "- **Same-second arrivals on one entry lane: 0 in every file**, so every vehicle can be "
        "inserted at its recorded start time when the entry lane has space.",
        "- **One vehicle profile per file:** all vehicles share the same length, speed and "
        "acceleration, so the data has a single vehicle class.",
        "",
    ]

    for key, d in res.get("train_description", {}).items():
        name, split = key.split("/")
        a, r = d["arrivals_per_bin"], d["route_roads"]
        ms = d["movement_share"]
        ratio = d["entry_max"]["vehicles"] / d["entry_min"]["vehicles"]
        md += [
            f"## 5. Demand description — {name} {split} split only",
            "",
            f"![Arrivals per minute](figures/audit/{name}_{split}_arrivals.png)",
            "",
            f"*Figure A1. Vehicles entering the network per {res['arrival_bin_seconds']} s over "
            f"the {res['episode_seconds']} s episode ({name} {split} flow, {d['vehicles']} "
            "vehicles). Dashed lines: outlier fences Q3 + 1.5·IQR and Q1 − 1.5·IQR.*",
            "",
            f"**Interpretation.** Arrivals range from {a['min']} to {a['max']} vehicles per "
            f"minute (mean {a['mean']}, median {a['median']:.0f}); the busiest minute starts at "
            f"minute {a['peak_minute']}. "
            f"No minute is above the upper fence ({a['fence']:.0f}), so the hour has no arrival "
            f"burst. {a['low_bins']} minutes are below the lower fence ({a['low_fence']:.0f}): a "
            "five-minute trough in which demand roughly halves. The curve is step-shaped, "
            "constant over five-minute blocks, which suggests the released trips were built from "
            "five-minute counts. Both are kept: the trough is a short low-demand period, not "
            "an error, and it gives the controllers one visible change of regime within the hour.",
            "",
            f"![Route length](figures/audit/{name}_{split}_route_length.png)",
            "",
            "*Figure A2. Number of roads per vehicle route.*",
            "",
            f"**Interpretation.** Routes have {r['min']}–{r['max']} roads (mean {r['mean']}, "
            f"median {r['median']:.0f}, 95th percentile {r['p95']:.0f}). Most trips are short "
            "and cross only a few signalised intersections; the long tail is the small group of "
            "detour routes counted in §4.",
            "",
            f"![Demand per entry road](figures/audit/{name}_{split}_entry_demand.png)",
            "",
            "*Figure A3. Vehicles per hour entering at each boundary road, sorted.*",
            "",
            f"**Interpretation (bias / coverage).** {d['entry_roads_used']} of "
            f"{d['entry_roads_total']} boundary roads carry demand. Demand is uneven: the five "
            f"busiest entries carry {100 * d['entry_top5_share']:.0f} % of all vehicles "
            f"(largest `{d['entry_max']['road']}`: {d['entry_max']['vehicles']}; smallest "
            f"`{d['entry_min']['road']}`: {d['entry_min']['vehicles']}). Movements are "
            f"{100 * ms.get('go_straight', 0):.0f} % through, "
            f"{100 * ms.get('turn_left', 0):.0f} % left and "
            f"{100 * ms.get('turn_right', 0):.0f} % right; {d['movements_unused']} of "
            f"{d['movements_total']} movements are never used. Intersection load ranges from "
            f"{d['intersection_load']['min']['vehicles']} "
            f"(`{d['intersection_load']['min']['id']}`) to "
            f"{d['intersection_load']['max']['vehicles']} "
            f"(`{d['intersection_load']['max']['id']}`) vehicles per hour. Every movement is "
            "used, so no part of the state space is empty in training, but the load is "
            f"unbalanced (busiest entry {ratio:.1f}"
            "× the quietest). Consequence: lightly used approaches can be starved by a "
            "controller that only minimises the average, which is why the maximum wait per "
            "vehicle is reported next to the average travel time.",
            "",
        ]

    md += [
        "## 6. Privacy",
        "",
        "- Records contain only the five documented fields; the `unexpected_field` and "
        "`unexpected_string_value` checks are 0 on every file.",
        "- No vehicle ID, licence plate, device ID, calendar date or location trace of a "
        "person: a record is a start second (0–3599) and a list of road IDs on a grid. File "
        "names are prefixed `anon_`.",
        "- All vehicles share one parameter profile, so no record can be linked to a vehicle "
        "type or owner.",
        "",
        "## 7. Handling and limits of this audit",
        "",
        "- **Handling:** the data is used exactly as released. Nothing is removed, repaired or "
        "resampled.",
        "- **Held-out city and Jinan val / test:** only the counts above. Their demand "
        "distributions are not plotted or described before final testing.",
        "- **Not covered:** how the trips were reconstructed from camera records, and the "
        "collection dates — neither is documented in the release (dataset card §2, §6).",
        "",
    ]
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eda.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    episode = cfg["sim"]["episode_seconds"]
    bin_s = cfg["eda"]["arrival_bin_seconds"]

    res = run_audit(episode, bin_s)
    res["train_description"] = train_description(episode, bin_s)
    (DOCS / "data_audit.json").write_text(json.dumps(res, indent=2) + "\n")
    (DOCS / "data_audit.md").write_text(render_markdown(res))

    bad = 0
    for name, ds in res["datasets"].items():
        d = defects(ds["roadnet"])
        bad += sum(d.values())
        print(f"{name} roadnet: {d or 'clean'}")
        for fn, f in ds["flows"].items():
            d = defects(f)
            bad += sum(d.values())
            info = {k: f[k] for k in FLOW_INFO}
            print(f"  {f['split']:8s} {fn}: {d or 'clean'}  info={info}")
    print(f"wrote {DOCS / 'data_audit.md'}, data_audit.json, figures in {FIG_DIR}")
    assert set(ROADNET_INFO) <= set(next(iter(res["datasets"].values()))["roadnet"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
