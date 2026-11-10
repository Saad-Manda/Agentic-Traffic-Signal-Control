"""Size and structure statistics for every dataset file in the split (ROADMAP 1.6).

Produces the numbers in ``docs/dataset_card.md``. Deliberately limited to sizes and
structural facts (counts, ranges, field sets, route validity) — no demand distributions
or plots — so it is safe to run on the held-out city (``database.md`` leakage rules).

    python scripts/dataset_stats.py                       # print markdown tables
    python scripts/dataset_stats.py --json docs/dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from trafficrl.data.splits import load_splits, verify_hashes


def roadnet_stats(path: Path) -> dict:
    rn = json.loads(path.read_text())
    inters = rn["intersections"]
    real = [i for i in inters if not i["virtual"]]
    roads = rn["roads"]
    lengths = [
        math.dist(
            (r["points"][0]["x"], r["points"][0]["y"]), (r["points"][-1]["x"], r["points"][-1]["y"])
        )
        for r in roads
    ]
    lanes = [ln for r in roads for ln in r["lanes"]]
    return {
        "intersections_signalised": len(real),
        "intersections_virtual": len(inters) - len(real),
        "roads": len(roads),
        "lanes": len(lanes),
        "road_length_m": dict(sorted(Counter(round(x) for x in lengths).items())),
        "lane_max_speed_mps": sorted({ln["maxSpeed"] for ln in lanes}),
        "lane_width_m": sorted({ln["width"] for ln in lanes}),
        "phases_per_intersection": sorted({len(i["trafficLight"]["lightphases"]) for i in real}),
        "road_links_per_intersection": sorted({len(i["roadLinks"]) for i in real}),
    }


def flow_stats(path: Path, road_ids: set[str]) -> dict:
    flows = json.loads(path.read_text())
    start = [v["startTime"] for v in flows]
    route_len = [len(v["route"]) for v in flows]
    return {
        "vehicles": len(flows),
        "start_time_s": [min(start), max(start)],
        "single_vehicle_records": sum(v["startTime"] == v["endTime"] for v in flows),
        "route_roads": [min(route_len), max(route_len)],
        "route_roads_mean": round(sum(route_len) / len(route_len), 2),
        "invalid_road_refs": sum(r not in road_ids for v in flows for r in v["route"]),
        "vehicle_profiles": len({json.dumps(v["vehicle"], sort_keys=True) for v in flows}),
        "vehicle_profile": flows[0]["vehicle"],
        "fields": sorted(flows[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="also write the stats to this JSON file")
    args = ap.parse_args()

    splits = load_splits()
    verify_hashes(splits)
    out = {}
    for name, ds in splits.items():
        rn_path = ds["roadnet"].path
        road_ids = {r["id"] for r in json.loads(rn_path.read_text())["roads"]}
        out[name] = {
            "role": ds["role"],
            "roadnet": {"file": rn_path.name, **roadnet_stats(rn_path)},
            "flows": {
                f.path.name: {"split": split, "sha256": f.sha256, **flow_stats(f.path, road_ids)}
                for split, files in ds["splits"].items()
                for f in files
            },
        }

    for name, d in out.items():
        r = d["roadnet"]
        print(f"\n### {name} ({d['role']}) — {r['file']}\n")
        print(
            f"{r['intersections_signalised']} signalised + {r['intersections_virtual']} virtual"
            f" intersections, {r['roads']} roads, {r['lanes']} lanes; road lengths (m: count)"
            f" {r['road_length_m']}; lane speed limit {r['lane_max_speed_mps']} m/s;"
            f" phases/intersection {r['phases_per_intersection']}\n"
        )
        print(
            "| Flow file | Split | Vehicles | startTime range (s) | Roads/route (min–max, mean)"
            " | Invalid road refs |"
        )
        print("|---|---|---|---|---|---|")
        for fn, s in d["flows"].items():
            print(
                f"| `{fn}` | {s['split']} | {s['vehicles']} | {s['start_time_s'][0]}–"
                f"{s['start_time_s'][1]} | {s['route_roads'][0]}–{s['route_roads'][1]},"
                f" {s['route_roads_mean']} | {s['invalid_road_refs']} |"
            )

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
