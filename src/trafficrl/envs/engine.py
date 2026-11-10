"""CityFlow engine factory shared by the environment wrapper and the EDA traces."""

from __future__ import annotations

import json
from pathlib import Path

from trafficrl.utils.seed import seed_cityflow


def make_engine(roadnet_path: Path, flow_path: Path, seed: int, threads: int, workdir: Path):
    """Create a seeded ``cityflow.Engine`` for one roadnet / flow pair (no replay output)."""
    import cityflow  # imported here so the rest of the package works without the simulator

    roadnet_path, flow_path = Path(roadnet_path).resolve(), Path(flow_path).resolve()
    if roadnet_path.parent != flow_path.parent:
        raise ValueError("roadnet and flow file must be in the same directory")
    cfg = {
        "interval": 1.0,
        "seed": seed,
        "dir": str(roadnet_path.parent) + "/",
        "roadnetFile": roadnet_path.name,
        "flowFile": flow_path.name,
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": False,
        "roadnetLogFile": "",
        "replayLogFile": "",
    }
    cfg_path = Path(workdir) / "cityflow_config.json"
    cfg_path.write_text(json.dumps(cfg))
    engine = cityflow.Engine(str(cfg_path), thread_num=threads)
    seed_cityflow(engine, seed)
    return engine
