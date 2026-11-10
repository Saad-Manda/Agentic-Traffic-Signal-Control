"""Hand-built one-intersection roadnet and flow records shared by the data / env tests.

Layout: signalised ``C`` at the origin, virtual nodes ``E``, ``N``, ``W``, ``S`` 300 m away.
Incoming roads are named ``in_<d>`` and outgoing roads ``out_<d>`` where ``d`` is the
direction of travel (0 E, 1 N, 2 W, 3 S). Three lanes per road (left, through, right).
"""

from __future__ import annotations

LENGTH = 300
# direction of travel -> unit vector
VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
NODE = {0: "E", 1: "N", 2: "W", 3: "S"}  # node reached when leaving C in direction d
# turn -> change of direction (counter-clockwise numbering: left = +1)
TURN = {"go_straight": 0, "turn_left": 1, "turn_right": -1}
LANE = {"turn_left": 0, "go_straight": 1, "turn_right": 2}


def toy_roadnet() -> dict:
    lanes = [{"width": 4, "maxSpeed": 11.111}] * 3
    roads = []
    for d, (vx, vy) in VEC.items():
        far = {"x": vx * LENGTH, "y": vy * LENGTH}
        near = {"x": 0, "y": 0}
        # out_d leaves C towards NODE[d]; in_d arrives at C travelling in direction d,
        # i.e. it comes from the node on the opposite side.
        opp = {"x": -vx * LENGTH, "y": -vy * LENGTH}
        roads.append(
            {
                "id": f"out_{d}",
                "points": [near, far],
                "lanes": lanes,
                "startIntersection": "C",
                "endIntersection": NODE[d],
            }
        )
        roads.append(
            {
                "id": f"in_{d}",
                "points": [opp, near],
                "lanes": lanes,
                "startIntersection": NODE[(d + 2) % 4],
                "endIntersection": "C",
            }
        )

    links = []
    for d in range(4):
        for kind in ("go_straight", "turn_left", "turn_right"):
            out_d = (d + TURN[kind]) % 4
            links.append(
                {
                    "type": kind,
                    "startRoad": f"in_{d}",
                    "endRoad": f"out_{out_d}",
                    "laneLinks": [
                        {"startLaneIndex": LANE[kind], "endLaneIndex": k, "points": []}
                        for k in range(3)
                    ],
                }
            )
    right = [i for i, link in enumerate(links) if link["type"] == "turn_right"]

    def idx(d: int, kind: str) -> int:
        return d * 3 + ("go_straight", "turn_left", "turn_right").index(kind)

    # 8 green phases: 4 "straight pair / left pair" + 4 "one approach straight + left"
    greens = [
        [idx(0, "go_straight"), idx(2, "go_straight")],
        [idx(1, "go_straight"), idx(3, "go_straight")],
        [idx(0, "turn_left"), idx(2, "turn_left")],
        [idx(1, "turn_left"), idx(3, "turn_left")],
        [idx(0, "go_straight"), idx(0, "turn_left")],
        [idx(2, "go_straight"), idx(2, "turn_left")],
        [idx(1, "go_straight"), idx(1, "turn_left")],
        [idx(3, "go_straight"), idx(3, "turn_left")],
    ]
    phases = [{"time": 5, "availableRoadLinks": right}] + [
        {"time": 30, "availableRoadLinks": sorted(g + right)} for g in greens
    ]

    inters = [
        {
            "id": "C",
            "point": {"x": 0, "y": 0},
            "width": 15,
            "roads": [r["id"] for r in roads],
            "roadLinks": links,
            "trafficLight": {"lightphases": phases},
            "virtual": False,
        }
    ]
    for d, name in NODE.items():
        vx, vy = VEC[d]
        inters.append(
            {
                "id": name,
                "point": {"x": vx * LENGTH, "y": vy * LENGTH},
                "width": 0,
                "roads": [f"out_{d}", f"in_{(d + 2) % 4}"],
                "roadLinks": [],
                "trafficLight": {"lightphases": []},
                "virtual": True,
            }
        )
    return {"intersections": inters, "roads": roads}


VEHICLE = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.111,
    "headwayTime": 2,
}


def toy_vehicle(start: int, route=("in_0", "out_0")) -> dict:
    return {
        "vehicle": dict(VEHICLE),
        "route": list(route),
        "interval": 1.0,
        "startTime": start,
        "endTime": start,
    }
