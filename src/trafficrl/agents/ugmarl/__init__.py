"""UG-MARL: neighbour attention, Q-ensemble and uncertainty gate."""

from trafficrl.agents.ugmarl.agent import UGMARLAgent, variant_name
from trafficrl.agents.ugmarl.network import UGMARLNet, expected_params

__all__ = ["UGMARLAgent", "UGMARLNet", "expected_params", "variant_name"]
