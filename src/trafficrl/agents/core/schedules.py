"""Exploration schedule (plan phase5-rl-baselines D7)."""

from __future__ import annotations


class LinearSchedule:
    """``start`` at step 0, linear to ``end`` at step ``duration``, constant afterwards."""

    def __init__(self, start: float, end: float, duration: float) -> None:
        self.start, self.end, self.duration = float(start), float(end), float(duration)

    def value(self, t: float) -> float:
        if self.duration <= 0:
            return self.end
        frac = min(max(t / self.duration, 0.0), 1.0)
        return self.start + frac * (self.end - self.start)
