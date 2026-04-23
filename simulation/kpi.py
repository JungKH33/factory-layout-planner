from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_v = sorted(float(v) for v in values)
    idx = (len(sorted_v) - 1) * float(q)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_v[lo])
    ratio = idx - lo
    return float(sorted_v[lo] * (1.0 - ratio) + sorted_v[hi] * ratio)


@dataclass
class KPICollector:
    created: int = 0
    completed: int = 0
    dropped: int = 0
    latencies_sec: List[float] = field(default_factory=list)
    timeline: List[Dict[str, float]] = field(default_factory=list)

    def on_created(self) -> None:
        self.created += 1

    def on_completed(self, latency_sec: float) -> None:
        self.completed += 1
        self.latencies_sec.append(max(0.0, float(latency_sec)))

    def on_dropped(self) -> None:
        self.dropped += 1

    def snapshot(self, sim_time: float, wip: int) -> None:
        self.timeline.append({"t": float(sim_time), "wip": float(max(0, int(wip)))})

    def summary(self, *, horizon_sec: float, warmup_sec: float) -> Dict[str, float]:
        effective_horizon = max(1e-6, float(horizon_sec) - float(warmup_sec))
        throughput = float(self.completed) * 3600.0 / effective_horizon
        avg_latency = (sum(self.latencies_sec) / len(self.latencies_sec)) if self.latencies_sec else 0.0
        return {
            "created_count": int(self.created),
            "completed_count": int(self.completed),
            "dropped_count": int(self.dropped),
            "completion_rate": float(self.completed / self.created) if self.created > 0 else 0.0,
            "throughput_per_hour": float(throughput),
            "latency_avg_sec": float(avg_latency),
            "latency_p95_sec": float(_percentile(self.latencies_sec, 0.95)),
            "wip_avg": float(
                sum(float(v["wip"]) for v in self.timeline) / len(self.timeline)
            )
            if self.timeline
            else 0.0,
        }

