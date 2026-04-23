from __future__ import annotations

from typing import Iterable


_SUPPORTED = {"nearest_idle", "least_loaded"}


def normalize_dispatch_rule(name: str) -> str:
    rule = str(name).strip().lower()
    if rule not in _SUPPORTED:
        return "nearest_idle"
    return rule


def choose_wait_time(rule: str, waits: Iterable[float]) -> float:
    values = [float(v) for v in waits]
    if not values:
        return 0.0
    normalized = normalize_dispatch_rule(rule)
    if normalized == "least_loaded":
        return min(values)
    return min(values)

