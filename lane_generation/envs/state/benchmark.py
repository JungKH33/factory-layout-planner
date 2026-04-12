"""Construction-time routing benchmark.

Times each candidate algorithm on a few real flows from the state, separately
for ``single_src`` and ``multi_src`` flows, and returns the best per-method
algorithm name as a ``Dict[str, str]``.
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .state import LaneState, RoutingConfig


logger = logging.getLogger(__name__)


_METHOD_SINGLE = "single_src"
_METHOD_MULTI = "multi_src"
_ALL_METHODS = (_METHOD_SINGLE, _METHOD_MULTI)


def _method_key(*, src_port_count: int) -> str:
    return _METHOD_MULTI if src_port_count > 1 else _METHOD_SINGLE


def _candidate_algorithms(*, method: str) -> List[str]:
    if method == _METHOD_SINGLE:
        return ["astar", "wavefront", "dijkstra"]
    return ["wavefront", "astar", "dijkstra"]


def _collect_method_samples(
    state: "LaneState",
    *,
    max_per_method: int,
) -> Dict[str, List[int]]:
    samples: Dict[str, List[int]] = {m: [] for m in _ALL_METHODS}
    cap = max(1, int(max_per_method))
    for fi in range(int(state.flow_count)):
        sp, dp = state.valid_ports(fi)
        if sp.numel() == 0 or dp.numel() == 0:
            continue
        method = _method_key(src_port_count=int(sp.shape[0]))
        if len(samples[method]) >= cap:
            continue
        samples[method].append(fi)
        if all(len(samples[m]) >= cap for m in _ALL_METHODS):
            break
    return samples


def _bench_algorithm(
    state: "LaneState",
    algorithm: str,
    *,
    flow_indices: List[int],
    warmup: int,
    rounds: int,
) -> Tuple[float, float]:
    """Time pathfind() calls for sample flows with a specific algorithm."""
    saved = state._algorithm
    state._algorithm = algorithm

    def _run_once():
        for fi in flow_indices:
            src_ports, dst_ports = state.valid_ports(fi)
            src_gid, dst_gid, _ = state.flow_pair(fi)
            allow_mask = state.allow_mask() if state.forbid_opposite(fi) else None
            for j in range(int(src_ports.shape[0])):
                state.pathfind(
                    src_xy=(int(src_ports[j, 0]), int(src_ports[j, 1])),
                    dst_xy=dst_ports,
                    src_gid=src_gid,
                    dst_gid=dst_gid,
                    allow_mask=allow_mask,
                    rng=None,
                )

    try:
        for _ in range(max(0, int(warmup))):
            _run_once()

        times: List[float] = []
        for _ in range(max(1, int(rounds))):
            t0 = time.perf_counter()
            _run_once()
            times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        state._algorithm = saved

    if not times:
        return float("inf"), 0.0
    mean = float(statistics.mean(times))
    std = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
    return mean, std


def pick_best_algorithms(
    state: "LaneState",
    config: "RoutingConfig",
) -> Dict[str, str]:
    """Time each candidate algorithm per method and return the best.

    Returns ``{"single_src": ..., "multi_src": ...}`` so that flows with one
    source port and flows with several can each use the fastest algorithm.
    """
    samples = _collect_method_samples(
        state,
        max_per_method=int(config.benchmark_max_flows_per_method),
    )

    warmup = max(0, int(config.benchmark_warmup))
    rounds = max(1, int(config.benchmark_rounds))

    logger.info("=== lane route algorithm selection (mode=benchmark) ===")
    logger.info("  benchmarking per method (warmup=%d, rounds=%d)", warmup, rounds)

    selected: Dict[str, str] = {}
    for method in _ALL_METHODS:
        method_samples = samples.get(method, [])
        candidates = _candidate_algorithms(method=method)
        if not method_samples:
            selected[method] = candidates[0]
            logger.info("  %-10s | %s (no sample flow)", method, candidates[0])
            continue

        perf: List[Tuple[str, float, float]] = []
        for algo in candidates:
            mean_ms, std_ms = _bench_algorithm(
                state, algo,
                flow_indices=method_samples,
                warmup=warmup,
                rounds=rounds,
            )
            perf.append((algo, mean_ms, std_ms))

        parts = [f"{a} {mean:.3f}(+-{std:.3f})" for a, mean, std in perf]
        logger.info("  %-10s | %s", method, " | ".join(parts))
        best = min(perf, key=lambda x: x[1])[0]
        selected[method] = best

    logger.info("  selected lane algorithms:")
    for method in _ALL_METHODS:
        logger.info("  %-10s -> %s", method, selected[method])
    logger.info("lane route algorithm selection done")

    return selected
