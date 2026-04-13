"""Per-scenario lane routing benchmark report.

For each (device, scenario, algorithm) combo, measures:
  Timing:
    - mean / std of routing time (building candidate edge sets from scratch)
    - ref(ms): reward_composer.delta_batch() time on prebuilt candidates
               (the "Manhattan distance measurement" from
                lane_generation/envs/reward/flow.py)
    - x_time: mean / ref  --- how many times slower routing is vs pure reward eval
  Length:
    - path:  sum over flows of the shortest routed candidate length (edges)
    - manh:  sum over flows of min Manhattan distance across src/dst port pairs
    - x_len: path / manh  --- detour factor vs the Manhattan lower bound

Devices:
  - runs on CPU always
  - runs on CUDA additionally if available
  Note: astar/dijkstra in LaneAdapter._route_graph_search hardcode
  `free_map.detach().cpu().numpy()`, so those algorithms always execute on CPU
  regardless of the tensor device. wavefront stays on-device but its backtrace
  has per-step .item() sync, which hurts GPU throughput.

Run:
    python -m tmp.lane_bench_report
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from lane_generation.agents.placement.greedy import LaneAdapter, LaneAdapterConfig
from lane_generation.envs.env import FactoryLaneEnv
from lane_generation.envs.routing import (
    AStarStrategy,
    DijkstraStrategy,
    RoutingConfig,
    RoutingStrategy,
    WavefrontStrategy,
)
from lane_generation.envs.state import LaneFlowSpec


ALGORITHMS = ("wavefront", "astar", "dijkstra")
CANDIDATE_K = 4
WARMUP = 2
ROUNDS = 6


@dataclass
class Scenario:
    name: str
    grid_w: int
    grid_h: int
    blocked: torch.Tensor
    flows: List[LaneFlowSpec]


def _empty(gw: int, gh: int) -> torch.Tensor:
    return torch.zeros((gh, gw), dtype=torch.bool)


def _sparse(gw: int, gh: int) -> torch.Tensor:
    m = _empty(gw, gh)
    m[int(0.27 * gh):int(0.47 * gh), int(0.20 * gw):int(0.37 * gw)] = True
    m[int(0.27 * gh):int(0.47 * gh), int(0.63 * gw):int(0.80 * gw)] = True
    m[int(0.63 * gh):int(0.80 * gh), int(0.40 * gw):int(0.67 * gw)] = True
    return m


def _dense(gw: int, gh: int) -> torch.Tensor:
    m = _empty(gw, gh)
    step = max(20, min(gw, gh) // 8)
    block = step // 2
    for y in range(step, gh - step, step):
        for x in range(step, gw - step, step):
            m[y:y + block, x:x + block] = True
    return m


def _maze(gw: int, gh: int) -> torch.Tensor:
    m = _empty(gw, gh)
    row_step = max(24, gh // 6)
    gap = max(20, gw // 6)
    for i, y in enumerate(range(row_step, gh - row_step, row_step)):
        m[y:y + 3, :] = True
        if i % 2 == 0:
            m[y:y + 3, 0:gap] = False
        else:
            m[y:y + 3, gw - gap:gw] = False
    return m


def _flows_simple(gw: int, gh: int) -> List[LaneFlowSpec]:
    return [
        LaneFlowSpec(
            src_gid="A", dst_gid="B", weight=1.0,
            src_ports=((10, 10),), dst_ports=((gw - 10, gh - 10),),
        ),
        LaneFlowSpec(
            src_gid="A", dst_gid="C", weight=0.9,
            src_ports=((10, 10),), dst_ports=((gw - 10, 10),),
        ),
        LaneFlowSpec(
            src_gid="D", dst_gid="E", weight=0.7,
            src_ports=((10, gh - 10),), dst_ports=((gw - 10, gh - 10),),
        ),
    ]


def _flows_mixed(gw: int, gh: int) -> List[LaneFlowSpec]:
    mid_y = gh // 2
    return [
        LaneFlowSpec(
            src_gid="A", dst_gid="B", weight=1.0,
            src_ports=((10, 10),), dst_ports=((gw - 10, gh - 10),),
        ),
        LaneFlowSpec(
            src_gid="C", dst_gid="D", weight=0.8,
            src_ports=((10, gh - 10),), dst_ports=((gw - 10, 10),),
        ),
        LaneFlowSpec(
            src_gid="E", dst_gid="F", weight=0.6,
            src_ports=((10, mid_y - 20), (10, mid_y), (10, mid_y + 20)),
            dst_ports=((gw - 10, mid_y),),
        ),
        LaneFlowSpec(
            src_gid="G", dst_gid="H", weight=0.5,
            src_ports=((20, 20), (20, 30)),
            dst_ports=((gw - 20, gh - 20), (gw - 20, gh - 30)),
        ),
    ]


def make_scenarios() -> List[Scenario]:
    out: List[Scenario] = []
    gw, gh = 80, 80
    out.append(Scenario("empty_80", gw, gh, _empty(gw, gh), _flows_simple(gw, gh)))
    gw, gh = 150, 150
    out.append(Scenario("empty_150", gw, gh, _empty(gw, gh), _flows_mixed(gw, gh)))
    out.append(Scenario("sparse_150", gw, gh, _sparse(gw, gh), _flows_mixed(gw, gh)))
    gw, gh = 200, 200
    out.append(Scenario("dense_200", gw, gh, _dense(gw, gh), _flows_mixed(gw, gh)))
    out.append(Scenario("maze_200", gw, gh, _maze(gw, gh), _flows_mixed(gw, gh)))
    return out


def _collect_samples(state) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Return ``(flow_idx, src_ports, dst_ports)`` for every routable flow."""
    samples: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for fi in range(int(state.flow_count)):
        sp, dp = state.valid_ports(fi)
        if sp.numel() == 0 or dp.numel() == 0:
            continue
        samples.append((int(fi), sp.detach().clone(), dp.detach().clone()))
    return samples


def _make_strategy(name: str) -> RoutingStrategy:
    n = name.lower()
    if n == "wavefront":
        return WavefrontStrategy()
    if n == "astar":
        return AStarStrategy()
    if n == "dijkstra":
        return DijkstraStrategy()
    raise ValueError(f"unknown algo {name!r}")


def _manhattan_lb(src_ports: torch.Tensor, dst_ports: torch.Tensor) -> int:
    sp = src_ports.detach().cpu().tolist()
    dp = dst_ports.detach().cpu().tolist()
    best: int | None = None
    for s in sp:
        for d in dp:
            m = abs(int(s[0]) - int(d[0])) + abs(int(s[1]) - int(d[1]))
            if best is None or m < best:
                best = m
    return int(best or 0)


def _pack_candidates(
    cands: List[torch.Tensor],
    turns_l: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = len(cands)
    lmax = max(int(c.numel()) for c in cands)
    edge_idx = torch.zeros((k, lmax), dtype=torch.long, device=device)
    edge_mask = torch.zeros((k, lmax), dtype=torch.bool, device=device)
    for i, e in enumerate(cands):
        n = int(e.numel())
        if n > 0:
            edge_idx[i, :n] = e.to(device=device, dtype=torch.long)
            edge_mask[i, :n] = True
    turns = torch.tensor(turns_l, dtype=torch.float32, device=device)
    return edge_idx, edge_mask, turns


def bench_scenario(scn: Scenario, device: torch.device) -> Dict[str, dict]:
    is_cuda = device.type == "cuda"

    env = FactoryLaneEnv(
        grid_width=scn.grid_w,
        grid_height=scn.grid_h,
        blocked_static=scn.blocked.to(device),
        flows=scn.flows,
        device=device,
        routing_config=RoutingConfig(
            selection="static",
            algorithm="wavefront",
            candidate_k=CANDIDATE_K,
        ),
    )
    env.set_adapter(LaneAdapter(config=LaneAdapterConfig(candidate_k=CANDIDATE_K)))

    state = env.get_state()
    samples = _collect_samples(state)

    manh_total = sum(_manhattan_lb(sp, dp) for _fi, sp, dp in samples)

    # Prebuild one candidate set per flow (using a fresh wavefront strategy)
    # so we can time the reward eval in isolation.
    prebuild_strat = WavefrontStrategy()
    prebuilt: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for fi, _sp, _dp in samples:
        cands, turns_l = prebuild_strat.route_flow(
            state=state, flow_idx=fi, k=CANDIDATE_K, rng=None,
        )
        if not cands:
            continue
        prebuilt.append(_pack_candidates(cands, turns_l, device))

    def _sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    # --- Reference: delta_batch (reward eval) only ---
    reward = env.reward_composer
    for _ in range(WARMUP):
        for ei, em, tn in prebuilt:
            reward.delta_batch(
                state,
                candidate_edge_idx=ei,
                candidate_edge_mask=em,
                candidate_turns=tn,
            )
    _sync()
    ref_times: List[float] = []
    for _ in range(ROUNDS):
        _sync()
        t0 = time.perf_counter()
        for ei, em, tn in prebuilt:
            reward.delta_batch(
                state,
                candidate_edge_idx=ei,
                candidate_edge_mask=em,
                candidate_turns=tn,
            )
        _sync()
        ref_times.append((time.perf_counter() - t0) * 1000.0)
    ref_mean = float(statistics.mean(ref_times)) if ref_times else 0.0
    ref_std = float(statistics.pstdev(ref_times)) if len(ref_times) > 1 else 0.0

    results: Dict[str, dict] = {
        "_ref": {"mean_ms": ref_mean, "std_ms": ref_std},
        "_manh": {"total": int(manh_total), "flows": len(samples)},
    }

    for algo in ALGORITHMS:
        strat = _make_strategy(algo)
        for _ in range(WARMUP):
            for fi, _sp, _dp in samples:
                strat.route_flow(state=state, flow_idx=fi, k=CANDIDATE_K, rng=None)
        _sync()
        times: List[float] = []
        path_total_last = 0
        missing_last = 0
        for _ in range(ROUNDS):
            _sync()
            t0 = time.perf_counter()
            round_cands: List[Tuple[List[torch.Tensor], int]] = []
            for fi, _sp, _dp in samples:
                cands, _turns = strat.route_flow(
                    state=state, flow_idx=fi, k=CANDIDATE_K, rng=None,
                )
                round_cands.append((cands, 0))
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)

            path_total = 0
            missing = 0
            for cands, _ in round_cands:
                if cands:
                    path_total += min(int(c.numel()) for c in cands)
                else:
                    missing += 1
            path_total_last = path_total
            missing_last = missing

        mean_ms = float(statistics.mean(times))
        std_ms = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
        results[algo] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "x_time": (mean_ms / ref_mean) if ref_mean > 0 else float("inf"),
            "path_total": int(path_total_last),
            "x_len": (path_total_last / manh_total) if manh_total > 0 else float("inf"),
            "missing": int(missing_last),
        }
    return results


def print_report(all_results: Dict[Tuple[str, str], Dict[str, dict]]) -> None:
    header = (
        f"{'dev':<5}{'scenario':<14}{'algo':<11}"
        f"{'mean(ms)':>11}{'std(ms)':>10}"
        f"{'ref(ms)':>10}{'x_time':>10}"
        f"{'path':>8}{'manh':>8}{'x_len':>8}{'miss':>6}"
    )
    print(header)
    print("-" * len(header))
    for (device_name, scn_name), per_algo in all_results.items():
        ref = per_algo.get("_ref", {"mean_ms": 0.0, "std_ms": 0.0})
        manh = per_algo.get("_manh", {"total": 0})
        for algo in ALGORITHMS:
            r = per_algo.get(algo)
            if r is None:
                continue
            print(
                f"{device_name:<5}{scn_name:<14}{algo:<11}"
                f"{r['mean_ms']:>11.3f}{r['std_ms']:>10.3f}"
                f"{ref['mean_ms']:>10.3f}{r['x_time']:>9.1f}x"
                f"{r['path_total']:>8d}{int(manh['total']):>8d}"
                f"{r['x_len']:>7.2f}x{r['missing']:>6d}"
            )
        print()


def main() -> int:
    devices: List[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    scenarios = make_scenarios()
    all_results: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for device in devices:
        for scn in scenarios:
            tag = device.type
            print(f"[run] device={tag} scenario={scn.name} ({scn.grid_w}x{scn.grid_h}, flows={len(scn.flows)})")
            all_results[(tag, scn.name)] = bench_scenario(scn, device)
    print()
    print_report(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
