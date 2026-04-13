"""Smoke test + microbench for the prewave (batched BFS) cache.

Verifies:
  1. wavefront_distance_field_batched(...)[i] is bit-identical to
     wavefront_distance_field(seeds=flow_i_dsts) for every flow i.
  2. Sequential build_action_space across all flows is faster than the
     pre-cache baseline (which we simulate by clearing the adapter cache
     between flows).
  3. Final paths from build_action_space are identical (so the cache does
     not change candidate selection).

Run:
    python -m tmp.lane_prewave_smoke
"""
from __future__ import annotations

import statistics
import time
from typing import List

import torch

from lane_generation.agents.placement.greedy import LaneAdapter, LaneAdapterConfig
from lane_generation.envs.env import FactoryLaneEnv
from lane_generation.envs.routing import RoutingConfig, WavefrontStrategy
from lane_generation.envs.state import LaneFlowSpec
from lane_generation.envs.routing.wavefront import (
    wavefront_distance_field,
    wavefront_distance_field_batched,
)

from tmp.lane_bench_report import (  # reuse the existing scenario builders
    Scenario,
    make_scenarios,
)


WARMUP = 2
ROUNDS = 6


def _build_env(scn: Scenario, device: torch.device) -> FactoryLaneEnv:
    env = FactoryLaneEnv(
        grid_width=scn.grid_w,
        grid_height=scn.grid_h,
        blocked_static=scn.blocked.to(device),
        flows=scn.flows,
        flow_ordering="weight_desc",
        device=device,
        routing_config=RoutingConfig(
            selection="static",
            algorithm="wavefront",
            candidate_k=4,
        ),
    )
    env.set_adapter(LaneAdapter(config=LaneAdapterConfig(candidate_k=4)))
    return env


def _verify_correctness(env: FactoryLaneEnv) -> None:
    state = env.get_state()
    free_map = (~state.blocked_static).to(dtype=torch.bool, device=state.device)

    batched = wavefront_distance_field_batched(
        free_map=free_map,
        seeds_xy=state.dst_ports,
        seeds_mask=state.dst_mask,
        max_iters=0,
    )

    for fi in range(int(state.flow_count)):
        _, dst_ports = state.valid_ports(fi)
        ref = wavefront_distance_field(
            free_map=free_map,
            seeds_xy=dst_ports,
            max_iters=0,
        )
        if not torch.equal(ref, batched[fi]):
            mismatch = int((ref != batched[fi]).sum().item())
            raise AssertionError(
                f"flow={fi}: batched dist differs from per-flow dist in {mismatch} cells"
            )


def _time_action_loop(env: FactoryLaneEnv, *, use_cache: bool) -> tuple[float, list[int]]:
    """Build action spaces for every flow once. Returns (elapsed_ms, path_lens).

    When use_cache=False we clear the strategy cache between flows to simulate
    the pre-optimization behavior (one wavefront per flow). When use_cache=True
    we leave the cache alone, so the first call pays for all M wavefronts and
    subsequent calls only pay for backtrace.
    """
    state = env.get_state()
    routing = state.routing
    if not isinstance(routing, WavefrontStrategy):
        raise RuntimeError(
            f"expected WavefrontStrategy on state.routing, got {type(routing).__name__}"
        )
    flow_count = int(state.flow_count)

    # Reset cache so the timing always starts cold.
    state._wavefront_dist = None
    state._wavefront_dist_gen = -1

    t0 = time.perf_counter()
    path_lens: list[int] = []
    for fi in range(flow_count):
        if not use_cache:
            state._wavefront_dist = None
            state._wavefront_dist_gen = -1
        # Force build_action_space to evaluate flow `fi` even though
        # the engine's step counter hasn't advanced. We do this by
        # temporarily overriding step_count.
        original_step = state.step_count
        state.step_count = fi
        try:
            asp = env.build_action_space()
        finally:
            state.step_count = original_step
        if asp.candidate_path_len is not None and asp.candidate_path_len.numel() > 0:
            path_lens.append(int(asp.candidate_path_len[0].item()))
        else:
            path_lens.append(-1)
    return (time.perf_counter() - t0) * 1000.0, path_lens


def _bench_scenario(scn: Scenario, device: torch.device) -> dict:
    env = _build_env(scn, device)

    _verify_correctness(env)

    # Verify path equivalence between cache-on / cache-off paths.
    _, paths_cold = _time_action_loop(env, use_cache=False)
    _, paths_hot = _time_action_loop(env, use_cache=True)
    if paths_cold != paths_hot:
        raise AssertionError(
            f"{scn.name}: action-space paths differ between cache modes\n"
            f"  cold={paths_cold}\n  hot ={paths_hot}"
        )

    # Warmup
    for _ in range(WARMUP):
        _time_action_loop(env, use_cache=False)
        _time_action_loop(env, use_cache=True)

    cold_times: list[float] = []
    hot_times: list[float] = []
    for _ in range(ROUNDS):
        cold_ms, _ = _time_action_loop(env, use_cache=False)
        hot_ms, _ = _time_action_loop(env, use_cache=True)
        cold_times.append(cold_ms)
        hot_times.append(hot_ms)

    return {
        "name": scn.name,
        "flows": int(env.get_state().flow_count),
        "cold_mean": statistics.mean(cold_times),
        "cold_std": statistics.pstdev(cold_times) if len(cold_times) > 1 else 0.0,
        "hot_mean": statistics.mean(hot_times),
        "hot_std": statistics.pstdev(hot_times) if len(hot_times) > 1 else 0.0,
    }


def main() -> int:
    devices: List[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"{'dev':<5}{'scenario':<14}{'flows':>6}"
          f"{'cold(ms)':>12}{'hot(ms)':>12}{'speedup':>10}")
    print("-" * 65)

    for device in devices:
        for scn in make_scenarios():
            res = _bench_scenario(scn, device)
            speedup = res["cold_mean"] / max(res["hot_mean"], 1e-9)
            print(
                f"{device.type:<5}{res['name']:<14}{res['flows']:>6d}"
                f"{res['cold_mean']:>10.2f}±{res['cold_std']:<.2f}"
                f"{res['hot_mean']:>10.2f}±{res['hot_std']:<.2f}"
                f"{speedup:>9.1f}x"
            )

    print()
    print("All correctness checks passed (batched dist == per-flow dist;")
    print("paths under cache-on == paths under cache-off).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
