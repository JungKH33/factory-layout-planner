"""Lane-generation inference: route flows on a placed layout.

배치 완료된 group placement 결과를 기반으로 설비 간 물류 동선을 생성합니다.

사용법:
    python -m lane_generation.inference
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import torch

from lane_generation.envs import (
    FactoryLaneEnv,
    LaneAction,
    LaneAdapter,
    LaneAdapterConfig,
    LoadedLaneEnv,
    RoutingConfig,
    load_lane_env,
)
from lane_generation.envs.interchange import export_lane_generation, print_summary, save_lane_generation
from lane_generation.agents import GreedyLaneAgent

# --- config (module-level constants) ---
ENV_JSON: str = "group_placement/envs/env_configs/clearance_03.json"
GROUP_PLACEMENT_JSON: str = "results/inference/sample_placement.json"

CANDIDATE_K: int = 8
ROUTING_ALGORITHM: str = "wavefront"  # "wavefront" | "dijkstra" | "astar"
ROUTING_SELECTION: str = "benchmark"  # "static" | "benchmark"
FLOW_ORDERING: str = "weight_desc"  # "weight_desc" | "given"

REWARD_SCALE: float = 100.0
PENALTY_WEIGHT: float = 50000.0

logger = logging.getLogger(__name__)


@torch.no_grad()
def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    device = torch.device("cpu")

    logger.info(
        "ENV_JSON=%s GROUP_PLACEMENT=%s ROUTING=%s/%s device=%s",
        ENV_JSON, GROUP_PLACEMENT_JSON, ROUTING_SELECTION, ROUTING_ALGORITHM, device,
    )

    loaded = load_lane_env(
        env_json=ENV_JSON,
        group_placement=GROUP_PLACEMENT_JSON,
        device=device,
        flow_ordering=FLOW_ORDERING,
        adapter_config=LaneAdapterConfig(candidate_k=CANDIDATE_K),
        routing_config=RoutingConfig(
            selection=ROUTING_SELECTION,
            algorithm=ROUTING_ALGORITHM,
        ),
        reward_scale=REWARD_SCALE,
        penalty_weight=PENALTY_WEIGHT,
    )
    engine = loaded.env
    engine.reset()

    agent = GreedyLaneAgent(prior_temperature=1.0)

    start = time.perf_counter()
    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False
    collected_routes = []

    while not (terminated or truncated):
        state = engine.get_state()
        if state.done:
            terminated = True
            break

        step += 1
        flow_idx = state.current_flow_index()
        if flow_idx is None:
            terminated = True
            break

        src_gid, dst_gid, weight = state.flow_pair(flow_idx)

        action_space = engine.build_action_space()
        valid_n = int(action_space.valid_mask.sum().item())

        if valid_n == 0:
            logger.warning(
                "step %s flow=%s (%s->%s) no valid candidates, skipping",
                step, flow_idx, src_gid, dst_gid,
            )
            obs, reward, terminated, truncated, info = engine.step_action(
                LaneAction(candidate_index=0, flow_index=flow_idx),
                action_space=action_space,
            )
            total_reward += reward
            continue

        action_idx = agent.select_action(obs={}, action_space=action_space)

        flow_idx_resolved, route, _space = engine.resolve_action(
            LaneAction(candidate_index=action_idx, flow_index=flow_idx),
            action_space=action_space,
        )
        if route is None:
            logger.warning(
                "step %s flow=%s (%s->%s) resolve failed for action=%s",
                step, flow_idx, src_gid, dst_gid, action_idx,
            )
            obs, reward, terminated, truncated, info = engine.step_action(
                LaneAction(candidate_index=action_idx, flow_index=flow_idx),
                action_space=action_space,
            )
            total_reward += reward
            continue

        obs, reward, terminated, truncated, info = engine.step_route(route)
        total_reward += reward
        collected_routes.append(route)

        cost_val = action_space.candidate_cost
        cost_str = f"{float(cost_val[action_idx].item()):.1f}" if cost_val is not None else "?"
        logger.info(
            "step %s flow=%s (%s->%s w=%.1f) action=%s/%s cost=%s edges=%s",
            step, flow_idx, src_gid, dst_gid, weight,
            action_idx, valid_n, cost_str, int(route.edge_indices.numel()),
        )

    elapsed = time.perf_counter() - start
    logger.info(
        "done: steps=%s routes=%s reward=%.3f elapsed=%.4fs",
        step, len(collected_routes), total_reward, elapsed,
    )

    print_summary(engine, collected_routes)

    out_dir = Path("results") / "lane_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{ts}_{ROUTING_ALGORITHM}.json"
    save_lane_generation(engine, collected_routes, str(json_path))


if __name__ == "__main__":
    main()
