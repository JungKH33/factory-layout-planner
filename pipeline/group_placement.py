from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Optional

import torch

from group_placement.agents.ordering import DifficultyOrderingAgent
from group_placement.agents.registry import create as create_agent
from group_placement.envs.interchange import export_group_placement, save_group_placement
from group_placement.envs.env_loader import load_env
from group_placement.envs.visualizer import save_layout
from group_placement.search import (
    AStarConfig,
    AStarSearch,
    BeamConfig,
    BeamSearch,
    BestFirstConfig,
    BestFirstSearch,
    HierarchicalBeamConfig,
    HierarchicalBeamSearch,
    HierarchicalBestFirstConfig,
    HierarchicalBestFirstSearch,
    HierarchicalMCTSConfig,
    HierarchicalMCTSSearch,
    MCTSConfig,
    MCTSSearch,
)
from group_placement.trace.explorer import Explorer
from pipeline.schema import GroupPlacementArtifact, save_json, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupPlacementConfig:
    env_json: str
    output_dir: str
    top_n: int = 1
    device: Optional[str] = None
    backend_selection: str = "benchmark"
    wrapper_mode: str = "greedyv3"
    agent_mode: str = "greedy"
    search_mode: str = "mcts"  # none|mcts|astar|hierarchical_mcts|best_first|h_best_first|beam|hierarchical_beam
    ordering_mode: str = "none"  # none|difficulty
    max_decisions: int = 0
    mcts_sims: int = 1000
    rollout_enabled: bool = True
    rollout_depth: int = 10
    beam_width: int = 8
    search_depth: int = 5
    expansion_topk: int = 16
    max_expansions: int = 200


def _select_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_search(cfg: GroupPlacementConfig):
    mode = str(cfg.search_mode).strip().lower()
    track_top_k = max(0, int(cfg.top_n))
    if mode == "none":
        return None
    if mode == "mcts":
        return MCTSSearch(
            config=MCTSConfig(
                num_simulations=int(cfg.mcts_sims),
                rollout_enabled=bool(cfg.rollout_enabled),
                rollout_depth=int(cfg.rollout_depth),
                track_top_k=track_top_k,
            )
        )
    if mode == "hierarchical_mcts":
        return HierarchicalMCTSSearch(
            config=HierarchicalMCTSConfig(
                num_simulations=int(cfg.mcts_sims),
                rollout_enabled=bool(cfg.rollout_enabled),
                rollout_depth=int(cfg.rollout_depth),
                track_top_k=track_top_k,
            )
        )
    if mode in {"best_first", "best"}:
        return BestFirstSearch(
            config=BestFirstConfig(
                max_expansions=int(cfg.max_expansions),
                depth=int(cfg.search_depth),
                expansion_topk=int(cfg.expansion_topk),
                track_top_k=track_top_k,
            )
        )
    if mode in {"astar", "a_star"}:
        return AStarSearch(
            config=AStarConfig(
                max_expansions=int(cfg.max_expansions),
                depth=int(cfg.search_depth),
                expansion_topk=int(cfg.expansion_topk),
                track_top_k=track_top_k,
            )
        )
    if mode in {"h_best_first", "h_best"}:
        return HierarchicalBestFirstSearch(
            config=HierarchicalBestFirstConfig(
                max_expansions=int(cfg.max_expansions),
                depth=int(cfg.search_depth),
                manager_topk=int(cfg.expansion_topk),
                track_top_k=track_top_k,
            )
        )
    if mode == "beam":
        return BeamSearch(
            config=BeamConfig(
                beam_width=int(cfg.beam_width),
                depth=int(cfg.search_depth),
                expansion_topk=int(cfg.expansion_topk),
                track_top_k=track_top_k,
            )
        )
    if mode == "hierarchical_beam":
        return HierarchicalBeamSearch(
            config=HierarchicalBeamConfig(
                beam_width=int(cfg.beam_width),
                depth=int(cfg.search_depth),
                manager_topk=int(cfg.expansion_topk),
                track_top_k=track_top_k,
            )
        )
    raise ValueError(f"unsupported search_mode={cfg.search_mode!r}")


def _collect_top_k(all_top_k: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    if not all_top_k or top_n <= 0:
        return []
    seen_cost_keys: set[int] = set()
    unique_top_k: list[dict[str, Any]] = []
    for entry in sorted(all_top_k, key=lambda r: float(r["cost"])):
        cost_key = int(round(float(entry["cost"]) * 1000.0))
        if cost_key in seen_cost_keys:
            continue
        seen_cost_keys.add(cost_key)
        unique_top_k.append(entry)
    return unique_top_k[:top_n]


def _execute_group_placement(cfg: GroupPlacementConfig):
    start = time.perf_counter()
    device = _select_device(cfg.device)
    loaded = load_env(cfg.env_json, device=device, backend_selection=cfg.backend_selection)

    agent, adapter = create_agent(method=cfg.wrapper_mode, agent=cfg.agent_mode)
    search = _build_search(cfg)
    ordering_agent = DifficultyOrderingAgent() if cfg.ordering_mode == "difficulty" else None

    exp = Explorer(loaded.env, adapter, agent, search=search, ordering_agent=ordering_agent)
    exp.reset(options=loaded.reset_kwargs)

    terminated = False
    truncated = False
    decisions = 0
    total_reward = 0.0
    all_top_k: list[dict[str, Any]] = []

    while not (terminated or truncated):
        node = exp.current()
        if node.terminal:
            break
        if int(cfg.max_decisions) > 0 and decisions >= int(cfg.max_decisions):
            truncated = True
            break
        try:
            if search is not None:
                sig = exp.predict_search()
                step_top_k = sig.metadata.get("top_k")
                if step_top_k:
                    all_top_k.extend(step_top_k)
                child = exp.step_with(sig.source)
            else:
                child = exp.step_with("agent")
        except (ValueError, KeyError):
            truncated = True
            break
        decisions += 1
        total_reward += float(child.cum_reward - node.cum_reward)
        terminated = child.terminal and len(loaded.env.get_state().remaining) == 0
        truncated = child.terminal and not terminated

    state = loaded.env.get_state()
    payload = export_group_placement(loaded)

    artifact = GroupPlacementArtifact(
        stage="group_placement",
        created_at=utc_now_iso(),
        env_json=str(cfg.env_json),
        group_placement=payload,
        metrics={
            "device": str(device),
            "search_mode": str(cfg.search_mode),
            "wrapper_mode": str(cfg.wrapper_mode),
            "agent_mode": str(cfg.agent_mode),
            "decisions": int(decisions),
            "placed_count": int(len(state.placed)),
            "remaining_count": int(len(state.remaining)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_cost": float(loaded.env.cost()),
            "episode_reward": float(total_reward),
            "elapsed_sec": float(time.perf_counter() - start),
        },
    )
    top_k_results = _collect_top_k(all_top_k, max(0, int(cfg.top_n)))
    return artifact, loaded, adapter, top_k_results


def run_group_placement(cfg: GroupPlacementConfig) -> GroupPlacementArtifact:
    artifact, _loaded, _adapter, _top_k_results = _execute_group_placement(cfg)
    return artifact


def _artifact_path(output_dir: str) -> Path:
    return Path(output_dir) / "group_placement.json"


def run_and_save_group_placement(cfg: GroupPlacementConfig) -> GroupPlacementArtifact:
    artifact, loaded, adapter, top_k_results = _execute_group_placement(cfg)
    out_path = _artifact_path(cfg.output_dir)
    save_json(artifact, out_path)

    top_n = max(1, int(cfg.top_n))
    final_state = loaded.env.get_state().copy()
    candidates = top_k_results if top_k_results else [{"engine_state": final_state, "cost": loaded.env.cost()}]
    for idx, result in enumerate(candidates[:top_n], start=1):
        loaded.env.set_state(result["engine_state"])
        top_json_path = Path(cfg.output_dir) / f"top_{idx:03d}.json"
        top_png_path = Path(cfg.output_dir) / f"top_{idx:03d}.png"
        save_group_placement(loaded, str(top_json_path))
        save_layout(
            adapter,
            save_path=str(top_png_path),
            show_masks=True,
            show_flow=True,
            show_score=False,
            show_zones=False,
            action_space=None,
        )
        logger.info("group_placement top_%03d json saved: %s", idx, top_json_path)
        logger.info("group_placement top_%03d png saved: %s", idx, top_png_path)
    loaded.env.set_state(final_state)

    logger.info("group_placement output_dir: %s", cfg.output_dir)
    logger.info("group_placement artifact saved: %s", out_path)
    logger.info("group_placement metrics: %s", artifact.metrics)
    return artifact


__all__ = [
    "GroupPlacementConfig",
    "run_group_placement",
    "run_and_save_group_placement",
]
