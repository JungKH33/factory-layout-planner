from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import torch

from group_placement.agents.ordering import DifficultyOrderingAgent
from group_placement.agents.registry import create as create_agent
from group_placement.envs.export import export_group_placement
from group_placement.envs.env_loader import load_env
from group_placement.search import (
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


@dataclass(frozen=True)
class GroupPlacementConfig:
    env_json: str
    device: Optional[str] = None
    backend_selection: str = "benchmark"
    wrapper_mode: str = "greedyv3"
    agent_mode: str = "greedy"
    search_mode: str = "mcts"  # none|mcts|hierarchical_mcts|best_first|h_best_first|beam|hierarchical_beam
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
    if mode == "none":
        return None
    if mode == "mcts":
        return MCTSSearch(
            config=MCTSConfig(
                num_simulations=int(cfg.mcts_sims),
                rollout_enabled=bool(cfg.rollout_enabled),
                rollout_depth=int(cfg.rollout_depth),
            )
        )
    if mode == "hierarchical_mcts":
        return HierarchicalMCTSSearch(
            config=HierarchicalMCTSConfig(
                num_simulations=int(cfg.mcts_sims),
                rollout_enabled=bool(cfg.rollout_enabled),
                rollout_depth=int(cfg.rollout_depth),
            )
        )
    if mode in {"best_first", "best"}:
        return BestFirstSearch(
            config=BestFirstConfig(
                max_expansions=int(cfg.max_expansions),
                depth=int(cfg.search_depth),
                expansion_topk=int(cfg.expansion_topk),
            )
        )
    if mode in {"h_best_first", "h_best"}:
        return HierarchicalBestFirstSearch(
            config=HierarchicalBestFirstConfig(
                max_expansions=int(cfg.max_expansions),
                depth=int(cfg.search_depth),
                manager_topk=int(cfg.expansion_topk),
            )
        )
    if mode == "beam":
        return BeamSearch(
            config=BeamConfig(
                beam_width=int(cfg.beam_width),
                depth=int(cfg.search_depth),
                expansion_topk=int(cfg.expansion_topk),
            )
        )
    if mode == "hierarchical_beam":
        return HierarchicalBeamSearch(
            config=HierarchicalBeamConfig(
                beam_width=int(cfg.beam_width),
                depth=int(cfg.search_depth),
                manager_topk=int(cfg.expansion_topk),
            )
        )
    raise ValueError(f"unsupported search_mode={cfg.search_mode!r}")


def run_group_placement(cfg: GroupPlacementConfig) -> GroupPlacementArtifact:
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
            "total_cost": float(loaded.env.total_cost()),
            "episode_reward": float(total_reward),
            "elapsed_sec": float(time.perf_counter() - start),
        },
    )
    return artifact


def run_and_save_group_placement(cfg: GroupPlacementConfig, output_json: str) -> GroupPlacementArtifact:
    artifact = run_group_placement(cfg)
    save_json(artifact, output_json)
    return artifact


__all__ = [
    "GroupPlacementConfig",
    "run_group_placement",
    "run_and_save_group_placement",
]
