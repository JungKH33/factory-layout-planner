from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import time
import torch

from group_placement.envs.env_loader import load_env
from group_placement.envs.interchange import export_group_placement, save_group_placement
from group_placement.envs.visualizer import plot_layout, save_layout, browse_steps, StepFrame
from facility_placement import save_facility_layout

from group_placement.trace.explorer import Explorer
from group_placement.search.astar import AStarConfig, AStarSearch
from group_placement.search.beam import BeamConfig, BeamSearch
from group_placement.search.best import BestFirstConfig, BestFirstSearch
from group_placement.search.mcts import MCTSConfig, MCTSSearch
from group_placement.search.hierarchical_beam import HierarchicalBeamConfig, HierarchicalBeamSearch
from group_placement.search.hierarchical_best import HierarchicalBestFirstConfig, HierarchicalBestFirstSearch
from group_placement.search.hierarchical_mcts import HierarchicalMCTSConfig, HierarchicalMCTSSearch

from group_placement.agents.registry import create as create_agent
from group_placement.agents.ordering import DifficultyOrderingAgent
from group_placement.envs.action_space import ActionSpace


# --- config (module-level constants, keep simple) ---
ENV_JSON: str = "group_placement/envs/env_configs/facility_placement_demo.json"
#ENV_JSON: str = "preprocess/조립.json"
WRAPPER_MODE: str = "greedyv3"  # "greedy" | "greedyv2" | "greedyv3" | "greedyv4" | "greedyv5" | "alphachip" | "maskplace"
AGENT_MODE: str = "greedy"  # "greedy" | "alphachip" | "maskplace"
ALPHACHIP_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-26_00-50_b156aa\best.ckpt"
MASKPLACE_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-24_01-49_4e9e28\best.ckpt"

TOPK_K: int = 100
TOPK_SCAN_STEP: float = 5.0
TOPK_QUANT_STEP: float = 10.0
TOPK_CELL_SIZE: int = 50
TOPK_PER_CELL: int = 20
ALPHACHIP_GRID: int = 128

SEARCH_MODE: str = "mcts"  # "none" | "mcts" | "astar" | "hierarchical_mcts" | "h_best_first" | "hierarchical_beam" | "best_first" | "beam"
ORDERING_MODE: str = "none"  # "none" | "difficulty"
MCTS_SIMS: int = 1000
BEST_MAX_EXPANSIONS: int = 20
HBEST_MAX_EXPANSIONS: int = 20
MCTS_ROLLOUT_ENABLED: bool = True
ROLLOUT_DEPTH: int = 10

MCTS_TEMPERATURE: float = 0.0
# Progressive widening (useful for large action spaces, e.g. maskplace)
MCTS_PW_ENABLED: bool = False
MCTS_PW_C: float = 1.5
MCTS_PW_ALPHA: float = 0.5
MCTS_PW_MIN_CHILDREN: int = 1
MCTS_CACHE_DECISION_STATE: bool = True
BEAM_WIDTH: int = 8
BEAM_DEPTH: int = 5
BEAM_EXPANSION_TOPK: int = 16
BEAM_CACHE_DECISION_STATE: bool = False
HBEAM_WORKER_TOPK: int = 4
BEST_USE_VALUE_HEURISTIC: bool = True
HBEST_USE_VALUE_HEURISTIC: bool = True
ASTAR_USE_VALUE_HEURISTIC: bool = False

# Top-K tracking: search 중 최고 결과 K개 저장
TRACK_TOP_K: int = 5  # 0이면 비활성화

BACKEND_SELECTION: str = "benchmark"  # "static" | "benchmark"

SHOW_FLOW: bool = True
SHOW_SCORE: bool = False
SHOW_MASKS: bool = True

logger = logging.getLogger(__name__)


@torch.no_grad()
def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device, backend_selection=BACKEND_SELECTION)
    engine = loaded.env
    engine.log = True

    adapter_kwargs: dict = {
        "greedy": {"k": TOPK_K, "scan_step": TOPK_SCAN_STEP, "quant_step": TOPK_QUANT_STEP, "random_seed": 5},
        "greedyv2": {"k": TOPK_K, "scan_step": TOPK_SCAN_STEP, "quant_step": TOPK_QUANT_STEP, "random_seed": 5},
        "greedyv3": {"k": TOPK_K, "quant_step": TOPK_QUANT_STEP, "oversample_factor": 2, "edge_ratio": 0.8, "random_seed": 5},
        "greedyv4": {"k": TOPK_K, "cell_size": TOPK_CELL_SIZE, "top_per_cell": TOPK_PER_CELL, "quant_step": TOPK_QUANT_STEP, "random_seed": 5},
        "greedyv5": {"cell_size": TOPK_CELL_SIZE, "top_per_cell": TOPK_PER_CELL, "quant_step": TOPK_QUANT_STEP, "random_seed": 5},
        "alphachip": {"coarse_grid": int(ALPHACHIP_GRID)},
        "maskplace": {"grid": 224, "soft_coefficient": 1.0},
    }
    agent_kwargs: dict = {
        "greedy": {"prior_temperature": 1.0},
        "alphachip": {"coarse_grid": int(ALPHACHIP_GRID), "checkpoint_path": str(ALPHACHIP_CHECKPOINT_PATH), "device": device},
        "maskplace": {"device": device, "grid": 224, "soft_coefficient": 1.0,
                      "checkpoint_path": str(MASKPLACE_CHECKPOINT_PATH) if MASKPLACE_CHECKPOINT_PATH else None},
    }
    agent, adapter = create_agent(
        method=WRAPPER_MODE,
        agent=AGENT_MODE,
        agent_kwargs=agent_kwargs.get(AGENT_MODE, {}),
        adapter_kwargs=adapter_kwargs.get(WRAPPER_MODE, {}),
    )

    if SEARCH_MODE == "none":
        search = None
    elif SEARCH_MODE == "mcts":
        search = MCTSSearch(
            config=MCTSConfig(
                num_simulations=MCTS_SIMS,
                rollout_enabled=bool(MCTS_ROLLOUT_ENABLED),
                rollout_depth=int(ROLLOUT_DEPTH),
                cache_decision_state=bool(MCTS_CACHE_DECISION_STATE),
                track_top_k=TRACK_TOP_K,
            )
        )
    elif SEARCH_MODE == "hierarchical_mcts":
        search = HierarchicalMCTSSearch(
            config=HierarchicalMCTSConfig(
                num_simulations=MCTS_SIMS,
                rollout_enabled=bool(MCTS_ROLLOUT_ENABLED),
                rollout_depth=int(ROLLOUT_DEPTH),
                track_top_k=TRACK_TOP_K,
                pw_enabled=bool(MCTS_PW_ENABLED),
                pw_c=MCTS_PW_C,
                pw_alpha=MCTS_PW_ALPHA,
            )
        )
    elif SEARCH_MODE in {"h_best_first", "h_best"}:
        search = HierarchicalBestFirstSearch(
            config=HierarchicalBestFirstConfig(
                max_expansions=HBEST_MAX_EXPANSIONS,
                depth=BEAM_DEPTH,
                manager_topk=BEAM_EXPANSION_TOPK,
                worker_topk=HBEAM_WORKER_TOPK,
                cache_decision_state=bool(BEAM_CACHE_DECISION_STATE),
                use_value_heuristic=bool(HBEST_USE_VALUE_HEURISTIC),
                track_top_k=TRACK_TOP_K,
            )
        )
    elif SEARCH_MODE == "hierarchical_beam":
        search = HierarchicalBeamSearch(
            config=HierarchicalBeamConfig(
                beam_width=BEAM_WIDTH,
                depth=BEAM_DEPTH,
                manager_topk=BEAM_EXPANSION_TOPK,
                worker_topk=HBEAM_WORKER_TOPK,
                cache_decision_state=bool(BEAM_CACHE_DECISION_STATE),
                track_top_k=TRACK_TOP_K,
            )
        )
    elif SEARCH_MODE in {"best_first", "best"}:
        search = BestFirstSearch(
            config=BestFirstConfig(
                max_expansions=BEST_MAX_EXPANSIONS,
                depth=BEAM_DEPTH,
                expansion_topk=BEAM_EXPANSION_TOPK,
                cache_decision_state=bool(BEAM_CACHE_DECISION_STATE),
                use_value_heuristic=bool(BEST_USE_VALUE_HEURISTIC),
                track_top_k=TRACK_TOP_K,
            )
        )
    elif SEARCH_MODE in {"astar", "a_star"}:
        search = AStarSearch(
            config=AStarConfig(
                max_expansions=BEST_MAX_EXPANSIONS,
                depth=BEAM_DEPTH,
                expansion_topk=BEAM_EXPANSION_TOPK,
                cache_decision_state=bool(BEAM_CACHE_DECISION_STATE),
                use_value_heuristic=bool(ASTAR_USE_VALUE_HEURISTIC),
                track_top_k=TRACK_TOP_K,
            )
        )
    elif SEARCH_MODE == "beam":
        search = BeamSearch(
            config=BeamConfig(
                beam_width=BEAM_WIDTH,
                depth=BEAM_DEPTH,
                expansion_topk=BEAM_EXPANSION_TOPK,
                cache_decision_state=bool(BEAM_CACHE_DECISION_STATE),
                track_top_k=TRACK_TOP_K,
            )
        )
    else:
        raise ValueError(
            f"Unknown SEARCH_MODE={SEARCH_MODE!r} "
            "(expected 'none'|'mcts'|'astar'|'hierarchical_mcts'|'h_best_first'|'hierarchical_beam'|'best_first'|'beam')"
        )

    if ORDERING_MODE == "none":
        ordering_agent = None
    elif ORDERING_MODE == "difficulty":
        ordering_agent = DifficultyOrderingAgent()
    else:
        raise ValueError(f"Unknown ORDERING_MODE={ORDERING_MODE!r} (expected 'none'|'difficulty')")

    # Use Explorer instead of DecisionPipeline
    exp = Explorer(engine, adapter, agent, search=search, ordering_agent=ordering_agent)
    exp.reset(options=loaded.reset_kwargs)

    # Determine auto_play source
    if search is not None:
        source = f"search:{type(search).__name__}"
    else:
        source = "agent"

    start = time.perf_counter()
    frames: list[StepFrame] = []
    all_top_k: list[dict] = []

    logger.info("inference")
    logger.info(
        "ENV_JSON=%s WRAPPER_MODE=%s AGENT_MODE=%s SEARCH_MODE=%s device=%s",
        ENV_JSON,
        WRAPPER_MODE,
        AGENT_MODE,
        SEARCH_MODE,
        device,
    )

    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        node = exp.current()
        if node.terminal:
            break

        step += 1
        next_gid = node.group_id

        # Capture state BEFORE action for frame
        state_before = engine.get_state().copy()

        # Get agent signal (always, for visualization scores/value)
        agent_sig = exp.predict_agent()

        # Get search signal and decide source
        result_placement = None
        try:
            if search is not None:
                search_sig = exp.predict_search()
                child = exp.step_with(search_sig.source)
                # Collect top-K from this search step
                step_top_k = search_sig.metadata.get("top_k")
                if step_top_k:
                    all_top_k.extend(step_top_k)
            else:
                child = exp.step_with("agent")
        except (ValueError, KeyError):
            # No valid actions or signal missing
            reward = float(engine.failure_penalty())
            terminated = False
            truncated = True
            logger.warning(
                "step %s next_gid=%s search=%s reason=no_valid_actions",
                step, next_gid, SEARCH_MODE,
            )
            frames.append(
                StepFrame(
                    state=state_before,
                    cost=float(engine.cost()),
                    step_idx=int(step),
                    action_space=node._snapshot.action_space if node._snapshot else None,
                    scores=agent_sig.scores if agent_sig.scores is not None else None,
                    selected_action=None,
                    value=agent_sig.metadata.get("value_estimate"),
                )
            )
            total_reward += reward
            break

        reward = child.cum_reward - node.cum_reward
        total_reward += reward
        terminated = child.terminal and len(engine.get_state().remaining) == 0
        truncated = child.terminal and not terminated

        # Build StepFrame for visualization
        frames.append(
            StepFrame(
                state=state_before,
                cost=child.cost_after or float(engine.cost()),
                step_idx=int(step),
                action_space=node._snapshot.action_space if node._snapshot else None,
                scores=agent_sig.scores if agent_sig.scores is not None else None,
                selected_action=node.chosen_action,
                value=agent_sig.metadata.get("value_estimate"),
            )
        )

        logger.info(
            "step %s next_gid=%s search=%s action=%s cost=%.1f",
            step, next_gid, SEARCH_MODE, node.chosen_action,
            child.cost_after or 0.0,
        )

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            logger.info(
                "end: terminated=%s truncated=%s step=%s placed=%s cost=%.3f",
                terminated, truncated, step,
                len(engine.get_state().placed),
                engine.total_cost(),
            )

    end = time.perf_counter()
    logger.info("Total computation time: %.4f seconds", end - start)
    logger.info(
        "episode_reward=%.3f terminated=%s truncated=%s",
        total_reward, terminated, truncated,
    )

    # Sort and deduplicate top-K results across all search steps
    if all_top_k:
        seen_cost_keys: set[int] = set()
        unique_top_k = []
        for entry in sorted(all_top_k, key=lambda r: r["cost"]):
            cost_key = int(round(float(entry["cost"]) * 1000.0))
            if cost_key not in seen_cost_keys:
                seen_cost_keys.add(cost_key)
                unique_top_k.append(entry)
        all_top_k = unique_top_k[:TRACK_TOP_K]
        logger.info("Top-%s Search Results (aggregated across steps)", len(all_top_k))
        for i, result in enumerate(all_top_k):
            logger.info(
                "#%s: cost=%.2f, placed=%s, cum_reward=%.3f",
                i + 1, result["cost"], len(result["positions"]), result["cum_reward"],
            )

    out_dir = Path("results") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{AGENT_MODE}_{WRAPPER_MODE}_{SEARCH_MODE}.png"

    # Interactive slide viewer (←/→): policy scatter colored by agent.policy scores.
    if frames:
        # IMPORTANT: browse_steps restores state internally. Save & restore the final state
        # so that the final plot/save below always reflects the true end layout.
        final_state = engine.get_state().copy()
        browse_steps(adapter, frames=frames, title="Inference browser (←/→ to navigate, q to quit)")
        engine.set_state(final_state)  # type: ignore[arg-type]

    # Route planning
    # planner = RoutePlanner(engine, algorithm="astar")
    # routes = planner.plan_all()

    # Preview before saving (interactive; close the window to continue).
    plot_layout(adapter, action_space=None, routes= None, backend="matplotlib")

    save_layout(
        adapter,
        show_masks=SHOW_MASKS,
        show_flow=SHOW_FLOW,
        show_score=SHOW_SCORE,
        show_zones=False,
        action_space=None,
        save_path=str(out_path),
    )
    logger.info("saved_layout=%s", out_path)

    # Save placement JSON
    placement_path = out_dir / f"{ts}_{AGENT_MODE}_{WRAPPER_MODE}_{SEARCH_MODE}.json"
    save_group_placement(loaded, str(placement_path))
    logger.info("saved_placement=%s", placement_path)

    # --- Phase 2: facility-level placement overlay ---
    # Phase 1 and Phase 2 are decoupled: export to a plain dict, then the
    # postprocess helper resolves + renders + saves in one call.
    state_dict = export_group_placement(loaded)
    facility_path = out_dir / f"{ts}_{AGENT_MODE}_{WRAPPER_MODE}_{SEARCH_MODE}_facility.png"
    if save_facility_layout(state_dict, save_path=str(facility_path)):
        logger.info("saved_facility_layout=%s", facility_path)

    # Save top-K result layouts
    if all_top_k:
        for i, result in enumerate(all_top_k):
            engine.set_state(result["engine_state"])
            top_path = out_dir / f"{ts}_top{i+1}_cost{result['cost']:.1f}.png"
            save_layout(
                adapter,
                show_masks=SHOW_MASKS,
                show_flow=SHOW_FLOW,
                show_score=SHOW_SCORE,
                show_zones=False,
                action_space=None,
                save_path=str(top_path),
            )
            logger.info("saved_top_%s=%s", i + 1, top_path)


if __name__ == "__main__":
    main()
