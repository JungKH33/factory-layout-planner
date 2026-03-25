from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import time
import torch

from envs.env_loader import load_env
from envs.visualizer import plot_layout, save_layout, browse_steps, StepFrame
from postprocess import RoutePlanner

from pipeline import DecisionPipeline
from search.beam import BeamConfig, BeamSearch
from search.mcts import MCTSConfig, MCTSSearch

from agents.registry import create as create_agent
from agents.ordering import DifficultyOrderingAgent
from envs.action_space import ActionSpace


# --- config (module-level constants, keep simple) ---
ENV_JSON: str = "envs/env_configs/basic_01.json"
#ENV_JSON: str = "preprocess/조립.json"
WRAPPER_MODE: str = "greedyv4"  # "greedy" | "greedyv2" | "greedyv3" | "greedyv4" | "alphachip" | "maskplace"
AGENT_MODE: str = "greedy"  # "greedy" | "alphachip" | "maskplace"
ALPHACHIP_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-26_00-50_b156aa\best.ckpt"
MASKPLACE_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-24_01-49_4e9e28\best.ckpt"

TOPK_K: int = 50
TOPK_SCAN_STEP: float = 5.0
TOPK_QUANT_STEP: float = 10.0
TOPK_CELL_SIZE: int = 50
ALPHACHIP_GRID: int = 128

SEARCH_MODE: str = "mcts"  # "none" | "mcts"
ORDERING_MODE: str = "none"  # "none" | "difficulty"
MCTS_SIMS: int = 1000
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

# Orientation expansion: adapter가 (center, orientation) 쌍을 후보로 생성
EXPAND_ORIENTATIONS: bool = False
MAX_ORIENTATIONS: int = 3

# Top-K tracking: search 중 최고 결과 K개 저장
TRACK_TOP_K: int = 5  # 0이면 비활성화
TRACK_VERBOSE: bool = True  # 리스트 변경 시 print

BACKEND_SELECTION: str = "benchmark"  # "static" | "benchmark"

SHOW_FLOW: bool = True
SHOW_SCORE: bool = True
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
        "greedy": {"k": TOPK_K, "scan_step": TOPK_SCAN_STEP, "quant_step": TOPK_QUANT_STEP, "random_seed": 5,
                   "expand_orientations": EXPAND_ORIENTATIONS, "max_orientations": MAX_ORIENTATIONS},
        "greedyv2": {"k": TOPK_K, "scan_step": TOPK_SCAN_STEP, "quant_step": TOPK_QUANT_STEP, "random_seed": 5,
                     "expand_orientations": EXPAND_ORIENTATIONS, "max_orientations": MAX_ORIENTATIONS},
        "greedyv3": {"k": TOPK_K, "quant_step": TOPK_QUANT_STEP, "oversample_factor": 2, "edge_ratio": 0.8, "random_seed": 5,
                     "expand_orientations": EXPAND_ORIENTATIONS, "max_orientations": MAX_ORIENTATIONS},
        "greedyv4": {"k": TOPK_K, "cell_size": TOPK_CELL_SIZE, "quant_step": TOPK_QUANT_STEP, "random_seed": 5,
                     "expand_orientations": EXPAND_ORIENTATIONS, "max_orientations": MAX_ORIENTATIONS},
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
                track_verbose=TRACK_VERBOSE,
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
                track_verbose=TRACK_VERBOSE,
            )
        )
    else:
        raise ValueError(f"Unknown SEARCH_MODE={SEARCH_MODE!r} (expected 'none'|'mcts'|'beam')")

    if ORDERING_MODE == "none":
        ordering_agent = None
    elif ORDERING_MODE == "difficulty":
        ordering_agent = DifficultyOrderingAgent()
    else:
        raise ValueError(f"Unknown ORDERING_MODE={ORDERING_MODE!r} (expected 'none'|'difficulty')")

    pipe = DecisionPipeline(agent=agent, adapter=adapter, search=search, ordering_agent=ordering_agent)
    pipe.bind(engine=engine)
    obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    terminated = truncated = False
    total_reward = 0.0

    start = time.perf_counter()
    step = 0
    frames: list[StepFrame] = []

    logger.info("inference")
    logger.info(
        "ENV_JSON=%s WRAPPER_MODE=%s AGENT_MODE=%s SEARCH_MODE=%s device=%s",
        ENV_JSON,
        WRAPPER_MODE,
        AGENT_MODE,
        SEARCH_MODE,
        device,
    )

    while not (terminated or truncated):
        step += 1
        next_gid = engine.get_state().remaining[0] if engine.get_state().remaining else None

        state = engine.get_state().copy()
        action = None
        dbg: dict[str, object] = {}
        try:
            action, dbg = pipe.decide()
            obs_env_next, reward, terminated, truncated, info = engine.step_action(action)
        except ValueError as e:
            if str(e) != "no_valid_actions":
                raise
            reward = float(engine.failure_penalty())
            terminated = False
            truncated = True
            info = {"reason": "no_valid_actions"}
            obs_env_next = None

        action_space_obj = dbg.get("action_space")
        scores_obj = dbg.get("scores")
        action_obj = dbg.get("action_index")
        value_obj = dbg.get("value")
        selected_action = None
        if action_obj is not None:
            try:
                selected_action = int(action_obj)
            except Exception:
                selected_action = None
        value = None
        if value_obj is not None:
            try:
                value = float(value_obj)
            except Exception:
                value = None

        frames.append(
            StepFrame(
                state=state,
                cost=float(engine.cost()),
                step_idx=int(step),
                action_space=action_space_obj if isinstance(action_space_obj, ActionSpace) else None,
                scores=scores_obj if hasattr(scores_obj, "shape") else None,
                selected_action=selected_action,
                value=value,
            )
        )

        obs_env = obs_env_next
        total_reward += float(reward)
        if action is None:
            logger.warning(
                "step %s next_gid=%s search=%s reason=no_valid_actions",
                step,
                next_gid,
                SEARCH_MODE,
            )
        else:
            logger.info(
                "step %s next_gid=%s search=%s action=(%.1f,%.1f)",
                step,
                next_gid,
                dbg.get("search", SEARCH_MODE),
                float(action.x_c),
                float(action.y_c),
            )

        if terminated or truncated:
            reason = info.get("reason", None)
            logger.info(
                "end: terminated=%s truncated=%s step=%s placed=%s cost=%.3f reason=%s",
                terminated,
                truncated,
                step,
                len(engine.get_state().placed),
                engine.total_cost(),
                reason,
            )

    end = time.perf_counter()
    logger.info("Total computation time: %.4f seconds", end - start)
    logger.info(
        "episode_reward=%.3f terminated=%s truncated=%s",
        total_reward,
        terminated,
        truncated,
    )

    # Print top-K results if tracking was enabled
    if search is not None and hasattr(search, "top_tracker") and search.top_tracker is not None:
        top_results = search.top_tracker.get_results()
        if top_results:
            logger.info("Top-%s Search Results", len(top_results))
            for i, result in enumerate(top_results):
                logger.info(
                    "#%s: cost=%.2f, placed=%s, cum_reward=%.3f",
                    i + 1,
                    result.cost,
                    len(result.positions),
                    result.cum_reward,
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
    engine.save_placement(str(placement_path))
    logger.info("saved_placement=%s", placement_path)

    # Save top-K results if tracking was enabled
    if search is not None and hasattr(search, "top_tracker") and search.top_tracker is not None:
        top_results = search.top_tracker.get_results()
        for i, result in enumerate(top_results):
            engine.set_state(result.engine_state)
            top_path = out_dir / f"{ts}_top{i+1}_cost{result.cost:.1f}.png"
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
