from __future__ import annotations

from datetime import datetime
from pathlib import Path

import time
import torch

from envs.json_loader import load_env
from envs.visualizer import plot_layout, save_layout, browse_steps, StepFrame
from postprocess import RoutePlanner

from pipeline import DecisionPipeline
from search.beam import BeamConfig, BeamSearch
from search.mcts import MCTSConfig, MCTSSearch

from agents.greedy import GreedyAgent
from agents.alphachip.agent import AlphaChipAgent
from agents.maskplace import MaskPlaceAgent

from envs.wrappers.greedy import GreedyWrapperEnv
from envs.wrappers.greedyv2 import GreedyWrapperV2Env
from envs.wrappers.greedyv3 import GreedyWrapperV3Env

from envs.wrappers.alphachip import AlphaChipWrapperEnv
from envs.wrappers.maskplace import MaskPlaceWrapperEnv
from envs.wrappers.candidate_set import CandidateSet


# --- config (module-level constants, keep simple) ---
ENV_JSON: str = "env_configs/basic_01.json"
#ENV_JSON: str = "preprocess/조립.json"
WRAPPER_MODE: str = "greedyv3"  # "greedy" | "alphachip" | "maskplace"
AGENT_MODE: str = "greedy"  # "greedy" | "alphachip" | "maskplace"
ALPHACHIP_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-26_00-50_b156aa\best.ckpt"
MASKPLACE_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-24_01-49_4e9e28\best.ckpt"

TOPK_K: int = 50
TOPK_SCAN_STEP: float = 5.0
TOPK_QUANT_STEP: float = 10.0
ALPHACHIP_GRID: int = 128

SEARCH_MODE: str = "mcts"  # "none" | "mcts"
MCTS_SIMS: int = 1000
MCTS_ROLLOUT_ENABLED: bool = True
ROLLOUT_DEPTH: int = 10

MCTS_TEMPERATURE: float = 0.0
# Progressive widening (useful for large action spaces, e.g. maskplace)
MCTS_PW_ENABLED: bool = False
MCTS_PW_C: float = 1.5
MCTS_PW_ALPHA: float = 0.5
MCTS_PW_MIN_CHILDREN: int = 1
BEAM_WIDTH: int = 8
BEAM_DEPTH: int = 5
BEAM_EXPANSION_TOPK: int = 16

# Top-K tracking: search 중 최고 결과 K개 저장
TRACK_TOP_K: int = 5  # 0이면 비활성화
TRACK_VERBOSE: bool = True  # 리스트 변경 시 print

SHOW_FLOW: bool = True
SHOW_SCORE: bool = True
SHOW_MASKS: bool = True


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = True

    if WRAPPER_MODE == "greedy":
        env = GreedyWrapperEnv(
            engine=engine,
            k=TOPK_K,
            scan_step=TOPK_SCAN_STEP,
            quant_step=TOPK_QUANT_STEP,
            random_seed=5,
        )
    elif WRAPPER_MODE == "greedyv2":
        env = GreedyWrapperV2Env(
            engine=engine,
            k=TOPK_K,
            scan_step=TOPK_SCAN_STEP,
            quant_step=TOPK_QUANT_STEP,
            random_seed=5,
        )

    elif WRAPPER_MODE == "greedyv3":
        env = GreedyWrapperV3Env(
            engine=engine,
            quant_step=TOPK_QUANT_STEP,
            k=TOPK_K,
            oversample_factor=2,
            edge_ratio=0.8,
            random_seed=5,
        )

    elif WRAPPER_MODE == "alphachip":
        env = AlphaChipWrapperEnv(engine=engine, coarse_grid=int(ALPHACHIP_GRID), rot=0)
    elif WRAPPER_MODE == "maskplace":
        # Defaults are fixed here (per request): grid=224, rot=0, soft_coefficient=1.0
        env = MaskPlaceWrapperEnv(engine=engine, grid=224, rot=0, soft_coefficient=1.0)
    else:
        raise ValueError(f"Unknown WRAPPER_MODE={WRAPPER_MODE!r} (expected 'greedy'|'alphachip'|'maskplace')")

    if AGENT_MODE == "greedy":
        agent = GreedyAgent(prior_temperature=1.0)
    elif AGENT_MODE == "alphachip":
        if WRAPPER_MODE != "alphachip":
            raise ValueError("AlphaChipAgent supports WRAPPER_MODE='alphachip' only.")
        if not ALPHACHIP_CHECKPOINT_PATH:
            raise ValueError("ALPHACHIP_CHECKPOINT_PATH must be set when AGENT_MODE='alphachip'.")
        agent = AlphaChipAgent(coarse_grid=int(ALPHACHIP_GRID), checkpoint_path=str(ALPHACHIP_CHECKPOINT_PATH), device=device)
    elif AGENT_MODE == "maskplace":
        if WRAPPER_MODE != "maskplace":
            raise ValueError("MaskPlaceAgent supports WRAPPER_MODE='maskplace' only.")
        agent = MaskPlaceAgent(
            device=device,
            grid=224,
            soft_coefficient=1.0,
            checkpoint_path=str(MASKPLACE_CHECKPOINT_PATH) if MASKPLACE_CHECKPOINT_PATH else None,
        )
    else:
        raise ValueError(f"Unknown AGENT_MODE={AGENT_MODE!r} (expected 'greedy'|'alphachip'|'maskplace')")

    if SEARCH_MODE == "none":
        search = None
    elif SEARCH_MODE == "mcts":
        search = MCTSSearch(
            config=MCTSConfig(
                num_simulations=MCTS_SIMS,
                rollout_enabled=bool(MCTS_ROLLOUT_ENABLED),
                rollout_depth=int(ROLLOUT_DEPTH),
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
                track_top_k=TRACK_TOP_K,
                track_verbose=TRACK_VERBOSE,
            )
        )
    else:
        raise ValueError(f"Unknown SEARCH_MODE={SEARCH_MODE!r} (expected 'none'|'mcts'|'beam')")

    pipe = DecisionPipeline(agent=agent, search=search)

    obs, _info = env.reset(options=loaded.reset_kwargs)
    terminated = truncated = False
    total_reward = 0.0

    start = time.perf_counter()
    step = 0
    last_candidates: CandidateSet | None = None
    frames: list[StepFrame] = []

    print("[inference]")
    print(" ENV_JSON=", ENV_JSON, "WRAPPER_MODE=", WRAPPER_MODE, "AGENT_MODE=", AGENT_MODE, "SEARCH_MODE=", SEARCH_MODE, "device=", device)

    while not (terminated or truncated):
        step += 1
        next_gid = env.engine.remaining[0] if env.engine.remaining else None

        # For visualization: Greedy(TopK) wrapper provides action_xyrot; alphachip can be visualized by decoding valid actions (omitted for now).
        if isinstance(obs, dict) and ("action_mask" in obs):
            if "action_xyrot" in obs:
                last_candidates = CandidateSet(xyrot=obs["action_xyrot"], mask=obs["action_mask"], gid=next_gid)
            elif WRAPPER_MODE == "maskplace":
                # Avoid allocating all 224*224 candidates for plotting; subsample valid actions.
                mask = obs["action_mask"].to(device=device, dtype=torch.bool).view(-1)
                idxs = torch.where(mask)[0][:5000]
                xy = torch.zeros((int(idxs.numel()), 3), dtype=torch.long, device=device)
                for t, ai in enumerate(idxs.tolist()):
                    x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))  # type: ignore[attr-defined]
                    xy[t, 0] = int(x_bl)
                    xy[t, 1] = int(y_bl)
                    xy[t, 2] = int(rot)
                last_candidates = CandidateSet(
                    xyrot=xy,
                    mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device),
                    gid=next_gid,
                    meta={"subsampled": True},
                )
            else:
                last_candidates = None

        # Build candidates + action first so we can snapshot and visualize pre-step policy.
        action, dbg_act, candidates = pipe.act(env=env, obs=obs)
        snap = env.get_snapshot()
        scores = agent.policy(env=env.engine, obs=obs, candidates=candidates).detach().to(device="cpu").numpy()
        v = float(agent.value(env=env.engine, obs=obs, candidates=candidates))
        cost = float(env.engine.cost())
        frames.append(
            StepFrame(
                snapshot=snap,
                candidates=candidates,
                scores=scores,
                selected_action=int(action),
                value=float(v),
                cost=float(cost),
                step_idx=int(step),
            )
        )

        obs, reward, terminated, truncated, info = env.step(int(action))
        dbg = dict(dbg_act)
        dbg["candidates"] = candidates
        total_reward += float(reward)
        print(f"[step] {step} next_gid={next_gid} dbg={dbg}")

        if terminated or truncated:
            reason = info.get("reason", None)
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"step={step} placed={len(env.engine.placed)} cost={env.engine.total_cost():.3f} reason={reason}"
            )

    end = time.perf_counter()
    print(f"Total computation time: {end - start:.4f} seconds")
    print(f"episode_reward={total_reward:.3f} terminated={terminated} truncated={truncated}")

    # Print top-K results if tracking was enabled
    if search is not None and hasattr(search, "top_tracker") and search.top_tracker is not None:
        top_results = search.top_tracker.get_results()
        if top_results:
            print(f"\n[Top-{len(top_results)} Search Results]")
            for i, result in enumerate(top_results):
                print(f"  #{i+1}: cost={result.cost:.2f}, placed={len(result.positions)}, cum_reward={result.cum_reward:.3f}")

    out_dir = Path("results") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{AGENT_MODE}_{WRAPPER_MODE}_{SEARCH_MODE}.png"

    # Interactive slide viewer (←/→): policy scatter colored by agent.policy scores.
    if frames:
        # IMPORTANT: browse_steps restores snapshots internally. Save & restore the final state
        # so that the final plot/save below always reflects the true end layout.
        final_snap = env.get_snapshot()
        browse_steps(env, frames=frames, title="Inference browser (←/→ to navigate, q to quit)")
        env.set_snapshot(final_snap)

    # Route planning
    # planner = RoutePlanner(env.engine, algorithm="astar")
    # routes = planner.plan_all()

    # Preview before saving (interactive; close the window to continue).
    plot_layout(env, candidate_set=None, routes= None)

    save_layout(
        env,
        show_masks=SHOW_MASKS,
        show_flow=SHOW_FLOW,
        show_score=SHOW_SCORE,
        show_zones=False,
        candidate_set=None,
        save_path=str(out_path),
    )
    print(f"saved_layout={out_path}")
    
    # Save placement JSON
    placement_path = out_dir / f"{ts}_{AGENT_MODE}_{WRAPPER_MODE}_{SEARCH_MODE}.json"
    env.engine.save_placement(str(placement_path))
    print(f"saved_placement={placement_path}")

    # Save top-K results if tracking was enabled
    if search is not None and hasattr(search, "top_tracker") and search.top_tracker is not None:
        top_results = search.top_tracker.get_results()
        for i, result in enumerate(top_results):
            env.set_snapshot(result.snapshot)
            top_path = out_dir / f"{ts}_top{i+1}_cost{result.cost:.1f}.png"
            save_layout(
                env,
                show_masks=SHOW_MASKS,
                show_flow=SHOW_FLOW,
                show_score=SHOW_SCORE,
                show_zones=False,
                candidate_set=None,
                save_path=str(top_path),
            )
            print(f"saved_top_{i+1}={top_path}")


if __name__ == "__main__":
    main()

