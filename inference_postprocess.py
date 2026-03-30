"""Inference Postprocess - Dynamic Storage Placement.

기존 그룹 배치가 완료된 환경에서 동적 storage를 추가 배치합니다.
inference.py와 유사한 파이프라인을 사용하되, DynamicStorageEnv를 사용합니다.

사용법:
    python inference_postprocess.py
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List

import time
import torch

from envs.action import EnvAction
from envs.env_loader import load_env
from envs.visualizer import plot_layout, save_layout, draw_layout_layers as _draw_layout_layers

from search.mcts import MCTSConfig, MCTSSearch

from agents.placement.greedy import GreedyAgent

from postprocess.dynamic_env import DynamicStorageEnv, DynamicGroupConfig
from postprocess.dynamic_wrapper import DynamicStorageWrapper


# ============================================================
# Config
# ============================================================

# 기존 배치 결과 (또는 배치할 환경)
BASE_ENV_JSON: str = "envs/env_configs/zones_01.json"

# Storage 설정
STORAGE_CONFIGS: List[Dict] = [
    {
        "gid": "storage_1",
        "total_cells": 300,
        "cell_width": 20,
        "cell_depth": 30,
        "clearance_w": 5,
        "clearance_h": 5,
        "gap_w": 0,
        "gap_h": 5,
        "rotatable": True,
        "cell_z_height": 2.0,
        "z_overhead": 1.0,
        # "max_units_per_step": 10,  # step당 최대 4개 unit
    },
    # 다양한 파라미터 케이스 추가
    {
        "gid": "storage_2",
        "total_cells": 260,
        "cell_width": 18,
        "cell_depth": 24,
        "clearance_w": 4,
        "clearance_h": 6,
        "gap_w": 0,
        "gap_h": 6,
        "rotatable": True,
        "cell_z_height": 1.6,
        "z_overhead": 0.8,
        # "max_units_per_step": 10,
    },
    {
        "gid": "storage_3",
        "total_cells": 160,
        "cell_width": 16,
        "cell_depth": 28,
        "clearance_w": 3,
        "clearance_h": 3,
        "gap_w": 3,
        "gap_h": 0,
        "rotatable": False,
        "cell_z_height": 2.4,
        "z_overhead": 1.2,
        # "max_units_per_step": 10,
    },
]

# Flow 연결 (그룹 간 연결)
# None이면 storage 그룹 간 + 기존 배치된 그룹과 양방향 연결 자동 생성
GROUP_FLOW: Dict[str, Dict[str, float]] | None = None

# Wrapper 설정
TOPK_K: int = 50

# Search 설정
SEARCH_MODE: str = "mcts"  # "none" | "mcts"
MCTS_SIMS: int = 100
MCTS_ROLLOUT_ENABLED: bool = True
ROLLOUT_DEPTH: int = 5
MCTS_CACHE_DECISION_STATE: bool = True

# Visualization
SHOW_FLOW: bool = True
SHOW_SCORE: bool = False
SHOW_MASKS: bool = True

logger = logging.getLogger(__name__)


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # CPU 강제 (디버깅용)
    
    logger.info("=" * 60)
    logger.info("Inference Postprocess - Dynamic Storage Placement")
    logger.info("=" * 60)
    
    # ===== 1. Base 환경 로드 =====
    logger.info("Loading base environment: %s", BASE_ENV_JSON)
    loaded = load_env(BASE_ENV_JSON, device=device)
    base_env = loaded.env
    base_env.reset(options=loaded.reset_kwargs)
    
    logger.info("Grid: %s x %s", base_env.grid_width, base_env.grid_height)
    logger.info("Placed groups: %s", len(base_env.get_state().placed))
    logger.info("Remaining groups: %s", len(base_env.get_state().remaining))
    
    # ===== 1.5. 기존 그룹 미리 배치 (하드코딩) =====
    # (gid, x_center, y_center, variant_index) 형식 — center 기반
    # forbidden area: [0, 0, 150, 200] 피해서 배치
    PRE_PLACEMENTS = [
        ("A", 180.0, 240.0, 0),
        ("B", 210.0, 360.0, 0),
        ("E", 370.0, 400.0, 0),
        ("G", 215.0, 455.0, 0),
        ("H", 520.0, 250.0, 0),
        ("I", 405.0, 125.0, 0),
    ]

    if PRE_PLACEMENTS:
        logger.info("Pre-placing %s groups (hardcoded)", len(PRE_PLACEMENTS))
        for gid, x_center, y_center, oi in PRE_PLACEMENTS:
            if gid in base_env.get_state().remaining:
                _obs, _reward, _terminated, _truncated, info = base_env.step_action(
                    EnvAction(group_id=gid, x_center=x_center, y_center=y_center, variant_index=oi)
                )
                if info.get("reason") == "placed":
                    logger.info("Placed: %s at center (%.1f, %.1f) oi=%d", gid, x_center, y_center, oi)
                else:
                    logger.warning("Failed to pre-place %s: reason=%s", gid, info.get("reason"))
            else:
                logger.info("Skip: %s not in remaining (or already placed)", gid)
        
        logger.info("Now placed: %s", list(base_env.get_state().placed))
    
    # ===== 2. Storage 설정 =====
    logger.info("Storage configuration")
    configs = [DynamicGroupConfig(**cfg) for cfg in STORAGE_CONFIGS]
    
    for cfg in configs:
        logger.info(
            "- %s: %s cells, %sx%s, rotatable=%s",
            cfg.group_id,
            cfg.total_cells,
            cfg.cell_width,
            cfg.cell_depth,
            cfg.rotatable,
        )
    
    # ===== 3. Flow 설정 =====
    logger.info("Flow configuration")
    if GROUP_FLOW is not None:
        group_flow = GROUP_FLOW
    else:
        # 자동 생성: storage 그룹 간 + 기존 배치된 그룹과 양방향 연결
        group_flow: Dict[str, Dict[str, float]] = {}
        storage_gids = [cfg.group_id for cfg in configs]
        
        # storage 그룹 간 양방향 연결
        for i, gid1 in enumerate(storage_gids):
            group_flow[gid1] = {}
            for j, gid2 in enumerate(storage_gids):
                if i != j:
                    group_flow[gid1][gid2] = 1.0
        
        # 기존 배치된 그룹과 양방향 연결
        for gid in base_env.get_state().placed:
            for cfg in configs:
                group_flow.setdefault(gid, {})[cfg.group_id] = 1.0
                group_flow[cfg.group_id][gid] = 1.0
    
    logger.info("Group flow: %s", group_flow)
    
    # ===== 4. Dynamic Env 생성 =====
    logger.info("Creating DynamicStorageEnv")
    dynamic_env = DynamicStorageEnv(
        base_env=base_env,
        configs=configs,
        group_flow=group_flow,
    )
    
    # ===== 5. Wrapper 적용 =====
    logger.info("Creating DynamicStorageWrapper (k=%s)", TOPK_K)
    env = DynamicStorageWrapper(
        dynamic_env=dynamic_env,
        k=TOPK_K,
        random_seed=42,
    )
    
    # ===== 6. Agent & Search =====
    logger.info("Setting up Agent and Search")
    agent = GreedyAgent(prior_temperature=1.0)
    
    if SEARCH_MODE == "none":
        search = None
        logger.info("Search: None (greedy)")
    elif SEARCH_MODE == "mcts":
        search = MCTSSearch(
            config=MCTSConfig(
                num_simulations=MCTS_SIMS,
                rollout_enabled=MCTS_ROLLOUT_ENABLED,
                rollout_depth=ROLLOUT_DEPTH,
                cache_decision_state=MCTS_CACHE_DECISION_STATE,
            )
        )
        logger.info("Search: MCTS (sims=%s)", MCTS_SIMS)
    else:
        raise ValueError(f"Unknown SEARCH_MODE: {SEARCH_MODE}")

    if search is not None:
        logger.warning(
            "DynamicStorageWrapper does not support env-owned search after pipeline refactor; fallback to no-search."
        )
        search = None
    
    # ===== 7. 실행 =====
    logger.info("Running inference")
    logger.info("=" * 60)
    
    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    step = 0
    
    start = time.perf_counter()
    
    while not (terminated or truncated):
        step += 1
        current_gid = dynamic_env.group_id
        
        # Wrapper decide (no pipeline/search in dynamic wrapper path)
        obs_decision = env.build_observation()
        action_space = env.build_action_space()
        valid_n = int(action_space.valid_mask.to(torch.int64).sum().item())
        if valid_n <= 0:
            reward = -1.0
            terminated = False
            truncated = True
            info = {"reason": "no_valid_actions"}
            logger.warning("step %s gid=%s, reason=no_valid_actions", step, current_gid)
            total_reward += float(reward)
            break
        action = int(agent.select_action(obs=obs_decision, action_space=action_space))
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        
        # 로그
        x, y, rot, _, _ = env.decode_action(int(action))
        logger.info(
            "step %s gid=%s, action=%s, pos=(%s,%s), rot=%s, units=%s, cells=%s, reward=%.3f",
            step,
            current_gid,
            action,
            x,
            y,
            rot,
            info.get("num_units", 0),
            info.get("total_cells", 0),
            reward,
        )
        
        if terminated:
            logger.info("DONE: All storage groups placed!")
        elif truncated:
            logger.warning("TRUNCATED: reason=%s", info.get("reason", "unknown"))
    
    end = time.perf_counter()
    
    logger.info("=" * 60)
    logger.info("Total time: %.3fs", end - start)
    logger.info("Total steps: %s", step)
    logger.info("Total reward: %.3f", total_reward)
    
    # ===== 8. 시각화 =====
    logger.info("Visualization")
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, dynamic_env.W)
    ax.set_ylim(0, dynamic_env.H)
    ax.set_aspect("equal")
    
    # Base layout
    _draw_layout_layers(ax=ax, engine=base_env)
    
    # Storage overlay
    palette = [
        ("lightgreen", "darkgreen"),
        ("lightskyblue", "royalblue"),
        ("moccasin", "peru"),
        ("plum", "purple"),
    ]
    legend_patches = []

    total_units_all = 0
    total_cells_all = 0

    cfg_map = {c.group_id: c for c in configs}

    for j, (gid, units) in enumerate(dynamic_env.placed_history.items()):
        face, edge = palette[j % len(palette)]
        legend_patches.append(mpatches.Patch(facecolor=face, edgecolor=edge, label=gid))

        cfg = cfg_map.get(gid)
        if cfg is None:
            # config를 못 찾으면 스킵
            continue

        for ux, uy, rot, stack in units:
            # gid별 config 기준으로 cell/clearance size 계산
            if rot == 90:
                cw, ch = int(cfg.cell_depth), int(cfg.cell_width)
                cl_w, cl_h = int(cfg.clearance_h), int(cfg.clearance_w)
            else:
                cw, ch = int(cfg.cell_width), int(cfg.cell_depth)
                cl_w, cl_h = int(cfg.clearance_w), int(cfg.clearance_h)
            
            # block (clearance 포함) 영역 - 연한 색으로
            bx, by = ux - cl_w, uy - cl_h
            bw, bh = cw + 2 * cl_w, ch + 2 * cl_h
            block_rect = mpatches.Rectangle(
                (bx, by), bw, bh,
                facecolor=face, edgecolor=edge,
                alpha=0.25, linewidth=1, linestyle='--'
            )
            ax.add_patch(block_rect)
            
            # unit 영역 - 진한 색으로
            rect = mpatches.Rectangle(
                (ux, uy), cw, ch,
                facecolor=face, edgecolor=edge,
                alpha=0.85, linewidth=2
            )
            ax.add_patch(rect)
            ax.text(ux + cw/2, uy + ch/2, str(stack),
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='black')

            total_units_all += 1
            total_cells_all += int(stack)

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    
    # Title
    ax.set_title(f"Dynamic Storage Placement (units={total_units_all}, cells={total_cells_all})")
    
    plt.tight_layout()
    
    # Save
    out_dir = Path("results") / "inference_postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_storage.png"
    
    plt.savefig(out_path, dpi=150)
    logger.info("Saved: %s", out_path)
    
    plt.show()


if __name__ == "__main__":
    main()
