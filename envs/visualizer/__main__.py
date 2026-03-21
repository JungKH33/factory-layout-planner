"""Demo for the visualizer package.

Usage:
    python -m envs.visualizer              # matplotlib (default)
    python -m envs.visualizer --plotly     # plotly
"""
from __future__ import annotations

import sys

import torch

from agents.placement.alphachip import AlphaChipAdapter
from agents.placement.greedy import GreedyAdapter
from envs.env import FactoryLayoutEnv
from envs.placement.static import StaticRectSpec
from envs.action_space import ActionSpace as CandidateSet
from envs.visualizer import plot_layout, plot_flow_graph


def main():
    backend = "matplotlib"
    if "--plotly" in sys.argv:
        backend = "plotly"

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    groups = {
        "A": StaticRectSpec(
            device=dev, id="A", width=20, height=10,
            entries_rel=[(10.0, 5.0)], exits_rel=[(10.0, 5.0)],
            rotatable=True, zone_values={"weight": 3.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "B": StaticRectSpec(
            device=dev, id="B", width=16, height=16,
            entries_rel=[(8.0, 8.0)], exits_rel=[(8.0, 8.0)],
            rotatable=True, zone_values={"weight": 4.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "C": StaticRectSpec(
            device=dev, id="C", width=18, height=12,
            entries_rel=[(9.0, 6.0)], exits_rel=[(9.0, 6.0)],
            rotatable=True, zone_values={"weight": 12.0, "height": 10.0, "dry": 2.0, "placeable": 1},
        ),
    }
    flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    forbidden_areas = [{"rect": [0, 0, 30, 20]}]

    zone_constraints = {
        "weight": {"dtype": "float", "op": "<=", "default": 10.0, "areas": [{"rect": [60, 0, 120, 80], "value": 20.0}]},
        "height": {"dtype": "float", "op": "<=", "default": 20.0, "areas": [{"rect": [0, 60, 120, 80], "value": 5.0}]},
        "dry": {"dtype": "float", "op": ">=", "default": 0.0, "areas": [{"rect": [0, 40, 60, 80], "value": 2.0}]},
        "placeable": {"dtype": "int", "op": "==", "default": 0, "areas": [{"rect": [30, 20, 120, 80], "value": 1}]},
    }

    initial_positions = {
        "A": (80, 15, 0),
        "B": (82, 32, 0),
    }
    remaining_order = ["C", "A", "B"]

    engine = FactoryLayoutEnv(
        grid_width=120,
        grid_height=80,
        group_specs=groups,
        group_flow=flow,
        forbidden_areas=forbidden_areas,
        zone_constraints=zone_constraints,
        device=dev,
        max_steps=10,
        log=False,
    )

    # 1) AlphaChip adapter demo
    env1 = AlphaChipAdapter(engine=engine, coarse_grid=32)
    _obs1, _ = env1.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    plot_layout(env1, action_space=None, backend=backend)
    plot_flow_graph(env1, backend=backend)

    # 2) Greedy adapter demo
    env2 = GreedyAdapter(
        engine=engine,
        k=70,
        scan_step=5.0,
        quant_step=5.0,
        p_high=0.2,
        p_near=0.8,
        p_coarse=0.0,
        oversample_factor=2,
        random_seed=7,
    )
    topk_obs, _ = env2.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    cand = None
    if isinstance(topk_obs, dict) and ("action_mask" in topk_obs) and ("action_poses" in topk_obs):
        cand = CandidateSet(
            poses=topk_obs["action_poses"],
            mask=topk_obs["action_mask"],
            gid=engine.get_state().remaining[0] if engine.get_state().remaining else None,
        )
    plot_layout(env2, action_space=cand, backend=backend)
    plot_flow_graph(env2, backend=backend)


if __name__ == "__main__":
    main()
