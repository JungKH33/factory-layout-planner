"""Ad-hoc runner: synthesize a lane env and benchmark route algorithms per method."""
from __future__ import annotations

import logging
import sys

import torch

from lane_generation.agents.placement.greedy import LaneAdapter, LaneAdapterConfig
from lane_generation.envs.env import FactoryLaneEnv
from lane_generation.envs.routing import RoutingConfig
from lane_generation.envs.state import LaneFlowSpec


GRID_W = 150
GRID_H = 150


def _make_blocked(grid_w: int, grid_h: int) -> torch.Tensor:
    """Create a blocked_static map with a few rectangular obstacles."""
    m = torch.zeros((grid_h, grid_w), dtype=torch.bool)
    m[40:70, 30:55] = True
    m[40:70, 95:120] = True
    m[95:120, 60:100] = True
    return m


def _make_flows() -> list[LaneFlowSpec]:
    """Mix of single-source (1 port) and multi-source (several ports) flows."""
    return [
        # single_src
        LaneFlowSpec(src_gid="A", dst_gid="B", weight=1.0,
                     src_ports=((10, 10),), dst_ports=((140, 140),)),
        LaneFlowSpec(src_gid="A", dst_gid="C", weight=0.8,
                     src_ports=((10, 10),), dst_ports=((140, 10),)),
        LaneFlowSpec(src_gid="D", dst_gid="E", weight=0.6,
                     src_ports=((10, 140),), dst_ports=((140, 140),)),
        LaneFlowSpec(src_gid="D", dst_gid="F", weight=0.5,
                     src_ports=((75, 10),), dst_ports=((75, 140),)),
        # multi_src
        LaneFlowSpec(src_gid="G", dst_gid="H", weight=0.9,
                     src_ports=((10, 30), (10, 40), (10, 50)),
                     dst_ports=((140, 75),)),
        LaneFlowSpec(src_gid="G", dst_gid="I", weight=0.7,
                     src_ports=((15, 85), (15, 95), (15, 105), (15, 115)),
                     dst_ports=((135, 30),)),
        LaneFlowSpec(src_gid="J", dst_gid="K", weight=0.5,
                     src_ports=((20, 20), (20, 30)),
                     dst_ports=((130, 130), (130, 140))),
    ]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    device = torch.device("cpu")
    grid_w, grid_h = GRID_W, GRID_H

    blocked = _make_blocked(grid_w, grid_h).to(device)

    routing_cfg = RoutingConfig(
        selection="benchmark",
        candidate_k=4,
        benchmark_warmup=2,
        benchmark_rounds=6,
        benchmark_max_flows_per_method=6,
    )
    env = FactoryLaneEnv(
        grid_width=grid_w,
        grid_height=grid_h,
        blocked_static=blocked,
        flows=_make_flows(),
        device=device,
        routing_config=routing_cfg,
    )

    env.set_adapter(LaneAdapter(config=LaneAdapterConfig(candidate_k=4)))
    print(f"[done] selected={type(env.get_state().routing).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
