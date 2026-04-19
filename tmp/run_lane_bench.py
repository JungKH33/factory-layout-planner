"""Ad-hoc runner: synthesize a lane env and benchmark route algorithms per method."""
from __future__ import annotations

import logging
import sys

import torch

from lane_generation.agents.placement.greedy import LaneAdapter, LaneAdapterConfig
from lane_generation.envs.env import FactoryLaneEnv
from lane_generation.envs.routing import RoutingConfig
from lane_generation.envs.state import LaneFlowSpec, PortSelector, PortSpec


GRID_W = 150
GRID_H = 150


def _make_blocked(grid_w: int, grid_h: int) -> torch.Tensor:
    """Create a blocked_static map with a few rectangular obstacles."""
    m = torch.zeros((grid_h, grid_w), dtype=torch.bool)
    m[40:70, 30:55] = True
    m[40:70, 95:120] = True
    m[95:120, 60:100] = True
    return m


def _mk_flow(sg, dg, w, src_xy, dst_xy, catalog):
    src_ids = []
    for i, (x, y) in enumerate(src_xy):
        pid = f"{sg}.ex.{i}"
        catalog[pid] = PortSpec(port_id=pid, gid=sg, xy=(int(x), int(y)), kind="exit")
        src_ids.append(pid)
    dst_ids = []
    for i, (x, y) in enumerate(dst_xy):
        pid = f"{dg}.en.{i}"
        catalog[pid] = PortSpec(port_id=pid, gid=dg, xy=(int(x), int(y)), kind="entry")
        dst_ids.append(pid)
    return LaneFlowSpec(
        src=PortSelector(gid=sg, port_ids=tuple(src_ids)),
        dst=PortSelector(gid=dg, port_ids=tuple(dst_ids)),
        weight=float(w),
    )


def _make_flows() -> tuple[list[LaneFlowSpec], dict]:
    """Mix of single-source (1 port) and multi-source (several ports) flows."""
    cat: dict = {}
    flows = [
        _mk_flow("A", "B", 1.0, ((10, 10),), ((140, 140),), cat),
        _mk_flow("A", "C", 0.8, ((10, 10),), ((140, 10),), cat),
        _mk_flow("D", "E", 0.6, ((10, 140),), ((140, 140),), cat),
        _mk_flow("D", "F", 0.5, ((75, 10),), ((75, 140),), cat),
        _mk_flow("G", "H", 0.9, ((10, 30), (10, 40), (10, 50)), ((140, 75),), cat),
        _mk_flow("G", "I", 0.7,
                 ((15, 85), (15, 95), (15, 105), (15, 115)),
                 ((135, 30),), cat),
        _mk_flow("J", "K", 0.5,
                 ((20, 20), (20, 30)),
                 ((130, 130), (130, 140)), cat),
    ]
    return flows, cat


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
    flows, port_catalog = _make_flows()
    env = FactoryLaneEnv(
        grid_width=grid_w,
        grid_height=grid_h,
        blocked_static=blocked,
        flows=flows,
        port_specs=port_catalog,
        device=device,
        routing_config=routing_cfg,
    )

    env.set_adapter(LaneAdapter(config=LaneAdapterConfig(candidate_k=4)))
    print(f"[done] selected={type(env.get_state().routing).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
