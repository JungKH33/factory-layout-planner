from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple

import torch

from lane_generation.agents.base import BaseAdapter
from .adapter import LaneAdapterConfig

if TYPE_CHECKING:
    from lane_generation.envs.state import LaneState


class HananAdapter(BaseAdapter):
    """Generate candidates by selecting one Hanan point per action."""

    def __init__(self, *, config: LaneAdapterConfig | None = None) -> None:
        cfg = config or LaneAdapterConfig()
        self.config = cfg
        super().__init__(candidate_k=int(cfg.candidate_k), random_seed=int(cfg.random_seed))

    @staticmethod
    def _build_hanan_points(*, state: "LaneState", src_ports: torch.Tensor, dst_ports: torch.Tensor) -> List[Tuple[int, int]]:
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return []
        ports = torch.cat([src_ports, dst_ports], dim=0).to(dtype=torch.long, device=state.device)
        xs = torch.unique(ports[:, 0]).detach().cpu().tolist()
        ys = torch.unique(ports[:, 1]).detach().cpu().tolist()

        h = int(state.grid_height)
        w = int(state.grid_width)
        blocked = state.blocked_static
        points: List[Tuple[int, int]] = []
        for yv in ys:
            y = int(yv)
            if y < 0 or y >= h:
                continue
            for xv in xs:
                x = int(xv)
                if x < 0 or x >= w:
                    continue
                if bool(blocked[y, x].item()):
                    continue
                points.append((x, y))
        return points

    def _generate_candidates(self, *, state, flow_idx, k, rng):
        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return [], []

        src_gid, dst_gid, _ = state.flow_pair(flow_idx)
        allow_mask = state.combined_mask(flow_idx)
        hanan_points = self._build_hanan_points(state=state, src_ports=src_ports, dst_ports=dst_ports)
        if len(hanan_points) == 0:
            return [], []

        if len(hanan_points) > 1:
            perm = torch.randperm(len(hanan_points), generator=rng).detach().cpu().tolist()
            ordered_points = [hanan_points[int(i)] for i in perm]
        else:
            ordered_points = hanan_points

        src_list = [(int(p[0]), int(p[1])) for p in src_ports.detach().cpu().tolist()]
        seen: Set[Tuple[int, ...]] = set()
        packed: List[Tuple[torch.Tensor, int, int]] = []

        for hx, hy in ordered_points:
            hp_xy = (int(hx), int(hy))
            hp_t = torch.tensor([[hx, hy]], dtype=torch.long, device=state.device)

            p2 = state.pathfind(
                src_xy=hp_xy,
                dst_xy=dst_ports,
                dst_gid=dst_gid,
                allow_mask=allow_mask,
                rng=rng,
            )
            if not p2:
                continue

            best_path = None
            best_cost = None
            for sxy in src_list:
                p1 = state.pathfind(
                    src_xy=sxy,
                    dst_xy=hp_t,
                    src_gid=src_gid,
                    allow_mask=allow_mask,
                    rng=rng,
                )
                if not p1:
                    continue
                merged = p1 + p2[1:]
                if len(merged) < 2:
                    continue
                edge_est = len(merged) - 1
                if best_path is None or edge_est < int(best_cost):
                    best_path = merged
                    best_cost = edge_est

            if best_path is None:
                continue

            edges, turns = state.path_to_edge_ids_and_turns(best_path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            packed.append((edges, int(turns), int(edges.numel())))

        if len(packed) == 0:
            return [], []

        packed.sort(key=lambda it: (int(it[2]), int(it[1])))
        top = packed[: max(1, int(k))]
        return [it[0] for it in top], [int(it[1]) for it in top]

