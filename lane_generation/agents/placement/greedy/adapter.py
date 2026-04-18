from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch

from lane_generation.agents.base import BaseAdapter

if TYPE_CHECKING:
    from lane_generation.envs.state import LaneState


@dataclass(frozen=True)
class LaneAdapterConfig:
    candidate_k: int = 8
    random_seed: int = 0


def _build_candidates_with_builder(
    *,
    state: "LaneState",
    src_ports: torch.Tensor,
    k_target: int,
    build_path,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Run primary + diversity passes around a path-building closure."""
    candidates: List[torch.Tensor] = []
    turns_l: List[int] = []
    seen: Set[Tuple[int, ...]] = set()
    src_list = src_ports.detach().cpu().tolist()

    for p in src_list:
        path = build_path((int(p[0]), int(p[1])))
        if not path:
            continue
        edges, turns = state.path_to_edge_ids_and_turns(path)
        key = tuple(int(x) for x in edges.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(edges)
        turns_l.append(int(turns))
        if len(candidates) >= int(k_target):
            break

    retry = 0
    while len(candidates) < int(k_target) and retry < int(k_target) * 4 and len(src_list) > 0:
        p = src_list[retry % len(src_list)]
        path = build_path((int(p[0]), int(p[1])))
        retry += 1
        if not path:
            continue
        edges, turns = state.path_to_edge_ids_and_turns(path)
        key = tuple(int(x) for x in edges.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(edges)
        turns_l.append(int(turns))

    return candidates, turns_l


class DirectRouteAdapter(BaseAdapter):
    """Generate K shortest-path candidates via :meth:`LaneState.pathfind`."""

    def __init__(self, *, config: Optional[LaneAdapterConfig] = None) -> None:
        cfg = config or LaneAdapterConfig()
        self.config = cfg
        super().__init__(candidate_k=int(cfg.candidate_k), random_seed=int(cfg.random_seed))

    def _generate_candidates(self, *, state, flow_idx, k, rng):
        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return [], []

        src_gid, dst_gid, _ = state.flow_pair(flow_idx)
        allow_mask = state.combined_mask(flow_idx)

        def build_path(src_xy):
            return state.pathfind(
                src_xy=src_xy,
                dst_xy=dst_ports,
                src_gid=src_gid,
                dst_gid=dst_gid,
                allow_mask=allow_mask,
                rng=rng,
            )

        return _build_candidates_with_builder(
            state=state,
            src_ports=src_ports,
            k_target=int(k),
            build_path=build_path,
        )


LaneAdapter = DirectRouteAdapter

