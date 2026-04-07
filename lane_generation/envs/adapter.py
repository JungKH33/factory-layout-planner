from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .action import LaneRoute
from .action_space import ActionSpace
from .wavefront import backtrace_shortest_path, wavefront_distance_field


@dataclass(frozen=True)
class LaneAdapterConfig:
    candidate_k: int = 8
    max_wave_iters: int = 0
    max_backtrace_steps: int = 0
    random_seed: int = 0


class LaneAdapter:
    """Build lane candidates for current flow and resolve action index to route."""

    def __init__(self, *, config: Optional[LaneAdapterConfig] = None) -> None:
        self.config = config or LaneAdapterConfig()
        self.env = None
        self._rng = torch.Generator()
        self._rng.manual_seed(int(self.config.random_seed))

    def bind(self, env) -> None:
        self.env = env

    def _empty_action_space(self, *, flow_index: int, device: torch.device) -> ActionSpace:
        return ActionSpace(
            flow_index=int(flow_index),
            candidate_edge_idx=torch.zeros((0, 0), dtype=torch.long, device=device),
            candidate_edge_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
            valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    def build_action_space(self) -> ActionSpace:
        if self.env is None:
            raise RuntimeError("LaneAdapter is not bound to env")
        state = self.env.get_state()
        flow_idx = state.current_flow_index()
        if flow_idx is None:
            return self._empty_action_space(flow_index=-1, device=state.device)

        src_ports, dst_ports = state.flow.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        free_map = (~state.maps.blocked_static).to(dtype=torch.bool, device=state.device)
        dist = wavefront_distance_field(
            free_map=free_map,
            seeds_xy=dst_ports,
            max_iters=int(self.config.max_wave_iters),
        )

        candidates: List[torch.Tensor] = []
        turns_l: List[int] = []
        seen = set()

        src_list = src_ports.detach().cpu().tolist()
        k_target = max(1, int(self.config.candidate_k))

        # Primary pass: one route per source port.
        for p in src_list:
            path = backtrace_shortest_path(
                dist=dist,
                src_xy=(int(p[0]), int(p[1])),
                rng=self._rng,
                max_steps=int(self.config.max_backtrace_steps),
            )
            if not path:
                continue
            edges, turns = state.maps.path_to_edge_ids_and_turns(path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(edges)
            turns_l.append(int(turns))
            if len(candidates) >= k_target:
                break

        # Diversity pass: stochastic tie-break retries.
        retry = 0
        while len(candidates) < k_target and retry < k_target * 4:
            p = src_list[retry % len(src_list)]
            path = backtrace_shortest_path(
                dist=dist,
                src_xy=(int(p[0]), int(p[1])),
                rng=self._rng,
                max_steps=int(self.config.max_backtrace_steps),
            )
            retry += 1
            if not path:
                continue
            edges, turns = state.maps.path_to_edge_ids_and_turns(path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(edges)
            turns_l.append(int(turns))

        if len(candidates) == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        k = len(candidates)
        lmax = max(int(c.numel()) for c in candidates)
        edge_idx = torch.zeros((k, lmax), dtype=torch.long, device=state.device)
        edge_mask = torch.zeros((k, lmax), dtype=torch.bool, device=state.device)
        path_len = torch.zeros((k,), dtype=torch.float32, device=state.device)
        turns_t = torch.tensor(turns_l, dtype=torch.float32, device=state.device)

        for i, edges in enumerate(candidates):
            n = int(edges.numel())
            if n > 0:
                edge_idx[i, :n] = edges
                edge_mask[i, :n] = True
            path_len[i] = float(n)

        rev = state.maps.reverse_edge_lut[edge_idx]
        reverse_hit = (state.maps.lane_dir_flat[rev] & edge_mask).any(dim=1)
        edge_ok = (state.maps.edge_valid_flat[edge_idx] | (~edge_mask)).all(dim=1)
        valid = (~reverse_hit) & edge_ok & (edge_mask.any(dim=1))

        costs = self.env.reward_composer.delta_batch(
            state,
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            candidate_turns=turns_t,
        )

        return ActionSpace(
            flow_index=int(flow_idx),
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            valid_mask=valid,
            candidate_path_len=path_len,
            candidate_turns=turns_t,
            candidate_cost=costs,
        )

    def resolve_action(self, action_idx: int, action_space: ActionSpace) -> Optional[LaneRoute]:
        i = int(action_idx)
        k = int(action_space.valid_mask.shape[0])
        if i < 0 or i >= k:
            return None
        if not bool(action_space.valid_mask[i].item()):
            return None

        edges = action_space.candidate_edge_idx[i][action_space.candidate_edge_mask[i]]
        turns = int(action_space.candidate_turns[i].item()) if action_space.candidate_turns is not None else 0
        path_len = float(action_space.candidate_path_len[i].item()) if action_space.candidate_path_len is not None else float(edges.numel())
        return LaneRoute(
            flow_index=int(action_space.flow_index),
            candidate_index=i,
            edge_indices=edges,
            path_length=path_len,
            turns=turns,
        )
