from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple

import torch

if TYPE_CHECKING:
    from lane_generation.envs.action import LaneRoute
    from lane_generation.envs.action_space import ActionSpace
    from lane_generation.envs.state import LaneState


class Agent(Protocol):
    def policy(self, *, obs: dict, action_space: "ActionSpace") -> torch.Tensor:
        """Return float32 [K] policy probabilities."""

    def select_action(self, *, obs: dict, action_space: "ActionSpace") -> int:
        """Return action index in [0, K)."""

    def value(self, *, obs: dict, action_space: "ActionSpace") -> float:
        """Return scalar expected remaining return."""


class BaseAdapter(ABC):
    """Base decision-adapter for FactoryLaneEnv."""

    metadata = {"render_modes": []}

    def __init__(self, *, candidate_k: int = 8, random_seed: int = 0) -> None:
        self.candidate_k = int(candidate_k)
        self.env = None
        self._rng = torch.Generator()
        self._rng.manual_seed(int(random_seed))

    def bind(self, env) -> None:
        self.env = env

    @abstractmethod
    def _generate_candidates(
        self,
        *,
        state: "LaneState",
        flow_idx: int,
        k: int,
        rng: torch.Generator,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Return ``(edge_id_tensors, turn_counts)`` for up to *k* candidates."""

    def _empty_action_space(self, *, flow_index: int, device: torch.device) -> "ActionSpace":
        from lane_generation.envs.action_space import ActionSpace

        return ActionSpace(
            flow_index=int(flow_index),
            candidate_edge_idx=torch.zeros((0, 0), dtype=torch.long, device=device),
            candidate_edge_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
            valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    def build_action_space(self) -> "ActionSpace":
        from lane_generation.envs.action_space import ActionSpace

        if self.env is None:
            raise RuntimeError("adapter is not bound to env")
        state = self.env.get_state()
        flow_idx = state.current_flow_index()
        if flow_idx is None:
            return self._empty_action_space(flow_index=-1, device=state.device)

        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        candidates, turns_l = self._generate_candidates(
            state=state,
            flow_idx=int(flow_idx),
            k=max(1, int(self.candidate_k)),
            rng=self._rng,
        )

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

        valid = edge_mask.any(dim=1)
        planned_slot_idx = state.preview_lane_slots_batch(
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
        )

        costs = self.env.reward_composer.delta_batch(
            state,
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            candidate_lane_slot_idx=planned_slot_idx,
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
            candidate_lane_slot_idx=planned_slot_idx,
        )

    def resolve_action(self, action_idx: int, action_space: "ActionSpace") -> Optional["LaneRoute"]:
        from lane_generation.envs.action import LaneRoute

        i = int(action_idx)
        k = int(action_space.valid_mask.shape[0])
        if i < 0 or i >= k:
            return None
        if not bool(action_space.valid_mask[i].item()):
            return None

        edges = action_space.candidate_edge_idx[i][action_space.candidate_edge_mask[i]]
        planned_slots = None
        if action_space.candidate_lane_slot_idx is not None:
            planned_slots = action_space.candidate_lane_slot_idx[i][action_space.candidate_edge_mask[i]]
        turns = int(action_space.candidate_turns[i].item()) if action_space.candidate_turns is not None else 0
        path_len = (
            float(action_space.candidate_path_len[i].item())
            if action_space.candidate_path_len is not None
            else float(edges.numel())
        )
        return LaneRoute(
            flow_index=int(action_space.flow_index),
            candidate_index=i,
            edge_indices=edges,
            path_length=path_len,
            turns=turns,
            planned_lane_slots=planned_slots,
        )
