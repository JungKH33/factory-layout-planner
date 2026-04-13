from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..state import LaneState


@dataclass
class TerminalReward:
    penalty_weight: float
    reward_scale: float

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_scale must be > 0")

    def ratio(self, state: LaneState) -> float:
        return float(state.remaining_weight_ratio())

    def penalty(self, state: LaneState) -> float:
        return float(self.penalty_weight) * self.ratio(state)

    def failure_reward(self, state: LaneState) -> float:
        return -self.penalty(state) / float(self.reward_scale)


@dataclass
class RewardComposer:
    components: Dict[str, object]
    weights: Dict[str, float] = field(default_factory=dict)
    reward_scale: float = 100.0

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_scale must be > 0")

    def to_reward(self, value: torch.Tensor | float) -> torch.Tensor | float:
        if torch.is_tensor(value):
            return -value.to(dtype=torch.float32) / float(self.reward_scale)
        return -float(value) / float(self.reward_scale)

    def score(self, state: LaneState) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=state.device)
        kw = dict(
            edge_map=state.edge_map,
            edge_lane_mask=state.edge_lane_mask,
            edge_valid_flat=state.edge_valid_flat,
            reverse_edge_lut=state.reverse_edge_lut,
        )
        for name, comp in self.components.items():
            w = float(self.weights.get(name, 1.0))
            if w == 0.0:
                continue
            if hasattr(comp, "score"):
                total = total + w * comp.score(**kw)
        return total

    def delta_batch(
        self,
        state: LaneState,
        *,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_lane_slot_idx: Optional[torch.Tensor] = None,
        candidate_turns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = int(candidate_edge_idx.shape[0])
        total = torch.zeros((k,), dtype=torch.float32, device=state.device)
        kw = dict(
            edge_map=state.edge_map,
            edge_lane_mask=state.edge_lane_mask,
            edge_valid_flat=state.edge_valid_flat,
            reverse_edge_lut=state.reverse_edge_lut,
            candidate_edge_idx=candidate_edge_idx,
            candidate_edge_mask=candidate_edge_mask,
            candidate_lane_slot_idx=candidate_lane_slot_idx,
            candidate_turns=candidate_turns,
        )
        for name, comp in self.components.items():
            w = float(self.weights.get(name, 1.0))
            if w == 0.0:
                continue
            total = total + w * comp.delta_batch(**kw)
        return total
