from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..state.base import EnvState


@dataclass
class TerminalReward:
    penalty_weight: float
    reward_scale: float

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_scale must be > 0")

    def ratio(self, state: EnvState) -> float:
        return float(state.flow.remaining_weight_ratio(state.routed_mask))

    def penalty(self, state: EnvState) -> float:
        return float(self.penalty_weight) * self.ratio(state)

    def failure_reward(self, state: EnvState) -> float:
        return -self.penalty(state) / float(self.reward_scale)


@dataclass
class RewardComposer:
    components: Dict[str, object]
    weights: Dict[str, float]
    reward_scale: float = 100.0

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_scale must be > 0")

    def to_reward(self, value: torch.Tensor | float) -> torch.Tensor | float:
        if torch.is_tensor(value):
            return -value.to(dtype=torch.float32) / float(self.reward_scale)
        return -float(value) / float(self.reward_scale)

    def score(self, state: EnvState) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=state.device)
        lane = self.components.get("lane", None)
        if lane is not None:
            total = total + float(self.weights.get("lane", 1.0)) * lane.score(
                lane_dir_flat=state.maps.lane_dir_flat,
            )
        return total

    def delta_batch(
        self,
        state: EnvState,
        *,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_turns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = int(candidate_edge_idx.shape[0])
        total = torch.zeros((k,), dtype=torch.float32, device=state.device)

        lane = self.components.get("lane", None)
        if lane is not None:
            lane_cost, _invalid = lane.delta_batch(
                lane_dir_flat=state.maps.lane_dir_flat,
                reverse_edge_lut=state.maps.reverse_edge_lut,
                edge_valid_flat=state.maps.edge_valid_flat,
                candidate_edge_idx=candidate_edge_idx,
                candidate_edge_mask=candidate_edge_mask,
                candidate_turns=candidate_turns,
            )
            total = total + float(self.weights.get("lane", 1.0)) * lane_cost

        return total
