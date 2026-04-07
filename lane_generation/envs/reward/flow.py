from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LaneLengthReward:
    """Lane delta-cost reward component.

    Core rule:
    - new edge cost dominates (new_len)
    - same-direction reuse can be rewarded (negative cost term)
    - reverse-direction overlap is invalid (infinite cost by default)
    """

    new_edge_weight: float = 1.0
    path_len_weight: float = 0.0
    reuse_weight: float = 0.0
    turn_weight: float = 0.0
    invalid_penalty: float = float("inf")

    def required(self) -> set[str]:
        return {"candidate_edge_idx", "candidate_edge_mask"}

    def score(self, *, lane_dir_flat: torch.Tensor) -> torch.Tensor:
        return lane_dir_flat.to(dtype=torch.float32).sum()

    def delta_batch(
        self,
        *,
        lane_dir_flat: torch.Tensor,
        reverse_edge_lut: torch.Tensor,
        edge_valid_flat: torch.Tensor,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_turns: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_edge_idx.dim() != 2 or candidate_edge_mask.dim() != 2:
            raise ValueError("candidate_edge_idx/mask must be [K,L]")
        if candidate_edge_idx.shape != candidate_edge_mask.shape:
            raise ValueError("candidate_edge_idx/mask shape mismatch")

        idx = candidate_edge_idx.to(dtype=torch.long)
        mask = candidate_edge_mask.to(dtype=torch.bool)

        occ = lane_dir_flat[idx] & mask
        rev_occ = lane_dir_flat[reverse_edge_lut[idx]] & mask
        edge_valid = edge_valid_flat[idx] | (~mask)

        invalid = rev_occ.any(dim=1) | (~edge_valid).any(dim=1)
        new_edge = (~occ) & mask
        reuse = occ

        new_len = new_edge.to(dtype=torch.float32).sum(dim=1)
        path_len = mask.to(dtype=torch.float32).sum(dim=1)
        reuse_len = reuse.to(dtype=torch.float32).sum(dim=1)

        cost = (
            float(self.new_edge_weight) * new_len
            + float(self.path_len_weight) * path_len
            - float(self.reuse_weight) * reuse_len
        )
        if candidate_turns is not None:
            cost = cost + float(self.turn_weight) * candidate_turns.to(dtype=torch.float32)

        if torch.isinf(torch.tensor(float(self.invalid_penalty))):
            cost = torch.where(invalid, torch.full_like(cost, float("inf")), cost)
        else:
            cost = cost + invalid.to(dtype=torch.float32) * float(self.invalid_penalty)

        return cost, invalid
