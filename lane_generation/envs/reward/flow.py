"""Per-step lane reward components.

Three independent cost terms:

* :class:`LanePathLengthReward` — total edge count per candidate (full
  routed path length, regardless of overlap with existing lanes).
* :class:`LaneTurnReward` — number of 90-degree direction changes.
* :class:`LaneNewEdgeReward` — edges that are *new* to the grid (edges
  already placed by earlier same-direction flows cost zero, so merged
  lanes are free).

None of these components judge reverse-direction overlap. That is a hard
routing constraint enforced by :meth:`LaneState.pathfind` when
``forbid_opposite=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LanePathLengthReward:
    """Cost = total candidate edge count (full path length per flow)."""

    def delta_batch(
        self,
        *,
        candidate_edge_mask: torch.Tensor,
        **_kw,
    ) -> torch.Tensor:
        return candidate_edge_mask.to(dtype=torch.float32).sum(dim=1)

    def score(self, *, lane_dir_flat: torch.Tensor, **_kw) -> torch.Tensor:
        return lane_dir_flat.to(dtype=torch.float32).sum()


@dataclass
class LaneTurnReward:
    """Cost = number of 90-degree turns along the candidate path."""

    def delta_batch(
        self,
        *,
        candidate_turns: Optional[torch.Tensor] = None,
        **_kw,
    ) -> torch.Tensor:
        if candidate_turns is None:
            raise ValueError("LaneTurnReward requires candidate_turns")
        return candidate_turns.to(dtype=torch.float32)

    def score(self, **_kw) -> torch.Tensor:
        return torch.tensor(0.0)


@dataclass
class LaneNewEdgeReward:
    """Cost = edges not already occupied by a same-direction lane.

    If an earlier flow already placed a directed edge and the current
    candidate reuses it (same direction), that edge contributes zero cost.
    This is the "merged lane" metric.
    """

    def delta_batch(
        self,
        *,
        lane_dir_flat: torch.Tensor,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        **_kw,
    ) -> torch.Tensor:
        idx = candidate_edge_idx.to(dtype=torch.long)
        mask = candidate_edge_mask.to(dtype=torch.bool)
        occ = lane_dir_flat[idx] & mask
        new_edge = (~occ) & mask
        return new_edge.to(dtype=torch.float32).sum(dim=1)

    def score(self, *, lane_dir_flat: torch.Tensor, **_kw) -> torch.Tensor:
        return lane_dir_flat.to(dtype=torch.float32).sum()
