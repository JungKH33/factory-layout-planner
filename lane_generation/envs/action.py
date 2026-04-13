from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class LaneAction:
    """Lane routing action.

    - ``flow_index``: target flow row index. If None, env routes current flow.
    - ``candidate_index``: index in current ActionSpace candidates.
    """

    candidate_index: int
    flow_index: Optional[int] = None


@dataclass
class LaneRoute:
    """Resolved lane route candidate (edge-index representation)."""

    flow_index: int
    candidate_index: int
    edge_indices: torch.Tensor  # [L]
    path_length: float
    turns: int = 0
    planned_lane_slots: Optional[torch.Tensor] = None  # [L], int slot index
