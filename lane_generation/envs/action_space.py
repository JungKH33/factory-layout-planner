from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ActionSpace:
    """Lane candidate action space for one flow step.

    - ``flow_index``: current flow row index in FlowGraph.
    - ``candidate_edge_idx``: [K, Lmax] padded directed-edge indices.
    - ``candidate_edge_mask``: [K, Lmax] validity mask for edge_idx.
    - ``valid_mask``: [K] candidate validity.
    """

    flow_index: int
    candidate_edge_idx: torch.Tensor
    candidate_edge_mask: torch.Tensor
    valid_mask: torch.Tensor
    candidate_path_len: Optional[torch.Tensor] = None
    candidate_turns: Optional[torch.Tensor] = None
    candidate_cost: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_edge_idx, torch.Tensor):
            raise TypeError("candidate_edge_idx must be torch.Tensor")
        if not isinstance(self.candidate_edge_mask, torch.Tensor):
            raise TypeError("candidate_edge_mask must be torch.Tensor")
        if not isinstance(self.valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be torch.Tensor")
        if self.candidate_edge_idx.dim() != 2:
            raise ValueError(f"candidate_edge_idx must be [K,L], got {tuple(self.candidate_edge_idx.shape)}")
        if self.candidate_edge_mask.shape != self.candidate_edge_idx.shape:
            raise ValueError(
                "candidate_edge_mask shape mismatch: "
                f"{tuple(self.candidate_edge_mask.shape)} vs {tuple(self.candidate_edge_idx.shape)}"
            )
        k = int(self.candidate_edge_idx.shape[0])
        if self.valid_mask.dim() != 1 or int(self.valid_mask.shape[0]) != k:
            raise ValueError(f"valid_mask must be [K], got {tuple(self.valid_mask.shape)} for K={k}")
        if self.candidate_edge_mask.dtype != torch.bool:
            raise TypeError("candidate_edge_mask must be bool")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be bool")
        if self.candidate_path_len is not None:
            if self.candidate_path_len.shape != (k,):
                raise ValueError("candidate_path_len must be [K]")
        if self.candidate_turns is not None:
            if self.candidate_turns.shape != (k,):
                raise ValueError("candidate_turns must be [K]")
        if self.candidate_cost is not None:
            if self.candidate_cost.shape != (k,):
                raise ValueError("candidate_cost must be [K]")
