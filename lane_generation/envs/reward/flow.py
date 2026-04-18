"""Per-step lane reward components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import torch


def _total_lane_length(
    *,
    edge_lane_mask: torch.Tensor,
    reverse_edge_lut: torch.Tensor,
    edge_valid_flat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dev = edge_lane_mask.device
    mask_flat = edge_lane_mask.view(-1)
    rev_flat = reverse_edge_lut.view(-1)
    valid_flat = (
        edge_valid_flat.view(-1).to(device=dev, dtype=torch.bool)
        if isinstance(edge_valid_flat, torch.Tensor)
        else torch.ones((int(mask_flat.shape[0]),), dtype=torch.bool, device=dev)
    )

    total = 0
    n = int(mask_flat.shape[0])
    for e in range(n):
        if not bool(valid_flat[e].item()):
            continue
        r = int(rev_flat[e].item())
        if e > r:
            continue
        me = int(mask_flat[e].item())
        mr = int(mask_flat[r].item()) if 0 <= r < n else 0
        total += int((me | mr).bit_count())
    return torch.tensor(float(total), dtype=torch.float32, device=dev)


def _lane_length_delta_from_planned_slots_batch(
    *,
    edge_lane_mask: torch.Tensor,
    reverse_edge_lut: torch.Tensor,
    edge_valid_flat: Optional[torch.Tensor],
    candidate_edge_idx: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    candidate_lane_slot_idx: torch.Tensor,
) -> torch.Tensor:
    dev = edge_lane_mask.device
    idx = candidate_edge_idx.to(device=dev, dtype=torch.long)
    cand_mask = candidate_edge_mask.to(device=dev, dtype=torch.bool)
    slot_idx = candidate_lane_slot_idx.to(device=dev, dtype=torch.long)
    if idx.shape != cand_mask.shape or idx.shape != slot_idx.shape:
        raise ValueError(
            "candidate tensors must share shape: "
            f"idx={tuple(idx.shape)} mask={tuple(cand_mask.shape)} slot={tuple(slot_idx.shape)}"
        )
    mask_flat = edge_lane_mask.view(-1)
    rev_flat = reverse_edge_lut.view(-1)
    valid_flat = (
        edge_valid_flat.view(-1).to(device=dev, dtype=torch.bool)
        if isinstance(edge_valid_flat, torch.Tensor)
        else torch.ones((int(mask_flat.shape[0]),), dtype=torch.bool, device=dev)
    )

    k = int(idx.shape[0])
    out = torch.zeros((k,), dtype=torch.float32, device=dev)

    for ci in range(k):
        overrides: Dict[int, int] = {}
        touched_leaders: Set[int] = set()
        l = int(idx.shape[1]) if idx.dim() == 2 else 0
        infeasible = False
        for li in range(l):
            if not bool(cand_mask[ci, li].item()):
                continue
            e = int(idx[ci, li].item())
            if e < 0 or e >= int(mask_flat.shape[0]):
                continue
            if not bool(valid_flat[e].item()):
                continue
            slot = int(slot_idx[ci, li].item())
            if slot < 0:
                # Infeasible candidate — should have been caught upstream by valid_mask
                infeasible = True
                break
            r = int(rev_flat[e].item())
            dir_mask = int(overrides.get(e, int(mask_flat[e].item())))
            new_dir_mask = int(dir_mask) | (1 << int(slot))
            if new_dir_mask != int(dir_mask):
                overrides[e] = int(new_dir_mask)
                leader = int(e) if int(e) <= int(r) else int(r)
                touched_leaders.add(leader)

        if infeasible:
            out[ci] = float("inf")
            continue

        delta = 0
        for leader in touched_leaders:
            if leader < 0 or leader >= int(mask_flat.shape[0]):
                continue
            if not bool(valid_flat[leader].item()):
                continue
            r = int(rev_flat[leader].item())
            before_a = int(mask_flat[leader].item())
            before_b = int(mask_flat[r].item()) if 0 <= r < int(mask_flat.shape[0]) else 0
            after_a = int(overrides.get(leader, before_a))
            after_b = int(overrides.get(r, before_b))
            delta += int((after_a | after_b).bit_count()) - int((before_a | before_b).bit_count())
        out[ci] = float(delta)

    return out


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

    def score(
        self,
        *,
        edge_lane_mask: Optional[torch.Tensor] = None,
        reverse_edge_lut: Optional[torch.Tensor] = None,
        edge_valid_flat: Optional[torch.Tensor] = None,
        **_kw,
    ) -> torch.Tensor:
        if not isinstance(edge_lane_mask, torch.Tensor) or not isinstance(reverse_edge_lut, torch.Tensor):
            return torch.tensor(0.0, dtype=torch.float32)
        return _total_lane_length(
            edge_lane_mask=edge_lane_mask,
            reverse_edge_lut=reverse_edge_lut,
            edge_valid_flat=edge_valid_flat,
        )


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
    """Cost = increment of total lane length on undirected edge pairs."""

    def delta_batch(
        self,
        *,
        edge_lane_mask: Optional[torch.Tensor] = None,
        reverse_edge_lut: Optional[torch.Tensor] = None,
        edge_valid_flat: Optional[torch.Tensor] = None,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        candidate_lane_slot_idx: Optional[torch.Tensor] = None,
        **_kw,
    ) -> torch.Tensor:
        if not isinstance(edge_lane_mask, torch.Tensor) or not isinstance(reverse_edge_lut, torch.Tensor):
            return candidate_edge_mask.to(dtype=torch.float32).sum(dim=1)
        if not isinstance(candidate_lane_slot_idx, torch.Tensor):
            raise ValueError("LaneNewEdgeReward requires candidate_lane_slot_idx")
        return _lane_length_delta_from_planned_slots_batch(
            edge_lane_mask=edge_lane_mask,
            reverse_edge_lut=reverse_edge_lut,
            edge_valid_flat=edge_valid_flat,
            candidate_edge_idx=candidate_edge_idx,
            candidate_edge_mask=candidate_edge_mask,
            candidate_lane_slot_idx=candidate_lane_slot_idx,
        )

    def score(
        self,
        *,
        edge_lane_mask: Optional[torch.Tensor] = None,
        reverse_edge_lut: Optional[torch.Tensor] = None,
        edge_valid_flat: Optional[torch.Tensor] = None,
        **_kw,
    ) -> torch.Tensor:
        if isinstance(edge_lane_mask, torch.Tensor) and isinstance(reverse_edge_lut, torch.Tensor):
            return _total_lane_length(
                edge_lane_mask=edge_lane_mask,
                reverse_edge_lut=reverse_edge_lut,
                edge_valid_flat=edge_valid_flat,
            )
        return torch.tensor(0.0, dtype=torch.float32)
