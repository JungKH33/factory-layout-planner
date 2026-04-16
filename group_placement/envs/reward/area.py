from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GridOccupancyReward:
    """Grid-cell occupancy reward using coarse cells (any overlap => occupied)."""

    cell_w: int = 1
    cell_h: int = 1

    def required(self) -> set[str]:
        return {"min_x", "max_x", "min_y", "max_y"}

    def _candidate_mask(
        self,
        *,
        candidate_min_x: torch.Tensor,
        candidate_max_x: torch.Tensor,
        candidate_min_y: torch.Tensor,
        candidate_max_y: torch.Tensor,
        gh: int,
        gw: int,
    ) -> torch.Tensor:
        cw = int(self.cell_w)
        ch = int(self.cell_h)
        if cw <= 0 or ch <= 0:
            raise ValueError("cell_w/cell_h must be > 0")
        if gh <= 0 or gw <= 0:
            return torch.zeros((candidate_min_x.shape[0], 0, 0), dtype=torch.bool, device=candidate_min_x.device)

        x0 = torch.floor(candidate_min_x / float(cw)).to(dtype=torch.long)
        x1 = (torch.ceil(candidate_max_x / float(cw)) - 1.0).to(dtype=torch.long)
        y0 = torch.floor(candidate_min_y / float(ch)).to(dtype=torch.long)
        y1 = (torch.ceil(candidate_max_y / float(ch)) - 1.0).to(dtype=torch.long)

        x0 = x0.clamp(0, gw - 1)
        x1 = x1.clamp(0, gw - 1)
        y0 = y0.clamp(0, gh - 1)
        y1 = y1.clamp(0, gh - 1)

        valid = (x1 >= x0) & (y1 >= y0)
        gx = torch.arange(gw, device=candidate_min_x.device, dtype=torch.long).view(1, 1, gw)
        gy = torch.arange(gh, device=candidate_min_x.device, dtype=torch.long).view(1, gh, 1)
        mask = (gx >= x0.view(-1, 1, 1)) & (gx <= x1.view(-1, 1, 1))
        mask = mask & (gy >= y0.view(-1, 1, 1)) & (gy <= y1.view(-1, 1, 1))
        mask = mask & valid.view(-1, 1, 1)
        return mask

    def score(
        self,
        *,
        placed_cell_occupied: torch.Tensor,
        return_meta: bool = False,
    ):
        if placed_cell_occupied is None:
            raise ValueError("placed_cell_occupied is required for GridOccupancyReward.score")
        score = placed_cell_occupied.to(dtype=torch.float32).sum()
        if not return_meta:
            return score
        meta = {
            "occupied_cells": float(score.item()),
            "cell_w": int(self.cell_w),
            "cell_h": int(self.cell_h),
        }
        return score, meta

    def delta(
        self,
        *,
        placed_cell_occupied: torch.Tensor,
        candidate_min_x: torch.Tensor,
        candidate_max_x: torch.Tensor,
        candidate_min_y: torch.Tensor,
        candidate_max_y: torch.Tensor,
        return_meta: bool = False,
    ):
        if placed_cell_occupied is None:
            raise ValueError("placed_cell_occupied is required for GridOccupancyReward.delta")
        base = placed_cell_occupied.to(dtype=torch.bool, device=candidate_min_x.device)
        gh, gw = int(base.shape[0]), int(base.shape[1])
        cand = self._candidate_mask(
            candidate_min_x=candidate_min_x,
            candidate_max_x=candidate_max_x,
            candidate_min_y=candidate_min_y,
            candidate_max_y=candidate_max_y,
            gh=gh,
            gw=gw,
        )
        delta = (cand & (~base.view(1, gh, gw))).to(dtype=torch.float32).sum(dim=(1, 2))
        if not return_meta:
            return delta
        return delta, {"candidate_count": int(delta.shape[0])}


@dataclass
class AreaReward:
    def required(self) -> set[str]:
        return {"min_x", "max_x", "min_y", "max_y"}

    def score(
        self,
        *,
        placed_count: int,
        min_x: torch.Tensor,
        max_x: torch.Tensor,
        min_y: torch.Tensor,
        max_y: torch.Tensor,
        return_meta: bool = False,
    ):
        if placed_count == 0:
            score = torch.tensor(0.0, dtype=torch.float32, device=min_x.device)
        else:
            score = 0.5 * ((max_x - min_x) + (max_y - min_y))
        if not return_meta:
            return score
        meta = {
            "placed_count": int(placed_count),
            "bbox": {
                "min_x": float(min_x.item()),
                "max_x": float(max_x.item()),
                "min_y": float(min_y.item()),
                "max_y": float(max_y.item()),
            },
        }
        return score, meta

    def delta(
        self,
        *,
        placed_count: int,
        cur_min_x: float,
        cur_max_x: float,
        cur_min_y: float,
        cur_max_y: float,
        candidate_min_x: torch.Tensor,
        candidate_max_x: torch.Tensor,
        candidate_min_y: torch.Tensor,
        candidate_max_y: torch.Tensor,
        return_meta: bool = False,
    ):
        if placed_count == 0:
            delta = 0.5 * ((candidate_max_x - candidate_min_x) + (candidate_max_y - candidate_min_y))
        else:
            s_min_x = candidate_min_x.new_tensor(cur_min_x)
            s_max_x = candidate_max_x.new_tensor(cur_max_x)
            s_min_y = candidate_min_y.new_tensor(cur_min_y)
            s_max_y = candidate_max_y.new_tensor(cur_max_y)

            new_min_x = torch.minimum(candidate_min_x, s_min_x)
            new_max_x = torch.maximum(candidate_max_x, s_max_x)
            new_min_y = torch.minimum(candidate_min_y, s_min_y)
            new_max_y = torch.maximum(candidate_max_y, s_max_y)

            cur_hpwl = 0.5 * ((float(cur_max_x) - float(cur_min_x)) + (float(cur_max_y) - float(cur_min_y)))
            new_hpwl = 0.5 * ((new_max_x - new_min_x) + (new_max_y - new_min_y))
            delta = new_hpwl - cur_hpwl
        if not return_meta:
            return delta
        return delta, {"candidate_count": int(delta.shape[0])}
