from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..action import GroupId
    from ..state.base import EnvState


@dataclass
class FlowReward:
    def required(self) -> set[str]:
        return {"entry_points", "exit_points"}

    @staticmethod
    def _to_port_tensor(
        xy: torch.Tensor,
        *,
        name: str,
        allow_2d: bool,
    ) -> torch.Tensor:
        if not torch.is_tensor(xy):
            raise TypeError(f"{name} must be a torch.Tensor")
        if xy.dim() == 3 and int(xy.shape[-1]) == 2:
            return xy.to(dtype=torch.float32)
        if allow_2d and xy.dim() == 2 and int(xy.shape[-1]) == 2:
            return xy.to(dtype=torch.float32).view(int(xy.shape[0]), 1, 2)
        raise ValueError(f"{name} must be [N,P,2]" + (" or [N,2]" if allow_2d else ""))

    @staticmethod
    def _port_mask(
        ports: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        name: str,
    ) -> torch.Tensor:
        expected = (int(ports.shape[0]), int(ports.shape[1]))
        if mask is None:
            return torch.ones(expected, dtype=torch.bool, device=ports.device)
        m = mask.to(device=ports.device, dtype=torch.bool)
        if tuple(m.shape) != expected:
            raise ValueError(f"{name} must be shape {expected}, got {tuple(m.shape)}")
        return m

    @staticmethod
    def _as_weight_matrix(
        weight: torch.Tensor,
        *,
        m: int,
        t: int,
        device: torch.device,
    ) -> torch.Tensor:
        w = weight.to(device=device, dtype=torch.float32)
        if w.dim() == 1:
            if int(w.shape[0]) != t:
                raise ValueError(f"weight length must match target count {t}, got {tuple(w.shape)}")
            return w.view(1, t)
        if w.dim() == 2:
            if tuple(w.shape) != (m, t):
                raise ValueError(f"weight matrix must be shape {(m, t)}, got {tuple(w.shape)}")
            return w
        raise ValueError(f"weight must be [T] or [M,T], got dim={w.dim()}")

    @staticmethod
    def _mode_tensor(mode: str, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """Convert ``"min"``/``"mean"`` to bool tensor. Returns None for ``"min"`` (fast-path)."""
        if mode == "mean":
            return torch.ones((n,), dtype=torch.bool, device=device)
        return None

    @staticmethod
    def _masked_pair_reduce(
        cost: torch.Tensor,
        valid: torch.Tensor,
        c_modes: Optional[torch.Tensor] = None,
        t_modes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce [M,T,C,P] cost to [M,T] using per-facility port aggregation.

        c_modes: [M] bool (True=mean, False=min). None → all min.
        t_modes: [T] bool (True=mean, False=min). None → all min.

        Returns (values [M,T], c_idx [M,T], p_idx [M,T]).
        c_idx/p_idx are argmin indices from the min path; for mean-mode
        facilities they serve as placeholders (callers use the valid mask
        to enumerate all ports instead).
        """
        if cost.dim() != 4 or valid.dim() != 4:
            raise ValueError("cost/valid must be rank-4 tensors [M,T,C,P]")
        if tuple(cost.shape) != tuple(valid.shape):
            raise ValueError(f"cost and valid shape mismatch: {tuple(cost.shape)} vs {tuple(valid.shape)}")

        has_valid = valid.any(dim=3).any(dim=2)  # [M,T]

        # Always compute min over P for argmin indices
        masked_inf = cost.masked_fill(~valid, float("inf"))
        min_p = masked_inf.min(dim=3)  # values [M,T,C], indices [M,T,C]

        # --- P-dim reduction (target ports) ---
        t_all_min = t_modes is None or not t_modes.any().item()
        if t_all_min:
            reduced_p = min_p.values
        else:
            zero_filled = cost.masked_fill(~valid, 0.0)
            count_p = valid.sum(dim=3).clamp(min=1).to(torch.float32)
            mean_p = zero_filled.sum(dim=3) / count_p  # [M,T,C]
            if t_modes.all().item():
                reduced_p = mean_p
            else:
                reduced_p = torch.where(t_modes.view(1, -1, 1), mean_p, min_p.values)

        # C-dim validity after P reduction
        c_valid = valid.any(dim=3)  # [M,T,C]
        reduced_p_inf = reduced_p.masked_fill(~c_valid, float("inf"))
        min_c = reduced_p_inf.min(dim=2)  # values [M,T], indices [M,T]

        # --- C-dim reduction (candidate ports) ---
        c_all_min = c_modes is None or not c_modes.any().item()
        if c_all_min:
            result = torch.where(has_valid, min_c.values, torch.zeros_like(min_c.values))
        else:
            reduced_p_zero = reduced_p.masked_fill(~c_valid, 0.0)
            count_c = c_valid.sum(dim=2).clamp(min=1).to(torch.float32)
            mean_c = reduced_p_zero.sum(dim=2) / count_c  # [M,T]
            if c_modes.all().item():
                raw = mean_c
            else:
                raw = torch.where(c_modes.view(-1, 1), mean_c, min_c.values)
            result = torch.where(has_valid, raw, torch.zeros_like(raw))

        c_idx = min_c.indices
        p_idx = min_p.indices.gather(2, c_idx.unsqueeze(2)).squeeze(2)
        return result, c_idx, p_idx

    def _reduce_distance(
        self,
        *,
        candidate_ports: torch.Tensor,          # [M,C,2]
        candidate_mask: Optional[torch.Tensor], # [M,C]
        target_ports: torch.Tensor,             # [T,P,2]
        target_mask: Optional[torch.Tensor],    # [T,P]
        target_weight: torch.Tensor,            # [T] or [M,T]
        c_modes: Optional[torch.Tensor] = None, # [M] bool (True=mean)
        t_modes: Optional[torch.Tensor] = None, # [T] bool (True=mean)
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns ([M], c_idx [M,T], p_idx [M,T]).

        c_idx / p_idx are None when result is a zero early-exit (no candidates / targets).
        """
        m = int(candidate_ports.shape[0])
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=candidate_ports.device), None, None
        t = int(target_ports.shape[0])
        if t == 0 or int(target_weight.numel()) == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device), None, None
        if int(candidate_ports.shape[1]) == 0 or int(target_ports.shape[1]) == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device), None, None

        weight_mt = self._as_weight_matrix(
            target_weight,
            m=m,
            t=t,
            device=candidate_ports.device,
        )

        cand = candidate_ports[:, None, :, None, :]  # [M,1,C,1,2]
        tgt = target_ports[None, :, None, :, :]      # [1,T,1,P,2]
        dist = (cand - tgt).abs().sum(dim=-1)        # [M,T,C,P]

        cand_mask = self._port_mask(candidate_ports, candidate_mask, name="candidate_mask")
        tgt_mask = self._port_mask(target_ports, target_mask, name="target_mask")
        valid = cand_mask[:, None, :, None] & tgt_mask[None, :, None, :]  # [M,T,C,P]
        dist_reduced, c_idx, p_idx = self._masked_pair_reduce(dist, valid, c_modes, t_modes)
        return (dist_reduced * weight_mt).sum(dim=1), c_idx, p_idx

    def score(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        flow_w: torch.Tensor,
        exit_modes: Optional[torch.Tensor] = None,
        entry_modes: Optional[torch.Tensor] = None,
        return_argmin: bool = False,
    ):
        """Compute absolute flow score for the placed state (tensor-only).

        exit_modes / entry_modes: [P] bool (True=mean) per placed facility.
        None → all min (backward compatible).

        If return_argmin=True, returns (scalar, c_idx [P,P], p_idx [P,P]).
        """
        placed_en = self._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = self._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        zero = torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if placed_en.shape[0] == 0:
            return (zero, None, None) if return_argmin else zero
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            return (zero, None, None) if return_argmin else zero
        per_src, c_idx, p_idx = self._reduce_distance(
            candidate_ports=placed_ex,
            candidate_mask=placed_exits_mask,
            target_ports=placed_en,
            target_mask=placed_entries_mask,
            target_weight=flow_w,
            c_modes=exit_modes,
            t_modes=entry_modes,
        )
        total = per_src.sum()
        return (total, c_idx, p_idx) if return_argmin else total

    def delta(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        w_out: torch.Tensor,
        w_in: torch.Tensor,
        candidate_entries: torch.Tensor,
        candidate_exits: torch.Tensor,
        candidate_entries_mask: Optional[torch.Tensor],
        candidate_exits_mask: Optional[torch.Tensor],
        c_exit_mode: str = "min",
        c_entry_mode: str = "min",
        t_entry_modes: Optional[torch.Tensor] = None,
        t_exit_modes: Optional[torch.Tensor] = None,
        return_argmin: bool = False,
    ):
        """Compute incremental flow cost for M candidate placements.

        Port aggregation modes:
          c_exit_mode / c_entry_mode: candidate facility's mode (uniform for all M).
          t_entry_modes / t_exit_modes: [T] bool per placed facility. None → all min.

        If return_argmin=True, returns a tuple:
          (delta [M],
           out_argmin: (c_idx [M,T_out], p_idx [M,T_out], out_idx BoolTensor),
           in_argmin:  (c_idx [M,T_in],  p_idx [M,T_in],  in_idx BoolTensor))
        """
        cand_entries = self._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = self._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = self._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = self._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        w_out_t = w_out.view(-1)
        w_in_t = w_in.view(-1)
        m = int(cand_entries.shape[0])
        device = cand_entries.device

        out_idx = (w_out_t != 0)
        in_idx = (w_in_t != 0)
        if placed_entries_mask is not None:
            out_idx = out_idx & placed_entries_mask.any(dim=1)
        if placed_exits_mask is not None:
            in_idx = in_idx & placed_exits_mask.any(dim=1)
        has_out = bool(out_idx.any().item())
        has_in = bool(in_idx.any().item())
        if not (has_out or has_in):
            if return_argmin:
                return torch.zeros((m,), dtype=torch.float32, device=device), None, None
            return torch.zeros((m,), dtype=torch.float32, device=device)

        out_term = torch.zeros((m,), dtype=torch.float32, device=device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=device)
        out_am = None
        in_am = None

        if has_out:
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            out_term, oc_idx, op_idx = self._reduce_distance(
                candidate_ports=cand_exits,
                candidate_mask=candidate_exits_mask,
                target_ports=out_entries,
                target_mask=out_entries_mask,
                target_weight=out_w,
                c_modes=self._mode_tensor(c_exit_mode, m, device),
                t_modes=t_entry_modes[out_idx] if t_entry_modes is not None else None,
            )
            if return_argmin:
                out_am = (oc_idx, op_idx, out_idx)
        if has_in:
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            in_term, ic_idx, ip_idx = self._reduce_distance(
                candidate_ports=cand_entries,
                candidate_mask=candidate_entries_mask,
                target_ports=in_exits,
                target_mask=in_exits_mask,
                target_weight=in_w,
                c_modes=self._mode_tensor(c_entry_mode, m, device),
                t_modes=t_exit_modes[in_idx] if t_exit_modes is not None else None,
            )
            if return_argmin:
                in_am = (ic_idx, ip_idx, in_idx)
        if return_argmin:
            return out_term + in_term, out_am, in_am
        return out_term + in_term


@dataclass
class FlowCollisionReward:
    """Flow reward with route-collision penalty on a blocked grid map."""

    collision_weight: float = 10.0

    def required(self) -> set[str]:
        return {"entry_points", "exit_points"}

    @staticmethod
    def _prefix(blocked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = blocked.to(dtype=torch.int32)
        zrow = torch.zeros((b.shape[0], 1), dtype=torch.int32, device=b.device)
        zcol = torch.zeros((1, b.shape[1]), dtype=torch.int32, device=b.device)
        row_ps = torch.cat([zrow, torch.cumsum(b, dim=1)], dim=1)  # [H, W+1]
        col_ps = torch.cat([zcol, torch.cumsum(b, dim=0)], dim=0)  # [H+1, W]
        return row_ps, col_ps

    @staticmethod
    def _min_lshape_hits(
        *,
        blocked: torch.Tensor,
        row_ps: torch.Tensor,
        col_ps: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        H, W = blocked.shape
        x1i = torch.round(x1).to(dtype=torch.long)
        y1i = torch.round(y1).to(dtype=torch.long)
        x2i = torch.round(x2).to(dtype=torch.long)
        y2i = torch.round(y2).to(dtype=torch.long)

        oob = (
            (x1i < 0) | (x1i >= W) | (x2i < 0) | (x2i >= W) |
            (y1i < 0) | (y1i >= H) | (y2i < 0) | (y2i >= H)
        )

        x1c = x1i.clamp(0, W - 1)
        y1c = y1i.clamp(0, H - 1)
        x2c = x2i.clamp(0, W - 1)
        y2c = y2i.clamp(0, H - 1)

        xlo = torch.minimum(x1c, x2c)
        xhi = torch.maximum(x1c, x2c)
        ylo = torch.minimum(y1c, y2c)
        yhi = torch.maximum(y1c, y2c)

        # x-then-y
        h1 = row_ps[y1c, xhi + 1] - row_ps[y1c, xlo]
        v1 = col_ps[yhi + 1, x2c] - col_ps[ylo, x2c]
        t1 = blocked[y1c, x2c].to(dtype=torch.int32)
        c1 = h1 + v1 - t1

        # y-then-x
        v2 = col_ps[yhi + 1, x1c] - col_ps[ylo, x1c]
        h2 = row_ps[y2c, xhi + 1] - row_ps[y2c, xlo]
        t2 = blocked[y2c, x1c].to(dtype=torch.int32)
        c2 = v2 + h2 - t2

        c = torch.minimum(c1, c2).to(dtype=torch.float32)
        bad = torch.full_like(c, float(H + W))
        return torch.where(oob, bad, c)

    def _reduce_cost_with_collision(
        self,
        *,
        candidate_ports: torch.Tensor,          # [M,C,2]
        candidate_mask: Optional[torch.Tensor], # [M,C]
        target_ports: torch.Tensor,             # [T,P,2]
        target_mask: Optional[torch.Tensor],    # [T,P]
        target_weight: torch.Tensor,            # [T] or [M,T]
        blocked: torch.Tensor,
        row_ps: torch.Tensor,
        col_ps: torch.Tensor,
        c_modes: Optional[torch.Tensor] = None,
        t_modes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = int(candidate_ports.shape[0])
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=candidate_ports.device)
        t = int(target_ports.shape[0])
        if t == 0 or int(target_weight.numel()) == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)
        if int(candidate_ports.shape[1]) == 0 or int(target_ports.shape[1]) == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)

        weight_mt = FlowReward._as_weight_matrix(
            target_weight,
            m=m,
            t=t,
            device=candidate_ports.device,
        )

        cand = candidate_ports[:, None, :, None, :]  # [M,1,C,1,2]
        tgt = target_ports[None, :, None, :, :]      # [1,T,1,P,2]
        dist = (cand - tgt).abs().sum(dim=-1)        # [M,T,C,P]
        hits = self._min_lshape_hits(
            blocked=blocked,
            row_ps=row_ps,
            col_ps=col_ps,
            x1=cand[..., 0],
            y1=cand[..., 1],
            x2=tgt[..., 0],
            y2=tgt[..., 1],
        )
        cost = dist + float(self.collision_weight) * hits

        cand_valid = FlowReward._port_mask(candidate_ports, candidate_mask, name="candidate_mask")
        tgt_valid = FlowReward._port_mask(target_ports, target_mask, name="target_mask")
        valid = cand_valid[:, None, :, None] & tgt_valid[None, :, None, :]
        cost_reduced, _, _ = FlowReward._masked_pair_reduce(cost, valid, c_modes, t_modes)
        return (cost_reduced * weight_mt).sum(dim=1)

    def score(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        flow_w: torch.Tensor,
        route_blocked: Optional[torch.Tensor],
        exit_modes: Optional[torch.Tensor] = None,
        entry_modes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if route_blocked is None:
            return FlowReward().score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_modes=exit_modes,
                entry_modes=entry_modes,
            )
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        if placed_en.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)

        blocked = route_blocked.to(dtype=torch.bool, device=placed_en.device)
        row_ps, col_ps = self._prefix(blocked)
        per_src = self._reduce_cost_with_collision(
            candidate_ports=placed_ex,
            candidate_mask=placed_exits_mask,
            target_ports=placed_en,
            target_mask=placed_entries_mask,
            target_weight=flow_w,
            blocked=blocked,
            row_ps=row_ps,
            col_ps=col_ps,
            c_modes=exit_modes,
            t_modes=entry_modes,
        )
        return per_src.sum()

    def delta(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        w_out: torch.Tensor,
        w_in: torch.Tensor,
        candidate_entries: torch.Tensor,
        candidate_exits: torch.Tensor,
        candidate_entries_mask: Optional[torch.Tensor],
        candidate_exits_mask: Optional[torch.Tensor],
        route_blocked: Optional[torch.Tensor],
        c_exit_mode: str = "min",
        c_entry_mode: str = "min",
        t_entry_modes: Optional[torch.Tensor] = None,
        t_exit_modes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if route_blocked is None:
            return FlowReward().delta(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                w_out=w_out,
                w_in=w_in,
                candidate_entries=candidate_entries,
                candidate_exits=candidate_exits,
                candidate_entries_mask=candidate_entries_mask,
                candidate_exits_mask=candidate_exits_mask,
                c_exit_mode=c_exit_mode,
                c_entry_mode=c_entry_mode,
                t_entry_modes=t_entry_modes,
                t_exit_modes=t_exit_modes,
            )
        cand_entries = FlowReward._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = FlowReward._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        m = int(cand_entries.shape[0])
        device = cand_entries.device
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=device)

        w_out_t = w_out.view(-1)
        w_in_t = w_in.view(-1)
        out_idx = (w_out_t != 0)
        in_idx = (w_in_t != 0)
        if placed_entries_mask is not None:
            out_idx = out_idx & placed_entries_mask.any(dim=1)
        if placed_exits_mask is not None:
            in_idx = in_idx & placed_exits_mask.any(dim=1)
        has_out = bool(out_idx.any().item())
        has_in = bool(in_idx.any().item())
        if not (has_out or has_in):
            return torch.zeros((m,), dtype=torch.float32, device=device)

        blocked = route_blocked.to(dtype=torch.bool, device=device)
        row_ps, col_ps = self._prefix(blocked)

        out_term = torch.zeros((m,), dtype=torch.float32, device=device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=device)

        if has_out:
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            out_term = self._reduce_cost_with_collision(
                candidate_ports=cand_exits,
                candidate_mask=candidate_exits_mask,
                target_ports=out_entries,
                target_mask=out_entries_mask,
                target_weight=out_w,
                blocked=blocked,
                row_ps=row_ps,
                col_ps=col_ps,
                c_modes=FlowReward._mode_tensor(c_exit_mode, m, device),
                t_modes=t_entry_modes[out_idx] if t_entry_modes is not None else None,
            )

        if has_in:
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            in_term = self._reduce_cost_with_collision(
                candidate_ports=cand_entries,
                candidate_mask=candidate_entries_mask,
                target_ports=in_exits,
                target_mask=in_exits_mask,
                target_weight=in_w,
                blocked=blocked,
                row_ps=row_ps,
                col_ps=col_ps,
                c_modes=FlowReward._mode_tensor(c_entry_mode, m, device),
                t_modes=t_exit_modes[in_idx] if t_exit_modes is not None else None,
            )

        return out_term + in_term


