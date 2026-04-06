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


@dataclass
class FlowLaneDistanceReward:
    """Flow reward using obstacle-aware lane distance (wavefront BFS on grid).

    Notes:
    - Uses exact grid shortest-path distance (4-neighbor) on a blocked map.
    - Supports chunked target processing for memory control.
    - Supports optional coarse routing with ``distance_stride`` (approximation).
    - Deliberately does not include turn minimization in reward (speed-first).
    """

    target_chunk: int = 6
    distance_stride: int = 1
    unreachable_penalty: Optional[float] = None
    max_wave_iters: int = 0
    _body_idx_cache: Dict[Tuple[int, Tuple[object, ...], object], torch.Tensor] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def required(self) -> set[str]:
        return {"entry_points", "exit_points"}

    @staticmethod
    def _ports_to_grid(
        ports_xy: torch.Tensor,
        *,
        stride: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.round(ports_xy[..., 0]).to(dtype=torch.long)
        y = torch.round(ports_xy[..., 1]).to(dtype=torch.long)
        if int(stride) > 1:
            x = torch.div(x, int(stride), rounding_mode="floor")
            y = torch.div(y, int(stride), rounding_mode="floor")
        oob = (x < 0) | (x >= int(w)) | (y < 0) | (y >= int(h))
        x = x.clamp(0, int(w) - 1)
        y = y.clamp(0, int(h) - 1)
        return x, y, oob

    @staticmethod
    def _coarsen_blocked(blocked: torch.Tensor, stride: int) -> torch.Tensor:
        s = int(stride)
        if s <= 1:
            return blocked
        x = blocked.to(dtype=torch.float32).unsqueeze(1)
        y = F.max_pool2d(x, kernel_size=s, stride=s, ceil_mode=True)
        return y.squeeze(1).to(dtype=torch.bool)

    @staticmethod
    def _wavefront_distance_fields(
        *,
        free_map: torch.Tensor,     # [B,H,W] bool
        seed_xy: torch.Tensor,      # [B,S,2] float
        seed_mask: torch.Tensor,    # [B,S] bool
        max_iters: int,
    ) -> torch.Tensor:
        if free_map.dim() != 3:
            raise ValueError(f"free_map must be [B,H,W], got {tuple(free_map.shape)}")
        b, h, w = int(free_map.shape[0]), int(free_map.shape[1]), int(free_map.shape[2])
        dist = torch.full((b, h, w), -1, dtype=torch.int32, device=free_map.device)
        if b == 0:
            return dist

        frontier = torch.zeros((b, h, w), dtype=torch.bool, device=free_map.device)
        sx = torch.round(seed_xy[..., 0]).to(dtype=torch.long)
        sy = torch.round(seed_xy[..., 1]).to(dtype=torch.long)
        inb = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        valid = seed_mask.to(dtype=torch.bool, device=free_map.device) & inb
        if not bool(valid.any().item()):
            return dist

        sidx = sy.clamp(0, h - 1) * w + sx.clamp(0, w - 1)
        bidx = torch.arange(b, device=free_map.device, dtype=torch.long).view(b, 1).expand_as(sidx)
        flat = frontier.view(b, -1)
        flat[bidx[valid], sidx[valid]] = True
        frontier &= free_map
        if not bool(frontier.any().item()):
            return dist

        cap = int(max_iters)
        step = 0
        while bool(frontier.any().item()):
            dist[frontier] = int(step)
            nxt = torch.zeros_like(frontier)
            nxt[:, 1:, :] |= frontier[:, :-1, :]
            nxt[:, :-1, :] |= frontier[:, 1:, :]
            nxt[:, :, 1:] |= frontier[:, :, :-1]
            nxt[:, :, :-1] |= frontier[:, :, 1:]
            nxt &= free_map
            nxt &= (dist < 0)
            frontier = nxt
            step += 1
            if cap > 0 and step >= cap:
                break
        return dist

    def _body_indices_for_gid(
        self,
        *,
        state: "EnvState",
        gid: object,
    ) -> torch.Tensor:
        dev = state.maps.occ_invalid.device
        nodes_key = tuple(state.placed_nodes())
        key = (id(state.maps.occ_invalid), nodes_key, gid)
        cached = self._body_idx_cache.get(key, None)
        if isinstance(cached, torch.Tensor):
            return cached

        placement = state.placements.get(gid, None)
        if placement is None:
            out = torch.empty((0,), dtype=torch.long, device=dev)
            self._body_idx_cache[key] = out
            return out

        body = getattr(placement, "body_mask", None)
        if not isinstance(body, torch.Tensor):
            out = torch.empty((0,), dtype=torch.long, device=dev)
            self._body_idx_cache[key] = out
            return out
        body = body.to(device=dev, dtype=torch.bool)
        bh = int(body.shape[0])
        bw = int(body.shape[1])
        if bh <= 0 or bw <= 0:
            out = torch.empty((0,), dtype=torch.long, device=dev)
            self._body_idx_cache[key] = out
            return out

        x0 = int(getattr(placement, "x_bl", math.floor(float(getattr(placement, "min_x", 0.0)))))
        y0 = int(getattr(placement, "y_bl", math.floor(float(getattr(placement, "min_y", 0.0)))))
        H = int(state.maps.grid_height)
        W = int(state.maps.grid_width)

        if bool(getattr(placement, "is_rectangular", False)):
            xs = torch.arange(x0, x0 + bw, device=dev, dtype=torch.long).view(1, -1).expand(bh, bw)
            ys = torch.arange(y0, y0 + bh, device=dev, dtype=torch.long).view(-1, 1).expand(bh, bw)
        else:
            yy, xx = torch.where(body)
            xs = xx.to(dtype=torch.long) + int(x0)
            ys = yy.to(dtype=torch.long) + int(y0)

        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if not bool(valid.any().item()):
            out = torch.empty((0,), dtype=torch.long, device=dev)
            self._body_idx_cache[key] = out
            return out

        flat = (ys[valid] * int(W) + xs[valid]).to(dtype=torch.long).contiguous()
        if int(flat.numel()) > 1:
            flat = torch.unique(flat)
        self._body_idx_cache[key] = flat
        return flat

    def _blocked_batch_for_targets(
        self,
        *,
        blocked_base: torch.Tensor,      # [H,W] bool
        state: Optional["EnvState"],
        current_gid: Optional[object],
        target_gids: List[Optional[object]],
    ) -> torch.Tensor:
        b = int(len(target_gids))
        if b <= 0:
            return blocked_base.new_zeros((0, int(blocked_base.shape[0]), int(blocked_base.shape[1])), dtype=torch.bool)
        out = blocked_base.unsqueeze(0).expand(b, -1, -1).clone()
        if state is None:
            return out

        flat = out.view(b, -1)
        cur_idx = None
        if current_gid is not None and current_gid in state.placed:
            cur_idx = self._body_indices_for_gid(state=state, gid=current_gid)
        for i, tg in enumerate(target_gids):
            if cur_idx is not None and int(cur_idx.numel()) > 0:
                flat[i, cur_idx] = False
            if tg is None:
                continue
            if tg in state.placed:
                t_idx = self._body_indices_for_gid(state=state, gid=tg)
                if int(t_idx.numel()) > 0:
                    flat[i, t_idx] = False
        return out

    def _reduce_cost_with_lane(
        self,
        *,
        candidate_ports: torch.Tensor,          # [M,C,2]
        candidate_mask: Optional[torch.Tensor], # [M,C]
        target_ports: torch.Tensor,             # [T,P,2]
        target_mask: Optional[torch.Tensor],    # [T,P]
        target_weight: torch.Tensor,            # [T] or [M,T]
        blocked_base: torch.Tensor,             # [H,W]
        state: Optional["EnvState"],
        current_gid: Optional[object],
        target_gids: Optional[List[object]],
        c_modes: Optional[torch.Tensor] = None, # [M] bool (True=mean)
        t_modes: Optional[torch.Tensor] = None, # [T] bool (True=mean)
    ) -> torch.Tensor:
        m = int(candidate_ports.shape[0])
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=candidate_ports.device)
        t = int(target_ports.shape[0])
        if t == 0 or int(target_weight.numel()) == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)
        c = int(candidate_ports.shape[1])
        p = int(target_ports.shape[1])
        if c == 0 or p == 0:
            return torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)

        device = candidate_ports.device
        blocked_base = blocked_base.to(device=device, dtype=torch.bool)
        H0 = int(blocked_base.shape[0])
        W0 = int(blocked_base.shape[1])
        stride = max(1, int(self.distance_stride))
        unreachable = (
            float(self.unreachable_penalty)
            if self.unreachable_penalty is not None
            else float((H0 + W0) * 4.0)
        )

        weight_mt = FlowReward._as_weight_matrix(
            target_weight,
            m=m,
            t=t,
            device=device,
        )
        cand_valid = FlowReward._port_mask(candidate_ports, candidate_mask, name="candidate_mask")
        tgt_valid = FlowReward._port_mask(target_ports, target_mask, name="target_mask")
        t_has = tgt_valid.any(dim=1)
        if not bool(t_has.any().item()):
            return torch.zeros((m,), dtype=torch.float32, device=device)

        if target_gids is None or len(target_gids) != t:
            t_gid_list: List[Optional[object]] = [None] * t
        else:
            t_gid_list = [target_gids[i] for i in range(t)]

        t_is_mean = torch.zeros((t,), dtype=torch.bool, device=device) if t_modes is None else t_modes.to(
            device=device,
            dtype=torch.bool,
        )

        # Seed groups:
        # - target mean: one field per target port
        # - target min : one field per target (multi-source over valid ports)
        g_tidx: List[int] = []
        g_tgid: List[Optional[object]] = []
        g_seeds: List[torch.Tensor] = []
        g_is_mean: List[bool] = []
        mean_cnt = torch.zeros((t,), dtype=torch.int32, device=device)
        for ti in range(t):
            p_idx = torch.where(tgt_valid[ti])[0]
            if int(p_idx.numel()) == 0:
                continue
            if bool(t_is_mean[ti].item()):
                mean_cnt[ti] = int(p_idx.numel())
                for pj_t in p_idx:
                    pj = int(pj_t.item())
                    g_tidx.append(ti)
                    g_tgid.append(t_gid_list[ti])
                    g_seeds.append(target_ports[ti, pj:pj + 1, :])
                    g_is_mean.append(True)
            else:
                g_tidx.append(ti)
                g_tgid.append(t_gid_list[ti])
                g_seeds.append(target_ports[ti, p_idx, :])
                g_is_mean.append(False)

        g_total = len(g_tidx)
        if g_total == 0:
            return torch.zeros((m,), dtype=torch.float32, device=device)

        reduced_cp = torch.full((m, t, c), float("inf"), dtype=torch.float32, device=device)
        mean_acc = torch.zeros((m, t, c), dtype=torch.float32, device=device)

        # Candidate port indices on (possibly) coarse grid.
        Hc = (H0 + stride - 1) // stride
        Wc = (W0 + stride - 1) // stride
        cx, cy, c_oob = self._ports_to_grid(
            candidate_ports,
            stride=stride,
            h=Hc,
            w=Wc,
        )
        c_lin = cy * int(Wc) + cx
        c_lin_flat = c_lin.view(1, -1)
        c_oob_m1c = c_oob.view(m, 1, c)

        chunk = max(1, int(self.target_chunk))
        for g0 in range(0, g_total, chunk):
            g1 = min(g_total, g0 + chunk)
            gids = list(range(g0, g1))
            b = len(gids)
            chunk_target_gids = [g_tgid[g] for g in gids]

            blocked_b = self._blocked_batch_for_targets(
                blocked_base=blocked_base,
                state=state,
                current_gid=current_gid,
                target_gids=chunk_target_gids,
            )
            blocked_b = self._coarsen_blocked(blocked_b, stride)
            free_b = ~blocked_b
            Hb = int(free_b.shape[1])
            Wb = int(free_b.shape[2])

            smax = max(int(g_seeds[g].shape[0]) for g in gids)
            seed_xy = torch.zeros((b, smax, 2), dtype=torch.float32, device=device)
            seed_mask = torch.zeros((b, smax), dtype=torch.bool, device=device)
            for bi, g in enumerate(gids):
                seeds = g_seeds[g].to(device=device, dtype=torch.float32).view(-1, 2)
                n = int(seeds.shape[0])
                sx, sy, s_oob = self._ports_to_grid(
                    seeds,
                    stride=stride,
                    h=Hb,
                    w=Wb,
                )
                seed_xy[bi, :n, 0] = sx.to(dtype=torch.float32)
                seed_xy[bi, :n, 1] = sy.to(dtype=torch.float32)
                seed_mask[bi, :n] = (~s_oob)

            dist_b = self._wavefront_distance_fields(
                free_map=free_b,
                seed_xy=seed_xy,
                seed_mask=seed_mask,
                max_iters=int(self.max_wave_iters),
            )  # [B,Hb,Wb], -1 unreachable
            dist_flat = dist_b.view(b, -1)
            idx = c_lin_flat.expand(b, -1)  # [B,M*C]
            d = dist_flat.gather(1, idx).view(b, m, c).permute(1, 0, 2)  # [M,B,C]
            d = torch.where(
                d >= 0,
                d.to(dtype=torch.float32) * float(stride),
                torch.full_like(d, float(unreachable), dtype=torch.float32),
            )
            d = torch.where(
                c_oob_m1c,
                torch.full_like(d, float(unreachable), dtype=torch.float32),
                d,
            )

            for bi, g in enumerate(gids):
                ti = g_tidx[g]
                if g_is_mean[g]:
                    mean_acc[:, ti, :] += d[:, bi, :]
                else:
                    reduced_cp[:, ti, :] = d[:, bi, :]

        mean_idx = torch.where(mean_cnt > 0)[0]
        for ti_t in mean_idx:
            ti = int(ti_t.item())
            reduced_cp[:, ti, :] = mean_acc[:, ti, :] / float(mean_cnt[ti].item())

        valid_c = cand_valid[:, None, :].expand(m, t, c)
        has_valid = valid_c.any(dim=2)
        masked_inf = reduced_cp.masked_fill(~valid_c, float("inf"))
        min_c = masked_inf.min(dim=2).values

        c_all_min = c_modes is None or not c_modes.any().item()
        if c_all_min:
            reduced_mt = torch.where(has_valid, min_c, torch.zeros_like(min_c))
        else:
            c_mode = c_modes.to(device=device, dtype=torch.bool).view(-1, 1)
            reduced_zero = reduced_cp.masked_fill(~valid_c, 0.0)
            c_count = valid_c.sum(dim=2).clamp(min=1).to(dtype=torch.float32)
            mean_c = reduced_zero.sum(dim=2) / c_count
            if c_mode.all().item():
                raw = mean_c
            else:
                raw = torch.where(c_mode, mean_c, min_c)
            reduced_mt = torch.where(has_valid, raw, torch.zeros_like(raw))

        reduced_mt = torch.where(
            t_has.view(1, -1),
            reduced_mt,
            torch.zeros_like(reduced_mt),
        )
        return (reduced_mt * weight_mt).sum(dim=1)

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
        state: Optional["EnvState"] = None,
        placed_nodes: Optional[List["GroupId"]] = None,
    ) -> torch.Tensor:
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        p = int(placed_en.shape[0])
        if p == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if int(placed_en.shape[1]) == 0 or int(placed_ex.shape[1]) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)

        if placed_nodes is None and state is not None:
            placed_nodes = state.placed_nodes()
        if placed_nodes is None or len(placed_nodes) != p:
            return FlowReward().score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_modes=exit_modes,
                entry_modes=entry_modes,
            )

        if state is not None:
            blocked_base = state.maps.static_invalid | state.maps.occ_invalid
        elif route_blocked is not None:
            blocked_base = route_blocked.to(dtype=torch.bool, device=placed_en.device)
        else:
            return FlowReward().score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_modes=exit_modes,
                entry_modes=entry_modes,
            )

        total = torch.tensor(0.0, dtype=torch.float32, device=placed_en.device)
        for src in range(p):
            row_w = flow_w[src:src + 1, :]
            if not bool((row_w != 0).any().item()):
                continue
            per = self._reduce_cost_with_lane(
                candidate_ports=placed_ex[src:src + 1],
                candidate_mask=placed_exits_mask[src:src + 1] if placed_exits_mask is not None else None,
                target_ports=placed_en,
                target_mask=placed_entries_mask,
                target_weight=row_w,
                blocked_base=blocked_base,
                state=state,
                current_gid=placed_nodes[src],
                target_gids=placed_nodes,
                c_modes=exit_modes[src:src + 1] if exit_modes is not None else None,
                t_modes=entry_modes,
            )
            total = total + per[0]
        return total

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
        state: Optional["EnvState"] = None,
        current_gid: Optional["GroupId"] = None,
        placed_nodes: Optional[List["GroupId"]] = None,
    ) -> torch.Tensor:
        cand_entries = FlowReward._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = FlowReward._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        m = int(cand_entries.shape[0])
        device = cand_entries.device
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=device)

        if placed_nodes is None and state is not None:
            placed_nodes = state.placed_nodes()
        if placed_nodes is not None and len(placed_nodes) != int(placed_en.shape[0]):
            placed_nodes = None

        if state is not None:
            blocked_base = state.maps.static_invalid | state.maps.occ_invalid
        elif route_blocked is not None:
            blocked_base = route_blocked.to(dtype=torch.bool, device=device)
        else:
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

        out_term = torch.zeros((m,), dtype=torch.float32, device=device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=device)

        if has_out:
            out_rows = torch.where(out_idx)[0]
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            out_gids = [placed_nodes[int(i.item())] for i in out_rows] if placed_nodes is not None else None
            out_term = self._reduce_cost_with_lane(
                candidate_ports=cand_exits,
                candidate_mask=candidate_exits_mask,
                target_ports=out_entries,
                target_mask=out_entries_mask,
                target_weight=out_w,
                blocked_base=blocked_base,
                state=state,
                current_gid=current_gid,
                target_gids=out_gids,
                c_modes=FlowReward._mode_tensor(c_exit_mode, m, device),
                t_modes=t_entry_modes[out_idx] if t_entry_modes is not None else None,
            )

        if has_in:
            in_rows = torch.where(in_idx)[0]
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            in_gids = [placed_nodes[int(i.item())] for i in in_rows] if placed_nodes is not None else None
            in_term = self._reduce_cost_with_lane(
                candidate_ports=cand_entries,
                candidate_mask=candidate_entries_mask,
                target_ports=in_exits,
                target_mask=in_exits_mask,
                target_weight=in_w,
                blocked_base=blocked_base,
                state=state,
                current_gid=current_gid,
                target_gids=in_gids,
                c_modes=FlowReward._mode_tensor(c_entry_mode, m, device),
                t_modes=t_exit_modes[in_idx] if t_exit_modes is not None else None,
            )

        return out_term + in_term
