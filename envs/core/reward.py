from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class RewardContext:
    placed_count: int
    cur_min_x: float
    cur_max_x: float
    cur_min_y: float
    cur_max_y: float
    placed_entries: torch.Tensor      # [P, Emax, 2]
    placed_exits: torch.Tensor        # [P, Xmax, 2]
    placed_entries_mask: torch.Tensor # [P, Emax]
    placed_exits_mask: torch.Tensor   # [P, Xmax]
    # Optional — only required for delta() (delta_cost), not for score() (cost).
    flow_w: Optional[torch.Tensor] = None   # [P,P]   — score() only
    w_out: Optional[torch.Tensor] = None    # [P]     — delta() only
    w_in: Optional[torch.Tensor] = None     # [P]     — delta() only
    # Optional maps for extended rewards.
    route_blocked: Optional[torch.Tensor] = None        # [H,W] bool
    placed_cell_occupied: Optional[torch.Tensor] = None # [Gh,Gw] bool

def _require(x: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if x is None:
        raise ValueError(f"{name} is required but None")
    return x


@dataclass
class CandidateBatch:
    """Reward input batch for candidate placements."""

    entries: Optional[torch.Tensor] = None       # [M, Emax, 2]
    exits: Optional[torch.Tensor] = None         # [M, Xmax, 2]
    min_x: Optional[torch.Tensor] = None         # [M]
    max_x: Optional[torch.Tensor] = None         # [M]
    min_y: Optional[torch.Tensor] = None         # [M]
    max_y: Optional[torch.Tensor] = None         # [M]
    entries_mask: Optional[torch.Tensor] = None  # [M, Emax]
    exits_mask: Optional[torch.Tensor] = None    # [M, Xmax]

    @classmethod
    def from_feature_map(
        cls,
        feature_map: Dict[str, torch.Tensor],
        *,
        required: Optional[set[str]] = None,
        device: Optional[torch.device] = None,
    ) -> "CandidateBatch":
        valid_keys = {
            "entries",
            "exits",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "entries_mask",
            "exits_mask",
        }
        unknown = set(feature_map.keys()) - valid_keys
        if unknown:
            raise ValueError(f"unknown candidate feature keys: {sorted(unknown)}")

        kwargs: Dict[str, Optional[torch.Tensor]] = {}
        for key in valid_keys:
            val = feature_map.get(key, None)
            if val is None:
                kwargs[key] = None
                continue
            if not torch.is_tensor(val):
                raise TypeError(f"feature {key!r} must be a torch.Tensor")
            kwargs[key] = val.to(device=device) if device is not None else val

        out = cls(**kwargs)
        out.validate(required=required)
        return out

    def validate(self, *, required: Optional[set[str]] = None) -> None:
        required_keys = set(required or ())
        for key in required_keys:
            if not hasattr(self, key):
                raise ValueError(f"unknown required candidate field: {key!r}")
            if getattr(self, key) is None:
                raise ValueError(f"required candidate field missing: {key!r}")

        entries = self.entries
        exits = self.exits
        if entries is not None:
            if entries.dim() != 3 or int(entries.shape[-1]) != 2:
                raise ValueError(f"entries must be [M,E,2], got {tuple(entries.shape)}")
        if exits is not None:
            if exits.dim() != 3 or int(exits.shape[-1]) != 2:
                raise ValueError(f"exits must be [M,X,2], got {tuple(exits.shape)}")

        if self.entries_mask is not None:
            if entries is None:
                raise ValueError("entries_mask requires entries")
            if tuple(self.entries_mask.shape) != (int(entries.shape[0]), int(entries.shape[1])):
                raise ValueError(
                    f"entries_mask must be {(int(entries.shape[0]), int(entries.shape[1]))}, got {tuple(self.entries_mask.shape)}"
                )
        if self.exits_mask is not None:
            if exits is None:
                raise ValueError("exits_mask requires exits")
            if tuple(self.exits_mask.shape) != (int(exits.shape[0]), int(exits.shape[1])):
                raise ValueError(
                    f"exits_mask must be {(int(exits.shape[0]), int(exits.shape[1]))}, got {tuple(self.exits_mask.shape)}"
                )

        m: Optional[int] = None
        for name in ("entries", "exits", "min_x", "max_x", "min_y", "max_y"):
            t = getattr(self, name)
            if t is None:
                continue
            if name in {"min_x", "max_x", "min_y", "max_y"} and t.dim() != 1:
                raise ValueError(f"{name} must be [M], got {tuple(t.shape)}")
            cur_m = int(t.shape[0])
            if m is None:
                m = cur_m
            elif m != cur_m:
                raise ValueError(f"inconsistent candidate batch length: {name} has M={cur_m}, expected M={m}")


@dataclass
class FlowReward:
    def required(self) -> set[str]:
        return {"entries", "exits"}

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
    def _masked_pair_min(
        cost: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce [M,T,C,P] cost to [M,T] with invalid pairs mapped to zero.

        Always returns (values [M,T], c_idx [M,T], p_idx [M,T]).
        c_idx: best candidate port index (C dim), p_idx: best target port index (P dim).
        Indices are meaningful only where has_valid[m,t] is True.
        PyTorch min() already computes indices in the same kernel pass — gather is the only extra op.
        """
        if cost.dim() != 4 or valid.dim() != 4:
            raise ValueError("cost/valid must be rank-4 tensors [M,T,C,P]")
        if tuple(cost.shape) != tuple(valid.shape):
            raise ValueError(f"cost and valid shape mismatch: {tuple(cost.shape)} vs {tuple(valid.shape)}")
        has_valid = valid.any(dim=3).any(dim=2)  # [M,T]
        masked = cost.masked_fill(~valid, float("inf"))
        min_p = masked.min(dim=3)               # values [M,T,C], indices [M,T,C]
        min_cp = min_p.values.min(dim=2)        # values [M,T],   indices [M,T]
        result = torch.where(has_valid, min_cp.values, torch.zeros_like(min_cp.values))
        c_idx = min_cp.indices                  # [M,T]: best candidate port
        p_idx = min_p.indices.gather(2, c_idx.unsqueeze(2)).squeeze(2)  # [M,T]: best target port
        return result, c_idx, p_idx

    def _min_distance(
        self,
        *,
        candidate_ports: torch.Tensor,          # [M,C,2]
        candidate_mask: Optional[torch.Tensor], # [M,C]
        target_ports: torch.Tensor,             # [T,P,2]
        target_mask: Optional[torch.Tensor],    # [T,P]
        target_weight: torch.Tensor,            # [T] or [M,T]
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
        dist_min, c_idx, p_idx = self._masked_pair_min(dist, valid)
        return (dist_min * weight_mt).sum(dim=1), c_idx, p_idx

    def score(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        flow_w: torch.Tensor,
        return_argmin: bool = False,
    ):
        """Compute absolute flow score for the placed state (tensor-only).

        If return_argmin=True, returns (scalar, c_idx [P,P], p_idx [P,P]) where:
          c_idx[m,t] = exit port index of placed facility m toward facility t
          p_idx[m,t] = entry port index of placed facility t from facility m
        Indices are valid only for (m,t) pairs with nonzero flow_w.
        """
        placed_en = self._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = self._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        zero = torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if placed_en.shape[0] == 0:
            return (zero, None, None) if return_argmin else zero
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            return (zero, None, None) if return_argmin else zero
        per_src, c_idx, p_idx = self._min_distance(
            candidate_ports=placed_ex,
            candidate_mask=placed_exits_mask,
            target_ports=placed_en,
            target_mask=placed_entries_mask,
            target_weight=flow_w,
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
        return_argmin: bool = False,
    ):
        """Compute incremental flow cost for M candidate placements.

        If return_argmin=True, returns a tuple:
          (delta [M],
           out_argmin: (c_idx [M,T_out], p_idx [M,T_out], out_idx BoolTensor),
           in_argmin:  (c_idx [M,T_in],  p_idx [M,T_in],  in_idx BoolTensor))
        out_idx / in_idx are the boolean masks over placed facilities used to
        filter w_out / w_in; they allow the caller to recover which placed
        facility each column corresponds to.  None if the respective term is absent.
        """
        cand_entries = self._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = self._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = self._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = self._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        w_out_t = w_out.view(-1)
        w_in_t = w_in.view(-1)
        m = int(cand_entries.shape[0])

        # Keep only placed facilities that are actual flow targets for this candidate.
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
                return torch.zeros((m,), dtype=torch.float32, device=cand_entries.device), None, None
            return torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)

        out_term = torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)
        out_am = None
        in_am = None

        if has_out:
            # OUT term: candidate exit -> placed entry (weighted).
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            out_term, oc_idx, op_idx = self._min_distance(
                candidate_ports=cand_exits,
                candidate_mask=candidate_exits_mask,
                target_ports=out_entries,
                target_mask=out_entries_mask,
                target_weight=out_w,
            )
            if return_argmin:
                out_am = (oc_idx, op_idx, out_idx)
        if has_in:
            # IN term: placed exit -> candidate entry (weighted).
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            in_term, ic_idx, ip_idx = self._min_distance(
                candidate_ports=cand_entries,
                candidate_mask=candidate_entries_mask,
                target_ports=in_exits,
                target_mask=in_exits_mask,
                target_weight=in_w,
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
        return {"entries", "exits"}

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

    def _min_cost_with_collision(
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
        cost_min, _, _ = FlowReward._masked_pair_min(cost, valid)
        return (cost_min * weight_mt).sum(dim=1)

    def score(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        flow_w: torch.Tensor,
        route_blocked: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Fallback to pure flow when blocked map is absent.
        if route_blocked is None:
            return FlowReward().score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
            )
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        if placed_en.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)

        blocked = route_blocked.to(dtype=torch.bool, device=placed_en.device)
        row_ps, col_ps = self._prefix(blocked)
        per_src = self._min_cost_with_collision(
            candidate_ports=placed_ex,
            candidate_mask=placed_exits_mask,
            target_ports=placed_en,
            target_mask=placed_entries_mask,
            target_weight=flow_w,
            blocked=blocked,
            row_ps=row_ps,
            col_ps=col_ps,
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
    ) -> torch.Tensor:
        # Fallback to pure flow when blocked map is absent.
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
            )
        cand_entries = FlowReward._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = FlowReward._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        m = int(cand_entries.shape[0])
        if m == 0:
            return torch.zeros((0,), dtype=torch.float32, device=cand_entries.device)

        w_out_t = w_out.view(-1)
        w_in_t = w_in.view(-1)
        # Keep only placed facilities that have a non-zero flow relation.
        out_idx = (w_out_t != 0)
        in_idx = (w_in_t != 0)
        if placed_entries_mask is not None:
            out_idx = out_idx & placed_entries_mask.any(dim=1)
        if placed_exits_mask is not None:
            in_idx = in_idx & placed_exits_mask.any(dim=1)
        has_out = bool(out_idx.any().item())
        has_in = bool(in_idx.any().item())
        if not (has_out or has_in):
            return torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)

        blocked = route_blocked.to(dtype=torch.bool, device=cand_entries.device)
        row_ps, col_ps = self._prefix(blocked)

        out_term = torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=cand_entries.device)

        if has_out:
            # OUT term with collision-aware L-shape routing cost.
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            out_term = self._min_cost_with_collision(
                candidate_ports=cand_exits,
                candidate_mask=candidate_exits_mask,
                target_ports=out_entries,
                target_mask=out_entries_mask,
                target_weight=out_w,
                blocked=blocked,
                row_ps=row_ps,
                col_ps=col_ps,
            )

        if has_in:
            # IN term with collision-aware L-shape routing cost.
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            in_term = self._min_cost_with_collision(
                candidate_ports=cand_entries,
                candidate_mask=candidate_entries_mask,
                target_ports=in_exits,
                target_mask=in_exits_mask,
                target_weight=in_w,
                blocked=blocked,
                row_ps=row_ps,
                col_ps=col_ps,
            )

        return out_term + in_term


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

    def score(self, *, placed_cell_occupied: torch.Tensor) -> torch.Tensor:
        if placed_cell_occupied is None:
            raise ValueError("placed_cell_occupied is required for GridOccupancyReward.score")
        return placed_cell_occupied.to(dtype=torch.float32).sum()

    def delta(
        self,
        *,
        placed_cell_occupied: torch.Tensor,
        candidate_min_x: torch.Tensor,
        candidate_max_x: torch.Tensor,
        candidate_min_y: torch.Tensor,
        candidate_max_y: torch.Tensor,
    ) -> torch.Tensor:
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
        new_occ = cand & (~base.view(1, gh, gw))
        return new_occ.to(dtype=torch.float32).sum(dim=(1, 2))


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
    ) -> torch.Tensor:
        """Compute absolute compactness score (HPWL of placed bbox, tensor-only)."""
        if placed_count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=min_x.device)
        return 0.5 * ((max_x - min_x) + (max_y - min_y))

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
    ) -> torch.Tensor:
        if placed_count == 0:
            return 0.5 * ((candidate_max_x - candidate_min_x) + (candidate_max_y - candidate_min_y))

        # scalar tensors broadcast against [M] candidates — avoids [M] allocation x4
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
        return new_hpwl - cur_hpwl


@dataclass
class RewardComposer:
    components: Dict[str, object]
    weights: Dict[str, float]

    def required(self) -> set[str]:
        needed: set[str] = set()
        for comp in self.components.values():
            if comp is not None and hasattr(comp, "required"):
                needed |= set(comp.required())
        return needed

    def score(self, ctx: RewardContext) -> torch.Tensor:
        device = ctx.placed_entries.device
        total = torch.tensor(0.0, dtype=torch.float32, device=device)

        flow = self.components.get("flow", None)
        if flow is not None:
            w = float(self.weights.get("flow", 1.0))
            flow_w = _require(ctx.flow_w, "ctx.flow_w")
            total = total + w * flow.score(
                placed_entries=ctx.placed_entries,
                placed_exits=ctx.placed_exits,
                placed_entries_mask=ctx.placed_entries_mask,
                placed_exits_mask=ctx.placed_exits_mask,
                flow_w=flow_w,
            )

        flow_collision = self.components.get("flow_collision", None)
        if flow_collision is not None:
            w = float(self.weights.get("flow_collision", 1.0))
            flow_w = _require(ctx.flow_w, "ctx.flow_w")
            total = total + w * flow_collision.score(
                placed_entries=ctx.placed_entries,
                placed_exits=ctx.placed_exits,
                placed_entries_mask=ctx.placed_entries_mask,
                placed_exits_mask=ctx.placed_exits_mask,
                flow_w=flow_w,
                route_blocked=ctx.route_blocked,
            )

        area = self.components.get("area", None)
        if area is not None:
            w = float(self.weights.get("area", 1.0))
            min_x_t = torch.tensor(float(ctx.cur_min_x), dtype=torch.float32, device=device)
            max_x_t = torch.tensor(float(ctx.cur_max_x), dtype=torch.float32, device=device)
            min_y_t = torch.tensor(float(ctx.cur_min_y), dtype=torch.float32, device=device)
            max_y_t = torch.tensor(float(ctx.cur_max_y), dtype=torch.float32, device=device)
            total = total + w * area.score(
                placed_count=ctx.placed_count,
                min_x=min_x_t,
                max_x=max_x_t,
                min_y=min_y_t,
                max_y=max_y_t,
            )

        grid_occ = self.components.get("grid_occupancy", None)
        if grid_occ is not None:
            w = float(self.weights.get("grid_occupancy", 1.0))
            total = total + w * grid_occ.score(
                placed_cell_occupied=ctx.placed_cell_occupied,
            )
        return total

    def delta(self, ctx: RewardContext, batch: CandidateBatch) -> torch.Tensor:
        batch.validate(required=self.required())

        m = 0
        device = ctx.placed_entries.device
        for name in ("entries", "exits", "min_x", "max_x", "min_y", "max_y"):
            t = getattr(batch, name)
            if t is None:
                continue
            m = int(t.shape[0])
            device = t.device
            break

        total = torch.zeros((m,), dtype=torch.float32, device=device)
        flow = self.components.get("flow", None)
        flow_collision = self.components.get("flow_collision", None)
        area = self.components.get("area", None)
        grid_occ = self.components.get("grid_occupancy", None)

        entries = batch.entries
        exits = batch.exits
        if flow is not None or flow_collision is not None:
            entries = _require(entries, "batch.entries")
            exits = _require(exits, "batch.exits")

        if flow is not None:
            w = float(self.weights.get("flow", 1.0))
            total = total + w * flow.delta(
                placed_entries=ctx.placed_entries,
                placed_exits=ctx.placed_exits,
                placed_entries_mask=ctx.placed_entries_mask,
                placed_exits_mask=ctx.placed_exits_mask,
                w_out=_require(ctx.w_out, "ctx.w_out"),
                w_in=_require(ctx.w_in, "ctx.w_in"),
                candidate_entries=entries,
                candidate_exits=exits,
                candidate_entries_mask=batch.entries_mask,
                candidate_exits_mask=batch.exits_mask,
            )
        if flow_collision is not None:
            w = float(self.weights.get("flow_collision", 1.0))
            total = total + w * flow_collision.delta(
                placed_entries=ctx.placed_entries,
                placed_exits=ctx.placed_exits,
                placed_entries_mask=ctx.placed_entries_mask,
                placed_exits_mask=ctx.placed_exits_mask,
                w_out=_require(ctx.w_out, "ctx.w_out"),
                w_in=_require(ctx.w_in, "ctx.w_in"),
                candidate_entries=entries,
                candidate_exits=exits,
                candidate_entries_mask=batch.entries_mask,
                candidate_exits_mask=batch.exits_mask,
                route_blocked=ctx.route_blocked,
            )

        min_x = batch.min_x
        max_x = batch.max_x
        min_y = batch.min_y
        max_y = batch.max_y
        if area is not None or grid_occ is not None:
            min_x = _require(min_x, "batch.min_x")
            max_x = _require(max_x, "batch.max_x")
            min_y = _require(min_y, "batch.min_y")
            max_y = _require(max_y, "batch.max_y")

        if area is not None:
            w = float(self.weights.get("area", 1.0))
            total = total + w * area.delta(
                placed_count=ctx.placed_count,
                cur_min_x=ctx.cur_min_x,
                cur_max_x=ctx.cur_max_x,
                cur_min_y=ctx.cur_min_y,
                cur_max_y=ctx.cur_max_y,
                candidate_min_x=min_x,
                candidate_max_x=max_x,
                candidate_min_y=min_y,
                candidate_max_y=max_y,
            )
        if grid_occ is not None:
            w = float(self.weights.get("grid_occupancy", 1.0))
            total = total + w * grid_occ.delta(
                placed_cell_occupied=ctx.placed_cell_occupied,
                candidate_min_x=min_x,
                candidate_max_x=max_x,
                candidate_min_y=min_y,
                candidate_max_y=max_y,
            )
        return total
