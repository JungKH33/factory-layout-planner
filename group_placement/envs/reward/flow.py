from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
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
    def _select_k_tensor(select_k: int, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """Convert scalar ``select_k`` to ``[n]`` int tensor. Returns None for ``1`` fast-path."""
        k = int(select_k)
        if k == 1:
            return None
        if k < 1:
            raise ValueError(f"select_k must be >= 1, got {select_k!r}")
        return torch.full((n,), k, dtype=torch.int32, device=device)

    @staticmethod
    def _normalize_k_tensor(
        k: Optional[torch.Tensor],
        *,
        n: int,
        name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if k is None:
            return None
        out = k.to(device=device, dtype=torch.int32).view(-1)
        if int(out.numel()) != int(n):
            raise ValueError(f"{name} must have shape ({n},), got {tuple(out.shape)}")
        if not bool((out >= 1).all().item()):
            bad = out[out < 1][0].item()
            raise ValueError(f"{name} must be >= 1, got {bad!r}")
        if bool((out == 1).all().item()):
            return None
        return out

    @staticmethod
    def _masked_pair_reduce(
        cost: torch.Tensor,
        valid: torch.Tensor,
        c_k: Optional[torch.Tensor] = None,
        t_k: Optional[torch.Tensor] = None,
        return_detail: bool = False,
    ):
        """Reduce [M,T,C,P] to [M,T] using per-facility k-port averaging.

        - ``k=1``: min reduction (closest port)
        - ``k>1``: average over top-k closest valid ports, clamped per
          candidate to the actual valid count (so ``k >= max_ports`` is
          equivalent to averaging over all valid ports).
        """
        if cost.dim() != 4 or valid.dim() != 4:
            raise ValueError("cost/valid must be rank-4 tensors [M,T,C,P]")
        if tuple(cost.shape) != tuple(valid.shape):
            raise ValueError(f"cost and valid shape mismatch: {tuple(cost.shape)} vs {tuple(valid.shape)}")

        m = int(cost.shape[0])
        t = int(cost.shape[1])
        device = cost.device
        c_k_t = FlowReward._normalize_k_tensor(c_k, n=m, name="c_k", device=device)
        t_k_t = FlowReward._normalize_k_tensor(t_k, n=t, name="t_k", device=device)
        has_valid = valid.any(dim=3).any(dim=2)  # [M,T]

        # Always compute min over P for argmin indices
        masked_inf = cost.masked_fill(~valid, float("inf"))
        min_p = masked_inf.min(dim=3)  # values [M,T,C], indices [M,T,C]
        c_valid = valid.any(dim=3)  # [M,T,C]

        # --- P-dim reduction (target ports) ---
        # k=1 → min, k>1 → average over top-k closest valid ports (clamped to
        # the actual valid count, so "use all" is expressed as any k >= max ports).
        t_all_one = t_k_t is None or bool((t_k_t == 1).all().item())
        selected_p_mask = torch.zeros_like(valid) if bool(return_detail) else None
        if t_all_one:
            reduced_p = min_p.values
            if selected_p_mask is not None:
                selected_p_mask.scatter_(3, min_p.indices.unsqueeze(3), True)
                selected_p_mask &= c_valid.unsqueeze(3)
        else:
            sorted_p_raw = masked_inf.sort(dim=3)
            sorted_p = sorted_p_raw.values  # inf values sort to tail
            cumsum_p = sorted_p.cumsum(dim=3)
            count_p = valid.sum(dim=3).to(dtype=torch.int32)  # [M,T,C]
            p_has = count_p > 0
            req_t = t_k_t.view(1, -1, 1).expand_as(count_p)
            eff_p = torch.minimum(req_t, count_p)
            # gather at (eff_p - 1) always points to the last valid port in the
            # sorted order, so inf tail values are never included in the sum.
            eff_p_safe = torch.where(p_has, eff_p.clamp(min=1), torch.ones_like(eff_p))
            gather_idx = (eff_p_safe - 1).to(dtype=torch.long).unsqueeze(3)
            sum_p = cumsum_p.gather(3, gather_idx).squeeze(3)
            avg_p = sum_p / eff_p_safe.to(dtype=torch.float32)
            reduced_p = torch.where(p_has, avg_p, torch.zeros_like(avg_p))
            if selected_p_mask is not None:
                sorted_p_idx = sorted_p_raw.indices
                p_rank = torch.arange(int(cost.shape[3]), dtype=torch.int32, device=device).view(1, 1, 1, -1)
                sorted_take = (p_rank < eff_p_safe.unsqueeze(3)) & p_has.unsqueeze(3)
                selected_p_mask.scatter_(3, sorted_p_idx, sorted_take)
                selected_p_mask &= c_valid.unsqueeze(3)

        reduced_p_inf = reduced_p.masked_fill(~c_valid, float("inf"))
        min_c = reduced_p_inf.min(dim=2)  # values [M,T], indices [M,T]

        # --- C-dim reduction (candidate ports) ---
        c_all_one = c_k_t is None or bool((c_k_t == 1).all().item())
        selected_c_mask = torch.zeros_like(c_valid) if bool(return_detail) else None
        if c_all_one:
            result = torch.where(has_valid, min_c.values, torch.zeros_like(min_c.values))
            if selected_c_mask is not None:
                selected_c_mask.scatter_(2, min_c.indices.unsqueeze(2), True)
                selected_c_mask &= has_valid.unsqueeze(2)
        else:
            sorted_c_raw = reduced_p_inf.sort(dim=2)
            sorted_c = sorted_c_raw.values  # inf values sort to tail
            cumsum_c = sorted_c.cumsum(dim=2)
            count_c = c_valid.sum(dim=2).to(dtype=torch.int32)  # [M,T]
            c_has = count_c > 0
            req_c = c_k_t.view(-1, 1).expand_as(count_c)
            eff_c = torch.minimum(req_c, count_c)
            eff_c_safe = torch.where(c_has, eff_c.clamp(min=1), torch.ones_like(eff_c))
            gather_idx = (eff_c_safe - 1).to(dtype=torch.long).unsqueeze(2)
            sum_c = cumsum_c.gather(2, gather_idx).squeeze(2)
            avg_c = sum_c / eff_c_safe.to(dtype=torch.float32)
            result = torch.where(has_valid, avg_c, torch.zeros_like(avg_c))
            if selected_c_mask is not None:
                sorted_c_idx = sorted_c_raw.indices
                c_rank = torch.arange(int(cost.shape[2]), dtype=torch.int32, device=device).view(1, 1, -1)
                sorted_take = (c_rank < eff_c_safe.unsqueeze(2)) & c_has.unsqueeze(2)
                selected_c_mask.scatter_(2, sorted_c_idx, sorted_take)
                selected_c_mask &= c_valid

        c_idx = min_c.indices
        p_idx = min_p.indices.gather(2, c_idx.unsqueeze(2)).squeeze(2)
        if return_detail:
            return result, c_idx, p_idx, {
                "selected_c_mask": selected_c_mask if selected_c_mask is not None else torch.zeros_like(c_valid),
                "selected_p_mask": selected_p_mask if selected_p_mask is not None else torch.zeros_like(valid),
            }
        return result, c_idx, p_idx

    def _reduce_distance(
        self,
        *,
        candidate_ports: torch.Tensor,          # [M,C,2]
        candidate_mask: Optional[torch.Tensor], # [M,C]
        target_ports: torch.Tensor,             # [T,P,2]
        target_mask: Optional[torch.Tensor],    # [T,P]
        target_weight: torch.Tensor,            # [T] or [M,T]
        c_k: Optional[torch.Tensor] = None, # [M] int (1|min, -1|all, k>1 top-k)
        t_k: Optional[torch.Tensor] = None, # [T] int (1|min, -1|all, k>1 top-k)
        return_detail: bool = False,
    ):
        """Returns ([M], c_idx [M,T], p_idx [M,T]).

        c_idx / p_idx are None when result is a zero early-exit (no candidates / targets).
        """
        m = int(candidate_ports.shape[0])
        if m == 0:
            zero = torch.zeros((0,), dtype=torch.float32, device=candidate_ports.device)
            if return_detail:
                return zero, None, None, {
                    "edge_distance": torch.zeros((0, 0), dtype=torch.float32, device=candidate_ports.device),
                    "selected_c_mask": None,
                    "selected_p_mask": None,
                }
            return zero, None, None
        t = int(target_ports.shape[0])
        if t == 0 or int(target_weight.numel()) == 0:
            zero = torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)
            if return_detail:
                return zero, None, None, {
                    "edge_distance": torch.zeros((m, t), dtype=torch.float32, device=candidate_ports.device),
                    "selected_c_mask": None,
                    "selected_p_mask": None,
                }
            return zero, None, None
        if int(candidate_ports.shape[1]) == 0 or int(target_ports.shape[1]) == 0:
            zero = torch.zeros((m,), dtype=torch.float32, device=candidate_ports.device)
            if return_detail:
                return zero, None, None, {
                    "edge_distance": torch.zeros((m, t), dtype=torch.float32, device=candidate_ports.device),
                    "selected_c_mask": None,
                    "selected_p_mask": None,
                }
            return zero, None, None

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
        c_k_t = self._normalize_k_tensor(c_k, n=m, name="c_k", device=candidate_ports.device)
        t_k_t = self._normalize_k_tensor(t_k, n=t, name="t_k", device=candidate_ports.device)
        need_selection = (c_k_t is not None) or (t_k_t is not None)
        valid = cand_mask[:, None, :, None] & tgt_mask[None, :, None, :]  # [M,T,C,P]
        if return_detail and need_selection:
            dist_reduced, c_idx, p_idx, detail = self._masked_pair_reduce(
                dist, valid, c_k_t, t_k_t, return_detail=True,
            )
        else:
            dist_reduced, c_idx, p_idx = self._masked_pair_reduce(dist, valid, c_k_t, t_k_t)
            detail = {
                "selected_c_mask": None,
                "selected_p_mask": None,
            }
        out = (dist_reduced * weight_mt).sum(dim=1)
        if return_detail:
            detail["edge_distance"] = dist_reduced
            return out, c_idx, p_idx, detail
        return out, c_idx, p_idx

    def score(
        self,
        *,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: Optional[torch.Tensor],
        placed_exits_mask: Optional[torch.Tensor],
        flow_w: torch.Tensor,
        exit_k: Optional[torch.Tensor] = None,
        entry_k: Optional[torch.Tensor] = None,
        return_argmin: bool = False,
        return_meta: bool = False,
    ):
        """Compute absolute flow score for the placed state (tensor-only).

        exit_k / entry_k: [P] int per placed facility.
        None → all span=1.

        If return_argmin=True, returns (scalar, c_idx [P,P], p_idx [P,P]).
        """
        placed_en = self._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = self._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        zero = torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
        if placed_en.shape[0] == 0:
            if return_argmin:
                return zero, None, None
            if return_meta:
                return zero, {"edges": {}, "edge_key_kind": "row_index"}
            return zero
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            if return_argmin:
                return zero, None, None
            if return_meta:
                return zero, {"edges": {}, "edge_key_kind": "row_index"}
            return zero
        if return_meta:
            per_src, c_idx, p_idx, detail = self._reduce_distance(
                candidate_ports=placed_ex,
                candidate_mask=placed_exits_mask,
                target_ports=placed_en,
                target_mask=placed_entries_mask,
                target_weight=flow_w,
                c_k=exit_k,
                t_k=entry_k,
                return_detail=True,
            )
        else:
            per_src, c_idx, p_idx = self._reduce_distance(
                candidate_ports=placed_ex,
                candidate_mask=placed_exits_mask,
                target_ports=placed_en,
                target_mask=placed_entries_mask,
                target_weight=flow_w,
                c_k=exit_k,
                t_k=entry_k,
            )
            detail = {}
        total = per_src.sum()
        if return_argmin:
            return total, c_idx, p_idx
        if not return_meta:
            return total
        meta: Dict[str, object] = {"edges": {}, "edge_key_kind": "row_index"}
        if c_idx is not None and p_idx is not None:
            selected_c_mask = detail.get("selected_c_mask", None) if isinstance(detail, dict) else None
            selected_p_mask = detail.get("selected_p_mask", None) if isinstance(detail, dict) else None
            edge_dist = detail.get("edge_distance", None) if isinstance(detail, dict) else None
            c_valid = self._port_mask(placed_ex, placed_exits_mask, name="placed_exits_mask")
            t_valid = self._port_mask(placed_en, placed_entries_mask, name="placed_entries_mask")
            if flow_w.dim() == 2:
                active_ij = torch.nonzero(flow_w > 0, as_tuple=False)
                active_w = flow_w[active_ij[:, 0], active_ij[:, 1]] if int(active_ij.numel()) > 0 else flow_w.new_zeros((0,))
            else:
                active_j = torch.nonzero(flow_w > 0, as_tuple=False).view(-1)
                if int(active_j.numel()) == 0:
                    active_ij = torch.zeros((0, 2), dtype=torch.long, device=flow_w.device)
                    active_w = flow_w.new_zeros((0,))
                else:
                    src_idx = torch.arange(int(c_idx.shape[0]), dtype=torch.long, device=flow_w.device)
                    src_grid = src_idx[:, None].expand(-1, int(active_j.numel())).reshape(-1)
                    dst_grid = active_j[None, :].expand(int(src_idx.numel()), -1).reshape(-1)
                    active_ij = torch.stack((src_grid, dst_grid), dim=1)
                    active_w = flow_w[active_ij[:, 1]]

            n_active = int(active_ij.shape[0])
            if n_active > 0:
                active_i = active_ij[:, 0]
                active_j = active_ij[:, 1]
                active_dist = edge_dist[active_i, active_j] if edge_dist is not None else None

                # span=1 fast-path: gather selected (ci,pj) for active edges in a batch.
                if selected_c_mask is None or selected_p_mask is None:
                    active_ci = c_idx[active_i, active_j]
                    active_pj = p_idx[active_i, active_j]
                    active_pair_valid = c_valid[active_i, active_ci] & t_valid[active_j, active_pj]
                    active_ex = placed_ex[active_i, active_ci]
                    active_en = placed_en[active_j, active_pj]
                    for row in range(n_active):
                        i = int(active_i[row].item())
                        j = int(active_j[row].item())
                        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                        if bool(active_pair_valid[row].item()):
                            ex = (
                                float(active_ex[row, 0].item()),
                                float(active_ex[row, 1].item()),
                            )
                            en = (
                                float(active_en[row, 0].item()),
                                float(active_en[row, 1].item()),
                            )
                            pairs = [(ex, en)]
                        k = f"{i}->{j}"
                        meta["edges"][k] = {
                            "weight": float(active_w[row].item()),
                            "distance": float(active_dist[row].item()) if active_dist is not None else 0.0,
                            "pair_count": int(len(pairs)),
                            "models": {
                                "estimated": {
                                    "port_pairs": [[list(ex), list(en)] for ex, en in pairs],
                                }
                            },
                        }
                else:
                    # span>1 path: avoid nested python loops by extracting selected (ci,pj)
                    # pairs via a single nonzero per active edge.
                    for row in range(n_active):
                        i = int(active_i[row].item())
                        j = int(active_j[row].item())
                        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                        pair_mask = (
                            selected_p_mask[i, j]
                            & selected_c_mask[i, j].unsqueeze(1)
                            & c_valid[i].unsqueeze(1)
                            & t_valid[j].unsqueeze(0)
                        )
                        pair_idx = torch.nonzero(pair_mask, as_tuple=False)
                        if int(pair_idx.shape[0]) > 0:
                            ci_idx = pair_idx[:, 0]
                            pj_idx = pair_idx[:, 1]
                            ex_xy = placed_ex[i, ci_idx]
                            en_xy = placed_en[j, pj_idx]
                            for k_idx in range(int(pair_idx.shape[0])):
                                pairs.append((
                                    (float(ex_xy[k_idx, 0].item()), float(ex_xy[k_idx, 1].item())),
                                    (float(en_xy[k_idx, 0].item()), float(en_xy[k_idx, 1].item())),
                                ))
                        if not pairs:
                            ci = int(c_idx[i, j].item())
                            pj = int(p_idx[i, j].item())
                            if bool(c_valid[i, ci].item()) and bool(t_valid[j, pj].item()):
                                ex = (
                                    float(placed_ex[i, ci, 0].item()),
                                    float(placed_ex[i, ci, 1].item()),
                                )
                                en = (
                                    float(placed_en[j, pj, 0].item()),
                                    float(placed_en[j, pj, 1].item()),
                                )
                                pairs = [(ex, en)]
                        k = f"{i}->{j}"
                        meta["edges"][k] = {
                            "weight": float(active_w[row].item()),
                            "distance": float(active_dist[row].item()) if active_dist is not None else 0.0,
                            "pair_count": int(len(pairs)),
                            "models": {
                                "estimated": {
                                    "port_pairs": [[list(ex), list(en)] for ex, en in pairs],
                                }
                            },
                        }
        return total, meta

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
        c_exit_k: int = 1,
        c_entry_k: int = 1,
        t_entry_k: Optional[torch.Tensor] = None,
        t_exit_k: Optional[torch.Tensor] = None,
        return_argmin: bool = False,
        return_meta: bool = False,
        placed_node_ids: Optional[Sequence[str | int]] = None,
        candidate_gid: Optional[str | int] = None,
        previous_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Compute incremental flow cost for M candidate placements.

        Port span selection:
          c_exit_k / c_entry_k: candidate facility span (uniform for all M).
          t_entry_k / t_exit_k: [T] span tensors for placed facilities. None → all 1.

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
            zero = torch.zeros((m,), dtype=torch.float32, device=device)
            if return_argmin:
                return zero, None, None
            if return_meta:
                return zero, {"candidate_count": int(m), "edges": {}}
            return zero

        out_term = torch.zeros((m,), dtype=torch.float32, device=device)
        in_term = torch.zeros((m,), dtype=torch.float32, device=device)
        out_am = None
        in_am = None

        out_detail: Optional[Dict[str, torch.Tensor]] = None
        in_detail: Optional[Dict[str, torch.Tensor]] = None
        out_full_idx = torch.nonzero(out_idx, as_tuple=False).view(-1)
        in_full_idx = torch.nonzero(in_idx, as_tuple=False).view(-1)

        if has_out:
            out_entries = placed_en[out_idx]
            out_entries_mask = placed_entries_mask[out_idx] if placed_entries_mask is not None else None
            out_w = w_out_t[out_idx]
            if return_meta:
                out_term, oc_idx, op_idx, out_detail = self._reduce_distance(
                    candidate_ports=cand_exits,
                    candidate_mask=candidate_exits_mask,
                    target_ports=out_entries,
                    target_mask=out_entries_mask,
                    target_weight=out_w,
                    c_k=self._select_k_tensor(c_exit_k, m, device),
                    t_k=t_entry_k[out_idx] if t_entry_k is not None else None,
                    return_detail=True,
                )
            else:
                out_term, oc_idx, op_idx = self._reduce_distance(
                    candidate_ports=cand_exits,
                    candidate_mask=candidate_exits_mask,
                    target_ports=out_entries,
                    target_mask=out_entries_mask,
                    target_weight=out_w,
                    c_k=self._select_k_tensor(c_exit_k, m, device),
                    t_k=t_entry_k[out_idx] if t_entry_k is not None else None,
                )
            if return_argmin:
                out_am = (oc_idx, op_idx, out_idx)
        if has_in:
            in_exits = placed_ex[in_idx]
            in_exits_mask = placed_exits_mask[in_idx] if placed_exits_mask is not None else None
            in_w = w_in_t[in_idx]
            if return_meta:
                in_term, ic_idx, ip_idx, in_detail = self._reduce_distance(
                    candidate_ports=cand_entries,
                    candidate_mask=candidate_entries_mask,
                    target_ports=in_exits,
                    target_mask=in_exits_mask,
                    target_weight=in_w,
                    c_k=self._select_k_tensor(c_entry_k, m, device),
                    t_k=t_exit_k[in_idx] if t_exit_k is not None else None,
                    return_detail=True,
                )
            else:
                in_term, ic_idx, ip_idx = self._reduce_distance(
                    candidate_ports=cand_entries,
                    candidate_mask=candidate_entries_mask,
                    target_ports=in_exits,
                    target_mask=in_exits_mask,
                    target_weight=in_w,
                    c_k=self._select_k_tensor(c_entry_k, m, device),
                    t_k=t_exit_k[in_idx] if t_exit_k is not None else None,
                )
            if return_argmin:
                in_am = (ic_idx, ip_idx, in_idx)
        delta = out_term + in_term
        if return_argmin:
            return delta, out_am, in_am
        if not return_meta:
            return delta

        if int(m) != 1:
            raise ValueError("FlowReward.delta(return_meta=True) supports only single-candidate input")
        if placed_node_ids is None or candidate_gid is None:
            raise ValueError("FlowReward.delta(return_meta=True) requires placed_node_ids and candidate_gid")

        if int(len(placed_node_ids)) != int(placed_en.shape[0]):
            raise ValueError(
                f"placed_node_ids length {len(placed_node_ids)} does not match placed tensor rows {int(placed_en.shape[0])}"
            )

        edge_meta: Dict[str, Dict[str, object]] = {}
        candidate_entry_valid = self._port_mask(cand_entries, candidate_entries_mask, name="candidate_entries_mask")[0]
        candidate_exit_valid = self._port_mask(cand_exits, candidate_exits_mask, name="candidate_exits_mask")[0]
        placed_entry_valid = self._port_mask(placed_en, placed_entries_mask, name="placed_entries_mask")
        placed_exit_valid = self._port_mask(placed_ex, placed_exits_mask, name="placed_exits_mask")

        def _build_pair_indices(
            *,
            candidate_valid_row: torch.Tensor,
            target_valid_row: torch.Tensor,
            chosen_c_idx: int,
            chosen_p_idx: int,
            selected_c_mask_row: Optional[torch.Tensor],
            selected_p_mask_row: Optional[torch.Tensor],
        ) -> List[Tuple[int, int]]:
            if selected_c_mask_row is None or selected_p_mask_row is None:
                if bool(candidate_valid_row[chosen_c_idx].item()) and bool(target_valid_row[chosen_p_idx].item()):
                    return [(int(chosen_c_idx), int(chosen_p_idx))]
                return []
            pair_mask = (
                selected_p_mask_row
                & selected_c_mask_row.unsqueeze(1)
                & candidate_valid_row.unsqueeze(1)
                & target_valid_row.unsqueeze(0)
            )
            pair_idx = torch.nonzero(pair_mask, as_tuple=False)
            if int(pair_idx.shape[0]) > 0:
                out: List[Tuple[int, int]] = []
                for i in range(int(pair_idx.shape[0])):
                    out.append((int(pair_idx[i, 0].item()), int(pair_idx[i, 1].item())))
                return out
            if bool(candidate_valid_row[chosen_c_idx].item()) and bool(target_valid_row[chosen_p_idx].item()):
                return [(int(chosen_c_idx), int(chosen_p_idx))]
            return []

        candidate_gid_s = str(candidate_gid)
        if has_out and out_detail is not None and oc_idx is not None and op_idx is not None:
            selected_c_mask = out_detail.get("selected_c_mask", None)
            selected_p_mask = out_detail.get("selected_p_mask", None)
            edge_distance = out_detail.get("edge_distance", None)
            out_weight = w_out_t[out_idx]
            for local_t in range(int(out_full_idx.shape[0])):
                full_j = int(out_full_idx[local_t].item())
                dst_gid_s = str(placed_node_ids[full_j])
                key = f"{candidate_gid_s}->{dst_gid_s}"
                chosen_c = int(oc_idx[0, local_t].item())
                chosen_p = int(op_idx[0, local_t].item())
                raw_pairs = _build_pair_indices(
                    candidate_valid_row=candidate_exit_valid,
                    target_valid_row=placed_entry_valid[full_j],
                    chosen_c_idx=chosen_c,
                    chosen_p_idx=chosen_p,
                    selected_c_mask_row=selected_c_mask[0, local_t] if selected_c_mask is not None else None,
                    selected_p_mask_row=selected_p_mask[0, local_t] if selected_p_mask is not None else None,
                )
                pair_indices = [
                    [int(src_idx), int(dst_idx)]
                    for src_idx, dst_idx in raw_pairs
                ]
                edge_meta[key] = {
                    "weight": float(out_weight[local_t].item()),
                    "distance": float(edge_distance[0, local_t].item()) if edge_distance is not None else 0.0,
                    "pair_count": int(len(raw_pairs)),
                    "models": {
                        "estimated": {
                            "pair_indices": pair_indices,
                        }
                    },
                }

        if has_in and in_detail is not None and ic_idx is not None and ip_idx is not None:
            selected_c_mask = in_detail.get("selected_c_mask", None)
            selected_p_mask = in_detail.get("selected_p_mask", None)
            edge_distance = in_detail.get("edge_distance", None)
            in_weight = w_in_t[in_idx]
            for local_t in range(int(in_full_idx.shape[0])):
                full_i = int(in_full_idx[local_t].item())
                src_gid_s = str(placed_node_ids[full_i])
                key = f"{src_gid_s}->{candidate_gid_s}"
                chosen_c = int(ic_idx[0, local_t].item())
                chosen_p = int(ip_idx[0, local_t].item())
                raw_pairs = _build_pair_indices(
                    candidate_valid_row=candidate_entry_valid,
                    target_valid_row=placed_exit_valid[full_i],
                    chosen_c_idx=chosen_c,
                    chosen_p_idx=chosen_p,
                    selected_c_mask_row=selected_c_mask[0, local_t] if selected_c_mask is not None else None,
                    selected_p_mask_row=selected_p_mask[0, local_t] if selected_p_mask is not None else None,
                )
                # In this branch, raw pair=(candidate_entry_idx, placed_exit_idx).
                # Normalize to edge direction: src(exit)->dst(entry) => (placed_exit_idx, candidate_entry_idx).
                pair_indices = [
                    [int(src_idx), int(dst_idx)]
                    for dst_idx, src_idx in raw_pairs
                ]
                edge_meta[key] = {
                    "weight": float(in_weight[local_t].item()),
                    "distance": float(edge_distance[0, local_t].item()) if edge_distance is not None else 0.0,
                    "pair_count": int(len(raw_pairs)),
                    "models": {
                        "estimated": {
                            "pair_indices": pair_indices,
                        }
                    },
                }

        metadata: Dict[str, object] = {
            "edge_key_kind": "gid",
            "edges": edge_meta,
            "candidate_count": int(m),
            "has_out": bool(has_out),
            "has_in": bool(has_in),
        }
        if isinstance(previous_metadata, dict):
            metadata["previous_edge_count"] = int(len(dict(previous_metadata.get("edges", {}) or {})))
        return delta, metadata


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
        c_k: Optional[torch.Tensor] = None,
        t_k: Optional[torch.Tensor] = None,
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
        cost_reduced, _, _ = FlowReward._masked_pair_reduce(cost, valid, c_k, t_k)
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
        exit_k: Optional[torch.Tensor] = None,
        entry_k: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
        if route_blocked is None:
            return FlowReward().score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
                return_meta=return_meta,
            )
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        if placed_en.shape[0] == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
            if not return_meta:
                return zero
            return zero, {"edges": {}, "collision_weight": float(self.collision_weight)}
        if placed_en.shape[1] == 0 or placed_ex.shape[1] == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=placed_entries.device)
            if not return_meta:
                return zero
            return zero, {"edges": {}, "collision_weight": float(self.collision_weight)}

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
            c_k=exit_k,
            t_k=entry_k,
        )
        total = per_src.sum()
        if not return_meta:
            return total
        return total, {"collision_weight": float(self.collision_weight)}

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
        c_exit_k: int = 1,
        c_entry_k: int = 1,
        t_entry_k: Optional[torch.Tensor] = None,
        t_exit_k: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
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
                c_exit_k=c_exit_k,
                c_entry_k=c_entry_k,
                t_entry_k=t_entry_k,
                t_exit_k=t_exit_k,
                return_meta=return_meta,
            )
        cand_entries = FlowReward._to_port_tensor(candidate_entries, name="candidate_entries", allow_2d=True)
        cand_exits = FlowReward._to_port_tensor(candidate_exits, name="candidate_exits", allow_2d=True)
        placed_en = FlowReward._to_port_tensor(placed_entries, name="placed_entries", allow_2d=True)
        placed_ex = FlowReward._to_port_tensor(placed_exits, name="placed_exits", allow_2d=True)
        m = int(cand_entries.shape[0])
        device = cand_entries.device
        if m == 0:
            zero = torch.zeros((0,), dtype=torch.float32, device=device)
            if not return_meta:
                return zero
            return zero, {"candidate_count": 0, "collision_weight": float(self.collision_weight)}

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
            zero = torch.zeros((m,), dtype=torch.float32, device=device)
            if not return_meta:
                return zero
            return zero, {"candidate_count": int(m), "collision_weight": float(self.collision_weight)}

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
                c_k=FlowReward._select_k_tensor(c_exit_k, m, device),
                t_k=t_entry_k[out_idx] if t_entry_k is not None else None,
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
                c_k=FlowReward._select_k_tensor(c_entry_k, m, device),
                t_k=t_exit_k[in_idx] if t_exit_k is not None else None,
            )

        delta = out_term + in_term
        if not return_meta:
            return delta
        return delta, {
            "candidate_count": int(m),
            "collision_weight": float(self.collision_weight),
        }
