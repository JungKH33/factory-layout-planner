from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from group_placement.envs.reward.flow import FlowReward

if TYPE_CHECKING:
    from group_placement.envs.action import GroupId
    from group_placement.envs.state.base import EnvState

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
        c_k: Optional[torch.Tensor] = None, # [M] int
        t_k: Optional[torch.Tensor] = None, # [T] int
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

        # Build one BFS seed per valid target port. Reduction by k is handled
        # afterwards by FlowReward._masked_pair_reduce.
        g_tidx: List[int] = []
        g_pidx: List[int] = []
        g_tgid: List[Optional[object]] = []
        g_seed: List[torch.Tensor] = []
        for ti in range(t):
            p_idx = torch.where(tgt_valid[ti])[0].tolist()
            for pj in p_idx:
                g_tidx.append(ti)
                g_pidx.append(int(pj))
                g_tgid.append(t_gid_list[ti])
                g_seed.append(target_ports[ti, int(pj):int(pj) + 1, :])

        g_total = len(g_tidx)
        if g_total == 0:
            return torch.zeros((m,), dtype=torch.float32, device=device)

        lane_cost = torch.full((m, t, c, p), float(unreachable), dtype=torch.float32, device=device)

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

            smax = max(int(g_seed[g].shape[0]) for g in gids)
            seed_xy = torch.zeros((b, smax, 2), dtype=torch.float32, device=device)
            seed_mask = torch.zeros((b, smax), dtype=torch.bool, device=device)
            for bi, g in enumerate(gids):
                seeds = g_seed[g].to(device=device, dtype=torch.float32).view(-1, 2)
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
                pj = g_pidx[g]
                lane_cost[:, ti, :, pj] = d[:, bi, :]

        valid = cand_valid[:, None, :, None] & tgt_valid[None, :, None, :]
        reduced_mt, _, _ = FlowReward._masked_pair_reduce(
            lane_cost,
            valid,
            c_k=c_k,
            t_k=t_k,
        )

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
        exit_k: Optional[torch.Tensor] = None,
        entry_k: Optional[torch.Tensor] = None,
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
                exit_k=exit_k,
                entry_k=entry_k,
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
                exit_k=exit_k,
                entry_k=entry_k,
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
                c_k=exit_k[src:src + 1] if exit_k is not None else None,
                t_k=entry_k,
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
        c_exit_k: int = 1,
        c_entry_k: int = 1,
        t_entry_k: Optional[torch.Tensor] = None,
        t_exit_k: Optional[torch.Tensor] = None,
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
                c_exit_k=c_exit_k,
                c_entry_k=c_entry_k,
                t_entry_k=t_entry_k,
                t_exit_k=t_exit_k,
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
                c_k=FlowReward._select_k_tensor(c_exit_k, m, device),
                t_k=t_entry_k[out_idx] if t_entry_k is not None else None,
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
                c_k=FlowReward._select_k_tensor(c_entry_k, m, device),
                t_k=t_exit_k[in_idx] if t_exit_k is not None else None,
            )

        return out_term + in_term


__all__ = ["FlowLaneDistanceReward"]
