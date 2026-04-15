from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Set, Tuple

import torch

from .flow import FlowReward

if TYPE_CHECKING:
    from .core import RewardComposer
    from ..state.base import EnvState
    from ..state.maps import GridMaps


_WAVEFRONT_SYNC_EVERY = 16


def _wavefront_distance_field(
    *,
    free_map: torch.Tensor,
    seeds_xy: torch.Tensor,
    max_iters: int = 0,
) -> torch.Tensor:
    """Compute shortest-path distance field on a 4-neighbor grid."""
    if free_map.dim() != 2:
        raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")
    h, w = int(free_map.shape[0]), int(free_map.shape[1])
    device = free_map.device
    dist = torch.full((h, w), -1, dtype=torch.int32, device=device)

    if seeds_xy.numel() == 0:
        return dist

    seeds = seeds_xy.to(device=device, dtype=torch.long).view(-1, 2)
    sx = seeds[:, 0]
    sy = seeds[:, 1]
    inb = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sx = sx[inb]
    sy = sy[inb]
    if sx.numel() == 0:
        return dist

    frontier = torch.zeros((h, w), dtype=torch.bool, device=device)
    frontier[sy, sx] = True
    frontier &= free_map

    cap = int(max_iters) if int(max_iters) > 0 else (h * w)
    step = 0
    while step < cap:
        dist.masked_fill_(frontier, step)

        nxt = torch.zeros_like(frontier)
        nxt[1:, :] |= frontier[:-1, :]
        nxt[:-1, :] |= frontier[1:, :]
        nxt[:, 1:] |= frontier[:, :-1]
        nxt[:, :-1] |= frontier[:, 1:]
        nxt &= free_map
        nxt &= (dist < 0)
        frontier = nxt
        step += 1

        if step % _WAVEFRONT_SYNC_EVERY == 0:
            if not bool(frontier.any().item()):
                break
    return dist


def _wavefront_distance_field_batched(
    *,
    free_map: torch.Tensor,
    seeds_xy: torch.Tensor,
    seeds_mask: torch.Tensor,
    max_iters: int = 0,
) -> torch.Tensor:
    """Batched wavefront distance fields (one row per seed set)."""
    if free_map.dim() != 2:
        raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")
    if seeds_xy.dim() != 3 or int(seeds_xy.shape[-1]) != 2:
        raise ValueError(f"seeds_xy must be [M,K,2], got {tuple(seeds_xy.shape)}")
    if seeds_mask.dim() != 2 or tuple(seeds_mask.shape) != tuple(seeds_xy.shape[:2]):
        raise ValueError(
            f"seeds_mask must be [M,K] matching seeds_xy[:2], got {tuple(seeds_mask.shape)}"
        )

    h, w = int(free_map.shape[0]), int(free_map.shape[1])
    m = int(seeds_xy.shape[0])
    device = free_map.device
    dist = torch.full((m, h, w), -1, dtype=torch.int32, device=device)
    if m == 0:
        return dist

    xs = seeds_xy[..., 0].to(device=device, dtype=torch.long)
    ys = seeds_xy[..., 1].to(device=device, dtype=torch.long)
    inb = (
        seeds_mask.to(device=device, dtype=torch.bool)
        & (xs >= 0) & (xs < w)
        & (ys >= 0) & (ys < h)
    )

    frontier = torch.zeros((m, h, w), dtype=torch.bool, device=device)
    if bool(inb.any().item()):
        flow_idx = torch.arange(m, device=device, dtype=torch.long).unsqueeze(1).expand_as(inb)
        frontier[flow_idx[inb], ys[inb], xs[inb]] = True
    frontier &= free_map.unsqueeze(0)

    cap = int(max_iters) if int(max_iters) > 0 else (h * w)
    step = 0
    while step < cap:
        dist.masked_fill_(frontier, step)

        nxt = torch.zeros_like(frontier)
        nxt[:, 1:, :] |= frontier[:, :-1, :]
        nxt[:, :-1, :] |= frontier[:, 1:, :]
        nxt[:, :, 1:] |= frontier[:, :, :-1]
        nxt[:, :, :-1] |= frontier[:, :, 1:]
        nxt &= free_map.unsqueeze(0)
        nxt &= (dist < 0)
        frontier = nxt
        step += 1

        if step % _WAVEFRONT_SYNC_EVERY == 0:
            if not bool(frontier.any().item()):
                break
    return dist


@dataclass
class TerminalPenaltyReward:
    """Failure-area terminal penalty component."""

    penalty_weight: float
    group_areas: Mapping[object, float]
    total_area: float = field(init=False)

    def __post_init__(self) -> None:
        self.total_area = float(sum(float(v) for v in self.group_areas.values()))

    @staticmethod
    def remaining_area_ratio(*, remaining_area: float, total_area: float) -> float:
        total = float(total_area)
        if total <= 0.0:
            return 1.0
        ratio = float(remaining_area) / total
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return float(ratio)

    def remaining_area(self, remaining_gids: Iterable[object]) -> float:
        remain = 0.0
        for gid in set(remaining_gids):
            remain += float(self.group_areas.get(gid, 0.0))
        return float(remain)

    def penalty_cost(self, *, state: "EnvState") -> float:
        remaining_area = self.remaining_area(state.remaining)
        ratio = self.remaining_area_ratio(remaining_area=remaining_area, total_area=self.total_area)
        return float(self.penalty_weight) * float(ratio)


@dataclass
class TerminalFlowReward:
    """
    Terminal exact flow component based on wavefront shortest paths.

    This component is terminal-only by design:
    - During placement, exact routing is unstable because future obstacles and
      ports are not fixed yet.
    - Recomputing full-map shortest paths at each step is expensive.
    """

    group_specs: Optional[Dict[object, object]] = None
    unreachable_cost: float = 1e6
    max_wave_iters: int = 0
    batched_wavefront: bool = True
    batch_chunk_size: int = 64
    include_clear_invalid: bool = False
    base_key: Optional[str] = None

    @staticmethod
    def _port_span_tensors(
        *,
        gids: list,
        group_specs: Optional[Dict[object, object]],
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if group_specs is None:
            return None, None
        n = len(gids)
        if n == 0:
            return None, None
        exit_k = torch.ones((n,), dtype=torch.int32, device=device)
        entry_k = torch.ones((n,), dtype=torch.int32, device=device)
        any_non_one = False
        for i, gid in enumerate(gids):
            spec = group_specs.get(gid, None)
            if spec is None:
                continue
            ek = int(getattr(spec, "exit_port_span", 1))
            ik = int(getattr(spec, "entry_port_span", 1))
            exit_k[i] = ek
            entry_k[i] = ik
            if ek != 1 or ik != 1:
                any_non_one = True
        if not any_non_one:
            return None, None
        return exit_k, entry_k

    @staticmethod
    def _placement_cell_sets(
        *,
        placement: object,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], torch.Tensor]:
        body_mask = getattr(placement, "body_mask", None)
        if torch.is_tensor(body_mask):
            body = body_mask.to(device=device, dtype=torch.bool)
            if body.numel() > 0 and bool(body.any().item()):
                x_bl = int(round(float(getattr(placement, "x_bl", getattr(placement, "min_x", 0.0)))))
                y_bl = int(round(float(getattr(placement, "y_bl", getattr(placement, "min_y", 0.0)))))
                by, bx = torch.where(body)
                abs_x = bx.to(dtype=torch.long) + int(x_bl)
                abs_y = by.to(dtype=torch.long) + int(y_bl)
                inb = (abs_x >= 0) & (abs_x < grid_w) & (abs_y >= 0) & (abs_y < grid_h)
                abs_x = abs_x[inb]
                abs_y = abs_y[inb]
                if int(abs_x.numel()) > 0:
                    up = torch.zeros_like(body)
                    dn = torch.zeros_like(body)
                    lt = torch.zeros_like(body)
                    rt = torch.zeros_like(body)
                    up[1:, :] = body[:-1, :]
                    dn[:-1, :] = body[1:, :]
                    lt[:, 1:] = body[:, :-1]
                    rt[:, :-1] = body[:, 1:]
                    interior = body & up & dn & lt & rt
                    boundary_local = body & (~interior)
                    bdy, bdx = torch.where(boundary_local)
                    b_abs_x = bdx.to(dtype=torch.long) + int(x_bl)
                    b_abs_y = bdy.to(dtype=torch.long) + int(y_bl)
                    b_inb = (b_abs_x >= 0) & (b_abs_x < grid_w) & (b_abs_y >= 0) & (b_abs_y < grid_h)
                    b_abs_x = b_abs_x[b_inb]
                    b_abs_y = b_abs_y[b_inb]

                    body_cells = {(int(x), int(y)) for x, y in zip(abs_x.tolist(), abs_y.tolist())}
                    boundary_cells = {(int(x), int(y)) for x, y in zip(b_abs_x.tolist(), b_abs_y.tolist())}
                    if not boundary_cells:
                        boundary_cells = set(body_cells)
                    if boundary_cells:
                        boundary_t = torch.tensor(list(boundary_cells), dtype=torch.long, device=device)
                    else:
                        boundary_t = torch.empty((0, 2), dtype=torch.long, device=device)
                    return body_cells, boundary_cells, boundary_t

        x0 = int(math.floor(float(getattr(placement, "min_x", 0.0))))
        y0 = int(math.floor(float(getattr(placement, "min_y", 0.0))))
        x1 = int(math.ceil(float(getattr(placement, "max_x", 0.0)))) - 1
        y1 = int(math.ceil(float(getattr(placement, "max_y", 0.0)))) - 1
        body_cells: Set[Tuple[int, int]] = set()
        boundary_cells: Set[Tuple[int, int]] = set()
        if x1 < x0 or y1 < y0:
            return body_cells, boundary_cells, torch.empty((0, 2), dtype=torch.long, device=device)

        for y in range(y0, y1 + 1):
            if y < 0 or y >= grid_h:
                continue
            for x in range(x0, x1 + 1):
                if x < 0 or x >= grid_w:
                    continue
                body_cells.add((x, y))
                if x == x0 or x == x1 or y == y0 or y == y1:
                    boundary_cells.add((x, y))
        if not boundary_cells:
            boundary_cells = set(body_cells)
        boundary_t = (
            torch.tensor(list(boundary_cells), dtype=torch.long, device=device)
            if boundary_cells
            else torch.empty((0, 2), dtype=torch.long, device=device)
        )
        return body_cells, boundary_cells, boundary_t

    def _anchor_ports_by_group(
        self,
        *,
        state: "EnvState",
        placed_nodes: list,
        ports_xy: torch.Tensor,
        ports_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.round(ports_xy).to(dtype=torch.long)
        anchored = raw.clone()
        valid = ports_mask.to(dtype=torch.bool, device=ports_xy.device).clone()

        anchor_cache: Dict[object, tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], torch.Tensor]] = {}
        for gid in placed_nodes:
            placement = state.placements.get(gid, None)
            if placement is None:
                anchor_cache[gid] = (set(), set(), torch.empty((0, 2), dtype=torch.long, device=ports_xy.device))
            else:
                anchor_cache[gid] = self._placement_cell_sets(
                    placement=placement,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    device=ports_xy.device,
                )

        n = int(anchored.shape[0])
        p = int(anchored.shape[1])
        for i in range(n):
            gid = placed_nodes[i]
            body_cells, boundary_cells, boundary_t = anchor_cache[gid]
            for j in range(p):
                if not bool(valid[i, j].item()):
                    continue
                x = int(anchored[i, j, 0].item())
                y = int(anchored[i, j, 1].item())
                if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
                    valid[i, j] = False
                    continue
                key = (x, y)
                if key not in body_cells:
                    continue
                if key in boundary_cells:
                    continue
                if int(boundary_t.numel()) == 0:
                    valid[i, j] = False
                    continue
                d = (boundary_t[:, 0] - int(x)).abs() + (boundary_t[:, 1] - int(y)).abs()
                k = int(torch.argmin(d).item())
                anchored[i, j, 0] = int(boundary_t[k, 0].item())
                anchored[i, j, 1] = int(boundary_t[k, 1].item())
        return raw, anchored, valid

    def exact_flow_cost(
        self,
        *,
        state: "EnvState",
        maps: "GridMaps",
    ) -> torch.Tensor:
        (
            placed_nodes,
            placed_entries,
            placed_exits,
            placed_entries_mask,
            placed_exits_mask,
        ) = state.io_tensors()
        n = len(placed_nodes)
        device = placed_entries.device
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        if n == 0:
            return zero
        if int(placed_entries.shape[1]) == 0 or int(placed_exits.shape[1]) == 0:
            return zero

        flow_w = state.build_flow_w().to(device=device, dtype=torch.float32)
        if int(flow_w.numel()) == 0 or not bool((flow_w != 0).any().item()):
            return zero

        h = int(maps.static_invalid.shape[0])
        w = int(maps.static_invalid.shape[1])
        blocked = maps.static_invalid.to(device=device, dtype=torch.bool) | maps.occ_invalid.to(device=device, dtype=torch.bool)
        if self.include_clear_invalid:
            blocked = blocked | maps.clear_invalid.to(device=device, dtype=torch.bool)
        walkable = ~blocked

        raw_entries, anchored_entries, entries_valid = self._anchor_ports_by_group(
            state=state,
            placed_nodes=placed_nodes,
            ports_xy=placed_entries,
            ports_mask=placed_entries_mask,
            grid_h=h,
            grid_w=w,
        )
        raw_exits, anchored_exits, exits_valid = self._anchor_ports_by_group(
            state=state,
            placed_nodes=placed_nodes,
            ports_xy=placed_exits,
            ports_mask=placed_exits_mask,
            grid_h=h,
            grid_w=w,
        )

        if bool(entries_valid.any().item()) or bool(exits_valid.any().item()):
            all_ports = torch.cat(
                [anchored_entries[entries_valid], anchored_exits[exits_valid]],
                dim=0,
            )
            px = all_ports[:, 0]
            py = all_ports[:, 1]
            inb = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if bool(inb.any().item()):
                walkable[py[inb], px[inb]] = True

        t = int(anchored_entries.shape[0])
        p = int(anchored_entries.shape[1])
        m = int(anchored_exits.shape[0])
        c = int(anchored_exits.shape[1])
        if m == 0 or c == 0 or t == 0 or p == 0:
            return zero

        target_uid = torch.full((t, p), -1, dtype=torch.long, device=device)
        target_cells = anchored_entries[entries_valid]
        if int(target_cells.numel()) == 0:
            return zero
        unique_targets, inverse = torch.unique(target_cells, dim=0, return_inverse=True)
        if int(unique_targets.numel()) == 0:
            return zero
        target_uid[entries_valid] = inverse

        if self.batched_wavefront:
            chunk = max(1, int(self.batch_chunk_size))
            parts = []
            k_total = int(unique_targets.shape[0])
            for s in range(0, k_total, chunk):
                part = unique_targets[s:s + chunk]
                mm = int(part.shape[0])
                seeds_xy = part.view(mm, 1, 2)
                seeds_mask = torch.ones((mm, 1), dtype=torch.bool, device=device)
                parts.append(
                    _wavefront_distance_field_batched(
                        free_map=walkable,
                        seeds_xy=seeds_xy,
                        seeds_mask=seeds_mask,
                        max_iters=int(self.max_wave_iters),
                    )
                )
            dist_batch = torch.cat(parts, dim=0) if parts else torch.empty((0, h, w), dtype=torch.int32, device=device)
        else:
            ds = []
            for i in range(int(unique_targets.shape[0])):
                ds.append(
                    _wavefront_distance_field(
                        free_map=walkable,
                        seeds_xy=unique_targets[i:i + 1],
                        max_iters=int(self.max_wave_iters),
                    ).unsqueeze(0)
                )
            dist_batch = torch.cat(ds, dim=0) if ds else torch.empty((0, h, w), dtype=torch.int32, device=device)

        anchored_exits_clamped = anchored_exits.clamp_min(0)
        anchored_exits_clamped[..., 0] = anchored_exits_clamped[..., 0].clamp_max(w - 1)
        anchored_exits_clamped[..., 1] = anchored_exits_clamped[..., 1].clamp_max(h - 1)
        src_x = anchored_exits_clamped[..., 0]
        src_y = anchored_exits_clamped[..., 1]

        exit_anchor_extra = (
            (raw_exits[..., 0] - anchored_exits[..., 0]).abs()
            + (raw_exits[..., 1] - anchored_exits[..., 1]).abs()
        ).to(dtype=torch.float32)
        entry_anchor_extra = (
            (raw_entries[..., 0] - anchored_entries[..., 0]).abs()
            + (raw_entries[..., 1] - anchored_entries[..., 1]).abs()
        ).to(dtype=torch.float32)

        valid = exits_valid[:, None, :, None] & entries_valid[None, :, None, :]
        cost = torch.full((m, t, c, p), float(self.unreachable_cost), dtype=torch.float32, device=device)

        for ti in range(t):
            for pj in range(p):
                if not bool(entries_valid[ti, pj].item()):
                    continue
                uid = int(target_uid[ti, pj].item())
                if uid < 0:
                    continue
                dist = dist_batch[uid]
                vals = dist[src_y, src_x].to(dtype=torch.float32)
                vals = torch.where(
                    vals >= 0.0,
                    vals,
                    torch.full_like(vals, float(self.unreachable_cost)),
                )
                vals = vals + exit_anchor_extra + float(entry_anchor_extra[ti, pj].item())
                cost[:, ti, :, pj] = vals

        exit_k, entry_k = self._port_span_tensors(
            gids=placed_nodes,
            group_specs=self.group_specs,
            device=device,
        )
        reduced_mt, _, _ = FlowReward._masked_pair_reduce(cost, valid, exit_k, entry_k)
        return (reduced_mt * flow_w).sum()

    def terminal_score(
        self,
        *,
        state: "EnvState",
        maps: "GridMaps",
    ) -> float:
        return float(self.exact_flow_cost(state=state, maps=maps).item())


@dataclass
class TerminalRewardComposer:
    """Composable terminal reward manager."""

    components: Dict[str, object]
    weights: Dict[str, float]
    reward_scale: float = 100.0

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")

    @staticmethod
    def _to_float_dict(v: Mapping[str, object]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, vv in v.items():
            if torch.is_tensor(vv):
                out[str(k)] = float(vv.item())
            else:
                out[str(k)] = float(vv)
        return out

    def delta_dict(
        self,
        *,
        state: "EnvState",
        maps: "GridMaps",
        reward_composer: "RewardComposer",
        failed: bool,
        base_scores_unweighted: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if bool(failed):
            for name, comp in self.components.items():
                fn = getattr(comp, "penalty_cost", None)
                if not callable(fn):
                    continue
                tw = float(self.weights.get(name, 1.0))
                out[name] = tw * float(fn(state=state))
            return out

        unweighted = (
            self._to_float_dict(base_scores_unweighted)
            if base_scores_unweighted is not None
            else reward_composer.score_dict(state, weighted=False)
        )
        for name, comp in self.components.items():
            fn = getattr(comp, "terminal_score", None)
            if not callable(fn):
                continue
            base_key = getattr(comp, "base_key", None) or name
            if base_key not in unweighted:
                raise KeyError(
                    f"terminal component {name!r} targets base key {base_key!r}, "
                    f"but available base keys are {sorted(unweighted.keys())}"
                )
            term_score = float(fn(state=state, maps=maps))
            base_score = float(unweighted[base_key])
            rw = float(reward_composer.weights.get(base_key, 1.0))
            tw = float(self.weights.get(name, 1.0))
            out[name] = tw * rw * (term_score - base_score)
        return out

    def delta_total(
        self,
        *,
        state: "EnvState",
        maps: "GridMaps",
        reward_composer: "RewardComposer",
        failed: bool,
        base_scores_unweighted: Optional[Mapping[str, object]] = None,
    ) -> float:
        delta = self.delta_dict(
            state=state,
            maps=maps,
            reward_composer=reward_composer,
            failed=bool(failed),
            base_scores_unweighted=base_scores_unweighted,
        )
        return float(sum(float(v) for v in delta.values()))
