from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from ..reward import FlowReward
from .base import GroupSpec

GridXY = Tuple[int, int]
StrideBBox = Tuple[int, int, int, int]  # (min_sx, min_sy, max_sx, max_sy), inclusive


@dataclass(frozen=True)
class DynamicSpec(GroupSpec):
    """Resolved dynamic placement spec in placed orientation."""

    rotation: int
    unit_w: int
    unit_h: int
    unit_z: float
    unit_capacity: int
    max_capacity: int
    stride_w: int
    stride_h: int
    clearance_left: int
    clearance_right: int
    clearance_bottom: int
    clearance_top: int
    device: torch.device
    id: object | None = None
    zone_values: Dict[str, object] = field(default_factory=dict)
    _entry_port_mode: str = "min"
    _exit_port_mode: str = "min"

    def validate(self) -> None:
        if int(self.unit_w) <= 0 or int(self.unit_h) <= 0:
            raise ValueError("unit_w/unit_h must be > 0")
        if float(self.unit_z) <= 0.0:
            raise ValueError("unit_z must be > 0")
        if int(self.unit_capacity) <= 0:
            raise ValueError("unit_capacity must be > 0")
        if int(self.max_capacity) <= 0:
            raise ValueError("max_capacity must be > 0")
        if int(self.stride_w) <= 0 or int(self.stride_h) <= 0:
            raise ValueError("stride_w/stride_h must be > 0")

    @property
    def body_area(self) -> float:
        return float(self.unit_w) * float(self.unit_h)

    def set_device(self, device: torch.device) -> None:
        object.__setattr__(self, "device", torch.device(device))


@dataclass
class DynamicPlacement:
    """Planned placement result from one bottom-left start anchor."""

    x_bl: int
    y_bl: int
    rotation: int
    placed_units: Set[GridXY] = field(default_factory=set)  # unit-anchor grid cells
    unit_rows: List[Dict[str, float]] = field(default_factory=list)
    total_capacity: int = 0
    max_capacity: int = 0
    success: bool = False
    reason: str = "ok"
    bbox_stride: Optional[StrideBBox] = None
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    entry_points: List[Tuple[float, float]] = field(default_factory=list)
    exit_points: List[Tuple[float, float]] = field(default_factory=list)


class DynamicPlanner:
    """Frontier planner for dynamic expansion from a bottom-left start point.

    Priority order:
    1) height (more available capacity first)
    2) delta bbox area in stride space
    3) flow penalty (precomputed on stride-grid from unit-center candidates)
    4) tie breaker (FIFO by push order)

    TODO(reward): hook flow/reward term after reward input contract is finalized.
    """

    def __init__(self, *, grid_unit: float = 1.0):
        self.grid_unit = float(grid_unit)
        self.flow_reward = FlowReward()

    @staticmethod
    def _bbox_area(bbox: Optional[StrideBBox]) -> int:
        if bbox is None:
            return 0
        minx, miny, maxx, maxy = bbox
        return int(maxx - minx + 1) * int(maxy - miny + 1)

    @staticmethod
    def _bbox_after_add(bbox: Optional[StrideBBox], sx: int, sy: int) -> StrideBBox:
        if bbox is None:
            return (int(sx), int(sy), int(sx), int(sy))
        minx, miny, maxx, maxy = bbox
        return (min(minx, int(sx)), min(miny, int(sy)), max(maxx, int(sx)), max(maxy, int(sy)))

    @classmethod
    def _delta_bbox_area(cls, bbox: Optional[StrideBBox], sx: int, sy: int) -> int:
        return cls._bbox_area(cls._bbox_after_add(bbox, sx, sy)) - cls._bbox_area(bbox)

    @staticmethod
    def _build_local_occupied(
        *,
        placed_units: Set[GridXY],
        unit_w: int,
        unit_h: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        """Build local occupied mask for placed unit rectangles."""
        if not placed_units:
            return None, 0, 0
        min_x = min(int(px) for px, _ in placed_units)
        min_y = min(int(py) for _, py in placed_units)
        max_x = max(int(px) + int(unit_w) for px, _ in placed_units)
        max_y = max(int(py) + int(unit_h) for _, py in placed_units)
        local_w = int(max_x - min_x)
        local_h = int(max_y - min_y)
        if local_w <= 0 or local_h <= 0:
            return None, min_x, min_y

        occ = torch.zeros((local_h, local_w), dtype=torch.bool, device=device)
        uw = int(unit_w)
        uh = int(unit_h)
        for px, py in placed_units:
            lx = int(px) - min_x
            ly = int(py) - min_y
            occ[ly : ly + uh, lx : lx + uw] = True
        return occ, min_x, min_y

    @staticmethod
    def _edge_points_from_boundary(
        *,
        occ_local: torch.Tensor,
        origin_x: int,
        origin_y: int,
    ) -> torch.Tensor:
        """Extract boundary edge points (not cell centers) as [B,2] tensor."""
        if occ_local is None or occ_local.numel() == 0:
            dev = occ_local.device if occ_local is not None else torch.device("cpu")
            return torch.empty((0, 2), dtype=torch.float32, device=dev)
        if not bool(occ_local.any().item()):
            return torch.empty((0, 2), dtype=torch.float32, device=occ_local.device)

        h, w = occ_local.shape
        up_nb = torch.zeros_like(occ_local)
        down_nb = torch.zeros_like(occ_local)
        left_nb = torch.zeros_like(occ_local)
        right_nb = torch.zeros_like(occ_local)

        if h > 1:
            up_nb[1:, :] = occ_local[:-1, :]
            down_nb[:-1, :] = occ_local[1:, :]
        if w > 1:
            left_nb[:, 1:] = occ_local[:, :-1]
            right_nb[:, :-1] = occ_local[:, 1:]

        edge_up = occ_local & (~up_nb)
        edge_down = occ_local & (~down_nb)
        edge_left = occ_local & (~left_nb)
        edge_right = occ_local & (~right_nb)

        pts_t: List[torch.Tensor] = []

        yu, xu = torch.where(edge_up)
        if yu.numel() > 0:
            pts_t.append(
                torch.stack(
                    [
                        xu.to(dtype=torch.float32) + 0.5 + float(origin_x),
                        yu.to(dtype=torch.float32) + float(origin_y),
                    ],
                    dim=1,
                )
            )

        yd, xd = torch.where(edge_down)
        if yd.numel() > 0:
            pts_t.append(
                torch.stack(
                    [
                        xd.to(dtype=torch.float32) + 0.5 + float(origin_x),
                        yd.to(dtype=torch.float32) + 1.0 + float(origin_y),
                    ],
                    dim=1,
                )
            )

        yl, xl = torch.where(edge_left)
        if yl.numel() > 0:
            pts_t.append(
                torch.stack(
                    [
                        xl.to(dtype=torch.float32) + float(origin_x),
                        yl.to(dtype=torch.float32) + 0.5 + float(origin_y),
                    ],
                    dim=1,
                )
            )

        yr, xr = torch.where(edge_right)
        if yr.numel() > 0:
            pts_t.append(
                torch.stack(
                    [
                        xr.to(dtype=torch.float32) + 1.0 + float(origin_x),
                        yr.to(dtype=torch.float32) + 0.5 + float(origin_y),
                    ],
                    dim=1,
                )
            )

        if not pts_t:
            return torch.empty((0, 2), dtype=torch.float32, device=occ_local.device)
        return torch.cat(pts_t, dim=0)

    @staticmethod
    def _build_capacity_grid(
        *,
        geom: DynamicSpec,
        height_map: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> torch.Tensor:
        """Build stride-grid capacity map on unit footprint (tensorized).

        This computes only stride-lattice anchors directly (no dense map + gather).
        """
        if height_map.dim() != 2:
            raise ValueError("height_map must be [H,W] float tensor")
        device = height_map.device
        gh = int(ys.numel())
        gw = int(xs.numel())
        out = torch.zeros((gh, gw), dtype=torch.int32, device=device)
        if gh <= 0 or gw <= 0:
            return out

        h, w = height_map.shape

        uw = int(geom.unit_w)
        uh = int(geom.unit_h)
        sw = int(geom.stride_w)
        sh = int(geom.stride_h)
        if uw <= 0 or uh <= 0:
            return out
        if sw <= 0 or sh <= 0:
            return out
        valid_h = int(h) - uh + 1
        valid_w = int(w) - uw + 1
        if valid_h <= 0 or valid_w <= 0:
            return out

        if float(geom.unit_z) <= 0.0:
            return out

        x0 = int(xs[0].item())
        y0 = int(ys[0].item())
        if x0 < 0 or y0 < 0 or x0 >= int(w) or y0 >= int(h):
            return out

        # NaN is treated as invalid height for capacity.
        hmap = torch.nan_to_num(height_map.to(dtype=torch.float32), nan=-1e30)
        hmap_crop = hmap[y0:, x0:]
        if int(hmap_crop.shape[0]) < uh or int(hmap_crop.shape[1]) < uw:
            return out

        # Min height over unit window at stride anchors.
        min_h = -F.max_pool2d(
            (-hmap_crop).unsqueeze(0).unsqueeze(0),
            kernel_size=(uh, uw),
            stride=(sh, sw),
        ).squeeze()
        if min_h.dim() == 0:
            min_h = min_h.unsqueeze(0).unsqueeze(0)
        elif min_h.dim() == 1:
            min_h = min_h.unsqueeze(0)

        floors = torch.floor(min_h / float(geom.unit_z))
        cap = floors * float(int(geom.unit_capacity))
        cap = torch.where(torch.isposinf(min_h), torch.full_like(cap, float(int(geom.max_capacity))), cap)
        cap = torch.nan_to_num(cap, nan=0.0, posinf=float(int(geom.max_capacity)), neginf=0.0)
        cap = cap.clamp(min=0.0, max=float(int(geom.max_capacity))).to(dtype=torch.int32)

        hh = min(int(cap.shape[0]), gh)
        ww = min(int(cap.shape[1]), gw)
        if hh <= 0 or ww <= 0:
            return out
        out[:hh, :ww] = cap[:hh, :ww]
        return out

    @staticmethod
    def _build_placeable_grid(
        *,
        geom: DynamicSpec,
        height_map: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        invalid: Optional[torch.Tensor],
        clear_invalid: Optional[torch.Tensor],
        occupied: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build stride-grid placeable mask using stride-aware conv2d."""
        if height_map.dim() != 2:
            raise ValueError("height_map must be [H,W] float tensor")
        device = height_map.device
        gh = int(ys.numel())
        gw = int(xs.numel())
        result = torch.zeros((gh, gw), dtype=torch.bool, device=device)
        if gh <= 0 or gw <= 0:
            return result

        h, w = height_map.shape

        bw = int(geom.unit_w)
        bh = int(geom.unit_h)
        sw = int(geom.stride_w)
        sh = int(geom.stride_h)
        cl = int(geom.clearance_left)
        cr = int(geom.clearance_right)
        cb = int(geom.clearance_bottom)
        ct = int(geom.clearance_top)

        if bw <= 0 or bh <= 0:
            return result
        if sw <= 0 or sh <= 0:
            return result
        if bw > int(w) or bh > int(h):
            return result

        kw = bw + cl + cr
        kh = bh + cb + ct
        if kw <= 0 or kh <= 0:
            return result
        if kw > int(w) or kh > int(h):
            return result

        x0 = int(xs[0].item())
        y0 = int(ys[0].item())
        if x0 < 0 or y0 < 0 or x0 >= int(w) or y0 >= int(h):
            return result

        invalid_b = invalid if invalid is not None else torch.zeros((h, w), dtype=torch.bool, device=device)
        clear_b = clear_invalid if clear_invalid is not None else torch.zeros((h, w), dtype=torch.bool, device=device)
        occ_b = occupied if occupied is not None else torch.zeros((h, w), dtype=torch.bool, device=device)

        invalid_b = invalid_b.to(device=device, dtype=torch.bool)
        clear_b = clear_b.to(device=device, dtype=torch.bool)
        occ_b = occ_b.to(device=device, dtype=torch.bool)

        block_src = (invalid_b | clear_b | occ_b).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        invalid_src = invalid_b.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        invalid_src_padded = F.pad(invalid_src, (cl, cr, cb, ct), mode="constant", value=1.0)

        block_kernel = torch.ones((1, 1, bh, bw), dtype=torch.float32, device=device)
        bwc_kernel = torch.ones((1, 1, kh, kw), dtype=torch.float32, device=device)

        block_crop = block_src[:, :, y0:, x0:]
        bwc_crop = invalid_src_padded[:, :, y0:, x0:]
        if int(block_crop.shape[-2]) < bh or int(block_crop.shape[-1]) < bw:
            return result
        if int(bwc_crop.shape[-2]) < kh or int(bwc_crop.shape[-1]) < kw:
            return result

        block_hit = F.conv2d(block_crop, block_kernel, stride=(sh, sw)).squeeze()
        bwc_hit = F.conv2d(bwc_crop, bwc_kernel, stride=(sh, sw)).squeeze()

        if block_hit.dim() == 0:
            block_hit = block_hit.unsqueeze(0).unsqueeze(0)
        elif block_hit.dim() == 1:
            block_hit = block_hit.unsqueeze(0)
        if bwc_hit.dim() == 0:
            bwc_hit = bwc_hit.unsqueeze(0).unsqueeze(0)
        elif bwc_hit.dim() == 1:
            bwc_hit = bwc_hit.unsqueeze(0)

        hh = min(int(block_hit.shape[0]), int(bwc_hit.shape[0]), gh)
        ww = min(int(block_hit.shape[1]), int(bwc_hit.shape[1]), gw)
        if hh <= 0 or ww <= 0:
            return result
        result[:hh, :ww] = (block_hit[:hh, :ww] == 0) & (bwc_hit[:hh, :ww] == 0)
        return result

    def _build_flow_penalty_grid(
        self,
        *,
        geom: DynamicSpec,
        xs: torch.Tensor,
        ys: torch.Tensor,
        flow_out_target_entries_xy: Optional[torch.Tensor],
        flow_out_weights: Optional[torch.Tensor],
        flow_in_target_exits_xy: Optional[torch.Tensor],
        flow_in_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build stride-grid flow penalty map from unit-center candidates via FlowReward."""
        gh = int(ys.numel())
        gw = int(xs.numel())
        out = torch.zeros((gh, gw), dtype=torch.float32, device=xs.device)
        if gh <= 0 or gw <= 0:
            return out

        cx = xs.to(dtype=torch.float32).view(1, gw) + float(geom.unit_w) * 0.5
        cy = ys.to(dtype=torch.float32).view(gh, 1) + float(geom.unit_h) * 0.5
        candidate_xy = torch.stack([cx.expand(gh, gw), cy.expand(gh, gw)], dim=-1).view(-1, 2)  # [Gh*Gw,2]
        empty_xy = torch.empty((0, 2), dtype=torch.float32, device=xs.device)
        empty_w = torch.empty((0,), dtype=torch.float32, device=xs.device)

        out_xy = (
            flow_out_target_entries_xy.to(device=xs.device, dtype=torch.float32).view(-1, 2)
            if (flow_out_target_entries_xy is not None and flow_out_weights is not None)
            else empty_xy
        )
        out_w = (
            flow_out_weights.to(device=xs.device, dtype=torch.float32).view(-1)
            if (flow_out_target_entries_xy is not None and flow_out_weights is not None)
            else empty_w
        )
        in_xy = (
            flow_in_target_exits_xy.to(device=xs.device, dtype=torch.float32).view(-1, 2)
            if (flow_in_target_exits_xy is not None and flow_in_weights is not None)
            else empty_xy
        )
        in_w = (
            flow_in_weights.to(device=xs.device, dtype=torch.float32).view(-1)
            if (flow_in_target_exits_xy is not None and flow_in_weights is not None)
            else empty_w
        )

        penalty = self.flow_reward.delta(
            placed_entries=out_xy,
            placed_exits=in_xy,
            placed_entries_mask=None,
            placed_exits_mask=None,
            w_out=out_w,
            w_in=in_w,
            candidate_entries=candidate_xy,
            candidate_exits=candidate_xy,
            candidate_entries_mask=None,
            candidate_exits_mask=None,
        )
        return penalty.view(gh, gw)

    def plan_from_bl(
        self,
        *,
        x_bl: int,
        y_bl: int,
        geom: DynamicSpec,
        height_map: torch.Tensor,
        invalid: Optional[torch.Tensor] = None,
        clear_invalid: Optional[torch.Tensor] = None,
        occupied: Optional[torch.Tensor] = None,
        flow_out_target_entries_xy: Optional[torch.Tensor] = None,
        flow_out_weights: Optional[torch.Tensor] = None,
        flow_in_target_exits_xy: Optional[torch.Tensor] = None,
        flow_in_weights: Optional[torch.Tensor] = None,
    ) -> DynamicPlacement:
        """Plan dynamic placement from one start point.

        Inputs:
        - x_bl/y_bl: start bottom-left anchor on unit grid
        - height_map: [H,W] float ceiling map
        - geom.unit_w/unit_h: capacity footprint
        - geom.unit_z/unit_capacity: capacity scale
        - geom.max_capacity: target capacity to reach
        - invalid/clear_invalid/occupied: optional local feasibility maps
        - flow_out_target_entries_xy/flow_out_weights: optional flow targets for choosing exit_points
        - flow_in_target_exits_xy/flow_in_weights: optional flow targets for choosing entry_points
        """
        geom.validate()
        device = geom.device
        if height_map.dim() != 2:
            raise ValueError("height_map must be [H,W] float tensor")
        for name, t in (
            ("invalid", invalid),
            ("clear_invalid", clear_invalid),
            ("occupied", occupied),
        ):
            if t is None:
                continue
            if t.dim() != 2:
                raise ValueError(f"{name} must be [H,W] tensor")
            if tuple(t.shape) != tuple(height_map.shape):
                raise ValueError(f"{name} and height_map must have the same shape")
        for xy_name, w_name, xy_t, w_t in (
            ("flow_out_target_entries_xy", "flow_out_weights", flow_out_target_entries_xy, flow_out_weights),
            ("flow_in_target_exits_xy", "flow_in_weights", flow_in_target_exits_xy, flow_in_weights),
        ):
            if (xy_t is None) != (w_t is None):
                raise ValueError(f"{xy_name} and {w_name} must be provided together")
            if xy_t is None:
                continue
            if xy_t.dim() != 2 or int(xy_t.shape[-1]) != 2:
                raise ValueError(f"{xy_name} must be [K,2] tensor")
            if w_t.dim() != 1:
                raise ValueError(f"{w_name} must be [K] tensor")
            if int(w_t.shape[0]) != int(xy_t.shape[0]):
                raise ValueError(f"{w_name} length must match {xy_name}")

        # Normalize all tensors onto geom.device (group spec-style device ownership).
        height_map = height_map.to(device=device)
        if invalid is not None:
            invalid = invalid.to(device=device)
        if clear_invalid is not None:
            clear_invalid = clear_invalid.to(device=device)
        if occupied is not None:
            occupied = occupied.to(device=device)
        if flow_out_target_entries_xy is not None:
            flow_out_target_entries_xy = flow_out_target_entries_xy.to(device=device, dtype=torch.float32)
        if flow_out_weights is not None:
            flow_out_weights = flow_out_weights.to(device=device, dtype=torch.float32)
        if flow_in_target_exits_xy is not None:
            flow_in_target_exits_xy = flow_in_target_exits_xy.to(device=device, dtype=torch.float32)
        if flow_in_weights is not None:
            flow_in_weights = flow_in_weights.to(device=device, dtype=torch.float32)

        start_x = int(x_bl)
        start_y = int(y_bl)
        target_capacity = int(geom.max_capacity)
        result = DynamicPlacement(
            x_bl=start_x,
            y_bl=start_y,
            rotation=int(geom.rotation),
            max_capacity=target_capacity,
        )
        if target_capacity <= 0:
            result.reason = "invalid_max_capacity"
            return result

        h, w = height_map.shape
        if start_x < 0 or start_y < 0 or start_x >= int(w) or start_y >= int(h):
            result.reason = "start_out_of_bounds"
            return result

        bw = int(geom.unit_w)
        bh = int(geom.unit_h)
        max_anchor_x = int(w) - bw
        max_anchor_y = int(h) - bh
        if start_x > max_anchor_x or start_y > max_anchor_y:
            result.reason = "start_out_of_bounds"
            return result

        sw = int(geom.stride_w)
        sh = int(geom.stride_h)
        x0 = int(start_x) % sw
        y0 = int(start_y) % sh
        xs = torch.arange(x0, max_anchor_x + 1, sw, device=device, dtype=torch.long)
        ys = torch.arange(y0, max_anchor_y + 1, sh, device=device, dtype=torch.long)
        if int(xs.numel()) == 0 or int(ys.numel()) == 0:
            result.reason = "start_not_placeable"
            return result

        gx0 = (int(start_x) - x0) // sw
        gy0 = (int(start_y) - y0) // sh
        placeable_grid = self._build_placeable_grid(
            geom=geom,
            height_map=height_map,
            xs=xs,
            ys=ys,
            invalid=invalid,
            clear_invalid=clear_invalid,
            occupied=occupied,
        )
        capacity_grid = self._build_capacity_grid(
            geom=geom,
            height_map=height_map,
            xs=xs,
            ys=ys,
        )
        flow_penalty_grid = self._build_flow_penalty_grid(
            geom=geom,
            xs=xs,
            ys=ys,
            flow_out_target_entries_xy=flow_out_target_entries_xy,
            flow_out_weights=flow_out_weights,
            flow_in_target_exits_xy=flow_in_target_exits_xy,
            flow_in_weights=flow_in_weights,
        )
        # Candidate prefilter on stride-grid only.
        valid_grid = placeable_grid & (capacity_grid > 0)
        gh, gw = int(valid_grid.shape[0]), int(valid_grid.shape[1])
        if gx0 < 0 or gy0 < 0 or gx0 >= gw or gy0 >= gh:
            result.reason = "start_not_placeable"
            return result
        if not bool(valid_grid[gy0, gx0].item()):
            result.reason = "start_not_placeable"
            return result

        # Frontier is expanded on stride-grid indices, then mapped to world anchors.
        pq: List[Tuple[int, int, float, int, int, int]] = []  # (height_pen, area_pen, flow_pen, tie, gx, gy)
        tie = 0
        start_flow_pen = float(flow_penalty_grid[int(gy0), int(gx0)].item())
        heapq.heappush(pq, (0, 0, start_flow_pen, tie, int(gx0), int(gy0)))
        tie += 1

        accepted_mask = torch.zeros((gh, gw), dtype=torch.bool, device=device)
        enqueued_mask = torch.zeros((gh, gw), dtype=torch.bool, device=device)
        enqueued_mask[int(gy0), int(gx0)] = True
        current_bbox: Optional[StrideBBox] = None
        current_capacity = 0

        while pq and current_capacity < target_capacity:
            _hpen, _apen, _fpen, _t, gx, gy = heapq.heappop(pq)
            if bool(accepted_mask[gy, gx].item()):
                continue

            available_capacity = int(capacity_grid[gy, gx].item())

            # accept
            accepted_mask[gy, gx] = True
            x = int(xs[gx].item())
            y = int(ys[gy].item())
            result.placed_units.add((x, y))
            current_capacity += int(available_capacity)

            current_bbox = self._bbox_after_add(current_bbox, int(gx), int(gy))

            result.unit_rows.append(
                {
                    "unit_x": float(x) * self.grid_unit,
                    "unit_y": float(y) * self.grid_unit,
                    "unit_w": float(int(geom.unit_w)) * self.grid_unit,
                    "unit_h": float(int(geom.unit_h)) * self.grid_unit,
                    "unit_x_grid": float(x),
                    "unit_y_grid": float(y),
                    "unit_w_grid": float(int(geom.unit_w)),
                    "unit_h_grid": float(int(geom.unit_h)),
                    "rotation": float(int(geom.rotation)),
                    "capacity": float(int(available_capacity)),
                }
            )

            # expand frontier: height + bbox + flow
            for dgx, dgy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ):
                ngx = gx + dgx
                ngy = gy + dgy
                if ngx < 0 or ngy < 0 or ngx >= gw or ngy >= gh:
                    continue
                if bool(enqueued_mask[ngy, ngx].item()):
                    continue
                if not bool(valid_grid[ngy, ngx].item()):
                    continue

                area_pen = int(self._delta_bbox_area(current_bbox, int(ngx), int(ngy)))

                n_available_capacity = int(capacity_grid[ngy, ngx].item())
                h_pen = -int(n_available_capacity)
                flow_pen = float(flow_penalty_grid[ngy, ngx].item())

                heapq.heappush(pq, (h_pen, area_pen, flow_pen, tie, int(ngx), int(ngy)))
                tie += 1
                enqueued_mask[ngy, ngx] = True

        result.total_capacity = int(current_capacity)
        result.bbox_stride = current_bbox
        result.success = bool(current_capacity >= target_capacity)
        result.reason = "ok" if result.success else "insufficient_capacity"

        if result.placed_units:
            min_x = min(float(px) for px, _ in result.placed_units)
            max_x = max(float(px + int(geom.unit_w)) for px, _ in result.placed_units)
            min_y = min(float(py) for _, py in result.placed_units)
            max_y = max(float(py + int(geom.unit_h)) for _, py in result.placed_units)
            result.min_x = min_x
            result.max_x = max_x
            result.min_y = min_y
            result.max_y = max_y

            need_entry = (
                flow_in_target_exits_xy is not None
                and flow_in_weights is not None
                and int(flow_in_target_exits_xy.numel()) > 0
                and int(flow_in_weights.numel()) > 0
                and bool((flow_in_weights > 0).any().item())
            )
            need_exit = (
                flow_out_target_entries_xy is not None
                and flow_out_weights is not None
                and int(flow_out_target_entries_xy.numel()) > 0
                and int(flow_out_weights.numel()) > 0
                and bool((flow_out_weights > 0).any().item())
            )
            if not (need_entry or need_exit):
                return result

            occ_local, ox, oy = self._build_local_occupied(
                placed_units=result.placed_units,
                unit_w=int(geom.unit_w),
                unit_h=int(geom.unit_h),
                device=device,
            )
            boundary_xy = self._edge_points_from_boundary(
                occ_local=occ_local,
                origin_x=int(ox),
                origin_y=int(oy),
            )
            if int(boundary_xy.shape[0]) > 0:
                empty_xy = torch.empty((0, 2), dtype=torch.float32, device=boundary_xy.device)
                empty_w = torch.empty((0,), dtype=torch.float32, device=boundary_xy.device)

                # entry port selection (incoming term only)
                if need_entry:
                    in_xy = flow_in_target_exits_xy.to(device=boundary_xy.device, dtype=torch.float32).view(-1, 2)
                    in_w = flow_in_weights.to(device=boundary_xy.device, dtype=torch.float32).view(-1)
                    entry_pen = self.flow_reward.delta(
                        placed_entries=empty_xy,
                        placed_exits=in_xy,
                        placed_entries_mask=None,
                        placed_exits_mask=None,
                        w_out=empty_w,
                        w_in=in_w,
                        candidate_entries=boundary_xy,
                        candidate_exits=boundary_xy,
                        candidate_entries_mask=None,
                        candidate_exits_mask=None,
                    )
                    if int(entry_pen.numel()) > 0 and bool((entry_pen == entry_pen).any().item()):
                        i_ent = int(torch.argmin(entry_pen).item())
                        e = boundary_xy[i_ent]
                        result.entry_points = [(float(e[0].item()), float(e[1].item()))]

                # exit port selection (outgoing term only)
                if need_exit:
                    out_xy = flow_out_target_entries_xy.to(device=boundary_xy.device, dtype=torch.float32).view(-1, 2)
                    out_w = flow_out_weights.to(device=boundary_xy.device, dtype=torch.float32).view(-1)
                    exit_pen = self.flow_reward.delta(
                        placed_entries=out_xy,
                        placed_exits=empty_xy,
                        placed_entries_mask=None,
                        placed_exits_mask=None,
                        w_out=out_w,
                        w_in=empty_w,
                        candidate_entries=boundary_xy,
                        candidate_exits=boundary_xy,
                        candidate_entries_mask=None,
                        candidate_exits_mask=None,
                    )
                    if int(exit_pen.numel()) > 0 and bool((exit_pen == exit_pen).any().item()):
                        i_ex = int(torch.argmin(exit_pen).item())
                        ex = boundary_xy[i_ex]
                        result.exit_points = [(float(ex[0].item()), float(ex[1].item()))]

        return result
