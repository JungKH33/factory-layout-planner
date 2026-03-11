from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from ..action import EnvAction, GroupId
from ..action_space import ActionSpace
from ..placement.static import StaticSpec
from .flow import FlowGraph
from .maps import GridMaps


@dataclass
class EnvState:
    """Runtime engine state.

    Runtime copy policy:
    - copied: placements/placed/remaining/step_count, maps(runtime), flow(runtime)
    - shared: maps static tensors/cache, flow graph edges
    """

    _COPY_PUBLIC_FIELDS: ClassVar[Tuple[str, ...]] = (
        "placements",
        "placed",
        "remaining",
        "step_count",
        "placed_nodes_order",
        "maps",
        "flow",
    )

    placements: Dict[GroupId, object]
    placed: set[GroupId]
    remaining: List[GroupId]
    step_count: int
    placed_nodes_order: List[GroupId]
    maps: GridMaps
    flow: FlowGraph
    _state_sig: Tuple[int, int, Tuple[str, ...]]

    @staticmethod
    def _make_state_sig(*, grid_height: int, grid_width: int, gids: List[GroupId]) -> Tuple[int, int, Tuple[str, ...]]:
        key = tuple(sorted(str(gid) for gid in gids))
        return int(grid_height), int(grid_width), key

    @classmethod
    def empty(
        cls,
        *,
        maps: GridMaps,
        group_specs: Dict[GroupId, object],
        flow: FlowGraph,
    ) -> "EnvState":
        maps.bind_group_specs(group_specs)
        maps.reset_runtime()
        flow.reset_runtime()
        h, w = maps.shape
        return cls(
            placements={},
            placed=set(),
            remaining=[],
            step_count=0,
            placed_nodes_order=[],
            maps=maps,
            flow=flow,
            _state_sig=cls._make_state_sig(grid_height=h, grid_width=w, gids=list(group_specs.keys())),
        )

    def copy(self) -> "EnvState":
        """Return a runtime-safe copy.

        Placement objects inside ``placements`` are shared by reference
        (shallow dict copy).  This is intentional – placements are treated
        as **immutable** after creation (``build_placement``).  Do NOT
        mutate a placement object in-place once it has been stored here.
        """
        return EnvState(
            placements=dict(self.placements),
            placed=set(self.placed),
            remaining=list(self.remaining),
            step_count=int(self.step_count),
            placed_nodes_order=list(self.placed_nodes_order),
            maps=self.maps.copy(),
            flow=self.flow.copy(),
            _state_sig=self._state_sig,
        )

    def restore(self, src: "EnvState") -> None:
        """In-place restore from another state with the same signature.

        Placement objects are shared by reference (same contract as ``copy``).
        """
        if not isinstance(src, EnvState):
            raise TypeError(f"src must be EnvState, got {type(src).__name__}")
        if self._state_sig and src._state_sig and self._state_sig != src._state_sig:
            raise ValueError(f"state signature mismatch: source={src._state_sig}, target={self._state_sig}")
        self.placements.clear()
        self.placements.update(src.placements)
        self.placed.clear()
        self.placed.update(src.placed)
        self.remaining[:] = list(src.remaining)
        self.step_count = int(src.step_count)
        self.placed_nodes_order[:] = list(src.placed_nodes_order)
        self.maps.restore(src.maps)
        self.flow.restore(src.flow)

    def reset_runtime(self, *, remaining: List[GroupId]) -> None:
        self.placements = {}
        self.placed = set()
        self.remaining = list(remaining)
        self.step_count = 0
        self.placed_nodes_order = []
        self.maps.reset_runtime()
        self.flow.reset_runtime()

    def set_remaining(self, remaining: List[GroupId]) -> None:
        self.remaining = list(remaining)

    def place(self, *, gid: GroupId, placement: object) -> None:
        is_new = gid not in self.placed
        self.placements[gid] = placement
        self.placed.add(gid)
        if is_new:
            self.placed_nodes_order.append(gid)
            self.flow.invalidate_on_nodes_changed()
        else:
            self.flow.clear_flow_port_pairs()
        if gid in self.remaining:
            self.remaining.remove(gid)

        min_x = getattr(placement, "min_x", None)
        max_x = getattr(placement, "max_x", None)
        min_y = getattr(placement, "min_y", None)
        max_y = getattr(placement, "max_y", None)
        if min_x is None or max_x is None or min_y is None or max_y is None:
            raise ValueError(
                "placement must define min_x/max_x/min_y/max_y for map painting"
            )
        x0 = int(math.floor(float(min_x)))
        y0 = int(math.floor(float(min_y)))
        x1 = int(math.ceil(float(max_x)))
        y1 = int(math.ceil(float(max_y)))
        cL = int(getattr(placement, "clearance_left", 0) or 0)
        cR = int(getattr(placement, "clearance_right", 0) or 0)
        cB = int(getattr(placement, "clearance_bottom", 0) or 0)
        cT = int(getattr(placement, "clearance_top", 0) or 0)
        self.maps.paint_rects(
            bbox_min_x=float(min_x),
            bbox_max_x=float(max_x),
            bbox_min_y=float(min_y),
            bbox_max_y=float(max_y),
            body_rect=(x0, y0, x1, y1),
            clear_rect=(x0 - cL, y0 - cB, x1 + cR, y1 + cT),
        )
        self.flow.upsert_io(
            gid=gid,
            placement=placement,
            nodes=self.placed_nodes_order,
        )

    def step(
        self,
        *,
        apply: bool,
        gid: Optional[GroupId] = None,
        placement: Optional[object] = None,
    ) -> None:
        self.step_count += 1
        if not bool(apply):
            return
        if gid is None or placement is None:
            raise ValueError("step(apply=True) requires gid and placement")
        self.place(
            gid=gid,
            placement=placement,
        )

    def placed_nodes(self) -> List[GroupId]:
        return list(self.placed_nodes_order)

    def io_tensors(self) -> Tuple[List[GroupId], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.flow.io_tensors(self.placed_nodes_order)

    def build_flow_w(self) -> torch.Tensor:
        return self.flow.build_flow_w(self.placed_nodes_order)

    def build_delta_flow_weights_for(self, gid: Optional[GroupId]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.build_delta_flow_weights(gid, self.placed_nodes_order)

    def clear_flow_port_pairs(self) -> None:
        self.flow.clear_flow_port_pairs()

    def set_flow_port_pairs(self, pairs: Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        self.flow.set_flow_port_pairs(pairs)

    @property
    def flow_port_pairs(self) -> Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self.flow.flow_port_pairs

    def is_placeable(
        self,
        *,
        action: EnvAction,
        spec: StaticSpec,
    ) -> bool:
        if not isinstance(action, EnvAction):
            raise TypeError(f"action must be EnvAction, got {type(action).__name__}")
        gid = action.gid
        x_bl = int(action.x)
        y_bl = int(action.y)
        r = spec._resolve_rot(int(action.rot))
        w, h = spec._rotated_size(r)
        cL, cR, cB, cT = spec._clearance_lrtb(r)

        x0 = int(x_bl)
        y0 = int(y_bl)
        x1 = x0 + int(w)
        y1 = y0 + int(h)
        body_rect = (x0, y0, x1, y1)
        pad_rect = (x0 - int(cL), y0 - int(cB), x1 + int(cR), y1 + int(cT))
        return self.maps.is_placeable(
            gid=gid,
            body_rect=body_rect,
            pad_rect=pad_rect,
        )

    def is_placeable_map(
        self,
        *,
        gid: GroupId,
        spec: StaticSpec,
        rot: int,
    ) -> torch.Tensor:
        r = spec._resolve_rot(int(rot))
        w, h = spec._rotated_size(r)
        cL, cR, cB, cT = spec._clearance_lrtb(r)
        return self.maps.is_placeable_map(
            gid=gid,
            body_w=int(w),
            body_h=int(h),
            cL=int(cL),
            cR=int(cR),
            cB=int(cB),
            cT=int(cT),
        )

    def is_placeable_mask(
        self,
        *,
        action_space: ActionSpace,
        spec: StaticSpec,
    ) -> torch.Tensor:
        gid = action_space.gid
        if gid is None:
            raise ValueError("action_space.gid is required")
        device = self.maps.invalid.device
        xyrot = action_space.xyrot.to(dtype=torch.long, device=device).view(-1, 3)
        valid = action_space.mask.to(dtype=torch.bool, device=device).view(-1)
        if int(xyrot.shape[0]) == 0:
            return valid

        w0, h0 = spec._rotated_size(0)
        cL0, cR0, cB0, cT0 = spec._clearance_lrtb(0)
        map_0 = self.maps.is_placeable_map(
            gid=gid,
            body_w=int(w0),
            body_h=int(h0),
            cL=int(cL0),
            cR=int(cR0),
            cB=int(cB0),
            cT=int(cT0),
        )
        w90, h90 = spec._rotated_size(90)
        cL90, cR90, cB90, cT90 = spec._clearance_lrtb(90)
        map_90 = self.maps.is_placeable_map(
            gid=gid,
            body_w=int(w90),
            body_h=int(h90),
            cL=int(cL90),
            cR=int(cR90),
            cB=int(cB90),
            cT=int(cT90),
        )

        H, W = self.maps.shape
        x = xyrot[:, 0]
        y = xyrot[:, 1]
        rot = xyrot[:, 2]
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        rot_norm = torch.remainder(rot, 360)
        if torch.any((rot_norm % 90) != 0):
            raise ValueError("rot must be multiples of 90")
        if not bool(spec.rotatable):
            rot_norm = torch.zeros_like(rot_norm)
        is_rot0 = (rot_norm == 0) | (rot_norm == 180)
        x_clamped = x.clamp(0, W - 1)
        y_clamped = y.clamp(0, H - 1)
        result_0 = map_0[y_clamped, x_clamped]
        result_90 = map_90[y_clamped, x_clamped]
        can = torch.where(is_rot0, result_0, result_90) & in_bounds
        return can & valid

    def placed_bbox(self) -> Tuple[float, float, float, float]:
        return self.maps.placed_bbox()
