from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from ..action import EnvAction, GroupId
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
        "device",
        "maps",
        "flow",
    )

    placements: Dict[GroupId, object]
    placed: set[GroupId]
    remaining: List[GroupId]
    step_count: int
    placed_nodes_order: List[GroupId]
    device: torch.device
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
        device: torch.device,
    ) -> "EnvState":
        dev = torch.device(device)
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
            device=dev,
            maps=maps,
            flow=flow,
            _state_sig=cls._make_state_sig(grid_height=h, grid_width=w, gids=list(group_specs.keys())),
        )

    def copy(self) -> "EnvState":
        """Return a runtime-safe copy.

        Placement objects inside ``placements`` are shared by reference
        (shallow dict copy).  This is intentional – placements are treated
        as **immutable** after creation (``resolve``).  Do NOT
        mutate a placement object in-place once it has been stored here.
        """
        return EnvState(
            placements=dict(self.placements),
            placed=set(self.placed),
            remaining=list(self.remaining),
            step_count=int(self.step_count),
            placed_nodes_order=list(self.placed_nodes_order),
            device=self.device,
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

    def place(self, *, placement: object) -> None:
        gid = placement.group_id
        is_new = gid not in self.placed
        self.placements[gid] = placement
        self.placed.add(gid)
        if is_new:
            self.placed_nodes_order.append(gid)
            self.flow.invalidate_on_nodes_changed()
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
        body_mask = getattr(placement, "body_mask", None)
        clearance_mask = getattr(placement, "clearance_mask", None)
        clearance_origin = getattr(placement, "clearance_origin", None)
        is_rectangular = bool(getattr(placement, "is_rectangular", False))
        if not isinstance(body_mask, torch.Tensor) or not isinstance(clearance_mask, torch.Tensor) or not isinstance(clearance_origin, tuple):
            raise ValueError(
                "placement must define body_mask, clearance_mask, clearance_origin, "
                "and is_rectangular for map painting"
            )
        self.maps.paint_placement(
            bbox_min_x=float(min_x),
            bbox_max_x=float(max_x),
            bbox_min_y=float(min_y),
            bbox_max_y=float(max_y),
            x_bl=int(x0),
            y_bl=int(y0),
            body_mask=body_mask,
            clearance_mask=clearance_mask,
            clearance_origin=(int(clearance_origin[0]), int(clearance_origin[1])),
            is_rectangular=is_rectangular,
        )
        self.flow.upsert_io(
            placement=placement,
            nodes=self.placed_nodes_order,
        )

    def step(
        self,
        *,
        apply: bool,
        placement: Optional[object] = None,
    ) -> None:
        self.step_count += 1
        if not bool(apply):
            return
        if placement is None:
            raise ValueError("step(apply=True) requires placement")
        self.place(
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

    def set_flow_port_pairs(
        self,
        pairs: Dict[Tuple[GroupId, GroupId], list],
        *,
        nodes: Optional[List[GroupId]] = None,
    ) -> None:
        self.flow.set_flow_port_pairs(pairs, nodes=nodes)

    @property
    def flow_port_pairs(self) -> Dict[Tuple[GroupId, GroupId], list]:
        return self.flow.flow_port_pairs

    @property
    def flow_port_pairs_nodes_key(self) -> Tuple[GroupId, ...]:
        return self.flow.flow_port_pairs_nodes_key

    def placeable(
        self,
        *,
        placement: object,
    ) -> bool:
        gid = placement.group_id
        x_bl = int(getattr(placement, "x_bl", placement.min_x))
        y_bl = int(getattr(placement, "y_bl", placement.min_y))
        return self.maps.placeable(
            gid=gid,
            x_bl=x_bl,
            y_bl=y_bl,
            body_mask=placement.body_mask,
            clearance_mask=placement.clearance_mask,
            clearance_origin=placement.clearance_origin,
            is_rectangular=placement.is_rectangular,
        )

    def placeable_batch(
        self,
        *,
        gid: GroupId,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> torch.Tensor:
        return self.maps.placeable_batch(
            gid=gid,
            x_bl=x_bl,
            y_bl=y_bl,
            body_mask=body_mask,
            clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
            is_rectangular=is_rectangular,
        )

    def placeable_map(
        self,
        *,
        gid: GroupId,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> torch.Tensor:
        return self.maps.placeable_map(
            gid=gid,
            body_mask=body_mask,
            clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
            is_rectangular=is_rectangular,
        )

    def placed_bbox(self) -> Tuple[float, float, float, float]:
        return self.maps.placed_bbox()
