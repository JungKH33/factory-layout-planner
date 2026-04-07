from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..action import LaneRoute
from .flow import LaneFlowGraph
from .maps import LaneMaps


@dataclass
class EnvState:
    maps: LaneMaps
    flow: LaneFlowGraph
    device: torch.device
    step_count: int
    routed_mask: torch.Tensor  # [F] bool

    @classmethod
    def empty(
        cls,
        *,
        maps: LaneMaps,
        flow: LaneFlowGraph,
        device: torch.device,
    ) -> "EnvState":
        f = int(flow.flow_count)
        return cls(
            maps=maps,
            flow=flow,
            device=torch.device(device),
            step_count=0,
            routed_mask=torch.zeros((f,), dtype=torch.bool, device=torch.device(device)),
        )

    def copy(self) -> "EnvState":
        return EnvState(
            maps=self.maps.copy(),
            flow=self.flow.copy(),
            device=self.device,
            step_count=int(self.step_count),
            routed_mask=self.routed_mask.clone(),
        )

    def restore(self, src: "EnvState") -> None:
        if not isinstance(src, EnvState):
            raise TypeError(f"src must be EnvState, got {type(src).__name__}")
        self.maps.restore(src.maps)
        self.step_count = int(src.step_count)
        self.routed_mask.copy_(src.routed_mask.to(device=self.device, dtype=torch.bool))

    def reset_runtime(self) -> None:
        self.maps.reset_runtime()
        self.step_count = 0
        self.routed_mask.zero_()

    def current_flow_index(self) -> Optional[int]:
        return self.flow.ordered_flow_index(self.step_count)

    @property
    def done(self) -> bool:
        return bool(self.step_count >= int(self.flow.flow_count))

    def remaining_flow_count(self) -> int:
        return int((~self.routed_mask).sum().item())

    def apply_route(self, route: LaneRoute) -> None:
        if self.done:
            raise RuntimeError("cannot apply route: already done")
        cur = self.current_flow_index()
        if cur is None:
            raise RuntimeError("current flow is None")
        if int(route.flow_index) != int(cur):
            raise ValueError(f"route.flow_index={route.flow_index} does not match current flow={cur}")
        self.maps.apply_edges(route.edge_indices)
        self.routed_mask[int(cur)] = True
        self.step_count += 1

    def step(self, *, apply: bool, route: Optional[LaneRoute] = None) -> None:
        if not apply:
            return
        if route is None:
            raise ValueError("step(apply=True) requires route")
        self.apply_route(route)
