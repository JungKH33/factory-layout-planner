from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import torch

from .area import AreaReward, GridOccupancyReward
from .flow import FlowCollisionReward, FlowReward

if TYPE_CHECKING:
    from ..action import GroupId
    from ..action_space import ActionSpace
    from ..state.base import EnvState


def _require(x: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if x is None:
        raise ValueError(f"{name} is required but None")
    return x


@dataclass
class RewardComposer:
    components: Dict[str, object]
    weights: Dict[str, float]
    reward_scale: float = 100.0
    group_specs: Optional[Dict["GroupId", object]] = None

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")

    def to_reward(self, value: object):
        if torch.is_tensor(value):
            return -value.to(dtype=torch.float32) / float(self.reward_scale)
        return -float(value) / float(self.reward_scale)

    def find_component(self, cls: Type[object]) -> Tuple[Optional[str], Optional[object]]:
        for name, comp in self.components.items():
            if isinstance(comp, cls):
                return name, comp
        return None, None

    def required(self) -> set[str]:
        needed: set[str] = set()
        for comp in self.components.values():
            if comp is not None and hasattr(comp, "required"):
                needed |= set(comp.required())
        return needed

    def _port_span_tensors(
        self,
        gids: list,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build (exit_k, entry_k) int tensors from group_specs.

        Returns (None, None) when all facilities use span=1 (fast-path).
        """
        if self.group_specs is None:
            return None, None
        n = len(gids)
        if n == 0:
            return None, None
        exit_k = torch.ones((n,), dtype=torch.int32, device=device)
        entry_k = torch.ones((n,), dtype=torch.int32, device=device)
        any_non_one = False
        for i, gid in enumerate(gids):
            spec = self.group_specs.get(gid)
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

    def _score_component(
        self,
        *,
        comp: object,
        placed_count: int,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: torch.Tensor,
        placed_exits_mask: torch.Tensor,
        flow_w: torch.Tensor,
        exit_k: Optional[torch.Tensor],
        entry_k: Optional[torch.Tensor],
        cur_min_x: float,
        cur_max_x: float,
        cur_min_y: float,
        cur_max_y: float,
        route_blocked: Optional[torch.Tensor],
        placed_cell_occupied: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = placed_entries.device
        if isinstance(comp, FlowReward):
            return comp.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
            )
        if isinstance(comp, FlowCollisionReward):
            return comp.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                route_blocked=route_blocked,
                exit_k=exit_k,
                entry_k=entry_k,
            )
        if isinstance(comp, AreaReward):
            return comp.score(
                placed_count=placed_count,
                min_x=torch.tensor(float(cur_min_x), dtype=torch.float32, device=device),
                max_x=torch.tensor(float(cur_max_x), dtype=torch.float32, device=device),
                min_y=torch.tensor(float(cur_min_y), dtype=torch.float32, device=device),
                max_y=torch.tensor(float(cur_max_y), dtype=torch.float32, device=device),
            )
        if isinstance(comp, GridOccupancyReward):
            return comp.score(placed_cell_occupied=placed_cell_occupied)
        raise TypeError(f"unsupported reward component type: {type(comp).__name__}")

    def score_dict(
        self,
        state: "EnvState",
        *,
        weighted: bool = True,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        (
            placed_nodes,
            placed_entries,
            placed_exits,
            placed_entries_mask,
            placed_exits_mask,
        ) = state.io_tensors()
        placed_count = len(placed_nodes)
        cur_min_x, cur_max_x, cur_min_y, cur_max_y = state.placed_bbox()
        flow_w = state.build_flow_w()
        exit_k, entry_k = self._port_span_tensors(placed_nodes, placed_entries.device)

        out: Dict[str, float] = {}
        total = 0.0
        for name, comp in self.components.items():
            if comp is None:
                continue
            val_t = self._score_component(
                comp=comp,
                placed_count=placed_count,
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
                cur_min_x=cur_min_x,
                cur_max_x=cur_max_x,
                cur_min_y=cur_min_y,
                cur_max_y=cur_max_y,
                route_blocked=route_blocked,
                placed_cell_occupied=placed_cell_occupied,
            )
            val = float(val_t.item())
            if weighted:
                val *= float(self.weights.get(name, 1.0))
            out[name] = val
            total += val
        out["total"] = float(total)
        return out

    def score(
        self,
        state: "EnvState",
        *,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.score_dict(
            state,
            weighted=True,
            route_blocked=route_blocked,
            placed_cell_occupied=placed_cell_occupied,
        )
        return torch.tensor(
            float(scores.get("total", 0.0)),
            dtype=torch.float32,
            device=state.device,
        )

    def delta_batch(
        self,
        state: "EnvState",
        *,
        gid: Optional["GroupId"] = None,
        entry_points: Optional[torch.Tensor] = None,
        exit_points: Optional[torch.Tensor] = None,
        entry_mask: Optional[torch.Tensor] = None,
        exit_mask: Optional[torch.Tensor] = None,
        min_x: Optional[torch.Tensor] = None,
        max_x: Optional[torch.Tensor] = None,
        min_y: Optional[torch.Tensor] = None,
        max_y: Optional[torch.Tensor] = None,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute incremental cost from keyword feature tensors.

        Same logic as ``delta()`` but reads features from kwargs
        instead of an ActionSpace object.
        """
        (
            placed_nodes,
            placed_entries,
            placed_exits,
            placed_entries_mask,
            placed_exits_mask,
        ) = state.io_tensors()
        placed_count = len(placed_nodes)
        cur_min_x, cur_max_x, cur_min_y, cur_max_y = state.placed_bbox()
        if gid is None:
            raise ValueError("RewardComposer.delta_batch requires explicit gid")
        w_out, w_in = state.build_delta_flow_weights_for(gid)

        m = 0
        device = placed_entries.device
        for t in (entry_points, exit_points, min_x, max_x, min_y, max_y):
            if t is not None:
                m = int(t.shape[0])
                device = t.device
                break

        total = torch.zeros((m,), dtype=torch.float32, device=device)
        has_flow_like = any(
            isinstance(comp, (FlowReward, FlowCollisionReward))
            for comp in self.components.values()
            if comp is not None
        )
        has_area_like = any(
            isinstance(comp, (AreaReward, GridOccupancyReward))
            for comp in self.components.values()
            if comp is not None
        )

        if has_flow_like:
            entry_points = _require(entry_points, "entry_points")
            exit_points = _require(exit_points, "exit_points")

        # Build port span tensors for candidate and placed facilities
        t_exit_k, t_entry_k = self._port_span_tensors(placed_nodes, device)
        c_exit_k = 1
        c_entry_k = 1
        if self.group_specs is not None and gid is not None:
            c_spec = self.group_specs.get(gid)
            if c_spec is not None:
                c_exit_k = int(getattr(c_spec, "exit_port_span", 1))
                c_entry_k = int(getattr(c_spec, "entry_port_span", 1))

        if has_area_like:
            min_x = _require(min_x, "min_x")
            max_x = _require(max_x, "max_x")
            min_y = _require(min_y, "min_y")
            max_y = _require(max_y, "max_y")

        for name, comp in self.components.items():
            if comp is None:
                continue
            w = float(self.weights.get(name, 1.0))
            if isinstance(comp, FlowReward):
                total = total + w * comp.delta(
                    placed_entries=placed_entries,
                    placed_exits=placed_exits,
                    placed_entries_mask=placed_entries_mask,
                    placed_exits_mask=placed_exits_mask,
                    w_out=w_out,
                    w_in=w_in,
                    candidate_entries=entry_points,
                    candidate_exits=exit_points,
                    candidate_entries_mask=entry_mask,
                    candidate_exits_mask=exit_mask,
                    c_exit_k=c_exit_k,
                    c_entry_k=c_entry_k,
                    t_entry_k=t_entry_k,
                    t_exit_k=t_exit_k,
                )
                continue
            if isinstance(comp, FlowCollisionReward):
                total = total + w * comp.delta(
                    placed_entries=placed_entries,
                    placed_exits=placed_exits,
                    placed_entries_mask=placed_entries_mask,
                    placed_exits_mask=placed_exits_mask,
                    w_out=w_out,
                    w_in=w_in,
                    candidate_entries=entry_points,
                    candidate_exits=exit_points,
                    candidate_entries_mask=entry_mask,
                    candidate_exits_mask=exit_mask,
                    route_blocked=route_blocked,
                    c_exit_k=c_exit_k,
                    c_entry_k=c_entry_k,
                    t_entry_k=t_entry_k,
                    t_exit_k=t_exit_k,
                )
                continue
            if isinstance(comp, AreaReward):
                total = total + w * comp.delta(
                    placed_count=placed_count,
                    cur_min_x=cur_min_x,
                    cur_max_x=cur_max_x,
                    cur_min_y=cur_min_y,
                    cur_max_y=cur_max_y,
                    candidate_min_x=min_x,
                    candidate_max_x=max_x,
                    candidate_min_y=min_y,
                    candidate_max_y=max_y,
                )
                continue
            if isinstance(comp, GridOccupancyReward):
                total = total + w * comp.delta(
                    placed_cell_occupied=placed_cell_occupied,
                    candidate_min_x=min_x,
                    candidate_max_x=max_x,
                    candidate_min_y=min_y,
                    candidate_max_y=max_y,
                )
                continue
            raise TypeError(f"unsupported reward component type: {type(comp).__name__}")
        return total

    def delta(
        self,
        state: "EnvState",
        action_space: "ActionSpace",
        *,
        gid: Optional["GroupId"] = None,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute incremental cost for each candidate in *action_space*.

        Delegates to ``delta_batch`` using ActionSpace fields.
        """
        return self.delta_batch(
            state,
            gid=gid,
            entry_points=action_space.entry_points,
            exit_points=action_space.exit_points,
            entry_mask=action_space.entry_mask,
            exit_mask=action_space.exit_mask,
            min_x=action_space.min_x,
            max_x=action_space.max_x,
            min_y=action_space.min_y,
            max_y=action_space.max_y,
            route_blocked=route_blocked,
            placed_cell_occupied=placed_cell_occupied,
        )
