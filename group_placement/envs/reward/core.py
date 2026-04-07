from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional

import torch

if TYPE_CHECKING:
    from ..action import GroupId
    from ..action_space import ActionSpace
    from ..state.base import EnvState


def _require(x: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if x is None:
        raise ValueError(f"{name} is required but None")
    return x


@dataclass
class TerminalReward:
    """Terminal/failure reward helper based on remaining-area penalty."""

    penalty_weight: float
    reward_scale: float
    group_areas: Mapping[object, float]
    total_area: float = field(init=False)

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
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

    def ratio(self, state: "EnvState") -> float:
        remaining_area = self.remaining_area(state.remaining)
        return self.remaining_area_ratio(
            remaining_area=remaining_area,
            total_area=self.total_area,
        )

    def remaining_area(self, remaining_gids: Iterable[object]) -> float:
        remain = 0.0
        for gid in set(remaining_gids):
            remain += float(self.group_areas.get(gid, 0.0))
        return float(remain)

    def penalty(self, state: "EnvState") -> float:
        return float(self.penalty_weight) * self.ratio(state)

    def failure_reward(self, state: "EnvState") -> float:
        return -self.penalty(state) / float(self.reward_scale)


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

    def score(
        self,
        state: "EnvState",
        *,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        device = placed_entries.device
        total = torch.tensor(0.0, dtype=torch.float32, device=device)

        exit_k, entry_k = self._port_span_tensors(placed_nodes, device)

        flow = self.components.get("flow", None)
        if flow is not None:
            w = float(self.weights.get("flow", 1.0))
            total = total + w * flow.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
            )

        flow_collision = self.components.get("flow_collision", None)
        if flow_collision is not None:
            w = float(self.weights.get("flow_collision", 1.0))
            total = total + w * flow_collision.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                route_blocked=route_blocked,
                exit_k=exit_k,
                entry_k=entry_k,
            )

        area = self.components.get("area", None)
        if area is not None:
            w = float(self.weights.get("area", 1.0))
            min_x_t = torch.tensor(float(cur_min_x), dtype=torch.float32, device=device)
            max_x_t = torch.tensor(float(cur_max_x), dtype=torch.float32, device=device)
            min_y_t = torch.tensor(float(cur_min_y), dtype=torch.float32, device=device)
            max_y_t = torch.tensor(float(cur_max_y), dtype=torch.float32, device=device)
            total = total + w * area.score(
                placed_count=placed_count,
                min_x=min_x_t,
                max_x=max_x_t,
                min_y=min_y_t,
                max_y=max_y_t,
            )

        grid_occ = self.components.get("grid_occupancy", None)
        if grid_occ is not None:
            w = float(self.weights.get("grid_occupancy", 1.0))
            total = total + w * grid_occ.score(
                placed_cell_occupied=placed_cell_occupied,
            )
        return total

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
        flow = self.components.get("flow", None)
        flow_collision = self.components.get("flow_collision", None)
        area = self.components.get("area", None)
        grid_occ = self.components.get("grid_occupancy", None)

        if flow is not None or flow_collision is not None:
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

        if flow is not None:
            w = float(self.weights.get("flow", 1.0))
            total = total + w * flow.delta(
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
        if flow_collision is not None:
            w = float(self.weights.get("flow_collision", 1.0))
            total = total + w * flow_collision.delta(
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

        if area is not None or grid_occ is not None:
            min_x = _require(min_x, "min_x")
            max_x = _require(max_x, "max_x")
            min_y = _require(min_y, "min_y")
            max_y = _require(max_y, "max_y")

        if area is not None:
            w = float(self.weights.get("area", 1.0))
            total = total + w * area.delta(
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
        if grid_occ is not None:
            w = float(self.weights.get("grid_occupancy", 1.0))
            total = total + w * grid_occ.delta(
                placed_cell_occupied=placed_cell_occupied,
                candidate_min_x=min_x,
                candidate_max_x=max_x,
                candidate_min_y=min_y,
                candidate_max_y=max_y,
            )
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
