from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import torch

from .action import LaneAction, LaneRoute
from .action_space import ActionSpace
from .adapter import BaseLaneAdapter
from .reward import LaneNewEdgeReward, LanePathLengthReward, LaneTurnReward, RewardComposer, TerminalReward
from .state import LaneFlowSpec, LaneState, RoutingConfig

logger = logging.getLogger(__name__)


class FactoryLaneEnv(gym.Env):
    """Sequential lane-generation env (one routed flow per step)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_width: int,
        grid_height: int,
        blocked_static: torch.Tensor,
        flows: Sequence[LaneFlowSpec],
        device: Optional[torch.device] = None,
        flow_ordering: str = "weight_desc",
        routing_config: Optional[RoutingConfig] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 100.0,
        reward_weights: Optional[Dict[str, float]] = None,
        penalty_weight: float = 50000.0,
        log: bool = False,
    ) -> None:
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.device = torch.device(
            device or (blocked_static.device if torch.is_tensor(blocked_static) else "cpu")
        )
        self.max_steps = max_steps
        self.log = bool(log)

        self._state = LaneState.build(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            blocked_static=blocked_static,
            flows=flows,
            device=self.device,
            flow_ordering=flow_ordering,
        )

        self._state.configure_routing(routing_config or RoutingConfig())

        rw = dict(reward_weights or {})
        self._reward = RewardComposer(
            components={
                "path_length": LanePathLengthReward(),
                "turn": LaneTurnReward(),
                "new_edge": LaneNewEdgeReward(),
            },
            weights={
                "path_length": float(rw.get("path_length", 0.0)),
                "turn": float(rw.get("turn", 0.0)),
                "new_edge": float(rw.get("new_edge", 1.0)),
            },
            reward_scale=float(reward_scale),
        )
        self._terminal = TerminalReward(
            penalty_weight=float(penalty_weight),
            reward_scale=float(reward_scale),
        )

        self.adapter: Optional[BaseLaneAdapter] = None
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict({})

    def set_adapter(self, adapter: BaseLaneAdapter) -> None:
        self.adapter = adapter
        self.adapter.bind(self)

    def get_state(self) -> LaneState:
        return self._state

    def set_state(self, state: LaneState) -> None:
        if not isinstance(state, LaneState):
            raise TypeError(f"state must be LaneState, got {type(state).__name__}")
        self._state.restore(state)

    @property
    def reward_composer(self) -> RewardComposer:
        return self._reward

    def build_action_space(self) -> ActionSpace:
        if self.adapter is None:
            raise RuntimeError("FactoryLaneEnv.adapter is not set")
        return self.adapter.build_action_space()

    def _normalize_action(self, action: LaneAction) -> Tuple[int, int]:
        if not isinstance(action, LaneAction):
            raise TypeError(f"expected LaneAction, got {type(action).__name__}")
        cur = self._state.current_flow_index()
        if cur is None:
            raise ValueError("no remaining flows")
        flow_idx = int(cur if action.flow_index is None else action.flow_index)
        return flow_idx, int(action.candidate_index)

    def resolve_action(
        self,
        action: LaneAction,
        *,
        action_space: Optional[ActionSpace] = None,
    ) -> Tuple[int, Optional[LaneRoute], ActionSpace]:
        flow_idx, candidate_idx = self._normalize_action(action)
        space = action_space if action_space is not None else self.build_action_space()
        if int(space.flow_index) != int(flow_idx):
            return flow_idx, None, space
        if self.adapter is None:
            raise RuntimeError("FactoryLaneEnv.adapter is not set")
        route = self.adapter.resolve_action(candidate_idx, space)
        return flow_idx, route, space

    def cost(self) -> float:
        return float(self._reward.score(self._state).item())

    def total_cost(self) -> float:
        return float(self.cost()) + float(self._terminal.penalty(self._state))

    def failure_penalty(self) -> float:
        return self._terminal.failure_reward(self._state)

    def _fail(self, reason: str):
        reward = float(self.failure_penalty())
        return {}, reward, False, True, {"reason": str(reason)}

    def step_route(self, route: LaneRoute) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        if self._state.done:
            return {}, 0.0, True, False, {"reason": "done"}

        e = route.edge_indices.to(device=self.device, dtype=torch.long).view(-1)
        l = int(e.numel())
        edge_idx = torch.zeros((1, max(1, l)), dtype=torch.long, device=self.device)
        edge_mask = torch.zeros((1, max(1, l)), dtype=torch.bool, device=self.device)
        if l > 0:
            edge_idx[0, :l] = e
            edge_mask[0, :l] = True
        turns = torch.tensor([float(route.turns)], dtype=torch.float32, device=self.device)
        delta_cost = self._reward.delta_batch(
            self._state,
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            candidate_turns=turns,
        )[0]
        if not bool(torch.isfinite(delta_cost).item()):
            return self._fail("invalid_route")

        self._state.step(apply=True, route=route)
        reward = float(self._reward.to_reward(delta_cost).item())
        terminated = self._state.done
        truncated = self.max_steps is not None and int(self._state.step_count) >= int(self.max_steps)
        return {}, reward, bool(terminated), bool(truncated), {"reason": "routed"}

    def step_action(
        self,
        action: LaneAction,
        *,
        action_space: Optional[ActionSpace] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        if self._state.done:
            return {}, 0.0, True, False, {"reason": "done"}

        try:
            flow_idx, route, _space = self.resolve_action(action, action_space=action_space)
        except Exception:
            return self._fail("invalid_action_payload")

        cur = self._state.current_flow_index()
        if cur is None or int(flow_idx) != int(cur):
            return self._fail("flow_index_mismatch")
        if route is None:
            return self._fail("invalid_candidate")

        return self.step_route(route)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = dict(options or {})
        self._state.reset_runtime()

        initial_edges = options.get("initial_edges", None)
        if initial_edges is not None:
            t = torch.as_tensor(initial_edges, dtype=torch.long, device=self.device).view(-1)
            valid = (t >= 0) & (t < int(self._state.edge_count))
            self._state.apply_edges(t[valid])

        return {}, {}

    def step(self, action: LaneAction):
        if not isinstance(action, LaneAction):
            raise TypeError(f"FactoryLaneEnv.step expects LaneAction, got {type(action).__name__}")
        return self.step_action(action)
