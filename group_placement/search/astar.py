from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from group_placement.agents.base import Agent
from group_placement.envs.action_space import ActionSpace
from group_placement.search.base import (
    BaseSearch,
    BaseSearchConfig,
    ProgressFn,
    SearchOutput,
    SearchSnapshot,
    collect_top_k,
    track_terminal,
)


@dataclass(frozen=True)
class AStarConfig(BaseSearchConfig):
    max_expansions: int = 200
    depth: int = 5
    expansion_topk: int = 16
    cache_decision_state: bool = False
    # False => h=0 (uniform-cost search). Provably finds the lowest-cost
    # terminal reachable within the max_expansions/depth budget.
    #
    # True => h = max(0, -agent.value()). WARNING: this sacrifices A*
    # optimality. agent.value() is a learned/heuristic utility, not a proven
    # lower bound on remaining cost, so the heuristic may be inadmissible.
    # Use only as a search-direction hint when you care more about speed
    # to a good-enough solution than about the optimality guarantee.
    use_value_heuristic: bool = False
    track_top_k: int = 0


@dataclass
class _AStarItem:
    f: float
    g: float
    h: float
    depth: int
    first_action: int
    state_key: Tuple[object, ...]
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


class AStarSearch(BaseSearch):
    """A* tree search over adapter actions with state dedup by best g-score."""

    def __init__(self, *, config: AStarConfig):
        super().__init__()
        self.config = config

    def select(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
        progress_fn: Optional[ProgressFn] = None,
        progress_interval: int = 10,
    ) -> SearchOutput:
        adapter = self.adapter
        if adapter is None:
            raise ValueError("AStarSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("AStarSearch requires adapter.engine. Bind adapter to env before search.")

        total_expansions = max(1, int(self.config.max_expansions))
        max_depth = max(1, int(self.config.depth))
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
        root_key = self._state_key(engine=engine)
        root_h = self._heuristic_cost(agent=agent, obs=obs, action_space=root_action_space)

        # Local top-k heap
        topk_heap: list = []
        topk_ctr = [0]
        max_k = int(self.config.track_top_k)

        has_callback = progress_fn is not None
        n_actions = int(root_action_space.valid_mask.shape[0])
        root_mask = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)
        root_visits = np.zeros((n_actions,), dtype=np.int32)
        # Keep reward-like values for compatibility with existing UI/callbacks:
        # value = -g (higher is better, i.e., lower accumulated cost).
        root_values = np.full((n_actions,), float("-inf"), dtype=np.float32)

        frontier: List[Tuple[float, int, _AStarItem]] = []
        push_order = 0
        heapq.heappush(
            frontier,
            (
                float(root_h),
                push_order,
                _AStarItem(
                    f=float(root_h),
                    g=0.0,
                    h=float(root_h),
                    depth=0,
                    first_action=-1,
                    state_key=root_key,
                    snapshot=root_snapshot,
                    obs=obs,
                    action_space=root_action_space,
                ),
            ),
        )
        push_order += 1

        # Best known g-score per deduplicated state.
        best_g_by_state: Dict[Tuple[object, ...], float] = {root_key: 0.0}

        best_terminal_action = -1
        best_terminal_g = float("inf")
        best_frontier_action = -1
        best_frontier_f = float("inf")

        expansions = 0
        while frontier and expansions < total_expansions:
            popped_f, _, node = heapq.heappop(frontier)
            known_g = best_g_by_state.get(node.state_key, float("inf"))
            if float(node.g) > float(known_g) + 1e-12:
                # Stale heap item superseded by a better path to this state.
                continue

            self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node.snapshot)
            node_snapshot = node.snapshot
            if node.obs is not None and node.action_space is not None:
                obs_node = node.obs
                action_space = node.action_space
            else:
                obs_node = adapter.build_observation()
                action_space = adapter.build_action_space()
                node_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)

            if node.first_action >= 0 and float(popped_f) < best_frontier_f:
                best_frontier_f = float(popped_f)
                best_frontier_action = int(node.first_action)

            if node.depth >= max_depth:
                expansions += 1
                if has_callback and (expansions % progress_interval == 0 or expansions >= total_expansions):
                    self._emit_progress_fn(progress_fn, expansions, total_expansions, root_visits, root_values, root_mask)
                continue

            valid_mask = action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
            valid_n = int(valid_mask.to(torch.int64).sum().item())
            if valid_n <= 0:
                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, -float(node.g), max_k)
                if node.first_action >= 0 and float(node.g) < best_terminal_g:
                    best_terminal_g = float(node.g)
                    best_terminal_action = int(node.first_action)
                expansions += 1
                if has_callback and (expansions % progress_interval == 0 or expansions >= total_expansions):
                    self._emit_progress_fn(progress_fn, expansions, total_expansions, root_visits, root_values, root_mask)
                continue

            priors = self._policy_many(
                agent=agent,
                obs_batch=[obs_node],
                action_space_batch=[action_space],
                device=adapter.device,
            )[0]
            m = int(valid_mask.shape[0])
            if int(priors.shape[0]) < m:
                padded = torch.full((m,), float("-inf"), dtype=torch.float32, device=adapter.device)
                if int(priors.shape[0]) > 0:
                    padded[: int(priors.shape[0])] = priors
                priors = padded
            elif int(priors.shape[0]) > m:
                priors = priors[:m]
            priors = priors.masked_fill(~valid_mask, float("-inf"))
            topk = min(max(1, int(self.config.expansion_topk)), valid_n)
            top_actions = torch.topk(priors, k=topk).indices

            for action in top_actions:
                action = int(action)

                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                reward, terminated, truncated, _info = self._apply_action_index(
                    engine=engine,
                    adapter=adapter,
                    action=action,
                    action_space=action_space,
                )
                terminal = bool(terminated or truncated)
                child_depth = int(node.depth) + 1
                # Reward is defined as -delta_cost/reward_scale in env. Use -reward
                # as step cost so g is accumulated cost (to minimize).
                child_g = float(node.g) + float(-reward)
                root_action = action if node.first_action < 0 else int(node.first_action)

                if 0 <= root_action < n_actions:
                    root_visits[root_action] += 1
                    root_values[root_action] = max(float(root_values[root_action]), float(-child_g))

                if terminal:
                    topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, -float(child_g), max_k)
                    if root_action >= 0 and float(child_g) < best_terminal_g:
                        best_terminal_g = float(child_g)
                        best_terminal_action = int(root_action)
                    if root_action >= 0 and float(child_g) < best_frontier_f:
                        best_frontier_f = float(child_g)
                        best_frontier_action = int(root_action)
                    continue

                if bool(self.config.cache_decision_state):
                    child_obs = adapter.build_observation()
                    child_action_space = adapter.build_action_space()
                    child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                else:
                    child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                    child_obs = None
                    child_action_space = None

                child_key = self._state_key(engine=engine)
                prev_best = best_g_by_state.get(child_key, float("inf"))
                if float(child_g) >= float(prev_best) - 1e-12:
                    continue
                best_g_by_state[child_key] = float(child_g)

                if bool(self.config.use_value_heuristic):
                    if child_obs is not None and child_action_space is not None:
                        child_h = self._heuristic_cost(agent=agent, obs=child_obs, action_space=child_action_space)
                    else:
                        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=child_snapshot)
                        eval_obs = adapter.build_observation()
                        eval_action_space = adapter.build_action_space()
                        child_h = self._heuristic_cost(agent=agent, obs=eval_obs, action_space=eval_action_space)
                        child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                else:
                    child_h = 0.0
                child_f = float(child_g) + float(child_h)

                if root_action >= 0 and float(child_f) < best_frontier_f:
                    best_frontier_f = float(child_f)
                    best_frontier_action = int(root_action)

                heapq.heappush(
                    frontier,
                    (
                        float(child_f),
                        push_order,
                        _AStarItem(
                            f=float(child_f),
                            g=float(child_g),
                            h=float(child_h),
                            depth=child_depth,
                            first_action=int(root_action),
                            state_key=child_key,
                            snapshot=child_snapshot,
                            obs=child_obs,
                            action_space=child_action_space,
                        ),
                    ),
                )
                push_order += 1

            expansions += 1
            if has_callback and (expansions % progress_interval == 0 or expansions >= total_expansions):
                self._emit_progress_fn(progress_fn, expansions, total_expansions, root_visits, root_values, root_mask)

        if best_terminal_action >= 0:
            best_action = int(best_terminal_action)
        elif best_frontier_action >= 0:
            best_action = int(best_frontier_action)
        else:
            best_action = self._fallback_action(
                agent=agent,
                obs=obs,
                root_action_space=root_action_space,
                device=adapter.device,
            )

        values_out = np.where(np.isfinite(root_values), root_values, 0.0).astype(np.float32, copy=False)

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return SearchOutput(
            action=int(best_action),
            visits=root_visits.copy(),
            values=values_out.copy(),
            iterations=expansions,
            top_k=collect_top_k(topk_heap),
        )

    @staticmethod
    def _q(v: object) -> int:
        try:
            return int(round(float(v) * 10000.0))
        except Exception:
            return 0

    def _state_key(self, *, engine) -> Tuple[object, ...]:
        state = engine.get_state()
        remaining_key = tuple(str(gid) for gid in state.remaining)
        placed_order_key = tuple(str(gid) for gid in state.placed_nodes_order)

        placements_key = []
        for gid in sorted(state.placements.keys(), key=lambda x: str(x)):
            p = state.placements[gid]
            x_center = float(
                getattr(
                    p,
                    "x_center",
                    (float(getattr(p, "min_x", 0.0)) + float(getattr(p, "max_x", 0.0))) / 2.0,
                )
            )
            y_center = float(
                getattr(
                    p,
                    "y_center",
                    (float(getattr(p, "min_y", 0.0)) + float(getattr(p, "max_y", 0.0))) / 2.0,
                )
            )
            x_bl = int(getattr(p, "x_bl", int(round(float(getattr(p, "min_x", x_center))))))
            y_bl = int(getattr(p, "y_bl", int(round(float(getattr(p, "min_y", y_center))))))
            placements_key.append(
                (
                    str(gid),
                    int(x_bl),
                    int(y_bl),
                    self._q(x_center),
                    self._q(y_center),
                    int(getattr(p, "variant_index", -1)),
                    int(getattr(p, "rotation", 0)),
                    int(bool(getattr(p, "mirror", False))),
                )
            )
        return (
            int(state.step_count),
            remaining_key,
            placed_order_key,
            tuple(placements_key),
        )

    def _heuristic_cost(
        self,
        *,
        agent: Agent,
        obs: dict,
        action_space: ActionSpace,
    ) -> float:
        if not bool(self.config.use_value_heuristic):
            return 0.0
        value = self._safe_value(agent=agent, obs=obs, action_space=action_space)
        # Agent value is utility (reward-like): higher is better.
        # Convert to non-negative remaining cost estimate for A*.
        return max(0.0, -float(value))

    @staticmethod
    def _emit_progress_fn(
        progress_fn: ProgressFn,
        iteration: int,
        total: int,
        visits: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        n_actions = int(visits.shape[0])
        values_out = np.where(np.isfinite(values), values, 0.0).astype(np.float32, copy=False)
        if n_actions > 0:
            valid_values = np.where(mask, values, -np.inf)
            if np.isfinite(valid_values).any():
                best_a = int(np.argmax(valid_values))
                best_v = float(values_out[best_a])
            else:
                valid_visits = np.where(mask, visits, -1)
                best_a = int(np.argmax(valid_visits))
                best_v = 0.0
        else:
            best_a = 0
            best_v = 0.0
        progress_fn(iteration, total, visits.copy(), values_out.copy(), best_a, best_v)

