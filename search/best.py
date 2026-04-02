from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Optional, Tuple

import numpy as np
import torch

from agents.base import Agent
from envs.action_space import ActionSpace
from search.base import (
    BaseSearch,
    BaseSearchConfig,
    SearchProgress,
    SearchSnapshot,
    TopKTracker,
)


@dataclass(frozen=True)
class BestFirstConfig(BaseSearchConfig):
    max_expansions: int = 200
    depth: int = 5
    expansion_topk: int = 16
    expansion_batch_size: int = 1
    cache_decision_state: bool = False
    use_value_heuristic: bool = True
    track_top_k: int = 0
    track_verbose: bool = False


@dataclass
class _BestItem:
    score: float
    cum_reward: float
    depth: int
    first_action: int
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


class BestFirstSearch(BaseSearch):
    """Best-first tree search using f = cumulative_reward + value_heuristic."""

    def __init__(self, *, config: BestFirstConfig):
        super().__init__()
        self.config = config
        if self.config.track_top_k > 0:
            self.top_tracker = TopKTracker(
                k=self.config.track_top_k,
                verbose=self.config.track_verbose,
            )
        else:
            self.top_tracker = None

    def select(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
    ) -> int:
        adapter = self.adapter
        if adapter is None:
            raise ValueError("BestFirstSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("BestFirstSearch requires adapter.engine. Bind adapter to env before search.")

        total_expansions = max(1, int(self.config.max_expansions))
        max_depth = max(1, int(self.config.depth))
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
        root_score = self._safe_value(agent=agent, obs=obs, action_space=root_action_space) if bool(self.config.use_value_heuristic) else 0.0

        has_callback = self._progress_callback is not None
        if has_callback:
            n_actions = int(root_action_space.valid_mask.shape[0])
            root_mask = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)
            root_visits = np.zeros((n_actions,), dtype=np.int32)
            root_values = np.full((n_actions,), float("-inf"), dtype=np.float32)
        else:
            n_actions = 0
            root_mask = np.zeros((0,), dtype=bool)
            root_visits = np.zeros((0,), dtype=np.int32)
            root_values = np.zeros((0,), dtype=np.float32)

        frontier: List[Tuple[float, int, _BestItem]] = []
        push_order = 0
        heapq.heappush(
            frontier,
            (
                -float(root_score),
                push_order,
                _BestItem(
                    score=float(root_score),
                    cum_reward=0.0,
                    depth=0,
                    first_action=-1,
                    snapshot=root_snapshot,
                    obs=obs,
                    action_space=root_action_space,
                ),
            ),
        )
        push_order += 1

        best_action = -1
        best_score = float("-inf")
        expansions = 0

        batch_size = max(1, int(self.config.expansion_batch_size))
        while frontier and expansions < total_expansions:
            budget = int(total_expansions - expansions)
            take_n = min(batch_size, budget, len(frontier))
            node_batch: List[_BestItem] = []
            for _ in range(int(take_n)):
                _, _, node = heapq.heappop(frontier)
                node_batch.append(node)

            expandable = []
            for node in node_batch:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node.snapshot)
                node_snapshot = node.snapshot

                if node.obs is not None and node.action_space is not None:
                    obs_node = node.obs
                    action_space = node.action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                    node_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)

                if node.depth >= max_depth:
                    if node.first_action >= 0 and float(node.score) > best_score:
                        best_action = int(node.first_action)
                        best_score = float(node.score)
                    expansions += 1
                    self._maybe_emit_progress(
                        iteration=expansions,
                        total=total_expansions,
                        visits=root_visits,
                        values=root_values,
                        mask=root_mask,
                        frontier_size=len(frontier),
                    )
                    continue

                valid_mask = action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    self._track_terminal(engine=engine, cum_reward=float(node.cum_reward))
                    if node.first_action >= 0 and float(node.score) > best_score:
                        best_action = int(node.first_action)
                        best_score = float(node.score)
                    expansions += 1
                    self._maybe_emit_progress(
                        iteration=expansions,
                        total=total_expansions,
                        visits=root_visits,
                        values=root_values,
                        mask=root_mask,
                        frontier_size=len(frontier),
                    )
                    continue

                expandable.append((node, node_snapshot, obs_node, action_space, valid_mask, valid_n))

            if not expandable:
                continue

            priors_batch = self._policy_many(
                agent=agent,
                obs_batch=[ctx[2] for ctx in expandable],
                action_space_batch=[ctx[3] for ctx in expandable],
                device=adapter.device,
            )

            for (node, node_snapshot, _obs_node, action_space, valid_mask, valid_n), priors in zip(expandable, priors_batch):
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
                    child_cum = float(node.cum_reward) + float(reward)
                    root_action = action if node.first_action < 0 else int(node.first_action)

                    if terminal:
                        child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                        child_obs: Optional[dict] = {}
                        child_action_space: Optional[ActionSpace] = self._empty_action_space(device=adapter.device)
                        value_term = 0.0
                        score = float(child_cum)
                        self._track_terminal(engine=engine, cum_reward=child_cum)
                    else:
                        if bool(self.config.cache_decision_state):
                            child_obs = adapter.build_observation()
                            child_action_space = adapter.build_action_space()
                            child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                        else:
                            child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                            child_obs = None
                            child_action_space = None

                        if bool(self.config.use_value_heuristic):
                            if child_obs is not None and child_action_space is not None:
                                value_term = self._safe_value(
                                    agent=agent,
                                    obs=child_obs,
                                    action_space=child_action_space,
                                )
                            else:
                                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=child_snapshot)
                                eval_obs = adapter.build_observation()
                                eval_action_space = adapter.build_action_space()
                                value_term = self._safe_value(
                                    agent=agent,
                                    obs=eval_obs,
                                    action_space=eval_action_space,
                                )
                                child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                        else:
                            value_term = 0.0
                        score = float(child_cum) + float(value_term)

                    if root_action >= 0 and score > best_score:
                        best_action = int(root_action)
                        best_score = float(score)

                    if has_callback and 0 <= root_action < n_actions:
                        root_visits[root_action] += 1
                        root_values[root_action] = max(float(root_values[root_action]), float(score))

                    if (not terminal) and child_depth <= max_depth:
                        heapq.heappush(
                            frontier,
                            (
                                -float(score),
                                push_order,
                                _BestItem(
                                    score=float(score),
                                    cum_reward=float(child_cum),
                                    depth=int(child_depth),
                                    first_action=int(root_action),
                                    snapshot=child_snapshot,
                                    obs=child_obs,
                                    action_space=child_action_space,
                                ),
                            ),
                        )
                        push_order += 1

                expansions += 1
                self._maybe_emit_progress(
                    iteration=expansions,
                    total=total_expansions,
                    visits=root_visits,
                    values=root_values,
                    mask=root_mask,
                    frontier_size=len(frontier),
                )

        if best_action < 0:
            best_action = self._fallback_action(
                agent=agent,
                obs=obs,
                root_action_space=root_action_space,
                device=adapter.device,
            )

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return int(best_action)

    def _safe_value(
        self,
        *,
        agent: Agent,
        obs: dict,
        action_space: ActionSpace,
    ) -> float:
        try:
            value = agent.value(obs=obs, action_space=action_space)
        except Exception:
            return 0.0
        try:
            return float(value)
        except Exception:
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return float(value.view(-1)[0].item())
            return 0.0

    def _fallback_action(
        self,
        *,
        agent: Agent,
        obs: dict,
        root_action_space: ActionSpace,
        device: torch.device,
    ) -> int:
        valid = root_action_space.valid_mask.to(dtype=torch.bool, device=device).view(-1)
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) <= 0:
            return 0
        try:
            scores = agent.policy(obs=obs, action_space=root_action_space).to(dtype=torch.float32, device=device).view(-1)
            scores = scores.masked_fill(~valid, float("-inf"))
            if bool(torch.isfinite(scores).any().item()):
                return int(torch.argmax(scores).item())
        except Exception:
            pass
        return int(valid_idx[0].item())

    def _maybe_emit_progress(
        self,
        *,
        iteration: int,
        total: int,
        visits: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        frontier_size: int,
    ) -> None:
        if self._progress_callback is None:
            return
        if (iteration % self._progress_interval) != 0 and iteration < total:
            return
        self._emit_best_progress(
            iteration=iteration,
            total=total,
            visits=visits,
            values=values,
            mask=mask,
            frontier_size=frontier_size,
        )

    def _emit_best_progress(
        self,
        *,
        iteration: int,
        total: int,
        visits: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        frontier_size: int,
    ) -> None:
        n_actions = int(visits.shape[0])
        values_out = np.where(np.isfinite(values), values, 0.0).astype(np.float32, copy=False)

        if n_actions > 0:
            valid_values = np.where(mask, values, -np.inf)
            if np.isfinite(valid_values).any():
                best_action = int(np.argmax(valid_values))
                best_value = float(values_out[best_action])
            else:
                valid_visits = np.where(mask, visits, -1)
                best_action = int(np.argmax(valid_visits))
                best_value = 0.0
        else:
            best_action = 0
            best_value = 0.0

        self._emit_progress(
            SearchProgress(
                iteration=iteration,
                total=total,
                visits=visits.copy(),
                values=values_out.copy(),
                best_action=best_action,
                best_value=best_value,
                extra={"frontier_size": int(frontier_size), "hierarchical": False},
            )
        )


# Backward-compatible aliases
BestConfig = BestFirstConfig
BestSearch = BestFirstSearch
