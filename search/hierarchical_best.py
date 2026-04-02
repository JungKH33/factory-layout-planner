from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Optional, Tuple

import numpy as np
import torch

from agents.base import Agent, BaseAdapter
from envs.action_space import ActionSpace
from search.base import (
    BaseHierarchicalSearch,
    BaseSearchConfig,
    SearchProgress,
    SearchSnapshot,
    TopKTracker,
)


@dataclass(frozen=True)
class HierarchicalBestFirstConfig(BaseSearchConfig):
    max_expansions: int = 200
    depth: int = 5
    manager_topk: int = 8
    worker_topk: int = 4
    expansion_batch_size: int = 1
    cache_decision_state: bool = False
    use_value_heuristic: bool = True
    track_top_k: int = 0
    track_verbose: bool = False


@dataclass
class _HierBestItem:
    score: float
    cum_reward: float
    depth: int
    first_manager_action: int
    first_worker_action: int
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


class HierarchicalBestFirstSearch(BaseHierarchicalSearch):
    """Best-first tree search over hierarchical manager/worker actions."""

    def __init__(self, *, config: HierarchicalBestFirstConfig):
        super().__init__()
        self.config = config
        if self.config.track_top_k > 0:
            self.top_tracker = TopKTracker(
                k=self.config.track_top_k,
                verbose=self.config.track_verbose,
            )
        else:
            self.top_tracker = None

    def select_h(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
    ) -> Tuple[int, int]:
        adapter = self.adapter
        if adapter is None:
            raise ValueError("HierarchicalBestFirstSearch.adapter is not set. Call search.set_adapter(...).")
        if not adapter.supports_hierarchical:
            raise TypeError(
                "HierarchicalBestFirstSearch requires adapter with "
                f"supports_hierarchical=True, got {type(adapter).__name__}"
            )
        engine = adapter.engine

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

        frontier: List[Tuple[float, int, _HierBestItem]] = []
        push_order = 0
        heapq.heappush(
            frontier,
            (
                -float(root_score),
                push_order,
                _HierBestItem(
                    score=float(root_score),
                    cum_reward=0.0,
                    depth=0,
                    first_manager_action=-1,
                    first_worker_action=-1,
                    snapshot=root_snapshot,
                    obs=obs,
                    action_space=root_action_space,
                ),
            ),
        )
        push_order += 1

        best_manager_action = -1
        best_worker_action = -1
        best_score = float("-inf")
        expansions = 0

        batch_size = max(1, int(self.config.expansion_batch_size))
        while frontier and expansions < total_expansions:
            budget = int(total_expansions - expansions)
            take_n = min(batch_size, budget, len(frontier))
            node_batch: List[_HierBestItem] = []
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
                    if node.first_manager_action >= 0 and float(node.score) > best_score:
                        best_manager_action = int(node.first_manager_action)
                        best_worker_action = int(node.first_worker_action) if node.first_worker_action >= 0 else 0
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
                    if node.first_manager_action >= 0 and float(node.score) > best_score:
                        best_manager_action = int(node.first_manager_action)
                        best_worker_action = int(node.first_worker_action) if node.first_worker_action >= 0 else 0
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
                manager_topk = min(max(1, int(self.config.manager_topk)), valid_n)
                top_manager_actions = torch.topk(priors, k=manager_topk).indices

                for manager_action in top_manager_actions:
                    manager_action = int(manager_action)

                    self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                    try:
                        worker_as = adapter.sub_action_space(manager_action)
                        worker_candidates = self._top_worker_candidates(
                            adapter=adapter,
                            parent_idx=manager_action,
                            worker_action_space=worker_as,
                        )
                    except (IndexError, ValueError):
                        worker_candidates = []

                    if not worker_candidates:
                        reward = float(engine.failure_penalty())
                        child_cum = float(node.cum_reward) + float(reward)
                        root_manager_action = manager_action if node.first_manager_action < 0 else int(node.first_manager_action)
                        root_worker_action = 0 if node.first_worker_action < 0 else int(node.first_worker_action)
                        score = float(child_cum)
                        self._track_terminal(engine=engine, cum_reward=child_cum)

                        if root_manager_action >= 0 and score > best_score:
                            best_manager_action = int(root_manager_action)
                            best_worker_action = int(root_worker_action)
                            best_score = float(score)

                        if has_callback and 0 <= root_manager_action < n_actions:
                            root_visits[root_manager_action] += 1
                            root_values[root_manager_action] = max(float(root_values[root_manager_action]), float(score))
                        continue

                    for worker_action in worker_candidates:
                        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                        try:
                            placement = adapter.resolve_sub_action(
                                int(worker_action),
                                worker_as,
                                parent_idx=manager_action,
                            )
                            _, reward, terminated, truncated, _info = engine.step_placement(placement)
                        except (IndexError, ValueError):
                            reward = float(engine.failure_penalty())
                            terminated = False
                            truncated = True

                        terminal = bool(terminated or truncated)
                        child_depth = int(node.depth) + 1
                        child_cum = float(node.cum_reward) + float(reward)
                        root_manager_action = manager_action if node.first_manager_action < 0 else int(node.first_manager_action)
                        root_worker_action = int(worker_action) if node.first_worker_action < 0 else int(node.first_worker_action)

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

                        if root_manager_action >= 0 and score > best_score:
                            best_manager_action = int(root_manager_action)
                            best_worker_action = int(root_worker_action)
                            best_score = float(score)

                        if has_callback and 0 <= root_manager_action < n_actions:
                            root_visits[root_manager_action] += 1
                            root_values[root_manager_action] = max(float(root_values[root_manager_action]), float(score))

                        if (not terminal) and child_depth <= max_depth:
                            heapq.heappush(
                                frontier,
                                (
                                    -float(score),
                                    push_order,
                                    _HierBestItem(
                                        score=float(score),
                                        cum_reward=float(child_cum),
                                        depth=child_depth,
                                        first_manager_action=int(root_manager_action),
                                        first_worker_action=int(root_worker_action),
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

        if best_manager_action < 0:
            best_manager_action, best_worker_action = self._fallback_pair(
                adapter=adapter,
                agent=agent,
                obs=obs,
                root_action_space=root_action_space,
            )

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return int(best_manager_action), int(best_worker_action)

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

    def _top_worker_candidates(
        self,
        *,
        adapter: BaseAdapter,
        parent_idx: int,
        worker_action_space: ActionSpace,
    ) -> List[int]:
        valid = worker_action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) <= 0:
            return []

        costs = adapter.sub_action_costs(parent_idx).to(dtype=torch.float32, device=adapter.device).view(-1)
        m = int(valid.shape[0])
        if int(costs.shape[0]) < m:
            padded = torch.full((m,), float("inf"), dtype=torch.float32, device=adapter.device)
            if int(costs.shape[0]) > 0:
                padded[: int(costs.shape[0])] = costs
            costs = padded
        elif int(costs.shape[0]) > m:
            costs = costs[:m]

        valid_costs = costs.index_select(0, valid_idx)
        finite_valid_idx = torch.where(torch.isfinite(valid_costs))[0]
        candidate_idx = valid_idx.index_select(0, finite_valid_idx) if int(finite_valid_idx.numel()) > 0 else valid_idx
        candidate_n = int(candidate_idx.numel())
        if candidate_n <= 0:
            return []

        worker_topk = min(max(1, int(self.config.worker_topk)), candidate_n)
        rank_scores = -costs.index_select(0, candidate_idx)
        top_local = torch.topk(rank_scores, k=worker_topk).indices
        top_idx = candidate_idx.index_select(0, top_local)
        return [int(i) for i in top_idx.detach().cpu().tolist()]

    def _fallback_pair(
        self,
        *,
        adapter: BaseAdapter,
        agent: Agent,
        obs: dict,
        root_action_space: ActionSpace,
    ) -> Tuple[int, int]:
        valid = root_action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) <= 0:
            return 0, 0

        best_manager_action = int(valid_idx[0].item())
        try:
            priors = agent.policy(obs=obs, action_space=root_action_space).to(dtype=torch.float32, device=adapter.device).view(-1)
            priors = priors.masked_fill(~valid, float("-inf"))
            if bool(torch.isfinite(priors).any().item()):
                best_manager_action = int(torch.argmax(priors).item())
        except Exception:
            pass

        try:
            worker_as = adapter.sub_action_space(best_manager_action)
            locals_top = self._top_worker_candidates(
                adapter=adapter,
                parent_idx=best_manager_action,
                worker_action_space=worker_as,
            )
            if locals_top:
                return best_manager_action, int(locals_top[0])
        except Exception:
            pass
        return best_manager_action, 0

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
        self._emit_hbest_progress(
            iteration=iteration,
            total=total,
            visits=visits,
            values=values,
            mask=mask,
            frontier_size=frontier_size,
        )

    def _emit_hbest_progress(
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
                extra={"frontier_size": int(frontier_size), "hierarchical": True},
            )
        )


# Backward-compatible aliases
HierarchicalBestConfig = HierarchicalBestFirstConfig
HierarchicalBestSearch = HierarchicalBestFirstSearch
