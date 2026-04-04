from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base import Agent, BaseAdapter
from envs.action_space import ActionSpace
from search.base import (
    BaseHierarchicalSearch,
    BaseSearchConfig,
    ProgressFn,
    SearchOutput,
    SearchSnapshot,
    collect_top_k,
    track_terminal,
)


@dataclass(frozen=True)
class HierarchicalBeamConfig(BaseSearchConfig):
    beam_width: int = 8
    depth: int = 5
    manager_topk: int = 8
    worker_topk: int = 4
    cache_decision_state: bool = False
    track_top_k: int = 0


@dataclass
class _HierBeamItem:
    cum_reward: float
    first_manager_action: int
    first_worker_action: int
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


class HierarchicalBeamSearch(BaseHierarchicalSearch):
    """Beam search over hierarchical adapter (manager action -> worker action)."""

    def __init__(self, *, config: HierarchicalBeamConfig):
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
            raise ValueError("HierarchicalBeamSearch.adapter is not set. Call search.set_adapter(...).")
        if not adapter.supports_hierarchical:
            raise TypeError(
                "HierarchicalBeamSearch requires adapter with "
                f"supports_hierarchical=True, got {type(adapter).__name__}"
            )
        engine = adapter.engine
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
        total_depth = int(self.config.depth)

        # Local top-k heap
        topk_heap: list = []
        topk_ctr = [0]
        max_k = int(self.config.track_top_k)

        has_callback = progress_fn is not None
        if has_callback:
            n_actions = int(root_action_space.valid_mask.shape[0])
            mask_np = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)
            root_manager_scores: Dict[int, float] = {}
        else:
            root_manager_scores = {}
            n_actions = 0
            mask_np = np.zeros((0,), dtype=bool)

        beams: List[_HierBeamItem] = [
            _HierBeamItem(
                cum_reward=0.0,
                first_manager_action=-1,
                first_worker_action=-1,
                snapshot=root_snapshot,
                obs=obs,
                action_space=root_action_space,
            )
        ]

        for depth in range(total_depth):
            new_beams: List[_HierBeamItem] = []
            expand_contexts = []
            for beam in beams:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=beam.snapshot)
                node_snapshot = beam.snapshot

                if beam.obs is not None and beam.action_space is not None:
                    obs_node = beam.obs
                    action_space = beam.action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                    node_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)

                valid_mask = action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, beam.cum_reward, max_k)
                    new_beams.append(
                        _HierBeamItem(
                            cum_reward=beam.cum_reward,
                            first_manager_action=beam.first_manager_action if beam.first_manager_action >= 0 else 0,
                            first_worker_action=beam.first_worker_action if beam.first_worker_action >= 0 else 0,
                            snapshot=node_snapshot,
                            obs=obs_node if bool(self.config.cache_decision_state) else None,
                            action_space=action_space if bool(self.config.cache_decision_state) else None,
                        )
                    )
                    continue

                expand_contexts.append((beam, node_snapshot, action_space, valid_mask, int(valid_n), obs_node))

            if expand_contexts:
                priors_batch = self._policy_many(
                    agent=agent,
                    obs_batch=[ctx[5] for ctx in expand_contexts],
                    action_space_batch=[ctx[2] for ctx in expand_contexts],
                    device=adapter.device,
                )

                for (beam, node_snapshot, action_space, valid_mask, valid_n, _obs_node), priors in zip(expand_contexts, priors_batch):
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
                            terminated = False
                            truncated = True
                            terminal = True
                            new_cum = float(beam.cum_reward) + float(reward)
                            root_manager_action = manager_action if beam.first_manager_action < 0 else int(beam.first_manager_action)
                            root_worker_action = 0 if beam.first_worker_action < 0 else int(beam.first_worker_action)
                            child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                            new_beams.append(
                                _HierBeamItem(
                                    cum_reward=new_cum,
                                    first_manager_action=root_manager_action,
                                    first_worker_action=root_worker_action,
                                    snapshot=child_snapshot,
                                    obs={} if terminal else None,
                                    action_space=self._empty_action_space(device=adapter.device) if terminal else None,
                                )
                            )
                            topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, new_cum, max_k)
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

                            new_cum = float(beam.cum_reward) + float(reward)
                            root_manager_action = manager_action if beam.first_manager_action < 0 else int(beam.first_manager_action)
                            root_worker_action = int(worker_action) if beam.first_worker_action < 0 else int(beam.first_worker_action)

                            if terminal:
                                child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                                child_obs: Optional[dict] = {}
                                child_action_space: Optional[ActionSpace] = self._empty_action_space(device=adapter.device)
                            else:
                                if bool(self.config.cache_decision_state):
                                    child_obs = adapter.build_observation()
                                    child_action_space = adapter.build_action_space()
                                    child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                                else:
                                    child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                                    child_obs = None
                                    child_action_space = None

                            new_beams.append(
                                _HierBeamItem(
                                    cum_reward=new_cum,
                                    first_manager_action=root_manager_action,
                                    first_worker_action=root_worker_action,
                                    snapshot=child_snapshot,
                                    obs=child_obs,
                                    action_space=child_action_space,
                                )
                            )

                            if terminal:
                                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, new_cum, max_k)

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item.cum_reward, reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            if has_callback:
                for beam in beams:
                    if beam.first_manager_action >= 0:
                        best = root_manager_scores.get(beam.first_manager_action, float("-inf"))
                        if beam.cum_reward > best:
                            root_manager_scores[beam.first_manager_action] = beam.cum_reward

                visits = np.zeros(n_actions, dtype=np.int32)
                values = np.zeros(n_actions, dtype=np.float32)
                for beam in beams:
                    if 0 <= beam.first_manager_action < n_actions:
                        visits[beam.first_manager_action] += 1
                        values[beam.first_manager_action] = max(values[beam.first_manager_action], float(beam.cum_reward))
                for rma, score in root_manager_scores.items():
                    if 0 <= rma < n_actions:
                        values[rma] = max(values[rma], float(score))
                if n_actions > 0:
                    valid_values = np.where(mask_np, values, -np.inf)
                    best_a = int(np.argmax(valid_values))
                    best_v = float(values[best_a]) if best_a < len(values) else 0.0
                else:
                    best_a = 0
                    best_v = 0.0
                progress_fn(depth + 1, total_depth, visits, values, best_a, best_v)

        best_manager_action = int(beams[0].first_manager_action) if beams else -1
        best_worker_action = int(beams[0].first_worker_action) if beams else -1
        if best_manager_action < 0:
            best_manager_action, best_worker_action = self._fallback_pair(
                adapter=adapter,
                agent=agent,
                obs=obs,
                root_action_space=root_action_space,
            )

        # Build output arrays
        n_out = int(root_action_space.valid_mask.shape[0])
        visits_out = np.zeros(n_out, dtype=np.int32)
        values_out = np.zeros(n_out, dtype=np.float32)
        for beam in beams:
            if 0 <= beam.first_manager_action < n_out:
                visits_out[beam.first_manager_action] += 1
                values_out[beam.first_manager_action] = max(values_out[beam.first_manager_action], beam.cum_reward)

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return SearchOutput(
            action=best_manager_action,
            worker_action=best_worker_action,
            visits=visits_out,
            values=values_out,
            iterations=total_depth,
            top_k=collect_top_k(topk_heap),
        )

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
            return -1, -1

        best_ma = int(valid_idx[0].item())
        try:
            priors = agent.policy(obs=obs, action_space=root_action_space).to(dtype=torch.float32, device=adapter.device).view(-1)
            priors = priors.masked_fill(~valid, float("-inf"))
            if bool(torch.isfinite(priors).any().item()):
                best_ma = int(torch.argmax(priors).item())
        except Exception:
            pass

        try:
            worker_as = adapter.sub_action_space(best_ma)
            locals_top = self._top_worker_candidates(
                adapter=adapter,
                parent_idx=best_ma,
                worker_action_space=worker_as,
            )
            if locals_top:
                return best_ma, int(locals_top[0])
        except Exception:
            pass
        return best_ma, -1

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
