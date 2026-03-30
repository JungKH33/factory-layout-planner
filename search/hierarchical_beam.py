from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base import Agent, BaseHierarchicalAdapter
from envs.action_space import ActionSpace
from search.base import (
    BaseHierarchicalSearch,
    BaseSearchConfig,
    SearchProgress,
    SearchSnapshot,
    TopKTracker,
)


@dataclass(frozen=True)
class HierarchicalBeamConfig(BaseSearchConfig):
    beam_width: int = 8
    depth: int = 5
    manager_topk: int = 8
    worker_topk: int = 4
    cache_decision_state: bool = False
    track_top_k: int = 0
    track_verbose: bool = False


@dataclass
class _HierBeamItem:
    cum_reward: float
    first_cell: int
    first_local: int
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


class HierarchicalBeamSearch(BaseHierarchicalSearch):
    """Beam search over hierarchical adapter (manager cell -> worker candidate)."""

    def __init__(self, *, config: HierarchicalBeamConfig):
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
            raise ValueError("HierarchicalBeamSearch.adapter is not set. Call search.set_adapter(...).")
        if not isinstance(adapter, BaseHierarchicalAdapter):
            raise TypeError(
                "HierarchicalBeamSearch requires BaseHierarchicalAdapter, "
                f"got {type(adapter).__name__}"
            )
        engine = adapter.engine
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
        total_depth = int(self.config.depth)

        has_callback = self._progress_callback is not None
        if has_callback:
            n_actions = int(root_action_space.valid_mask.shape[0])
            mask_np = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)
            root_cell_scores: Dict[int, float] = {}
        else:
            root_cell_scores = {}  # unused when no callback
            n_actions = 0
            mask_np = np.zeros((0,), dtype=bool)

        beams: List[_HierBeamItem] = [
            _HierBeamItem(
                cum_reward=0.0,
                first_cell=-1,
                first_local=-1,
                snapshot=root_snapshot,
                obs=obs,
                action_space=root_action_space,
            )
        ]

        for depth in range(total_depth):
            new_beams: List[_HierBeamItem] = []
            for beam in beams:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=beam.snapshot)
                node_snapshot = beam.snapshot

                if beam.obs is not None and beam.action_space is not None:
                    obs_node = beam.obs
                    action_space = beam.action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                    # In non-cached mode, rebuild node snapshot so worker-level adapter cache matches this state.
                    node_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)

                valid_mask = action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    self._track_terminal(engine=engine, cum_reward=beam.cum_reward)
                    new_beams.append(
                        _HierBeamItem(
                            cum_reward=beam.cum_reward,
                            first_cell=beam.first_cell if beam.first_cell >= 0 else 0,
                            first_local=beam.first_local if beam.first_local >= 0 else 0,
                            snapshot=node_snapshot,
                            obs=obs_node if bool(self.config.cache_decision_state) else None,
                            action_space=action_space if bool(self.config.cache_decision_state) else None,
                        )
                    )
                    continue

                priors = agent.policy(obs=obs_node, action_space=action_space).to(
                    dtype=torch.float32, device=adapter.device
                ).view(-1)
                priors = priors.masked_fill(~valid_mask, float("-inf"))
                manager_topk = min(max(1, int(self.config.manager_topk)), valid_n)
                top_cells = torch.topk(priors, k=manager_topk).indices.tolist()

                for cell_idx in top_cells:
                    cell_idx = int(cell_idx)
                    if not bool(valid_mask[cell_idx].item()):
                        continue

                    self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                    try:
                        worker_as = adapter.cell_action_space(cell_idx)
                        local_candidates = self._top_worker_candidates(
                            adapter=adapter,
                            cell_idx=cell_idx,
                            worker_action_space=worker_as,
                        )
                    except Exception:
                        local_candidates = []

                    if not local_candidates:
                        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                        reward = float(engine.failure_penalty())
                        terminated = False
                        truncated = True
                        terminal = True
                        new_cum = float(beam.cum_reward) + float(reward)
                        root_cell = cell_idx if beam.first_cell < 0 else int(beam.first_cell)
                        root_local = 0 if beam.first_local < 0 else int(beam.first_local)
                        child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                        new_beams.append(
                            _HierBeamItem(
                                cum_reward=new_cum,
                                first_cell=root_cell,
                                first_local=root_local,
                                snapshot=child_snapshot,
                                obs={} if terminal else None,
                                action_space=self._empty_action_space(device=adapter.device) if terminal else None,
                            )
                        )
                        self._track_terminal(engine=engine, cum_reward=new_cum)
                        continue

                    for local_idx in local_candidates:
                        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node_snapshot)
                        worker_as = adapter.cell_action_space(cell_idx)
                        try:
                            placement = adapter.resolve_worker_action(
                                int(local_idx),
                                worker_as,
                                cell_idx=cell_idx,
                            )
                            _, reward, terminated, truncated, _info = engine.step_placement(placement)
                        except (IndexError, ValueError):
                            reward = float(engine.failure_penalty())
                            terminated = False
                            truncated = True
                        terminal = bool(terminated or truncated)

                        new_cum = float(beam.cum_reward) + float(reward)
                        root_cell = cell_idx if beam.first_cell < 0 else int(beam.first_cell)
                        root_local = int(local_idx) if beam.first_local < 0 else int(beam.first_local)

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
                                first_cell=root_cell,
                                first_local=root_local,
                                snapshot=child_snapshot,
                                obs=child_obs,
                                action_space=child_action_space,
                            )
                        )

                        if terminal:
                            self._track_terminal(engine=engine, cum_reward=new_cum)

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item.cum_reward, reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            if has_callback:
                for beam in beams:
                    if beam.first_cell >= 0:
                        best = root_cell_scores.get(beam.first_cell, float("-inf"))
                        if beam.cum_reward > best:
                            root_cell_scores[beam.first_cell] = beam.cum_reward
                self._emit_hbeam_progress(
                    depth=depth + 1,
                    total_depth=total_depth,
                    n_actions=n_actions,
                    mask=mask_np,
                    root_cell_scores=root_cell_scores,
                    beams=beams,
                )

        best_cell = int(beams[0].first_cell) if beams else 0
        best_local = int(beams[0].first_local) if beams else 0

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        if best_cell < 0:
            return 0, 0
        if best_local < 0:
            return best_cell, 0
        return best_cell, best_local

    def _top_worker_candidates(
        self,
        *,
        adapter: BaseHierarchicalAdapter,
        cell_idx: int,
        worker_action_space: ActionSpace,
    ) -> List[int]:
        valid = worker_action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
        valid_n = int(valid.to(torch.int64).sum().item())
        if valid_n <= 0:
            return []

        costs = adapter.worker_costs(cell_idx).to(dtype=torch.float32, device=adapter.device).view(-1)
        m = int(valid.shape[0])
        if int(costs.shape[0]) < m:
            padded = torch.full((m,), float("inf"), dtype=torch.float32, device=adapter.device)
            if int(costs.shape[0]) > 0:
                padded[: int(costs.shape[0])] = costs
            costs = padded
        elif int(costs.shape[0]) > m:
            costs = costs[:m]

        costs = costs.masked_fill(~valid, float("inf"))
        finite_valid = valid & torch.isfinite(costs)
        candidate_mask = finite_valid if bool(finite_valid.any().item()) else valid
        candidate_n = int(candidate_mask.to(torch.int64).sum().item())
        if candidate_n <= 0:
            return []

        worker_topk = min(max(1, int(self.config.worker_topk)), candidate_n)
        rank_scores = (-costs).masked_fill(~candidate_mask, float("-inf"))
        top_local = torch.topk(rank_scores, k=worker_topk).indices
        return [int(i.item()) for i in top_local]

    def _emit_hbeam_progress(
        self,
        *,
        depth: int,
        total_depth: int,
        n_actions: int,
        mask: np.ndarray,
        root_cell_scores: Dict[int, float],
        beams: List[_HierBeamItem],
    ) -> None:
        visits = np.zeros(n_actions, dtype=np.int32)
        values = np.zeros(n_actions, dtype=np.float32)

        for beam in beams:
            if 0 <= beam.first_cell < n_actions:
                visits[beam.first_cell] += 1
                values[beam.first_cell] = max(values[beam.first_cell], float(beam.cum_reward))

        for root_cell, score in root_cell_scores.items():
            if 0 <= root_cell < n_actions:
                values[root_cell] = max(values[root_cell], float(score))

        if n_actions > 0:
            valid_values = np.where(mask, values, -np.inf)
            best_action = int(np.argmax(valid_values))
            best_value = float(values[best_action]) if best_action < len(values) else 0.0
        else:
            best_action = 0
            best_value = 0.0

        self._emit_progress(
            SearchProgress(
                iteration=depth,
                total=total_depth,
                visits=visits,
                values=values,
                best_action=best_action,
                best_value=best_value,
                extra={
                    "beam_size": len(beams),
                    "active_root_cells": len(root_cell_scores),
                    "hierarchical": True,
                },
            )
        )
