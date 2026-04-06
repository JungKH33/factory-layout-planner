"""2-Level Hierarchical MCTS.

ManagerNode (manager): selects which manager action to take.
WorkerNode (worker): selects which (position, variant) under a manager action.

Manager uses agent priors (softmax over manager-action costs).
Worker uses greedy priors (softmax over within-parent candidate costs).
WorkerNode shares parent ManagerNode's snapshot — manager-action selection doesn't change env.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from group_placement.envs.env import FactoryLayoutEnv
from group_placement.envs.action_space import ActionSpace
from group_placement.agents.base import Agent, BaseAdapter
from group_placement.search.base import (
    BaseHierarchicalSearch,
    BaseSearchConfig,
    DecisionCache,
    ProgressFn,
    SearchOutput,
    SearchSnapshot,
    collect_top_k,
    track_terminal,
)


@dataclass(frozen=True)
class HierarchicalMCTSConfig(BaseSearchConfig):
    num_simulations: int = 200
    c_puct: float = 2.0
    worker_c_puct: float = 2.0
    worker_temperature: float = 0.5
    rollout_enabled: bool = True
    rollout_depth: int = 5
    dirichlet_epsilon: float = 0.2
    dirichlet_concentration: float = 0.5
    temperature: float = 0.0
    # Progressive widening for manager level
    pw_enabled: bool = False
    pw_c: float = 1.5
    pw_alpha: float = 0.5
    # Top-K tracking
    track_top_k: int = 0


def _greedy_worker_priors(costs: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """softmax(-cost / T) for worker-level MCTS priors."""
    if costs.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=costs.device)
    t = max(temperature, 1e-6)
    logits = -costs / t
    logits = logits - logits.max()
    probs = torch.exp(logits)
    s = float(probs.sum().item())
    if s > 0:
        probs = probs / s
    else:
        probs = torch.ones_like(probs) / max(1, probs.numel())
    return probs


class _ManagerNode:
    """Manager level — which action to select."""

    __slots__ = (
        "decision_cache", "priors", "action", "reward",
        "terminal", "visits", "total_value", "children", "valid_actions",
    )

    def __init__(
        self,
        *,
        decision_cache: DecisionCache,
        priors: torch.Tensor,
        action: Optional[int] = None,
        reward: float = 0.0,
        terminal: bool = False,
    ):
        self.decision_cache = decision_cache
        self.priors = priors
        self.action = action
        self.reward = float(reward)
        self.terminal = bool(terminal)
        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, _WorkerNode] = {}

        valid = decision_cache.action_space.valid_mask
        self.valid_actions = torch.where(
            valid.to(dtype=torch.bool, device=priors.device).view(-1)
        )[0].to(dtype=torch.long)

    def best_manager_action(self, c_puct: float, pw_enabled: bool = False, pw_c: float = 1.5, pw_alpha: float = 0.5) -> int:
        """PUCT selection over manager actions. Returns manager_action or -1."""
        acts = self.valid_actions
        if acts.numel() == 0:
            return -1

        # Progressive widening check
        if pw_enabled:
            allowed = max(1, int(math.ceil(pw_c * (max(1, self.visits) ** pw_alpha))))
            allowed = min(int(acts.numel()), allowed)
            if len(self.children) < allowed:
                return self._best_unexpanded_manager_action()
            return self._best_expanded_manager_action(c_puct)

        fpu_val = (self.total_value / self.visits) if self.visits > 0 else 0.0
        pri = self.priors.index_select(0, acts)
        q = torch.full_like(pri, float(fpu_val))
        n = torch.zeros_like(pri)

        if self.children:
            sorted_ch = sorted((int(a), ch) for a, ch in self.children.items())
            ch_acts = torch.tensor([a for a, _ in sorted_ch], dtype=torch.long, device=acts.device)
            pos = torch.searchsorted(acts, ch_acts)
            ch_q = torch.tensor(
                [float(ch.total_value / max(1, ch.visits)) for _, ch in sorted_ch],
                dtype=torch.float32, device=acts.device,
            )
            ch_n = torch.tensor(
                [float(ch.visits) for _, ch in sorted_ch],
                dtype=torch.float32, device=acts.device,
            )
            q[pos] = ch_q
            n[pos] = ch_n

        u = c_puct * pri * math.sqrt(self.visits + 1) / (1.0 + n)
        score = q + u
        return int(acts[int(torch.argmax(score).item())].item())

    def _best_unexpanded_manager_action(self) -> int:
        acts = self.valid_actions
        if not self.children:
            return int(acts[int(torch.argmax(self.priors.index_select(0, acts)).item())].item())
        ch_acts = torch.tensor(sorted(int(a) for a in self.children), dtype=torch.long, device=acts.device)
        pos = torch.searchsorted(acts, ch_acts)
        expanded = torch.zeros(acts.shape[0], dtype=torch.bool, device=acts.device)
        expanded[pos] = True
        pri = self.priors.index_select(0, acts).masked_fill(expanded, float("-inf"))
        if not torch.isfinite(pri).any():
            return -1
        return int(acts[int(torch.argmax(pri).item())].item())

    def _best_expanded_manager_action(self, c_puct: float) -> int:
        if not self.children:
            return -1
        sorted_ch = sorted((int(a), ch) for a, ch in self.children.items())
        ch_acts = torch.tensor([a for a, _ in sorted_ch], dtype=torch.long, device=self.priors.device)
        q = torch.tensor(
            [float(ch.total_value / max(1, ch.visits)) for _, ch in sorted_ch],
            dtype=torch.float32, device=self.priors.device,
        )
        n = torch.tensor(
            [float(ch.visits) for _, ch in sorted_ch],
            dtype=torch.float32, device=self.priors.device,
        )
        p = self.priors.index_select(0, ch_acts)
        u = c_puct * p * math.sqrt(self.visits + 1) / (1.0 + n)
        score = q + u
        return int(ch_acts[int(torch.argmax(score).item())].item())


class _WorkerNode:
    """Worker level — which (pos, variant) under the selected manager action."""

    __slots__ = (
        "manager_action", "priors", "worker_action_space",
        "visits", "total_value", "children",
    )

    def __init__(
        self,
        *,
        manager_action: int,
        priors: torch.Tensor,
        worker_action_space: ActionSpace,
    ):
        self.manager_action = manager_action
        self.priors = priors
        self.worker_action_space = worker_action_space
        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, _ManagerNode] = {}

    def best_worker_action(self, c_puct: float) -> int:
        """PUCT over within-parent candidates. Returns local_cand_idx or -1."""
        M = int(self.priors.shape[0])
        if M == 0:
            return -1

        acts = torch.arange(M, dtype=torch.long, device=self.priors.device)
        fpu_val = (self.total_value / self.visits) if self.visits > 0 else 0.0
        q = torch.full((M,), float(fpu_val), dtype=torch.float32, device=self.priors.device)
        n = torch.zeros((M,), dtype=torch.float32, device=self.priors.device)

        if self.children:
            sorted_ch = sorted((int(a), ch) for a, ch in self.children.items())
            ch_acts = torch.tensor([a for a, _ in sorted_ch], dtype=torch.long, device=self.priors.device)
            ch_q = torch.tensor(
                [float(ch.total_value / max(1, ch.visits)) for _, ch in sorted_ch],
                dtype=torch.float32, device=self.priors.device,
            )
            ch_n = torch.tensor(
                [float(ch.visits) for _, ch in sorted_ch],
                dtype=torch.float32, device=self.priors.device,
            )
            q[ch_acts] = ch_q
            n[ch_acts] = ch_n

        u = c_puct * self.priors * math.sqrt(self.visits + 1) / (1.0 + n)
        score = q + u
        return int(torch.argmax(score).item())


class HierarchicalMCTSSearch(BaseHierarchicalSearch):
    """2-Level MCTS: manager (manager-action selection) + worker (pos/var selection)."""

    def __init__(self, *, config: HierarchicalMCTSConfig):
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
        """Run H-MCTS. Returns SearchOutput with action (manager) and worker_action."""
        adapter = self.adapter
        if adapter is None:
            raise ValueError("adapter not set")
        if not adapter.supports_hierarchical:
            raise TypeError(
                "HierarchicalMCTSSearch requires adapter with "
                f"supports_hierarchical=True, got {type(adapter).__name__}"
            )
        engine = adapter.engine

        root_cache = self._capture_decision_cache(
            engine=engine, adapter=adapter, obs=obs, action_space=root_action_space,
        )
        priors = self._safe_priors(agent=agent, obs=obs, action_space=root_action_space)
        priors = self._apply_dirichlet(priors, root_action_space.valid_mask)
        root = _ManagerNode(decision_cache=root_cache, priors=priors)

        # Local top-k heap
        topk_heap: list = []
        topk_ctr = [0]
        max_k = int(self.config.track_top_k)

        cfg = self.config
        num_sims = int(cfg.num_simulations)
        for _ in range(num_sims):
            self._simulate(
                engine=engine, adapter=adapter, root=root, agent=agent,
                topk_heap=topk_heap, topk_ctr=topk_ctr, max_k=max_k,
            )

        # Select best manager action by visit count
        if not root.children:
            fallback_manager = -1
            valid = root_action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
            if bool(torch.isfinite(priors).any().item()) and int(valid.to(torch.int64).sum().item()) > 0:
                masked_priors = priors.masked_fill(~valid, float("-inf"))
                if bool(torch.isfinite(masked_priors).any().item()):
                    fallback_manager = int(torch.argmax(masked_priors).item())
            self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_cache.snapshot)
            return SearchOutput(
                action=fallback_manager,
                worker_action=-1,
                iterations=num_sims,
                top_k=collect_top_k(topk_heap),
            )

        best_manager_action: int
        temp = float(cfg.temperature)
        if temp <= 0.0:
            best_manager_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        else:
            manager_actions = list(root.children.keys())
            visits = torch.tensor([float(root.children[a].visits) for a in manager_actions], dtype=torch.float32)
            w = torch.pow(torch.clamp(visits, min=0.0), 1.0 / temp)
            s = float(w.sum().item())
            if s <= 0:
                best_manager_action = manager_actions[0]
            else:
                idx = int(torch.multinomial(w / s, 1).item())
                best_manager_action = manager_actions[idx]

        # Select best worker action by visit count
        worker_node = root.children[best_manager_action]
        if not worker_node.children:
            best_worker_action = 0
        else:
            best_worker_action = max(worker_node.children.items(), key=lambda kv: kv[1].visits)[0]

        # Build output arrays
        n_out = int(root_action_space.valid_mask.shape[0])
        visits_out = np.zeros(n_out, dtype=np.int32)
        values_out = np.zeros(n_out, dtype=np.float32)
        for ma, wn in root.children.items():
            if 0 <= ma < n_out:
                visits_out[ma] = wn.visits
                values_out[ma] = wn.total_value / max(1, wn.visits)

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_cache.snapshot)
        return SearchOutput(
            action=best_manager_action,
            worker_action=best_worker_action,
            visits=visits_out,
            values=values_out,
            iterations=num_sims,
            top_k=collect_top_k(topk_heap),
        )

    def _simulate(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        root: _ManagerNode,
        agent: Agent,
        topk_heap: list,
        topk_ctr: List[int],
        max_k: int,
    ) -> None:
        cfg = self.config
        manager_node = root
        path: List[Tuple[_ManagerNode, Optional[_WorkerNode]]] = []
        path_rewards: List[float] = []

        while not manager_node.terminal:
            # Manager: select action
            manager_action = manager_node.best_manager_action(
                cfg.c_puct,
                pw_enabled=cfg.pw_enabled,
                pw_c=cfg.pw_c,
                pw_alpha=cfg.pw_alpha,
            )
            if manager_action == -1:
                break

            # Get or create worker node
            if manager_action not in manager_node.children:
                self._restore_snapshot(
                    engine=engine, adapter=adapter,
                    snapshot=manager_node.decision_cache.snapshot,
                )
                worker_as = adapter.sub_action_space(manager_action)
                worker_priors = _greedy_worker_priors(
                    adapter.sub_action_costs(manager_action),
                    cfg.worker_temperature,
                )
                worker_node = _WorkerNode(
                    manager_action=manager_action,
                    priors=worker_priors,
                    worker_action_space=worker_as,
                )
                manager_node.children[manager_action] = worker_node
            else:
                worker_node = manager_node.children[manager_action]

            # Worker: select action
            worker_action = worker_node.best_worker_action(cfg.worker_c_puct)
            if worker_action == -1:
                break

            if worker_action in worker_node.children:
                # Traverse existing child
                child_step = worker_node.children[worker_action]
                path.append((manager_node, worker_node))
                path_rewards.append(child_step.reward)
                manager_node = child_step
            else:
                # Expand: restore parent snapshot, apply action, create child ManagerNode
                self._restore_snapshot(
                    engine=engine, adapter=adapter,
                    snapshot=manager_node.decision_cache.snapshot,
                )

                worker_as = worker_node.worker_action_space
                try:
                    placement = adapter.resolve_sub_action(
                        worker_action, worker_as, parent_idx=manager_action,
                    )
                    _, reward, terminated, truncated, _info = engine.step_placement(
                        placement,
                    )
                except (IndexError, ValueError):
                    reward = float(engine.failure_penalty())
                    terminated = False
                    truncated = True
                terminal = bool(terminated or truncated)

                if terminal:
                    child_cache = self._capture_terminal_cache(engine=engine, adapter=adapter)
                    child_priors = torch.zeros((0,), dtype=torch.float32, device=adapter.device)
                else:
                    child_obs = adapter.build_observation()
                    child_as = adapter.build_action_space()
                    child_priors = self._safe_priors(agent=agent, obs=child_obs, action_space=child_as)
                    child_cache = self._capture_decision_cache(
                        engine=engine, adapter=adapter,
                        obs=child_obs, action_space=child_as,
                    )
                    next_valid = int(child_as.valid_mask.to(torch.int64).sum().item())
                    if next_valid <= 0:
                        dead_reward, _, _, _ = self._apply_action_index(
                            engine=engine, adapter=adapter,
                            action=0, action_space=child_as,
                        )
                        reward = float(reward) + float(dead_reward)
                        terminal = True
                        child_cache = self._capture_terminal_cache(engine=engine, adapter=adapter)
                        child_priors = torch.zeros((0,), dtype=torch.float32, device=adapter.device)

                child_step = _ManagerNode(
                    decision_cache=child_cache,
                    priors=child_priors,
                    action=worker_action,
                    reward=float(reward),
                    terminal=terminal,
                )
                worker_node.children[worker_action] = child_step

                path.append((manager_node, worker_node))
                path_rewards.append(float(reward))
                manager_node = child_step

                if child_step.terminal:
                    cum_reward = sum(path_rewards)
                    topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, cum_reward, max_k)
                break

        # Leaf evaluation
        leaf_value = 0.0
        if not manager_node.terminal:
            if cfg.rollout_enabled:
                self._restore_snapshot(
                    engine=engine, adapter=adapter,
                    snapshot=manager_node.decision_cache.snapshot,
                )
                leaf_value = self._rollout(
                    engine=engine, adapter=adapter, agent=agent,
                    path_reward_offset=float(sum(path_rewards)),
                    topk_heap=topk_heap, topk_ctr=topk_ctr, max_k=max_k,
                )
            else:
                leaf_value = float(agent.value(
                    obs=manager_node.decision_cache.obs,
                    action_space=manager_node.decision_cache.action_space,
                ))

        # Backup through path — update ALL nodes (leaf → root)
        total = float(leaf_value)

        # Update leaf manager_node
        manager_node.visits += 1
        manager_node.total_value += total

        # Walk backwards: each path entry is (parent_step, worker_node)
        for i in range(len(path) - 1, -1, -1):
            total += path_rewards[i]
            parent_step, worker_node = path[i]
            if worker_node is not None:
                worker_node.visits += 1
                worker_node.total_value += total
            parent_step.visits += 1
            parent_step.total_value += total

    def _rollout(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        agent: Agent,
        path_reward_offset: float = 0.0,
        topk_heap: list,
        topk_ctr: List[int],
        max_k: int,
    ) -> float:
        """Greedy rollout — flat (no hierarchical structure during rollout)."""
        total = 0.0
        for _ in range(int(self.config.rollout_depth)):
            obs = adapter.build_observation()
            action_space = adapter.build_action_space()
            valid_n = int(action_space.valid_mask.to(torch.int64).sum().item())
            if valid_n == 0:
                reward, _, _, _ = self._apply_action_index(
                    engine=engine, adapter=adapter, action=0, action_space=action_space,
                )
                total += float(reward)
                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, path_reward_offset + total, max_k)
                break
            a = agent.select_action(obs=obs, action_space=action_space)
            reward, terminated, truncated, _ = self._apply_action_index(
                engine=engine, adapter=adapter, action=int(a), action_space=action_space,
            )
            total += float(reward)
            if terminated or truncated:
                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, path_reward_offset + total, max_k)
                break
        return float(total)

    # ---- helpers ----

    def _safe_priors(self, *, agent: Agent, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        device = action_space.centers.device if action_space.centers.numel() > 0 else torch.device("cpu")
        pri = agent.policy(obs=obs, action_space=action_space)
        if not isinstance(pri, torch.Tensor):
            N = int(action_space.valid_mask.shape[0])
            return torch.ones(N, dtype=torch.float32, device=device) / max(1, N)
        pri = pri.to(dtype=torch.float32, device=device).view(-1)
        pri = torch.clamp(pri, min=0.0)
        pri = pri.masked_fill(~action_space.valid_mask.to(dtype=torch.bool, device=device), 0.0)
        s = float(pri.sum().item())
        if s > 0:
            pri = pri / s
        else:
            valid = action_space.valid_mask.to(dtype=torch.bool, device=device)
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                pri = torch.zeros_like(pri)
                pri[valid] = 1.0 / float(cnt)
        return pri

    def _apply_dirichlet(self, priors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.dirichlet_epsilon)
        c = float(self.config.dirichlet_concentration)
        if eps <= 0.0 or c <= 0.0:
            return priors
        valid = mask.to(dtype=torch.bool, device=priors.device).view(-1)
        valid_count = int(valid.to(torch.int64).sum().item())
        if valid_count <= 0:
            return priors
        alpha = c / float(valid_count)
        if alpha <= 0.0:
            return priors
        noise = torch.distributions.Dirichlet(
            torch.full((valid_count,), alpha, dtype=torch.float32, device=priors.device)
        ).sample()
        mixed = priors.clone()
        mixed_valid = (1.0 - eps) * mixed[valid] + eps * noise
        total = float(mixed_valid.sum().item())
        if total > 0:
            mixed_valid = mixed_valid / total
        mixed[valid] = mixed_valid
        return mixed
