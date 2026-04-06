from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from group_placement.envs.env import FactoryLayoutEnv
from group_placement.agents.base import Agent, BaseAdapter
from group_placement.envs.action_space import ActionSpace
from group_placement.search.base import (
    BaseSearch,
    BaseSearchConfig,
    DecisionCache,
    ProgressFn,
    SearchOutput,
    collect_top_k,
    track_terminal,
)

@dataclass(frozen=True)
class MCTSConfig(BaseSearchConfig):
    num_simulations: int = 50
    c_puct: float = 2.0
    # Rollout control:
    # - rollout_enabled=True  => run rollouts of `rollout_depth`
    # - rollout_enabled=False => no rollout; evaluate leaf via agent.value()
    rollout_enabled: bool = True
    rollout_depth: int = 5
    # Root Dirichlet noise (parity with legacy).
    dirichlet_epsilon: float = 0.2
    dirichlet_concentration: float = 0.5
    # Root action selection temperature:
    # - 0.0 => argmax by visit count (deterministic)
    # - >0  => sample with p(a) ∝ N(a)^(1/T)
    temperature: float = 0.0
    # Progressive widening (for large action spaces).
    # If enabled, a node expands up to k(s)=ceil(pw_c * N(s)^pw_alpha) children.
    pw_enabled: bool = False
    pw_c: float = 1.5
    pw_alpha: float = 0.5
    pw_min_children: int = 1
    # Cache obs/action_space in each tree node to avoid rebuilding on revisit.
    # True by default: MCTS revisits nodes many times, so caching pays off.
    cache_decision_state: bool = True
    # Top-K tracking: 0이면 비활성화
    track_top_k: int = 0


class _Node:
    def __init__(
        self,
        *,
        decision_cache: DecisionCache,
        priors: torch.Tensor,  # float32 [N]
        action: Optional[int] = None,
        reward: float = 0.0,
        terminal: bool = False,
    ):
        self.decision_cache = decision_cache
        self.obs = decision_cache.obs
        self.action_space = decision_cache.action_space
        self.priors = priors
        self.action = action
        self.reward = float(reward)
        self.terminal = bool(terminal)

        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, "_Node"] = {}

        valid = self.action_space.valid_mask
        self.valid_actions = torch.where(valid.to(dtype=torch.bool, device=priors.device).view(-1))[0].to(dtype=torch.long)

    def _allowed_children(self, cfg: MCTSConfig) -> int:
        if not cfg.pw_enabled:
            return int(self.valid_actions.numel())
        if int(self.valid_actions.numel()) == 0:
            return 0
        n = max(1, int(self.visits))
        k = int(math.ceil(float(cfg.pw_c) * (float(n) ** float(cfg.pw_alpha))))
        k = max(int(cfg.pw_min_children), k)
        return min(int(self.valid_actions.numel()), k)

    def _best_unexpanded_action_by_prior(self) -> int:
        """Pick an unexpanded action with highest prior (deterministic PW expansion)."""
        acts = self.valid_actions
        if int(acts.numel()) == 0:
            return -1
        if not self.children:
            return int(acts[int(torch.argmax(self.priors.index_select(0, acts)).item())].item())
        child_acts = torch.tensor(sorted(int(a) for a in self.children.keys()), dtype=torch.long, device=acts.device)
        pos = torch.searchsorted(acts, child_acts)
        expanded = torch.zeros((int(acts.shape[0]),), dtype=torch.bool, device=acts.device)
        expanded[pos] = True
        pri = self.priors.index_select(0, acts).masked_fill(expanded, float("-inf"))
        if not torch.isfinite(pri).any():
            return -1
        return int(acts[int(torch.argmax(pri).item())].item())

    def best_action(self, c_puct: float) -> int:
        acts = self.valid_actions
        if int(acts.numel()) == 0:
            return -1

        fpu_val = (self.total_value / self.visits) if self.visits > 0 else 0.0
        pri = self.priors.index_select(0, acts)
        q = torch.full_like(pri, float(fpu_val))
        n = torch.zeros((int(acts.shape[0]),), dtype=torch.float32, device=acts.device)
        if self.children:
            sorted_children = sorted((int(a), child) for a, child in self.children.items())
            child_acts = torch.tensor([a for a, _child in sorted_children], dtype=torch.long, device=acts.device)
            pos = torch.searchsorted(acts, child_acts)
            child_q = torch.tensor(
                [float(child.total_value / max(1, child.visits)) for _a, child in sorted_children],
                dtype=torch.float32,
                device=acts.device,
            )
            child_n = torch.tensor(
                [float(child.visits) for _a, child in sorted_children],
                dtype=torch.float32,
                device=acts.device,
            )
            q[pos] = child_q
            n[pos] = child_n
        u = float(c_puct) * pri * math.sqrt(self.visits + 1) / (1.0 + n)
        score = q + u
        return int(acts[int(torch.argmax(score).item())].item())

    def best_action_expanded(self, c_puct: float) -> int:
        """PUCT over expanded children only (used after PW saturation)."""
        if not self.children:
            return -1
        sorted_children = sorted((int(a), child) for a, child in self.children.items())
        child_acts = torch.tensor([a for a, _child in sorted_children], dtype=torch.long, device=self.priors.device)
        q = torch.tensor(
            [float(child.total_value / max(1, child.visits)) for _a, child in sorted_children],
            dtype=torch.float32,
            device=self.priors.device,
        )
        n = torch.tensor(
            [float(child.visits) for _a, child in sorted_children],
            dtype=torch.float32,
            device=self.priors.device,
        )
        p = self.priors.index_select(0, child_acts)
        u = float(c_puct) * p * math.sqrt(self.visits + 1) / (1.0 + n)
        score = q + u
        return int(child_acts[int(torch.argmax(score).item())].item())


class MCTSSearch(BaseSearch):
    """MCTS over decision adapter (Discrete + action_mask + env state)."""

    def __init__(self, *, config: MCTSConfig):
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
            raise ValueError("MCTSSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("MCTSSearch requires adapter.engine. Bind adapter to env before search.")
        root_cache = self._capture_decision_cache(
            engine=engine,
            adapter=adapter,
            obs=obs,
            action_space=root_action_space,
        )

        priors = self._safe_priors(agent=agent, adapter=adapter, obs=obs, action_space=root_action_space)
        priors = self._apply_root_dirichlet(priors=priors, mask=root_action_space.valid_mask)
        root = _Node(
            decision_cache=root_cache,
            priors=priors,
            action=None,
            reward=0.0,
            terminal=False,
        )

        # Local top-k heap
        topk_heap: list = []
        topk_ctr = [0]
        max_k = int(self.config.track_top_k)

        has_callback = progress_fn is not None
        if has_callback:
            n_actions = int(root_action_space.valid_mask.shape[0])
            mask_np = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)

        num_sims = int(self.config.num_simulations)
        for sim in range(num_sims):
            self._simulate(
                engine=engine, adapter=adapter, root=root, agent=agent,
                topk_heap=topk_heap, topk_ctr=topk_ctr, max_k=max_k,
            )

            if has_callback:
                if (sim + 1) % progress_interval == 0 or sim == num_sims - 1:
                    visits = np.zeros(n_actions, dtype=np.int32)
                    values = np.zeros(n_actions, dtype=np.float32)
                    for act, child in root.children.items():
                        if 0 <= act < n_actions:
                            visits[act] = child.visits
                            values[act] = child.total_value / max(1, child.visits)
                    valid_visits = np.where(mask_np, visits, -1)
                    best_a = int(np.argmax(valid_visits))
                    best_v = float(values[best_a]) if best_a < len(values) else 0.0
                    progress_fn(sim + 1, num_sims, visits, values, best_a, best_v)

        if not root.children:
            fallback_action = -1
            valid = root_action_space.valid_mask.to(dtype=torch.bool, device=adapter.device).view(-1)
            if bool(torch.isfinite(priors).any().item()) and int(valid.to(torch.int64).sum().item()) > 0:
                masked_priors = priors.masked_fill(~valid, float("-inf"))
                if bool(torch.isfinite(masked_priors).any().item()):
                    fallback_action = int(torch.argmax(masked_priors).item())
            self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_cache.snapshot)
            return SearchOutput(action=fallback_action, iterations=num_sims, top_k=collect_top_k(topk_heap))

        best_action: int
        temp = float(self.config.temperature)
        if temp <= 0.0:
            best_action = int(max(root.children.items(), key=lambda item: item[1].visits)[0])
        else:
            acts = list(root.children.keys())
            visits_t = torch.tensor([float(root.children[a].visits) for a in acts], dtype=torch.float32, device=adapter.device)
            if float(visits_t.sum().item()) <= 0.0:
                best_action = int(acts[0])
            else:
                w = torch.pow(torch.clamp(visits_t, min=0.0), 1.0 / float(temp))
                s = float(w.sum().item())
                if s <= 0.0:
                    best_action = int(acts[int(torch.argmax(visits_t).item())])
                else:
                    p = w / s
                    idx = int(torch.multinomial(p, num_samples=1).item())
                    best_action = int(acts[idx])

        # Build output arrays
        n_out = int(root_action_space.valid_mask.shape[0])
        visits_out = np.zeros(n_out, dtype=np.int32)
        values_out = np.zeros(n_out, dtype=np.float32)
        for act, child in root.children.items():
            if 0 <= act < n_out:
                visits_out[act] = child.visits
                values_out[act] = child.total_value / max(1, child.visits)

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_cache.snapshot)
        return SearchOutput(
            action=int(best_action),
            visits=visits_out,
            values=values_out,
            iterations=num_sims,
            top_k=collect_top_k(topk_heap),
        )

    def _apply_root_dirichlet(self, *, priors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

        alpha_vec = torch.full((valid_count,), float(alpha), dtype=torch.float32, device=priors.device)
        noise = torch.distributions.Dirichlet(alpha_vec).sample().to(dtype=torch.float32)

        mixed = priors.clone()
        mixed_valid = (1.0 - eps) * mixed[valid] + eps * noise
        total = float(mixed_valid.sum().item())
        if total > 0.0:
            mixed_valid = mixed_valid / total
        mixed[valid] = mixed_valid
        return mixed

    def _safe_priors(
        self,
        *,
        agent: Agent,
        adapter: BaseAdapter,
        obs: dict,
        action_space: ActionSpace,
    ) -> torch.Tensor:
        pri = agent.policy(obs=obs, action_space=action_space)
        if not isinstance(pri, torch.Tensor):
            raise TypeError("Agent.policy must return torch.Tensor")
        pri = pri.to(dtype=torch.float32, device=adapter.device).view(-1)
        if int(pri.shape[0]) != int(action_space.valid_mask.shape[0]):
            out = torch.zeros((int(action_space.valid_mask.shape[0]),), dtype=torch.float32, device=adapter.device)
            valid = action_space.valid_mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                out[valid] = 1.0 / float(cnt)
            return out
        pri = torch.clamp(pri, min=0.0)
        pri = pri.masked_fill(~action_space.valid_mask, 0.0)
        s = float(pri.sum().item())
        if s > 0:
            pri = pri / s
        else:
            valid = action_space.valid_mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                pri = torch.zeros_like(pri)
                pri[valid] = 1.0 / float(cnt)
        return pri

    # ---- main simulation loop ----

    def _simulate(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        root: _Node,
        agent: Agent,
        topk_heap: list,
        topk_ctr: List[int],
        max_k: int,
    ) -> None:
        node = root

        path_nodes = [root]
        path_rewards: List[float] = []

        while not node.terminal:
            if self.config.pw_enabled:
                allowed = node._allowed_children(self.config)
                if len(node.children) < allowed:
                    action = node._best_unexpanded_action_by_prior()
                else:
                    action = node.best_action_expanded(self.config.c_puct)
            else:
                action = node.best_action(self.config.c_puct)
            if action == -1:
                break

            if action in node.children:
                node = node.children[action]
                path_nodes.append(node)
                path_rewards.append(node.reward)
            else:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node.decision_cache.snapshot)

                cand = node.action_space
                reward, terminated, truncated, _info = self._apply_action_index(
                    engine=engine,
                    adapter=adapter,
                    action=int(action),
                    action_space=cand,
                )
                terminal = bool(terminated or truncated)

                if terminal:
                    child_cache = self._capture_terminal_cache(engine=engine, adapter=adapter)
                    next_action_space = child_cache.action_space
                    priors = torch.zeros((0,), dtype=torch.float32, device=adapter.device)
                else:
                    obs2 = adapter.build_observation()
                    next_action_space = adapter.build_action_space()
                    priors = self._safe_priors(
                        agent=agent,
                        adapter=adapter,
                        obs=obs2,
                        action_space=next_action_space,
                    )
                    child_cache = self._capture_decision_cache(
                        engine=engine,
                        adapter=adapter,
                        obs=obs2,
                        action_space=next_action_space,
                    )
                    next_valid = int(next_action_space.valid_mask.to(torch.int64).sum().item())
                    if next_valid <= 0:
                        dead_reward, _dead_term, _dead_trunc, _dead_info = self._apply_action_index(
                            engine=engine,
                            adapter=adapter,
                            action=0,
                            action_space=next_action_space,
                        )
                        reward = float(reward) + float(dead_reward)
                        terminal = True
                        child_cache = self._capture_terminal_cache(engine=engine, adapter=adapter)
                        next_action_space = child_cache.action_space
                        priors = torch.zeros((0,), dtype=torch.float32, device=adapter.device)

                child = _Node(
                    decision_cache=child_cache,
                    priors=priors,
                    action=int(action),
                    reward=float(reward),
                    terminal=terminal or (not terminal and int(next_action_space.valid_mask.to(torch.int64).sum().item()) == 0),
                )
                node.children[int(action)] = child
                node = child
                path_nodes.append(node)
                path_rewards.append(float(reward))

                if child.terminal:
                    cum_reward = sum(path_rewards)
                    topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, cum_reward, max_k)
                break

        leaf_value = 0.0
        if not node.terminal:
            if not bool(self.config.rollout_enabled):
                if bool(self.config.cache_decision_state):
                    leaf_value = float(agent.value(obs=node.obs, action_space=node.action_space))
                else:
                    self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node.decision_cache.snapshot)
                    obs_leaf = adapter.build_observation()
                    leaf_action_space = adapter.build_action_space()
                    leaf_value = float(agent.value(obs=obs_leaf, action_space=leaf_action_space))
            else:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=node.decision_cache.snapshot)
                leaf_value = self._rollout(
                    engine=engine, adapter=adapter, agent=agent,
                    path_reward_offset=float(sum(path_rewards)),
                    initial_cache=node.decision_cache if bool(self.config.cache_decision_state) else None,
                    topk_heap=topk_heap, topk_ctr=topk_ctr, max_k=max_k,
                )

        total = float(leaf_value)
        for reward, path_node in zip(reversed(path_rewards), reversed(path_nodes[1:])):
            total += float(reward)
            path_node.visits += 1
            path_node.total_value += float(total)

        root.visits += 1
        root.total_value += float(total)

    def _rollout(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        agent: Agent,
        path_reward_offset: float = 0.0,
        initial_cache: Optional[DecisionCache] = None,
        topk_heap: list,
        topk_ctr: List[int],
        max_k: int,
    ) -> float:
        total = 0.0
        if not bool(self.config.rollout_enabled):
            return 0.0
        cache = initial_cache
        for _ in range(int(self.config.rollout_depth)):
            if cache is not None:
                obs = cache.obs
                action_space = cache.action_space
                cache = None
            else:
                obs = adapter.build_observation()
                action_space = adapter.build_action_space()
            if int(action_space.valid_mask.to(torch.int64).sum().item()) == 0:
                reward, terminated, truncated, _ = self._apply_action_index(
                    engine=engine,
                    adapter=adapter,
                    action=0,
                    action_space=action_space,
                )
                total += float(reward)
                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, path_reward_offset + total, max_k)
                break

            a = agent.select_action(obs=obs, action_space=action_space)
            reward, terminated, truncated, _ = self._apply_action_index(
                engine=engine,
                adapter=adapter,
                action=int(a),
                action_space=action_space,
            )
            total += float(reward)
            if terminated or truncated:
                topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, path_reward_offset + total, max_k)
                break
        return float(total)


if __name__ == "__main__":
    import time

    from group_placement.envs.env_loader import load_env
    from group_placement.agents.placement.greedy import GreedyAgent, GreedyAdapter

    ENV_JSON = "group_placement/envs/env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyAdapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    root_action_space = adapter.build_action_space()

    agent = GreedyAgent(prior_temperature=1.0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_enabled=True, rollout_depth=5, dirichlet_epsilon=0.0))
    search.set_adapter(adapter)

    t0 = time.perf_counter()
    next_gid = root_action_space.group_id
    output = search.select(obs=obs, agent=agent, root_action_space=root_action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_action_space.valid_mask.sum().item())
    a = output.action
    pose = root_action_space.centers[a].tolist() if int(root_action_space.centers.shape[0]) > 0 else [0, 0]

    print("search.mcts demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_actions=", valid_n, "pose=", pose)
    print(f" elapsed_ms={dt_ms:.2f}")
    print(f" iterations={output.iterations}, visits_sum={output.visits.sum() if output.visits is not None else 0}")
