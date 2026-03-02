from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch

from envs.wrappers.base import BaseWrapper
from envs.wrappers.candidate_set import CandidateSet
from agents.base import Agent
from search.base import BaseSearch, SearchProgress, SearchResult, TopKTracker


@dataclass(frozen=True)
class MCTSConfig:
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
    # Top-K tracking: 0이면 비활성화
    track_top_k: int = 0
    track_verbose: bool = False  # True면 리스트 변경 시 print


class _Node:
    def __init__(
        self,
        *,
        snapshot: Dict[str, object],
        candidates: CandidateSet,
        priors: torch.Tensor,  # float32 [N]
        parent: Optional["_Node"] = None,
        action: Optional[int] = None,
        reward: float = 0.0,
        terminal: bool = False,
    ):
        self.snapshot = snapshot
        self.candidates = candidates
        self.priors = priors
        self.parent = parent
        self.action = action
        self.reward = float(reward)
        self.terminal = bool(terminal)

        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, "_Node"] = {}

        valid = candidates.mask
        self.valid_actions = [i for i in range(int(valid.shape[0])) if bool(valid[i].item())]

    def _allowed_children(self, cfg: MCTSConfig) -> int:
        if not cfg.pw_enabled:
            return len(self.valid_actions)
        if not self.valid_actions:
            return 0
        n = max(1, int(self.visits))
        k = int(math.ceil(float(cfg.pw_c) * (float(n) ** float(cfg.pw_alpha))))
        k = max(int(cfg.pw_min_children), k)
        return min(len(self.valid_actions), k)

    def _best_unexpanded_action_by_prior(self) -> int:
        """Pick an unexpanded action with highest prior (deterministic PW expansion)."""
        best_act = -1
        best_p = float("-inf")
        for act in self.valid_actions:
            if act in self.children:
                continue
            p = float(self.priors[act].item()) if act < int(self.priors.shape[0]) else 0.0
            if p > best_p:
                best_p = p
                best_act = act
        return best_act

    def best_action(self, c_puct: float) -> int:
        if not self.valid_actions:
            return -1

        fpu_val = (self.total_value / self.visits) if self.visits > 0 else 0.0
        best_score = float("-inf")
        best_act = -1
        for act in self.valid_actions:
            if act in self.children:
                child = self.children[act]
                q = child.total_value / max(1, child.visits)
                n = child.visits
            else:
                q = fpu_val
                n = 0
            p = float(self.priors[act].item()) if act < int(self.priors.shape[0]) else 0.0
            u = float(c_puct) * p * math.sqrt(self.visits + 1) / (1 + n)
            score = float(q) + float(u)
            if score > best_score:
                best_score = score
                best_act = act
        return best_act

    def best_action_expanded(self, c_puct: float) -> int:
        """PUCT over expanded children only (used after PW saturation)."""
        if not self.children:
            return -1
        best_score = float("-inf")
        best_act = -1
        for act, child in self.children.items():
            q = child.total_value / max(1, child.visits)
            n = child.visits
            p = float(self.priors[int(act)].item()) if int(act) < int(self.priors.shape[0]) else 0.0
            u = float(c_puct) * p * math.sqrt(self.visits + 1) / (1 + n)
            score = float(q) + float(u)
            if score > best_score:
                best_score = score
                best_act = int(act)
        return best_act


class MCTSSearch(BaseSearch):
    """MCTS over wrapper env (Discrete + action_mask + snapshot) using Agent priors."""

    def __init__(self, *, config: MCTSConfig):
        super().__init__()
        self.config = config
        # Initialize top-K tracker if enabled
        if self.config.track_top_k > 0:
            self.top_tracker = TopKTracker(
                k=self.config.track_top_k, verbose=self.config.track_verbose
            )
        else:
            self.top_tracker = None

    def select(
        self,
        *,
        env: BaseWrapper,
        obs: dict,
        agent: Agent,
        root_candidates: CandidateSet,
    ) -> int:
        root_snapshot = env.get_snapshot()

        priors = self._safe_priors(agent=agent, env=env, obs=obs, candidates=root_candidates)
        priors = self._apply_root_dirichlet(priors=priors, mask=root_candidates.mask)
        root = _Node(
            snapshot=root_snapshot,
            candidates=root_candidates,
            priors=priors,
            parent=None,
            action=None,
            reward=0.0,
            terminal=False,
        )

        # Only compute numpy arrays if callback is set (avoid overhead when not needed)
        has_callback = self._progress_callback is not None
        if has_callback:
            n_actions = int(root_candidates.mask.shape[0])
            mask_np = root_candidates.mask.detach().cpu().numpy().astype(bool)

        num_sims = int(self.config.num_simulations)
        for sim in range(num_sims):
            self._simulate(env=env, root=root, agent=agent)
            
            # Emit progress at intervals (only if callback is set)
            if has_callback:
                if (sim + 1) % self._progress_interval == 0 or sim == num_sims - 1:
                    self._emit_mcts_progress(root, sim + 1, num_sims, n_actions, mask_np)

        if not root.children:
            env.set_snapshot(root_snapshot)
            return 0

        best_action: int
        temp = float(self.config.temperature)
        if temp <= 0.0:
            # Deterministic: pick the most visited child.
            best_action = int(max(root.children.items(), key=lambda item: item[1].visits)[0])
        else:
            # Stochastic: sample proportionally to visit counts with temperature.
            acts = list(root.children.keys())
            visits = torch.tensor([float(root.children[a].visits) for a in acts], dtype=torch.float32, device=env.device)
            if float(visits.sum().item()) <= 0.0:
                best_action = int(acts[0])
            else:
                w = torch.pow(torch.clamp(visits, min=0.0), 1.0 / float(temp))
                s = float(w.sum().item())
                if s <= 0.0:
                    best_action = int(acts[int(torch.argmax(visits).item())])
                else:
                    p = w / s
                    idx = int(torch.multinomial(p, num_samples=1).item())
                    best_action = int(acts[idx])
        env.set_snapshot(root_snapshot)
        return int(best_action)

    def _emit_mcts_progress(
        self,
        root: _Node,
        iteration: int,
        total: int,
        n_actions: int,
        mask: np.ndarray,
    ) -> None:
        """Extract visit counts and Q-values from root and emit progress."""
        visits = np.zeros(n_actions, dtype=np.int32)
        values = np.zeros(n_actions, dtype=np.float32)
        
        for act, child in root.children.items():
            if 0 <= act < n_actions:
                visits[act] = child.visits
                values[act] = child.total_value / max(1, child.visits)
        
        # Find best action by visit count among valid actions
        valid_visits = np.where(mask, visits, -1)
        best_action = int(np.argmax(valid_visits))
        best_value = float(values[best_action]) if best_action < len(values) else 0.0
        
        progress = SearchProgress(
            iteration=iteration,
            total=total,
            visits=visits,
            values=values,
            best_action=best_action,
            best_value=best_value,
            extra={"root_visits": root.visits, "root_value": root.total_value / max(1, root.visits)},
        )
        self._emit_progress(progress)

    def _track_terminal(self, env: BaseWrapper, cum_reward: float) -> None:
        """Track terminal state if tracking is enabled."""
        if self.top_tracker is None:
            return
        cost = env.engine.total_cost()
        positions = {str(gid): pos for gid, pos in env.engine.positions.items()}
        self.top_tracker.add(SearchResult(
            cost=cost,
            cum_reward=cum_reward,
            positions=positions,
            snapshot=env.get_snapshot(),
        ))

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

    def _safe_priors(self, *, agent: Agent, env: BaseWrapper, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        pri = agent.policy(env=env.engine, obs=obs, candidates=candidates)
        if not isinstance(pri, torch.Tensor):
            raise TypeError("Agent.policy must return torch.Tensor")
        pri = pri.to(dtype=torch.float32, device=env.device).view(-1)
        if int(pri.shape[0]) != int(candidates.mask.shape[0]):
            out = torch.zeros((int(candidates.mask.shape[0]),), dtype=torch.float32, device=env.device)
            valid = candidates.mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                out[valid] = 1.0 / float(cnt)
            return out
        pri = torch.clamp(pri, min=0.0)
        pri = pri.masked_fill(~candidates.mask, 0.0)
        s = float(pri.sum().item())
        if s > 0:
            pri = pri / s
        else:
            valid = candidates.mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                pri = torch.zeros_like(pri)
                pri[valid] = 1.0 / float(cnt)
        return pri

    def _simulate(self, *, env: BaseWrapper, root: _Node, agent: Agent) -> None:
        node = root
        env.set_snapshot(root.snapshot)

        path_nodes = [root]
        path_rewards: List[float] = []

        while not node.terminal:
            # Progressive widening:
            # - expand new actions until allowed children count is reached
            # - then select among expanded children using PUCT
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
                env.set_snapshot(node.snapshot)
                path_nodes.append(node)
                path_rewards.append(node.reward)
            else:
                env.set_snapshot(node.snapshot)
                cand = node.candidates
                obs2, reward, terminated, truncated, _info = env.step(int(action))
                terminal = bool(terminated or truncated)

                if terminal:
                    next_candidates = CandidateSet(
                        xyrot=torch.zeros((0, 3), dtype=torch.long, device=env.device),
                        mask=torch.zeros((0,), dtype=torch.bool, device=env.device),
                        meta={"terminal": True},
                    )
                    priors = torch.zeros((0,), dtype=torch.float32, device=env.device)
                else:
                    next_candidates = self._candidates_from_obs(env, obs2)
                    priors = self._safe_priors(agent=agent, env=env, obs=obs2, candidates=next_candidates)

                child = _Node(
                    snapshot=env.get_snapshot(),
                    candidates=next_candidates,
                    priors=priors,
                    parent=node,
                    action=int(action),
                    reward=float(reward),
                    terminal=terminal or (not terminal and int(next_candidates.mask.to(torch.int64).sum().item()) == 0),
                )
                node.children[int(action)] = child
                node = child
                path_nodes.append(node)
                path_rewards.append(float(reward))

                # Track terminal state (expansion 시점)
                if child.terminal:
                    cum_reward = sum(path_rewards)
                    self._track_terminal(env, cum_reward)
                break

        leaf_value = 0.0
        if not node.terminal:
            if not bool(self.config.rollout_enabled):
                # Leaf evaluation via value head.
                env.set_snapshot(node.snapshot)
                obs_leaf = env._build_obs()  # type: ignore[attr-defined]
                cand_leaf = self._candidates_from_obs(env, obs_leaf)
                leaf_value = float(agent.value(env=env.engine, obs=obs_leaf, candidates=cand_leaf))
            else:
                leaf_value = self._rollout(env=env, agent=agent)

        total = float(leaf_value)
        for reward, path_node in zip(reversed(path_rewards), reversed(path_nodes[1:])):
            total += float(reward)
            path_node.visits += 1
            path_node.total_value += float(total)

        root.visits += 1
        root.total_value += float(total)

    def _candidates_from_obs(self, env: BaseWrapper, obs: dict) -> CandidateSet:
        mask = obs.get("action_mask", None)
        if not isinstance(mask, torch.Tensor):
            raise ValueError("MCTSSearch(wrapper) requires obs['action_mask'] (torch.Tensor)")
        mask = mask.to(dtype=torch.bool, device=env.device).view(-1)
        A = int(mask.shape[0])

        gid = env.engine.remaining[0] if env.engine.remaining else None
        if "action_xyrot" in obs and isinstance(obs["action_xyrot"], torch.Tensor):
            xyrot = obs["action_xyrot"].to(dtype=torch.long, device=env.device)
            if xyrot.ndim != 2 or int(xyrot.shape[0]) != A or int(xyrot.shape[1]) != 3:
                raise ValueError(f"obs['action_xyrot'] must have shape [A,3], got {tuple(xyrot.shape)} for A={A}")
            return CandidateSet(xyrot=xyrot, mask=mask, gid=gid)

        xyrot = torch.zeros((A, 3), dtype=torch.long, device=env.device)
        for a in range(A):
            x_bl, y_bl, rot, _i, _j = env.decode_action(int(a))  # type: ignore[attr-defined]
            xyrot[a, 0] = int(x_bl)
            xyrot[a, 1] = int(y_bl)
            xyrot[a, 2] = int(rot)
        return CandidateSet(xyrot=xyrot, mask=mask, gid=gid)

    def _rollout(self, *, env: BaseWrapper, agent: Agent) -> float:
        total = 0.0
        if not bool(self.config.rollout_enabled):
            return 0.0
        for _ in range(int(self.config.rollout_depth)):
            obs = env._build_obs()  # type: ignore[attr-defined]
            candidates = self._candidates_from_obs(env, obs)
            if int(candidates.mask.to(torch.int64).sum().item()) == 0:
                _obs2, reward, terminated, truncated, _ = env.step(int(0))
                total += float(reward)
                # Track terminal during rollout
                self._track_terminal(env, total)
                break

            a = agent.select_action(env=env.engine, obs=obs, candidates=candidates)
            _obs2, reward, terminated, truncated, _ = env.step(int(a))
            total += float(reward)
            if terminated or truncated:
                # Track terminal during rollout
                self._track_terminal(env, total)
                break
        return float(total)


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from agents.greedy import GreedyAgent
    from envs.wrappers.greedy import GreedyWrapperEnv

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    wenv = GreedyWrapperEnv(engine=engine, k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    obs, _info = wenv.reset(options=loaded.reset_kwargs)

    agent = GreedyAgent(prior_temperature=1.0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_enabled=True, rollout_depth=5, dirichlet_epsilon=0.0))

    t0 = time.perf_counter()
    next_gid = wenv.engine.remaining[0] if wenv.engine.remaining else None
    root_candidates = CandidateSet(xyrot=obs["action_xyrot"], mask=obs["action_mask"], gid=next_gid)
    a = search.select(
        env=wenv,
        obs=obs,
        agent=agent,
        root_candidates=root_candidates,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_candidates.mask.sum().item())
    xyrot = root_candidates.xyrot[a].tolist() if int(root_candidates.xyrot.shape[0]) > 0 else [0, 0, 0]

    print("[search.mcts demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_candidates=", valid_n, "xyrot=", xyrot)
    print(f" elapsed_ms={dt_ms:.2f}")

