from __future__ import annotations

from dataclasses import dataclass

import torch

from group_placement.envs.action_space import ActionSpace
from ...base import Agent


@dataclass(frozen=True)
class GreedyAgent:
    """Greedy agent (parity with legacy `agents/greedy.py` logic).

    - Select action: argmin(delta_obj) among valid action_space.
    - Priors: softmax(-delta_obj / prior_temperature) over valid action_space.
    """

    prior_temperature: float = 1.0
    value_topk: int = 3
    value_risk_beta: float = 0.3
    value_min_reward_scale: float = 1e-6

    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        device = action_space.centers.device
        N = int(action_space.centers.shape[0])
        priors = torch.zeros((N,), dtype=torch.float32, device=device)
        valid = action_space.valid_mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return priors

        scores_obs = obs.get("action_costs", None)
        if isinstance(scores_obs, torch.Tensor) and int(scores_obs.numel()) == N:
            scores = scores_obs.to(dtype=torch.float32, device=device).view(-1)[valid_idx]
        else:
            # Fallback: uniform over valid actions.
            priors[valid_idx] = 1.0 / float(max(1, int(valid_idx.numel())))
            return priors

        temp = float(self.prior_temperature) if float(self.prior_temperature) > 0.0 else 1.0
        logits = -scores / temp
        logits = logits - torch.max(logits)
        probs = torch.exp(logits)
        probs_sum = float(probs.sum().item())
        if probs_sum <= 0.0:
            probs = torch.full_like(probs, 1.0 / float(max(1, int(valid_idx.numel()))))
        else:
            probs = probs / probs_sum

        priors[valid_idx] = probs
        return priors

    def policy_batch(
        self,
        *,
        obs_batch: list[dict],
        action_space_batch: list[ActionSpace],
    ) -> list[torch.Tensor]:
        if len(obs_batch) != len(action_space_batch):
            raise ValueError("obs_batch and action_space_batch length mismatch")
        return [
            self.policy(obs=obs, action_space=action_space)
            for obs, action_space in zip(obs_batch, action_space_batch)
        ]

    def select_action(self, *, obs: dict, action_space: ActionSpace) -> int:
        N = int(action_space.centers.shape[0])
        if N <= 0:
            return 0

        valid = action_space.valid_mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return 0

        scores_obs = obs.get("action_costs", None)
        if not (isinstance(scores_obs, torch.Tensor) and int(scores_obs.numel()) == N):
            return int(valid_idx[0].item())
        scores = scores_obs.to(dtype=torch.float32, device=action_space.centers.device).view(-1)[valid_idx]
        best_k = int(torch.argmin(scores).item()) if scores.numel() > 0 else 0
        return int(valid_idx[best_k].item()) if int(valid_idx.numel()) > 0 else 0

    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        # Optional externally provided scalar estimate takes precedence.
        v = obs.get("state_value", None)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return float(v.view(-1)[0].item())
        if isinstance(v, (float, int)):
            return float(v)

        device = action_space.centers.device
        valid = action_space.valid_mask.to(dtype=torch.bool, device=device).view(-1)
        valid_idx = torch.where(valid)[0]
        valid_n = int(valid_idx.numel())

        step_term = 0.0
        scores_obs = obs.get("action_costs", None)
        if (
            isinstance(scores_obs, torch.Tensor)
            and int(scores_obs.numel()) == int(valid.numel())
            and valid_n > 0
        ):
            costs = scores_obs.to(dtype=torch.float32, device=device).view(-1)
            valid_costs = costs[valid_idx]
            finite = torch.isfinite(valid_costs)
            if bool(finite.any().item()):
                finite_costs = valid_costs[finite]
                k = min(max(1, int(self.value_topk)), int(finite_costs.numel()))
                top_small = torch.topk(finite_costs, k=k, largest=False).values
                mean_cost = float(top_small.mean().item())

                reward_scale = obs.get("reward_scale", None)
                if isinstance(reward_scale, torch.Tensor) and reward_scale.numel() > 0:
                    rs = float(reward_scale.view(-1)[0].item())
                elif isinstance(reward_scale, (float, int)):
                    rs = float(reward_scale)
                else:
                    rs = 1.0
                if rs <= float(self.value_min_reward_scale):
                    rs = 1.0
                step_term = -mean_cost / rs

        failure_penalty = obs.get("failure_penalty", 0.0)
        if isinstance(failure_penalty, torch.Tensor) and failure_penalty.numel() > 0:
            fail = float(failure_penalty.view(-1)[0].item())
        elif isinstance(failure_penalty, (float, int)):
            fail = float(failure_penalty)
        else:
            fail = 0.0

        risk = 1.0 / float(valid_n + 1)
        beta = max(0.0, float(self.value_risk_beta))
        return float(step_term + beta * risk * fail)

    def value_batch(
        self,
        *,
        obs_batch: list[dict],
        action_space_batch: list[ActionSpace],
    ) -> list[float]:
        if len(obs_batch) != len(action_space_batch):
            raise ValueError("obs_batch and action_space_batch length mismatch")
        return [
            self.value(obs=obs, action_space=action_space)
            for obs, action_space in zip(obs_batch, action_space_batch)
        ]


if __name__ == "__main__":
    import time

    from group_placement.envs.env_loader import load_env
    from group_placement.agents.placement.greedy.adapter import GreedyAdapter
    from group_placement.envs.action_space import ActionSpace

    ENV_JSON = "group_placement/envs/env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False
    adapter = GreedyAdapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    action_space = adapter.build_action_space()
    next_gid = action_space.group_id
    agent = GreedyAgent(prior_temperature=1.0)

    t0 = time.perf_counter()
    pri = agent.policy(obs=obs, action_space=action_space)
    a = agent.select_action(obs=obs, action_space=action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(action_space.valid_mask.sum().item())
    pose = action_space.centers[a].tolist() if int(action_space.centers.shape[0]) > 0 else [0, 0]

    print("agents.greedy demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_actions=", valid_n, "pose=", pose, "prior=", (float(pri[a].item()) if pri.numel() > 0 else 0.0))
    print(f" elapsed_ms={dt_ms:.3f}")
