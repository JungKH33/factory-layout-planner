from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.action_space import ActionSpace
from ...base import Agent


@dataclass(frozen=True)
class GreedyAgent:
    """Greedy agent (parity with legacy `agents/greedy.py` logic).

    - Select action: argmin(delta_obj) among valid action_space.
    - Priors: softmax(-delta_obj / prior_temperature) over valid action_space.
    """

    prior_temperature: float = 1.0

    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        device = action_space.poses.device
        N = int(action_space.poses.shape[0])
        priors = torch.zeros((N,), dtype=torch.float32, device=device)
        valid = action_space.mask
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

    def select_action(self, *, obs: dict, action_space: ActionSpace) -> int:
        N = int(action_space.poses.shape[0])
        if N <= 0:
            return 0

        valid = action_space.mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return 0

        scores_obs = obs.get("action_costs", None)
        if not (isinstance(scores_obs, torch.Tensor) and int(scores_obs.numel()) == N):
            return int(valid_idx[0].item())
        scores = scores_obs.to(dtype=torch.float32, device=action_space.poses.device).view(-1)[valid_idx]
        best_k = int(torch.argmin(scores).item()) if scores.numel() > 0 else 0
        return int(valid_idx[best_k].item()) if int(valid_idx.numel()) > 0 else 0

    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        # Optional adapter-provided scalar estimate.
        v = obs.get("state_value", None)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return float(v.view(-1)[0].item())
        if isinstance(v, (float, int)):
            return float(v)
        return 0.0


if __name__ == "__main__":
    import time

    from envs.env_loader import load_env
    from agents.placement.greedy.adapter import GreedyAdapter
    from envs.action_space import ActionSpace

    ENV_JSON = "envs/env_configs/basic_01.json"
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
    next_gid = action_space.gid
    agent = GreedyAgent(prior_temperature=1.0)

    t0 = time.perf_counter()
    pri = agent.policy(obs=obs, action_space=action_space)
    a = agent.select_action(obs=obs, action_space=action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(action_space.mask.sum().item())
    pose = action_space.poses[a].tolist() if int(action_space.poses.shape[0]) > 0 else [0, 0]

    print("agents.greedy demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_actions=", valid_n, "pose=", pose, "prior=", (float(pri[a].item()) if pri.numel() > 0 else 0.0))
    print(f" elapsed_ms={dt_ms:.3f}")
