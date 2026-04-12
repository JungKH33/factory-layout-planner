from __future__ import annotations

from dataclasses import dataclass

import torch

from lane_generation.envs.action_space import ActionSpace


@dataclass(frozen=True)
class GreedyLaneAgent:
    """Greedy lane agent: pick the lowest-cost candidate route.

    - select_action: argmin(candidate_cost) among valid candidates.
    - policy: softmax(-cost / temperature) over valid candidates.
    - value: heuristic estimate from top-k candidate costs.
    """

    prior_temperature: float = 1.0
    value_topk: int = 3
    value_risk_beta: float = 0.3
    value_min_reward_scale: float = 1e-6

    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        k = int(action_space.valid_mask.shape[0])
        device = action_space.valid_mask.device
        priors = torch.zeros((k,), dtype=torch.float32, device=device)
        valid = action_space.valid_mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return priors

        costs = action_space.candidate_cost
        if not isinstance(costs, torch.Tensor) or int(costs.numel()) != k:
            priors[valid_idx] = 1.0 / float(max(1, int(valid_idx.numel())))
            return priors

        scores = costs.to(dtype=torch.float32, device=device).view(-1)[valid_idx]
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
        k = int(action_space.valid_mask.shape[0])
        if k <= 0:
            return 0

        valid = action_space.valid_mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return 0

        costs = action_space.candidate_cost
        if not (isinstance(costs, torch.Tensor) and int(costs.numel()) == k):
            return int(valid_idx[0].item())

        scores = costs.to(dtype=torch.float32, device=valid.device).view(-1)[valid_idx]
        best_k = int(torch.argmin(scores).item()) if scores.numel() > 0 else 0
        return int(valid_idx[best_k].item()) if int(valid_idx.numel()) > 0 else 0

    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        v = obs.get("state_value", None)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return float(v.view(-1)[0].item())
        if isinstance(v, (float, int)):
            return float(v)

        device = action_space.valid_mask.device
        valid = action_space.valid_mask.to(dtype=torch.bool, device=device).view(-1)
        valid_idx = torch.where(valid)[0]
        valid_n = int(valid_idx.numel())

        step_term = 0.0
        costs = action_space.candidate_cost
        if (
            isinstance(costs, torch.Tensor)
            and int(costs.numel()) == int(valid.numel())
            and valid_n > 0
        ):
            valid_costs = costs.to(dtype=torch.float32, device=device).view(-1)[valid_idx]
            finite = torch.isfinite(valid_costs)
            if bool(finite.any().item()):
                finite_costs = valid_costs[finite]
                topk = min(max(1, int(self.value_topk)), int(finite_costs.numel()))
                top_small = torch.topk(finite_costs, k=topk, largest=False).values
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


if __name__ == "__main__":
    import time

    from lane_generation.envs import load_lane_env, LaneAdapterConfig

    GROUP_PLACEMENT_JSON = "results/inference/sample_placement.json"
    ENV_JSON = "group_placement/envs/env_configs/clearance_03.json"

    device = torch.device("cpu")
    loaded = load_lane_env(
        env_json=ENV_JSON,
        group_placement=GROUP_PLACEMENT_JSON,
        device=device,
        adapter_config=LaneAdapterConfig(candidate_k=8),
    )
    engine = loaded.env
    engine.reset()

    agent = GreedyLaneAgent(prior_temperature=1.0)
    action_space = engine.build_action_space()

    t0 = time.perf_counter()
    pri = agent.policy(obs={}, action_space=action_space)
    a = agent.select_action(obs={}, action_space=action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(action_space.valid_mask.sum().item())
    print("lane_generation.agents.greedy demo")
    print(f"  flow_index={action_space.flow_index} valid_actions={valid_n}")
    print(f"  action={a} prior={float(pri[a].item()) if pri.numel() > 0 else 0.0:.4f}")
    print(f"  elapsed_ms={dt_ms:.3f}")
