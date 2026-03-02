from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.env import FactoryLayoutEnv
from envs.wrappers.candidate_set import CandidateSet
from .base import Agent


@dataclass(frozen=True)
class GreedyAgent:
    """Greedy agent (parity with legacy `agents/greedy.py` logic).

    - Select action: argmin(delta_obj) among valid candidates.
    - Priors: softmax(-delta_obj / prior_temperature) over valid candidates.
    """

    prior_temperature: float = 1.0

    def policy(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        device = env.device
        N = int(candidates.xyrot.shape[0])
        gid = env.remaining[0] if env.remaining else None
        if gid is None:
            return torch.zeros((N,), dtype=torch.float32, device=device)

        priors = torch.zeros((N,), dtype=torch.float32, device=device)
        valid = candidates.mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return priors

        xy = candidates.xyrot[valid_idx].to(device=device)
        scores = env.delta_cost(gid=gid, x=xy[:, 0], y=xy[:, 1], rot=xy[:, 2]).to(dtype=torch.float32, device=device)

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

    def select_action(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> int:
        N = int(candidates.xyrot.shape[0])
        gid = env.remaining[0] if env.remaining else None
        if gid is None or N <= 0:
            return 0

        valid = candidates.mask
        valid_idx = torch.where(valid.view(-1))[0]
        if int(valid_idx.numel()) == 0:
            return 0

        xy = candidates.xyrot[valid_idx].to(device=env.device)
        scores = env.delta_cost(gid=gid, x=xy[:, 0], y=xy[:, 1], rot=xy[:, 2]).to(dtype=torch.float32, device=env.device)
        best_k = int(torch.argmin(scores).item()) if scores.numel() > 0 else 0
        return int(valid_idx[best_k].item()) if int(valid_idx.numel()) > 0 else 0

    def value(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> float:
        # Leaf value for MCTS: 현재 상태의 예상 최종 reward
        return env.terminal_reward()


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from envs.wrappers.greedy import GreedyWrapperEnv
    from envs.wrappers.candidate_set import CandidateSet

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False
    wenv = GreedyWrapperEnv(engine=engine, k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    obs, _info = wenv.reset(options=loaded.reset_kwargs)
    next_gid = wenv.engine.remaining[0] if wenv.engine.remaining else None
    candidates = CandidateSet(xyrot=obs["action_xyrot"], mask=obs["action_mask"], gid=next_gid)
    agent = GreedyAgent(prior_temperature=1.0)

    t0 = time.perf_counter()
    pri = agent.policy(env=engine, obs=obs, candidates=candidates)
    a = agent.select_action(env=engine, obs=obs, candidates=candidates)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(candidates.mask.sum().item())
    xyrot = candidates.xyrot[a].tolist() if int(candidates.xyrot.shape[0]) > 0 else [0, 0, 0]

    print("[agents.greedy demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_candidates=", valid_n, "xyrot=", xyrot, "prior=", (float(pri[a].item()) if pri.numel() > 0 else 0.0))
    print(f" elapsed_ms={dt_ms:.3f}")
