from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from envs.wrappers.base import BaseWrapper
from envs.wrappers.candidate_set import CandidateSet
from agents.base import Agent
from search.base import BaseSearch, SearchProgress, SearchResult, TopKTracker


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 8
    depth: int = 5
    expansion_topk: int = 16
    track_top_k: int = 0  # 0이면 tracking 비활성화
    track_verbose: bool = False  # True면 리스트 변경 시 print


class BeamSearch(BaseSearch):
    """Beam search over wrapper env (Discrete + action_mask + snapshot)."""

    def __init__(self, *, config: BeamConfig):
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
        total_depth = int(self.config.depth)

        # Only compute numpy arrays if callback is set (avoid overhead when not needed)
        has_callback = self._progress_callback is not None
        if has_callback:
            n_actions = int(root_candidates.mask.shape[0])
            mask_np = root_candidates.mask.detach().cpu().numpy().astype(bool)
            # Track scores for each root action (for progress reporting)
            root_action_scores: Dict[int, float] = {}
        else:
            root_action_scores = None  # type: ignore

        # Each beam item:
        # (cum_reward, first_action, snapshot, obs_at_snapshot)
        beams: List[Tuple[float, int, Dict[str, object], dict]] = [(0.0, -1, root_snapshot, obs)]

        for depth in range(total_depth):
            new_beams: List[Tuple[float, int, Dict[str, object], dict]] = []
            for cum_reward, first_action, snap, obs_node in beams:
                env.set_snapshot(snap)

                # Use provided root candidates at depth=0 on root node to avoid rebuild mismatch.
                if depth == 0 and snap is root_snapshot:
                    candidates = root_candidates
                else:
                    # If wrapper-specific keys are missing (e.g. env returned core obs after truncated),
                    # treat this node as a leaf and do not expand further.
                    if not (isinstance(obs_node, dict) and ("action_mask" in obs_node)):
                        # Track terminal state
                        self._track_if_terminal(env, cum_reward, is_terminal=True)
                        new_beams.append(
                            (
                                cum_reward,
                                first_action if first_action >= 0 else 0,
                                env.get_snapshot(),
                                obs_node,
                            )
                        )
                        continue
                    candidates = self._candidates_from_obs(env, obs_node)
                valid_mask = candidates.mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    # Track terminal state (no valid actions = terminal)
                    self._track_if_terminal(env, cum_reward, is_terminal=True)
                    new_beams.append(
                        (
                            cum_reward,
                            first_action if first_action >= 0 else 0,
                            env.get_snapshot(),
                            obs_node,
                        )
                    )
                    continue

                # IMPORTANT: policy must be computed from the obs corresponding to this node state.
                priors = (
                    agent.policy(env=env.engine, obs=obs_node, candidates=candidates).to(dtype=torch.float32, device=env.device)
                    .view(-1)
                )
                priors = priors.masked_fill(~valid_mask, float("-inf"))

                # Expand only among VALID actions. (Avoid selecting masked actions when valid_n < expansion_topk.)
                topk = min(int(self.config.expansion_topk), int(valid_n))
                if topk <= 0:
                    continue

                top_actions = torch.topk(priors, k=topk).indices.tolist()

                for a in top_actions:
                    a = int(a)
                    if not bool(valid_mask[a].item()):
                        continue
                    env.set_snapshot(snap)
                    obs2, reward, terminated, truncated, _info = env.step(int(a))

                    new_cum = float(cum_reward) + float(reward)
                    root_a = a if first_action < 0 else int(first_action)
                    new_beams.append((new_cum, root_a, env.get_snapshot(), obs2))

                    # Track completed placements
                    if terminated or truncated:
                        self._track_if_terminal(env, new_cum, is_terminal=True)

            if not new_beams:
                break

            new_beams.sort(key=lambda t: t[0], reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            # Emit progress (only if callback is set)
            if has_callback:
                # Update root action scores from current beams
                for cum_reward, root_a, _, _ in beams:
                    if root_a >= 0:
                        if root_a not in root_action_scores or cum_reward > root_action_scores[root_a]:
                            root_action_scores[root_a] = cum_reward
                
                self._emit_beam_progress(
                    depth + 1, total_depth, n_actions, mask_np, root_action_scores, beams
                )

        best = beams[0][1] if beams else 0

        env.set_snapshot(root_snapshot)
        return int(best) if int(best) >= 0 else 0

    def _emit_beam_progress(
        self,
        depth: int,
        total_depth: int,
        n_actions: int,
        mask: np.ndarray,
        root_action_scores: Dict[int, float],
        beams: List[Tuple[float, int, Dict[str, object], dict]],
    ) -> None:
        """Emit progress for beam search."""
        # visits: count how many beams have each root action
        visits = np.zeros(n_actions, dtype=np.int32)
        values = np.zeros(n_actions, dtype=np.float32)
        
        for cum_reward, root_a, _, _ in beams:
            if 0 <= root_a < n_actions:
                visits[root_a] += 1
                values[root_a] = max(values[root_a], cum_reward)
        
        # Also include best scores from root_action_scores
        for root_a, score in root_action_scores.items():
            if 0 <= root_a < n_actions:
                values[root_a] = max(values[root_a], score)
        
        # Find best action
        valid_values = np.where(mask, values, -np.inf)
        best_action = int(np.argmax(valid_values))
        best_value = float(values[best_action]) if best_action < len(values) else 0.0
        
        progress = SearchProgress(
            iteration=depth,
            total=total_depth,
            visits=visits,
            values=values,
            best_action=best_action,
            best_value=best_value,
            extra={"beam_size": len(beams), "active_root_actions": len(root_action_scores)},
        )
        self._emit_progress(progress)

    def _track_if_terminal(self, env: BaseWrapper, cum_reward: float, *, is_terminal: bool) -> None:
        """Track terminal state if tracking is enabled."""
        if self.top_tracker is None or not is_terminal:
            return
        cost = env.engine.total_cost()
        positions = {str(gid): pos for gid, pos in env.engine.positions.items()}
        self.top_tracker.add(SearchResult(
            cost=cost,
            cum_reward=cum_reward,
            positions=positions,
            snapshot=env.get_snapshot(),
        ))

    def _candidates_from_obs(self, env: BaseWrapper, obs: dict) -> CandidateSet:
        mask = obs.get("action_mask", None)
        if not isinstance(mask, torch.Tensor):
            raise ValueError("BeamSearch(wrapper) requires obs['action_mask'] (torch.Tensor)")
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
    env = GreedyWrapperEnv(engine=engine, k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    obs, _info = env.reset(options=loaded.reset_kwargs)

    agent = GreedyAgent(prior_temperature=1.0)
    search = BeamSearch(config=BeamConfig(beam_width=8, depth=3, expansion_topk=16))

    t0 = time.perf_counter()
    next_gid = env.engine.remaining[0] if env.engine.remaining else None
    root_candidates = CandidateSet(xyrot=obs["action_xyrot"], mask=obs["action_mask"], gid=next_gid)
    a = search.select(env=env, obs=obs, agent=agent, root_candidates=root_candidates)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_candidates.mask.sum().item())
    xyrot = root_candidates.xyrot[a].tolist() if int(root_candidates.xyrot.shape[0]) > 0 else [0, 0, 0]

    print("[search.beam demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_candidates=", valid_n, "xyrot=", xyrot)
    print(f" elapsed_ms={dt_ms:.2f}")

