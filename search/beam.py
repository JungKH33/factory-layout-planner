from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from envs.env import FactoryLayoutEnv
from envs.state import EnvState
from agents.base import Agent, BaseAdapter
from envs.action_space import ActionSpace as CandidateSet
from search.base import BaseSearch, SearchProgress, SearchResult, TopKTracker


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 8
    depth: int = 5
    expansion_topk: int = 16
    track_top_k: int = 0  # 0이면 tracking 비활성화
    track_verbose: bool = False  # True면 리스트 변경 시 print


class BeamSearch(BaseSearch):
    """Beam search over decision adapter (Discrete + action_mask + env state)."""

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
        obs: dict,
        agent: Agent,
        root_action_space: CandidateSet,
    ) -> int:
        adapter = self.adapter
        if adapter is None:
            raise ValueError("BeamSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("BeamSearch requires adapter.engine. Bind adapter to env before search.")
        root_state = self._get_engine_state(engine=engine, adapter=adapter)
        total_depth = int(self.config.depth)

        # Only compute numpy arrays if callback is set (avoid overhead when not needed)
        has_callback = self._progress_callback is not None
        if has_callback:
            n_actions = int(root_action_space.mask.shape[0])
            mask_np = root_action_space.mask.detach().cpu().numpy().astype(bool)
            # Track scores for each root action (for progress reporting)
            root_action_scores: Dict[int, float] = {}
        else:
            root_action_scores = None  # type: ignore

        # Each beam item: (cum_reward, first_action, engine_state)
        beams: List[Tuple[float, int, EnvState]] = [(0.0, -1, root_state)]

        for depth in range(total_depth):
            new_beams: List[Tuple[float, int, EnvState]] = []
            for cum_reward, first_action, state in beams:
                self._set_engine_state(engine=engine, adapter=adapter, engine_state=state)

                # Use provided root action_space at depth=0 on root node to avoid rebuild mismatch.
                if depth == 0 and state is root_state:
                    obs_node = obs
                    action_space = root_action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                valid_mask = action_space.mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    # Track terminal state (no valid actions = terminal)
                    self._track_if_terminal(engine=engine, adapter=adapter, cum_reward=cum_reward, is_terminal=True)
                    new_beams.append(
                        (
                            cum_reward,
                            first_action if first_action >= 0 else 0,
                            self._get_engine_state(engine=engine, adapter=adapter),
                        )
                    )
                    continue

                # IMPORTANT: policy must be computed from the obs corresponding to this node state.
                priors = (
                    agent.policy(obs=obs_node, action_space=action_space).to(
                        dtype=torch.float32, device=adapter.device
                    )
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
                    self._set_engine_state(engine=engine, adapter=adapter, engine_state=state)
                    reward, terminated, truncated, _info = self._apply_action_index(
                        engine=engine,
                        adapter=adapter,
                        action=int(a),
                        action_space=action_space,
                    )

                    new_cum = float(cum_reward) + float(reward)
                    root_a = a if first_action < 0 else int(first_action)
                    new_beams.append((new_cum, root_a, self._get_engine_state(engine=engine, adapter=adapter)))

                    # Track completed placements
                    if terminated or truncated:
                        self._track_if_terminal(engine=engine, adapter=adapter, cum_reward=new_cum, is_terminal=True)

            if not new_beams:
                break

            new_beams.sort(key=lambda t: t[0], reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            # Emit progress (only if callback is set)
            if has_callback:
                # Update root action scores from current beams
                for cum_reward, root_a, _ in beams:
                    if root_a >= 0:
                        if root_a not in root_action_scores or cum_reward > root_action_scores[root_a]:
                            root_action_scores[root_a] = cum_reward

                self._emit_beam_progress(
                    depth + 1, total_depth, n_actions, mask_np, root_action_scores, beams
                )

        best = beams[0][1] if beams else 0

        self._set_engine_state(engine=engine, adapter=adapter, engine_state=root_state)
        return int(best) if int(best) >= 0 else 0

    def _get_engine_state(self, *, engine: FactoryLayoutEnv, adapter: BaseAdapter) -> EnvState:
        """Snapshot current engine state.

        The base implementation only copies the engine state.  Adapter state
        (mask, action_poses, etc.) is intentionally NOT included – it is
        always rebuilt via ``adapter.build_action_space()`` after restoration.
        Subclasses may override to include adapter state if needed.
        """
        return engine.get_state().copy()

    def _set_engine_state(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        engine_state: EnvState,
    ) -> None:
        """Restore engine to a previously captured state.

        Only the engine state is restored.  Adapter state is NOT restored
        here – callers must call ``adapter.build_action_space()`` afterwards
        if they need a consistent adapter.  Subclasses may override to
        include adapter restoration.
        """
        engine.set_state(engine_state)

    def _apply_action_index(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        action: int,
        action_space: CandidateSet,
    ):
        """Apply discrete action index via decode -> engine.step_action.

        Returns (reward, terminated, truncated, info) — no observation.
        Callers build observation via adapter.build_observation() only when needed.
        """
        try:
            placement = adapter.decode_action(int(action), action_space)
        except IndexError:
            return float(engine.failure_penalty()), False, True, {"reason": "action_out_of_range"}
        except ValueError as e:
            reason = "no_valid_actions" if str(e) == "no_valid_actions" else "masked_action"
            return float(engine.failure_penalty()), False, True, {"reason": reason}
        _, reward, terminated, truncated, info = engine.step_action(placement)
        return float(reward), bool(terminated), bool(truncated), info

    def _emit_beam_progress(
        self,
        depth: int,
        total_depth: int,
        n_actions: int,
        mask: np.ndarray,
        root_action_scores: Dict[int, float],
        beams: List[Tuple[float, int, EnvState]],
    ) -> None:
        """Emit progress for beam search."""
        # visits: count how many beams have each root action
        visits = np.zeros(n_actions, dtype=np.int32)
        values = np.zeros(n_actions, dtype=np.float32)
        
        for cum_reward, root_a, _ in beams:
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

    def _track_if_terminal(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        cum_reward: float,
        is_terminal: bool,
    ) -> None:
        """Track terminal state if tracking is enabled."""
        if self.top_tracker is None or not is_terminal:
            return
        cost = engine.total_cost()
        positions = {str(gid): p.position() for gid, p in engine.get_state().placements.items()}
        self.top_tracker.add(SearchResult(
            cost=cost,
            cum_reward=cum_reward,
            positions=positions,
            engine_state=self._get_engine_state(engine=engine, adapter=adapter),
        ))


if __name__ == "__main__":
    import time

    from envs.env_loader import load_env
    from agents.placement.greedy import GreedyAgent, GreedyAdapter

    ENV_JSON = "envs/env_configs/basic_01.json"
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
    search = BeamSearch(config=BeamConfig(beam_width=8, depth=3, expansion_topk=16))
    search.set_adapter(adapter)

    t0 = time.perf_counter()
    next_gid = root_action_space.gid
    a = search.select(obs=obs, agent=agent, root_action_space=root_action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_action_space.mask.sum().item())
    pose = root_action_space.poses[a].tolist() if int(root_action_space.poses.shape[0]) > 0 else [0, 0]

    print("search.beam demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_actions=", valid_n, "pose=", pose)
    print(f" elapsed_ms={dt_ms:.2f}")
