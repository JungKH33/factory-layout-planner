from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from agents.base import Agent, BaseAdapter
from envs.action_space import ActionSpace
from search.base import (
    BaseSearch,
    BaseSearchConfig,
    SearchProgress,
    SearchSnapshot,
    TopKTracker,
)


@dataclass(frozen=True)
class BeamConfig(BaseSearchConfig):
    beam_width: int = 8
    depth: int = 5
    expansion_topk: int = 16
    # Cache obs/action_space per beam item. False by default: beam expands
    # width * topk children per depth, so caching all items is memory-heavy.
    cache_decision_state: bool = False
    track_top_k: int = 0  # 0이면 tracking 비활성화
    track_verbose: bool = False  # True면 리스트 변경 시 print


@dataclass
class _BeamItem:
    cum_reward: float
    first_action: int
    snapshot: SearchSnapshot
    obs: Optional[dict] = None
    action_space: Optional[ActionSpace] = None


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
        root_action_space: ActionSpace,
    ) -> int:
        adapter = self.adapter
        if adapter is None:
            raise ValueError("BeamSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("BeamSearch requires adapter.engine. Bind adapter to env before search.")
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
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

        beams: List[_BeamItem] = [
            _BeamItem(
                cum_reward=0.0,
                first_action=-1,
                snapshot=root_snapshot,
                obs=obs,
                action_space=root_action_space,
            )
        ]

        for depth in range(total_depth):
            new_beams: List[_BeamItem] = []
            for beam in beams:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=beam.snapshot)

                if beam.obs is not None and beam.action_space is not None:
                    obs_node = beam.obs
                    action_space = beam.action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                valid_mask = action_space.mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    # Track terminal state (no valid actions = terminal)
                    self._track_terminal(engine=engine, cum_reward=beam.cum_reward)
                    new_beams.append(
                        _BeamItem(
                            cum_reward=beam.cum_reward,
                            first_action=beam.first_action if beam.first_action >= 0 else 0,
                            snapshot=beam.snapshot,
                            obs=obs_node,
                            action_space=action_space,
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

                    self._restore_snapshot(engine=engine, adapter=adapter, snapshot=beam.snapshot)
                    reward, terminated, truncated, _info = self._apply_action_index(
                        engine=engine, adapter=adapter,
                        action=int(a), action_space=action_space,
                    )

                    new_cum = float(beam.cum_reward) + float(reward)
                    root_a = a if beam.first_action < 0 else int(beam.first_action)
                    if terminated or truncated:
                        child_snapshot = SearchSnapshot(
                            engine_state=engine.get_state().copy(),
                            adapter_state={},
                        )
                        child_obs = {}
                        child_action_space = self._empty_action_space(device=adapter.device)
                    else:
                        if bool(self.config.cache_decision_state):
                            child_obs = adapter.build_observation()
                            child_action_space = adapter.build_action_space()
                            child_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
                        else:
                            child_snapshot = SearchSnapshot(
                                engine_state=engine.get_state().copy(),
                                adapter_state={},
                            )
                            child_obs = None
                            child_action_space = None
                    new_beams.append(
                        _BeamItem(
                            cum_reward=new_cum,
                            first_action=root_a,
                            snapshot=child_snapshot,
                            obs=child_obs,
                            action_space=child_action_space,
                        )
                    )

                    if terminated or truncated:
                        self._track_terminal(engine=engine, cum_reward=new_cum)

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item.cum_reward, reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            # Emit progress (only if callback is set)
            if has_callback:
                # Update root action scores from current beams
                for beam in beams:
                    if beam.first_action >= 0:
                        if beam.first_action not in root_action_scores or beam.cum_reward > root_action_scores[beam.first_action]:
                            root_action_scores[beam.first_action] = beam.cum_reward

                self._emit_beam_progress(
                    depth + 1, total_depth, n_actions, mask_np, root_action_scores, beams
                )

        best = beams[0].first_action if beams else 0

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return int(best) if int(best) >= 0 else 0

    def _emit_beam_progress(
        self,
        depth: int,
        total_depth: int,
        n_actions: int,
        mask: np.ndarray,
        root_action_scores: Dict[int, float],
        beams: List[_BeamItem],
    ) -> None:
        """Emit progress for beam search."""
        # visits: count how many beams have each root action
        visits = np.zeros(n_actions, dtype=np.int32)
        values = np.zeros(n_actions, dtype=np.float32)
        
        for beam in beams:
            if 0 <= beam.first_action < n_actions:
                visits[beam.first_action] += 1
                values[beam.first_action] = max(values[beam.first_action], beam.cum_reward)
        
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
