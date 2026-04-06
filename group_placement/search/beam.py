from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from group_placement.agents.base import Agent, BaseAdapter
from group_placement.envs.action_space import ActionSpace
from group_placement.search.base import (
    BaseSearch,
    BaseSearchConfig,
    ProgressFn,
    SearchOutput,
    SearchSnapshot,
    collect_top_k,
    track_terminal,
)


@dataclass(frozen=True)
class BeamConfig(BaseSearchConfig):
    beam_width: int = 8
    depth: int = 5
    expansion_topk: int = 16
    # Cache obs/action_space per beam item. False by default: beam expands
    # width * topk children per depth, so caching all items is memory-heavy.
    cache_decision_state: bool = False
    track_top_k: int = 0


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
            raise ValueError("BeamSearch.adapter is not set. Call search.set_adapter(...).")
        engine = getattr(adapter, "engine", None)
        if engine is None:
            raise ValueError("BeamSearch requires adapter.engine. Bind adapter to env before search.")
        root_snapshot = self._capture_snapshot(engine=engine, adapter=adapter)
        total_depth = int(self.config.depth)

        # Local top-k heap
        topk_heap: list = []
        topk_ctr = [0]
        max_k = int(self.config.track_top_k)

        has_callback = progress_fn is not None
        if has_callback:
            n_actions = int(root_action_space.valid_mask.shape[0])
            mask_np = root_action_space.valid_mask.detach().cpu().numpy().astype(bool)
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
            expand_contexts = []
            for beam in beams:
                self._restore_snapshot(engine=engine, adapter=adapter, snapshot=beam.snapshot)

                if beam.obs is not None and beam.action_space is not None:
                    obs_node = beam.obs
                    action_space = beam.action_space
                else:
                    obs_node = adapter.build_observation()
                    action_space = adapter.build_action_space()
                valid_mask = action_space.valid_mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, beam.cum_reward, max_k)
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

                expand_contexts.append((beam, action_space, valid_mask, int(valid_n), obs_node))

            if expand_contexts:
                priors_batch = self._policy_many(
                    agent=agent,
                    obs_batch=[ctx[4] for ctx in expand_contexts],
                    action_space_batch=[ctx[1] for ctx in expand_contexts],
                    device=adapter.device,
                )

                for (beam, action_space, valid_mask, valid_n, _obs_node), priors in zip(expand_contexts, priors_batch):
                    m = int(valid_mask.shape[0])
                    if int(priors.shape[0]) < m:
                        padded = torch.full((m,), float("-inf"), dtype=torch.float32, device=adapter.device)
                        if int(priors.shape[0]) > 0:
                            padded[: int(priors.shape[0])] = priors
                        priors = padded
                    elif int(priors.shape[0]) > m:
                        priors = priors[:m]
                    priors = priors.masked_fill(~valid_mask, float("-inf"))

                    topk = min(int(self.config.expansion_topk), int(valid_n))
                    if topk <= 0:
                        continue
                    top_actions = torch.topk(priors, k=topk).indices

                    for a in top_actions:
                        a = int(a)

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
                            topk_ctr[0] = track_terminal(topk_heap, topk_ctr[0], engine, new_cum, max_k)

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item.cum_reward, reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

            if has_callback:
                for beam in beams:
                    if beam.first_action >= 0:
                        if beam.first_action not in root_action_scores or beam.cum_reward > root_action_scores[beam.first_action]:
                            root_action_scores[beam.first_action] = beam.cum_reward

                visits = np.zeros(n_actions, dtype=np.int32)
                values = np.zeros(n_actions, dtype=np.float32)
                for beam in beams:
                    if 0 <= beam.first_action < n_actions:
                        visits[beam.first_action] += 1
                        values[beam.first_action] = max(values[beam.first_action], beam.cum_reward)
                for root_a, score in root_action_scores.items():
                    if 0 <= root_a < n_actions:
                        values[root_a] = max(values[root_a], score)
                valid_values = np.where(mask_np, values, -np.inf)
                best_a = int(np.argmax(valid_values))
                best_v = float(values[best_a]) if best_a < len(values) else 0.0
                progress_fn(depth + 1, total_depth, visits, values, best_a, best_v)

        best = beams[0].first_action if beams else -1
        if best < 0:
            best = self._fallback_action(
                agent=agent,
                obs=obs,
                root_action_space=root_action_space,
                device=adapter.device,
            )

        # Build output arrays
        n_out = int(root_action_space.valid_mask.shape[0])
        visits_out = np.zeros(n_out, dtype=np.int32)
        values_out = np.zeros(n_out, dtype=np.float32)
        for beam in beams:
            if 0 <= beam.first_action < n_out:
                visits_out[beam.first_action] += 1
                values_out[beam.first_action] = max(values_out[beam.first_action], beam.cum_reward)

        self._restore_snapshot(engine=engine, adapter=adapter, snapshot=root_snapshot)
        return SearchOutput(
            action=int(best),
            visits=visits_out,
            values=values_out,
            iterations=total_depth,
            top_k=collect_top_k(topk_heap),
        )

    def _fallback_action(
        self,
        *,
        agent: Agent,
        obs: dict,
        root_action_space: ActionSpace,
        device: torch.device,
    ) -> int:
        valid = root_action_space.valid_mask.to(dtype=torch.bool, device=device).view(-1)
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) <= 0:
            return -1
        try:
            scores = agent.policy(obs=obs, action_space=root_action_space).to(dtype=torch.float32, device=device).view(-1)
            scores = scores.masked_fill(~valid, float("-inf"))
            if bool(torch.isfinite(scores).any().item()):
                return int(torch.argmax(scores).item())
        except Exception:
            pass
        return int(valid_idx[0].item())



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
    search = BeamSearch(config=BeamConfig(beam_width=8, depth=3, expansion_topk=16))
    search.set_adapter(adapter)

    t0 = time.perf_counter()
    next_gid = root_action_space.group_id
    output = search.select(obs=obs, agent=agent, root_action_space=root_action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_action_space.valid_mask.sum().item())
    a = output.action
    pose = root_action_space.centers[a].tolist() if int(root_action_space.centers.shape[0]) > 0 else [0, 0]

    print("search.beam demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", next_gid)
    print(" action=", a, "valid_actions=", valid_n, "pose=", pose)
    print(f" elapsed_ms={dt_ms:.2f}")
