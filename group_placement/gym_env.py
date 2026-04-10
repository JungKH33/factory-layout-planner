"""Gym-compatible wrapper that composes engine + adapter for RL training.

Usage::

    from group_placement.envs.env_loader import load_env
    from group_placement.agents.placement.maskplace import MaskPlaceAdapter
    from gym_env import AdapterGymEnv

    loaded = load_env("group_placement/envs/env_configs/basic_01.json", device=device)
    adapter = MaskPlaceAdapter(grid=224)
    env = AdapterGymEnv(engine=loaded.env, adapter=adapter,
                        reset_kwargs=loaded.reset_kwargs)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(action_index)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch

from group_placement.envs.env import FactoryLayoutEnv
from group_placement.agents.base import BaseAdapter


class AdapterGymEnv(gym.Env):
    """Gym-compatible env that delegates observation to the adapter.

    The engine handles state/physics/cost; the adapter builds observations
    and action spaces tailored to the model/agent it serves.

    ``step()`` returns observations from ``adapter.build_observation()``
    merged with the current action mask.  The action mask is always
    included under the ``"action_mask"`` key so that training frameworks
    (Tianshou, TorchRL, SB3) can apply invalid-action masking.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.adapter = adapter
        self._reset_kwargs = dict(reset_kwargs or {})
        adapter.bind(engine)

        # Mirror adapter spaces for Gym compatibility.
        self.observation_space = adapter.observation_space
        self.action_space = adapter.action_space

    # ---- Gym API ----

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        opts = dict(options) if options is not None else dict(self._reset_kwargs)
        self.engine.reset(options=opts)
        self.adapter.bind(self.engine)

        obs = self.adapter.build_observation()
        action_space = self.adapter.build_action_space()
        obs["action_mask"] = action_space.valid_mask
        return obs, {}

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action_space = self.adapter.build_action_space()
        try:
            placement = self.adapter.resolve_action(int(action), action_space)
        except (IndexError, ValueError):
            reward = float(self.engine.failure_penalty())
            obs = self.adapter.build_observation()
            obs["action_mask"] = torch.zeros_like(action_space.valid_mask)
            return obs, reward, False, True, {"reason": "invalid_action"}

        _, reward, terminated, truncated, info = self.engine.step_placement(placement)
        obs = self.adapter.build_observation()

        if not (terminated or truncated):
            next_as = self.adapter.build_action_space()
            obs["action_mask"] = next_as.valid_mask
        else:
            obs["action_mask"] = torch.zeros_like(action_space.valid_mask)

        return obs, float(reward), bool(terminated), bool(truncated), info
