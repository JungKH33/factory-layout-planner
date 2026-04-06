from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from group_placement.envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyV3Adapter(BaseAdapter):
    """Top-K candidate wrapper: Discrete(K) actions over center-based sampling.

    Samples candidate **center** positions from the unified placeable map
    (all variants OR'd).  Variant is resolved by the engine at step
    time — the adapter never accesses rotation directly.

    Notes:
    - ``action_mask`` is ``torch.BoolTensor[K]`` (True means valid).
    - ``action_poses`` is ``torch.FloatTensor[K, 2]`` of ``(x_center, y_center)``
      center coordinates.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int = 50,
        quant_step: Optional[float] = 10.0,
        oversample_factor: int = 2,
        edge_ratio: float = 0.8,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.k = int(k)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.oversample_factor = int(oversample_factor)
        self.edge_ratio = float(edge_ratio)
        self._rng = random.Random(random_seed)

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        self.action_poses: Optional[torch.Tensor] = None  # float [K,2]
        self.action_costs: Optional[torch.Tensor] = None  # float [K]

    def build_observation(self) -> Dict[str, Any]:
        self.mask = self.create_mask()
        obs: Dict[str, Any] = {}
        if isinstance(self.action_costs, torch.Tensor):
            obs["action_costs"] = self.action_costs
        obs["reward_scale"] = float(self.engine.reward_scale)
        obs["failure_penalty"] = float(self.engine.failure_penalty())
        return obs

    def create_mask(self) -> torch.Tensor:
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.action_poses = torch.zeros((self.k, 2), dtype=torch.float32, device=self.device)
            self.action_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
            self.action_variant_indices = None
            return torch.zeros((self.k,), dtype=torch.bool, device=self.device)

        poses, mask = self._generate(self.engine, gid)

        self.action_poses = poses
        self.action_variant_indices = None

        delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
        vmask = mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]
        if int(vidx.numel()) > 0:
            d = self._score_poses(gid, poses[vidx]).to(dtype=torch.float32, device=self.device)
            delta[vidx] = d.view(-1)
        self.action_costs = delta
        return mask

    # ---- state api (for wrapped search/MCTS) ----
    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        snap["rng_state"] = self._rng.getstate()
        if isinstance(self.action_poses, torch.Tensor):
            snap["action_poses"] = self.action_poses.clone()
        else:
            snap["action_poses"] = None
        if isinstance(self.action_costs, torch.Tensor):
            snap["action_costs"] = self.action_costs.clone()
        else:
            snap["action_costs"] = None
        return snap

    def set_state(self, state: Dict[str, object]) -> None:
        super().set_state(state)
        rs = state.get("rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        ax = state.get("action_poses", None)
        if isinstance(ax, torch.Tensor):
            self.action_poses = ax.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_poses = None
        ad = state.get("action_costs", None)
        if isinstance(ad, torch.Tensor):
            self.action_costs = ad.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_costs = None

    # ---- candidate generation (center-based, variant-free) ----

    def _torch_gen(self, *, env: FactoryLayoutEnv) -> torch.Generator:
        seed = int(self._rng.randrange(0, 2**31 - 1))
        g = torch.Generator(device=env.device)
        g.manual_seed(seed)
        return g

    def _build_edge_map(self, valid_map: torch.Tensor) -> torch.Tensor:
        """Edge sampling mask: edge = valid & dilate(~valid)."""
        if (not isinstance(valid_map, torch.Tensor)) or valid_map.numel() == 0:
            return valid_map
        v = valid_map.to(dtype=torch.bool)
        inv = (~v).to(dtype=torch.float32).view(1, 1, int(v.shape[0]), int(v.shape[1]))
        kernel = torch.ones((1, 1, 3, 3), device=valid_map.device, dtype=inv.dtype)
        nbr = (F.conv2d(inv, kernel, padding=1) > 0).squeeze(0).squeeze(0)
        return v & nbr

    def _sample_centers(
        self,
        *,
        valid_map: torch.Tensor,
        count: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Sample up to *count* center positions from a center-based validity map.

        Returns ``[M, 2]`` float tensor of ``(x_center, y_center)`` positions.
        """
        idx = torch.nonzero(valid_map, as_tuple=False)  # [M, 2] of (y, x)
        if idx.numel() == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=valid_map.device)
        M = int(idx.shape[0])
        if M > count:
            perm = torch.randperm(M, generator=gen, device=valid_map.device)[:count]
            idx = idx[perm]
        # Convert (y, x) -> (x_center, y_center) with +0.5 offset for center of grid cell
        centers = torch.stack([idx[:, 1].float(), idx[:, 0].float()], dim=-1)
        return centers

    def _dedup_centers(
        self,
        tagged: List[Tuple[int, torch.Tensor]],
        q: float,
    ) -> List[Tuple[int, torch.Tensor]]:
        """Deduplicate (source_tag, center_tensor) pairs by quantised center."""
        if q <= 0:
            return tagged
        seen: set = set()
        unique: List[Tuple[int, torch.Tensor]] = []
        for src, center in tagged:
            qx = int(round(float(center[0].item()) / q))
            qy = int(round(float(center[1].item()) / q))
            key = (qx, qy)
            if key not in seen:
                seen.add(key)
                unique.append((src, center))
        return unique

    def _generate(
        self, env: FactoryLayoutEnv, gid: GroupId
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate candidate center poses and validity mask.

        Returns:
            poses: ``[K, 2]`` float tensor of center coordinates.
            mask:  ``[K]`` bool tensor.
        """
        device = env.device
        spec = env.group_specs[gid]
        state = env.get_state()

        # Unified center-based validity map (all variants OR'd).
        center_map = self._build_center_map(gid)
        edge_map = self._build_edge_map(center_map)
        gen = self._torch_gen(env=env)
        q = float(self.quant_step) if self.quant_step is not None else 1.0

        total_k = max(1, int(self.k * self.oversample_factor))
        edge_ratio = max(0.0, min(1.0, float(self.edge_ratio)))
        n_edge = int(round(float(total_k) * edge_ratio))
        n_fill = total_k - n_edge

        edge_centers = self._sample_centers(valid_map=edge_map, count=n_edge, gen=gen)
        fill_centers = self._sample_centers(valid_map=center_map, count=n_fill, gen=gen)

        # Tag: 0 = edge, 1 = fill.  Build combined list for dedup.
        raw_tagged: List[Tuple[int, torch.Tensor]] = []
        for i in range(int(edge_centers.shape[0])):
            raw_tagged.append((0, edge_centers[i]))
        for i in range(int(fill_centers.shape[0])):
            raw_tagged.append((1, fill_centers[i]))

        unique_tagged = self._dedup_centers(raw_tagged, q)

        # No explicit placeable_batch check needed: candidates come from
        # the center_map which guarantees at least one variant is valid.
        # Any false-positives from the integer center shift are handled by
        # score_batch (returns inf for non-placeable) and resolve_action.

        # Keep order: edge first, then fill.  Trim to K.
        final = [c for _src, c in unique_tagged][:self.k]

        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        for i, c in enumerate(final):
            poses[i] = c
            mask[i] = True
        return poses, mask


if __name__ == "__main__":
    import torch

    from group_placement.envs.action_space import ActionSpace
    from group_placement.envs.action import EnvAction
    from group_placement.envs.env_loader import load_env
    from group_placement.envs.visualizer import plot_layout

    ENV_JSON = "group_placement/envs/env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV3Adapter(k=50, quant_step=10.0, oversample_factor=2, edge_ratio=0.8, random_seed=0)

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.valid_mask.sum().item())
    a = int(torch.where(candidates.valid_mask)[0][0].item()) if valid > 0 else 0

    # Plot: initial candidates (interactive; close to continue)
    plot_layout(engine, action_space=candidates)

    t1 = time.perf_counter()
    placement = adapter.resolve_action(a, candidates)
    _obs_env2, _r, _term, _trunc, _info2 = engine.step_placement(placement)
    obs2 = adapter.build_observation()
    candidates2 = adapter.build_action_space()
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot: after 1 placement + new candidates (if any)
    if int(candidates2.valid_mask.shape[0]) > 0:
        plot_layout(engine, action_space=candidates2)
    else:
        plot_layout(engine, action_space=None)

    print("GreedyV3Adapter demo")
    print(" env=", ENV_JSON, "device=", device, "k=", 50)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")
