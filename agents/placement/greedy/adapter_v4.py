from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyV4Adapter(BaseAdapter):
    """Coarse-grid adapter: Discrete(K) actions over grid-cell-partitioned candidates.

    Divides the placeable center map into coarse cells of ``cell_size`` grid
    units.  Within each cell, candidates are scored and the top
    ``top_per_cell`` are retained.  The final action space is the union of
    per-cell top candidates, trimmed to K.

    Designed as a stepping-stone for RL: an RL policy selects a coarse cell,
    then the adapter supplies the precise position within that cell.

    Observation includes:
    - ``action_delta``: ``[K]`` float — incremental cost per candidate
    - ``cell_features``: ``[H_c, W_c, C]`` float — per-cell feature map
      (C channels: has_candidate, best_cost, candidate_count, mean_cost)
    - ``cell_indices``: ``[K]`` int — which coarse cell each action belongs to
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int = 50,
        cell_size: int = 10,
        quant_step: Optional[float] = None,
        random_seed: Optional[int] = None,
        expand_orientations: bool = False,
        max_orientations: int = 4,
        **kwargs: Any,
    ):
        super().__init__(expand_orientations=expand_orientations, max_orientations=max_orientations)
        self.k = int(k)
        self.cell_size = int(cell_size)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self._rng = random.Random(random_seed)

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        self.action_poses: Optional[torch.Tensor] = None   # float [K, 2]
        self.action_delta: Optional[torch.Tensor] = None    # float [K]
        self.cell_indices: Optional[torch.Tensor] = None    # int64 [K]
        self.cell_features: Optional[torch.Tensor] = None   # float [H_c, W_c, C]

    def build_observation(self) -> Dict[str, Any]:
        self.mask = self.create_mask()
        obs: Dict[str, Any] = {}
        if isinstance(self.action_delta, torch.Tensor):
            obs["action_delta"] = self.action_delta
        if isinstance(self.cell_features, torch.Tensor):
            obs["cell_features"] = self.cell_features
        if isinstance(self.cell_indices, torch.Tensor):
            obs["cell_indices"] = self.cell_indices
        return obs

    def create_mask(self) -> torch.Tensor:
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.cell_indices = torch.zeros((self.k,), dtype=torch.int64, device=self.device)
            self.cell_features = None
            return self._empty_orientation_output(self.k)

        poses, mask, cell_ids, cell_feat, costs = self._generate(self.engine, gid)

        self.cell_features = cell_feat

        if self.expand_orientations:
            self.cell_indices = cell_ids
            out_mask = self._apply_orientation_expansion(gid, poses, mask, self.k)
            if isinstance(self.action_poses, torch.Tensor):
                self._remap_cell_indices_after_expansion(poses, cell_ids)
            return out_mask

        self.action_poses = poses
        self.action_orientation_indices = None
        self.cell_indices = cell_ids
        self.action_delta = costs
        return mask

    # ---- state api (for MCTS) ----

    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        snap["rng_state"] = self._rng.getstate()
        snap["action_poses"] = self.action_poses.clone() if isinstance(self.action_poses, torch.Tensor) else None
        snap["action_delta"] = self.action_delta.clone() if isinstance(self.action_delta, torch.Tensor) else None
        snap["cell_indices"] = self.cell_indices.clone() if isinstance(self.cell_indices, torch.Tensor) else None
        snap["cell_features"] = self.cell_features.clone() if isinstance(self.cell_features, torch.Tensor) else None
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
        self.action_poses = ax.to(device=self.device, dtype=torch.float32).clone() if isinstance(ax, torch.Tensor) else None
        ad = state.get("action_delta", None)
        self.action_delta = ad.to(device=self.device, dtype=torch.float32).clone() if isinstance(ad, torch.Tensor) else None
        ci = state.get("cell_indices", None)
        self.cell_indices = ci.to(device=self.device, dtype=torch.int64).clone() if isinstance(ci, torch.Tensor) else None
        cf = state.get("cell_features", None)
        self.cell_features = cf.to(device=self.device, dtype=torch.float32).clone() if isinstance(cf, torch.Tensor) else None

    # ---- candidate generation ----

    def _generate(
        self, env: FactoryLayoutEnv, gid: GroupId
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate candidates partitioned by coarse grid cells.

        Steps:
        1. ``placeable_center_map`` → all valid center positions
        2. ``cost_batch`` → score all valid centers in one call
        3. (optional) quantize-dedup by ``quant_step``
        4. ``topk(K)`` → pick K lowest-cost centers globally
        5. Compute cell membership for each selected candidate

        Returns:
            poses:         [K, 2] center coordinates
            mask:          [K] bool
            cell_indices:  [K] int64 — flat cell index per candidate
            cell_features: [H_c, W_c, C] float — per-cell feature map
            costs:         [K] float — delta cost per candidate (inf for padding)
        """
        device = env.device
        state = env.get_state()
        spec = env.group_specs[gid]

        center_map = spec.placeable_center_map(state, gid)
        H, W = int(center_map.shape[0]), int(center_map.shape[1])
        cs = self.cell_size
        H_c = (H + cs - 1) // cs
        W_c = (W + cs - 1) // cs

        all_yx = torch.nonzero(center_map, as_tuple=False)  # [M, 2] (y, x)
        if all_yx.numel() == 0:
            return self._empty_generate(device, H_c, W_c)

        all_centers = torch.stack([all_yx[:, 1].float(), all_yx[:, 0].float()], dim=-1)  # [M, 2]
        flat_cell = (all_yx[:, 0] // cs) * W_c + (all_yx[:, 1] // cs)  # [M]

        # Score all valid centers in one batch call
        all_costs = self._score_poses(gid, all_centers)  # [M]

        # Filter to finite costs
        finite_mask = torch.isfinite(all_costs)
        if not finite_mask.any():
            cell_features = self._build_cell_features(all_costs, flat_cell, finite_mask, H_c, W_c, device)
            return self._empty_generate(device, H_c, W_c)

        finite_idx = torch.where(finite_mask)[0]
        f_centers = all_centers[finite_idx]
        f_costs = all_costs[finite_idx]
        f_cells = flat_cell[finite_idx]

        # Quantize-dedup: keep lowest cost per quantized bin
        if self.quant_step is not None and self.quant_step > 0:
            f_centers, f_costs, f_cells = self._quant_dedup(f_centers, f_costs, f_cells)

        # Global top-K
        pick = min(self.k, int(f_costs.shape[0]))
        _, topk_idx = torch.topk(f_costs, k=pick, largest=False)

        # Build output
        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        cell_ids = torch.zeros((self.k,), dtype=torch.int64, device=device)
        out_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=device)

        poses[:pick] = f_centers[topk_idx]
        mask[:pick] = True
        cell_ids[:pick] = f_cells[topk_idx]
        out_costs[:pick] = f_costs[topk_idx]

        cell_features = self._build_cell_features(all_costs, flat_cell, finite_mask, H_c, W_c, device)
        return poses, mask, cell_ids, cell_features, out_costs

    def _quant_dedup(
        self,
        centers: torch.Tensor,
        costs: torch.Tensor,
        cells: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep lowest-cost entry per quantized (x, y) bin. Pure torch."""
        q = self.quant_step
        qx = torch.div(centers[:, 0], q, rounding_mode="trunc").long()
        qy = torch.div(centers[:, 1], q, rounding_mode="trunc").long()
        # Pack (qx, qy) into a single key
        qy_range = qy.max() - qy.min() + 1 if qy.numel() > 0 else 1
        bin_key = qx * qy_range + (qy - qy.min())

        # Sort by (bin_key, cost) so first occurrence per bin is cheapest
        sort_key = bin_key.float() * (costs.max() - costs.min() + 1.0) + costs
        order = torch.argsort(sort_key)
        sorted_keys = bin_key[order]

        # First occurrence per bin: where key differs from previous
        first_mask = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=centers.device)
        if sorted_keys.shape[0] > 1:
            first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]

        keep = order[first_mask]
        return centers[keep], costs[keep], cells[keep]

    def _build_cell_features(
        self,
        costs: torch.Tensor,
        flat_cell: torch.Tensor,
        finite_mask: torch.Tensor,
        H_c: int,
        W_c: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [H_c, W_c, 4] cell feature map.

        Channels: has_candidate, best_cost, candidate_count, mean_cost.
        Costs are normalized to [0, 1] range.
        """
        n_cells = H_c * W_c

        if not finite_mask.any():
            return torch.zeros((H_c, W_c, 4), dtype=torch.float32, device=device)

        f_costs = costs[finite_mask]
        f_cells = flat_cell[finite_mask].to(dtype=torch.long)

        cand_count = torch.zeros(n_cells, dtype=torch.float32, device=device)
        sum_cost = torch.zeros(n_cells, dtype=torch.float32, device=device)
        best_cost = torch.full((n_cells,), float("inf"), dtype=torch.float32, device=device)

        ones = torch.ones_like(f_costs)
        cand_count.scatter_add_(0, f_cells, ones)
        sum_cost.scatter_add_(0, f_cells, f_costs)
        best_cost.scatter_reduce_(0, f_cells, f_costs, reduce="amin", include_self=False)

        has_cand = (cand_count > 0).float()
        best_cost = torch.where(torch.isfinite(best_cost), best_cost, torch.zeros_like(best_cost))
        mean_cost = torch.where(cand_count > 0, sum_cost / cand_count, torch.zeros_like(sum_cost))

        # Normalize
        cost_min = f_costs.min()
        cost_range = f_costs.max() - cost_min
        if float(cost_range.item()) > 1e-8:
            best_cost = (best_cost - cost_min) / cost_range
            mean_cost = torch.where(has_cand.bool(), (mean_cost - cost_min) / cost_range, mean_cost)
        else:
            best_cost.zero_()
            mean_cost.zero_()

        max_count = cand_count.max().clamp(min=1.0)
        cand_count_norm = cand_count / max_count

        return torch.stack([has_cand, best_cost, cand_count_norm, mean_cost], dim=-1).view(H_c, W_c, 4)

    def _empty_generate(
        self,
        device: torch.device,
        H_c: int,
        W_c: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        cell_ids = torch.zeros((self.k,), dtype=torch.int64, device=device)
        cell_feat = torch.zeros((H_c, W_c, 4), dtype=torch.float32, device=device)
        out_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=device)
        return poses, mask, cell_ids, cell_feat, out_costs

    def _remap_cell_indices_after_expansion(
        self,
        original_poses: torch.Tensor,
        original_cell_ids: torch.Tensor,
    ) -> None:
        """Remap cell_indices to match expanded action_poses from orientation expansion."""
        expanded_poses = self.action_poses  # [K, 2] — set by _apply_orientation_expansion
        if expanded_poses is None:
            return
        # Vectorized nearest-neighbor lookup: [K, 1, 2] - [1, N, 2] → [K, N] → argmin
        diffs = (expanded_poses[:, None, :] - original_poses[None, :, :]).abs().sum(dim=-1)  # [K, N]
        best = torch.argmin(diffs, dim=1)  # [K]
        self.cell_indices = original_cell_ids[best]


if __name__ == "__main__":
    import torch

    from envs.action_space import ActionSpace as CandidateSet
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV4Adapter(
        k=50, cell_size=10, top_per_cell=3,
        oversample_factor=2, random_seed=0,
    )

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.mask.sum().item())
    a = int(torch.where(candidates.mask)[0][0].item()) if valid > 0 else 0

    print("GreedyV4Adapter demo")
    print(f"  env={ENV_JSON}  device={device}  k=50  cell_size=10  top_per_cell=3")
    print(f"  valid_actions={valid}  first_valid_action={a}")
    if obs.get("cell_features") is not None:
        cf = obs["cell_features"]
        print(f"  cell_features shape={tuple(cf.shape)}")
    if obs.get("cell_indices") is not None:
        ci = obs["cell_indices"]
        unique_cells = int(ci[:valid].unique().shape[0]) if valid > 0 else 0
        print(f"  unique_cells_in_actions={unique_cells}")
    print(f"  reset_ms={dt_reset_ms:.3f}")

    # Plot candidates
    plot_layout(engine, action_space=candidates)

    # Step and show next
    t1 = time.perf_counter()
    placement = adapter.decode_action(a, candidates)
    _obs_env2, _r, _term, _trunc, _info2 = engine.step_action(placement)
    obs2 = adapter.build_observation()
    candidates2 = adapter.build_action_space()
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    valid2 = int(candidates2.mask.sum().item())
    print(f"  step_ms={dt_step_ms:.3f}  valid_after_step={valid2}")

    if int(candidates2.mask.shape[0]) > 0:
        plot_layout(engine, action_space=candidates2)
    else:
        plot_layout(engine, action_space=None)
