from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyV4Adapter(BaseAdapter):
    """Cell-capped top-K adapter with coarse-cell annotations.

    Scores every valid center from ``placeable_center_map`` in one
    ``cost_batch`` call, keeps the best ``top_per_cell`` centers per coarse
    cell, then interleaves those per-cell rankings until K actions are filled.

    Observation includes:
    - ``action_costs``: ``[K]`` float — incremental cost per candidate
    - ``cell_indices``: ``[K]`` int — which coarse cell each action belongs to
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int = 50,
        cell_size: int = 10,
        top_per_cell: int = 3,
        quant_step: Optional[float] = None,
        random_seed: Optional[int] = None,
        expand_orientations: bool = False,
        max_orientations: int = 4,
        **kwargs: Any,
    ):
        super().__init__(expand_orientations=expand_orientations, max_orientations=max_orientations)
        self.k = int(k)
        self.cell_size = int(cell_size)
        self.top_per_cell = int(top_per_cell)
        self.quant_step = float(quant_step) if quant_step is not None else None
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        if self.top_per_cell <= 0:
            raise ValueError("top_per_cell must be > 0")
        self._rng = random.Random(random_seed)

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        self.action_poses: Optional[torch.Tensor] = None   # float [K, 2]
        self.action_costs: Optional[torch.Tensor] = None    # float [K]
        self.cell_indices: Optional[torch.Tensor] = None    # int64 [K]

    def build_observation(self) -> Dict[str, Any]:
        self.mask = self.create_mask()
        obs: Dict[str, Any] = {}
        if isinstance(self.action_costs, torch.Tensor):
            obs["action_costs"] = self.action_costs
        if isinstance(self.cell_indices, torch.Tensor):
            obs["cell_indices"] = self.cell_indices
        return obs

    def create_mask(self) -> torch.Tensor:
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.cell_indices = torch.zeros((self.k,), dtype=torch.int64, device=self.device)
            return self._empty_orientation_output(self.k)

        poses, mask, cell_ids, costs = self._generate(self.engine, gid)

        if self.expand_orientations:
            self.cell_indices = cell_ids
            out_mask = self._apply_orientation_expansion(gid, poses, mask, self.k)
            if isinstance(self.action_poses, torch.Tensor):
                self._remap_cell_indices_after_expansion(poses, cell_ids)
            return out_mask

        self.action_poses = poses
        self.action_orientation_indices = None
        self.cell_indices = cell_ids
        self.action_costs = costs
        return mask

    # ---- state api (for MCTS) ----

    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        snap["rng_state"] = self._rng.getstate()
        snap["action_poses"] = self.action_poses.clone() if isinstance(self.action_poses, torch.Tensor) else None
        snap["action_costs"] = self.action_costs.clone() if isinstance(self.action_costs, torch.Tensor) else None
        snap["cell_indices"] = self.cell_indices.clone() if isinstance(self.cell_indices, torch.Tensor) else None
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
        ad = state.get("action_costs", None)
        self.action_costs = ad.to(device=self.device, dtype=torch.float32).clone() if isinstance(ad, torch.Tensor) else None
        ci = state.get("cell_indices", None)
        self.cell_indices = ci.to(device=self.device, dtype=torch.int64).clone() if isinstance(ci, torch.Tensor) else None

    # ---- candidate generation ----

    def _generate(
        self, env: FactoryLayoutEnv, gid: GroupId
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate per-cell top-N candidates with coarse-cell annotations.

        Steps:
        1. ``placeable_center_map`` → all valid center positions
        2. ``cost_batch`` → score all valid centers in one call
        3. (optional) quantize-dedup by ``quant_step``
        4. Keep the best ``top_per_cell`` centers in each coarse cell
        5. Round-robin interleave per-cell rankings until K candidates are filled
        6. Compute coarse-cell metadata for observations

        Returns:
            poses:         [K, 2] center coordinates
            mask:          [K] bool
            cell_indices:  [K] int64 — flat cell index per candidate
            costs:         [K] float — delta cost per candidate (inf for padding)
        """
        device = env.device
        state = env.get_state()
        spec = env.group_specs[gid]

        center_map = spec.placeable_center_map(state, gid)
        W = int(center_map.shape[1])
        cs = self.cell_size
        W_c = (W + cs - 1) // cs

        all_yx = torch.nonzero(center_map, as_tuple=False)  # [M, 2] (y, x)
        if all_yx.numel() == 0:
            return self._empty_generate(device)

        all_centers = torch.stack([all_yx[:, 1].float(), all_yx[:, 0].float()], dim=-1)  # [M, 2]
        flat_cell = (all_yx[:, 0] // cs) * W_c + (all_yx[:, 1] // cs)  # [M]

        # Score all valid centers in one batch call
        all_costs = self._score_poses(gid, all_centers)  # [M]

        # Filter to finite costs
        finite_mask = torch.isfinite(all_costs)
        if not finite_mask.any():
            return self._empty_generate(device)

        if bool(finite_mask.all()):
            f_centers = all_centers
            f_costs = all_costs
            f_cells = flat_cell
        else:
            finite_idx = torch.where(finite_mask)[0]
            f_centers = all_centers[finite_idx]
            f_costs = all_costs[finite_idx]
            f_cells = flat_cell[finite_idx]

        # Quantize-dedup: keep lowest cost per quantized bin
        if self.quant_step is not None and self.quant_step > 0:
            f_centers, f_costs, f_cells = self._quant_dedup(f_centers, f_costs, f_cells)

        selected_idx = self._select_top_per_cell(f_costs, f_cells)
        pick = min(self.k, int(selected_idx.shape[0]))

        # Build output
        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        cell_ids = torch.zeros((self.k,), dtype=torch.int64, device=device)
        out_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=device)

        if pick > 0:
            chosen = selected_idx[:pick]
            poses[:pick] = f_centers[chosen]
            mask[:pick] = True
            cell_ids[:pick] = f_cells[chosen]
            out_costs[:pick] = f_costs[chosen]

        return poses, mask, cell_ids, out_costs

    def _select_top_per_cell(
        self,
        costs: torch.Tensor,
        cells: torch.Tensor,
    ) -> torch.Tensor:
        """Select up to ``top_per_cell`` entries per cell, then round-robin trim.

        Cells are ordered by their best candidate cost so higher-quality cells
        contribute earlier when the union exceeds ``k``.

        Uses ``argsort`` to group by cell in O(M log M) instead of per-cell
        ``torch.where`` loops which are O(cells * M).
        """
        if costs.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=cells.device)

        # Sort by (cell, cost) in one pass via composite key.
        # Scale cell_id above cost range so primary sort is by cell.
        cost_range = costs.max() - costs.min() + 1.0
        sort_key = cells.float() * cost_range + costs
        order = torch.argsort(sort_key)

        sorted_cells = cells[order]
        unique_cells, counts = torch.unique_consecutive(sorted_cells, return_counts=True)

        tpc = self.top_per_cell
        n_cells = int(unique_cells.shape[0])

        # Per-cell top-N: slice each contiguous group (already cost-sorted)
        # and collect into a [n_cells, tpc] padded matrix for vectorized round-robin.
        per_cell_idx = torch.full(
            (n_cells, tpc), -1, dtype=torch.long, device=cells.device
        )
        per_cell_len = torch.zeros(n_cells, dtype=torch.long, device=cells.device)
        best_cost = torch.full((n_cells,), float("inf"), dtype=costs.dtype, device=cells.device)

        offset = 0
        for i, cnt in enumerate(counts.tolist()):
            take = min(tpc, int(cnt))
            per_cell_idx[i, :take] = order[offset : offset + take]
            per_cell_len[i] = take
            best_cost[i] = costs[order[offset]]
            offset += cnt

        # Sort cells by best candidate cost (ascending) for round-robin priority
        cell_order = torch.argsort(best_cost)
        per_cell_idx = per_cell_idx[cell_order]
        per_cell_len = per_cell_len[cell_order]

        # Round-robin interleave: rank 0 from all cells, then rank 1, etc.
        selected = []
        for rank in range(tpc):
            has_rank = per_cell_len > rank
            if not has_rank.any():
                break
            batch = per_cell_idx[has_rank, rank]
            for idx_val in batch.tolist():
                selected.append(idx_val)
                if len(selected) >= self.k:
                    break
            if len(selected) >= self.k:
                break

        if not selected:
            return torch.zeros((0,), dtype=torch.long, device=cells.device)
        return torch.tensor(selected, dtype=torch.long, device=cells.device)

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

    def _empty_generate(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        cell_ids = torch.zeros((self.k,), dtype=torch.int64, device=device)
        out_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=device)
        return poses, mask, cell_ids, out_costs

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

    from envs.action_space import ActionSpace
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV4Adapter(
        k=50, cell_size=10, top_per_cell=3, quant_step=10.0, random_seed=0,
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
    print(f"  env={ENV_JSON}  device={device}  k=50  cell_size=10  top_per_cell=3  quant_step=10.0")
    print(f"  valid_actions={valid}  first_valid_action={a}")
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
