from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import torch

from envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyV4Adapter(BaseAdapter):
    """Cell-capped top-K adapter with coarse-cell annotations.

    Scores every valid center from ``placeable_center_map`` in one
    ``score_batch`` call, keeps the best ``top_per_cell`` centers per coarse
    cell, then interleaves those per-cell rankings (round-robin).  Optionally
    re-ranks a round-robin prefix by score.

    Observation includes:
    - ``action_costs``: ``[K]`` float — incremental cost per candidate
    - ``cell_indices``: ``[K]`` int — which coarse cell each action belongs to
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int = 50,
        cell_size: int = 50,
        top_per_cell: int = 3,
        expand_factor: int = 2,
        quant_step: Optional[float] = None,
        random_seed: Optional[int] = None,
        expand_variants: bool = False,
        max_variants: int = 4,
        **kwargs: Any,
    ):
        super().__init__()
        self.k = int(k)
        self.cell_size = int(cell_size)
        self.top_per_cell = int(top_per_cell)
        self.expand_factor = int(expand_factor)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.expand_variants = bool(expand_variants)
        self.max_variants = int(max_variants)
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        if self.top_per_cell <= 0:
            raise ValueError("top_per_cell must be > 0")
        if self.expand_factor <= 0:
            raise ValueError("expand_factor must be > 0")
        if self.max_variants <= 0:
            raise ValueError("max_variants must be > 0")
        self._rng = random.Random(random_seed)

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
        obs["reward_scale"] = float(self.engine.reward_scale)
        obs["failure_penalty"] = float(self.engine.failure_penalty())
        return obs

    def create_mask(self) -> torch.Tensor:
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.cell_indices = torch.zeros((self.k,), dtype=torch.int64, device=self.device)
            return self._empty_variant_output(self.k)

        poses, mask, cell_ids, costs = self._generate(self.engine, gid)

        if self.expand_variants:
            self.cell_indices = cell_ids
            out_mask = self._apply_variant_expansion(gid, poses, mask, self.k)
            if isinstance(self.action_poses, torch.Tensor):
                self._remap_cell_indices_after_expansion(poses, cell_ids)
            return out_mask

        self.action_poses = poses
        self.action_variant_indices = None
        self.cell_indices = cell_ids
        self.action_costs = costs
        return mask

    # ---- state api (for search) ----

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
        """Generate per-cell top-N candidates with coarse-cell annotations."""
        device = env.device

        center_map = self._build_center_map(gid)
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
        """Select up to ``top_per_cell`` entries per cell via RR, then optional rerank."""
        if costs.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=cells.device)

        # Sort by (cell, cost) in one pass via composite key.
        cost_range = costs.max() - costs.min() + 1.0
        sort_key = cells.float() * cost_range + costs
        order = torch.argsort(sort_key)

        sorted_cells = cells[order]
        unique_cells, counts = torch.unique_consecutive(sorted_cells, return_counts=True)

        tpc = self.top_per_cell
        n_cells = int(unique_cells.shape[0])

        # Per-cell top-N and round-robin interleave.
        start = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(tpc, device=cells.device, dtype=torch.long)
        take = torch.clamp(counts, max=tpc)
        valid_rank = rank.unsqueeze(0) < take.unsqueeze(1)
        gather_pos = start.unsqueeze(1) + rank.unsqueeze(0)

        per_cell_idx = torch.full((n_cells, tpc), -1, dtype=torch.long, device=cells.device)
        if bool(valid_rank.any()):
            per_cell_idx[valid_rank] = order[gather_pos[valid_rank]]
        per_cell_len = take
        best_cost = costs[order[start]]

        cell_order = torch.argsort(best_cost)
        per_cell_idx = per_cell_idx[cell_order]
        per_cell_len = per_cell_len[cell_order]

        valid_rr = (
            torch.arange(tpc, device=cells.device, dtype=torch.long).unsqueeze(1)
            < per_cell_len.unsqueeze(0)
        )
        selected = per_cell_idx.transpose(0, 1)[valid_rr]

        if int(selected.numel()) == 0:
            return torch.zeros((0,), dtype=torch.long, device=cells.device)

        # Fast path: preserve legacy behavior exactly (no additional score sorting).
        if self.expand_factor == 1:
            if int(selected.shape[0]) > self.k:
                selected = selected[:self.k]
            return selected

        # Build a larger RR prefix, then sort by score within that pool.
        pool_k = min(int(selected.shape[0]), int(self.k * self.expand_factor))
        pool = selected[:pool_k]
        if int(pool.shape[0]) > 1:
            pool_costs = costs[pool]
            pool = pool[torch.argsort(pool_costs)]
        if int(pool.shape[0]) > self.k:
            pool = pool[:self.k]
        return pool

    def _quant_dedup(
        self,
        centers: torch.Tensor,
        costs: torch.Tensor,
        cells: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep lowest-cost entry per quantized (x, y) bin."""
        q = self.quant_step
        qx = torch.div(centers[:, 0], q, rounding_mode="trunc").long()
        qy = torch.div(centers[:, 1], q, rounding_mode="trunc").long()
        qy_range = qy.max() - qy.min() + 1 if qy.numel() > 0 else 1
        bin_key = qx * qy_range + (qy - qy.min())

        sort_key = bin_key.float() * (costs.max() - costs.min() + 1.0) + costs
        order = torch.argsort(sort_key)
        sorted_keys = bin_key[order]

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

    def _empty_variant_output(self, k: int) -> torch.Tensor:
        self.action_poses = torch.zeros((k, 2), dtype=torch.float32, device=self.device)
        self.action_costs = torch.full((k,), float("inf"), dtype=torch.float32, device=self.device)
        self.action_variant_indices = None
        return torch.zeros((k,), dtype=torch.bool, device=self.device)

    def _apply_variant_expansion(
        self,
        gid: GroupId,
        center_poses: torch.Tensor,
        center_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Expand center candidates into (center, variant) pairs and keep top-k."""
        spec = self.engine.group_specs[gid]
        V = len(spec.variants)
        max_vi = min(V, self.max_variants)

        vmask = center_mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]

        empty = self._empty_variant_output(k)
        if int(vidx.numel()) == 0:
            return empty

        valid_poses = center_poses[vidx]  # [M, 2]

        # Single batch call — inf where not placeable
        cost_nv = self._score_poses(gid, valid_poses, per_variant=True)  # [M, V]

        # Per-center: keep only top max_variants variants
        if V > max_vi:
            cost_nv = cost_nv.clone()
            _, top_vi = torch.topk(cost_nv, k=max_vi, dim=1, largest=False)
            keep = torch.zeros_like(cost_nv, dtype=torch.bool)
            keep.scatter_(1, top_vi, True)
            cost_nv[~keep] = float("inf")

        flat_cost = cost_nv.reshape(-1)  # [M*V]
        n_finite = int(torch.isfinite(flat_cost).sum().item())
        if n_finite == 0:
            return empty

        pick_k = min(k, n_finite)
        _, topk_flat = torch.topk(flat_cost, k=pick_k, largest=False)
        ci = topk_flat // V
        vi = topk_flat % V

        out_poses = torch.zeros((k, 2), dtype=torch.float32, device=self.device)
        out_delta = torch.full((k,), float("inf"), dtype=torch.float32, device=self.device)
        out_vi = torch.zeros((k,), dtype=torch.int64, device=self.device)
        out_mask = torch.zeros((k,), dtype=torch.bool, device=self.device)

        out_poses[:pick_k] = valid_poses[ci]
        out_delta[:pick_k] = flat_cost[topk_flat]
        out_vi[:pick_k] = vi
        out_mask[:pick_k] = True

        self.action_poses = out_poses
        self.action_costs = out_delta
        self.action_variant_indices = out_vi
        return out_mask

    def _remap_cell_indices_after_expansion(
        self,
        original_poses: torch.Tensor,
        original_cell_ids: torch.Tensor,
    ) -> None:
        """Remap cell_indices to match expanded action_poses from variant expansion."""
        expanded_poses = self.action_poses
        if expanded_poses is None:
            return
        diffs = (expanded_poses[:, None, :] - original_poses[None, :, :]).abs().sum(dim=-1)
        best = torch.argmin(diffs, dim=1)
        self.cell_indices = original_cell_ids[best]


if __name__ == "__main__":
    import time

    from envs.env_loader import load_env

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV4Adapter(
        k=50, cell_size=10, top_per_cell=3, expand_factor=2, quant_step=10.0, random_seed=0,
    )

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.valid_mask.sum().item())
    a = int(torch.where(candidates.valid_mask)[0][0].item()) if valid > 0 else 0

    print("GreedyV4Adapter demo")
    print(f"  env={ENV_JSON}  device={device}  k=50  cell_size=10  top_per_cell=3  expand_factor=2  quant_step=10.0")
    print(f"  valid_actions={valid}  first_valid_action={a}")
    if obs.get("cell_indices") is not None:
        ci = obs["cell_indices"]
        unique_cells = int(ci[:valid].unique().shape[0]) if valid > 0 else 0
        print(f"  unique_cells_in_actions={unique_cells}")
    print(f"  reset_ms={dt_reset_ms:.3f}")

    if valid > 0:
        placement = adapter.resolve_action(a, candidates)
        _obs_env2, _r, _term, _trunc, _info2 = engine.step_placement(placement)
        obs2 = adapter.build_observation()
        candidates2 = adapter.build_action_space()
        valid2 = int(candidates2.valid_mask.sum().item())
        print(f"  valid_after_step={valid2}")
