"""Cell-based adapter with hierarchical search support.

ActionSpace = [R] cells (variable R, non-empty cells only).
Each cell contains up to ``top_per_cell`` (center, variant) candidates.
Agent evaluates Q(cell) from per-cell data (greedy: min cost within cell).
Search operates on cells; ``resolve_action`` maps cell -> best placement.

Supports both flat search (via resolve_action) and hierarchical search
(via sub_action_space / resolve_sub_action / sub_action_costs).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from envs.action_space import ActionSpace
from envs.env import GroupId
from envs.placement.base import GroupPlacement
from ...base import BaseAdapter


@dataclass
class CellData:
    """Per-cell candidate data."""
    cell_id: int
    centroid_x: float
    centroid_y: float
    centers: torch.Tensor           # [M, 2] candidate center coords
    costs: torch.Tensor             # [M] best-variant delta cost per candidate (sorted asc)
    variant_indices: torch.Tensor   # [M] best variant index per candidate


class GreedyV5Adapter(BaseAdapter):
    """Cell-based adapter with hierarchical search support.

    Generates all valid center positions, scores them per-variant, groups
    by coarse cell, and exposes cells as the action space.  Greedy agent
    sees ``obs["action_costs"]`` = per-cell best cost.  Hierarchical search
    can drill into ``sub_action_space(parent_idx)`` for within-cell search.

    Works with:
    - No search (greedy agent) via resolve_action()
    - Flat search (MCTS, Beam, BestFirst) via resolve_action()
    - Hierarchical search (H-MCTS, H-Beam, H-BestFirst) via sub_* methods
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        cell_size: int = 50,
        top_per_cell: int = 20,
        quant_step: Optional[float] = None,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.cell_size = int(cell_size)
        self.top_per_cell = int(top_per_cell)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self._rng = random.Random(random_seed)

        # Per-step state
        self._cells: List[CellData] = []

    @property
    def supports_hierarchical(self) -> bool:
        return True

    # ---- observation / mask ----

    def build_observation(self) -> Dict[str, Any]:
        self.mask = self.create_mask()
        obs: Dict[str, Any] = {}
        if self._cells:
            best = torch.tensor(
                [float(c.costs[0].item()) for c in self._cells],
                dtype=torch.float32,
                device=self.device,
            )
            obs["action_costs"] = best
        return obs

    def create_mask(self) -> torch.Tensor:
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self._cells = []
            self.action_poses = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.action_costs = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.action_variant_indices = None
            return torch.zeros((0,), dtype=torch.bool, device=self.device)

        self._cells = self._generate_cells(gid)
        R = len(self._cells)
        if R == 0:
            self.action_poses = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.action_costs = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.action_variant_indices = None
            return torch.zeros((0,), dtype=torch.bool, device=self.device)

        # ActionSpace centers = cell centroids
        centroids = torch.tensor(
            [[c.centroid_x, c.centroid_y] for c in self._cells],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_poses = centroids
        # Per-cell best cost
        self.action_costs = torch.tensor(
            [float(c.costs[0].item()) for c in self._cells],
            dtype=torch.float32,
            device=self.device,
        )
        # Best variant of best candidate per cell
        self.action_variant_indices = torch.tensor(
            [int(c.variant_indices[0].item()) for c in self._cells],
            dtype=torch.int64,
            device=self.device,
        )
        return torch.ones((R,), dtype=torch.bool, device=self.device)

    # ---- cell generation ----

    def _generate_cells(self, gid: GroupId) -> List[CellData]:
        """Build per-cell candidate data from center_map + per-variant scoring."""
        center_map = self._build_center_map(gid)
        H, W = center_map.shape
        cs = self.cell_size
        W_c = (W + cs - 1) // cs

        all_yx = torch.nonzero(center_map, as_tuple=False)  # [N, 2]
        if all_yx.numel() == 0:
            return []

        all_centers = torch.stack([all_yx[:, 1].float(), all_yx[:, 0].float()], dim=-1)  # [N, 2]

        # Score per-variant, pick best variant per center
        cost_nv = self._score_poses(gid, all_centers, per_variant=True)  # [N, V]
        best_cost, best_vi = cost_nv.min(dim=1)  # [N], [N]

        # Filter out centers with no finite cost
        finite_mask = torch.isfinite(best_cost)
        if not finite_mask.any():
            return []

        if not bool(finite_mask.all()):
            finite_idx = torch.where(finite_mask)[0]
            all_centers = all_centers[finite_idx]
            best_cost = best_cost[finite_idx]
            best_vi = best_vi[finite_idx]

        # Optional quantize dedup: keep lowest-cost center per quantized bin.
        if self.quant_step is not None and self.quant_step > 0:
            all_centers, best_cost, best_vi = self._quant_dedup(
                all_centers, best_cost, best_vi
            )

        # Recompute yx for cell assignment
        all_yx_f = torch.stack([all_centers[:, 1], all_centers[:, 0]], dim=-1).long()  # [N, 2] (y, x)
        flat_cell = (all_yx_f[:, 0] // cs) * W_c + (all_yx_f[:, 1] // cs)  # [N]

        # Sort by (cell, cost)
        cost_range = best_cost.max() - best_cost.min() + 1.0
        sort_key = flat_cell.float() * cost_range + best_cost
        order = torch.argsort(sort_key)

        sorted_cells = flat_cell[order]
        unique_cells, counts = torch.unique_consecutive(sorted_cells, return_counts=True)

        tpc = self.top_per_cell
        start = torch.cumsum(counts, dim=0) - counts  # [C]
        n_cells = int(unique_cells.shape[0])

        cells: List[CellData] = []
        for i in range(n_cells):
            s = int(start[i].item())
            cnt = int(counts[i].item())
            pick = min(cnt, tpc)
            idx = order[s:s + pick]

            c_centers = all_centers[idx]  # [pick, 2]
            c_costs = best_cost[idx]      # [pick]
            c_vis = best_vi[idx]          # [pick]

            cx = float(c_centers[:, 0].mean().item())
            cy = float(c_centers[:, 1].mean().item())

            cells.append(CellData(
                cell_id=int(unique_cells[i].item()),
                centroid_x=cx,
                centroid_y=cy,
                centers=c_centers,
                costs=c_costs,
                variant_indices=c_vis,
            ))

        return cells

    def _quant_dedup(
        self,
        centers: torch.Tensor,
        costs: torch.Tensor,
        variant_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep lowest-cost center per quantized (x, y) bin."""
        if centers.numel() == 0:
            return centers, costs, variant_indices
        q = self.quant_step
        qx = torch.div(centers[:, 0], q, rounding_mode="trunc").long()
        qy = torch.div(centers[:, 1], q, rounding_mode="trunc").long()
        qy_range = qy.max() - qy.min() + 1 if qy.numel() > 0 else 1
        bin_key = qx * qy_range + (qy - qy.min())

        # Sort by (bin_key, cost) so first occurrence per bin is cheapest.
        sort_key = bin_key.float() * (costs.max() - costs.min() + 1.0) + costs
        order = torch.argsort(sort_key)
        sorted_keys = bin_key[order]

        # First occurrence per bin.
        first_mask = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=centers.device)
        if sorted_keys.shape[0] > 1:
            first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]

        keep = order[first_mask]
        return centers[keep], costs[keep], variant_indices[keep]

    # ---- resolve ----

    def resolve_action(self, action_idx: int, action_space: ActionSpace) -> GroupPlacement:
        """Cell index -> GroupPlacement using best candidate in cell."""
        a = self.validate_action_index(action_idx, action_space)
        cell = self._cells[a]

        # Best candidate (already sorted by cost)
        center = cell.centers[0]
        vi_idx = int(cell.variant_indices[0].item())

        gid = action_space.group_id
        spec = self.engine.group_specs[gid]
        vi = spec.variants[vi_idx]
        x_bl = int(round(float(center[0].item()) - float(vi.body_width) / 2.0))
        y_bl = int(round(float(center[1].item()) - float(vi.body_height) / 2.0))
        return spec.build_placement(variant_index=vi_idx, x_bl=x_bl, y_bl=y_bl)

    # ---- hierarchical support ----

    def sub_action_space(self, parent_idx: int) -> ActionSpace:
        """Within-cell candidates as ActionSpace for hierarchical search."""
        cell = self._cells[parent_idx]
        return ActionSpace(
            centers=cell.centers,
            valid_mask=torch.ones(cell.centers.shape[0], dtype=torch.bool, device=self.device),
            group_id=self.current_gid(),
            variant_indices=cell.variant_indices,
        )

    def resolve_sub_action(
        self,
        action_idx: int,
        action_space: ActionSpace,
        *,
        parent_idx: int,
    ) -> GroupPlacement:
        """Within-cell candidate index -> concrete placement."""
        a = self.validate_action_index(action_idx, action_space)
        if parent_idx < 0 or parent_idx >= len(self._cells):
            raise IndexError(f"parent_idx out of range: {parent_idx}")

        gid = action_space.group_id
        spec = self.engine.group_specs[gid]
        center = action_space.centers[a]
        vi_idx = int(action_space.variant_indices[a].item()) if action_space.variant_indices is not None else 0
        cell = self._cells[int(parent_idx)]
        if a >= int(cell.costs.shape[0]):
            raise IndexError(f"sub action index out of range for cell {parent_idx}: {a}")

        vi = spec.variants[vi_idx]
        x_bl = int(round(float(center[0].item()) - float(vi.body_width) / 2.0))
        y_bl = int(round(float(center[1].item()) - float(vi.body_height) / 2.0))
        return spec.build_placement(variant_index=vi_idx, x_bl=x_bl, y_bl=y_bl)

    def sub_action_costs(self, parent_idx: int) -> torch.Tensor:
        if parent_idx < 0 or parent_idx >= len(self._cells):
            raise IndexError(f"parent_idx out of range: {parent_idx}")
        return self._cells[parent_idx].costs

    # ---- state snapshot (for search) ----

    def get_state_copy(self) -> Dict[str, object]:
        state = super().get_state_copy()
        state["rng_state"] = self._rng.getstate()
        state["_cells"] = list(self._cells)
        ap = getattr(self, "action_poses", None)
        state["action_poses"] = ap.clone() if isinstance(ap, torch.Tensor) else None
        ac = getattr(self, "action_costs", None)
        state["action_costs"] = ac.clone() if isinstance(ac, torch.Tensor) else None
        return state

    def set_state(self, state: Dict[str, object]) -> None:
        super().set_state(state)
        rs = state.get("rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        self._cells = list(state.get("_cells", []))
        ap = state.get("action_poses", None)
        self.action_poses = ap.to(device=self.device, dtype=torch.float32).clone() if isinstance(ap, torch.Tensor) else None
        ac = state.get("action_costs", None)
        self.action_costs = ac.to(device=self.device, dtype=torch.float32).clone() if isinstance(ac, torch.Tensor) else None


if __name__ == "__main__":
    import time
    import torch

    from envs.env_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "envs/env_configs/mixed_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV5Adapter(
        cell_size=50, top_per_cell=20, quant_step=10.0,
    )

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    R = int(candidates.valid_mask.sum().item())
    print("GreedyV5Adapter demo")
    print(f"  env={ENV_JSON}  device={device}")
    print(f"  cell_size={adapter.cell_size}  top_per_cell={adapter.top_per_cell}")
    print(f"  cells={R}  total_candidates={sum(len(c.centers) for c in adapter._cells)}")
    print(f"  reset_ms={dt_ms:.3f}")

    if R > 0:
        costs = obs.get("action_costs")
        if costs is not None:
            print(f"  best_cell_cost={float(costs.min().item()):.3f}")
            print(f"  worst_cell_cost={float(costs.max().item()):.3f}")

        # Resolve best cell
        best_cell = int(torch.argmin(costs).item())
        placement = adapter.resolve_action(best_cell, candidates)
        print(
            f"  best_cell={best_cell}  gid={placement.group_id}  "
            f"pos=({placement.x_center:.1f},{placement.y_center:.1f})"
        )

    # Step through all groups
    print("\n  Full episode:")
    engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    step = 0
    total_reward = 0.0
    while engine.get_state().remaining:
        step += 1
        obs = adapter.build_observation()
        candidates = adapter.build_action_space()
        R = int(candidates.valid_mask.sum().item())
        if R == 0:
            print(f"    step {step}: no valid cells")
            break
        costs = obs["action_costs"]
        best = int(torch.argmin(costs).item())
        placement = adapter.resolve_action(best, candidates)
        _, reward, terminated, truncated, info = engine.step_placement(placement)
        total_reward += reward
        print(
            f"    step {step}: gid={placement.group_id}  cells={R}  "
            f"pos=({placement.x_center:.1f},{placement.y_center:.1f})  reward={reward:.3f}"
        )
        if terminated or truncated:
            break
    print(f"  total_cost={engine.total_cost():.3f}  total_reward={total_reward:.3f}")
