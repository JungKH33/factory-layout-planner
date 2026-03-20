from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId

from ...base import BaseAdapter
from envs.action_space import ActionSpace


class MaskPlaceAdapter(BaseAdapter):
    """MaskPlace dense-grid wrapper: Discrete(G*G) actions (default G=224).

    Obs provides:
    - action_mask: bool [G*G] (True means valid)
    - state: float32 [1 + 5*G*G + 2]  (pos_idx + 5 maps + extra2)
      maps are flattened in order: canvas, net_img, mask, net_img_2, mask_2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid: int = 224,
        soft_coefficient: float = 1.0,
    ):
        super().__init__()
        self.grid_width = 1
        self.grid_height = 1
        self.grid = int(grid)
        self.soft_coefficient = float(soft_coefficient)

        self.action_space = gym.spaces.Discrete(self.grid * self.grid)
        self.observation_space = gym.spaces.Dict({})

        # cached last-built maps (for debugging/visualization)
        self._last_maps: Optional[torch.Tensor] = None  # float32 [5,G,G]
        self.action_poses: Optional[torch.Tensor] = None  # float32 [G*G,2]

    def bind(self, engine: FactoryLayoutEnv) -> None:
        super().bind(engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)

    def cell_wh(self) -> Tuple[int, int]:
        g = int(self.grid)
        cell_w = int(math.ceil(self.grid_width / float(g)))
        cell_h = int(math.ceil(self.grid_height / float(g)))
        return cell_w, cell_h

    def _next_gid(self) -> Optional[GroupId]:
        return self.current_gid()

    def _gid_at(self, k: int) -> Optional[GroupId]:
        if k < 0:
            return None
        if len(self.engine.get_state().remaining) <= k:
            return None
        return self.engine.get_state().remaining[k]

    def _action_poses_for_gid(self, *, gid: Optional[GroupId]) -> torch.Tensor:
        """Return float32 [G*G,2] table mapping action index -> (x_c, y_c) center coordinates."""
        g = int(self.grid)
        if gid is None:
            return torch.zeros((g * g, 2), dtype=torch.float32, device=self.device)

        cell_w, cell_h = self.cell_wh()
        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)
        return torch.stack([cx, cy], dim=-1).view(g * g, 2).to(dtype=torch.float32)

    def _create_mask_for_gid(self, *, gid: Optional[GroupId]) -> torch.Tensor:
        """Return torch.BoolTensor[G*G] valid-action mask for a specific gid.

        Uses spec.placeable_batch which checks ALL rotation/mirror variants
        and ORs the results: a center cell is valid if the facility can be
        placed there in *any* variant.
        """
        if gid is None:
            return torch.zeros((self.grid * self.grid,), dtype=torch.bool, device=self.device)

        spec = self.engine.group_specs[gid]
        state = self.engine.get_state()
        g = int(self.grid)
        cell_w, cell_h = self.cell_wh()
        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        ok = spec.placeable_batch(
            state=state, gid=gid,
            x_c=cx.reshape(-1), y_c=cy.reshape(-1),
        )
        return ok

    def _score_map_for_gid(self, *, gid: Optional[GroupId]) -> torch.Tensor:
        """Return score map [G,G] for a specific gid using batch delta_cost (lower is better)."""
        g = int(self.grid)
        if gid is None:
            return torch.zeros((g, g), dtype=torch.float32, device=self.device)

        cell_w, cell_h = self.cell_wh()
        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        poses = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)  # [G*G, 2] float
        scores = self._score_poses(gid, poses).to(dtype=torch.float32)
        return scores.view(g, g)

    def create_mask(self) -> torch.Tensor:
        gid = self._next_gid()
        self.action_poses = self._action_poses_for_gid(gid=gid)
        return self._create_mask_for_gid(gid=gid)

    def _build_maps_and_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (maps[5,G,G], state[1+5*G*G+2])."""
        gid = self._gid_at(0)
        gid2 = self._gid_at(1)
        g = int(self.grid)
        g2 = g * g

        # --- hard mask map (1=invalid) derived from current wrapper mask ---
        assert self.mask is not None
        mask_valid = self.mask.view(g, g)
        mask_map = (~mask_valid).to(torch.float32)  # 1 where invalid

        # --- canvas: downsampled occupancy/static/clearance map ---
        # Values (before downsample, on engine grid):
        # - 1.0: occupied (placed body) or static blocked (forbidden/column)
        # - 0.5: clearance only (from already-placed facilities)
        # - 0.0: free
        #
        # Use area downsampling to preserve "partial coverage" information.
        occ_or_static = (self.engine.get_maps().occ_invalid | self.engine.get_maps().static_invalid)
        canvas_full = torch.zeros((self.grid_height, self.grid_width), dtype=torch.float32, device=self.device)
        canvas_full[occ_or_static] = 1.0
        canvas_src = canvas_full.view(1, 1, self.grid_height, self.grid_width)
        canvas = F.interpolate(canvas_src, size=(g, g), mode="area").squeeze(0).squeeze(0)

        # --- net_img / net_img_2: objective-based score maps (lower is better) ---
        # NOTE: We intentionally do NOT normalize or special-case invalid cells here (per request).
        net_img = self._score_map_for_gid(gid=gid)
        if gid2 is None:
            net_img_2 = net_img.clone()
            mask_2 = mask_map.clone()
        else:
            net_img_2 = self._score_map_for_gid(gid=gid2)
            mask2_valid = self._create_mask_for_gid(gid=gid2).view(g, g)
            mask_2 = (~mask2_valid).to(torch.float32)

        # --- normalize reward maps to [0,1] (global min/max over the grid; option-1) ---
        def _minmax01(x: torch.Tensor) -> torch.Tensor:
            x = x.to(dtype=torch.float32)
            x_min = torch.min(x)
            x_max = torch.max(x)
            denom = x_max - x_min
            eps = 1.0e-6
            if float(denom.item()) <= eps:
                return torch.zeros_like(x)
            return (x - x_min) / (denom + eps)

        net_img = _minmax01(net_img)
        net_img_2 = _minmax01(net_img_2)

        maps = torch.stack([canvas, net_img, mask_map, net_img_2, mask_2], dim=0)  # [5,G,G]

        # state = [pos_idx] + [5 maps flat] + [extra2]
        pos_idx = float(self.engine.gid_to_idx.get(gid, 0)) if gid is not None else 0.0
        if gid2 is None:
            extra2 = torch.zeros((2,), device=self.device, dtype=torch.float32)
        else:
            spec2 = self.engine.group_specs[gid2]
            extra2 = torch.tensor(
                [
                    float(spec2.width) / float(self.grid_width),
                    float(spec2.height) / float(self.grid_height),
                ],
                dtype=torch.float32,
                device=self.device,
            )
        state = torch.empty((1 + 5 * g2 + 2,), dtype=torch.float32, device=self.device)
        state[0] = float(pos_idx)
        state[1 : 1 + 5 * g2] = maps.reshape(-1)
        state[1 + 5 * g2 :] = extra2.to(dtype=torch.float32).view(-1)[:2]
        return maps, state

    def build_observation(self) -> Dict[str, Any]:
        """MaskPlace observation: state vector + 5 map channels."""
        self.mask = self.create_mask()
        maps, state = self._build_maps_and_state()
        self._last_maps = maps
        return {
            "state": state,
            "canvas": maps[0],
            "net_img": maps[1],
            "invalid_map": maps[2],
            "net_img_2": maps[3],
            "invalid_2_map": maps[4],
        }

    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        if isinstance(self.action_poses, torch.Tensor):
            snap["action_poses"] = self.action_poses.clone()
        else:
            snap["action_poses"] = None
        if isinstance(self._last_maps, torch.Tensor):
            snap["_last_maps"] = self._last_maps.clone()
        else:
            snap["_last_maps"] = None
        return snap

    def set_state(self, state: Dict[str, object]) -> None:
        super().set_state(state)
        ax = state.get("action_poses", None)
        if isinstance(ax, torch.Tensor):
            self.action_poses = ax.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_poses = None
        lm = state.get("_last_maps", None)
        if isinstance(lm, torch.Tensor):
            self._last_maps = lm.to(device=self.device, dtype=torch.float32).clone()
        else:
            self._last_maps = None

if __name__ == "__main__":
    import time

    import torch
    import matplotlib.pyplot as plt

    from envs.action_space import ActionSpace as CandidateSet
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.env_visualizer import plot_layout

    ENV_JSON = "envs/env_configs/placed_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = MaskPlaceAdapter(grid=224)

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.mask.sum().item())
    a = int(torch.where(candidates.mask)[0][0].item()) if valid > 0 else 0

    # For visualization, subsample valid points (224^2 can be large).
    idxs = torch.where(candidates.mask)[0][:5000]
    xy = candidates.poses[idxs]
    cand0 = CandidateSet(
        poses=xy,
        mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device),
        gid=candidates.gid,
    )
    plot_layout(engine, action_space=cand0)

    t1 = time.perf_counter()
    placement = adapter.decode_action(a, candidates)
    _obs_env2, _r, _term, _trunc, _info2 = engine.step_action(placement)
    obs2 = adapter.build_observation()
    candidates2 = adapter.build_action_space()
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot after one placement (subsample new valid points)
    if int(candidates2.mask.shape[0]) > 0:
        idxs2 = torch.where(candidates2.mask)[0][:5000]
        xy2 = candidates2.poses[idxs2]
        cand1 = CandidateSet(
            poses=xy2,
            mask=torch.ones((xy2.shape[0],), dtype=torch.bool, device=device),
            gid=candidates2.gid,
        )
        plot_layout(engine, action_space=cand1)
    else:
        plot_layout(engine, action_space=None)

    print("MaskPlaceAdapter demo")
    print(" env=", ENV_JSON, "device=", device, "grid=", adapter.grid)
    print(" valid_actions=", valid, "first_valid_action=", a, "plotted=", int(xy.shape[0]))
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")

    # Show maps: canvas(map0), net_img(map1), invalid(mask=map2), net_img_2(map3), mask_2(map4) (close to continue)
    if isinstance(adapter._last_maps, torch.Tensor) and adapter._last_maps.numel() > 0:
        maps = adapter._last_maps.detach().to(device="cpu", dtype=torch.float32).numpy()  # [5,G,G]
        canvas = maps[0]
        net_img = maps[1]
        invalid = maps[2]
        net_img_2 = maps[3]
        invalid_2 = maps[4]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs = axs.reshape(-1)
        axs[0].set_title("map0: canvas map")
        im0 = axs[0].imshow(canvas, origin="lower")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        axs[1].set_title("map1: estimate reward map")
        im1 = axs[1].imshow(net_img, origin="lower")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].set_title("map2: invalid map")
        im2 = axs[2].imshow(invalid, origin="lower")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # axs[3].set_title("map3: estimate reward map (next gid)")
        # im3 = axs[3].imshow(net_img_2, origin="lower")
        # fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        # axs[4].set_title("map4: invalid map (next gid)")
        # im4 = axs[4].imshow(invalid_2, origin="lower")
        # fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

        # axs[5].axis("off")

        plt.tight_layout()
        plt.show()
