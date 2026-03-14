from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId  # new env (renamed from env_new)
from envs.action_space import ActionSpace

from .base import BaseDecisionAdapter


class GreedyV3DecisionAdapter(BaseDecisionAdapter):
    """Top-K candidate wrapper: Discrete(K) actions over an in-file TopK generator.

    Notes:
    - Candidate coordinates are bottom-left integer coordinates (engine contract).
    - `action_mask` is torch.BoolTensor[K] (True means valid).
    - `action_xyrot` is torch.LongTensor[K,3] of (x_bl, y_bl, rot).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int = 50,
        # B-1 (edge-based): sample candidates from the *boundary* of valid top-left map.
        quant_step: Optional[float] = 10.0,
        oversample_factor: int = 2,
        edge_ratio: float = 0.8,
        random_seed: Optional[int] = None,
        optimize_rotation: bool = False,
    ):
        super().__init__()
        self.k = int(k)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.oversample_factor = int(oversample_factor)
        self.edge_ratio = float(edge_ratio)
        self._rng = random.Random(random_seed)
        self.optimize_rotation = optimize_rotation

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        self.action_xyrot: Optional[torch.Tensor] = None  # long [K,3]
        self.action_delta: Optional[torch.Tensor] = None  # float [K]

    def create_mask(self) -> torch.Tensor:
        # Keep candidate sampling deterministic from engine state.
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.action_xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
            self.action_delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
            return torch.zeros((self.k,), dtype=torch.bool, device=self.device)

        candidates, mask = self._generate(self.engine, gid)

        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
        for i, c in enumerate(candidates[: self.k]):
            _, x_bl, y_bl, rot = c
            xyrot[i, 0] = int(x_bl)
            xyrot[i, 1] = int(y_bl)
            xyrot[i, 2] = int(rot)
        self.action_xyrot = xyrot
        delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
        vmask = mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]
        if int(vidx.numel()) > 0:
            vv = xyrot[vidx]
            d = self.engine.delta_cost(gid=gid, x=vv[:, 0], y=vv[:, 1], rot=vv[:, 2]).to(
                dtype=torch.float32, device=self.device
            )
            delta[vidx] = d.view(-1)
        self.action_delta = delta
        return mask

    # ---- state api (for wrapped search/MCTS) ----
    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        snap["rng_state"] = self._rng.getstate()
        if isinstance(self.action_xyrot, torch.Tensor):
            snap["action_xyrot"] = self.action_xyrot.clone()
        else:
            snap["action_xyrot"] = None
        if isinstance(self.action_delta, torch.Tensor):
            snap["action_delta"] = self.action_delta.clone()
        else:
            snap["action_delta"] = None
        return snap

    def set_state(self, state: Dict[str, object]) -> None:
        super().set_state(state)
        rs = state.get("rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        ax = state.get("action_xyrot", None)
        if isinstance(ax, torch.Tensor):
            self.action_xyrot = ax.to(device=self.device, dtype=torch.long).clone()
        else:
            self.action_xyrot = None
        ad = state.get("action_delta", None)
        if isinstance(ad, torch.Tensor):
            self.action_delta = ad.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_delta = None

    # ---- candidate generation (BL int coords) ----

    def _wh_int(self, env: FactoryLayoutEnv, gid: GroupId, rot: int) -> Tuple[int, int]:
        g = env.group_specs[gid]
        w, h = g._rotated_size(int(rot))
        return int(round(float(w))), int(round(float(h)))

    def _clamp_bl(self, env: FactoryLayoutEnv, x_bl: int, y_bl: int, w: int, h: int) -> Tuple[int, int]:
        max_x = int(env.grid_width) - int(w)
        max_y = int(env.grid_height) - int(h)
        if max_x < 0 or max_y < 0:
            return 0, 0
        x2 = max(0, min(int(x_bl), max_x))
        y2 = max(0, min(int(y_bl), max_y))
        return int(x2), int(y2)

    def _pad_candidates(self, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        return [(gid, 0, 0, 0) for _ in range(count)]

    def _dedup_tagged(
        self,
        candidates: List[Tuple[int, Tuple[GroupId, int, int, int]]],
        q: float,
        group: object,
    ) -> List[Tuple[int, Tuple[GroupId, int, int, int]]]:
        if q <= 0:
            return candidates
        seen = set()
        unique: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        for src, c in candidates:
            qx = int(round(float(c[1]) / q))
            qy = int(round(float(c[2]) / q))
            key = (qx, qy, int(c[3]))
            if key not in seen:
                seen.add(key)
                unique.append((src, c))
        return unique

    def _cheap_reject_body(
        self,
        env: FactoryLayoutEnv,
        *,
        gid: GroupId,
        x_bl: int,
        y_bl: int,
        rot: int,
        wh_cache: Dict[int, Tuple[int, int]],
        valid_by_rot: Dict[int, torch.Tensor],
    ) -> bool:
        w, h = wh_cache[int(rot)]
        gw = int(env.grid_width)
        gh = int(env.grid_height)

        if x_bl < 0 or y_bl < 0 or (x_bl + w) > gw or (y_bl + h) > gh:
            return True
        if w <= 0 or h <= 0:
            return True
        valid_map = valid_by_rot.get(int(rot), None)
        if not isinstance(valid_map, torch.Tensor):
            return True
        H, W = int(valid_map.shape[0]), int(valid_map.shape[1])
        if x_bl < 0 or y_bl < 0 or x_bl >= W or y_bl >= H:
            return True
        return not bool(valid_map[int(y_bl), int(x_bl)].item())

    def _build_valid_map(self, env: FactoryLayoutEnv, *, gid: GroupId, rot: int) -> torch.Tensor:
        return env.get_state().is_placeable_map(gid=gid, spec=env.group_specs[gid], rot=int(rot))

    def _torch_gen(self, *, env: FactoryLayoutEnv) -> torch.Generator:
        # Use python RNG as the source-of-truth; seed a torch.Generator for tensor sampling.
        seed = int(self._rng.randrange(0, 2**31 - 1))
        g = torch.Generator(device=env.device)
        g.manual_seed(seed)
        return g

    def _sample_from_valid(
        self,
        *,
        env: FactoryLayoutEnv,
        gid: GroupId,
        rot: int,
        valid_map: torch.Tensor,
        count: int,
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        if not isinstance(valid_map, torch.Tensor) or valid_map.numel() == 0:
            return []
        idx = torch.nonzero(valid_map, as_tuple=False)  # [M,2] of (y,x)
        if idx.numel() == 0:
            return []
        M = int(idx.shape[0])
        if M <= count:
            pick = idx
        else:
            perm = torch.randperm(M, generator=gen, device=env.device)[: int(count)]
            pick = idx[perm]
        out: List[Tuple[GroupId, int, int, int]] = []
        for t in range(int(pick.shape[0])):
            y_bl = int(pick[t, 0].item())
            x_bl = int(pick[t, 1].item())
            out.append((gid, x_bl, y_bl, int(rot)))
        return out

    def _build_edge_map(self, valid_map: torch.Tensor) -> torch.Tensor:
        """B-1 edge sampling mask: edge = valid & dilate(~valid).

        - valid_map is bool[H2,W2] of *placeable top-left* positions (footprint+clearance aware).
        - edge_map marks valid positions adjacent to any invalid cell (3x3 neighborhood).
        """
        if (not isinstance(valid_map, torch.Tensor)) or valid_map.numel() == 0:
            return valid_map
        v = valid_map.to(dtype=torch.bool)
        inv = (~v).to(dtype=torch.float32).view(1, 1, int(v.shape[0]), int(v.shape[1]))
        kernel = torch.ones((1, 1, 3, 3), device=valid_map.device, dtype=inv.dtype)
        # padding=1 keeps same H/W
        nbr = (F.conv2d(inv, kernel, padding=1) > 0).squeeze(0).squeeze(0)
        return v & nbr

    def _generate(
        self, env: FactoryLayoutEnv, next_group_id: GroupId
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        group = env.group_specs[next_group_id]
        rotations = (0, 90) if group.rotatable else (0,)
        wh_cache = {int(r): self._wh_int(env, next_group_id, int(r)) for r in rotations}
        valid_by_rot = {int(r): self._build_valid_map(env, gid=next_group_id, rot=int(r)) for r in rotations}
        edge_by_rot = {int(r): self._build_edge_map(valid_by_rot[int(r)]) for r in rotations}
        gen = self._torch_gen(env=env)
        q = float(self.quant_step) if self.quant_step is not None else 1.0

        # B-1: sample mostly from edge of valid map, and fill the rest from valid interior.
        total_k = max(1, int(self.k * self.oversample_factor))
        edge_ratio = max(0.0, min(1.0, float(self.edge_ratio)))
        n_edge = int(round(float(total_k) * edge_ratio))
        n_fill = int(total_k - n_edge)

        rots = list(rotations)
        per_rot_edge = max(1, int(round(float(max(1, n_edge)) / float(max(1, len(rots))))))
        per_rot_fill = max(1, int(round(float(max(1, n_fill)) / float(max(1, len(rots))))))

        edge_pool: List[Tuple[GroupId, int, int, int]] = []
        fill_pool: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            edge_pool.extend(self._sample_from_valid(env=env, gid=next_group_id, rot=int(rot), valid_map=edge_by_rot[int(rot)], count=per_rot_edge, gen=gen))
            fill_pool.extend(self._sample_from_valid(env=env, gid=next_group_id, rot=int(rot), valid_map=valid_by_rot[int(rot)], count=per_rot_fill, gen=gen))

        # Trim to targets (keep edge first)
        edge_pool = edge_pool[: int(n_edge)]
        fill_pool = fill_pool[: int(n_fill)]

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in edge_pool)
        raw_tagged.extend((1, c) for c in fill_pool)

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        prefiltered_tagged = [
            (src, c)
            for src, c in unique_tagged
            if not self._cheap_reject_body(
                env,
                gid=next_group_id,
                x_bl=int(c[1]),
                y_bl=int(c[2]),
                rot=int(c[3]),
                wh_cache=wh_cache,
                valid_by_rot=valid_by_rot,
            )
        ]
        valid_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        if prefiltered_tagged:
            xyrot = torch.tensor(
                [[int(c[1]), int(c[2]), int(c[3])] for _, c in prefiltered_tagged],
                dtype=torch.long,
                device=device,
            )
            placeable = env.is_placeable_mask(
                ActionSpace(
                    xyrot=xyrot,
                    mask=torch.ones((int(xyrot.shape[0]),), dtype=torch.bool, device=device),
                    gid=next_group_id,
                )
            )
            for i, tagged in enumerate(prefiltered_tagged):
                if bool(placeable[i].item()):
                    valid_tagged.append(tagged)
        # Keep order: edge candidates first, then fill. No scoring in v3.
        final: List[Tuple[GroupId, int, int, int]] = [c for _src, c in valid_tagged][: int(self.k)]
        if self.optimize_rotation:
            final = self._optimize_rotation(env, next_group_id, final)
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(next_group_id, self.k - len(final)))
        return final, mask

    def _optimize_rotation(
        self, env: FactoryLayoutEnv, gid: GroupId, candidates: List[Tuple[GroupId, int, int, int]]
    ) -> List[Tuple[GroupId, int, int, int]]:
        """0 vs 180, 90 vs 270 중 점수가 더 좋은 회전을 선택"""
        if not candidates:
            return candidates

        x = torch.tensor([c[1] for c in candidates], device=env.device)
        y = torch.tensor([c[2] for c in candidates], device=env.device)
        rot_orig = torch.tensor([c[3] for c in candidates], device=env.device)
        rot_alt = (rot_orig + 180) % 360

        scores_orig = env.delta_cost(gid=gid, x=x, y=y, rot=rot_orig)
        scores_alt = env.delta_cost(gid=gid, x=x, y=y, rot=rot_alt)

        use_alt = scores_alt < scores_orig
        final_rot = torch.where(use_alt, rot_alt, rot_orig)

        return [(c[0], c[1], c[2], int(final_rot[i].item())) for i, c in enumerate(candidates)]


if __name__ == "__main__":
    import torch

    from envs.action_space import ActionSpace as CandidateSet
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.env_visualizer import plot_layout

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV3DecisionAdapter(k=50, quant_step=10.0, oversample_factor=2, edge_ratio=0.8, random_seed=0)

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.mask.sum().item())
    a = int(torch.where(candidates.mask)[0][0].item()) if valid > 0 else 0

    # Plot: initial candidates (interactive; close to continue)
    plot_layout(engine, action_space=candidates)

    t1 = time.perf_counter()
    placement = adapter.decode_action(a, candidates)
    _obs_env2, _r, _term, _trunc, _info2 = engine.step_action(placement)
    obs2 = adapter.build_observation()
    candidates2 = adapter.build_action_space()
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot: after 1 placement + new candidates (if any)
    if int(candidates2.mask.shape[0]) > 0:
        plot_layout(engine, action_space=candidates2)
    else:
        plot_layout(engine, action_space=None)

    print("[GreedyDecisionAdapter demo]")
    print(" env=", ENV_JSON, "device=", device, "k=", 50)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")
