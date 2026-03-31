from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyAdapter(BaseAdapter):
    """Top-K candidate wrapper: Discrete(K) actions over an in-file TopK generator.

    Notes:
    - Candidate coordinates are bottom-left integer coordinates (engine contract).
    - `action_mask` is torch.BoolTensor[K] (True means valid).
    - `action_poses` is torch.FloatTensor[K,2] of (x_center, y_center) center coordinates.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int,
        # Defaults MUST match `actionspace/topk.py:TopKSelector` for parity.
        scan_step: float = 2000.0,
        quant_step: Optional[float] = 10.0,
        p_high: float = 0.1,
        p_near: float = 0.8,
        p_coarse: float = 0.0,
        oversample_factor: int = 2,
        diversity_ratio: float = 0.0,  # parity (unused)
        min_diversity: int = 0,  # parity (unused)
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.k = int(k)
        self.scan_step = float(scan_step)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.p_high = float(p_high)
        self.p_near = float(p_near)
        self.p_coarse = float(p_coarse)
        self.oversample_factor = int(oversample_factor)
        self.diversity_ratio = float(diversity_ratio)
        self.min_diversity = int(min_diversity)
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
        return obs

    def create_mask(self) -> torch.Tensor:
        # Keep candidate sampling deterministic from engine state.
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.action_poses = torch.zeros((self.k, 2), dtype=torch.float32, device=self.device)
            self.action_costs = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
            return torch.zeros((self.k,), dtype=torch.bool, device=self.device)

        candidates, mask = self._generate(self.engine, gid)

        poses = torch.zeros((self.k, 2), dtype=torch.float32, device=self.device)
        spec = self.engine.group_specs[gid]
        for i, c in enumerate(candidates[: self.k]):
            _, x_bl, y_bl, rotation = c
            w, h = spec.rotated_size(int(rotation))
            poses[i, 0] = float(x_bl) + float(w) / 2.0
            poses[i, 1] = float(y_bl) + float(h) / 2.0

        self.action_poses = poses
        self.action_variant_indices = None
        delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
        vmask = mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]
        if int(vidx.numel()) > 0:
            vv = poses[vidx]
            d = self._score_poses(gid, vv).to(dtype=torch.float32, device=self.device)
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

    # ---- candidate generation (copied from actionspace/topk.py; BL int coords) ----
    def _quota(self, k: int) -> Tuple[int, int, int, int]:
        n_high = round(k * self.p_high)
        n_near = round(k * self.p_near)
        n_coarse = round(k * self.p_coarse)
        n_rand = k - (n_high + n_near + n_coarse)
        if n_rand < 0:
            n_rand = 0
        return int(n_high), int(n_near), int(n_coarse), int(n_rand)

    def _wh_int(self, env: FactoryLayoutEnv, gid: GroupId, rotation: int) -> Tuple[int, int]:
        g = env.group_specs[gid]
        w, h = g.rotated_size(int(rotation))
        return int(w), int(h)

    def _clamp_bl(self, env: FactoryLayoutEnv, x_bl: int, y_bl: int, w: int, h: int) -> Tuple[int, int]:
        max_x = int(env.grid_width) - int(w)
        max_y = int(env.grid_height) - int(h)
        if max_x < 0 or max_y < 0:
            return 0, 0
        x2 = max(0, min(int(x_bl), max_x))
        y2 = max(0, min(int(y_bl), max_y))
        return int(x2), int(y2)

    def _scan_axis_bl(self, limit: int, size: int, step: int) -> List[int]:
        if step <= 0:
            step = 1
        end = int(limit) - int(size)
        if end < 0:
            return []
        return list(range(0, end + 1, int(step)))

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
            _, x_bl, y_bl, rotation = c
            w, h = group.rotated_size(int(rotation))
            x_center = float(x_bl) + float(w) / 2.0
            y_center = float(y_bl) + float(h) / 2.0
            qx = int(round(x_center / q))
            qy = int(round(y_center / q))
            key = (qx, qy)
            if key not in seen:
                seen.add(key)
                unique.append((src, c))
        return unique

    def _validate_with_maps(
        self,
        candidates: torch.Tensor,
        valid_by_rotation: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Vectorized placement validation using pre-computed placeable maps."""
        N = int(candidates.shape[0])
        result = torch.zeros(N, dtype=torch.bool, device=self.device)
        x, y, rotation = candidates[:, 0], candidates[:, 1], candidates[:, 2]
        for o, vmap in valid_by_rotation.items():
            H, W = int(vmap.shape[0]), int(vmap.shape[1])
            rotation_match = (rotation == o)
            if not rotation_match.any():
                continue
            xi, yi = x[rotation_match], y[rotation_match]
            in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            xc = xi.clamp(0, W - 1)
            yc = yi.clamp(0, H - 1)
            result[rotation_match] = vmap[yc, xc] & in_bounds
        return result

    def _build_rotation_valid_map(self, env: FactoryLayoutEnv, *, gid: GroupId, rotation: int) -> torch.Tensor:
        """Placeable map for the given rotation (0/90/180/270)."""
        spec = env.group_specs[gid]
        state = env.get_state()
        rr = spec._resolve_rotation(rotation)
        result = None
        seen_shape: set = set()
        for vi in spec._variants:  # TODO: remove direct rotation filtering, use spec.variants
            if vi.rotation != rr:
                continue
            shape_key = vi.shape_key
            if shape_key in seen_shape:
                continue
            seen_shape.add(shape_key)
            body_mask, clearance_mask, clearance_origin, is_rectangular = spec.shape_tensors(shape_key)
            m = state.placeable_map(
                gid=gid,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                is_rectangular=is_rectangular,
            )
            if result is None:
                result = m
            else:
                result = result | m
        if result is None:
            H, W = state.maps.shape
            return torch.zeros((H, W), dtype=torch.bool, device=env.device)
        return result

    def _score_sorted(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        pool: List[Tuple[GroupId, int, int, int]],
    ) -> List[Tuple[GroupId, int, int, int]]:
        if not pool:
            return []
        spec = env.group_specs[gid]
        centers = []
        for c in pool:
            _, x_bl, y_bl, rotation = c
            w, h = spec.rotated_size(int(rotation))
            centers.append([float(x_bl) + float(w) / 2.0, float(y_bl) + float(h) / 2.0])
        xy = torch.tensor(centers, dtype=torch.float32, device=env.device)
        scores_t = self._score_poses(gid, xy)
        scores = scores_t.detach().to(device="cpu", dtype=torch.float32).tolist()
        order = sorted(range(len(pool)), key=lambda i: scores[i])
        return [pool[i] for i in order]

    def _random_take(self, pool: List[Tuple[GroupId, int, int, int]], count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            return list(pool)
        return self._rng.sample(pool, count)

    def _source_stratified(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        group = env.group_specs[gid]
        rotations = (0, 90, 180, 270) if group.rotatable else (0,)

        count_per_rotation = max(1, count // len(rotations))
        aspect_ratio = float(env.grid_width) / float(env.grid_height)
        nx = max(1, round(math.sqrt(count_per_rotation * aspect_ratio)))
        ny = max(1, round(count_per_rotation / nx))

        dx = float(env.grid_width) / float(nx)
        dy = float(env.grid_height) / float(ny)

        candidates: List[Tuple[GroupId, int, int, int]] = []
        for rotation in rotations:
            w, h = self._wh_int(env, gid, rotation)
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for i in range(nx):
                for j in range(ny):
                    x_bl = int(round(i * dx))
                    y_bl = int(round(j * dy))
                    x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                    candidates.append((gid, int(x_bl), int(y_bl), int(rotation)))
        return candidates

    def _source_high(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        group = env.group_specs[gid]
        rotations = (0, 90, 180, 270) if group.rotatable else (0,)
        step = max(int(round(self.scan_step)), 1)
        max_scan = 50000

        def _jump_x_bl(x_bl: int, y_bl: int, w: int, h: int, direction: int) -> Optional[int]:
            y0 = int(y_bl)
            y1 = int(y_bl) + int(h)
            best: Optional[int] = None
            for pid in env.get_state().placed:
                p = env.get_state().placements[pid]
                px0 = int(getattr(p, "min_x"))
                py0 = int(getattr(p, "min_y"))
                px1 = int(getattr(p, "max_x"))
                py1 = int(getattr(p, "max_y"))
                if py1 <= y0 or py0 >= y1:
                    continue
                if direction > 0 and px0 >= x_bl:
                    cand = int(px1)
                    if best is None or cand < best:
                        best = cand
                if direction < 0 and px1 <= x_bl:
                    cand = int(px0) - int(w)
                    if best is None or cand > best:
                        best = cand
            return best

        results: List[Tuple[GroupId, int, int, int]] = []
        for rotation in rotations:
            w, h = self._wh_int(env, gid, rotation)
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = 0
            y_bl = 0
            for _ in range(max_scan):
                x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                results.append((gid, int(x_bl), int(y_bl), int(rotation)))

                if x_bl < (int(env.grid_width) - w):
                    jump = _jump_x_bl(x_bl, y_bl, w, h, 1)
                    if jump is None:
                        x_bl += step
                    else:
                        x_bl = max(int(jump), x_bl + step)
                else:
                    y_bl += step
                    if y_bl > (int(env.grid_height) - h):
                        break
                    x_bl = 0

                if len(results) > count * 10:
                    break
        return results

    def _source_near(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if not env.get_state().placed:
            return []
        group = env.group_specs[gid]
        rotations = (0, 90, 180, 270) if group.rotatable else (0,)
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for rotation in rotations:
            w, h = self._wh_int(env, gid, rotation)
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for pid in env.get_state().placed:
                p = env.get_state().placements[pid]
                px0 = int(getattr(p, "min_x"))
                py0 = int(getattr(p, "min_y"))
                px1 = int(getattr(p, "max_x"))
                py1 = int(getattr(p, "max_y"))
                x_events = [int(px0) - w, int(px0), int(px1) - w, int(px1)]
                y_events = [int(py0) - h, int(py0), int(py1) - h, int(py1)]
                for x_bl in x_events:
                    for y_bl in y_events:
                        x2, y2 = self._clamp_bl(env, int(x_bl), int(y_bl), w, h)
                        candidates.append((gid, int(x2), int(y2), int(rotation)))
        return candidates

    def _source_coarse(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        step = max(int(round(float(self.scan_step * 3))), 1)
        w, h = self._wh_int(env, gid, 0)
        xs = self._scan_axis_bl(int(env.grid_width), w, step)
        ys = self._scan_axis_bl(int(env.grid_height), h, step)
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for x_bl in xs:
            for y_bl in ys:
                candidates.append((gid, int(x_bl), int(y_bl), 0))
        return candidates

    def _source_random(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        group = env.group_specs[gid]
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for _ in range(count):
            rotation = 0 if not group.rotatable else self._rng.choice([0, 90, 180, 270])
            w, h = self._wh_int(env, gid, rotation)
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_width) - w))))
            y_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_height) - h))))
            x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
            candidates.append((gid, int(x_bl), int(y_bl), int(rotation)))
        return candidates

    def _generate_initial(
        self, env: FactoryLayoutEnv, gid: GroupId, quant_step: float
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand))

        group = env.group_specs[gid]
        rotations = (0, 90, 180, 270) if getattr(group, "rotatable", False) else (0,)
        valid_by_rotation = {r: self._build_rotation_valid_map(env, gid=gid, rotation=r) for r in rotations}
        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates: List[Tuple[GroupId, int, int, int]] = []
        if unique_tagged:
            xyrot = torch.tensor(
                [[int(c[1]), int(c[2]), int(c[3])] for _, c in unique_tagged],
                dtype=torch.long,
                device=device,
            )
            placeable = self._validate_with_maps(xyrot, valid_by_rotation)
            for i, (_, c) in enumerate(unique_tagged):
                if bool(placeable[i].item()):
                    valid_candidates.append(c)

        final = valid_candidates[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(gid, self.k - len(final)))
        return final, mask

    def _generate(
        self, env: FactoryLayoutEnv, next_group_id: GroupId
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        q = self.quant_step if self.quant_step is not None else self.scan_step

        if len(env.get_state().placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.group_specs[next_group_id]
        rotations = (0, 90, 180, 270) if getattr(group, "rotatable", False) else (0,)
        valid_by_rotation = {r: self._build_rotation_valid_map(env, gid=next_group_id, rotation=r) for r in rotations}

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor))

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        valid_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        if unique_tagged:
            xyrot = torch.tensor(
                [[int(c[1]), int(c[2]), int(c[3])] for _, c in unique_tagged],
                dtype=torch.long,
                device=device,
            )
            placeable = self._validate_with_maps(xyrot, valid_by_rotation)
            for i, tagged in enumerate(unique_tagged):
                if bool(placeable[i].item()):
                    valid_tagged.append(tagged)

        pools: Dict[int, List[Tuple[GroupId, int, int, int]]] = {0: [], 1: [], 2: [], 3: []}
        for src, c in valid_tagged:
            pools[src].append(c)

        final: List[Tuple[GroupId, int, int, int]] = []
        final.extend(self._score_sorted(env, next_group_id, pools[0])[:n_high])
        final.extend(self._score_sorted(env, next_group_id, pools[1])[:n_near])
        final.extend(self._random_take(pools[2], n_coarse))
        final.extend(self._random_take(pools[3], n_rand))

        final = final[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(next_group_id, self.k - len(final)))
        return final, mask


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

    adapter = GreedyAdapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)

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

    print("GreedyAdapter demo")
    print(" env=", ENV_JSON, "device=", device, "k=", 50)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")
