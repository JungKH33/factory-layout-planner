from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId
from ...base import BaseAdapter


class GreedyV2Adapter(BaseAdapter):
    """Top-K candidate wrapper: Discrete(K) actions over an in-file TopK generator.

    Notes:
    - Candidate coordinates are bottom-left integer coordinates (engine contract).
    - `action_mask` is torch.BoolTensor[K] (True means valid).
    - `action_poses` is torch.FloatTensor[K,2] of (x_c, y_c) center coordinates.
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
        p_near: float = 0.1,
        p_coarse: float = 0.0,
        oversample_factor: int = 2,
        diversity_ratio: float = 0.0,  # parity (unused)
        min_diversity: int = 0,  # parity (unused)
        random_seed: Optional[int] = None,
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
        self.action_delta: Optional[torch.Tensor] = None  # float [K]

    def build_observation(self) -> Dict[str, Any]:
        self.mask = self.create_mask()
        obs: Dict[str, Any] = {}
        if isinstance(self.action_delta, torch.Tensor):
            obs["action_delta"] = self.action_delta
        return obs

    def create_mask(self) -> torch.Tensor:
        # Keep candidate sampling deterministic from engine state.
        self._rng = random.Random(self.action_space_seed())
        gid = self.current_gid()
        if gid is None:
            self.action_poses = torch.zeros((self.k, 2), dtype=torch.float32, device=self.device)
            self.action_delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
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
        delta = torch.full((self.k,), float("inf"), dtype=torch.float32, device=self.device)
        vmask = mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]
        if int(vidx.numel()) > 0:
            vv = poses[vidx]
            d = self._score_poses(gid, vv).to(dtype=torch.float32, device=self.device)
            delta[vidx] = d.view(-1)
        self.action_delta = delta
        return mask

    # ---- state api (for wrapped search/MCTS) ----
    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        snap["rng_state"] = self._rng.getstate()
        if isinstance(self.action_poses, torch.Tensor):
            snap["action_poses"] = self.action_poses.clone()
        else:
            snap["action_poses"] = None
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
        ax = state.get("action_poses", None)
        if isinstance(ax, torch.Tensor):
            self.action_poses = ax.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_poses = None
        ad = state.get("action_delta", None)
        if isinstance(ad, torch.Tensor):
            self.action_delta = ad.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_delta = None

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
            x_c = float(x_bl) + float(w) / 2.0
            y_c = float(y_bl) + float(h) / 2.0
            qx = int(round(x_c / q))
            qy = int(round(y_c / q))
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
        spec = env.group_specs[gid]
        state = env.get_state()
        rr = spec._resolve_rotation(rotation)
        result = None
        seen_shape: set = set()
        for vi in spec._variants:
            if vi.rotation != rr:
                continue
            shape_key = vi.shape_key
            if shape_key in seen_shape:
                continue
            seen_shape.add(shape_key)
            body_map, clearance_map, clearance_origin, is_rectangular = spec.shape_tensors(shape_key)
            m = state.is_placeable_map(
                gid=gid,
                body_map=body_map,
                clearance_map=clearance_map,
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
        rotation: int,
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
            out.append((gid, x_bl, y_bl, int(rotation)))
        return out

    def _source_stratified(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rotation: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or not valid_by_rotation:
            return []

        # Keep previous nx/ny heuristic but sample valid points inside each cell.
        aspect_ratio = float(env.grid_width) / float(env.grid_height)
        nx = max(1, round(math.sqrt(max(1, count) * aspect_ratio)))
        ny = max(1, round(max(1, count) / nx))
        dx = float(env.grid_width) / float(nx)
        dy = float(env.grid_height) / float(ny)

        rots = list(valid_by_rotation.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        results: List[Tuple[GroupId, int, int, int]] = []

        for rot in rots:
            vm = valid_by_rotation[int(rot)]
            H, W = int(vm.shape[0]), int(vm.shape[1])
            if H <= 0 or W <= 0:
                continue
            got = 0
            # We iterate cells in a pseudo-random order to diversify.
            cell_ids = list(range(int(nx * ny)))
            self._rng.shuffle(cell_ids)
            for cid in cell_ids:
                if got >= per_rot:
                    break
                i = int(cid // int(ny))
                j = int(cid % int(ny))
                x0 = int(round(i * dx))
                x1 = int(round((i + 1) * dx))
                y0 = int(round(j * dy))
                y1 = int(round((j + 1) * dy))
                x0 = max(0, min(W, x0))
                x1 = max(0, min(W, x1))
                y0 = max(0, min(H, y0))
                y1 = max(0, min(H, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = vm[y0:y1, x0:x1]
                if not bool(sub.any().item()):
                    continue
                sub_idx = torch.nonzero(sub, as_tuple=False)
                m = int(sub_idx.shape[0])
                # pick one valid point from this cell
                k = int(torch.randint(low=0, high=max(1, m), size=(1,), generator=gen, device=env.device).item())
                y_bl = int(y0 + int(sub_idx[k, 0].item()))
                x_bl = int(x0 + int(sub_idx[k, 1].item()))
                results.append((gid, x_bl, y_bl, int(rot)))
                got += 1

        # If still short, fill from uniform valid sampling.
        if len(results) < count:
            need = int(count) - int(len(results))
            for rot in rots:
                if need <= 0:
                    break
                add = self._sample_from_valid(env=env, gid=gid, rotation=int(rot), valid_map=valid_by_rotation[int(rot)], count=need, gen=gen)
                results.extend(add)
                need = int(count) - int(len(results))

        return results[: int(count)]

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

    def _source_high(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rotation: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        # v2: "high" is generated from valid space directly (then later scored by _score_sorted).
        if count <= 0 or not valid_by_rotation:
            return []
        rots = list(valid_by_rotation.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            out.extend(self._sample_from_valid(env=env, gid=gid, rotation=int(rot), valid_map=valid_by_rotation[int(rot)], count=per_rot, gen=gen))
        return out[: int(count)]

    def _source_near(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rotation: Dict[int, torch.Tensor],
        wh_cache: Dict[int, Tuple[int, int]],
        gen: torch.Generator,
        radius: int,
    ) -> List[Tuple[GroupId, int, int, int]]:
        # v2: generate near candidates by snapping around body-event points into nearby valid points.
        if count <= 0 or (not env.get_state().placed) or (not valid_by_rotation):
            return []
        rots = list(valid_by_rotation.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        placed_ids = list(env.get_state().placed)
        # Keep cost bounded: sample a limited number of placed facilities.
        self._rng.shuffle(placed_ids)
        placed_ids = placed_ids[: max(1, per_rot * 8)]

        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            vm = valid_by_rotation[int(rot)]
            H, W = int(vm.shape[0]), int(vm.shape[1])
            if H <= 0 or W <= 0:
                continue
            w, h = wh_cache[int(rot)]
            w = int(w)
            h = int(h)
            got = 0
            for pid in placed_ids:
                if got >= per_rot:
                    break
                p = env.get_state().placements[pid]
                px0 = int(getattr(p, "min_x"))
                py0 = int(getattr(p, "min_y"))
                px1 = int(getattr(p, "max_x"))
                py1 = int(getattr(p, "max_y"))
                x_events = [int(px0) - w, int(px0), int(px1) - w, int(px1)]
                y_events = [int(py0) - h, int(py0), int(py1) - h, int(py1)]
                self._rng.shuffle(x_events)
                self._rng.shuffle(y_events)
                for x_bl in x_events:
                    if got >= per_rot:
                        break
                    for y_bl in y_events:
                        if got >= per_rot:
                            break
                        x2, y2 = self._clamp_bl(env, int(x_bl), int(y_bl), w, h)
                        # local window search in valid map
                        x0 = max(0, int(x2) - int(radius))
                        x1 = min(W, int(x2) + int(radius) + 1)
                        y0 = max(0, int(y2) - int(radius))
                        y1 = min(H, int(y2) + int(radius) + 1)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        sub = vm[y0:y1, x0:x1]
                        if not bool(sub.any().item()):
                            continue
                        sub_idx = torch.nonzero(sub, as_tuple=False)
                        m = int(sub_idx.shape[0])
                        k = int(torch.randint(low=0, high=max(1, m), size=(1,), generator=gen, device=env.device).item())
                        yy = int(y0 + int(sub_idx[k, 0].item()))
                        xx = int(x0 + int(sub_idx[k, 1].item()))
                        out.append((gid, int(xx), int(yy), int(rot)))
                        got += 1
            # If still short for this rot, fill from uniform valid sampling
            if got < per_rot:
                out.extend(self._sample_from_valid(env=env, gid=gid, rotation=int(rot), valid_map=vm, count=(per_rot - got), gen=gen))

        return out[: int(count)]

    def _source_coarse(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rotation: Dict[int, torch.Tensor],
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or (not valid_by_rotation):
            return []
        step = max(int(round(float(self.scan_step * 3))), 1)
        out: List[Tuple[GroupId, int, int, int]] = []
        # coarse is always rot=0 in the original, keep parity
        rot = 0 if 0 in valid_by_rotation else list(valid_by_rotation.keys())[0]
        vm = valid_by_rotation[int(rot)]
        H, W = int(vm.shape[0]), int(vm.shape[1])
        if H <= 0 or W <= 0:
            return []
        for x_bl in range(0, W, step):
            for y_bl in range(0, H, step):
                if bool(vm[int(y_bl), int(x_bl)].item()):
                    out.append((gid, int(x_bl), int(y_bl), int(rot)))
                if len(out) >= int(count):
                    return out
        return out[: int(count)]

    def _source_random(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rotation: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or (not valid_by_rotation):
            return []
        rots = list(valid_by_rotation.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            out.extend(self._sample_from_valid(env=env, gid=gid, rotation=int(rot), valid_map=valid_by_rotation[int(rot)], count=per_rot, gen=gen))
        return out[: int(count)]

    def _generate_initial(
        self, env: FactoryLayoutEnv, gid: GroupId, quant_step: float
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        group = env.group_specs[gid]
        rotations = (0, 90, 180, 270) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {r: self._wh_int(env, gid, r) for r in rotations}
        valid_by_rotation = {r: self._build_rotation_valid_map(env, gid=gid, rotation=r) for r in rotations}
        gen = self._torch_gen(env=env)

        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target, valid_by_rotation=valid_by_rotation, gen=gen))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand, valid_by_rotation=valid_by_rotation, gen=gen))

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
        wh_cache = {r: self._wh_int(env, next_group_id, r) for r in rotations}
        valid_by_rotation = {r: self._build_rotation_valid_map(env, gid=next_group_id, rotation=r) for r in rotations}
        gen = self._torch_gen(env=env)
        radius = max(1, int(round(q)))

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend(
            (0, c)
            for c in self._source_high(env, next_group_id, n_high * self.oversample_factor, valid_by_rotation=valid_by_rotation, gen=gen)
        )
        raw_tagged.extend(
            (1, c)
            for c in self._source_near(
                env,
                next_group_id,
                n_near * self.oversample_factor,
                valid_by_rotation=valid_by_rotation,
                wh_cache=wh_cache,
                gen=gen,
                radius=radius,
            )
        )
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor, valid_by_rotation=valid_by_rotation))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor, valid_by_rotation=valid_by_rotation, gen=gen))

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

    from envs.action_space import ActionSpace as CandidateSet
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.env_visualizer import plot_layout

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = GreedyV2Adapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)

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

    print("GreedyAdapter demo")
    print(" env=", ENV_JSON, "device=", device, "k=", 50)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")
