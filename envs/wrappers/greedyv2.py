from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId

from .base import BaseWrapper


class GreedyWrapperV2Env(BaseWrapper):
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
        engine: FactoryLayoutEnv,
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
        optimize_rotation: bool = False,
    ):
        super().__init__(engine=engine)
        self.k = int(k)
        self.scan_step = float(scan_step)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.p_high = float(p_high)
        self.p_near = float(p_near)
        self.p_coarse = float(p_coarse)
        self.optimize_rotation = optimize_rotation
        self.oversample_factor = int(oversample_factor)
        self.diversity_ratio = float(diversity_ratio)
        self.min_diversity = int(min_diversity)
        self._rng = random.Random(random_seed)

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        self.action_xyrot: Optional[torch.Tensor] = None  # long [K,3]

    def create_mask(self) -> torch.Tensor:
        if not self.engine.remaining:
            self.action_xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
            return torch.zeros((self.k,), dtype=torch.bool, device=self.device)

        gid = self.engine.remaining[0]
        candidates, mask = self._generate(self.engine, gid)

        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
        for i, c in enumerate(candidates[: self.k]):
            _, x_bl, y_bl, rot = c
            xyrot[i, 0] = int(x_bl)
            xyrot[i, 1] = int(y_bl)
            xyrot[i, 2] = int(rot)
        self.action_xyrot = xyrot
        return mask

    def _build_obs(self) -> Dict[str, Any]:
        assert self.mask is not None
        assert self.action_xyrot is not None
        obs = dict(self.engine._build_obs())
        obs["action_mask"] = self.mask
        obs["action_xyrot"] = self.action_xyrot
        return obs

    def decode_action(self, action: int) -> Tuple[float, float, int, int, int]:
        a = int(action)
        if self.action_xyrot is None or a < 0 or a >= self.k:
            return 0.0, 0.0, 0, 0, 0
        xyz = self.action_xyrot[a]
        return float(xyz[0].item()), float(xyz[1].item()), int(xyz[2].item()), 0, a

    def step(self, action: int):
        assert self.mask is not None
        x, y, rot, _i, cand_idx = self.decode_action(int(action))
        obs_core, reward, terminated, truncated, info = self.engine.step_masked(
            action=int(action),
            x=float(x),
            y=float(y),
            rot=int(rot),
            mask=self.mask,
            action_space_n=int(self.k),
            extra_info={"cand_idx": int(cand_idx)},
        )
        if not (terminated or truncated):
            self.mask = self.create_mask()
            return self._build_obs(), reward, terminated, truncated, info
        return obs_core, reward, terminated, truncated, info

    # ---- snapshot api (for wrapped search/MCTS) ----
    def get_snapshot(self) -> Dict[str, object]:
        snap = dict(super().get_snapshot())
        snap["rng_state"] = self._rng.getstate()
        if isinstance(self.action_xyrot, torch.Tensor):
            snap["action_xyrot"] = self.action_xyrot.clone()
        else:
            snap["action_xyrot"] = None
        return snap

    def set_snapshot(self, snapshot: Dict[str, object]) -> None:
        super().set_snapshot(snapshot)
        rs = snapshot.get("rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        ax = snapshot.get("action_xyrot", None)
        if isinstance(ax, torch.Tensor):
            self.action_xyrot = ax.to(device=self.device, dtype=torch.long).clone()
        else:
            self.action_xyrot = None

    # ---- candidate generation (copied from actionspace/topk.py; BL int coords) ----
    def _quota(self, k: int) -> Tuple[int, int, int, int]:
        n_high = round(k * self.p_high)
        n_near = round(k * self.p_near)
        n_coarse = round(k * self.p_coarse)
        n_rand = k - (n_high + n_near + n_coarse)
        if n_rand < 0:
            n_rand = 0
        return int(n_high), int(n_near), int(n_coarse), int(n_rand)

    def _wh_int(self, env: FactoryLayoutEnv, gid: GroupId, rot: int) -> Tuple[int, int]:
        g = env.groups[gid]
        w, h = env.rotated_size(g, int(rot))
        return int(round(float(w))), int(round(float(h)))

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
    ) -> bool:
        w, h = wh_cache[int(rot)]
        gw = int(env.grid_width)
        gh = int(env.grid_height)

        if x_bl < 0 or y_bl < 0 or (x_bl + w) > gw or (y_bl + h) > gh:
            return True
        if w <= 0 or h <= 0:
            return True

        invalid = env._invalid  # torch.BoolTensor[H,W]
        clear_invalid = env._clear_invalid  # torch.BoolTensor[H,W]

        x0 = int(x_bl)
        y0 = int(y_bl)
        x1 = x0 + int(w) - 1
        y1 = y0 + int(h) - 1
        xc = (x0 + x1) // 2
        yc = (y0 + y1) // 2

        pts = ((x0, y0), (x1, y0), (x0, y1), (x1, y1), (xc, yc))
        for px, py in pts:
            if bool(invalid[py, px].item()):
                return True
            if bool(clear_invalid[py, px].item()):
                return True
        return False

    # ---- v2: valid-map based sampling (clearance-aware at generation time) ----
    def _build_valid_map(self, env: FactoryLayoutEnv, *, gid: GroupId, rot: int, wh_cache: Dict[int, Tuple[int, int]]) -> torch.Tensor:
        """Return bool[H2,W2] where True means this (x_bl,y_bl,rot) is *likely* placeable.

        This is a clearance-aware prefilter:
        - body window must not overlap env._invalid or env._clear_invalid
        - my clearance pad window must not overlap env._invalid

        NOTE: This is intentionally a *prefilter*. We still run env.is_placeable(...) as the final gate.
        """
        w, h = wh_cache[int(rot)]
        w = max(1, int(w))
        h = max(1, int(h))

        inv = env._invalid.to(dtype=torch.float32).view(1, 1, int(env.grid_height), int(env.grid_width))
        clr = env._clear_invalid.to(dtype=torch.float32).view(1, 1, int(env.grid_height), int(env.grid_width))

        # body window must avoid invalid + clear_invalid
        k_body = torch.ones((1, 1, int(h), int(w)), device=env.device, dtype=inv.dtype)
        ov_inv = F.conv2d(inv, k_body, padding=0).squeeze(0).squeeze(0)
        ov_clr = F.conv2d(clr, k_body, padding=0).squeeze(0).squeeze(0)
        body_ok = (ov_inv == 0) & (ov_clr == 0)  # bool[H2,W2]

        # pad window (my clearance) must avoid invalid
        group = env.groups[gid]
        cL, cR, cB, cT = env._clearance_lrtb(group, int(rot))
        cL_i, cR_i, cB_i, cT_i = int(cL), int(cR), int(cB), int(cT)
        kw = max(1, int(w) + cL_i + cR_i)
        kh = max(1, int(h) + cB_i + cT_i)
        k_pad = torch.ones((1, 1, int(kh), int(kw)), device=env.device, dtype=inv.dtype)
        ov_pad = F.conv2d(inv, k_pad, padding=0).squeeze(0).squeeze(0)
        pad_ok = (ov_pad == 0)  # bool[H3,W3], index is (x_bl - cL, y_bl - cB)

        H2, W2 = int(body_ok.shape[0]), int(body_ok.shape[1])
        H3, W3 = int(pad_ok.shape[0]), int(pad_ok.shape[1])

        # Align pad_ok into body_ok coordinates: body position (y,x) uses pad_ok[y-cB, x-cL]
        pad_aligned = torch.zeros((H2, W2), dtype=torch.bool, device=env.device)
        y0 = max(0, cB_i)
        x0 = max(0, cL_i)
        y1 = min(H2, cB_i + H3)
        x1 = min(W2, cL_i + W3)
        if y1 > y0 and x1 > x0:
            pad_aligned[y0:y1, x0:x1] = pad_ok[0 : (y1 - y0), 0 : (x1 - x0)]

        return body_ok & pad_aligned

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

    def _source_stratified(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rot: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or not valid_by_rot:
            return []

        # Keep previous nx/ny heuristic but sample valid points inside each cell.
        aspect_ratio = float(env.grid_width) / float(env.grid_height)
        nx = max(1, round(math.sqrt(max(1, count) * aspect_ratio)))
        ny = max(1, round(max(1, count) / nx))
        dx = float(env.grid_width) / float(nx)
        dy = float(env.grid_height) / float(ny)

        rots = list(valid_by_rot.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        results: List[Tuple[GroupId, int, int, int]] = []

        for rot in rots:
            vm = valid_by_rot[int(rot)]
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
                add = self._sample_from_valid(env=env, gid=gid, rot=int(rot), valid_map=valid_by_rot[int(rot)], count=need, gen=gen)
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
        xy = torch.tensor([[c[1], c[2], c[3]] for c in pool], dtype=torch.long, device=env.device)  # [M,3]
        scores_t = env.delta_cost(gid=gid, x=xy[:, 0], y=xy[:, 1], rot=xy[:, 2])
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
        valid_by_rot: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        # v2: "high" is generated from valid space directly (then later scored by _score_sorted).
        if count <= 0 or not valid_by_rot:
            return []
        rots = list(valid_by_rot.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            out.extend(self._sample_from_valid(env=env, gid=gid, rot=int(rot), valid_map=valid_by_rot[int(rot)], count=per_rot, gen=gen))
        return out[: int(count)]

    def _source_near(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rot: Dict[int, torch.Tensor],
        wh_cache: Dict[int, Tuple[int, int]],
        gen: torch.Generator,
        radius: int,
    ) -> List[Tuple[GroupId, int, int, int]]:
        # v2: generate near candidates by snapping around body-event points into nearby valid points.
        if count <= 0 or (not env.placed) or (not valid_by_rot):
            return []
        rots = list(valid_by_rot.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        placed_ids = list(env.placed)
        # Keep cost bounded: sample a limited number of placed facilities.
        self._rng.shuffle(placed_ids)
        placed_ids = placed_ids[: max(1, per_rot * 8)]

        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            vm = valid_by_rot[int(rot)]
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
                px0, py0, px1, py1 = env.placed_body_rect_bl(pid)
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
                out.extend(self._sample_from_valid(env=env, gid=gid, rot=int(rot), valid_map=vm, count=(per_rot - got), gen=gen))

        return out[: int(count)]

    def _source_coarse(
        self,
        env: FactoryLayoutEnv,
        gid: GroupId,
        count: int,
        *,
        valid_by_rot: Dict[int, torch.Tensor],
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or (not valid_by_rot):
            return []
        step = max(int(round(float(self.scan_step * 3))), 1)
        out: List[Tuple[GroupId, int, int, int]] = []
        # coarse is always rot=0 in the original, keep parity
        rot = 0 if 0 in valid_by_rot else list(valid_by_rot.keys())[0]
        vm = valid_by_rot[int(rot)]
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
        valid_by_rot: Dict[int, torch.Tensor],
        gen: torch.Generator,
    ) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or (not valid_by_rot):
            return []
        rots = list(valid_by_rot.keys())
        per_rot = max(1, int(round(float(count) / float(max(1, len(rots))))))
        out: List[Tuple[GroupId, int, int, int]] = []
        for rot in rots:
            out.extend(self._sample_from_valid(env=env, gid=gid, rot=int(rot), valid_map=valid_by_rot[int(rot)], count=per_rot, gen=gen))
        return out[: int(count)]

    def _generate_initial(
        self, env: FactoryLayoutEnv, gid: GroupId, quant_step: float
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        group = env.groups[gid]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, gid, int(r)) for r in rotations}
        valid_by_rot = {int(r): self._build_valid_map(env, gid=gid, rot=int(r), wh_cache=wh_cache) for r in rotations}
        gen = self._torch_gen(env=env)

        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target, valid_by_rot=valid_by_rot, gen=gen))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand, valid_by_rot=valid_by_rot, gen=gen))

        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [
            c
            for _, c in unique_tagged
            if (not self._cheap_reject_body(env, gid=gid, x_bl=int(c[1]), y_bl=int(c[2]), rot=int(c[3]), wh_cache=wh_cache))
            and env.is_placeable(gid, float(c[1]), float(c[2]), int(c[3]))
        ]

        final = valid_candidates[: self.k]
        if self.optimize_rotation:
            final = self._optimize_rotation(env, gid, final)
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

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.groups[next_group_id]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, next_group_id, int(r)) for r in rotations}
        valid_by_rot = {int(r): self._build_valid_map(env, gid=next_group_id, rot=int(r), wh_cache=wh_cache) for r in rotations}
        gen = self._torch_gen(env=env)
        radius = max(1, int(round(q)))

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend(
            (0, c)
            for c in self._source_high(env, next_group_id, n_high * self.oversample_factor, valid_by_rot=valid_by_rot, gen=gen)
        )
        raw_tagged.extend(
            (1, c)
            for c in self._source_near(
                env,
                next_group_id,
                n_near * self.oversample_factor,
                valid_by_rot=valid_by_rot,
                wh_cache=wh_cache,
                gen=gen,
                radius=radius,
            )
        )
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor, valid_by_rot=valid_by_rot))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor, valid_by_rot=valid_by_rot, gen=gen))

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        valid_tagged = [
            (src, c)
            for src, c in unique_tagged
            if (not self._cheap_reject_body(env, gid=next_group_id, x_bl=int(c[1]), y_bl=int(c[2]), rot=int(c[3]), wh_cache=wh_cache))
            and env.is_placeable(next_group_id, float(c[1]), float(c[2]), int(c[3]))
        ]

        pools: Dict[int, List[Tuple[GroupId, int, int, int]]] = {0: [], 1: [], 2: [], 3: []}
        for src, c in valid_tagged:
            pools[src].append(c)

        final: List[Tuple[GroupId, int, int, int]] = []
        final.extend(self._score_sorted(env, next_group_id, pools[0])[:n_high])
        final.extend(self._score_sorted(env, next_group_id, pools[1])[:n_near])
        final.extend(self._random_take(pools[2], n_coarse))
        final.extend(self._random_take(pools[3], n_rand))

        final = final[: self.k]
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

    from envs.wrappers.candidate_set import CandidateSet
    from envs.json_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    env = GreedyWrapperEnv(engine=engine, k=50, scan_step=10.0, quant_step=10.0, random_seed=0)

    t0 = time.perf_counter()
    obs, _info = env.reset(options=loaded.reset_kwargs)
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(obs["action_mask"].sum().item())
    a = int(torch.where(obs["action_mask"])[0][0].item()) if valid > 0 else 0

    # Plot: initial candidates (interactive; close to continue)
    next_gid = env.engine.remaining[0] if env.engine.remaining else None
    cand0 = CandidateSet(xyrot=obs["action_xyrot"], mask=obs["action_mask"], gid=next_gid, meta={"k": int(env.k)})
    plot_layout(env, candidate_set=cand0)

    t1 = time.perf_counter()
    obs2, _r, _term, _trunc, _info2 = env.step(a)
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot: after 1 placement + new candidates (if any)
    if isinstance(obs2, dict) and ("action_mask" in obs2) and ("action_xyrot" in obs2):
        next_gid2 = env.engine.remaining[0] if env.engine.remaining else None
        cand1 = CandidateSet(xyrot=obs2["action_xyrot"], mask=obs2["action_mask"], gid=next_gid2, meta={"k": int(env.k)})
        plot_layout(env, candidate_set=cand1)
    else:
        plot_layout(env, candidate_set=None)

    print("[GreedyWrapperEnv demo]")
    print(" env=", ENV_JSON, "device=", device, "k=", 50)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")

