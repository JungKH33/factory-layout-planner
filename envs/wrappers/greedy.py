from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId

from .base import BaseWrapper


class GreedyWrapperEnv(BaseWrapper):
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
        p_near: float = 0.8,
        p_coarse: float = 0.0,
        oversample_factor: int = 2,
        diversity_ratio: float = 0.0,  # parity (unused)
        min_diversity: int = 0,  # parity (unused)
        random_seed: Optional[int] = None,
    ):
        super().__init__(engine=engine)
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

    def _source_stratified(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)

        count_per_rot = max(1, count // len(rotations))
        aspect_ratio = float(env.grid_width) / float(env.grid_height)
        nx = max(1, round(math.sqrt(count_per_rot * aspect_ratio)))
        ny = max(1, round(count_per_rot / nx))

        dx = float(env.grid_width) / float(nx)
        dy = float(env.grid_height) / float(ny)

        candidates: List[Tuple[GroupId, int, int, int]] = []
        for rot in rotations:
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for i in range(nx):
                for j in range(ny):
                    x_bl = int(round(i * dx))
                    y_bl = int(round(j * dy))
                    x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                    candidates.append((gid, int(x_bl), int(y_bl), int(rot)))
        return candidates

    def _source_high(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        step = max(int(round(self.scan_step)), 1)
        max_scan = 50000

        def _jump_x_bl(x_bl: int, y_bl: int, w: int, h: int, direction: int) -> Optional[int]:
            y0 = int(y_bl)
            y1 = int(y_bl) + int(h)
            best: Optional[int] = None
            for pid in env.placed:
                px0, py0, px1, py1 = env.placed_body_rect_bl(pid)
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
        for rot in rotations:
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = 0
            y_bl = 0
            for _ in range(max_scan):
                x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                results.append((gid, int(x_bl), int(y_bl), int(rot)))

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
        if not env.placed:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for rot in rotations:
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for pid in env.placed:
                px0, py0, px1, py1 = env.placed_body_rect_bl(pid)
                x_events = [int(px0) - w, int(px0), int(px1) - w, int(px1)]
                y_events = [int(py0) - h, int(py0), int(py1) - h, int(py1)]
                for x_bl in x_events:
                    for y_bl in y_events:
                        x2, y2 = self._clamp_bl(env, int(x_bl), int(y_bl), w, h)
                        candidates.append((gid, int(x2), int(y2), int(rot)))
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
        group = env.groups[gid]
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for _ in range(count):
            rot = 0 if not group.rotatable else self._rng.choice([0, 90])
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_width) - w))))
            y_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_height) - h))))
            x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
            candidates.append((gid, int(x_bl), int(y_bl), int(rot)))
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

        group = env.groups[gid]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, gid, int(r)) for r in rotations}
        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [
            c
            for _, c in unique_tagged
            if (not self._cheap_reject_body(env, gid=gid, x_bl=int(c[1]), y_bl=int(c[2]), rot=int(c[3]), wh_cache=wh_cache))
            and env.is_placeable(gid, float(c[1]), float(c[2]), int(c[3]))
        ]

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

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.groups[next_group_id]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, next_group_id, int(r)) for r in rotations}

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor))

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
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(next_group_id, self.k - len(final)))
        return final, mask


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

