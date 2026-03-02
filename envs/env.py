from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch

from .core.base import PlacementBase
from .core.maps import GridMaps
from .core.reward import AreaReward, CandidateBatch, FlowReward, RewardComposer, RewardContext
from .core.static import StaticSpec


GroupId = Union[int, str]

# PlacementBase is defined in envs/base.py and imported above.
# Re-exported here so external code can do: from envs.env import PlacementBase, GridMaps
__all__ = ["PlacementBase", "GridMaps", "FactoryLayoutEnv"]


class FactoryLayoutEnv(gym.Env):
    """Tensor-first Gymnasium env for factory layout placement.

    Design notes:
    - The engine owns placement feasibility, constraints, and objective evaluation.
    - Action semantics / candidate generation are owned by wrapper envs.
    - Observations are torch.Tensors (GPU-friendly; TorchRL-compatible).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_width: int,
        grid_height: int,
        group_specs: Dict[GroupId, StaticSpec],
        group_flow: Optional[Dict[GroupId, Dict[GroupId, float]]] = None,
        # Forbidden areas: [{"rect": [x0, y0, x1, y1]}, ...]
        forbidden_areas: Optional[List[Dict[str, Any]]] = None,
        # --- zone/constraint configs (optional; fully map-based) ---
        # For each constraint, we define a per-cell float map:
        # - map is initialized from env.default_*
        # - areas override map values on their rects
        # Constraint invalidation is map comparison (same shape for all three):
        # - Weight/Height: invalid if map < facility_value
        # - Dry (reverse): invalid if map > facility_value
        default_weight: float = float("inf"),
        default_height: float = float("inf"),
        default_dry: float = -float("inf"),
        weight_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[x0,y0,x1,y1], "value": float}, ...]
        height_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[...], "value": float}, ...]
        dry_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[...], "value": float}, ...]
        placement_areas: Optional[List[Dict[str, Any]]] = None,  # [{"id": str, "rect":[x0,y0,x1,y1]}, ...]
        device: Optional[torch.device] = None,
        max_steps: Optional[int] = None,
        cost_scale: float = 100.0,
        penalty_weight: float = 50000.0,
        log: bool = False,
    ):
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.group_specs = self._normalize_group_specs(group_specs)
        self.group_flow = group_flow or {}
        self.forbidden_areas = list(forbidden_areas or [])
        self.default_weight = float(default_weight)
        self.default_height = float(default_height)
        self.default_dry = float(default_dry)
        self.weight_areas = list(weight_areas or [])
        self.height_areas = list(height_areas or [])
        self.dry_areas = list(dry_areas or [])
        self.placement_areas = list(placement_areas or [])
        self.max_steps = max_steps
        self.cost_scale = float(cost_scale)
        self.penalty_weight = float(penalty_weight)
        self.log = bool(log)

        # Engine does not own action semantics. Wrappers define action_space/obs additions.
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict({})

        # --- group id/index mapping (useful across wrappers) ---
        self.node_ids: List[GroupId] = sorted(self.group_specs.keys(), key=lambda x: str(x))
        self.gid_to_idx: Dict[GroupId, int] = {gid: i for i, gid in enumerate(self.node_ids)}

        # Placement objects are the single source of truth for placed state.
        self.placements: Dict[GroupId, PlacementBase] = {}
        self.placed: set[GroupId] = set()
        self.remaining: List[GroupId] = []
        self._step_count = 0
        # Per flow-edge argmin port pair cache: (src_gid, dst_gid) -> ((exit_x, exit_y), (entry_x, entry_y))
        # Populated after each placement by _rebuild_flow_port_pairs().
        self._flow_port_pairs: Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        # Grid map tensors (performance cache).
        self._maps = GridMaps(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            device=self.device,
            forbidden_areas=self.forbidden_areas,
            default_weight=self.default_weight,
            weight_areas=self.weight_areas,
            default_height=self.default_height,
            height_areas=self.height_areas,
            default_dry=self.default_dry,
            dry_areas=self.dry_areas,
            placement_areas=self.placement_areas,
        )

        # Single RewardComposer used for both score() (cost) and delta() (delta_cost).
        self._reward = RewardComposer(
            components={
                "flow": FlowReward(),
                "area": AreaReward(),
            },
            weights={
                "flow": 1.0,
                "area": 1.0,
            },
        )

    def _normalize_group_specs(self, group_specs: Dict[GroupId, StaticSpec]) -> Dict[GroupId, StaticSpec]:
        """Normalize group specs onto env device and validate id/key consistency."""
        out: Dict[GroupId, StaticSpec] = {}
        for gid, s in dict(group_specs).items():
            if not isinstance(s, StaticSpec):
                raise TypeError(f"group_specs[{gid!r}] must be StaticSpec, got {type(s).__name__}")
            sid = getattr(s, "id", gid)
            if sid != gid:
                raise ValueError(f"group_specs key/id mismatch: key={gid!r}, spec.id={sid!r}")
            out[gid] = StaticSpec(
                device=self.device,
                id=gid,
                width=int(s.width),
                height=int(s.height),
                entries_rel=[(float(p[0]), float(p[1])) for p in list(s.entries_rel)],
                exits_rel=[(float(p[0]), float(p[1])) for p in list(s.exits_rel)],
                clearance_left_rel=int(s.clearance_left_rel),
                clearance_right_rel=int(s.clearance_right_rel),
                clearance_bottom_rel=int(s.clearance_bottom_rel),
                clearance_top_rel=int(s.clearance_top_rel),
                rotatable=bool(s.rotatable),
                allowed_areas=list(s.allowed_areas) if s.allowed_areas is not None else None,
                facility_height=float(s.facility_height),
                facility_weight=float(s.facility_weight),
                facility_dry=float(s.facility_dry),
            )
        return out

    def _group_spec(self, gid: GroupId) -> StaticSpec:
        geom = self.group_specs.get(gid, None)
        if geom is None:
            raise KeyError(f"unknown gid={gid!r}")
        return geom

    def _as_long_tensor(self, v: object, *, name: str) -> torch.Tensor:
        """Coerce scalar/tensor values to integer tensor [N] with integer-value validation."""
        if torch.is_tensor(v):
            t = v.to(device=self.device).view(-1)
            if t.dtype == torch.bool:
                raise ValueError(f"{name} must be int-like tensor, got bool")
            if t.is_floating_point():
                r = torch.round(t)
                if not bool(torch.all(t == r).item()):
                    raise ValueError(f"{name} must contain integer-valued floats")
                t = r
            return t.to(dtype=torch.long).view(-1)
        return torch.tensor([int(v)], dtype=torch.long, device=self.device)

    def _ensure_placement(self, gid: GroupId) -> object:
        p = self.placements.get(gid, None)
        if p is not None:
            return p
        raise KeyError(f"placement missing for gid={gid!r}")

    def _placed_io_tensors(
        self,
    ) -> Tuple[List[GroupId], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        placed_nodes = list(self.placed)
        p = len(placed_nodes)
        if p == 0:
            empty_xy = torch.empty((0, 1, 2), dtype=torch.float32, device=self.device)
            empty_mask = torch.zeros((0, 1), dtype=torch.bool, device=self.device)
            return placed_nodes, empty_xy, empty_xy, empty_mask, empty_mask

        # Collect all ports per facility (fallback to center if none)
        all_entries: List[List[Tuple[float, float]]] = []
        all_exits: List[List[Tuple[float, float]]] = []
        for pgid in placed_nodes:
            placement = self._ensure_placement(pgid)
            cx = float(getattr(placement, "cx", 0.0))
            cy = float(getattr(placement, "cy", 0.0))

            def _to_pairs(src: Any, fallback: Tuple[float, float]) -> List[Tuple[float, float]]:
                if torch.is_tensor(src):
                    t = src.to(dtype=torch.float32).view(-1, 2)
                    return [(float(t[k, 0]), float(t[k, 1])) for k in range(int(t.shape[0]))] or [fallback]
                if isinstance(src, (list, tuple)) and len(src) > 0:
                    return [(float(pt[0]), float(pt[1])) for pt in src]
                return [fallback]

            all_entries.append(_to_pairs(getattr(placement, "entries", []), (cx, cy)))
            all_exits.append(_to_pairs(getattr(placement, "exits", []), (cx, cy)))

        max_e = max(len(e) for e in all_entries)
        max_x = max(len(x) for x in all_exits)

        entries_t = torch.zeros((p, max_e, 2), dtype=torch.float32, device=self.device)
        exits_t = torch.zeros((p, max_x, 2), dtype=torch.float32, device=self.device)
        entries_mask = torch.zeros((p, max_e), dtype=torch.bool, device=self.device)
        exits_mask = torch.zeros((p, max_x), dtype=torch.bool, device=self.device)

        for i, ports in enumerate(all_entries):
            for j, (x, y) in enumerate(ports):
                entries_t[i, j, 0] = x
                entries_t[i, j, 1] = y
                entries_mask[i, j] = True

        for i, ports in enumerate(all_exits):
            for j, (x, y) in enumerate(ports):
                exits_t[i, j, 0] = x
                exits_t[i, j, 1] = y
                exits_mask[i, j] = True

        return placed_nodes, entries_t, exits_t, entries_mask, exits_mask

    def _placed_bbox(self, placed_nodes: List[GroupId]) -> Tuple[float, float, float, float]:
        """Return (min_x, max_x, min_y, max_y) over all placed facilities (body bbox, no clearance)."""
        if not placed_nodes:
            return 0.0, 0.0, 0.0, 0.0
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        for pgid in placed_nodes:
            pl = self._ensure_placement(pgid)
            min_x = min(min_x, float(getattr(pl, "min_x")))
            max_x = max(max_x, float(getattr(pl, "max_x")))
            min_y = min(min_y, float(getattr(pl, "min_y")))
            max_y = max(max_y, float(getattr(pl, "max_y")))
        return min_x, max_x, min_y, max_y

    def _build_flow_w(self, placed_nodes: List[GroupId]) -> torch.Tensor:
        """Build [P, P] flow weight matrix from group_flow for the given placed_nodes order."""
        p = len(placed_nodes)
        flow_w = torch.zeros((p, p), dtype=torch.float32, device=self.device)
        for i, src in enumerate(placed_nodes):
            out_edges = self.group_flow.get(src, {})
            for j, dst in enumerate(placed_nodes):
                flow_w[i, j] = float(out_edges.get(dst, 0.0))
        return flow_w

    def _rebuild_flow_port_pairs(self) -> None:
        """FlowReward.score(return_argmin=True)로 flow edge별 최적 port pair 캐시 재구성.

        self._flow_port_pairs[(src_gid, dst_gid)] = ((exit_x, exit_y), (entry_x, entry_y))
        flow weight가 0인 엣지는 포함하지 않음.
        """
        flow_comp = self._reward.components.get("flow")
        if flow_comp is None:
            return
        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = (
            self._placed_io_tensors()
        )
        p = len(placed_nodes)
        if p == 0:
            self._flow_port_pairs = {}
            return
        flow_w = self._build_flow_w(placed_nodes)
        _, c_idx, p_idx = flow_comp.score(
            placed_entries=placed_entries,
            placed_exits=placed_exits,
            placed_entries_mask=placed_entries_mask,
            placed_exits_mask=placed_exits_mask,
            flow_w=flow_w,
            return_argmin=True,
        )
        if c_idx is None or p_idx is None:
            self._flow_port_pairs = {}
            return
        pairs: Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        for m, src_gid in enumerate(placed_nodes):
            for t, dst_gid in enumerate(placed_nodes):
                if float(flow_w[m, t].item()) <= 0.0:
                    continue
                ci = int(c_idx[m, t].item())
                pi = int(p_idx[m, t].item())
                exit_xy = (float(placed_exits[m, ci, 0].item()), float(placed_exits[m, ci, 1].item()))
                entry_xy = (float(placed_entries[t, pi, 0].item()), float(placed_entries[t, pi, 1].item()))
                pairs[(src_gid, dst_gid)] = (exit_xy, entry_xy)
        self._flow_port_pairs = pairs

    def get_placeable_mask(self, gid: GroupId, rot: int = 0) -> torch.Tensor:
        """전체 그리드의 배치 가능 마스크 계산."""
        if gid not in self.group_specs:
            return torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        geom = self._group_spec(gid)
        return geom.is_placeable_mask(
            rot=int(rot),
            invalid=self._maps.invalid,
            clear_invalid=self._maps.clear_invalid,
        )

    def count_placeable(self, gid: GroupId, rot: int = 0) -> int:
        """배치 가능한 위치 개수 반환.
        
        Args:
            gid: 그룹 ID
            rot: rotation (0, 90, 180, 270)
        
        Returns:
            int: 배치 가능한 위치 개수
        """
        mask = self.get_placeable_mask(gid, rot)
        return int(mask.sum().item())

    def is_placeable_batch(
        self, *, gid: GroupId, x: object, y: object, rot: object
    ) -> torch.Tensor:
        """주어진 좌표들의 배치 가능 여부 batch 검사."""
        if gid not in self.group_specs:
            scalar_input = not (torch.is_tensor(x) or torch.is_tensor(y) or torch.is_tensor(rot))
            if scalar_input:
                n = 1
            elif torch.is_tensor(x):
                n = int(x.to(device=self.device).view(-1).numel())
            elif torch.is_tensor(y):
                n = int(y.to(device=self.device).view(-1).numel())
            elif torch.is_tensor(rot):
                n = int(rot.to(device=self.device).view(-1).numel())
            else:
                n = 1
            return torch.zeros((n,), dtype=torch.bool, device=self.device)
        geom = self._group_spec(gid)
        return geom.is_placeable_batch(
            x=x,
            y=y,
            rot=rot,
            invalid=self._maps.invalid,
            clear_invalid=self._maps.clear_invalid,
        )

    def delta_cost(self, *, gid: GroupId, x: object, y: object, rot: object):
        """배치 시 예상 Δcost (unscaled, raw).

        - scalar (int/float) 또는 batched torch.Tensor[M] 입력 지원 (정수 격자 좌표).
        - is_placeable 체크 없이 빠른 계산.
        - reward backend(Flow + Area)로 Δcost를 계산.

        반환: raw cost 변화량 (낮을수록 좋은 배치)
        """
        scalar_input = not (torch.is_tensor(x) or torch.is_tensor(y) or torch.is_tensor(rot))
        if gid not in self.group_specs:
            raise KeyError(f"delta_cost(batch): unknown gid={gid!r}")

        x_bl = self._as_long_tensor(x, name="x")
        y_bl = self._as_long_tensor(y, name="y")
        r = self._as_long_tensor(rot, name="rot")
        M = int(x_bl.numel())
        if int(y_bl.numel()) != M or int(r.numel()) != M:
            raise ValueError("delta_cost(batch): x,y,rot must have same length")

        # Normalize rotations to {0,90,180,270}
        r = torch.remainder(r, 360)
        if torch.any((r % 90) != 0):
            raise ValueError("delta_cost(batch): rot must be multiples of 90")

        geom = self._group_spec(gid)
        needed = set(self._reward.required())
        feature_map = geom.build_candidate_features(
            x_bl=x_bl,
            y_bl=y_bl,
            rot=r,
            needed=needed,
        )
        if "entries" in feature_map:
            feature_map["entries_mask"] = torch.ones(
                feature_map["entries"].shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
        if "exits" in feature_map:
            feature_map["exits_mask"] = torch.ones(
                feature_map["exits"].shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
        batch = CandidateBatch.from_feature_map(
            feature_map,
            required=needed,
            device=self.device,
        )

        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = self._placed_io_tensors()
        p = len(placed_nodes)
        w_out = torch.zeros((p,), dtype=torch.float32, device=self.device)
        w_in = torch.zeros((p,), dtype=torch.float32, device=self.device)
        out_edges = self.group_flow.get(gid, {})
        for i, pgid in enumerate(placed_nodes):
            w_out[i] = float(out_edges.get(pgid, 0.0))
            w_in[i] = float(self.group_flow.get(pgid, {}).get(gid, 0.0))
        cur_min_x, cur_max_x, cur_min_y, cur_max_y = self._placed_bbox(placed_nodes)
        ctx = RewardContext(
            placed_count=len(placed_nodes),
            cur_min_x=cur_min_x,
            cur_max_x=cur_max_x,
            cur_min_y=cur_min_y,
            cur_max_y=cur_max_y,
            placed_entries=placed_entries,
            placed_exits=placed_exits,
            placed_entries_mask=placed_entries_mask,
            placed_exits_mask=placed_exits_mask,
            w_out=w_out,
            w_in=w_in,
        )
        delta = self._reward.delta(ctx, batch).to(dtype=torch.float32)
        if scalar_input:
            return float(delta[0].item())
        return delta

    def placement_reward(self, *, gid: GroupId, x: object, y: object, rot: object):
        """배치 시 예상 reward (scaled).

        반환: -Δcost / cost_scale (높을수록 좋은 배치)
        """
        dc = self.delta_cost(gid=gid, x=x, y=y, rot=rot)
        if torch.is_tensor(dc):
            return -dc / float(self.cost_scale)
        return -float(dc) / float(self.cost_scale)

    def terminal_reward(self) -> float:
        """현재 상태의 예상 최종 reward (MCTS leaf 평가용).

        반환: -total_cost() / cost_scale
        """
        return -float(self.total_cost()) / float(self.cost_scale)

    @property
    def positions(self) -> Dict[GroupId, Tuple[int, int, int]]:
        """(x_bl, y_bl, rot) per placed gid — backward compat with old env.py.

        호출처: search/mcts.py, search/beam.py, webui/session.py,
                postprocess/dynamic_env.py, postprocess/dynamic_group.py,
                wrappers/alphachip.py
        """
        return {gid: (int(p.x_bl), int(p.y_bl), int(p.rot)) for gid, p in self.placements.items()}

    def try_place(self, gid: GroupId, x: float, y: float, rot: int) -> bool:
        """Place a group if feasible; returns True on success.

        This mirrors `env.py` semantics (placements/placed/remaining only).
        NOTE: Engine-internal cached maps are NOT updated here.
        Use `_apply_place(..., update_caches=True)` when you need cached-map consistency.
        """
        if gid not in self.group_specs:
            return False
        x_bl = int(x)
        y_bl = int(y)
        r = int(rot)
        geom = self._group_spec(gid)
        try:
            placeable = bool(
                geom.is_placeable(
                    x_bl=int(x_bl),
                    y_bl=int(y_bl),
                    rot=int(r),
                    invalid=self._maps.invalid,
                    clear_invalid=self._maps.clear_invalid,
                )
            )
        except Exception:
            placeable = False
        if not placeable:
            return False
        try:
            p = self._group_spec(gid).build_placement(
                x_bl=int(x_bl),
                y_bl=int(y_bl),
                rot=int(r),
            )
        except Exception:
            return False
        self._apply_resolved_placement(gid, p, update_caches=False)
        return True

    def _apply_place(self, gid: GroupId, x: float, y: float, rot: int, *, update_caches: bool) -> None:
        """Internal: apply a placement and optionally update engine caches."""
        x_bl = int(x)
        y_bl = int(y)
        r = int(rot)
        p = self._group_spec(gid).build_placement(
            x_bl=int(x_bl),
            y_bl=int(y_bl),
            rot=int(r),
        )
        self._apply_resolved_placement(gid, p, update_caches=update_caches)

    def _apply_resolved_placement(
        self,
        gid: GroupId,
        placement: PlacementBase,
        *,
        update_caches: bool,
    ) -> None:
        self.placements[gid] = placement
        self.placed.add(gid)
        if gid in self.remaining:
            self.remaining.remove(gid)
        if update_caches:
            self._maps.paint(placement)
            # Keep zone invalid consistent for the *next* group after any real placement.
            # This prevents stale zone_invalid when callers use step_masked() (wrappers).
            next_geom = self.group_specs.get(self.remaining[0]) if self.remaining else None
            self._maps.update_zone(next_geom)
            self._rebuild_flow_port_pairs()

    def apply_dynamic_placement(self, gid: GroupId, placement: object, *, update_caches: bool = True) -> None:
        """Apply a pre-resolved dynamic placement object (DynamicPlacement-compatible)."""
        if gid not in self.group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        self._apply_resolved_placement(gid, placement, update_caches=bool(update_caches))

    # ---- objective ----

    def cost(self) -> float:
        """현재 배치 목적함수 절댓값: weighted L1 flow + HPWL compactness."""
        if not self.placed:
            return 0.0

        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = self._placed_io_tensors()
        min_x, max_x, min_y, max_y = self._placed_bbox(placed_nodes)
        score_ctx = RewardContext(
            placed_count=len(placed_nodes),
            cur_min_x=min_x,
            cur_max_x=max_x,
            cur_min_y=min_y,
            cur_max_y=max_y,
            placed_entries=placed_entries,
            placed_exits=placed_exits,
            placed_entries_mask=placed_entries_mask,
            placed_exits_mask=placed_exits_mask,
            flow_w=self._build_flow_w(placed_nodes),
        )
        return float(self._reward.score(score_ctx).item())

    def total_cost(self) -> float:
        """cost() + 미배치 페널티 (TopK 정렬·비교용).

        완료된 레이아웃: cost()와 동일.
        미완료: cost() + penalty_weight * remaining_ratio.
        """
        return self.cost() + float(self.penalty_weight) * self._remaining_area_ratio()

    def _remaining_area_ratio(self) -> float:
        """Remaining area / total area ratio in [0,1]. Mirrors env.py."""
        total_area = 0.0
        remaining_area = 0.0
        for gid, spec in self.group_specs.items():
            a = float(spec.width) * float(spec.height)
            total_area += a
            if gid in self.remaining:
                remaining_area += a
        if total_area <= 0.0:
            return 1.0
        ratio = remaining_area / total_area
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return float(ratio)

    def _failure_penalty(self) -> float:
        """Penalty reward for failed placement or no valid actions (negative).
        
        스케일 통일: -(penalty_weight * remaining) / cost_scale
        이로써 최종 reward = -total_cost() / cost_scale 관계 성립.
        """
        raw_penalty = float(self.penalty_weight) * self._remaining_area_ratio()
        return -raw_penalty / float(self.cost_scale)
    
    # ---- Export API ----
    
    def export_placement(self) -> Dict[str, Any]:
        """현재 배치 상태를 JSON-serializable dict로 export.
        
        Returns:
            환경 설정 + 배치 정보 + 그룹 정보 + flow 정보
        """
        # 배치 정보
        placements = []
        for gid in sorted(self.placed, key=lambda x: str(x)):
            p = self.placements.get(gid, None)
            if p is not None:
                placements.append({
                    "gid": gid,
                    "x": int(getattr(p, "x_bl")),
                    "y": int(getattr(p, "y_bl")),
                    "rot": int(getattr(p, "rot")),
                })
        
        # 그룹 정보
        groups_data = {}
        for gid, s in self.group_specs.items():
            ent0 = s.entries_rel[0] if len(s.entries_rel) > 0 else (0.0, 0.0)
            ex0 = s.exits_rel[0] if len(s.exits_rel) > 0 else (0.0, 0.0)
            groups_data[gid] = {
                "id": s.id,
                "width": float(s.width),
                "height": float(s.height),
                "rotatable": bool(s.rotatable),
                "allowed_areas": None if s.allowed_areas is None else list(s.allowed_areas),
                "clearance_left": float(s.clearance_left_rel),
                "clearance_right": float(s.clearance_right_rel),
                "clearance_bottom": float(s.clearance_bottom_rel),
                "clearance_top": float(s.clearance_top_rel),
                "ent_rel_x": float(ent0[0]),
                "ent_rel_y": float(ent0[1]),
                "exi_rel_x": float(ex0[0]),
                "exi_rel_y": float(ex0[1]),
                "facility_height": float(s.facility_height),
                "facility_weight": float(s.facility_weight),
                "facility_dry": float(s.facility_dry),
            }
        
        # Flow 정보
        flow_edges = []
        for src, dsts in self.group_flow.items():
            for dst, weight in dsts.items():
                flow_edges.append([src, dst, float(weight)])
        
        return {
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
            "placements": placements,
            "groups": groups_data,
            "flow_edges": flow_edges,
            "forbidden_areas": list(self.forbidden_areas),
        }
    
    def save_placement(self, path: str) -> None:
        """배치 상태를 JSON 파일로 저장.
        
        Args:
            path: 저장할 파일 경로
        """
        import json
        data = self.export_placement()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _fail(
        self, reason: str, **extra: Any
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """실패 처리 통합 헬퍼."""
        reward = self._failure_penalty()
        info: Dict[str, Any] = {"reason": reason, "invalid": True}
        info.update(extra)

        if self.log:
            base_cost = float(self.cost())
            penalty = float(self.penalty_weight) * self._remaining_area_ratio()
            total_cost = base_cost + penalty
            print(
                f"[env] fail: reason={reason} remaining={len(self.remaining)} "
                f"cost={total_cost:.3f} (base={base_cost:.3f} + penalty={penalty:.3f}) reward={reward:.3f}"
            )

        return self._build_obs(), float(reward), False, True, info

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        """Return *core* observation (model-agnostic).

        Policy-specific wrappers should attach their own extra observation fields
        on top of this core dict.
        """
        gid = self.remaining[0] if self.remaining else self.node_ids[0]
        spec = self._group_spec(gid)

        N = int(len(self.node_ids))
        placed_mask = torch.zeros((N,), dtype=torch.bool, device=self.device)
        pos = torch.full((N, 3), -1, dtype=torch.long, device=self.device)  # (x_bl,y_bl,rot) or -1
        for gid2 in self.placed:
            idx = self.gid_to_idx.get(gid2, None)
            if idx is None:
                continue
            placed_mask[int(idx)] = True
            p = self.placements.get(gid2, None)
            if p is None:
                continue
            pos[int(idx), 0] = int(getattr(p, "x_bl"))
            pos[int(idx), 1] = int(getattr(p, "y_bl"))
            pos[int(idx), 2] = int(getattr(p, "rot"))

        # current gid index (group node list)
        cur_idx = int(self.gid_to_idx.get(gid, 0))

        obs: Dict[str, torch.Tensor] = {
            "current_gid_idx": torch.tensor([cur_idx], dtype=torch.long, device=self.device),
            # NOTE: normalized to canvas (grid) size for scale-stable learning.
            "next_group_wh": torch.tensor(
                [float(spec.width) / float(self.grid_width), float(spec.height) / float(self.grid_height)],
                dtype=torch.float32,
                device=self.device,
            ),
            "placed_mask": placed_mask,
            "positions_bl": pos,
            "step_count": torch.tensor([int(self._step_count)], dtype=torch.long, device=self.device),
        }
        return obs

    def step_action(
        self,
        *,
        x: float,
        y: float,
        rot: int,
        # 옵션: mask 관련 (없으면 검사 안함)
        action: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        action_space_n: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """통합 step 메서드: 배치 시도 + 옵션 mask 검사.

        Args:
            x, y, rot: 배치 좌표 (bottom-left, rotation)
            action: (옵션) discrete action index (mask 검사용)
            mask: (옵션) action mask (True=valid)
            action_space_n: (옵션) action space 크기
            extra_info: (옵션) info에 추가할 정보
        """
        self._step_count += 1

        # 0. extra_info 처리
        info: Dict[str, Any] = {}
        if extra_info:
            info.update(dict(extra_info))

        # 1. 이미 완료된 경우
        if not self.remaining:
            return self._build_obs(), 0.0, True, False, {"reason": "done", "invalid": False}

        # 2. Mask 검사 (파라미터가 있을 때만)
        if mask is not None:
            if int(mask.to(torch.int64).sum().item()) == 0:
                return self._fail("no_valid_actions")
            if action is not None and action_space_n is not None:
                if int(action) < 0 or int(action) >= int(action_space_n):
                    return self._fail("action_out_of_range")
                if not bool(mask[int(action)].item()):
                    return self._fail("masked_action")

        # 3. 배치 시도
        gid = self.remaining[0]
        x_bl = int(x)
        y_bl = int(y)
        r = int(rot)
        geom = self._group_spec(gid)
        try:
            placeable = bool(
                geom.is_placeable(
                    x_bl=int(x_bl),
                    y_bl=int(y_bl),
                    rot=int(r),
                    invalid=self._maps.invalid,
                    clear_invalid=self._maps.clear_invalid,
                )
            )
        except Exception:
            placeable = False
        if not placeable:
            return self._fail("not_placeable", gid=gid, x=int(x_bl), y=int(y_bl), rot=int(r))

        # 4. 성공 배치 (delta_cost: placed 상태 변경 전 Δcost 계산 → apply)
        delta = float(self.delta_cost(gid=gid, x=x_bl, y=y_bl, rot=r))
        self._apply_place(gid, float(x_bl), float(y_bl), int(r), update_caches=True)

        reward = -delta / float(self.cost_scale)
        terminated = len(self.remaining) == 0
        truncated = self.max_steps is not None and self._step_count >= self.max_steps

        info.update({
            "reason": "placed",
            "invalid": False,
            "gid": gid,
            "x": int(x_bl),
            "y": int(y_bl),
            "rot": int(r),
        })

        # 5. 로깅
        if self.log and (terminated or truncated):
            total_cost = self.total_cost()
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"remaining={len(self.remaining)} placed={len(self.placed)} step={self._step_count} "
                f"cost={total_cost:.3f} reason=placed reward={reward:.3f}"
            )

        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info

    # --- Backward-compatible aliases ---
    def step_place(
        self, *, x: float, y: float, rot: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """단순 배치 API (backward-compatible). step_action() 호출."""
        return self.step_action(x=x, y=y, rot=rot)

    def step_masked(
        self,
        *,
        action: int,
        x: float,
        y: float,
        rot: int,
        mask: Optional[torch.Tensor],
        action_space_n: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Wrapper용 API (backward-compatible). step_action() 호출."""
        return self.step_action(
            x=x,
            y=y,
            rot=rot,
            action=action,
            mask=mask,
            action_space_n=action_space_n,
            extra_info=extra_info,
        )

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = dict(options or {})
        initial_positions = options.get("initial_positions", None)
        remaining_order = options.get("remaining_order", None)

        self.placements = {}
        self.placed = set()
        self._flow_port_pairs = {}

        # Base order: "harder-first" heuristic.
        #
        # TEMP (requested): difficulty = facility_area / free_area.
        # - invalid_area: inv.sum() where inv = static_only | zone_invalid_for_gid(gid)
        # - free_area: total_area - invalid_area
        # - facility_area: footprint area (w*h)
        # Larger difficulty => bigger footprint relative to available free space => placed earlier.
        #
        # TODO(ord1): Restore placeable(K/T) ordering using conv2d-based top-left feasibility.
        ordering: List[Tuple[float, float, GroupId, int]] = []
        static_only = self._maps.static_invalid  # no occupancy at reset-time
        for gid in self.group_specs.keys():
            gg = self.group_specs[gid]
            inv = static_only | self._maps.zone_for_geom(self.group_specs[gid])
            facility_area = float(gg.width) * float(gg.height)
            # invalid_area (requested): true invalid area on grid (cell units; 1 cell == 1 area unit here)
            invalid_area = int(inv.to(torch.int64).sum().item())
            total_area = int(self.grid_width) * int(self.grid_height)
            free_area = max(1, int(total_area) - int(invalid_area))
            denom = float(free_area)
            difficulty = float(facility_area) / float(denom)
            ordering.append((difficulty, facility_area, gid, invalid_area))

        # Sort:
        # 1) difficulty descending (harder first)
        # 2) facility_area descending (bigger first among equally hard)
        # 3) gid for stable order
        ordering.sort(key=lambda t: (-t[0], -t[1], str(t[2])))
        base_remaining = [gid for _diff, _area, gid, _inv in ordering]

        if self.log:
            print("[reset_order] harder-first by difficulty (=facility_area/free_area)")
            for rank, (diff, area, gid, invalid_area) in enumerate(ordering, start=1):
                total_area = int(self.grid_width) * int(self.grid_height)
                free_area = max(1, int(total_area) - int(invalid_area))
                print(
                    f"  {rank}/{len(ordering)} gid={gid} "
                    f"facility_area={area:.1f} "
                    f"invalid_area={invalid_area} "
                    f"free_area={free_area} "
                    f"difficulty={diff:.6g}"
                )
        if remaining_order is not None:
            if not isinstance(remaining_order, list):
                raise ValueError("reset(options): remaining_order must be a list of group ids")
            # Validate remaining_order elements and uniqueness.
            seen = set()
            for gid in remaining_order:
                if gid not in self.group_specs:
                    raise ValueError(f"reset(options): remaining_order contains unknown group id: {gid!r}")
                if gid in seen:
                    raise ValueError(f"reset(options): remaining_order contains duplicate group id: {gid!r}")
                seen.add(gid)
            # Keep order provided, append missing groups by base order.
            rest = [gid for gid in base_remaining if gid not in seen]
            self.remaining = list(remaining_order) + rest
        else:
            self.remaining = list(base_remaining)

        self._step_count = 0
        self._maps.reset()

        # Apply validated initial placements (and sync caches).
        if initial_positions is not None:
            if not isinstance(initial_positions, dict):
                raise ValueError("reset(options): initial_positions must be a dict {gid: (x,y,rot)}")
            for gid, pose in initial_positions.items():
                if gid not in self.group_specs:
                    raise ValueError(f"reset(options): initial_positions has unknown group id: {gid!r}")
                if (not isinstance(pose, (tuple, list))) or len(pose) != 3:
                    raise ValueError(f"reset(options): initial_positions[{gid!r}] must be (x,y,rot)")
                x = int(pose[0])
                y = int(pose[1])
                rot = int(pose[2])
                if gid in self.placed:
                    raise ValueError(f"reset(options): initial_positions contains duplicate gid: {gid!r}")
                geom = self._group_spec(gid)
                try:
                    placeable = bool(
                        geom.is_placeable(
                            x_bl=int(x),
                            y_bl=int(y),
                            rot=int(rot),
                            invalid=self._maps.invalid,
                            clear_invalid=self._maps.clear_invalid,
                        )
                    )
                except Exception:
                    placeable = False
                if not placeable:
                    warnings.warn(
                        f"reset(options): invalid initial placement gid={gid!r} pose=({x},{y},{rot}) - placing anyway",
                        stacklevel=2,
                    )
                self._apply_place(gid, float(x), float(y), int(rot), update_caches=True)

        # Recompute zone for the next group (runs regardless of initial_positions).
        next_geom = self.group_specs.get(self.remaining[0]) if self.remaining else None
        self._maps.update_zone(next_geom)
        return self._build_obs(), {}

    # ---- snapshot api (for search/MCTS) ----
    def get_snapshot(self) -> Dict[str, object]:
        """Return a deep-ish snapshot for deterministic restore in search algorithms.

        Notes:
        - This is additive (does not change training/inference behavior).
        - Tensors are cloned to ensure isolation across rollouts.
        """
        return {
            "placements": dict(self.placements),
            "placed": set(self.placed),
            "remaining": list(self.remaining),
            "_step_count": int(self._step_count),
            "_occ_invalid": self._maps.occ_invalid.clone(),
            "_clear_invalid": self._maps.clear_invalid.clone(),
            "_invalid": self._maps.invalid.clone(),
            "_zone_invalid": self._maps.zone_invalid.clone(),
        }

    def set_snapshot(self, snapshot: Dict[str, object]) -> None:
        """Restore a snapshot produced by `get_snapshot`."""
        self.placements = dict(snapshot.get("placements", {}))  # type: ignore[arg-type]
        self.placed = set(snapshot.get("placed", set()))  # type: ignore[arg-type]
        self.remaining = list(snapshot.get("remaining", []))  # type: ignore[arg-type]
        self._step_count = int(snapshot.get("_step_count", 0))

        occ = snapshot.get("_occ_invalid", None)
        clr = snapshot.get("_clear_invalid", None)
        zinv = snapshot.get("_zone_invalid", None)
        if isinstance(occ, torch.Tensor):
            self._maps.occ_invalid.copy_(occ.to(device=self.device, dtype=torch.bool))
        if isinstance(clr, torch.Tensor):
            self._maps.clear_invalid.copy_(clr.to(device=self.device, dtype=torch.bool))
        if isinstance(zinv, torch.Tensor):
            self._maps.zone_invalid.copy_(zinv.to(device=self.device, dtype=torch.bool))
        self._maps.recompute()

    def step(self, action: int):
        x, y, rot, i, j = self.decode_action(action)
        obs, reward, terminated, truncated, info = self.step_place(x=x, y=y, rot=rot)
        info.update({"cell_i": i, "cell_j": j})
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    import time
    from envs.visualizer import plot_flow_graph, plot_layout

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")

    # ---------------------------------------------------------------
    # Group specs — BL-relative entries/exits, 여러 포트 사용 예시
    #
    # A (20×10): 왼쪽 2개 entry, 오른쪽·위 각 1개 exit
    # B (15×15): 왼쪽 2개 entry, 오른쪽·아래 각 1개 exit
    # C (18×12): 왼쪽·위 각 2개 entry (weight=12 → x>=60 영역 필수)
    # ---------------------------------------------------------------
    group_specs = {
        "A": StaticSpec(
            device=dev, id="A", width=20, height=10,
            entries_rel=[(0.0, 3.0), (0.0, 7.0)],      # 왼쪽 끝 상·하
            exits_rel=[(20.0, 5.0), (10.0, 10.0)],     # 오른쪽 끝 중간, 위쪽 끝 중간
            clearance_left_rel=1, clearance_right_rel=1,
            clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, facility_weight=3.0, facility_height=2.0, facility_dry=0.0,
        ),
        "B": StaticSpec(
            device=dev, id="B", width=15, height=15,
            entries_rel=[(0.0, 5.0), (0.0, 10.0)],     # 왼쪽 끝 하·상
            exits_rel=[(15.0, 7.5), (7.5, 0.0)],       # 오른쪽 끝 중간, 아래쪽 끝 중간
            clearance_left_rel=1, clearance_right_rel=1,
            clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, facility_weight=4.0, facility_height=2.0, facility_dry=0.0,
        ),
        "C": StaticSpec(
            device=dev, id="C", width=18, height=12,
            entries_rel=[(0.0, 4.0), (0.0, 8.0), (9.0, 12.0)],  # 왼쪽 끝 하·상, 위쪽 끝 중간
            exits_rel=[(18.0, 4.0), (18.0, 8.0)],                # 오른쪽 끝 하·상
            clearance_left_rel=0, clearance_right_rel=0,
            clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, facility_weight=12.0, facility_height=10.0, facility_dry=2.0,
        ),
    }
    # A→B, B→C 양방향 흐름
    group_flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # 왼쪽 하단 forbidden, 오른쪽 영역(x>=60)은 weight=20 허용 (C 배치 가능)
    forbidden_areas = [{"rect": [0, 0, 30, 20]}]
    weight_areas  = [{"rect": [60, 0, 120, 80], "value": 20.0}]
    height_areas  = [{"rect": [0, 60, 120, 80], "value": 5.0}]
    dry_areas     = [{"rect": [0, 40, 60, 80],  "value": 2.0}]

    # --- env 생성 ---
    t0 = time.perf_counter()
    env = FactoryLayoutEnv(
        grid_width=120, grid_height=80,
        group_specs=group_specs, group_flow=group_flow,
        forbidden_areas=forbidden_areas,
        default_weight=10.0, weight_areas=weight_areas,
        default_height=20.0, height_areas=height_areas,
        default_dry=0.0,     dry_areas=dry_areas,
        device=dev, max_steps=10, log=True,
    )
    init_ms = (time.perf_counter() - t0) * 1000.0

    # --- reset: A·B 사전 배치, C만 step으로 배치 ---
    # forbidden [0,0,30,20] 밖, C는 weight_areas(x>=60) 영역 필수
    initial_positions = {
        "A": (32, 22, 0),   # A: 20×10, clearance 1 → x∈[31,53], y∈[22,32]
        "B": (55, 22, 0),   # B: 15×15, clearance 1 → x∈[54,71], y∈[22,37]
    }
    t1 = time.perf_counter()
    obs, _ = env.reset(options={"initial_positions": initial_positions})
    reset_ms = (time.perf_counter() - t1) * 1000.0

    print("[env_demo]")
    print(f" device={dev}  init_ms={init_ms:.2f}  reset_ms={reset_ms:.2f}")
    print(f" placed={sorted(env.placed)}  remaining={env.remaining}")
    print(f" flow_port_pairs after reset: {env._flow_port_pairs}")

    # --- step: C 배치 (x>=60+clearance, weight_areas 이내) ---
    t2 = time.perf_counter()
    obs2, reward, terminated, truncated, info = env.step_place(x=74.0, y=22.0, rot=0)
    step_ms = (time.perf_counter() - t2) * 1000.0

    print(f" step_ms={step_ms:.2f}  reason={info['reason']}  reward={reward:.4f}  terminated={terminated}")
    print(f" flow_port_pairs after step:  {env._flow_port_pairs}")
    print(f" cost={env.cost():.4f}")
    print(f" obs_keys={list(obs.keys())}")

    plot_layout(env, candidate_set=None)
    plot_flow_graph(env)
