from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from .placement.base import PlacementBase
from .reward import (
    AreaReward,
    FlowReward,
    RewardComposer,
    TerminalReward,
)
from .placement.static import StaticSpec
from .action import GroupId, EnvAction
from .action_space import ActionSpace
from .state import EnvState, FlowGraph, GridMaps

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
        reward_scale: float = 100.0,
        penalty_weight: float = 50000.0,
        log: bool = False,
    ):
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        group_specs_norm = self._normalize_group_specs(group_specs)
        group_flow_norm = dict(group_flow or {})
        self.forbidden_areas = list(forbidden_areas or [])
        self.default_weight = float(default_weight)
        self.default_height = float(default_height)
        self.default_dry = float(default_dry)
        self.weight_areas = list(weight_areas or [])
        self.height_areas = list(height_areas or [])
        self.dry_areas = list(dry_areas or [])
        self.placement_areas = list(placement_areas or [])
        self.max_steps = max_steps
        self.reward_scale = float(reward_scale)
        self.penalty_weight = float(penalty_weight)
        self.log = bool(log)

        # Engine does not own action semantics. Wrappers define action_space/obs additions.
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict({})

        # Engine static metadata (direct fields; no proxy properties).
        self.group_specs = group_specs_norm
        self.group_flow = group_flow_norm
        self.node_ids: List[GroupId] = sorted(self.group_specs.keys(), key=lambda x: str(x))
        self.gid_to_idx: Dict[GroupId, int] = {gid: i for i, gid in enumerate(self.node_ids)}

        maps = GridMaps(
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
        flow = FlowGraph(self.group_flow, device=self.device)
        self._state = EnvState.empty(
            maps=maps,
            flow=flow,
            group_specs=self.group_specs,
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
            reward_scale=float(self.reward_scale),
        )
        group_areas = {
            gid: float(spec.width) * float(spec.height)
            for gid, spec in self.group_specs.items()
        }
        self._terminal = TerminalReward(
            penalty_weight=float(self.penalty_weight),
            reward_scale=float(self.reward_scale),
            group_areas=group_areas,
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

    def get_maps(self) -> GridMaps:
        """Public map bundle for wrappers/agents/postprocess modules."""
        return self._state.maps

    def get_state(self) -> EnvState:
        """Runtime mutable state bundle."""
        return self._state

    def set_state(self, state: EnvState) -> None:
        """Restore runtime state bundle."""
        if not isinstance(state, EnvState):
            raise TypeError(f"state must be EnvState, got {type(state).__name__}")
        self._state.restore(state)

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

    def _rebuild_flow_port_pairs(self) -> None:
        """FlowReward.score(return_argmin=True)로 flow edge별 최적 port pair 캐시 재구성.

        self._state.flow_port_pairs[(src_gid, dst_gid)] = ((exit_x, exit_y), (entry_x, entry_y))
        flow weight가 0인 엣지는 포함하지 않음.
        """
        flow_comp = self._reward.components.get("flow")
        if flow_comp is None:
            return
        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = self._state.io_tensors()
        if len(placed_nodes) == 0:
            self._state.clear_flow_port_pairs()
            return
        flow_w = self._state.build_flow_w()
        _, c_idx, p_idx = flow_comp.score(
            placed_entries=placed_entries,
            placed_exits=placed_exits,
            placed_entries_mask=placed_entries_mask,
            placed_exits_mask=placed_exits_mask,
            flow_w=flow_w,
            return_argmin=True,
        )
        if c_idx is None or p_idx is None:
            self._state.clear_flow_port_pairs()
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
        self._state.set_flow_port_pairs(pairs)

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
        xyrot = torch.stack([x_bl, y_bl, r], dim=-1).to(dtype=torch.long, device=self.device)
        entries = feature_map.get("entries", None)
        exits = feature_map.get("exits", None)
        entries_mask = None
        exits_mask = None
        if entries is not None:
            entries = entries.to(device=self.device)
            entries_mask = torch.ones(
                entries.shape[:2], dtype=torch.bool, device=self.device,
            )
        if exits is not None:
            exits = exits.to(device=self.device)
            exits_mask = torch.ones(
                exits.shape[:2], dtype=torch.bool, device=self.device,
            )
        aspace = ActionSpace(
            xyrot=xyrot,
            mask=torch.ones((M,), dtype=torch.bool, device=self.device),
            gid=gid,
            entries=entries,
            exits=exits,
            entries_mask=entries_mask,
            exits_mask=exits_mask,
            min_x=feature_map.get("min_x", None),
            max_x=feature_map.get("max_x", None),
            min_y=feature_map.get("min_y", None),
            max_y=feature_map.get("max_y", None),
        )
        delta = self._reward.delta(self._state, aspace, gid=gid).to(dtype=torch.float32)
        if scalar_input:
            return float(delta[0].item())
        return delta

    def _normalize_action(self, action: EnvAction) -> Tuple[GroupId, int, int, int]:
        if not isinstance(action, EnvAction):
            raise TypeError(f"expected EnvAction, got {type(action).__name__}")
        gid_eff: GroupId = action.gid
        if gid_eff not in self.group_specs:
            raise KeyError(f"unknown gid={gid_eff!r}")
        return gid_eff, int(action.x), int(action.y), int(action.rot)

    def is_placeable(self, action: EnvAction) -> bool:
        gid, x, y, rot = self._normalize_action(action)
        geom = self._group_spec(gid)
        return self._state.is_placeable(
            action=EnvAction(gid=gid, x=int(x), y=int(y), rot=int(rot)),
            spec=geom,
        )

    def is_placeable_mask(self, action_space: ActionSpace) -> torch.Tensor:
        gid = action_space.gid
        if gid is None:
            raise ValueError("action_space.gid is required")
        geom = self._group_spec(gid)
        return self._state.is_placeable_mask(
            action_space=action_space,
            spec=geom,
        )

    def _apply_resolved_placement(
        self,
        gid: GroupId,
        placement: PlacementBase,
    ) -> None:
        self._state.place(gid=gid, placement=placement)
        self._rebuild_flow_port_pairs()

    def apply_dynamic_placement(self, gid: GroupId, placement: object) -> None:
        """Apply a pre-resolved dynamic placement object (DynamicPlacement-compatible)."""
        if gid not in self.group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        self._apply_resolved_placement(gid, placement)

    # ---- objective ----

    def cost(self) -> float:
        """현재 배치 목적함수 절댓값: weighted L1 flow + HPWL compactness."""
        if not self._state.placed:
            return 0.0
        return float(self._reward.score(self._state).item())

    def total_cost(self) -> float:
        """cost() + 미배치 페널티 (TopK 정렬·비교용).

        완료된 레이아웃: cost()와 동일.
        미완료: cost() + penalty_weight * remaining_ratio.
        """
        return float(self.cost()) + float(self._terminal.penalty(self._state))

    def failure_penalty(self) -> float:
        """Penalty reward for failed placement or no valid actions (negative).
        
        스케일 통일: -(penalty_weight * remaining) / reward_scale
        이로써 최종 reward = -total_cost() / reward_scale 관계 성립.
        """
        return self._terminal.failure_reward(self._state)

    def reorder_remaining(self, ordered_remaining: List[GroupId]) -> None:
        """Replace `remaining` order with a validated permutation of current remaining gids."""
        if not isinstance(ordered_remaining, list):
            raise TypeError("ordered_remaining must be a list")
        if len(ordered_remaining) != len(self._state.remaining):
            raise ValueError(
                f"ordered_remaining length mismatch: got {len(ordered_remaining)}, expected {len(self._state.remaining)}"
            )
        seen = set()
        for gid in ordered_remaining:
            if gid not in self._state.remaining:
                raise ValueError(f"ordered_remaining contains gid not in current remaining: {gid!r}")
            if gid in seen:
                raise ValueError(f"ordered_remaining contains duplicate gid: {gid!r}")
            seen.add(gid)
        self._state.set_remaining(list(ordered_remaining))
    
    # ---- Export API ----
    
    def export_placement(self) -> Dict[str, Any]:
        """현재 배치 상태를 JSON-serializable dict로 export.
        
        Returns:
            환경 설정 + 배치 정보 + 그룹 정보 + flow 정보
        """
        # 배치 정보
        placements = []
        for gid in sorted(self._state.placed, key=lambda x: str(x)):
            p = self._state.placements.get(gid, None)
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

    def _fail(self, reason: str) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """실패 처리 통합 헬퍼."""
        reward = self.failure_penalty()
        info: Dict[str, Any] = {"reason": reason}

        if self.log:
            base_cost = float(self.cost())
            penalty = float(self._terminal.penalty(self._state))
            total_cost = base_cost + penalty
            print(
                f"[env] fail: reason={reason} remaining={len(self._state.remaining)} "
                f"cost={total_cost:.3f} (base={base_cost:.3f} + penalty={penalty:.3f}) reward={reward:.3f}"
            )

        return self.build_observation(), float(reward), False, True, info

    def build_observation(self) -> Dict[str, torch.Tensor]:
        """Return model-agnostic engine observation.

        Policy-specific wrappers should attach their own extra observation fields
        on top of this base dict.
        """
        N = int(len(self.node_ids))
        placed_mask = torch.zeros((N,), dtype=torch.bool, device=self.device)
        pos = torch.full((N, 3), -1, dtype=torch.long, device=self.device)  # (x_bl,y_bl,rot) or -1
        for gid2 in self._state.placed:
            idx = self.gid_to_idx.get(gid2, None)
            if idx is None:
                continue
            placed_mask[int(idx)] = True
            p = self._state.placements.get(gid2, None)
            if p is None:
                continue
            pos[int(idx), 0] = int(getattr(p, "x_bl"))
            pos[int(idx), 1] = int(getattr(p, "y_bl"))
            pos[int(idx), 2] = int(getattr(p, "rot"))

        obs: Dict[str, torch.Tensor] = {
            "placed_mask": placed_mask,
            "positions_bl": pos,
            "step_count": torch.tensor([int(self._state.step_count)], dtype=torch.long, device=self.device),
        }
        return obs

    def step_action(
        self,
        action: EnvAction,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Place a single `EnvAction` (wrapper owns mask/index validation)."""
        # 1. 이미 완료된 경우
        if not self._state.remaining:
            self._state.step(apply=False)
            return self.build_observation(), 0.0, True, False, {"reason": "done"}

        try:
            gid_eff, x_bl, y_bl, r = self._normalize_action(action)
        except TypeError:
            raise
        except Exception:
            self._state.step(apply=False)
            return self._fail("invalid_action_payload")
        if gid_eff not in self._state.remaining:
            self._state.step(apply=False)
            return self._fail("gid_not_remaining")

        try:
            placeable = bool(self.is_placeable(EnvAction(gid=gid_eff, x=x_bl, y=y_bl, rot=r)))
        except Exception:
            placeable = False
        if not placeable:
            self._state.step(apply=False)
            return self._fail("not_placeable")

        # 4. 성공 배치 (delta_cost: placed 상태 변경 전 Δcost 계산 → apply)
        delta = float(self.delta_cost(gid=gid_eff, x=x_bl, y=y_bl, rot=r))
        placement = self._group_spec(gid_eff).build_placement(
            x_bl=int(x_bl),
            y_bl=int(y_bl),
            rot=int(r),
        )
        self._state.step(
            apply=True,
            gid=gid_eff,
            placement=placement,
        )
        self._rebuild_flow_port_pairs()

        reward = float(self._reward.to_reward(delta))
        terminated = len(self._state.remaining) == 0
        truncated = self.max_steps is not None and self._state.step_count >= self.max_steps

        info: Dict[str, Any] = {"reason": "placed"}

        # 5. 로깅
        if self.log and (terminated or truncated):
            total_cost = self.total_cost()
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"remaining={len(self._state.remaining)} placed={len(self._state.placed)} step={self._state.step_count} "
                f"cost={total_cost:.3f} reason=placed reward={reward:.3f}"
            )

        return self.build_observation(), float(reward), bool(terminated), bool(truncated), info

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = dict(options or {})
        initial_positions = options.get("initial_positions", None)
        remaining_order = options.get("remaining_order", None)

        # Base order: 그대로 입력 순서 (env-side remaining 정렬 없음).
        base_remaining = list(self.group_specs.keys())
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
            remaining = list(remaining_order) + rest
        else:
            remaining = list(base_remaining)
        self._state.reset_runtime(remaining=remaining)

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
                if gid in self._state.placed:
                    raise ValueError(f"reset(options): initial_positions contains duplicate gid: {gid!r}")
                geom = self._group_spec(gid)
                try:
                    placeable = bool(self.is_placeable(EnvAction(gid=gid, x=x, y=y, rot=rot)))
                except Exception:
                    placeable = False
                if not placeable:
                    warnings.warn(
                        f"reset(options): invalid initial placement gid={gid!r} pose=({x},{y},{rot}) - placing anyway",
                        stacklevel=2,
                    )
                placement = geom.build_placement(
                    x_bl=int(x),
                    y_bl=int(y),
                    rot=int(rot),
                )
                self._apply_resolved_placement(gid, placement)
        return self.build_observation(), {}

    def step(self, action: EnvAction):
        """Gym-compatible step proxy for engine-only usage."""
        if not isinstance(action, EnvAction):
            raise TypeError(f"FactoryLayoutEnv.step expects EnvAction, got {type(action).__name__}")
        return self.step_action(action)


if __name__ == "__main__":
    import time
    from envs.env_visualizer import plot_flow_graph, plot_layout

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
    print(f" placed={sorted(env.get_state().placed)}  remaining={env.get_state().remaining}")
    print(f" flow_port_pairs after reset: {env.get_state().flow_port_pairs}")

    # --- step: C 배치 (x>=60+clearance, weight_areas 이내) ---
    t2 = time.perf_counter()
    obs2, reward, terminated, truncated, info = env.step_action(
        EnvAction(gid="C", x=74, y=22, rot=0)
    )
    step_ms = (time.perf_counter() - t2) * 1000.0

    print(f" step_ms={step_ms:.2f}  reason={info['reason']}  reward={reward:.4f}  terminated={terminated}")
    print(f" flow_port_pairs after step:  {env.get_state().flow_port_pairs}")
    print(f" cost={env.cost():.4f}")
    print(f" obs_keys={list(obs.keys())}")

    plot_layout(env, action_space=None)
    plot_flow_graph(env)
