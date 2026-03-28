from __future__ import annotations

import logging
import platform
import warnings
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from .placement.base import GroupPlacement
from .reward import (
    AreaReward,
    FlowReward,
    RewardComposer,
    TerminalReward,
)
from .placement.base import GroupSpec
from .placement.static import StaticRectSpec
from .action import GroupId, EnvAction
from .state import EnvState, FlowGraph, GridMaps

# GroupPlacement is defined in envs/placement/base.py and imported above.
# Re-exported here so external code can do: from envs.env import GroupPlacement, GridMaps
__all__ = ["GroupPlacement", "GridMaps", "FactoryLayoutEnv"]

logger = logging.getLogger(__name__)


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
        group_specs: Dict[GroupId, GroupSpec],
        group_flow: Optional[Dict[GroupId, Dict[GroupId, float]]] = None,
        # Forbidden areas: [{"rect": [x0, y0, x1, y1]}, ...]
        forbidden_areas: Optional[List[Dict[str, Any]]] = None,
        # Generic map constraints:
        # zones.constraints.<name> = {"dtype": ..., "op": ..., "default": ..., "areas": [{"rect": [...], "value": ...}]}
        zone_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        device: Optional[torch.device] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 100.0,
        penalty_weight: float = 50000.0,
        log: bool = False,
        backend_selection: str = "static",
    ):
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        group_specs_norm = self._normalize_group_specs(group_specs)
        group_flow_norm = dict(group_flow or {})
        self.forbidden_areas = list(forbidden_areas or [])
        self.zone_constraints = dict(zone_constraints or {})
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

        # Log env info
        if self.device.type == "cuda":
            dev_name = torch.cuda.get_device_name(self.device)
        else:
            dev_name = platform.processor() or platform.machine()
        n_flows = sum(len(v) for v in self.group_flow.values())
        logger.info("=== env initialized ===")
        logger.info("  device: %s - %s", self.device.type, dev_name)
        logger.info("  grid: %dx%d", self.grid_width, self.grid_height)
        logger.info("  groups: %d  flows: %d", len(self.group_specs), n_flows)

        maps = GridMaps(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            device=self.device,
            forbidden_areas=self.forbidden_areas,
            zone_constraints=self.zone_constraints,
            backend_selection=backend_selection,
        )
        flow = FlowGraph(self.group_flow, device=self.device)
        self._state = EnvState.empty(
            maps=maps,
            flow=flow,
            group_specs=self.group_specs,
            device=self.device,
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
            group_specs=self.group_specs,
        )
        group_areas = {
            gid: float(spec.body_area)
            for gid, spec in self.group_specs.items()
        }
        self._terminal = TerminalReward(
            penalty_weight=float(self.penalty_weight),
            reward_scale=float(self.reward_scale),
            group_areas=group_areas,
        )

    def _normalize_group_specs(self, group_specs: Dict[GroupId, GroupSpec]) -> Dict[GroupId, GroupSpec]:
        """Normalize group specs onto env device and validate id/key consistency."""
        out: Dict[GroupId, GroupSpec] = {}
        for gid, s in dict(group_specs).items():
            if not isinstance(s, GroupSpec):
                raise TypeError(f"group_specs[{gid!r}] must be GroupSpec, got {type(s).__name__}")
            sid = getattr(s, "id", gid)
            if sid != gid:
                raise ValueError(f"group_specs key/id mismatch: key={gid!r}, spec.id={sid!r}")
            s.set_device(self.device)
            out[gid] = s
        return out

    def _group_spec(self, gid: GroupId) -> GroupSpec:
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

    def _flow_pair_list(
        self,
        *,
        src_row: int,
        dst_row: int,
        exit_argmin: int,
        entry_argmin: int,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: torch.Tensor,
        placed_exits_mask: torch.Tensor,
        exit_modes: Optional[torch.Tensor],
        entry_modes: Optional[torch.Tensor],
    ) -> list:
        src_exit_mean = exit_modes is not None and bool(exit_modes[src_row].item())
        dst_entry_mean = entry_modes is not None and bool(entry_modes[dst_row].item())

        if src_exit_mean:
            e_idxs = [int(j.item()) for j in torch.where(placed_exits_mask[src_row])[0]]
        else:
            e_idxs = [int(exit_argmin)]

        if dst_entry_mean:
            n_idxs = [int(j.item()) for j in torch.where(placed_entries_mask[dst_row])[0]]
        else:
            n_idxs = [int(entry_argmin)]

        pair_list = []
        for ei in e_idxs:
            ex = (float(placed_exits[src_row, ei, 0].item()), float(placed_exits[src_row, ei, 1].item()))
            for ni in n_idxs:
                en = (float(placed_entries[dst_row, ni, 0].item()), float(placed_entries[dst_row, ni, 1].item()))
                pair_list.append((ex, en))
        return pair_list

    def _try_update_flow_port_pairs_incremental(
        self,
        *,
        updated_gid: GroupId,
        flow_comp: FlowReward,
        placed_nodes: List[GroupId],
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: torch.Tensor,
        placed_exits_mask: torch.Tensor,
        flow_w: torch.Tensor,
        exit_modes: Optional[torch.Tensor],
        entry_modes: Optional[torch.Tensor],
    ) -> None:
        nodes_key = tuple(placed_nodes)
        if updated_gid not in nodes_key:
            raise KeyError(f"updated_gid={updated_gid!r} not found in placed_nodes")
        prev_key = self._state.flow_port_pairs_nodes_key
        updated_row = int(placed_nodes.index(updated_gid))

        is_append = (
            len(nodes_key) == len(prev_key) + 1
            and nodes_key[:-1] == prev_key
            and nodes_key[-1] == updated_gid
        )
        is_refresh = nodes_key == prev_key
        if not (is_append or is_refresh):
            raise RuntimeError(
                "flow_port_pairs incremental update requires append-only growth "
                f"or in-place refresh: prev={prev_key!r} current={nodes_key!r} updated_gid={updated_gid!r}"
            )

        if is_append:
            pairs: Dict[Tuple[GroupId, GroupId], list] = dict(self._state.flow_port_pairs)
        else:
            pairs = {
                key: value
                for key, value in self._state.flow_port_pairs.items()
                if key[0] != updated_gid and key[1] != updated_gid
            }

        _, out_c_idx, out_p_idx = flow_comp._reduce_distance(
            candidate_ports=placed_exits[updated_row:updated_row + 1],
            candidate_mask=placed_exits_mask[updated_row:updated_row + 1],
            target_ports=placed_entries,
            target_mask=placed_entries_mask,
            target_weight=flow_w[updated_row:updated_row + 1, :],
            c_modes=exit_modes[updated_row:updated_row + 1] if exit_modes is not None else None,
            t_modes=entry_modes,
        )
        _, in_c_idx, in_p_idx = flow_comp._reduce_distance(
            candidate_ports=placed_exits,
            candidate_mask=placed_exits_mask,
            target_ports=placed_entries[updated_row:updated_row + 1],
            target_mask=placed_entries_mask[updated_row:updated_row + 1],
            target_weight=flow_w[:, updated_row:updated_row + 1],
            c_modes=exit_modes,
            t_modes=entry_modes[updated_row:updated_row + 1] if entry_modes is not None else None,
        )
        if out_c_idx is None or out_p_idx is None or in_c_idx is None or in_p_idx is None:
            raise RuntimeError(
                f"flow_port_pairs incremental update failed to compute argmin indices for gid={updated_gid!r}"
            )

        for dst_row, dst_gid in enumerate(placed_nodes):
            if float(flow_w[updated_row, dst_row].item()) <= 0.0:
                continue
            pairs[(updated_gid, dst_gid)] = self._flow_pair_list(
                src_row=updated_row,
                dst_row=dst_row,
                exit_argmin=int(out_c_idx[0, dst_row].item()),
                entry_argmin=int(out_p_idx[0, dst_row].item()),
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                exit_modes=exit_modes,
                entry_modes=entry_modes,
            )

        for src_row, src_gid in enumerate(placed_nodes):
            if src_row == updated_row or float(flow_w[src_row, updated_row].item()) <= 0.0:
                continue
            pairs[(src_gid, updated_gid)] = self._flow_pair_list(
                src_row=src_row,
                dst_row=updated_row,
                exit_argmin=int(in_c_idx[src_row, 0].item()),
                entry_argmin=int(in_p_idx[src_row, 0].item()),
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                exit_modes=exit_modes,
                entry_modes=entry_modes,
            )

        self._state.set_flow_port_pairs(pairs, nodes=placed_nodes)

    def _update_flow_port_pairs(self, *, updated_gid: GroupId) -> None:
        """Update per-edge port pair cache for visualization.

        For min-mode ports: store the single argmin pair.
        For mean-mode ports: store all valid ports (cartesian product).
        """
        flow_comp = self._reward.components.get("flow")
        if flow_comp is None:
            return
        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = self._state.io_tensors()
        if len(placed_nodes) == 0:
            self._state.clear_flow_port_pairs()
            return
        flow_w = self._state.build_flow_w()
        exit_modes, entry_modes = self._reward._port_mode_tensors(
            placed_nodes, placed_entries.device,
        )
        self._try_update_flow_port_pairs_incremental(
            updated_gid=updated_gid,
            flow_comp=flow_comp,
            placed_nodes=placed_nodes,
            placed_entries=placed_entries,
            placed_exits=placed_exits,
            placed_entries_mask=placed_entries_mask,
            placed_exits_mask=placed_exits_mask,
            flow_w=flow_w,
            exit_modes=exit_modes,
            entry_modes=entry_modes,
        )

    @property
    def reward_composer(self) -> RewardComposer:
        """Public access to the RewardComposer for direct delta_batch calls."""
        return self._reward

    def _delta_cost_from_placements(
        self,
        gid: GroupId,
        placements: List[GroupPlacement],
    ) -> torch.Tensor:
        """Score already-resolved placements via reward delta.

        Reads center/geometry directly from GroupPlacement.
        """
        M = len(placements)
        if M == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)

        poses = torch.tensor(
            [[p.x_center, p.y_center] for p in placements],
            dtype=torch.float32, device=self.device,
        )

        # Build entry_points/exit_points tensors from placement geometry
        max_ent = max((len(p.entry_points) for p in placements), default=0)
        max_exi = max((len(p.exit_points) for p in placements), default=0)
        entry_points = None
        exit_points = None
        entry_mask = None
        exit_mask = None
        if max_ent > 0:
            entry_points = torch.zeros((M, max_ent, 2), dtype=torch.float32, device=self.device)
            entry_mask = torch.zeros((M, max_ent), dtype=torch.bool, device=self.device)
            for i, p in enumerate(placements):
                for j, (ex, ey) in enumerate(p.entry_points):
                    entry_points[i, j, 0] = float(ex)
                    entry_points[i, j, 1] = float(ey)
                    entry_mask[i, j] = True
        if max_exi > 0:
            exit_points = torch.zeros((M, max_exi, 2), dtype=torch.float32, device=self.device)
            exit_mask = torch.zeros((M, max_exi), dtype=torch.bool, device=self.device)
            for i, p in enumerate(placements):
                for j, (ex, ey) in enumerate(p.exit_points):
                    exit_points[i, j, 0] = float(ex)
                    exit_points[i, j, 1] = float(ey)
                    exit_mask[i, j] = True

        min_x = torch.tensor([p.min_x for p in placements], dtype=torch.float32, device=self.device)
        max_x = torch.tensor([p.max_x for p in placements], dtype=torch.float32, device=self.device)
        min_y = torch.tensor([p.min_y for p in placements], dtype=torch.float32, device=self.device)
        max_y = torch.tensor([p.max_y for p in placements], dtype=torch.float32, device=self.device)

        return self._reward.delta_batch(
            self._state, gid=gid,
            entry_points=entry_points, exit_points=exit_points,
            entry_mask=entry_mask, exit_mask=exit_mask,
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
        ).to(dtype=torch.float32)

    def _normalize_action(self, action: EnvAction) -> Tuple[GroupId, float, float]:
        if not isinstance(action, EnvAction):
            raise TypeError(f"expected EnvAction, got {type(action).__name__}")
        gid_eff: GroupId = action.group_id
        if gid_eff not in self.group_specs:
            raise KeyError(f"unknown gid={gid_eff!r}")
        return gid_eff, float(action.x_center), float(action.y_center)

    def resolve_action(self, action: EnvAction) -> Tuple[GroupId, 'GroupPlacement | None']:
        """Resolve a center-based EnvAction to (gid, concrete placement or None).

        Tries all variants at the given center and picks the cheapest placeable
        one.  If ``action.variant_index`` is set, only that specific variant
        is attempted.
        """
        gid, x_center, y_center = self._normalize_action(action)
        geom = self._group_spec(gid)

        def _check_placeable(x_bl, y_bl, body_mask, clearance_mask, clearance_origin, is_rectangular):
            return self._state.is_placeable(
                gid=gid,
                x_bl=int(x_bl),
                y_bl=int(y_bl),
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                is_rectangular=bool(is_rectangular),
            )

        placement = geom.resolve(
            x_center=float(x_center),
            y_center=float(y_center),
            is_placeable_fn=_check_placeable,
            score_fn=lambda ps: self._delta_cost_from_placements(gid, ps),
            variant_index=action.variant_index,
            source_index=action.source_index,
        )
        return gid, placement

    def _apply_resolved_placement(
        self,
        gid: GroupId,
        placement: GroupPlacement,
    ) -> None:
        self._state.place(gid=gid, placement=placement)
        self._update_flow_port_pairs(updated_gid=gid)

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
                    "x": int(float(getattr(p, "min_x", 0.0))),
                    "y": int(float(getattr(p, "min_y", 0.0))),
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
                "clearance_left": float(s.clearance_lrtb_rel[0]) if s.clearance_lrtb_rel else 0.0,
                "clearance_right": float(s.clearance_lrtb_rel[1]) if s.clearance_lrtb_rel else 0.0,
                "clearance_bottom": float(s.clearance_lrtb_rel[2]) if s.clearance_lrtb_rel else 0.0,
                "clearance_top": float(s.clearance_lrtb_rel[3]) if s.clearance_lrtb_rel else 0.0,
                "ent_rel_x": float(ent0[0]),
                "ent_rel_y": float(ent0[1]),
                "exi_rel_x": float(ex0[0]),
                "exi_rel_y": float(ex0[1]),
                "zone_values": dict(getattr(s, "zone_values", {}) or {}),
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
            "zone_constraints": dict(self.zone_constraints),
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

        base_cost = float(self.cost())
        penalty = float(self._terminal.penalty(self._state))
        total_cost = base_cost + penalty
        logger.warning(
            "fail: reason=%s remaining=%d cost=%.3f (base=%.3f + penalty=%.3f) reward=%.3f",
            reason,
            len(self._state.remaining),
            total_cost,
            base_cost,
            penalty,
            reward,
        )

        return {}, float(reward), False, True, info

    def step_action(
        self,
        action: EnvAction,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Place a single `EnvAction` (wrapper owns mask/index validation)."""
        # 1. 이미 완료된 경우
        if not self._state.remaining:
            self._state.step(apply=False)
            return {}, 0.0, True, False, {"reason": "done"}

        try:
            gid_eff, x_center, y_center = self._normalize_action(action)
        except TypeError:
            raise
        except Exception:
            self._state.step(apply=False)
            return self._fail("invalid_action_payload")
        if gid_eff not in self._state.remaining:
            self._state.step(apply=False)
            return self._fail("gid_not_remaining")

        try:
            _gid, placement = self.resolve_action(action)
        except Exception:
            placement = None
        if placement is None:
            self._state.step(apply=False)
            return self._fail("not_placeable")

        # TODO(perf): resolve_action already evaluates delta_cost internally
        # via score_fn to pick the best variant.  Here we re-compute it for
        # the chosen concrete placement.  Refactor to carry the score through
        # resolve() to avoid the double computation.
        delta = float(self._delta_cost_from_placements(gid_eff, [placement])[0].item())
        self._state.step(
            apply=True,
            gid=gid_eff,
            placement=placement,
        )
        self._update_flow_port_pairs(updated_gid=gid_eff)

        reward = float(self._reward.to_reward(delta))
        terminated = len(self._state.remaining) == 0
        truncated = self.max_steps is not None and self._state.step_count >= self.max_steps

        info: Dict[str, Any] = {"reason": "placed"}

        # 5. 로깅
        if terminated or truncated:
            total_cost = self.total_cost()
            logger.info(
                "end: terminated=%s truncated=%s remaining=%d placed=%d step=%d cost=%.3f reason=placed reward=%.3f",
                terminated,
                truncated,
                len(self._state.remaining),
                len(self._state.placed),
                self._state.step_count,
                total_cost,
                reward,
            )

        return {}, float(reward), bool(terminated), bool(truncated), info

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = dict(options or {})
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

        # Apply initial placements via the same resolve_action path as step_action.
        initial_placements = options.get("initial_placements", None)
        if initial_placements is not None:
            if not isinstance(initial_placements, dict):
                raise ValueError("reset(options): initial_placements must be a dict {gid: EnvAction}")
            for gid, action in initial_placements.items():
                if not isinstance(action, EnvAction):
                    raise TypeError(
                        f"reset(options): initial_placements[{gid!r}] must be EnvAction, "
                        f"got {type(action).__name__}"
                    )
                if gid != action.group_id:
                    raise ValueError(
                        f"reset(options): key {gid!r} does not match action.group_id={action.group_id!r}"
                    )
                if gid in self._state.placed:
                    raise ValueError(f"reset(options): initial_placements contains duplicate gid: {gid!r}")
                _gid, placement = self.resolve_action(action)
                if placement is None:
                    warnings.warn(
                        f"reset(options): not placeable gid={gid!r} "
                        f"x_center={action.x_center} y_center={action.y_center} vi={action.variant_index} — skipping",
                        stacklevel=2,
                    )
                    continue
                self._apply_resolved_placement(gid, placement)
        return {}, {}

    def step(self, action: EnvAction):
        """Gym-compatible step proxy for engine-only usage."""
        if not isinstance(action, EnvAction):
            raise TypeError(f"FactoryLayoutEnv.step expects EnvAction, got {type(action).__name__}")
        return self.step_action(action)


if __name__ == "__main__":
    import time
    from envs.visualizer import plot_flow_graph, plot_layout

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")

    # ---------------------------------------------------------------
    # Group specs — BL-relative entry_points/exit_points, 여러 포트 사용 예시
    #
    # A (20×10): 왼쪽 2개 entry, 오른쪽·위 각 1개 exit
    # B (15×15): 왼쪽 2개 entry, 오른쪽·아래 각 1개 exit
    # C (18×12): 왼쪽·위 각 2개 entry (weight=12 → x>=60 영역 필수)
    # ---------------------------------------------------------------
    group_specs = {
        "A": StaticRectSpec(
            device=dev, id="A", width=20, height=10,
            entries_rel=[(0.0, 3.0), (0.0, 7.0)],      # 왼쪽 끝 상·하
            exits_rel=[(20.0, 5.0), (10.0, 10.0)],     # 오른쪽 끝 중간, 위쪽 끝 중간
            clearance_lrtb_rel=(1, 1, 0, 0),
            rotatable=True,
            zone_values={"weight": 3.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "B": StaticRectSpec(
            device=dev, id="B", width=15, height=15,
            entries_rel=[(0.0, 5.0), (0.0, 10.0)],     # 왼쪽 끝 하·상
            exits_rel=[(15.0, 7.5), (7.5, 0.0)],       # 오른쪽 끝 중간, 아래쪽 끝 중간
            clearance_lrtb_rel=(1, 1, 0, 0),
            rotatable=True,
            zone_values={"weight": 4.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "C": StaticRectSpec(
            device=dev, id="C", width=18, height=12,
            entries_rel=[(0.0, 4.0), (0.0, 8.0), (9.0, 12.0)],  # 왼쪽 끝 하·상, 위쪽 끝 중간
            exits_rel=[(18.0, 4.0), (18.0, 8.0)],                # 오른쪽 끝 하·상
            rotatable=True,
            zone_values={"weight": 12.0, "height": 10.0, "dry": 2.0, "placeable": 1},
        ),
    }
    # A→B, B→C 양방향 흐름
    group_flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # 왼쪽 하단 forbidden + 제약 맵
    forbidden_areas = [{"rect": [0, 0, 30, 20]}]
    zone_constraints = {
        "weight": {
            "dtype": "float",
            "op": "<=",
            "default": 10.0,
            "areas": [{"rect": [60, 0, 120, 80], "value": 20.0}],
        },
        "height": {
            "dtype": "float",
            "op": "<=",
            "default": 20.0,
            "areas": [{"rect": [0, 60, 120, 80], "value": 5.0}],
        },
        "dry": {
            "dtype": "float",
            "op": ">=",
            "default": 0.0,
            "areas": [{"rect": [0, 40, 60, 80], "value": 2.0}],
        },
        "placeable": {
            "dtype": "int",
            "op": "==",
            "default": 0,
            "areas": [{"rect": [30, 20, 120, 80], "value": 1}],
        },
    }

    # --- env 생성 ---
    t0 = time.perf_counter()
    env = FactoryLayoutEnv(
        grid_width=120, grid_height=80,
        group_specs=group_specs, group_flow=group_flow,
        forbidden_areas=forbidden_areas,
        zone_constraints=zone_constraints,
        device=dev, max_steps=10, log=True,
    )
    init_ms = (time.perf_counter() - t0) * 1000.0

    # --- reset: A·B 사전 배치, C만 step으로 배치 ---
    # forbidden [0,0,30,20] 밖 + constraint map 조건
    initial_placements = {
        "A": EnvAction(group_id="A", x_center=42.0, y_center=27.0, variant_index=0),
        "B": EnvAction(group_id="B", x_center=62.5, y_center=29.5, variant_index=0),
    }
    t1 = time.perf_counter()
    obs, _ = env.reset(options={"initial_placements": initial_placements})
    reset_ms = (time.perf_counter() - t1) * 1000.0

    print("env_demo")
    print(f" device={dev}  init_ms={init_ms:.2f}  reset_ms={reset_ms:.2f}")
    print(f" placed={sorted(env.get_state().placed)}  remaining={env.get_state().remaining}")
    print(f" flow_port_pairs after reset: {env.get_state().flow_port_pairs}")

    # --- step: C 배치 ---
    t2 = time.perf_counter()
    # C: 18×12 at rotation=0 → center = (74 + 9, 22 + 6) = (83, 28)
    obs2, reward, terminated, truncated, info = env.step_action(
        EnvAction(group_id="C", x_center=83.0, y_center=28.0)
    )
    step_ms = (time.perf_counter() - t2) * 1000.0

    print(f" step_ms={step_ms:.2f}  reason={info['reason']}  reward={reward:.4f}  terminated={terminated}")
    print(f" flow_port_pairs after step:  {env.get_state().flow_port_pairs}")
    print(f" cost={env.cost():.4f}")
    print(f" obs_keys={list(obs.keys())}")

    plot_layout(env, action_space=None)
    plot_flow_graph(env)
