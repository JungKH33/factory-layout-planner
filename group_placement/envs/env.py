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
    TerminalPenaltyReward,
    TerminalFlowReward,
    TerminalRewardComposer,
)
from .placement.base import GroupSpec
from .placement.static import StaticRectSpec
from .state import EnvState, FlowGraph, GridMaps

# GroupPlacement is defined in envs/placement/base.py and imported above.
# Re-exported here so external code can do: from group_placement.envs.env import GroupPlacement, GridMaps
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
        group_specs: Dict[str | int, GroupSpec],
        group_flow: Optional[Dict[str | int, Dict[str | int, float]]] = None,
        # Forbidden zones:
        # - rect: {"shape_type": "rect", "rect": [x0, y0, x1, y1]}
        # - irregular: {"shape_type": "irregular", "polygon": [[x, y], ...]}
        forbidden: Optional[List[Dict[str, Any]]] = None,
        # Generic map constraints:
        # zones.constraints.<name> = {
        #   "dtype": ..., "op": ..., "default": ...,
        #   "areas": [{"shape_type": "rect", "rect": [...], "value": ...}] or
        #            [{"shape_type": "irregular", "polygon": [[x,y], ...], "value": ...}]
        # }
        zone_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        device: Optional[torch.device] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 100.0,
        penalty_weight: float = 50000.0,
        terminal_reward_components: Optional[Dict[str, object]] = None,
        terminal_reward_weights: Optional[Dict[str, float]] = None,
        terminal_flow_unreachable_cost: float = 1e6,
        terminal_flow_max_wave_iters: int = 0,
        terminal_flow_batched_wavefront: bool = True,
        terminal_flow_include_clearance: bool = False,
        log: bool = False,
        backend_selection: str = "static",
    ):
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        group_specs_norm = self._normalize_group_specs(group_specs)
        group_flow_norm = dict(group_flow or {})
        self.forbidden = list(forbidden or [])
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
        self.node_ids: List[str | int] = sorted(self.group_specs.keys(), key=lambda x: str(x))
        self.gid_to_idx: Dict[str | int, int] = {gid: i for i, gid in enumerate(self.node_ids)}

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
            forbidden=self.forbidden,
            zone_constraints=self.zone_constraints,
            backend_selection=backend_selection,
        )
        flow = FlowGraph(self.group_flow, device=self.device)
        self._state = EnvState.empty(
            maps=maps,
            flow=flow,
            group_specs=self.group_specs,
            device=self.device,
            reward_scale=float(self.reward_scale),
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
        if terminal_reward_components is None:
            t_components: Dict[str, object] = {
                "penalty": TerminalPenaltyReward(
                    penalty_weight=float(self.penalty_weight),
                    group_areas=group_areas,
                ),
                "flow": TerminalFlowReward(
                    group_specs=self.group_specs,
                    unreachable_cost=float(terminal_flow_unreachable_cost),
                    max_wave_iters=int(terminal_flow_max_wave_iters),
                    batched_wavefront=bool(terminal_flow_batched_wavefront),
                    include_clear_invalid=bool(terminal_flow_include_clearance),
                ),
            }
        else:
            t_components = dict(terminal_reward_components)
            t_components.setdefault(
                "penalty",
                TerminalPenaltyReward(
                    penalty_weight=float(self.penalty_weight),
                    group_areas=group_areas,
                ),
            )
        t_weights = dict(terminal_reward_weights or {})
        if not t_weights:
            t_weights = {name: 1.0 for name in t_components.keys()}
        self._terminal = TerminalRewardComposer(
            components=t_components,
            weights=t_weights,
            reward_scale=float(self.reward_scale),
        )
        self._reset_base_eval(reset_terminal=True)

    def _normalize_group_specs(self, group_specs: Dict[str | int, GroupSpec]) -> Dict[str | int, GroupSpec]:
        """Normalize group specs onto env device and validate id/key consistency."""
        out: Dict[str | int, GroupSpec] = {}
        for gid, s in dict(group_specs).items():
            if not isinstance(s, GroupSpec):
                raise TypeError(f"group_specs[{gid!r}] must be GroupSpec, got {type(s).__name__}")
            sid = getattr(s, "id", gid)
            if sid != gid:
                raise ValueError(f"group_specs key/id mismatch: key={gid!r}, spec.id={sid!r}")
            s.set_device(self.device)
            out[gid] = s
        return out

    def _group_spec(self, gid: str | int) -> GroupSpec:
        geom = self.group_specs.get(gid, None)
        if geom is None:
            raise KeyError(f"unknown gid={gid!r}")
        return geom

    def get_maps(self) -> GridMaps:
        """Public map bundle for wrappers/agents/lane/facility modules."""
        return self._state.maps

    def get_state(self) -> EnvState:
        """Runtime mutable state bundle."""
        return self._state

    def set_state(self, state: EnvState) -> None:
        """Restore runtime state bundle."""
        if not isinstance(state, EnvState):
            raise TypeError(f"state must be EnvState, got {type(state).__name__}")
        self._state.restore(state)
        objective = self._state.eval.objective
        if "cost_total" not in objective:
            self._state.eval.recompute_objective(
                finalized=bool(objective.get("finalized", False)),
            )

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

    @property
    def reward_composer(self) -> RewardComposer:
        """Public access to the RewardComposer for direct delta_batch calls."""
        return self._reward

    @property
    def terminal_reward_composer(self):
        """Public access to the TerminalRewardComposer (for visualization)."""
        return self._terminal

    def _placement_features_from_placements(
        self,
        placements: List[GroupPlacement],
    ) -> tuple[str | int, Dict[str, Optional[torch.Tensor]]]:
        """Build reward feature tensors for pre-resolved placements."""
        M = len(placements)
        if M == 0:
            raise ValueError("placements must not be empty")
        gid = placements[0].group_id
        for p in placements[1:]:
            if p.group_id != gid:
                raise ValueError(
                    "all placements passed to _placement_features_from_placements() "
                    "must share the same group_id"
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
        features: Dict[str, Optional[torch.Tensor]] = {
            "entry_points": entry_points,
            "exit_points": exit_points,
            "entry_mask": entry_mask,
            "exit_mask": exit_mask,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }
        return gid, features

    def _delta_cost_from_placements(
        self,
        placements: List[GroupPlacement],
    ) -> torch.Tensor:
        """Score already-resolved placements via reward delta."""
        if len(placements) == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        gid, features = self._placement_features_from_placements(placements)
        return self._reward.delta_batch(
            self._state,
            gid=gid,
            entry_points=features["entry_points"],
            exit_points=features["exit_points"],
            entry_mask=features["entry_mask"],
            exit_mask=features["exit_mask"],
            min_x=features["min_x"],
            max_x=features["max_x"],
            min_y=features["min_y"],
            max_y=features["max_y"],
        ).to(dtype=torch.float32)

    def _delta_single_for_placement(
        self,
        placement: GroupPlacement,
    ) -> tuple[float, Dict[str, float], Dict[str, Dict[str, Any]]]:
        gid, features = self._placement_features_from_placements([placement])
        return self._reward.delta_single(
            self._state,
            gid=gid,
            entry_points=features["entry_points"],
            exit_points=features["exit_points"],
            entry_mask=features["entry_mask"],
            exit_mask=features["exit_mask"],
            min_x=features["min_x"],
            max_x=features["max_x"],
            min_y=features["min_y"],
            max_y=features["max_y"],
            base_rewards=self._state.eval.base_rewards,
        )

    def _normalize_center_request(
        self,
        *,
        group_id: str | int,
        x_center: float,
        y_center: float,
    ) -> Tuple[str | int, float, float]:
        gid_eff: str | int = group_id
        if gid_eff not in self.group_specs:
            raise KeyError(f"unknown gid={gid_eff!r}")
        return gid_eff, float(x_center), float(y_center)

    def resolve_center_placement(
        self,
        *,
        group_id: str | int,
        x_center: float,
        y_center: float,
        variant_index: Optional[int] = None,
        source_index: Optional[int] = None,
    ) -> GroupPlacement | None:
        """Resolve a center-based request to a concrete placement.

        Center→BL conversion happens here, per shape_key, so each shape gets
        the correct BL from the shared center coordinate.
        """
        gid, x_center, y_center = self._normalize_center_request(
            group_id=group_id,
            x_center=x_center,
            y_center=y_center,
        )
        geom = self._group_spec(gid)

        def _check_placeable(xb, yb, body_mask, clearance_mask, clearance_origin, is_rectangular):
            return self._state.maps.placeable(
                gid=gid,
                x_bl=int(xb), y_bl=int(yb),
                body_mask=body_mask, clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                is_rectangular=bool(is_rectangular),
            )

        # Single variant — fast path
        if variant_index is not None:
            vi = geom._variants[int(variant_index)]
            x_bl = int(round(x_center - float(vi.body_width) / 2.0))
            y_bl = int(round(y_center - float(vi.body_height) / 2.0))
            p = geom.resolve(
                x_bl=x_bl, y_bl=y_bl,
                variant_index=int(variant_index),
                is_placeable_fn=_check_placeable,
            )
            return p

        # Multiple variants — determine candidates
        if source_index is not None:
            s, e = geom._source_ranges[int(source_index)]
            candidates = list(range(s, e))
        else:
            candidates = list(range(len(geom._variants)))

        # Group by shape_key → center→BL per shape, skip shape if not placeable.
        from collections import defaultdict
        by_shape: Dict[tuple, List[int]] = defaultdict(list)
        for vi_idx in candidates:
            by_shape[geom._variants[vi_idx].shape_key].append(vi_idx)

        placeable: List[GroupPlacement] = []
        for sk, vi_indices in by_shape.items():
            vi0 = geom._variants[vi_indices[0]]
            sk_x_bl = int(round(x_center - float(vi0.body_width) / 2.0))
            sk_y_bl = int(round(y_center - float(vi0.body_height) / 2.0))

            body_mask, clearance_mask, clearance_origin, is_rect = geom.shape_tensors(sk)
            if not self._state.maps.placeable(
                gid=gid, x_bl=sk_x_bl, y_bl=sk_y_bl,
                body_mask=body_mask, clearance_mask=clearance_mask,
                clearance_origin=clearance_origin, is_rectangular=bool(is_rect),
            ):
                continue
            for vi_idx in vi_indices:
                p = geom.resolve(
                    x_bl=sk_x_bl, y_bl=sk_y_bl,
                    variant_index=vi_idx,
                    is_placeable_fn=_check_placeable,
                )
                if p is not None:
                    placeable.append(p)

        if not placeable:
            return None
        if len(placeable) == 1:
            return placeable[0]
        scores = self._delta_cost_from_placements(placeable)
        scores = scores.to(dtype=torch.float32, device=self.device).view(-1)
        return placeable[int(torch.argmin(scores).item())]

    def _apply_resolved_placement(
        self,
        placement: GroupPlacement,
    ) -> None:
        _total_delta, reward_delta_by_name, metadata_by_reward = self._delta_single_for_placement(placement)
        self._state.place(placement=placement)
        self._apply_base_delta(
            reward_delta_by_name=reward_delta_by_name,
            metadata_by_reward=metadata_by_reward,
        )

    def apply_dynamic_placement(self, placement: object) -> None:
        """Apply a pre-resolved dynamic placement object (DynamicPlacement-compatible)."""
        gid = placement.group_id
        if gid not in self.group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        self._apply_resolved_placement(placement)

    # ---- objective ----

    def _runtime_metadata(self, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(raw or {})
        out["status"] = "ok"
        out["layout_rev"] = self._state.eval.layout_rev
        return out

    def _reset_base_eval(self, *, reset_terminal: bool) -> None:
        eval_state = self._state.eval
        base_snapshot: Dict[str, Dict[str, Any]] = {}
        for name in self._reward.components.keys():
            name_s = str(name)
            w = float(self._reward.weights.get(name, self._reward.weights.get(name_s, 1.0)))
            base_snapshot[name_s] = {
                "raw_cost": 0.0,
                "weight": w,
                "metadata": self._runtime_metadata({}),
            }
        eval_state.set_base_snapshot(base_snapshot)
        if bool(reset_terminal):
            eval_state.clear_terminal()

    def _apply_base_delta(
        self,
        *,
        reward_delta_by_name: Dict[str, float],
        metadata_by_reward: Dict[str, Dict[str, Any]],
    ) -> None:
        runtime_metadata_by_reward = {
            str(name): self._runtime_metadata(dict(metadata or {}))
            for name, metadata in dict(metadata_by_reward or {}).items()
        }
        self._state.eval.record_base_delta(
            reward_delta_by_name={str(k): float(v) for k, v in reward_delta_by_name.items()},
            metadata_by_reward=runtime_metadata_by_reward,
        )

    def _finalize_terminal_eval(self, *, failed: bool) -> float:
        eval_state = self._state.eval
        if bool(eval_state.objective.get("finalized", False)):
            return float(eval_state.objective.get("terminal_total", 0.0))
        base_unweighted = {
            str(name): float(rec.get("raw_cost", 0.0))
            for name, rec in eval_state.base_rewards.items()
            if isinstance(rec, dict)
        }

        delta_raw, delta_meta = self._terminal.delta_dict(
            state=self._state,
            maps=self.get_maps(),
            reward_composer=self._reward,
            failed=bool(failed),
            base_scores_unweighted=base_unweighted,
            return_metadata=True,
        )
        terminal_snapshot: Dict[str, Dict[str, Any]] = {}
        for name, delta in delta_raw.items():
            name_s = str(name)
            terminal_snapshot[name_s] = {
                "delta_cost": float(delta),
                "metadata": self._runtime_metadata(delta_meta.get(name_s, {})),
            }
        eval_state.set_terminal_snapshot(terminal_snapshot)
        eval_state.recompute_objective(finalized=True)
        return float(eval_state.objective.get("terminal_total", 0.0))

    def record_reward_meta(
        self,
        *,
        reward_name: str,
        metadata: Dict[str, Any],
        phase: str = "base",
    ) -> None:
        phase_key = str(phase)
        runtime_metadata = self._runtime_metadata(dict(metadata or {}))
        self._state.eval.merge_metadata(
            name=str(reward_name),
            metadata=runtime_metadata,
            phase=phase_key,
        )

    def cost(self) -> float:
        """Current objective value cached in state runtime.

        Non-finalized states return base_total only — terminal rewards
        are excluded until the episode actually ends.
        """
        obj = self._state.eval.objective
        if "cost_total" not in obj:
            self._state.eval.recompute_objective(
                finalized=bool(obj.get("finalized", False)),
            )
            obj = self._state.eval.objective
        if not bool(obj.get("finalized", False)):
            return float(obj.get("base_total", 0.0))
        return float(obj.get("cost_total", 0.0))

    def failure_penalty(self) -> float:
        """Estimated penalty if episode fails now (read-only, no state mutation)."""
        for comp in self._terminal.components.values():
            if isinstance(comp, TerminalPenaltyReward):
                return comp.penalty_cost(state=self._state) / float(self.reward_scale)
        return 0.0

    def reorder_remaining(self, ordered_remaining: List[str | int]) -> None:
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
    # Placement interchange: ``group_placement.envs.interchange`` (export_placement /
    # import_placement).

    def _end_episode(self, *, reason: str, failed: bool) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Finalize terminal reward once and return unified step-like outputs."""
        terminal_delta = self._finalize_terminal_eval(failed=bool(failed))
        reward = -float(terminal_delta) / float(self.reward_scale)
        terminated = not bool(failed)
        truncated = bool(failed)
        info: Dict[str, Any] = {"reason": reason}
        obj = self._state.eval.objective
        base_cost = float(obj.get("base_total", 0.0))
        terminal_total = float(obj.get("terminal_total", 0.0))
        final_cost = float(self.cost())
        if bool(failed):
            logger.warning(
                "fail: reason=%s remaining=%d cost=%.3f (base=%.3f + terminal=%.3f) reward=%.3f",
                reason,
                len(self._state.remaining),
                final_cost,
                base_cost,
                terminal_total,
                reward,
            )
        else:
            logger.info(
                "end: terminated=%s truncated=%s remaining=%d placed=%d step=%d cost=%.3f reason=%s reward=%.3f",
                terminated,
                truncated,
                len(self._state.remaining),
                len(self._state.placed),
                self._state.step_count,
                final_cost,
                reason,
                reward,
            )
        return {}, float(reward), bool(terminated), bool(truncated), info

    def fail(self, reason: str) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Public failed-termination endpoint with step-compatible outputs."""
        return self._end_episode(reason=reason, failed=True)

    def step(
        self,
        placement: GroupPlacement,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Apply a resolved GroupPlacement directly. No variant resolution."""
        gid = placement.group_id
        if not self._state.remaining:
            self._state.step(apply=False)
            return {}, 0.0, True, False, {"reason": "done"}
        if gid not in self._state.remaining:
            self._state.step(apply=False)
            return self._end_episode(reason="gid_not_remaining", failed=True)

        # Placeability validation — env never trusts external input
        if not self._state.placeable(placement=placement):
            self._state.step(apply=False)
            return self._end_episode(reason="not_placeable", failed=True)

        delta_cost, reward_delta_by_name, metadata_by_reward = self._delta_single_for_placement(placement)

        self._state.step(apply=True, placement=placement)
        self._apply_base_delta(
            reward_delta_by_name=reward_delta_by_name,
            metadata_by_reward=metadata_by_reward,
        )

        reward = float(self._reward.to_reward(delta_cost))
        terminated = len(self._state.remaining) == 0
        truncated = self.max_steps is not None and self._state.step_count >= self.max_steps
        info: Dict[str, Any] = {"reason": "placed"}

        if terminated:
            _obs, terminal_reward, terminated, truncated, info = self._end_episode(
                reason="placed",
                failed=False,
            )
            reward += float(terminal_reward)
            return _obs, float(reward), bool(terminated), bool(truncated), info
        if truncated:
            _obs, terminal_reward, terminated, truncated, info = self._end_episode(
                reason="placed",
                failed=True,
            )
            reward += float(terminal_reward)
            return _obs, float(reward), bool(terminated), bool(truncated), info

        return {}, float(reward), False, False, info

    # ---- reset ----
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
        self._reset_base_eval(reset_terminal=True)

        # Apply pre-resolved initial placements.
        initial_placements = options.get("initial_placements", None)
        placement_order = options.get("placement_order", None)
        strict_initial = bool(options.get("strict_initial_placements", False))
        if initial_placements is not None:
            if not isinstance(initial_placements, dict):
                raise ValueError("reset(options): initial_placements must be a dict {gid: GroupPlacement}")
            if placement_order is not None:
                if not isinstance(placement_order, list):
                    raise ValueError("reset(options): placement_order must be a list of group ids")
                iter_gids: List[str | int] = list(placement_order)
                seen_po: set = set()
                for gid in iter_gids:
                    if gid not in initial_placements:
                        raise ValueError(
                            f"reset(options): placement_order contains gid={gid!r} missing from initial_placements"
                        )
                    if gid in seen_po:
                        raise ValueError(f"reset(options): placement_order duplicate gid={gid!r}")
                    seen_po.add(gid)
                if strict_initial and len(seen_po) != len(initial_placements):
                    raise ValueError(
                        "reset(options): strict_initial_placements requires placement_order to list "
                        "every initial_placements key exactly once"
                    )
            else:
                iter_gids = list(initial_placements.keys())

            for gid in iter_gids:
                placement = initial_placements[gid]
                if not isinstance(placement, GroupPlacement):
                    raise TypeError(
                        f"reset(options): initial_placements[{gid!r}] must be GroupPlacement, "
                        f"got {type(placement).__name__}"
                    )
                if gid != placement.group_id:
                    raise ValueError(
                        f"reset(options): key {gid!r} does not match placement.group_id={placement.group_id!r}"
                    )
                if gid in self._state.placed:
                    raise ValueError(f"reset(options): initial_placements contains duplicate gid: {gid!r}")
                if not self._state.placeable(placement=placement):
                    msg = (
                        f"reset(options): not placeable gid={gid!r} "
                        f"x_center={float(placement.x_center):.3f} y_center={float(placement.y_center):.3f}"
                    )
                    if strict_initial:
                        raise RuntimeError(msg)
                    warnings.warn(f"{msg} — skipping", stacklevel=2)
                    continue
                self._apply_resolved_placement(placement)
        if len(self._state.remaining) == 0 and len(self._state.placed) > 0:
            self._finalize_terminal_eval(failed=False)
        return {}, {}


if __name__ == "__main__":
    import time
    from group_placement.envs.visualizer import plot_flow_graph, plot_layout

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
    forbidden = [{"shape_type": "rect", "rect": [0, 0, 30, 20]}]
    zone_constraints = {
        "weight": {
            "dtype": "float",
            "op": "<=",
            "default": 10.0,
            "areas": [{"shape_type": "rect", "rect": [60, 0, 120, 80], "value": 20.0}],
        },
        "height": {
            "dtype": "float",
            "op": "<=",
            "default": 20.0,
            "areas": [{"shape_type": "rect", "rect": [0, 60, 120, 80], "value": 5.0}],
        },
        "dry": {
            "dtype": "float",
            "op": ">=",
            "default": 0.0,
            "areas": [{"shape_type": "rect", "rect": [0, 40, 60, 80], "value": 2.0}],
        },
        "placeable": {
            "dtype": "int",
            "op": "==",
            "default": 0,
            "areas": [{"shape_type": "rect", "rect": [30, 20, 120, 80], "value": 1}],
        },
    }

    # --- env 생성 ---
    t0 = time.perf_counter()
    env = FactoryLayoutEnv(
        grid_width=120, grid_height=80,
        group_specs=group_specs, group_flow=group_flow,
        forbidden=forbidden,
        zone_constraints=zone_constraints,
        device=dev, max_steps=10, log=True,
    )
    init_ms = (time.perf_counter() - t0) * 1000.0

    # --- reset: A·B 사전 배치, C만 step으로 배치 ---
    # forbidden [0,0,30,20] 밖 + constraint map 조건
    p_a = env.resolve_center_placement(group_id="A", x_center=42.0, y_center=27.0, variant_index=0)
    p_b = env.resolve_center_placement(group_id="B", x_center=62.5, y_center=29.5, variant_index=0)
    assert p_a is not None and p_b is not None
    initial_placements = {"A": p_a, "B": p_b}
    t1 = time.perf_counter()
    obs, _ = env.reset(options={"initial_placements": initial_placements})
    reset_ms = (time.perf_counter() - t1) * 1000.0

    print("env_demo")
    print(f" device={dev}  init_ms={init_ms:.2f}  reset_ms={reset_ms:.2f}")
    print(f" placed={sorted(env.get_state().placed)}  remaining={env.get_state().remaining}")
    print(f" objective after reset: {env.get_state().eval.objective}")

    # --- step: C 배치 ---
    t2 = time.perf_counter()
    # C: 18×12 at rotation=0 → center = (74 + 9, 22 + 6) = (83, 28)
    p_c = env.resolve_center_placement(group_id="C", x_center=83.0, y_center=28.0)
    assert p_c is not None
    obs2, reward, terminated, truncated, info = env.step(p_c)
    step_ms = (time.perf_counter() - t2) * 1000.0

    print(f" step_ms={step_ms:.2f}  reason={info['reason']}  reward={reward:.4f}  terminated={terminated}")
    print(f" objective after step:  {env.get_state().eval.objective}")
    print(f" cost={env.cost():.4f}")
    print(f" obs_keys={list(obs.keys())}")

    plot_layout(env, action_space=None)
    plot_flow_graph(env)
