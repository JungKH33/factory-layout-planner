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
from .action import GroupId, EnvAction
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
        group_specs: Dict[GroupId, GroupSpec],
        group_flow: Optional[Dict[GroupId, Dict[GroupId, float]]] = None,
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
        self._refresh_base_cost_snapshot(reset_terminal=True)

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
        if bool(self._state.cost.get("finalized", False)):
            self._refresh_base_cost_snapshot(reset_terminal=False)
        else:
            self._refresh_base_cost_snapshot(reset_terminal=True)

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
        exit_k: Optional[torch.Tensor],
        entry_k: Optional[torch.Tensor],
    ) -> list:
        src_req_k = 1 if exit_k is None else int(exit_k[src_row].item())
        dst_req_k = 1 if entry_k is None else int(entry_k[dst_row].item())
        valid_exit_idx = torch.where(placed_exits_mask[src_row])[0]
        valid_entry_idx = torch.where(placed_entries_mask[dst_row])[0]

        if int(valid_exit_idx.numel()) == 0 or int(valid_entry_idx.numel()) == 0:
            return []

        if src_req_k == 1:
            e_idxs = [int(exit_argmin)] if bool((valid_exit_idx == int(exit_argmin)).any().item()) else [int(valid_exit_idx[0].item())]
        else:
            src_eff_k = min(int(src_req_k), int(valid_exit_idx.numel()))
            anchor_entry = placed_entries[dst_row, int(entry_argmin), :]
            exits = placed_exits[src_row, valid_exit_idx, :]
            dist = (exits - anchor_entry.view(1, 2)).abs().sum(dim=1)
            take = torch.topk(dist, k=src_eff_k, largest=False).indices
            e_idxs = [int(valid_exit_idx[int(i.item())].item()) for i in take]

        if dst_req_k == 1:
            n_idxs = [int(entry_argmin)] if bool((valid_entry_idx == int(entry_argmin)).any().item()) else [int(valid_entry_idx[0].item())]
        else:
            dst_eff_k = min(int(dst_req_k), int(valid_entry_idx.numel()))
            anchor_exit = placed_exits[src_row, int(exit_argmin), :]
            entries = placed_entries[dst_row, valid_entry_idx, :]
            dist = (entries - anchor_exit.view(1, 2)).abs().sum(dim=1)
            take = torch.topk(dist, k=dst_eff_k, largest=False).indices
            n_idxs = [int(valid_entry_idx[int(i.item())].item()) for i in take]

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
        exit_k: Optional[torch.Tensor],
        entry_k: Optional[torch.Tensor],
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
            c_k=exit_k[updated_row:updated_row + 1] if exit_k is not None else None,
            t_k=entry_k,
        )
        _, in_c_idx, in_p_idx = flow_comp._reduce_distance(
            candidate_ports=placed_exits,
            candidate_mask=placed_exits_mask,
            target_ports=placed_entries[updated_row:updated_row + 1],
            target_mask=placed_entries_mask[updated_row:updated_row + 1],
            target_weight=flow_w[:, updated_row:updated_row + 1],
            c_k=exit_k,
            t_k=entry_k[updated_row:updated_row + 1] if entry_k is not None else None,
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
                exit_k=exit_k,
                entry_k=entry_k,
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
                exit_k=exit_k,
                entry_k=entry_k,
            )

        self._state.set_flow_port_pairs(pairs, nodes=placed_nodes)

    def _update_flow_port_pairs(self, *, updated_gid: GroupId) -> None:
        """Update per-edge port pair cache for visualization.

        For span=1 ports: store the single argmin pair.
        For span>1 / span=all: store selected port combinations.
        """
        _flow_name, flow_comp = self._reward.find_component(FlowReward)
        if flow_comp is None:
            return
        placed_nodes, placed_entries, placed_exits, placed_entries_mask, placed_exits_mask = self._state.io_tensors()
        if len(placed_nodes) == 0:
            self._state.clear_flow_port_pairs()
            return
        flow_w = self._state.build_flow_w()
        exit_k, entry_k = self._reward._port_span_tensors(
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
            exit_k=exit_k,
            entry_k=entry_k,
        )

    @property
    def reward_composer(self) -> RewardComposer:
        """Public access to the RewardComposer for direct delta_batch calls."""
        return self._reward

    def _delta_cost_from_placements(
        self,
        placements: List[GroupPlacement],
    ) -> torch.Tensor:
        """Score already-resolved placements via reward delta.

        Reads center/geometry directly from GroupPlacement.
        """
        M = len(placements)
        if M == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        gid = placements[0].group_id
        for p in placements[1:]:
            if p.group_id != gid:
                raise ValueError(
                    "all placements passed to _delta_cost_from_placements() "
                    "must share the same group_id"
                )

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

        Center→BL conversion happens here, per shape_key, so each shape gets
        the correct BL from the shared center coordinate.
        """
        gid, x_center, y_center = self._normalize_action(action)
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
        if action.variant_index is not None:
            vi = geom._variants[action.variant_index]
            x_bl = int(round(x_center - float(vi.body_width) / 2.0))
            y_bl = int(round(y_center - float(vi.body_height) / 2.0))
            p = geom.resolve(
                x_bl=x_bl, y_bl=y_bl,
                variant_index=action.variant_index,
                is_placeable_fn=_check_placeable,
            )
            return gid, p

        # Multiple variants — determine candidates
        if action.source_index is not None:
            s, e = geom._source_ranges[action.source_index]
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
            return gid, None
        if len(placeable) == 1:
            return gid, placeable[0]
        scores = self._delta_cost_from_placements(placeable)
        scores = scores.to(dtype=torch.float32, device=self.device).view(-1)
        return gid, placeable[int(torch.argmin(scores).item())]

    def _apply_resolved_placement(
        self,
        placement: GroupPlacement,
    ) -> None:
        gid = placement.group_id
        self._state.place(placement=placement)
        self._update_flow_port_pairs(updated_gid=gid)
        self._refresh_base_cost_snapshot(reset_terminal=True)

    def apply_dynamic_placement(self, placement: object) -> None:
        """Apply a pre-resolved dynamic placement object (DynamicPlacement-compatible)."""
        gid = placement.group_id
        if gid not in self.group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        self._apply_resolved_placement(placement)

    # ---- objective ----

    @staticmethod
    def _normalize_cost_breakdown(values: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in values.items():
            out[str(k)] = float(v)
        if "total" not in out:
            out["total"] = float(sum(v for kk, v in out.items() if kk != "total"))
        return out

    def _refresh_base_cost_snapshot(self, *, reset_terminal: bool) -> Dict[str, float]:
        base = self._normalize_cost_breakdown(self._reward.score_dict(self._state, weighted=True))
        cs = self._state.cost
        cs["base"] = base
        if bool(reset_terminal):
            cs["terminal"] = {"delta": {}, "total": 0.0}
            cs["finalized"] = False
            terminal_total = 0.0
        else:
            terminal = dict(cs.get("terminal", {}) or {})
            terminal_delta = dict(terminal.get("delta", {}) or {})
            terminal["delta"] = terminal_delta
            terminal_total = float(terminal.get("total", 0.0))
            cs["terminal"] = terminal
        cs["final"] = {"total": float(base["total"]) + float(terminal_total)}
        return base

    def _finalize_terminal_snapshot(self, *, failed: bool) -> float:
        cs = self._state.cost
        if bool(cs.get("finalized", False)):
            terminal = dict(cs.get("terminal", {}) or {})
            return float(terminal.get("total", 0.0))

        base_weighted = self._normalize_cost_breakdown(self._reward.score_dict(self._state, weighted=True))
        base_unweighted = self._normalize_cost_breakdown(self._reward.score_dict(self._state, weighted=False))
        delta = self._terminal.delta_dict(
            state=self._state,
            maps=self.get_maps(),
            reward_composer=self._reward,
            failed=bool(failed),
            base_scores_unweighted=base_unweighted,
        )
        delta_norm = {str(k): float(v) for k, v in delta.items()}
        delta_total = float(sum(delta_norm.values()))

        cs["base"] = base_weighted
        cs["terminal"] = {"delta": delta_norm, "total": delta_total}
        cs["final"] = {"total": float(base_weighted["total"]) + delta_total}
        cs["finalized"] = True
        return delta_total

    def cost(self) -> float:
        """Current objective value cached in state runtime."""
        final = self._state.cost.get("final", {}) if isinstance(self._state.cost, dict) else {}
        if "total" not in final:
            self._refresh_base_cost_snapshot(reset_terminal=not bool(self._state.cost.get("finalized", False)))
            final = self._state.cost.get("final", {})
        return float(final.get("total", 0.0))

    def failure_penalty(self) -> float:
        """Terminal-failure reward adjustment (negative)."""
        delta = self._finalize_terminal_snapshot(failed=True)
        return -float(delta) / float(self.reward_scale)

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
    # Placement interchange: ``group_placement.envs.interchange`` (export_placement /
    # import_placement).

    def _fail(self, reason: str) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """실패 처리 통합 헬퍼."""
        reward = self.failure_penalty()
        info: Dict[str, Any] = {"reason": reason}
        base_cost = float(self._state.cost.get("base", {}).get("total", 0.0))
        terminal_delta = float(self._state.cost.get("terminal", {}).get("total", 0.0))
        final_cost = float(self.cost())
        logger.warning(
            "fail: reason=%s remaining=%d cost=%.3f (base=%.3f + terminal=%.3f) reward=%.3f",
            reason,
            len(self._state.remaining),
            final_cost,
            base_cost,
            terminal_delta,
            reward,
        )

        return {}, float(reward), False, True, info

    def step_placement(
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
            return self._fail("gid_not_remaining")

        # Placeability validation — env never trusts external input
        if not self._state.placeable(placement=placement):
            self._state.step(apply=False)
            return self._fail("not_placeable")

        delta_cost = float(self._delta_cost_from_placements([placement])[0].item())

        self._state.step(apply=True, placement=placement)
        self._update_flow_port_pairs(updated_gid=gid)
        self._refresh_base_cost_snapshot(reset_terminal=True)

        reward = float(self._reward.to_reward(delta_cost))
        terminated = len(self._state.remaining) == 0
        truncated = self.max_steps is not None and self._state.step_count >= self.max_steps
        info: Dict[str, Any] = {"reason": "placed"}

        if terminated:
            terminal_delta = self._finalize_terminal_snapshot(failed=False)
            reward += -float(terminal_delta) / float(self.reward_scale)
        elif truncated:
            terminal_delta = self._finalize_terminal_snapshot(failed=True)
            reward += -float(terminal_delta) / float(self.reward_scale)

        if terminated or truncated:
            logger.info(
                "end: terminated=%s truncated=%s remaining=%d placed=%d step=%d cost=%.3f reason=placed reward=%.3f",
                terminated, truncated,
                len(self._state.remaining), len(self._state.placed),
                self._state.step_count, self.cost(), reward,
            )

        return {}, float(reward), bool(terminated), bool(truncated), info

    def step_action(
        self,
        action: EnvAction,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Place a single `EnvAction` — resolves variant then delegates to step_placement."""
        if not self._state.remaining:
            self._state.step(apply=False)
            return {}, 0.0, True, False, {"reason": "done"}

        try:
            gid_eff, _x_center, _y_center = self._normalize_action(action)
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

        return self.step_placement(placement)

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
        placement_order = options.get("placement_order", None)
        strict_initial = bool(options.get("strict_initial_placements", False))
        if initial_placements is not None:
            if not isinstance(initial_placements, dict):
                raise ValueError("reset(options): initial_placements must be a dict {gid: EnvAction}")
            if placement_order is not None:
                if not isinstance(placement_order, list):
                    raise ValueError("reset(options): placement_order must be a list of group ids")
                iter_gids: List[GroupId] = list(placement_order)
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
                action = initial_placements[gid]
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
                    msg = (
                        f"reset(options): not placeable gid={gid!r} "
                        f"x_center={action.x_center} y_center={action.y_center} vi={action.variant_index}"
                    )
                    if strict_initial:
                        raise RuntimeError(msg)
                    warnings.warn(f"{msg} — skipping", stacklevel=2)
                    continue
                self._apply_resolved_placement(placement)
        self._refresh_base_cost_snapshot(reset_terminal=True)
        if len(self._state.remaining) == 0 and len(self._state.placed) > 0:
            self._finalize_terminal_snapshot(failed=False)
        return {}, {}

    def step(self, action: EnvAction):
        """Gym-compatible step proxy for engine-only usage."""
        if not isinstance(action, EnvAction):
            raise TypeError(f"FactoryLayoutEnv.step expects EnvAction, got {type(action).__name__}")
        return self.step_action(action)


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
