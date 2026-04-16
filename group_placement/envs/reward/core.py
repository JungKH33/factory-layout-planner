from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

import torch

from .area import AreaReward, GridOccupancyReward
from .flow import FlowCollisionReward, FlowReward

if TYPE_CHECKING:
    from ..action_space import ActionSpace
    from ..state.base import EnvState


def _require(x: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if x is None:
        raise ValueError(f"{name} is required but None")
    return x


@dataclass
class RewardComposer:
    components: Dict[str, object]
    weights: Dict[str, float]
    reward_scale: float = 100.0
    group_specs: Optional[Dict["str | int", object]] = None

    def __post_init__(self) -> None:
        if float(self.reward_scale) <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")

    def to_reward(self, value: object):
        if torch.is_tensor(value):
            return -value.to(dtype=torch.float32) / float(self.reward_scale)
        return -float(value) / float(self.reward_scale)

    def find_component(self, cls: Type[object]) -> Tuple[Optional[str], Optional[object]]:
        for name, comp in self.components.items():
            if isinstance(comp, cls):
                return name, comp
        return None, None

    def required(self) -> set[str]:
        needed: set[str] = set()
        for comp in self.components.values():
            if comp is not None and hasattr(comp, "required"):
                needed |= set(comp.required())
        return needed

    def _port_span_tensors(
        self,
        gids: list,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build (exit_k, entry_k) int tensors from group_specs.

        Returns (None, None) when all facilities use span=1 (fast-path).
        """
        if self.group_specs is None:
            return None, None
        n = len(gids)
        if n == 0:
            return None, None
        exit_k = torch.ones((n,), dtype=torch.int32, device=device)
        entry_k = torch.ones((n,), dtype=torch.int32, device=device)
        any_non_one = False
        for i, gid in enumerate(gids):
            spec = self.group_specs.get(gid)
            if spec is None:
                continue
            ek = int(getattr(spec, "exit_port_span", 1))
            ik = int(getattr(spec, "entry_port_span", 1))
            exit_k[i] = ek
            entry_k[i] = ik
            if ek != 1 or ik != 1:
                any_non_one = True
        if not any_non_one:
            return None, None
        return exit_k, entry_k

    @staticmethod
    def _to_gid_edge_key(
        *,
        raw_key: object,
        edge_key_kind: str,
        placed_nodes: list,
    ) -> Optional[tuple[str, str]]:
        if edge_key_kind == "row_index":
            i: Optional[int] = None
            j: Optional[int] = None
            if isinstance(raw_key, str) and "->" in raw_key:
                lhs, rhs = raw_key.split("->", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                if lhs.isdigit() and rhs.isdigit():
                    i = int(lhs)
                    j = int(rhs)
            elif isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
                if isinstance(raw_key[0], int) and isinstance(raw_key[1], int):
                    i = int(raw_key[0])
                    j = int(raw_key[1])
            if i is None or j is None:
                return None
            if i < 0 or j < 0 or i >= len(placed_nodes) or j >= len(placed_nodes):
                return None
            return str(placed_nodes[i]), str(placed_nodes[j])

        if isinstance(raw_key, str) and "->" in raw_key:
            lhs, rhs = raw_key.split("->", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs and rhs:
                return lhs, rhs
            return None
        if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
            return str(raw_key[0]), str(raw_key[1])
        return None

    def _normalize_component_meta(
        self,
        *,
        comp_meta: Dict[str, Any],
        placed_nodes: list,
    ) -> Dict[str, Any]:
        meta = dict(comp_meta or {})
        raw_edges = meta.get("edges", None)
        if not isinstance(raw_edges, dict):
            return meta

        edge_key_kind = str(meta.get("edge_key_kind", "gid"))
        edges: Dict[str, Dict[str, Any]] = {}
        for raw_key, raw_edge in raw_edges.items():
            parsed = self._to_gid_edge_key(
                raw_key=raw_key,
                edge_key_kind=edge_key_kind,
                placed_nodes=placed_nodes,
            )
            if parsed is None:
                continue
            src_gid, dst_gid = parsed
            edge_key = f"{src_gid}->{dst_gid}"
            edge_data = dict(raw_edge or {}) if isinstance(raw_edge, dict) else {}
            edge_data["src"] = src_gid
            edge_data["dst"] = dst_gid
            edges[edge_key] = edge_data

        meta["edge_key_kind"] = "gid"
        meta["edges"] = edges
        return meta

    def _score_component(
        self,
        *,
        comp: object,
        placed_count: int,
        placed_entries: torch.Tensor,
        placed_exits: torch.Tensor,
        placed_entries_mask: torch.Tensor,
        placed_exits_mask: torch.Tensor,
        flow_w: torch.Tensor,
        exit_k: Optional[torch.Tensor],
        entry_k: Optional[torch.Tensor],
        cur_min_x: float,
        cur_max_x: float,
        cur_min_y: float,
        cur_max_y: float,
        route_blocked: Optional[torch.Tensor],
        placed_cell_occupied: Optional[torch.Tensor],
        return_meta: bool = False,
    ):
        device = placed_entries.device
        if isinstance(comp, FlowReward):
            return comp.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
                return_meta=return_meta,
            )
        if isinstance(comp, FlowCollisionReward):
            return comp.score(
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                route_blocked=route_blocked,
                exit_k=exit_k,
                entry_k=entry_k,
                return_meta=return_meta,
            )
        if isinstance(comp, AreaReward):
            return comp.score(
                placed_count=placed_count,
                min_x=torch.tensor(float(cur_min_x), dtype=torch.float32, device=device),
                max_x=torch.tensor(float(cur_max_x), dtype=torch.float32, device=device),
                min_y=torch.tensor(float(cur_min_y), dtype=torch.float32, device=device),
                max_y=torch.tensor(float(cur_max_y), dtype=torch.float32, device=device),
                return_meta=return_meta,
            )
        if isinstance(comp, GridOccupancyReward):
            return comp.score(placed_cell_occupied=placed_cell_occupied, return_meta=return_meta)
        raise TypeError(f"unsupported reward component type: {type(comp).__name__}")

    def score_dict(
        self,
        state: "EnvState",
        *,
        weighted: bool = True,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
        (
            placed_nodes,
            placed_entries,
            placed_exits,
            placed_entries_mask,
            placed_exits_mask,
        ) = state.io_tensors()
        placed_count = len(placed_nodes)
        cur_min_x, cur_max_x, cur_min_y, cur_max_y = state.placed_bbox()
        flow_w = state.build_flow_w()
        exit_k, entry_k = self._port_span_tensors(placed_nodes, placed_entries.device)

        out: Dict[str, float] = {}
        meta_out: Dict[str, Dict[str, Any]] = {}
        total = 0.0
        for name, comp in self.components.items():
            if comp is None:
                continue
            val_raw = self._score_component(
                comp=comp,
                placed_count=placed_count,
                placed_entries=placed_entries,
                placed_exits=placed_exits,
                placed_entries_mask=placed_entries_mask,
                placed_exits_mask=placed_exits_mask,
                flow_w=flow_w,
                exit_k=exit_k,
                entry_k=entry_k,
                cur_min_x=cur_min_x,
                cur_max_x=cur_max_x,
                cur_min_y=cur_min_y,
                cur_max_y=cur_max_y,
                route_blocked=route_blocked,
                placed_cell_occupied=placed_cell_occupied,
                return_meta=return_meta,
            )
            if return_meta:
                if not (isinstance(val_raw, tuple) and len(val_raw) == 2):
                    raise TypeError(f"{type(comp).__name__}.score(return_meta=True) must return (score, meta)")
                val_t, comp_meta = val_raw
                meta_out[str(name)] = self._normalize_component_meta(
                    comp_meta=dict(comp_meta or {}),
                    placed_nodes=placed_nodes,
                )
            else:
                val_t = val_raw
            val = float(val_t.item())
            if weighted:
                val *= float(self.weights.get(name, 1.0))
            out[name] = val
            total += val
        out["total"] = float(total)
        if return_meta:
            return out, meta_out
        return out

    def score(
        self,
        state: "EnvState",
        *,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
        scores = self.score_dict(
            state,
            weighted=True,
            route_blocked=route_blocked,
            placed_cell_occupied=placed_cell_occupied,
            return_meta=return_meta,
        )
        meta_out: Optional[Dict[str, Dict[str, Any]]] = None
        if return_meta:
            if not (isinstance(scores, tuple) and len(scores) == 2):
                raise TypeError("score_dict(return_meta=True) must return (scores, meta)")
            scores, meta_out = scores
        return torch.tensor(
            float(scores.get("total", 0.0)),
            dtype=torch.float32,
            device=state.device,
        ) if not return_meta else (
            torch.tensor(
                float(scores.get("total", 0.0)),
                dtype=torch.float32,
                device=state.device,
            ),
            meta_out or {},
        )

    def delta_batch(
        self,
        state: "EnvState",
        *,
        gid: Optional["str | int"] = None,
        entry_points: Optional[torch.Tensor] = None,
        exit_points: Optional[torch.Tensor] = None,
        entry_mask: Optional[torch.Tensor] = None,
        exit_mask: Optional[torch.Tensor] = None,
        min_x: Optional[torch.Tensor] = None,
        max_x: Optional[torch.Tensor] = None,
        min_y: Optional[torch.Tensor] = None,
        max_y: Optional[torch.Tensor] = None,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
        """Compute incremental cost from keyword feature tensors.

        Same logic as ``delta()`` but reads features from kwargs
        instead of an ActionSpace object.
        """
        (
            placed_nodes,
            placed_entries,
            placed_exits,
            placed_entries_mask,
            placed_exits_mask,
        ) = state.io_tensors()
        placed_count = len(placed_nodes)
        cur_min_x, cur_max_x, cur_min_y, cur_max_y = state.placed_bbox()
        if gid is None:
            raise ValueError("RewardComposer.delta_batch requires explicit gid")
        w_out, w_in = state.build_delta_flow_weights_for(gid)

        m = 0
        device = placed_entries.device
        for t in (entry_points, exit_points, min_x, max_x, min_y, max_y):
            if t is not None:
                m = int(t.shape[0])
                device = t.device
                break

        total = torch.zeros((m,), dtype=torch.float32, device=device)
        meta_out: Dict[str, Dict[str, Any]] = {}
        has_flow_like = any(
            isinstance(comp, (FlowReward, FlowCollisionReward))
            for comp in self.components.values()
            if comp is not None
        )
        has_area_like = any(
            isinstance(comp, (AreaReward, GridOccupancyReward))
            for comp in self.components.values()
            if comp is not None
        )

        if has_flow_like:
            entry_points = _require(entry_points, "entry_points")
            exit_points = _require(exit_points, "exit_points")

        # Build port span tensors for candidate and placed facilities
        t_exit_k, t_entry_k = self._port_span_tensors(placed_nodes, device)
        c_exit_k = 1
        c_entry_k = 1
        if self.group_specs is not None and gid is not None:
            c_spec = self.group_specs.get(gid)
            if c_spec is not None:
                c_exit_k = int(getattr(c_spec, "exit_port_span", 1))
                c_entry_k = int(getattr(c_spec, "entry_port_span", 1))

        if has_area_like:
            min_x = _require(min_x, "min_x")
            max_x = _require(max_x, "max_x")
            min_y = _require(min_y, "min_y")
            max_y = _require(max_y, "max_y")

        for name, comp in self.components.items():
            if comp is None:
                continue
            w = float(self.weights.get(name, 1.0))
            if isinstance(comp, FlowReward):
                delta_raw = comp.delta(
                    placed_entries=placed_entries,
                    placed_exits=placed_exits,
                    placed_entries_mask=placed_entries_mask,
                    placed_exits_mask=placed_exits_mask,
                    w_out=w_out,
                    w_in=w_in,
                    candidate_entries=entry_points,
                    candidate_exits=exit_points,
                    candidate_entries_mask=entry_mask,
                    candidate_exits_mask=exit_mask,
                    c_exit_k=c_exit_k,
                    c_entry_k=c_entry_k,
                    t_entry_k=t_entry_k,
                    t_exit_k=t_exit_k,
                    return_meta=return_meta,
                )
                if return_meta:
                    if not (isinstance(delta_raw, tuple) and len(delta_raw) == 2):
                        raise TypeError(f"{type(comp).__name__}.delta(return_meta=True) must return (delta, meta)")
                    delta_t, comp_meta = delta_raw
                    meta_out[str(name)] = dict(comp_meta or {})
                else:
                    delta_t = delta_raw
                total = total + w * delta_t
                continue
            if isinstance(comp, FlowCollisionReward):
                delta_raw = comp.delta(
                    placed_entries=placed_entries,
                    placed_exits=placed_exits,
                    placed_entries_mask=placed_entries_mask,
                    placed_exits_mask=placed_exits_mask,
                    w_out=w_out,
                    w_in=w_in,
                    candidate_entries=entry_points,
                    candidate_exits=exit_points,
                    candidate_entries_mask=entry_mask,
                    candidate_exits_mask=exit_mask,
                    route_blocked=route_blocked,
                    c_exit_k=c_exit_k,
                    c_entry_k=c_entry_k,
                    t_entry_k=t_entry_k,
                    t_exit_k=t_exit_k,
                    return_meta=return_meta,
                )
                if return_meta:
                    if not (isinstance(delta_raw, tuple) and len(delta_raw) == 2):
                        raise TypeError(f"{type(comp).__name__}.delta(return_meta=True) must return (delta, meta)")
                    delta_t, comp_meta = delta_raw
                    meta_out[str(name)] = dict(comp_meta or {})
                else:
                    delta_t = delta_raw
                total = total + w * delta_t
                continue
            if isinstance(comp, AreaReward):
                delta_raw = comp.delta(
                    placed_count=placed_count,
                    cur_min_x=cur_min_x,
                    cur_max_x=cur_max_x,
                    cur_min_y=cur_min_y,
                    cur_max_y=cur_max_y,
                    candidate_min_x=min_x,
                    candidate_max_x=max_x,
                    candidate_min_y=min_y,
                    candidate_max_y=max_y,
                    return_meta=return_meta,
                )
                if return_meta:
                    if not (isinstance(delta_raw, tuple) and len(delta_raw) == 2):
                        raise TypeError(f"{type(comp).__name__}.delta(return_meta=True) must return (delta, meta)")
                    delta_t, comp_meta = delta_raw
                    meta_out[str(name)] = dict(comp_meta or {})
                else:
                    delta_t = delta_raw
                total = total + w * delta_t
                continue
            if isinstance(comp, GridOccupancyReward):
                delta_raw = comp.delta(
                    placed_cell_occupied=placed_cell_occupied,
                    candidate_min_x=min_x,
                    candidate_max_x=max_x,
                    candidate_min_y=min_y,
                    candidate_max_y=max_y,
                    return_meta=return_meta,
                )
                if return_meta:
                    if not (isinstance(delta_raw, tuple) and len(delta_raw) == 2):
                        raise TypeError(f"{type(comp).__name__}.delta(return_meta=True) must return (delta, meta)")
                    delta_t, comp_meta = delta_raw
                    meta_out[str(name)] = dict(comp_meta or {})
                else:
                    delta_t = delta_raw
                total = total + w * delta_t
                continue
            raise TypeError(f"unsupported reward component type: {type(comp).__name__}")
        if return_meta:
            return total, meta_out
        return total

    def delta(
        self,
        state: "EnvState",
        action_space: "ActionSpace",
        *,
        gid: Optional["str | int"] = None,
        route_blocked: Optional[torch.Tensor] = None,
        placed_cell_occupied: Optional[torch.Tensor] = None,
        return_meta: bool = False,
    ):
        """Compute incremental cost for each candidate in *action_space*.

        Delegates to ``delta_batch`` using ActionSpace fields.
        """
        return self.delta_batch(
            state,
            gid=gid,
            entry_points=action_space.entry_points,
            exit_points=action_space.exit_points,
            entry_mask=action_space.entry_mask,
            exit_mask=action_space.exit_mask,
            min_x=action_space.min_x,
            max_x=action_space.max_x,
            min_y=action_space.min_y,
            max_y=action_space.max_y,
            route_blocked=route_blocked,
            placed_cell_occupied=placed_cell_occupied,
            return_meta=return_meta,
        )
