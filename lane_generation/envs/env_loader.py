from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from group_placement.envs.env_loader import load_env as load_group_env

from .adapter import LaneAdapter, LaneAdapterConfig
from .env import FactoryLaneEnv
from .state import RoutingConfig
from .state import LaneFlowSpec


@dataclass(frozen=True)
class LoadedLaneEnv:
    env: FactoryLaneEnv
    reset_kwargs: Dict[str, Any]
    env_json: str
    group_placement: Dict[str, Any]


def _load_group_placement(payload_or_path: Mapping[str, Any] | str) -> Dict[str, Any]:
    if isinstance(payload_or_path, str):
        with open(payload_or_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = dict(payload_or_path)
    if "group_placement" in obj:
        obj = obj["group_placement"]
    if not isinstance(obj, dict):
        raise ValueError("group placement payload must be a dict")
    return obj


def _restore_group_env_with_group_placement(env, payload: Dict[str, Any], *, grid_size_mm: float) -> None:
    env.reset()
    state = env.get_state()
    placements = payload.get("placements", []) or []
    order = payload.get("placed_order", []) or []
    by_gid = {str(p.get("gid")): p for p in placements if isinstance(p, dict)}

    ordered: List[Dict[str, Any]] = []
    used = set()
    for gid in order:
        g = str(gid)
        p = by_gid.get(g)
        if p is not None:
            ordered.append(p)
            used.add(g)
    for p in placements:
        if not isinstance(p, dict):
            continue
        gid = str(p.get("gid"))
        if gid not in used:
            ordered.append(p)

    for p in ordered:
        gid = str(p.get("gid"))
        if gid not in env.group_specs:
            continue
        spec = env.group_specs[gid]
        vi = int(p.get("variant_index", 0))
        x_bl = int(round(float(p.get("x_bl_mm", 0.0)) / float(grid_size_mm)))
        y_bl = int(round(float(p.get("y_bl_mm", 0.0)) / float(grid_size_mm)))
        placement = spec.build_placement(variant_index=vi, x_bl=x_bl, y_bl=y_bl)
        state.place(placement=placement)


def _ports_from_placement(placement: object, *, kind: str) -> List[tuple[int, int]]:
    pts = getattr(placement, kind, None)
    out: List[tuple[int, int]] = []
    if isinstance(pts, list):
        for p in pts:
            out.append((int(round(float(p[0]))), int(round(float(p[1])))))
    if len(out) > 0:
        return out
    cx = int(round(0.5 * (float(getattr(placement, "min_x")) + float(getattr(placement, "max_x")))))
    cy = int(round(0.5 * (float(getattr(placement, "min_y")) + float(getattr(placement, "max_y")))))
    return [(cx, cy)]


def load_lane_env(
    *,
    env_json: str,
    group_placement: Mapping[str, Any] | str,
    device: Optional[torch.device] = None,
    backend_selection: str = "benchmark",
    flow_ordering: str = "weight_desc",
    adapter_config: Optional[LaneAdapterConfig] = None,
    routing_config: Optional[RoutingConfig] = None,
    reward_scale: float = 100.0,
    penalty_weight: float = 50000.0,
) -> LoadedLaneEnv:
    dev = torch.device(device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    payload = _load_group_placement(group_placement)

    loaded_group = load_group_env(env_json, device=dev, backend_selection=backend_selection)
    grid = payload.get("grid", {}) or {}
    grid_size_mm = float(grid.get("grid_size_mm", loaded_group.grid_size_mm))
    _restore_group_env_with_group_placement(loaded_group.env, payload, grid_size_mm=grid_size_mm)

    gstate = loaded_group.env.get_state()
    flow_specs: List[LaneFlowSpec] = []
    for src_gid, dsts in loaded_group.env.group_flow.items():
        src_p = gstate.placements.get(src_gid, None)
        if src_p is None:
            continue
        src_ports = tuple(_ports_from_placement(src_p, kind="exit_points"))
        for dst_gid, w in dsts.items():
            dst_p = gstate.placements.get(dst_gid, None)
            if dst_p is None:
                continue
            dst_ports = tuple(_ports_from_placement(dst_p, kind="entry_points"))
            flow_specs.append(
                LaneFlowSpec(
                    src_gid=str(src_gid),
                    dst_gid=str(dst_gid),
                    weight=float(w),
                    src_ports=src_ports,
                    dst_ports=dst_ports,
                )
            )

    blocked_static = (loaded_group.env.get_maps().static_invalid | loaded_group.env.get_maps().occ_invalid).to(
        device=dev,
        dtype=torch.bool,
    )

    lane_env = FactoryLaneEnv(
        grid_width=int(loaded_group.env.grid_width),
        grid_height=int(loaded_group.env.grid_height),
        blocked_static=blocked_static,
        flows=flow_specs,
        device=dev,
        flow_ordering=flow_ordering,
        routing_config=routing_config,
        reward_scale=float(reward_scale),
        penalty_weight=float(penalty_weight),
    )
    lane_env.set_adapter(LaneAdapter(config=adapter_config or LaneAdapterConfig()))

    return LoadedLaneEnv(
        env=lane_env,
        reset_kwargs={},
        env_json=str(env_json),
        group_placement=payload,
    )
