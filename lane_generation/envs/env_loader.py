from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from group_placement.envs.env_loader import load_env as load_group_env
from lane_generation.agents.placement.greedy import LaneAdapter, LaneAdapterConfig

from .env import FactoryLaneEnv
from .state import RoutingConfig
from .state import LaneFlowSpec, PortGroup, PortSelector, PortSpec


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
    flow_lane_widths: Optional[Dict[Tuple[str, str], float]] = None,
    flow_reverse_allow: Optional[Dict[Tuple[str, str], bool]] = None,
    flow_merge_allow: Optional[Dict[Tuple[str, str], bool]] = None,
    port_max_flows: Optional[Dict[str, int]] = None,
    port_groups: Optional[Dict[str, PortGroup]] = None,
    extra_port_specs: Optional[Dict[str, PortSpec]] = None,
) -> LoadedLaneEnv:
    """Load a lane-generation environment.

    ``flow_lane_widths`` maps ``(src_gid, dst_gid)`` → physical lane width in
    grid-cell units (default 1.0 when not specified).  Lane widths must be in
    ``(0, 1]``; values > 1.0 are reserved for future multi-cell lane support.

    ``flow_reverse_allow`` / ``flow_merge_allow`` map
    ``(src_gid, dst_gid)`` → per-flow override.  ``None`` (absent) falls back
    to the global ``RoutingConfig`` default.
    """
    dev = torch.device(device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    payload = _load_group_placement(group_placement)

    loaded_group = load_group_env(env_json, device=dev, backend_selection=backend_selection)
    grid = payload.get("grid", {}) or {}
    grid_size_mm = float(grid.get("grid_size_mm", loaded_group.grid_size_mm))
    _restore_group_env_with_group_placement(loaded_group.env, payload, grid_size_mm=grid_size_mm)

    gstate = loaded_group.env.get_state()
    _widths: Dict[Tuple[str, str], float] = flow_lane_widths or {}
    _rev: Dict[Tuple[str, str], bool] = flow_reverse_allow or {}
    _mrg: Dict[Tuple[str, str], bool] = flow_merge_allow or {}
    _maxflows: Dict[str, int] = port_max_flows or {}

    port_catalog: Dict[str, PortSpec] = {}
    exit_ids_by_gid: Dict[str, Tuple[str, ...]] = {}
    entry_ids_by_gid: Dict[str, Tuple[str, ...]] = {}
    for gid, placement in gstate.placements.items():
        if placement is None:
            continue
        g = str(gid)
        ex_ids: List[str] = []
        for i, xy in enumerate(_ports_from_placement(placement, kind="exit_points")):
            pid = f"{g}.ex.{i}"
            port_catalog[pid] = PortSpec(
                port_id=pid, gid=g, xy=(int(xy[0]), int(xy[1])),
                kind="exit", max_flows=int(_maxflows.get(pid, 0)),
            )
            ex_ids.append(pid)
        exit_ids_by_gid[g] = tuple(ex_ids)
        en_ids: List[str] = []
        for i, xy in enumerate(_ports_from_placement(placement, kind="entry_points")):
            pid = f"{g}.en.{i}"
            port_catalog[pid] = PortSpec(
                port_id=pid, gid=g, xy=(int(xy[0]), int(xy[1])),
                kind="entry", max_flows=int(_maxflows.get(pid, 0)),
            )
            en_ids.append(pid)
        entry_ids_by_gid[g] = tuple(en_ids)
    if extra_port_specs:
        for pid, spec in extra_port_specs.items():
            port_catalog[str(pid)] = spec

    flow_specs: List[LaneFlowSpec] = []
    for src_gid, dsts in loaded_group.env.group_flow.items():
        src_p = gstate.placements.get(src_gid, None)
        if src_p is None:
            continue
        src_pids = exit_ids_by_gid.get(str(src_gid), ())
        if not src_pids:
            continue
        for dst_gid, w in dsts.items():
            dst_p = gstate.placements.get(dst_gid, None)
            if dst_p is None:
                continue
            dst_pids = entry_ids_by_gid.get(str(dst_gid), ())
            if not dst_pids:
                continue
            key = (str(src_gid), str(dst_gid))
            lw = float(_widths.get(key, 1.0))
            ra = _rev.get(key)
            ma = _mrg.get(key)
            flow_specs.append(
                LaneFlowSpec(
                    src=PortSelector(gid=str(src_gid), port_ids=src_pids),
                    dst=PortSelector(gid=str(dst_gid), port_ids=dst_pids),
                    weight=float(w),
                    reverse_allow=ra,
                    merge_allow=ma,
                    lane_width=lw,
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
        port_specs=port_catalog,
        port_groups=dict(port_groups or {}),
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
