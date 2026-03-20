from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from envs.env import FactoryLayoutEnv
from envs.placement.static import StaticSpec

GroupId = Union[int, str]
RectI = Tuple[int, int, int, int]  # (x0, y0, x1, y1) half-open


@dataclass(frozen=True)
class LoadedEnv:
    """Result of loading an environment from a JSON spec file."""

    env: FactoryLayoutEnv
    reset_kwargs: Dict[str, Any]


def _edges_to_adj(edges: List[List[Any]]) -> Dict[GroupId, Dict[GroupId, float]]:
    adj: Dict[GroupId, Dict[GroupId, float]] = {}
    for e in edges:
        if len(e) != 3:
            raise ValueError(f"flow edge must be [src, dst, weight], got: {e}")
        src, dst, w = e
        adj.setdefault(src, {})[dst] = float(w)
    return adj


def load_env(
    json_path: str,
    *,
    device: torch.device | None = None,
) -> LoadedEnv:
    """Load a FactoryLayoutEnv from a JSON spec file."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    grid = data["grid"]
    env_cfg = data["env"]
    groups_cfg: Dict[str, Dict[str, Any]] = data["groups"]

    grid_w = int(grid["width"])
    grid_h = int(grid["height"])

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def _to_int(v: Any) -> int:
        try:
            return int(round(float(v)))
        except Exception as e:
            raise ValueError(f"expected number, got {v!r}") from e

    group_specs: Dict[GroupId, StaticSpec] = {}
    for gid, g in groups_cfg.items():
        w = _to_int(g["width"])
        h = _to_int(g["height"])
        # IO offsets: JSON stores BL-relative coords directly.
        # Default to center (w/2, h/2) so that facilities without explicit port
        # definitions use the facility centroid, not the BL corner (0,0).
        ent_x = float(g.get("ent_rel_x", w / 2.0))
        ent_y = float(g.get("ent_rel_y", h / 2.0))
        exi_x = float(g.get("exi_rel_x", w / 2.0))
        exi_y = float(g.get("exi_rel_y", h / 2.0))
        zone_values_raw = g.get("zone_values", {})
        if not isinstance(zone_values_raw, dict):
            raise ValueError(f"groups.{gid}.zone_values must be an object")
        group_specs[gid] = StaticSpec(
            device=dev,
            id=gid,
            width=w,
            height=h,
            entries_rel=[(ent_x, ent_y)],
            exits_rel=[(exi_x, exi_y)],
            clearance_left_rel=_to_int(g.get("facility_clearance_left", 0)),
            clearance_right_rel=_to_int(g.get("facility_clearance_right", 0)),
            clearance_bottom_rel=_to_int(g.get("facility_clearance_bottom", 0)),
            clearance_top_rel=_to_int(g.get("facility_clearance_top", 0)),
            rotatable=bool(g.get("rotatable", True)),
            mirrorable=bool(g.get("mirrorable", True)),
            zone_values=dict(zone_values_raw),
        )

    flow_raw = data.get("flow", {})
    if isinstance(flow_raw, list):
        flow = _edges_to_adj(flow_raw)
    elif isinstance(flow_raw, dict):
        flow = {src: {dst: float(w) for dst, w in dsts.items()} for src, dsts in flow_raw.items()}
    else:
        raise ValueError("flow must be a dict adjacency or an edge list")

    zones = data.get("zones", {}) if isinstance(data.get("zones"), dict) else {}
    forbidden_areas: List[Dict[str, Any]] = list(zones.get("forbidden_areas", []))
    if not isinstance(forbidden_areas, list):
        raise ValueError("zones.forbidden_areas must be a list")
    for i, a in enumerate(forbidden_areas):
        if not isinstance(a, dict) or "rect" not in a:
            raise ValueError(f"zones.forbidden_areas[{i}] must contain key 'rect'")

    constraints_raw = zones.get("constraints", {})
    if not isinstance(constraints_raw, dict):
        raise ValueError("zones.constraints must be an object")

    valid_ops = {"<", "<=", ">", ">=", "==", "!="}
    valid_dtypes = {"float", "int", "bool"}
    zone_constraints: Dict[str, Dict[str, Any]] = {}
    for cname, raw in constraints_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"zones.constraints.{cname} must be an object")
        dtype = str(raw.get("dtype", "")).lower()
        if dtype not in valid_dtypes:
            raise ValueError(
                f"zones.constraints.{cname}.dtype must be one of {sorted(valid_dtypes)}, got {dtype!r}"
            )
        op = str(raw.get("op", ""))
        if op not in valid_ops:
            raise ValueError(
                f"zones.constraints.{cname}.op must be one of {sorted(valid_ops)}, got {op!r}"
            )
        if "default" not in raw:
            raise ValueError(f"zones.constraints.{cname}.default is required")
        default_value = raw["default"]
        areas = raw.get("areas", [])
        if not isinstance(areas, list):
            raise ValueError(f"zones.constraints.{cname}.areas must be a list")
        norm_areas: List[Dict[str, Any]] = []
        for i, a in enumerate(areas):
            if not isinstance(a, dict) or "rect" not in a or "value" not in a:
                raise ValueError(
                    f"zones.constraints.{cname}.areas[{i}] must contain keys 'rect' and 'value'"
                )
            rect = a["rect"]
            if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
                raise ValueError(f"zones.constraints.{cname}.areas[{i}].rect must be [x0,y0,x1,y1]")
            norm_areas.append({"rect": list(rect), "value": a["value"]})
        zone_constraints[str(cname)] = {
            "dtype": dtype,
            "op": op,
            "default": default_value,
            "areas": norm_areas,
        }

    env = FactoryLayoutEnv(
        grid_width=grid_w,
        grid_height=grid_h,
        group_specs=group_specs,
        group_flow=flow,
        forbidden_areas=forbidden_areas,
        zone_constraints=zone_constraints,
        reward_scale=float(env_cfg.get("reward_scale", 100.0)),
        penalty_weight=float(env_cfg.get("penalty_weight", 50000.0)),
    )

    reset_cfg = data.get("reset", {})
    reset_kwargs: Dict[str, Any] = {}
    if "initial_positions" in reset_cfg and reset_cfg["initial_positions"] is not None:
        ip = {}
        for gid, pose in reset_cfg["initial_positions"].items():
            if not (isinstance(pose, list) and len(pose) == 3):
                raise ValueError(f"initial_positions[{gid}] must be [x, y, rot], got: {pose}")
            try:
                x_bl = int(round(float(pose[0])))
                y_bl = int(round(float(pose[1])))
            except Exception as e:
                raise ValueError(f"initial_positions[{gid}]: x/y must be numbers, got: {pose}") from e
            ip[gid] = (x_bl, y_bl, int(pose[2]))
        reset_kwargs["initial_positions"] = ip
    if "remaining_order" in reset_cfg and reset_cfg["remaining_order"] is not None:
        reset_kwargs["remaining_order"] = list(reset_cfg["remaining_order"])

    return LoadedEnv(env=env, reset_kwargs=reset_kwargs)
