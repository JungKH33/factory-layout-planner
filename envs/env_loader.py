from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from envs.action import EnvAction
from envs.env import FactoryLayoutEnv
from envs.placement.base import GroupSpec
from envs.placement.static import StaticRectSpec, StaticIrregularSpec

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
    backend_selection: str = "static",
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

    def _parse_ports(
        obj: Dict[str, Any], default_w: float, default_h: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Parse entries_rel / exits_rel from a group or variant dict."""
        entries_raw = obj.get("entries_rel")
        exits_raw = obj.get("exits_rel")
        if entries_raw is not None:
            entry_points = [(float(p[0]), float(p[1])) for p in entries_raw]
        else:
            ent_x = float(obj.get("ent_rel_x", default_w / 2.0))
            ent_y = float(obj.get("ent_rel_y", default_h / 2.0))
            entry_points = [(ent_x, ent_y)]
        if exits_raw is not None:
            exit_points = [(float(p[0]), float(p[1])) for p in exits_raw]
        else:
            exi_x = float(obj.get("exi_rel_x", default_w / 2.0))
            exi_y = float(obj.get("exi_rel_y", default_h / 2.0))
            exit_points = [(exi_x, exi_y)]
        return entry_points, exit_points

    def _parse_clearance_lrtb(obj: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        """Parse clearance LRTB from a group or variant dict."""
        cl_lrtb_raw = obj.get("clearance_lrtb")
        cl_uniform_raw = obj.get("clearance")
        if cl_lrtb_raw is not None:
            return (
                _to_int(cl_lrtb_raw[0]), _to_int(cl_lrtb_raw[1]),
                _to_int(cl_lrtb_raw[2]), _to_int(cl_lrtb_raw[3]),
            )
        elif cl_uniform_raw is not None:
            n = _to_int(cl_uniform_raw)
            return (n, n, n, n)
        return None

    group_specs: Dict[GroupId, GroupSpec] = {}
    for gid, g in groups_cfg.items():
        variants_raw = g.get("variants")
        group_type = str(g.get("type", "rect")).lower()

        # --- parse top-level width/height/ports ---
        # When "variants" is present, top-level width/height are optional
        # (defaults to first variant's values).
        if variants_raw is not None:
            if not isinstance(variants_raw, list) or len(variants_raw) == 0:
                raise ValueError(f"groups.{gid}.variants must be a non-empty list")
            first_v = variants_raw[0]
            if group_type == "rect":
                w = _to_int(g.get("width", first_v["width"]))
                h = _to_int(g.get("height", first_v["height"]))
            else:
                # irregular: width/height come from polygon rasterization;
                # use placeholder values (overwritten in __post_init__)
                w = _to_int(g.get("width", first_v.get("width", 1)))
                h = _to_int(g.get("height", first_v.get("height", 1)))
        else:
            w = _to_int(g["width"])
            h = _to_int(g["height"])

        entries_rel, exits_rel = _parse_ports(g, float(w), float(h))

        zone_values_raw = g.get("zone_values", {})
        if not isinstance(zone_values_raw, dict):
            raise ValueError(f"groups.{gid}.zone_values must be an object")

        # --- clearance: unified parsing (priority: polygon > lrtb > uniform) ---
        clearance_kwargs: Dict[str, Any] = {}
        irregular_kwargs: Dict[str, Any] = {}
        cl_polygon_raw = g.get("clearance_polygon")
        if cl_polygon_raw is not None:
            irregular_kwargs["clearance_polygon"] = [
                (float(p[0]), float(p[1])) for p in cl_polygon_raw
            ]
        top_cl = _parse_clearance_lrtb(g)
        if top_cl is not None:
            clearance_kwargs["clearance_lrtb_rel"] = top_cl

        # --- parse _variant_defs if "variants" key is present ---
        variant_defs: Optional[List[Dict[str, Any]]] = None
        if variants_raw is not None:
            variant_defs = []
            for vi, v in enumerate(variants_raw):
                if not isinstance(v, dict):
                    raise ValueError(f"groups.{gid}.variants[{vi}] must be an object")

                if group_type == "rect":
                    vw = _to_int(v["width"])
                    vh = _to_int(v["height"])
                elif group_type == "irregular":
                    # width/height computed from polygon; use placeholder
                    vw = _to_int(v.get("width", 1))
                    vh = _to_int(v.get("height", 1))
                else:
                    raise ValueError(f"groups.{gid}: unknown type {group_type!r}")

                v_entries, v_exits = _parse_ports(v, float(vw), float(vh))
                v_cl = _parse_clearance_lrtb(v)

                vdef: Dict[str, Any] = {
                    "width": vw,
                    "height": vh,
                    "entries_rel": v_entries,
                    "exits_rel": v_exits,
                    "rotatable": bool(v.get("rotatable", g.get("rotatable", True))),
                    "mirrorable": bool(v.get("mirrorable", g.get("mirrorable", True))),
                }
                if v_cl is not None:
                    vdef["clearance_lrtb_rel"] = v_cl

                # irregular-specific: body_polygon and clearance_polygon
                if group_type == "irregular":
                    bp_raw = v.get("body_polygon")
                    if bp_raw is None:
                        raise ValueError(
                            f"groups.{gid}.variants[{vi}]: type='irregular' requires 'body_polygon'"
                        )
                    vdef["body_polygon"] = [(float(p[0]), float(p[1])) for p in bp_raw]
                    cp_raw = v.get("clearance_polygon")
                    if cp_raw is not None:
                        vdef["clearance_polygon"] = [(float(p[0]), float(p[1])) for p in cp_raw]

                variant_defs.append(vdef)

            # Update top-level entry_points/exit_points to first variant for backward compat
            entries_rel = variant_defs[0]["entries_rel"]
            exits_rel = variant_defs[0]["exits_rel"]
            if group_type == "rect":
                w = _to_int(variant_defs[0]["width"])
                h = _to_int(variant_defs[0]["height"])

        common_kwargs = dict(
            device=dev,
            id=gid,
            width=w,
            height=h,
            entries_rel=entries_rel,
            exits_rel=exits_rel,
            rotatable=bool(g.get("rotatable", True)),
            mirrorable=bool(g.get("mirrorable", True)),
            zone_values=dict(zone_values_raw),
            _entry_port_mode=str(g.get("entry_port_mode", "min")),
            _exit_port_mode=str(g.get("exit_port_mode", "min")),
            _variant_defs=variant_defs,
            **clearance_kwargs,
        )

        if group_type == "rect":
            group_specs[gid] = StaticRectSpec(**common_kwargs)
        elif group_type == "irregular":
            body_polygon_raw = g.get("body_polygon")
            if body_polygon_raw is None and variants_raw is None:
                raise ValueError(f"groups.{gid}: type='irregular' requires a 'body_polygon' field")
            if body_polygon_raw is not None:
                body_polygon = [(float(p[0]), float(p[1])) for p in body_polygon_raw]
            else:
                # Use first variant's polygon as canonical
                body_polygon = variant_defs[0]["body_polygon"]
            group_specs[gid] = StaticIrregularSpec(
                **common_kwargs,
                **irregular_kwargs,
                body_polygon=body_polygon,
            )
        else:
            raise ValueError(f"groups.{gid}: unknown type {group_type!r} (expected 'rect' or 'irregular')")

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
        device=device,
        reward_scale=float(env_cfg.get("reward_scale", 100.0)),
        penalty_weight=float(env_cfg.get("penalty_weight", 50000.0)),
        backend_selection=backend_selection,
    )

    reset_cfg = data.get("reset", {})
    reset_kwargs: Dict[str, Any] = {}
    if "initial_placements" in reset_cfg and reset_cfg["initial_placements"] is not None:
        ip = {}
        for gid, pose in reset_cfg["initial_placements"].items():
            if not (isinstance(pose, list) and len(pose) in (2, 3, 4)):
                raise ValueError(
                    f"initial_placements[{gid}] must be [x_center, y_center], "
                    f"[x_center, y_center, variant_index], or "
                    f"[x_center, y_center, variant_index, source_index], got: {pose}"
                )
            try:
                x_center = float(pose[0])
                y_center = float(pose[1])
            except Exception as e:
                raise ValueError(f"initial_placements[{gid}]: x_center/y_center must be numbers, got: {pose}") from e
            vi = int(pose[2]) if len(pose) > 2 and pose[2] is not None else None
            si = int(pose[3]) if len(pose) > 3 and pose[3] is not None else None
            ip[gid] = EnvAction(group_id=gid, x_center=x_center, y_center=y_center, variant_index=vi, source_index=si)
        reset_kwargs["initial_placements"] = ip
    if "remaining_order" in reset_cfg and reset_cfg["remaining_order"] is not None:
        reset_kwargs["remaining_order"] = list(reset_cfg["remaining_order"])

    return LoadedEnv(env=env, reset_kwargs=reset_kwargs)
