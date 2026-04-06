from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from group_placement.envs.env_loader import LoadedEnv, load_env
from lane_generation.output import routes_to_dict
from lane_generation.pathfinder import RoutePlanner
from pipeline.schema import (
    GroupLaneGenerationArtifact,
    load_json,
    require_key,
    save_json,
    utc_now_iso,
)


@dataclass(frozen=True)
class GroupLaneGenerationConfig:
    group_placement_json: str
    output_json: Optional[str] = None
    env_json: Optional[str] = None
    device: Optional[str] = None
    backend_selection: str = "benchmark"
    algorithm: str = "astar"  # dijkstra|astar
    allow_diagonal: bool = False


def _placement_order(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    placements_raw = payload.get("placements", []) or []
    placements: List[Mapping[str, Any]] = [p for p in placements_raw if isinstance(p, Mapping)]
    by_gid: Dict[str, Mapping[str, Any]] = {str(p.get("gid")): p for p in placements}
    order = payload.get("placed_order", []) or []
    ordered: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for gid in order:
        k = str(gid)
        p = by_gid.get(k)
        if p is not None:
            ordered.append(p)
            seen.add(k)
    for p in placements:
        gid = str(p.get("gid"))
        if gid not in seen:
            ordered.append(p)
    return ordered


def _restore_env_with_group_placement(
    loaded: LoadedEnv,
    payload: Mapping[str, Any],
) -> None:
    env = loaded.env
    env.reset(options=loaded.reset_kwargs)
    state = env.get_state()

    grid = payload.get("grid", {}) or {}
    grid_size_mm = float(grid.get("grid_size_mm", loaded.grid_size_mm))
    if grid_size_mm <= 0:
        raise ValueError(f"invalid grid_size_mm={grid_size_mm}")

    for item in _placement_order(payload):
        gid = str(item.get("gid"))
        if gid not in env.group_specs:
            raise KeyError(f"group {gid!r} not found in env.group_specs")
        spec = env.group_specs[gid]
        variant_index = int(item.get("variant_index", 0))
        x_bl_mm = float(item.get("x_bl_mm", 0.0))
        y_bl_mm = float(item.get("y_bl_mm", 0.0))
        x_bl = int(round(x_bl_mm / grid_size_mm))
        y_bl = int(round(y_bl_mm / grid_size_mm))
        placement = spec.build_placement(
            variant_index=variant_index,
            x_bl=x_bl,
            y_bl=y_bl,
        )
        state.place(placement=placement)


def _summarize_routes(routes: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    routes_l = list(routes)
    success = [r for r in routes_l if bool(r.get("success"))]
    failed = [r for r in routes_l if not bool(r.get("success"))]
    total_cost = 0.0
    for r in success:
        c = r.get("cost")
        if c is None:
            continue
        total_cost += float(c)
    failed_pairs = [[str(r.get("src_group")), str(r.get("dst_group"))] for r in failed]
    return {
        "total_flows": int(len(routes_l)),
        "success_count": int(len(success)),
        "fail_count": int(len(failed)),
        "total_cost": float(total_cost),
        "failed_flows": failed_pairs,
    }


def run_group_lane_generation(cfg: GroupLaneGenerationConfig) -> GroupLaneGenerationArtifact:
    start = time.perf_counter()
    group_artifact = load_json(cfg.group_placement_json)
    if "group_placement" in group_artifact:
        payload = require_key(group_artifact, "group_placement")
        if not isinstance(payload, dict):
            raise ValueError("group_placement must be an object")
        env_json = str(cfg.env_json or require_key(group_artifact, "env_json"))
    else:
        payload = group_artifact
        if not isinstance(payload, dict):
            raise ValueError("group_placement json must be an object")
        if cfg.env_json is None:
            raise ValueError("env_json is required when group_placement_json is a raw placement dict")
        env_json = str(cfg.env_json)
    loaded = load_env(
        env_json,
        device=None if cfg.device is None else torch.device(cfg.device),
        backend_selection=cfg.backend_selection,
    )
    _restore_env_with_group_placement(loaded, payload)

    planner = RoutePlanner(
        loaded.env,
        allow_diagonal=bool(cfg.allow_diagonal),
        algorithm=str(cfg.algorithm),
    )
    routes = routes_to_dict(planner.plan_all())
    summary = _summarize_routes(routes)

    artifact = GroupLaneGenerationArtifact(
        stage="group_lane_generation",
        created_at=utc_now_iso(),
        env_json=env_json,
        group_placement=dict(payload),
        lane_generation={
            "algorithm": str(cfg.algorithm),
            "allow_diagonal": bool(cfg.allow_diagonal),
            "routes": routes,
            "summary": summary,
        },
        metrics={
            "route_count": int(len(routes)),
            "success_count": int(summary["success_count"]),
            "fail_count": int(summary["fail_count"]),
            "total_cost": float(summary["total_cost"]),
            "elapsed_sec": float(time.perf_counter() - start),
        },
    )
    return artifact


def run_and_save_group_lane_generation(cfg: GroupLaneGenerationConfig) -> GroupLaneGenerationArtifact:
    artifact = run_group_lane_generation(cfg)
    if cfg.output_json:
        save_json(artifact, cfg.output_json)
    return artifact


__all__ = [
    "GroupLaneGenerationConfig",
    "run_group_lane_generation",
    "run_and_save_group_lane_generation",
]
