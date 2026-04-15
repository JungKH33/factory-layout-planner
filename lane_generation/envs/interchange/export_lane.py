"""Phase 2 → downstream interchange: export lane-generation results.

Produces a JSON-serializable dict representing all routed lanes.  Each route
carries both ``edges`` (directed-edge indices for reimport) and ``cells``
(coordinate polyline for visualisation).

Downstream consumers read only this dict and never touch ``LaneState``
directly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

from ..action import LaneRoute
from ..env import FactoryLaneEnv

logger = logging.getLogger(__name__)


_DIR_TO_DXY = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _route_to_dict(route: LaneRoute, state) -> Dict[str, Any]:
    """Convert a single :class:`LaneRoute` to a serializable dict."""
    fi = int(route.flow_index)
    spec = state.flow_specs[fi]
    edges = route.edge_indices.detach().cpu()
    w = int(state.grid_width)

    edge_list = edges.tolist()

    cells: List[List[int]] = []
    seen: set[tuple[int, int]] = set()
    for eid in edge_list:
        cell = int(eid) // 4
        cy, cx = divmod(cell, w)
        pt = (cx, cy)
        if pt not in seen:
            seen.add(pt)
            cells.append([cx, cy])

    if edge_list:
        last_eid = int(edges[-1].item())
        dst_cell = int(state.edge_dst_cell[last_eid].item())
        if dst_cell >= 0:
            dy, dx = divmod(dst_cell, w)
            pt = (dx, dy)
            if pt not in seen:
                cells.append([dx, dy])

    lane_slots = None
    try:
        planned = route.planned_lane_slots
        if planned is not None:
            planned_list = planned.detach().to(dtype=edges.dtype, device="cpu").view(-1).tolist()
            if len(planned_list) == len(edge_list):
                lane_slots = [int(s) for s in planned_list]
    except Exception:
        lane_slots = None

    if lane_slots is None:
        try:
            slots = state.route_lane_slots_by_flow.get(fi, None)
            if isinstance(slots, tuple) and len(slots) == len(edge_list):
                lane_slots = [int(s) for s in slots]
        except Exception:
            lane_slots = None

    out = {
        "flow_index": fi,
        "src_gid": str(spec.src_gid),
        "dst_gid": str(spec.dst_gid),
        "weight": float(spec.weight),
        "success": True,
        "edges": edge_list,
        "cells": cells,
        "edge_count": int(edges.numel()),
        "path_length": float(route.path_length),
        "turns": int(route.turns),
    }
    if lane_slots is not None:
        out["lane_slots"] = lane_slots
    return out


def _failed_flow_dict(state, flow_index: int) -> Dict[str, Any]:
    spec = state.flow_specs[int(flow_index)]
    return {
        "flow_index": int(flow_index),
        "src_gid": str(spec.src_gid),
        "dst_gid": str(spec.dst_gid),
        "weight": float(spec.weight),
        "success": False,
        "edges": [],
        "cells": [],
        "edge_count": 0,
        "path_length": 0.0,
        "turns": 0,
    }


def _summarize(routes_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = [r for r in routes_out if r.get("success")]
    failed = [r for r in routes_out if not r.get("success")]
    total_cost = sum(float(r.get("path_length") or 0.0) for r in success)
    return {
        "total_flows": len(routes_out),
        "success_count": len(success),
        "fail_count": len(failed),
        "total_cost": total_cost,
        "failed_flows": [
            [str(r.get("src_gid")), str(r.get("dst_gid"))] for r in failed
        ],
    }


def export_lane_generation(
    env: FactoryLaneEnv,
    routes: Sequence[LaneRoute],
) -> Dict[str, Any]:
    """Flatten a lane-generation episode into an interchange dict.

    The returned dict is JSON-serializable and includes both ``edges``
    (directed-edge indices for :func:`import_lane.apply_interchange_to_env`)
    and ``cells`` (coordinate polyline for visualisation).
    """
    state = env.get_state()
    routes_out: List[Dict[str, Any]] = []

    route_by_flow = {int(r.flow_index): r for r in routes}
    for pos in range(int(state.flow_order.numel())):
        fi = int(state.flow_order[pos].item())
        if fi in route_by_flow:
            routes_out.append(_route_to_dict(route_by_flow[fi], state))
        else:
            routes_out.append(_failed_flow_dict(state, fi))

    return {
        "schema_version": 2,
        "grid": {
            "width_cells": int(env.grid_width),
            "height_cells": int(env.grid_height),
        },
        "routes": routes_out,
        "summary": _summarize(routes_out),
    }


def save_lane_generation(
    env: FactoryLaneEnv,
    routes: Sequence[LaneRoute],
    path: str,
) -> None:
    """Write ``export_lane_generation(env, routes)`` to ``path`` as JSON."""
    data = export_lane_generation(env, routes)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("saved lane generation: %s", path)


def print_summary(
    env: FactoryLaneEnv,
    routes: Sequence[LaneRoute],
) -> None:
    """Print a lane-generation summary to the logger."""
    data = export_lane_generation(env, routes)
    summary = data["summary"]
    logger.info("=" * 50)
    logger.info("Lane Generation Summary")
    logger.info("=" * 50)
    logger.info("Total flows: %s", summary["total_flows"])
    logger.info("Success: %s", summary["success_count"])
    logger.info("Failed: %s", summary["fail_count"])
    logger.info("Total cost: %.2f", summary["total_cost"])
    if summary["failed_flows"]:
        logger.info("Failed flows:")
        for src, dst in summary["failed_flows"]:
            logger.info("  - %s -> %s", src, dst)
    logger.info("=" * 50)


__all__ = [
    "export_lane_generation",
    "save_lane_generation",
    "print_summary",
]
