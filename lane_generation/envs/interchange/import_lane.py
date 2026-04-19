"""Load lane-generation interchange JSON and restore env state.

Mirrors ``group_placement.envs.interchange.import_placement``:

- :func:`load_lane_generation` reads the interchange dict from disk.
- :func:`apply_interchange_to_env` replays edges onto a
  :class:`FactoryLaneEnv`, returning the restored :class:`LaneRoute` list.
- :func:`restore_lane_from_files` combines load + apply in one call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

from ..action import LaneRoute
from ..env import FactoryLaneEnv


def load_lane_generation(path: Union[str, Path]) -> Dict[str, Any]:
    """Read lane-generation interchange JSON from disk."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_grid(env: FactoryLaneEnv, grid_block: Mapping[str, Any]) -> None:
    w = int(grid_block["width_cells"])
    h = int(grid_block["height_cells"])
    if int(env.grid_width) != w or int(env.grid_height) != h:
        raise ValueError(
            f"interchange grid {w}x{h} does not match env {env.grid_width}x{env.grid_height}"
        )


def apply_interchange_to_env(
    env: FactoryLaneEnv,
    data: Mapping[str, Any],
    *,
    check_grid: bool = True,
) -> List[LaneRoute]:
    """Replay interchange routes on *env*, returning restored routes.

    The env is ``reset()`` first, then edges from each successful route are
    applied in ``flow_order``.  Failed flows are skipped (no edges applied,
    but ``routed_mask`` is set and ``step_count`` advances so the state is
    consistent).

    Args:
        env: a :class:`FactoryLaneEnv` built from the same grid/flows.
        data: interchange dict (from :func:`export_lane_generation`).
        check_grid: if True, validates grid dimensions match.

    Returns:
        List of :class:`LaneRoute` for successfully restored routes.
    """
    grid_block = data.get("grid") or {}
    if check_grid:
        if "width_cells" not in grid_block or "height_cells" not in grid_block:
            raise ValueError("interchange dict missing grid.width_cells / grid.height_cells")
        _validate_grid(env, grid_block)

    env.reset()
    state = env.get_state()

    route_by_fi: Dict[int, Mapping[str, Any]] = {}
    for rd in data.get("routes", []):
        fi = int(rd["flow_index"])
        route_by_fi[fi] = rd  # keep all (success and failed) for lane_width restoration

    # Restore per-flow overrides from interchange if present
    for fi, rd in route_by_fi.items():
        if 0 > fi or fi >= len(state.flow_specs):
            continue
        old_spec = state.flow_specs[fi]
        new_lw = float(rd["lane_width"]) if "lane_width" in rd else old_spec.lane_width
        new_ra = rd.get("reverse_allow", old_spec.reverse_allow)
        new_ma = rd.get("merge_allow", old_spec.merge_allow)
        needs_update = (
            abs(float(new_lw) - old_spec.lane_width) > 1e-9
            or new_ra != old_spec.reverse_allow
            or new_ma != old_spec.merge_allow
        )
        if needs_update:
            from lane_generation.envs.state.lane import LaneFlowSpec
            new_specs = list(state.flow_specs)
            new_specs[fi] = LaneFlowSpec(
                src=old_spec.src,
                dst=old_spec.dst,
                weight=old_spec.weight,
                reverse_allow=new_ra if new_ra is not None else None,
                merge_allow=new_ma if new_ma is not None else None,
                lane_width=float(new_lw),
            )
            state.flow_specs = tuple(new_specs)

    # Re-filter to only successful routes for edge replay
    route_by_fi = {fi: rd for fi, rd in route_by_fi.items() if rd.get("success") and rd.get("edges")}

    routes_out: List[LaneRoute] = []
    routed_fis: List[int] = []

    for pos in range(int(state.flow_order.numel())):
        fi = int(state.flow_order[pos].item())
        rd = route_by_fi.get(fi)
        if rd is not None:
            edges_t = torch.tensor(rd["edges"], dtype=torch.long, device=state.device)
            spec = state.flow_specs[fi]
            lw = float(spec.lane_width)
            ma = state.flow_merge_allow(fi)
            ra = state.flow_reverse_allow(fi)
            lane_slots_raw = rd.get("lane_slots", None)
            lane_slots = None
            if isinstance(lane_slots_raw, list) and len(lane_slots_raw) == int(edges_t.numel()):
                lane_slots = [int(x) for x in lane_slots_raw]
            if lane_slots is None:
                l = int(edges_t.numel())
                edge_idx = edges_t.view(1, l)
                edge_mask = torch.ones((1, l), dtype=torch.bool, device=state.device)
                planned = state.preview_lane_slots_batch(
                    candidate_edge_idx=edge_idx,
                    candidate_edge_mask=edge_mask,
                    merge_allow=ma, reverse_allow=ra,
                )
                lane_slots = [int(x) for x in planned[0].to(dtype=torch.long).tolist()]
            state.apply_edges(edges_t, lane_slots=lane_slots, lane_width=lw)
            state.route_lane_slots_by_flow[fi] = tuple(lane_slots)
            routed_fis.append(fi)
            routes_out.append(LaneRoute(
                flow_index=fi,
                candidate_index=int(rd.get("candidate_index", 0)),
                edge_indices=edges_t,
                path_length=float(rd.get("path_length", edges_t.numel())),
                turns=int(rd.get("turns", 0)),
                planned_lane_slots=torch.tensor(lane_slots, dtype=torch.long, device=state.device),
            ))

    for fi in routed_fis:
        state.routed_mask[fi] = True

    state.step_count = int(state.flow_count)
    return routes_out


def restore_lane_from_files(
    env_json: Union[str, Path],
    group_placement: Union[str, Path, Mapping[str, Any]],
    interchange_json: Union[str, Path],
    *,
    check_grid: bool = True,
    **load_kw: Any,
) -> tuple:
    """:func:`load_lane_env` + :func:`apply_interchange_to_env` in one call.

    Returns ``(loaded, routes)`` where ``loaded`` is a :class:`LoadedLaneEnv`
    and ``routes`` is the list of restored :class:`LaneRoute`.
    """
    from ..env_loader import load_lane_env

    loaded = load_lane_env(
        env_json=str(Path(env_json)),
        group_placement=group_placement,
        **load_kw,
    )
    data = load_lane_generation(interchange_json)
    routes = apply_interchange_to_env(
        loaded.env, data, check_grid=check_grid,
    )
    return loaded, routes


__all__ = [
    "load_lane_generation",
    "apply_interchange_to_env",
    "restore_lane_from_files",
]
