from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from facility_placement import resolve_facilities
from pipeline.schema import load_json, require_key
from simulation.schema import FacilitySpec, FlowEdge, RunSpec, SimulationInput, StageSpec, TransportSpec


def _unwrap_payload(raw: Mapping[str, Any], key: str) -> Dict[str, Any]:
    if key in raw:
        payload = require_key(raw, key)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{key} must be a JSON object")
        return dict(payload)
    return dict(raw)


def _load_json_file(path: str | Path) -> Dict[str, Any]:
    return load_json(path)


def _load_env_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"env_json must be an object: {path}")
    return data


def _collect_flows(env_data: Mapping[str, Any], group_payload: Mapping[str, Any]) -> List[FlowEdge]:
    flow_obj = env_data.get("flow")
    flows: List[FlowEdge] = []

    if isinstance(flow_obj, list):
        for row in flow_obj:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            flows.append(
                FlowEdge(
                    src_gid=str(row[0]),
                    dst_gid=str(row[1]),
                    weight=float(row[2]),
                )
            )
    elif isinstance(flow_obj, Mapping):
        for src, dsts in flow_obj.items():
            if not isinstance(dsts, Mapping):
                continue
            for dst, weight in dsts.items():
                flows.append(FlowEdge(src_gid=str(src), dst_gid=str(dst), weight=float(weight)))

    if flows:
        return flows

    # Fallback: group placement eval metadata (if preprocess did not expose "flow")
    eval_edges = (
        group_payload.get("eval", {})
        .get("base_rewards", {})
        .get("flow", {})
        .get("metadata", {})
        .get("edges", {})
    )
    if isinstance(eval_edges, Mapping):
        for key, meta in eval_edges.items():
            if "->" not in str(key):
                continue
            src, dst = str(key).split("->", 1)
            weight = 1.0
            if isinstance(meta, Mapping):
                weight = float(meta.get("weight", 1.0))
            flows.append(FlowEdge(src_gid=src, dst_gid=dst, weight=weight))

    return flows


def _gid_positions(group_payload: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for item in group_payload.get("placements", []) or []:
        if not isinstance(item, Mapping):
            continue
        gid = str(item.get("gid", ""))
        x = float(item.get("x_bl_mm", 0.0)) + float(item.get("cluster_w_mm", 0.0)) * 0.5
        y = float(item.get("y_bl_mm", 0.0)) + float(item.get("cluster_h_mm", 0.0)) * 0.5
        out[gid] = (x, y)
    return out


def _build_facility_specs(
    env_data: Mapping[str, Any],
    group_payload: Mapping[str, Any],
    facility_payload: Mapping[str, Any],
    assumptions: List[str],
) -> Dict[str, FacilitySpec]:
    groups = env_data.get("groups", {})
    if not isinstance(groups, Mapping):
        groups = {}
    by_group = facility_payload.get("by_group", {})
    if not isinstance(by_group, Mapping):
        by_group = {}

    specs: Dict[str, FacilitySpec] = {}
    gids = set(_gid_positions(group_payload).keys())
    gids.update(str(k) for k in groups.keys())

    for gid in sorted(gids):
        group_cfg = groups.get(gid, {})
        sim_cfg = group_cfg.get("simulation", {}) if isinstance(group_cfg, Mapping) else {}
        if not isinstance(sim_cfg, Mapping):
            sim_cfg = {}

        cycle_time_sec = float(sim_cfg.get("cycle_time_sec", 60.0))
        if "cycle_time_sec" not in sim_cfg:
            assumptions.append(f"{gid}: cycle_time_sec default 60.0")

        parallel_slots = int(sim_cfg.get("parallel_slots", 1))
        if "parallel_slots" not in sim_cfg:
            assumptions.append(f"{gid}: parallel_slots default 1")

        batch_size = int(sim_cfg.get("batch_size", 1))
        if "batch_size" not in sim_cfg:
            assumptions.append(f"{gid}: batch_size default 1")

        raw_mode = sim_cfg.get("processing_mode")
        raw_stages = sim_cfg.get("stages", [])
        stages: List[StageSpec] = []
        if isinstance(raw_stages, list):
            for idx, item in enumerate(raw_stages):
                if not isinstance(item, Mapping):
                    assumptions.append(f"{gid}: stages[{idx}] ignored (not object)")
                    continue
                stage_cycle = float(item.get("cycle_time_sec", cycle_time_sec))
                stage_slots = int(item.get("parallel_slots", 1))
                stages.append(
                    StageSpec(
                        cycle_time_sec=max(0.001, stage_cycle),
                        parallel_slots=max(1, stage_slots),
                    )
                )
        elif raw_stages:
            assumptions.append(f"{gid}: stages ignored (must be list)")

        if raw_mode is None:
            if stages:
                processing_mode = "pipeline"
                assumptions.append(f"{gid}: processing_mode inferred as pipeline (stages provided)")
            elif batch_size > 1:
                processing_mode = "batch"
                assumptions.append(f"{gid}: processing_mode inferred as batch (batch_size>1)")
            elif parallel_slots > 1:
                processing_mode = "parallel"
                assumptions.append(f"{gid}: processing_mode inferred as parallel (parallel_slots>1)")
            else:
                processing_mode = "serial"
                assumptions.append(f"{gid}: processing_mode default serial")
        else:
            processing_mode = str(raw_mode).strip().lower()

        if processing_mode not in {"serial", "parallel", "batch", "pipeline"}:
            assumptions.append(f"{gid}: invalid processing_mode={raw_mode!r}, fallback serial")
            processing_mode = "serial"

        if processing_mode == "serial" and parallel_slots != 1:
            assumptions.append(f"{gid}: serial mode forces parallel_slots=1")
            parallel_slots = 1

        if processing_mode == "pipeline" and not stages:
            stages = [
                StageSpec(
                    cycle_time_sec=max(0.001, cycle_time_sec),
                    parallel_slots=max(1, parallel_slots),
                )
            ]
            assumptions.append(f"{gid}: pipeline mode without stages, created single default stage")

        default_buffer = max(4, parallel_slots * 2)
        buffer_in = int(sim_cfg.get("buffer_in", default_buffer))
        if "buffer_in" not in sim_cfg:
            assumptions.append(f"{gid}: buffer_in default {default_buffer}")
        buffer_out = int(sim_cfg.get("buffer_out", default_buffer))
        if "buffer_out" not in sim_cfg:
            assumptions.append(f"{gid}: buffer_out default {default_buffer}")

        # Optional heuristic: if facility unfolds to many stations but no slots defined,
        # keep slots as 1 for conservative baseline.
        if gid in by_group and "parallel_slots" not in sim_cfg:
            _ = by_group.get(gid)

        specs[gid] = FacilitySpec(
            gid=gid,
            cycle_time_sec=max(0.001, cycle_time_sec),
            parallel_slots=max(1, parallel_slots),
            batch_size=max(1, batch_size),
            buffer_in=max(1, buffer_in),
            buffer_out=max(1, buffer_out),
            processing_mode=processing_mode,
            stages=stages,
        )
    return specs


def _build_transport_spec(env_data: Mapping[str, Any], assumptions: List[str]) -> TransportSpec:
    sim = env_data.get("simulation", {})
    if not isinstance(sim, Mapping):
        sim = {}
    transport = sim.get("transport", {})
    if not isinstance(transport, Mapping):
        transport = {}

    fleet_size = int(transport.get("fleet_size", 4))
    if "fleet_size" not in transport:
        assumptions.append("transport.fleet_size default 4")
    speed_mps = float(transport.get("speed_mps", 1.0))
    if "speed_mps" not in transport:
        assumptions.append("transport.speed_mps default 1.0")
    load_unload_sec = float(transport.get("load_unload_sec", 5.0))
    if "load_unload_sec" not in transport:
        assumptions.append("transport.load_unload_sec default 5.0")
    dispatch_rule = str(transport.get("dispatch_rule", "nearest_idle"))
    if "dispatch_rule" not in transport:
        assumptions.append("transport.dispatch_rule default nearest_idle")

    return TransportSpec(
        fleet_size=max(1, fleet_size),
        speed_mps=max(0.001, speed_mps),
        load_unload_sec=max(0.0, load_unload_sec),
        dispatch_rule=dispatch_rule,
    )


def load_simulation_input(
    *,
    env_json: str,
    group_placement_json: str,
    facility_placement_json: str | None,
    lane_generation_json: str | None,
    horizon_sec: float,
    warmup_sec: float,
    seed: int,
    timeline_step_sec: float = 60.0,
) -> SimulationInput:
    assumptions: List[str] = []

    group_raw = _load_json_file(group_placement_json)
    group_payload = _unwrap_payload(group_raw, "group_placement")

    if facility_placement_json is not None and Path(facility_placement_json).exists():
        facility_raw = _load_json_file(facility_placement_json)
        facility_payload = _unwrap_payload(facility_raw, "facility_placement")
    else:
        facility_payload = {
            "by_group": {
                str(gid): [item.__dict__ for item in items]
                for gid, items in resolve_facilities(group_payload, on_missing="warn").items()
            }
        }
        assumptions.append("facility_placement generated from group_placement (file missing)")

    if lane_generation_json is not None and Path(lane_generation_json).exists():
        lane_raw = _load_json_file(lane_generation_json)
        lane_payload = _unwrap_payload(lane_raw, "lane_generation")
    else:
        lane_payload = {"routes": [], "summary": {"total_flows": 0}}
        assumptions.append("lane_generation file missing, fallback to geometric travel time")

    env_data = _load_env_json(env_json)
    flows = _collect_flows(env_data, group_payload)
    facility_specs = _build_facility_specs(env_data, group_payload, facility_payload, assumptions)
    transport_spec = _build_transport_spec(env_data, assumptions)
    run_spec = RunSpec(
        horizon_sec=float(horizon_sec),
        warmup_sec=float(warmup_sec),
        seed=int(seed),
        timeline_step_sec=float(timeline_step_sec),
    )

    return SimulationInput(
        env_json=str(env_json),
        group_placement=group_payload,
        facility_placement=facility_payload,
        lane_generation=lane_payload,
        flows=flows,
        facility_specs=facility_specs,
        transport_spec=transport_spec,
        run_spec=run_spec,
        assumptions=assumptions,
    )

