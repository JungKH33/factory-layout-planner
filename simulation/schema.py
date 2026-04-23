from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class StageSpec:
    cycle_time_sec: float
    parallel_slots: int = 1


@dataclass(frozen=True)
class FacilitySpec:
    gid: str
    cycle_time_sec: float
    parallel_slots: int
    batch_size: int
    buffer_in: int
    buffer_out: int
    processing_mode: str = "serial"  # serial|parallel|batch|pipeline
    stages: List[StageSpec] = field(default_factory=list)


@dataclass(frozen=True)
class TransportSpec:
    fleet_size: int
    speed_mps: float
    load_unload_sec: float
    dispatch_rule: str = "nearest_idle"


@dataclass(frozen=True)
class RunSpec:
    horizon_sec: float
    warmup_sec: float = 0.0
    seed: int = 0
    timeline_step_sec: float = 60.0


@dataclass(frozen=True)
class FlowEdge:
    src_gid: str
    dst_gid: str
    weight: float


@dataclass(frozen=True)
class SimulationInput:
    env_json: str
    group_placement: JsonDict
    facility_placement: JsonDict
    lane_generation: JsonDict
    flows: List[FlowEdge]
    facility_specs: Dict[str, FacilitySpec]
    transport_spec: TransportSpec
    run_spec: RunSpec
    assumptions: List[str] = field(default_factory=list)


def _positive_float(name: str, value: float) -> float:
    out = float(value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return out


def _non_negative_float(name: str, value: float) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return out


def _positive_int(name: str, value: int) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return out


def validate_input(data: SimulationInput) -> None:
    if not data.flows:
        raise ValueError("simulation requires at least one flow edge")

    _positive_float("run_spec.horizon_sec", data.run_spec.horizon_sec)
    _non_negative_float("run_spec.warmup_sec", data.run_spec.warmup_sec)
    _positive_float("run_spec.timeline_step_sec", data.run_spec.timeline_step_sec)
    _positive_int("transport_spec.fleet_size", data.transport_spec.fleet_size)
    _positive_float("transport_spec.speed_mps", data.transport_spec.speed_mps)
    _non_negative_float("transport_spec.load_unload_sec", data.transport_spec.load_unload_sec)

    for gid, spec in data.facility_specs.items():
        if gid != spec.gid:
            raise ValueError(f"facility spec key mismatch: key={gid} spec.gid={spec.gid}")
        _positive_float(f"{gid}.cycle_time_sec", spec.cycle_time_sec)
        _positive_int(f"{gid}.parallel_slots", spec.parallel_slots)
        _positive_int(f"{gid}.batch_size", spec.batch_size)
        _positive_int(f"{gid}.buffer_in", spec.buffer_in)
        _positive_int(f"{gid}.buffer_out", spec.buffer_out)
        mode = str(spec.processing_mode).strip().lower()
        if mode not in {"serial", "parallel", "batch", "pipeline"}:
            raise ValueError(f"{gid}.processing_mode invalid: {spec.processing_mode}")
        for idx, stage in enumerate(spec.stages):
            _positive_float(f"{gid}.stages[{idx}].cycle_time_sec", stage.cycle_time_sec)
            _positive_int(f"{gid}.stages[{idx}].parallel_slots", stage.parallel_slots)


def normalize_dict(data: Mapping[str, Any]) -> JsonDict:
    return {str(k): v for k, v in data.items()}

