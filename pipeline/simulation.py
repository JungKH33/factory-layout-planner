from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Optional

from pipeline.schema import (
    SimulationArtifact,
    save_json,
    utc_now_iso,
)
from simulation import SimulationRunner, load_simulation_input

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationConfig:
    group_placement_json: str
    output_dir: str
    env_json: Optional[str] = None
    facility_placement_json: Optional[str] = None
    lane_generation_json: Optional[str] = None
    horizon_sec: float = 3600.0
    warmup_sec: float = 300.0
    seed: int = 0
    timeline_step_sec: float = 60.0


def run_simulation(cfg: SimulationConfig) -> SimulationArtifact:
    start = time.perf_counter()
    if cfg.env_json is None:
        raise ValueError("env_json is required for simulation stage")

    sim_input = load_simulation_input(
        env_json=str(cfg.env_json),
        group_placement_json=str(cfg.group_placement_json),
        facility_placement_json=cfg.facility_placement_json,
        lane_generation_json=cfg.lane_generation_json,
        horizon_sec=float(cfg.horizon_sec),
        warmup_sec=float(cfg.warmup_sec),
        seed=int(cfg.seed),
        timeline_step_sec=float(cfg.timeline_step_sec),
    )
    result = SimulationRunner(sim_input).run()

    artifact = SimulationArtifact(
        stage="simulation",
        created_at=utc_now_iso(),
        env_json=str(cfg.env_json),
        group_placement=dict(sim_input.group_placement),
        facility_placement=dict(sim_input.facility_placement),
        lane_generation=dict(sim_input.lane_generation),
        simulation={
            "summary": result["summary"],
            "utilization": result["utilization"],
            "bottlenecks": result["bottlenecks"],
            "timeline": result["timeline"],
            "assumptions": list(sim_input.assumptions),
        },
        metrics={
            "elapsed_sec": float(time.perf_counter() - start),
            "flow_count": int(len(sim_input.flows)),
            "event_count": int(result["summary"]["created_count"]),
            "horizon_sec": float(cfg.horizon_sec),
            "warmup_sec": float(cfg.warmup_sec),
            "seed": int(cfg.seed),
        },
    )
    return artifact


def run_and_save_simulation(cfg: SimulationConfig) -> SimulationArtifact:
    artifact = run_simulation(cfg)
    out_path = Path(cfg.output_dir) / "simulation.json"
    save_json(artifact, out_path)
    logger.info("simulation output_dir: %s", cfg.output_dir)
    logger.info("simulation artifact saved: %s", out_path)
    logger.info("simulation metrics: %s", artifact.metrics)
    return artifact


__all__ = [
    "SimulationConfig",
    "run_simulation",
    "run_and_save_simulation",
]

