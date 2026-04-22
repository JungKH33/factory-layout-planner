from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Dict, Optional

from facility_placement import resolve_facilities
from pipeline.schema import (
    FacilityPlacementArtifact,
    load_json,
    require_key,
    save_json,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FacilityPlacementConfig:
    group_placement_json: str
    output_dir: str
    env_json: Optional[str] = None
    on_missing: str = "warn"  # warn|silent|error


def run_facility_placement(cfg: FacilityPlacementConfig) -> FacilityPlacementArtifact:
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

    resolved = resolve_facilities(payload, on_missing=str(cfg.on_missing))
    output: Dict[str, list] = {
        str(gid): [asdict(item) for item in placements]
        for gid, placements in resolved.items()
    }
    facility_count = sum(len(v) for v in output.values())

    artifact = FacilityPlacementArtifact(
        stage="facility_placement",
        created_at=utc_now_iso(),
        env_json=env_json,
        group_placement=dict(payload),
        facility_placement={
            "by_group": output,
            "summary": {
                "group_count": int(len(output)),
                "facility_count": int(facility_count),
            },
        },
        metrics={
            "group_count": int(len(output)),
            "facility_count": int(facility_count),
            "elapsed_sec": float(time.perf_counter() - start),
        },
    )
    return artifact


def run_and_save_facility_placement(cfg: FacilityPlacementConfig) -> FacilityPlacementArtifact:
    artifact = run_facility_placement(cfg)
    out_path = Path(cfg.output_dir) / "facility_placement.json"
    save_json(artifact, out_path)
    logger.info("facility_placement output_dir: %s", cfg.output_dir)
    logger.info("facility_placement artifact saved: %s", out_path)
    logger.info("facility_placement metrics: %s", artifact.metrics)
    return artifact


__all__ = [
    "FacilityPlacementConfig",
    "run_facility_placement",
    "run_and_save_facility_placement",
]
