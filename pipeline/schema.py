from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, TypeVar


JsonDict = Dict[str, Any]
_T = TypeVar("_T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PreprocessArtifact:
    stage: str
    created_at: str
    input_json: str
    input_copy_json: str
    env_json: str
    metrics: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class GroupPlacementArtifact:
    stage: str
    created_at: str
    env_json: str
    group_placement: JsonDict
    metrics: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class LaneGenerationArtifact:
    stage: str
    created_at: str
    env_json: str
    group_placement: JsonDict
    lane_generation: JsonDict
    metrics: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FacilityPlacementArtifact:
    stage: str
    created_at: str
    env_json: str
    group_placement: JsonDict
    facility_placement: JsonDict
    metrics: JsonDict = field(default_factory=dict)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def to_json_dict(data: Any) -> JsonDict:
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, dict):
        return dict(data)
    raise TypeError(f"expected dataclass or dict, got {type(data).__name__}")


def save_json(data: Any, path: str | Path) -> None:
    ensure_parent_dir(path)
    out = to_json_dict(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> JsonDict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"artifact root must be a JSON object: {path}")
    return obj


def require_key(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"missing required key '{key}'")
    return data[key]


__all__ = [
    "PreprocessArtifact",
    "GroupPlacementArtifact",
    "LaneGenerationArtifact",
    "FacilityPlacementArtifact",
    "utc_now_iso",
    "save_json",
    "load_json",
    "require_key",
]
