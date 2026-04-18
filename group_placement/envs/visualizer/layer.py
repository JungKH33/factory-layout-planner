"""Per-reward visualization layer dataclasses.

No matplotlib or engine dependencies — pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RewardVizSegment:
    """A single directed segment (arrow or polyline) within a reward layer."""
    src_xy: Tuple[float, float]
    dst_xy: Tuple[float, float]
    weight: float
    polyline: Optional[List[Tuple[float, float]]] = None


@dataclass
class RewardVizPort:
    """A port point with activity state."""
    xy: Tuple[float, float]
    kind: str    # "entry" | "exit"
    active: bool


@dataclass
class RewardLayer:
    """All visualization data for one reward component."""
    key: str
    label: str
    color: str
    style: str           # "solid" | "dashed"
    default_visible: bool
    phase: str           # "base" | "terminal"
    segments: List[RewardVizSegment] = field(default_factory=list)
    ports: List[RewardVizPort] = field(default_factory=list)
