"""Pydantic schemas for WebUI API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    env_json: str = "converters/test.json"
    wrapper_mode: str = "greedyv3"  # greedy | greedyv2 | greedyv3 | alphachip | maskplace
    agent_mode: str = "greedy"  # greedy | alphachip | maskplace
    search_mode: str = "none"  # none | mcts | beam
    
    # 동적 파라미터 (wrapper_*, agent_*, search_* 형식)
    params: Dict[str, Any] = {}


class StepRequest(BaseModel):
    action: int


class SearchRequest(BaseModel):
    simulations: int = 100
    broadcast_interval: int = 10  # Send update every N simulations


class PlacedFacility(BaseModel):
    gid: str
    x: float
    y: float
    w: float
    h: float
    rot: int


class CandidateInfo(BaseModel):
    index: int
    x: float
    y: float
    rot: int
    score: float
    valid: bool
    visits: int = 0
    q_value: float = 0.0


class ZoneRect(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    value: Optional[float] = None
    id: Optional[str] = None


class FlowEdge(BaseModel):
    src: str
    dst: str
    weight: float
    src_x: Optional[float] = None
    src_y: Optional[float] = None
    dst_x: Optional[float] = None
    dst_y: Optional[float] = None


class SessionState(BaseModel):
    grid_width: int
    grid_height: int
    placed: List[PlacedFacility]
    remaining: List[str]
    current_gid: Optional[str]
    candidates: List[CandidateInfo]
    value: float
    cost: float
    step: int
    history_length: int
    terminated: bool
    can_undo: bool
    can_redo: bool
    
    # Zones and overlays
    forbidden_areas: List[ZoneRect] = []
    placement_zones: List[ZoneRect] = []
    weight_zones: List[ZoneRect] = []
    dry_zones: List[ZoneRect] = []
    height_zones: List[ZoneRect] = []
    flow_edges: List[FlowEdge] = []


class SearchProgress(BaseModel):
    simulation: int
    total: int
    candidates: List[CandidateInfo]
    best_action: int
    best_value: float
