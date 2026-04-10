"""Pydantic schemas for WebUI API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    env_json: str = "converters/test.json"
    collision_check: str = "auto"  # auto | conv | prefixsum
    backend_selection: str = "benchmark"  # static | benchmark
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


class GotoRequest(BaseModel):
    node_id: int


class TopKRestoreRequest(BaseModel):
    item_id: str


class PortInfo(BaseModel):
    x: float
    y: float


class PlacedFacility(BaseModel):
    gid: str
    x: float
    y: float
    w: float
    h: float
    rot: int
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    entries: List[PortInfo] = []
    exits: List[PortInfo] = []
    variant_index: int = 0


class FlowDeltaInfo(BaseModel):
    src: str
    dst: str
    weight: float
    distance: float


class PhysicalContextInfo(BaseModel):
    """Physical placement result for the last step."""
    gid: str
    x: float
    y: float
    w: float
    h: float
    rotation: int
    variant_index: int
    x_center: float
    y_center: float
    entries: List[PortInfo] = []
    exits: List[PortInfo] = []
    delta_cost: float
    cost_before: float
    cost_after: float
    affected_flows: List[FlowDeltaInfo] = []


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
    forbidden: List[ZoneRect] = []
    constraint_zones: Dict[str, List[ZoneRect]] = {}
    flow_edges: List[FlowEdge] = []

    # Physical context from last step (None at root)
    last_physical: Optional[PhysicalContextInfo] = None


class SearchProgress(BaseModel):
    simulation: int
    total: int
    candidates: List[CandidateInfo]
    best_action: int
    best_value: float
