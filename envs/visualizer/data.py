"""Backend-agnostic data extraction from engine state.

Converts engine internals into plain dataclasses that visualization backends
can consume without importing torch, envs, or any engine code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FacilityData:
    gid: str
    x_bl: float
    y_bl: float
    w: float
    h: float
    x_c: float
    y_c: float
    body_polygon_abs: Optional[List[Tuple[float, float]]] = None
    clearance_polygon_abs: Optional[List[Tuple[float, float]]] = None
    entries: List[Tuple[float, float]] = field(default_factory=list)
    exits: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class FlowArrow:
    src_xy: Tuple[float, float]
    dst_xy: Tuple[float, float]
    weight: float


@dataclass
class PortData:
    active_entries: List[Tuple[float, float]] = field(default_factory=list)
    inactive_entries: List[Tuple[float, float]] = field(default_factory=list)
    active_exits: List[Tuple[float, float]] = field(default_factory=list)
    inactive_exits: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ConstraintZoneData:
    name: str
    op: str
    color: str
    heatmap: Optional[np.ndarray] = None  # [H,W] float
    rects: List[dict] = field(default_factory=list)  # area dicts from config


@dataclass
class RouteData:
    src_group: str
    dst_group: str
    path: List[Tuple[float, float]]
    color: str


@dataclass
class LayoutData:
    grid_width: int
    grid_height: int
    facilities: List[FacilityData]
    forbidden_rects: List[Tuple[int, int, int, int]]
    constraint_zones: dict  # name -> ConstraintZoneData
    constraint_names: List[str]
    invalid_mask: Optional[np.ndarray] = None   # [H,W] bool
    clearance_mask: Optional[np.ndarray] = None  # [H,W] bool
    flow_arrows: List[FlowArrow] = field(default_factory=list)
    ports: PortData = field(default_factory=PortData)
    cost: float = 0.0
    candidates_xy: Optional[List[Tuple[float, float]]] = None
    routes: Optional[List[RouteData]] = None


@dataclass(frozen=True)
class StepFrame:
    """A single step frame for interactive browsing."""
    state: Any  # engine/wrapper state copy
    cost: float
    step_idx: int
    action_space: Any = None  # Optional[ActionSpace]
    scores: Optional[np.ndarray] = None
    selected_action: Optional[int] = None
    value: Optional[float] = None


# ---------------------------------------------------------------------------
# Shared color palette
# ---------------------------------------------------------------------------

CONSTRAINT_BASE_COLORS = [
    "#1e90ff",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#bcbd22",
]

ROUTE_COLORS = [
    "#FF6B00", "#00CC66", "#9933FF", "#FF3366",
    "#00BFFF", "#FFD700", "#FF69B4", "#32CD32",
]


def constraint_color(name: str, names: list) -> str:
    idx = 0
    try:
        idx = names.index(str(name))
    except ValueError:
        idx = 0
    return CONSTRAINT_BASE_COLORS[idx % len(CONSTRAINT_BASE_COLORS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coord_key(x: float, y: float) -> Tuple[float, float]:
    return (round(float(x), 6), round(float(y), 6))


def _tensor_to_numpy(t) -> Optional[np.ndarray]:
    """Convert a torch tensor (or None) to numpy."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _constraint_names(engine: Any) -> list:
    constraints = getattr(engine, "zone_constraints", None)
    if not isinstance(constraints, dict):
        return []
    return [str(name) for name in constraints.keys()]


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_layout_data(
    engine: Any,
    *,
    action_space: Any = None,
    routes: Any = None,
) -> LayoutData:
    """Extract all rendering data from engine state into a plain LayoutData.

    Args:
        engine: FactoryLayoutEnv instance
        action_space: Optional ActionSpace (with .poses, .mask, .gid)
        routes: Optional list of RouteResult from postprocess.pathfinder
    """
    state = engine.get_state()
    c_names = _constraint_names(engine)

    # --- Facilities ---
    facilities: List[FacilityData] = []
    for gid in state.placed:
        p = state.placements[gid]
        facilities.append(FacilityData(
            gid=str(gid),
            x_bl=float(getattr(p, "x_bl", p.min_x)),
            y_bl=float(getattr(p, "y_bl", p.min_y)),
            w=float(getattr(p, "w", p.max_x - p.min_x)),
            h=float(getattr(p, "h", p.max_y - p.min_y)),
            x_c=float(p.x_c),
            y_c=float(p.y_c),
            body_polygon_abs=getattr(p, "body_polygon_abs", None),
            clearance_polygon_abs=getattr(p, "clearance_polygon_abs", None),
            entries=list(getattr(p, "entries", [])),
            exits=list(getattr(p, "exits", [])),
        ))

    # --- Forbidden areas ---
    forbidden_rects: List[Tuple[int, int, int, int]] = []
    if hasattr(engine, "forbidden_areas") and isinstance(engine.forbidden_areas, list):
        for a in engine.forbidden_areas:
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            if rect is None:
                continue
            x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            if x1 > x0 and y1 > y0:
                forbidden_rects.append((x0, y0, x1, y1))

    # --- Constraint zones ---
    constraint_zones: dict = {}
    constraints = getattr(engine, "zone_constraints", None)
    if isinstance(constraints, dict):
        maps_obj = engine.get_maps()
        constraint_maps = getattr(maps_obj, "constraint_maps", {})
        for name in c_names:
            cfg = constraints[name]
            color = constraint_color(name, c_names)
            heatmap = None
            if name in constraint_maps:
                arr = constraint_maps[name].detach().cpu().numpy().astype(np.float32, copy=False)
                heatmap = arr
            rects = []
            areas = cfg.get("areas", [])
            op = str(cfg.get("op", ""))
            if isinstance(areas, list):
                for area in areas:
                    if isinstance(area, dict):
                        rects.append(area)
            constraint_zones[name] = ConstraintZoneData(
                name=str(name),
                op=op,
                color=color,
                heatmap=heatmap,
                rects=rects,
            )

    # --- Masks ---
    maps = engine.get_maps()
    inv = _tensor_to_numpy(maps.invalid)
    if inv is not None:
        inv = inv.astype(bool)

    clr_raw = _tensor_to_numpy(maps.clear_invalid)
    occ_raw = _tensor_to_numpy(maps.occ_invalid)
    clr_mask = None
    if clr_raw is not None and occ_raw is not None:
        clr_mask = (clr_raw.astype(bool) & (~occ_raw.astype(bool)))

    # --- Flow arrows & ports ---
    flow_arrows: List[FlowArrow] = []
    ports = PortData()

    if state.placed:
        port_pairs = state.flow.flow_port_pairs
        for src, targets in engine.group_flow.items():
            if src not in state.placed:
                continue
            src_p = state.placements[src]
            for dst, weight in targets.items():
                if dst not in state.placed:
                    continue
                dst_p = state.placements[dst]
                cached = port_pairs.get((src, dst))
                if cached:
                    draw_pairs = cached
                else:
                    draw_pairs = [((src_p.x_c, src_p.y_c), (dst_p.x_c, dst_p.y_c))]
                for (sx, sy), (dx, dy) in draw_pairs:
                    flow_arrows.append(FlowArrow(
                        src_xy=(float(sx), float(sy)),
                        dst_xy=(float(dx), float(dy)),
                        weight=float(weight),
                    ))

        # Ports
        active_entries: set = set()
        active_exits: set = set()
        for cached in port_pairs.values():
            if not isinstance(cached, list):
                continue
            for pair in cached:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    continue
                exit_xy, entry_xy = pair
                active_exits.add(_coord_key(exit_xy[0], exit_xy[1]))
                active_entries.add(_coord_key(entry_xy[0], entry_xy[1]))

        for gid in state.placed:
            p = state.placements[gid]
            for x, y in getattr(p, "entries", []):
                if _coord_key(x, y) in active_entries:
                    ports.active_entries.append((float(x), float(y)))
                else:
                    ports.inactive_entries.append((float(x), float(y)))
            for x, y in getattr(p, "exits", []):
                if _coord_key(x, y) in active_exits:
                    ports.active_exits.append((float(x), float(y)))
                else:
                    ports.inactive_exits.append((float(x), float(y)))

    # --- Cost ---
    cost = float(engine.cost())

    # --- Action space candidates ---
    candidates_xy: Optional[List[Tuple[float, float]]] = None
    if action_space is not None:
        poses = action_space.poses[action_space.mask]
        if int(poses.shape[0]) > 0:
            candidates_xy = []
            for row in poses.detach().cpu().tolist():
                candidates_xy.append((float(row[0]), float(row[1])))

    # --- Routes ---
    route_data: Optional[List[RouteData]] = None
    if routes is not None:
        route_data = []
        for i, route in enumerate(routes):
            if not route.success or route.path is None:
                continue
            path = route.path
            if len(path) < 2:
                continue
            color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            route_data.append(RouteData(
                src_group=str(route.src_group),
                dst_group=str(route.dst_group),
                path=[(float(p[0]), float(p[1])) for p in path],
                color=color,
            ))

    return LayoutData(
        grid_width=int(engine.grid_width),
        grid_height=int(engine.grid_height),
        facilities=facilities,
        forbidden_rects=forbidden_rects,
        constraint_zones=constraint_zones,
        constraint_names=c_names,
        invalid_mask=inv,
        clearance_mask=clr_mask,
        flow_arrows=flow_arrows,
        ports=ports,
        cost=cost,
        candidates_xy=candidates_xy,
        routes=route_data,
    )
