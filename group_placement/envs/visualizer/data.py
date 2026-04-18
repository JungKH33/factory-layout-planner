"""Backend-agnostic data extraction from engine state.

Converts engine internals into plain dataclasses that visualization backends
can consume without importing torch, envs, or any engine code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, List, Tuple

import numpy as np

from group_placement.envs.visualizer.layer import RewardLayer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FacilityData:
    group_id: str
    x_bl: float
    y_bl: float
    w: float
    h: float
    x_center: float
    y_center: float
    body_polygon_abs: Optional[List[Tuple[float, float]]] = None
    clearance_polygon_abs: Optional[List[Tuple[float, float]]] = None
    clearance_left: int = 0
    clearance_right: int = 0
    clearance_bottom: int = 0
    clearance_top: int = 0
    entry_points: List[Tuple[float, float]] = field(default_factory=list)
    exit_points: List[Tuple[float, float]] = field(default_factory=list)


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
    reward_layers: Dict[str, RewardLayer] = field(default_factory=dict)


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


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)
        else:
            dst[k] = deepcopy(v)


def parse_flow_edge_key(raw_key: object) -> Optional[Tuple[str, str]]:
    if isinstance(raw_key, str) and "->" in raw_key:
        lhs, rhs = raw_key.split("->", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs and rhs:
            return lhs, rhs
    return None


def extract_flow_port_pairs_from_edge_meta(
    edge_meta: Dict[str, Any],
    *,
    model_priority: Tuple[str, ...] = ("routed", "estimated"),
    src_placement: Optional[Any] = None,
    dst_placement: Optional[Any] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    models = edge_meta.get("models", None)
    if not isinstance(models, dict):
        return []
    raw_pairs: object = None
    raw_pair_indices: object = None
    raw_pair_indices_encoded: object = None
    for model_name in model_priority:
        model = models.get(str(model_name), None)
        if not isinstance(model, dict):
            continue
        if isinstance(model.get("port_pairs", None), list):
            raw_pairs = model["port_pairs"]
            break
        if isinstance(model.get("pair_indices", None), list):
            raw_pair_indices = model["pair_indices"]
            break
        encoded = model.get("pair_indices_encoded", None)
        if isinstance(encoded, str):
            raw_pair_indices_encoded = encoded
            break
    if raw_pairs is None and raw_pair_indices is None and raw_pair_indices_encoded is None:
        return []

    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if raw_pairs is not None:
        for row in raw_pairs:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                continue
            ex, en = row
            if not isinstance(ex, (list, tuple)) or not isinstance(en, (list, tuple)):
                continue
            if len(ex) != 2 or len(en) != 2:
                continue
            out.append((
                (float(ex[0]), float(ex[1])),
                (float(en[0]), float(en[1])),
            ))
        return out
    if raw_pair_indices is None and raw_pair_indices_encoded is not None:
        parsed: List[List[int]] = []
        if raw_pair_indices_encoded.strip():
            for tok in raw_pair_indices_encoded.split(","):
                pair = tok.strip()
                if not pair or ":" not in pair:
                    continue
                lhs, rhs = pair.split(":", 1)
                try:
                    parsed.append([int(lhs), int(rhs)])
                except Exception:
                    continue
        raw_pair_indices = parsed
    if raw_pair_indices is None:
        return out
    if src_placement is None or dst_placement is None:
        return out
    src_exits = list(getattr(src_placement, "exit_points", []) or [])
    dst_entries = list(getattr(dst_placement, "entry_points", []) or [])
    for row in raw_pair_indices:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        s_idx = int(row[0])
        d_idx = int(row[1])
        if s_idx < 0 or d_idx < 0:
            continue
        if s_idx >= len(src_exits) or d_idx >= len(dst_entries):
            continue
        sx, sy = src_exits[s_idx]
        dx, dy = dst_entries[d_idx]
        out.append((
            (float(sx), float(sy)),
            (float(dx), float(dy)),
        ))
    return out


def extract_flow_edges_and_pairs(
    eval_state: Any,
    *,
    phase: str = "base",
    model_priority: Tuple[str, ...] = ("routed", "estimated"),
    placements_by_gid: Optional[Mapping[str | int, Any]] = None,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], List[Tuple[Tuple[float, float], Tuple[float, float]]]],
]:
    """Interpret reward flow metadata for visualizer/explorer consumers only."""
    edges_out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    phase_key = str(phase)
    if phase_key == "base":
        sources = [("base", getattr(eval_state, "base_rewards", {}))]
    elif phase_key == "terminal":
        sources = [("terminal", getattr(eval_state, "terminal_rewards", {}))]
    elif phase_key == "merged":
        sources = [
            ("base", getattr(eval_state, "base_rewards", {})),
            ("terminal", getattr(eval_state, "terminal_rewards", {})),
        ]
    else:
        raise ValueError(f"phase must be 'base'|'terminal'|'merged', got {phase!r}")

    for phase_name, comp_map in sources:
        if not isinstance(comp_map, dict):
            continue
        for comp_name, rec in comp_map.items():
            if not isinstance(rec, dict):
                continue
            metadata = rec.get("metadata", None)
            if not isinstance(metadata, dict):
                continue
            raw_edges = metadata.get("edges", None)
            if not isinstance(raw_edges, dict):
                continue
            for raw_key, raw_edge in raw_edges.items():
                key = parse_flow_edge_key(raw_key)
                if key is None:
                    continue
                src, dst = key
                edge_data = dict(raw_edge or {}) if isinstance(raw_edge, dict) else {}
                merged = edges_out.get((src, dst))
                if merged is None:
                    merged = {}
                    edges_out[(src, dst)] = merged
                _deep_merge_dict(merged, edge_data)
                contrib = merged.get("components", None)
                if not isinstance(contrib, list):
                    contrib = []
                    merged["components"] = contrib
                if str(comp_name) not in contrib:
                    contrib.append(str(comp_name))
                phase_src = merged.get("phases", None)
                if not isinstance(phase_src, list):
                    phase_src = []
                    merged["phases"] = phase_src
                if phase_name not in phase_src:
                    phase_src.append(phase_name)

    pairs_out: Dict[Tuple[str, str], List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    for key, edge_meta in edges_out.items():
        src_p = None
        dst_p = None
        if placements_by_gid is not None:
            src_p = placements_by_gid.get(key[0], None)
            dst_p = placements_by_gid.get(key[1], None)
            if src_p is None:
                src_p = placements_by_gid.get(str(key[0]), None)
            if dst_p is None:
                dst_p = placements_by_gid.get(str(key[1]), None)
        pairs = extract_flow_port_pairs_from_edge_meta(
            edge_meta,
            model_priority=model_priority,
            src_placement=src_p,
            dst_placement=dst_p,
        )
        if pairs:
            pairs_out[key] = pairs
    return edges_out, pairs_out


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_layout_data(
    engine: Any,
    *,
    action_space: Any = None,
    terminal_composer: Any = None,
) -> LayoutData:
    """Extract all rendering data from engine state into a plain LayoutData.

    Args:
        engine: FactoryLayoutEnv instance
        action_space: Optional ActionSpace (with .centers, .valid_mask, .group_id)
    """
    state = engine.get_state()
    c_names = _constraint_names(engine)

    # --- Facilities ---
    facilities: List[FacilityData] = []
    for gid in state.placed:
        p = state.placements[gid]
        facilities.append(FacilityData(
            group_id=str(gid),
            x_bl=float(getattr(p, "x_bl", p.min_x)),
            y_bl=float(getattr(p, "y_bl", p.min_y)),
            w=float(getattr(p, "w", p.max_x - p.min_x)),
            h=float(getattr(p, "h", p.max_y - p.min_y)),
            x_center=float(p.x_center),
            y_center=float(p.y_center),
            body_polygon_abs=getattr(p, "body_polygon_abs", None),
            clearance_polygon_abs=getattr(p, "clearance_polygon_abs", None),
            clearance_left=int(getattr(p, "clearance_left", 0)),
            clearance_right=int(getattr(p, "clearance_right", 0)),
            clearance_bottom=int(getattr(p, "clearance_bottom", 0)),
            clearance_top=int(getattr(p, "clearance_top", 0)),
            entry_points=list(getattr(p, "entry_points", [])),
            exit_points=list(getattr(p, "exit_points", [])),
        ))

    # --- Forbidden areas ---
    forbidden_rects: List[Tuple[int, int, int, int]] = []
    if hasattr(engine, "forbidden") and isinstance(engine.forbidden, list):
        for a in engine.forbidden:
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
        flow_edges, flow_pairs = extract_flow_edges_and_pairs(
            state.eval,
            phase="base",
            placements_by_gid=state.placements,
        )
        for edge_key, pairs in flow_pairs.items():
            edge = flow_edges.get(edge_key, {})
            weight = float(edge.get("weight", 0.0))
            pair_count = int(edge.get("pair_count", len(pairs))) if len(pairs) > 0 else 0
            if pair_count <= 0:
                pair_count = len(pairs) if len(pairs) > 0 else 1
            pair_weight = weight / float(pair_count)
            for (sx, sy), (dx, dy) in pairs:
                flow_arrows.append(FlowArrow(
                    src_xy=(float(sx), float(sy)),
                    dst_xy=(float(dx), float(dy)),
                    weight=pair_weight,
                ))

        # Ports
        active_entries: set = set()
        active_exits: set = set()
        for pairs in flow_pairs.values():
            for exit_xy, entry_xy in pairs:
                active_exits.add(_coord_key(exit_xy[0], exit_xy[1]))
                active_entries.add(_coord_key(entry_xy[0], entry_xy[1]))

        for gid in state.placed:
            p = state.placements[gid]
            for x, y in getattr(p, "entry_points", []):
                if _coord_key(x, y) in active_entries:
                    ports.active_entries.append((float(x), float(y)))
                else:
                    ports.inactive_entries.append((float(x), float(y)))
            for x, y in getattr(p, "exit_points", []):
                if _coord_key(x, y) in active_exits:
                    ports.active_exits.append((float(x), float(y)))
                else:
                    ports.inactive_exits.append((float(x), float(y)))

    # --- Reward layers (per-component) ---
    reward_layers: Dict[str, RewardLayer] = {}
    if state.placed:
        try:
            from group_placement.envs.visualizer.reward_layers import build_reward_layers
            reward_composer = getattr(engine, "reward_composer", None)
            t_composer = terminal_composer or getattr(engine, "terminal_reward_composer", None)
            reward_layers = build_reward_layers(
                reward_composer, t_composer, state, maps,
            )
        except Exception:
            pass

    # --- Shim: derive flow_arrows/ports from base flow layer (backward compat) ---
    if not flow_arrows and reward_layers:
        for layer in reward_layers.values():
            if layer.phase != "base":
                continue
            for seg in layer.segments:
                flow_arrows.append(FlowArrow(
                    src_xy=seg.src_xy,
                    dst_xy=seg.dst_xy,
                    weight=seg.weight,
                ))
            for port in layer.ports:
                if port.kind == "entry":
                    if port.active:
                        ports.active_entries.append(port.xy)
                    else:
                        ports.inactive_entries.append(port.xy)
                else:
                    if port.active:
                        ports.active_exits.append(port.xy)
                    else:
                        ports.inactive_exits.append(port.xy)
            break

    # --- Cost ---
    cost = float(engine.cost())

    # --- Action space candidates ---
    candidates_xy: Optional[List[Tuple[float, float]]] = None
    if action_space is not None:
        poses = action_space.centers[action_space.valid_mask]
        if int(poses.shape[0]) > 0:
            candidates_xy = []
            for row in poses.detach().cpu().tolist():
                candidates_xy.append((float(row[0]), float(row[1])))

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
        reward_layers=reward_layers,
    )
