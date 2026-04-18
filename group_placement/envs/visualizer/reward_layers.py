"""Per-reward-component layer builders.

All visualization logic lives here; reward components are not touched.
Dispatch is by component type (isinstance).
Visualizer is read-only: it reads EvalState metadata and renders it.
No computation (wavefront, backtracking, etc.) is performed here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from group_placement.envs.visualizer.layer import RewardLayer, RewardVizPort, RewardVizSegment
from group_placement.envs.visualizer.data import (
    extract_flow_port_pairs_from_edge_meta,
    parse_flow_edge_key,
    _coord_key,
)


_FLOW_COLOR = "royalblue"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_terminal_polyline(
    raw_polylines: Any,
) -> Optional[List[Tuple[float, float]]]:
    """Return first valid polyline from a terminal polylines list, or None."""
    if not isinstance(raw_polylines, list) or not raw_polylines:
        return None
    first = raw_polylines[0]
    if not isinstance(first, list) or not first:
        return None
    out: List[Tuple[float, float]] = []
    for pt in first:
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out.append((float(pt[0]), float(pt[1])))
    return out if len(out) >= 2 else None


def _build_flow_layer(
    *,
    name: str,
    metadata: Dict[str, Any],
    placements: Dict[Any, Any],
    phase: str,
    model_priority: Tuple[str, ...],
    style: str,
    default_visible: bool,
) -> Optional[RewardLayer]:
    """Build a RewardLayer from flow-style edges metadata."""
    raw_edges = metadata.get("edges", None)
    if not isinstance(raw_edges, dict) or not raw_edges:
        return None

    label_phase = "flow" if phase == "base" else "flow/terminal"
    label = f"{label_phase}:{name}"

    segments: List[RewardVizSegment] = []
    active_exits: set = set()
    active_entries: set = set()

    for raw_key, raw_edge in raw_edges.items():
        edge_key = parse_flow_edge_key(raw_key)
        if edge_key is None:
            continue
        src_gid, dst_gid = edge_key
        if not isinstance(raw_edge, dict):
            continue

        weight = float(raw_edge.get("weight", 0.0))
        pair_count = int(raw_edge.get("pair_count", 1)) or 1

        # Terminal: try polyline first
        terminal_model = raw_edge.get("models", {}).get("terminal", None)
        polyline: Optional[List[Tuple[float, float]]] = None
        if isinstance(terminal_model, dict):
            raw_pl = terminal_model.get("polylines", None)
            polyline = _resolve_terminal_polyline(raw_pl)

        # Resolve port-pair coordinates
        src_placement = placements.get(src_gid) or placements.get(str(src_gid))
        dst_placement = placements.get(dst_gid) or placements.get(str(dst_gid))
        pairs = extract_flow_port_pairs_from_edge_meta(
            raw_edge,
            model_priority=model_priority,
            src_placement=src_placement,
            dst_placement=dst_placement,
        )

        pair_weight = weight / float(pair_count)
        if pairs:
            for ex_xy, en_xy in pairs:
                active_exits.add(_coord_key(*ex_xy))
                active_entries.add(_coord_key(*en_xy))
                seg_polyline = polyline  # one polyline per edge (best pair only)
                polyline = None  # only attach polyline to first pair
                segments.append(RewardVizSegment(
                    src_xy=ex_xy,
                    dst_xy=en_xy,
                    weight=pair_weight,
                    polyline=seg_polyline,
                ))
        elif polyline is not None:
            # have polyline but no pair coords — use polyline endpoints
            src_xy = polyline[0]
            dst_xy = polyline[-1]
            segments.append(RewardVizSegment(
                src_xy=src_xy,
                dst_xy=dst_xy,
                weight=weight,
                polyline=polyline,
            ))

    if not segments:
        return None

    # Build port activity lists (only for base flow layers)
    ports: List[RewardVizPort] = []
    if phase == "base":
        for gid, placement in placements.items():
            for x, y in getattr(placement, "entry_points", []):
                active = _coord_key(x, y) in active_entries
                ports.append(RewardVizPort(xy=(float(x), float(y)), kind="entry", active=active))
            for x, y in getattr(placement, "exit_points", []):
                active = _coord_key(x, y) in active_exits
                ports.append(RewardVizPort(xy=(float(x), float(y)), kind="exit", active=active))

    return RewardLayer(
        key=name,
        label=label,
        color=_FLOW_COLOR,
        style=style,
        default_visible=default_visible,
        phase=phase,
        segments=segments,
        ports=ports,
    )


# ---------------------------------------------------------------------------
# Component-specific builders
# ---------------------------------------------------------------------------

def _build_flow_reward_layer(
    *,
    name: str,
    eval_state: Any,
    placements: Dict[Any, Any],
) -> Optional[RewardLayer]:
    base_rewards = getattr(eval_state, "base_rewards", {})
    rec = base_rewards.get(name, None) or base_rewards.get(str(name), None)
    if not isinstance(rec, dict):
        return None
    metadata = rec.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    return _build_flow_layer(
        name=name,
        metadata=metadata,
        placements=placements,
        phase="base",
        model_priority=("routed", "estimated"),
        style="dashed",
        default_visible=True,
    )


def _build_terminal_flow_layer(
    *,
    name: str,
    eval_state: Any,
    placements: Dict[Any, Any],
) -> Optional[RewardLayer]:
    """Read terminal flow metadata from EvalState and build a RewardLayer.

    Purely read-only: no computation is triggered here.
    Polylines must already be present in eval_state (populated by env at finalization).
    """
    terminal_rewards = getattr(eval_state, "terminal_rewards", {})
    rec = terminal_rewards.get(name, None) or terminal_rewards.get(str(name), None)
    if not isinstance(rec, dict):
        return None
    metadata = rec.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    return _build_flow_layer(
        name=name,
        metadata=metadata,
        placements=placements,
        phase="terminal",
        model_priority=("routed", "terminal", "estimated"),
        style="solid",
        default_visible=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_reward_layers(
    reward_composer: Any,
    terminal_composer: Any,
    state: Any,
    maps: Any,
) -> Dict[str, RewardLayer]:
    """Build per-component RewardLayer dict from EvalState metadata.

    Args:
        reward_composer:  RewardComposer with .components dict
        terminal_composer: TerminalRewardComposer (or None)
        state:  EnvState — supplies eval, placements
        maps:   GridMaps (unused; kept for API compatibility)
    """
    from group_placement.envs.reward.flow import FlowReward
    from group_placement.envs.reward.terminal import TerminalFlowReward

    if reward_composer is None or state is None:
        return {}

    eval_state = state.eval
    placements = state.placements if hasattr(state, "placements") else {}

    out: Dict[str, RewardLayer] = {}

    # Base components — always visible when any facility is placed
    for name, comp in getattr(reward_composer, "components", {}).items():
        name_s = str(name)
        if isinstance(comp, FlowReward):
            layer = _build_flow_reward_layer(
                name=name_s,
                eval_state=eval_state,
                placements=placements,
            )
            if layer is not None:
                out[name_s] = layer

    # Terminal components — only after the episode is finalized.
    # Uses prefix "terminal:" to avoid key collision with base components.
    if terminal_composer is not None:
        finalized = bool(
            getattr(eval_state, "objective", {}).get("finalized", False)
        )
        if finalized:
            for name, comp in getattr(terminal_composer, "components", {}).items():
                name_s = str(name)
                if isinstance(comp, TerminalFlowReward):
                    layer = _build_terminal_flow_layer(
                        name=name_s,
                        eval_state=eval_state,
                        placements=placements,
                    )
                    if layer is not None:
                        out[f"terminal:{name_s}"] = layer

    return out
