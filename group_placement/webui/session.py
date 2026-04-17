"""Session management for WebUI."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from group_placement.envs.env_loader import load_env
from group_placement.envs.action_space import ActionSpace
from group_placement.agents.placement.greedy import GreedyAgent, GreedyAdapter, GreedyV2Adapter, GreedyV3Adapter
from group_placement.agents.placement.alphachip import AlphaChipAgent, AlphaChipAdapter
from group_placement.agents.placement.maskplace import MaskPlaceAgent, MaskPlaceAdapter
from group_placement.agents.ordering import DifficultyOrderingAgent

from group_placement.search.mcts import MCTSConfig, MCTSSearch
from group_placement.search.beam import BeamConfig, BeamSearch

from group_placement.envs.visualizer.data import extract_flow_edges_and_pairs
from group_placement.trace.explorer import Explorer

from group_placement.webui.schemas import (
    SessionCreateRequest,
    SessionState,
    PlacedFacility,
    PortInfo,
    CandidateInfo,
    SearchProgress,
    ZoneRect,
    FlowEdge,
    FlowDeltaInfo,
    PhysicalContextInfo,
)


@dataclass
class Session:
    """A single interactive session backed by Explorer."""
    sid: str
    explorer: Explorer
    device: torch.device
    reset_kwargs: Dict[str, Any]

    # WebSocket connections
    websockets: List[Any] = field(default_factory=list)

    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    search_in_progress: bool = False
    search_timeline: List[Dict[str, Any]] = field(default_factory=list)
    topk_runs: List[Dict[str, Any]] = field(default_factory=list)
    topk_state_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    topk_run_seq: int = 0

    def can_undo(self) -> bool:
        node = self.explorer.current()
        return node.parent_id is not None

    def can_redo(self) -> bool:
        return len(self.explorer._redo_stack) > 0

    def undo(self) -> bool:
        result = self.explorer.undo()
        return result is not None

    def redo(self) -> bool:
        result = self.explorer.redo()
        return result is not None

    def get_state(self) -> SessionState:
        """Build SessionState for API response."""
        engine = self.explorer.engine
        adapter = self.explorer.adapter
        node = self.explorer.current()
        state = engine.get_state()

        # Placed facilities — include port info from GroupPlacement
        placed = []
        for gid in state.placed:
            p = state.placements[gid]
            entries = [PortInfo(x=float(pt[0]), y=float(pt[1]))
                       for pt in getattr(p, "entry_points", [])]
            exits = [PortInfo(x=float(pt[0]), y=float(pt[1]))
                     for pt in getattr(p, "exit_points", [])]
            placed.append(PlacedFacility(
                gid=str(gid),
                x=float(p.x_bl),
                y=float(p.y_bl),
                w=float(p.w),
                h=float(p.h),
                rot=int(p.rotation),
                x_center=float(p.x_center),
                y_center=float(p.y_center),
                entries=entries,
                exits=exits,
                variant_index=int(getattr(p, "variant_index", 0)),
            ))

        # Candidates from node snapshot or build fresh
        candidates = []
        action_space = None
        if not node.terminal:
            if node._snapshot and node._snapshot.action_space is not None:
                action_space = node._snapshot.action_space
            else:
                try:
                    adapter.build_observation()
                    action_space = adapter.build_action_space()
                except Exception:
                    pass

        # Agent scores/value from signal (if available)
        agent_sig = node.signals.get("agent")
        scores_np = agent_sig.scores if agent_sig else None
        value = agent_sig.metadata.get("value_estimate", 0.0) if agent_sig else 0.0

        # Search signal visits/values (if available)
        search_sig = None
        for k, sig in node.signals.items():
            if k.startswith("search:"):
                search_sig = sig
                break
        search_visits = search_sig.metadata.get("visits") if search_sig else None
        search_values = search_sig.values if search_sig else None

        if action_space is not None:
            mask = action_space.valid_mask.detach().cpu().numpy()
            poses = action_space.centers.detach().cpu().numpy()
            n = len(poses)

            for i in range(n):
                x_center, y_center = float(poses[i, 0]), float(poses[i, 1])
                score = float(scores_np[i]) if (scores_np is not None and i < len(scores_np)) else 0.0
                vis = int(search_visits[i]) if (search_visits is not None and i < len(search_visits)) else 0
                qv = float(search_values[i]) if (search_values is not None and i < len(search_values)) else 0.0
                candidates.append(CandidateInfo(
                    index=i,
                    x=x_center,
                    y=y_center,
                    rot=0,
                    score=score,
                    valid=bool(mask[i]),
                    visits=vis,
                    q_value=qv,
                ))

        current_gid = state.remaining[0] if state.remaining else None

        def _zone_rect_from_dict(a: dict, *, id_value: str | None = None) -> ZoneRect | None:
            if "rect" not in a:
                return None
            r = a["rect"]
            if not (isinstance(r, (list, tuple)) and len(r) == 4):
                return None
            raw_val = a.get("value", None)
            val: float | None = None
            if raw_val is not None and isinstance(raw_val, (int, float)):
                val = float(raw_val)
            return ZoneRect(
                x0=float(r[0]), y0=float(r[1]),
                x1=float(r[2]), y1=float(r[3]),
                value=val,
                id=id_value,
            )

        # Extract forbidden from engine
        forbidden_out: list[ZoneRect] = []
        areas = getattr(engine, "forbidden", None)
        if isinstance(areas, list):
            for a in areas:
                if not isinstance(a, dict):
                    continue
                z = _zone_rect_from_dict(a)
                if z is not None:
                    forbidden_out.append(z)

        # Extract generic constraints zones
        constraint_zones: Dict[str, List[ZoneRect]] = {}
        constraints = getattr(engine, "zone_constraints", None)
        if isinstance(constraints, dict):
            for cname, cfg in constraints.items():
                if not isinstance(cfg, dict):
                    continue
                op = str(cfg.get("op", ""))
                c_zones: list[ZoneRect] = []
                areas = cfg.get("areas", [])
                if isinstance(areas, list):
                    for a in areas:
                        if not isinstance(a, dict):
                            continue
                        raw_val = a.get("value", None)
                        label_val = str(raw_val) if raw_val is not None else ""
                        label = f"{cname}{op}{label_val}" if label_val else str(cname)
                        z = _zone_rect_from_dict(a, id_value=label)
                        if z is not None:
                            c_zones.append(z)
                constraint_zones[str(cname)] = c_zones

        # Extract flow edges with positions
        flow_edges = []
        edge_meta, edge_pairs = extract_flow_edges_and_pairs(state.eval, phase="base")
        for edge_key, meta in edge_meta.items():
            src, dst = edge_key
            edge = FlowEdge(src=str(src), dst=str(dst), weight=float(meta.get("weight", 0.0)))
            pairs = edge_pairs.get(edge_key, [])
            if pairs:
                (sx, sy), (dx, dy) = pairs[0]
                edge.src_x = float(sx)
                edge.src_y = float(sy)
                edge.dst_x = float(dx)
                edge.dst_y = float(dy)
            flow_edges.append(edge)

        # Physical context from the parent node (the node that performed the step to reach here)
        last_physical_info = None
        if node.parent_id is not None:
            parent = self.explorer.tree.nodes[node.parent_id]
            phys = parent.physical
            if phys is not None:
                last_physical_info = PhysicalContextInfo(
                    gid=phys.gid,
                    x=phys.x, y=phys.y, w=phys.w, h=phys.h,
                    rotation=phys.rotation, variant_index=phys.variant_index,
                    x_center=phys.x_center, y_center=phys.y_center,
                    entries=[PortInfo(x=pt[0], y=pt[1]) for pt in phys.entries],
                    exits=[PortInfo(x=pt[0], y=pt[1]) for pt in phys.exits],
                    delta_cost=phys.delta_cost,
                    cost_before=phys.cost_before, cost_after=phys.cost_after,
                    affected_flows=[
                        FlowDeltaInfo(src=fd.src, dst=fd.dst, weight=fd.weight, distance=fd.distance)
                        for fd in phys.affected_flows
                    ],
                )

        return SessionState(
            grid_width=int(engine.grid_width),
            grid_height=int(engine.grid_height),
            placed=placed,
            remaining=[str(g) for g in state.remaining],
            current_gid=str(current_gid) if current_gid else None,
            candidates=candidates,
            value=float(value),
            cost=float(engine.cost()),
            step=len(state.placed),
            history_length=len(self.explorer.tree.nodes),
            terminated=node.terminal or len(state.remaining) == 0,
            can_undo=self.can_undo(),
            can_redo=self.can_redo(),
            forbidden=forbidden_out,
            constraint_zones=constraint_zones,
            flow_edges=flow_edges,
            last_physical=last_physical_info,
        )


class SessionManager:
    """Manages multiple sessions."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    async def create_session(self, req: SessionCreateRequest) -> Session:
        """Create a new session."""
        async with self._lock:
            self._counter += 1
            sid = f"session_{self._counter}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = req.params or {}

        # Load environment
        loaded = load_env(
            req.env_json,
            device=device,
            backend_selection=req.backend_selection,
        )
        engine = loaded.env
        engine.log = False

        # Create adapter with dynamic params
        wrapper_params = {k.replace('wrapper_', ''): v
                        for k, v in params.items() if k.startswith('wrapper_')}

        if req.wrapper_mode == "greedy":
            adapter = GreedyAdapter(
                k=wrapper_params.get('k', 50),
                scan_step=wrapper_params.get('scan_step', 2000.0),
                quant_step=wrapper_params.get('quant_step', 10.0),
                p_high=wrapper_params.get('p_high', 0.1),
                p_near=wrapper_params.get('p_near', 0.8),
                p_coarse=wrapper_params.get('p_coarse', 0.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                random_seed=42,
            )
        elif req.wrapper_mode == "greedyv2":
            adapter = GreedyV2Adapter(
                k=wrapper_params.get('k', 50),
                scan_step=wrapper_params.get('scan_step', 2000.0),
                quant_step=wrapper_params.get('quant_step', 10.0),
                p_high=wrapper_params.get('p_high', 0.1),
                p_near=wrapper_params.get('p_near', 0.1),
                p_coarse=wrapper_params.get('p_coarse', 0.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                random_seed=42,
            )
        elif req.wrapper_mode == "greedyv3":
            adapter = GreedyV3Adapter(
                k=wrapper_params.get('k', 50),
                quant_step=wrapper_params.get('quant_step', 10.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                edge_ratio=wrapper_params.get('edge_ratio', 0.8),
                random_seed=42,
            )
        elif req.wrapper_mode == "alphachip":
            adapter = AlphaChipAdapter(
                coarse_grid=wrapper_params.get('coarse_grid', 128),
            )
        elif req.wrapper_mode == "maskplace":
            adapter = MaskPlaceAdapter(
                grid=wrapper_params.get('grid', 224),
                soft_coefficient=wrapper_params.get('soft_coefficient', 1.0),
            )
        else:
            raise ValueError(f"Unknown wrapper_mode: {req.wrapper_mode}")

        # Create agent with dynamic params
        agent_params = {k.replace('agent_', ''): v
                       for k, v in params.items() if k.startswith('agent_')}

        if req.agent_mode == "greedy":
            agent = GreedyAgent(
                prior_temperature=agent_params.get('prior_temperature', 1.0)
            )
        elif req.agent_mode == "alphachip":
            if req.wrapper_mode != "alphachip":
                raise ValueError(
                    f"AlphaChipAgent requires wrapper_mode='alphachip', got '{req.wrapper_mode}'"
                )
            agent = AlphaChipAgent(
                coarse_grid=wrapper_params.get('coarse_grid', 128),
                checkpoint_path=agent_params.get('checkpoint_path'),
                device=device,
            )
        elif req.agent_mode == "maskplace":
            if req.wrapper_mode != "maskplace":
                raise ValueError(
                    f"MaskPlaceAgent requires wrapper_mode='maskplace', got '{req.wrapper_mode}'"
                )
            agent = MaskPlaceAgent(
                device=device,
                grid=wrapper_params.get('grid', 224),
                soft_coefficient=wrapper_params.get('soft_coefficient', 1.0),
                checkpoint_path=agent_params.get('checkpoint_path'),
            )
        else:
            raise ValueError(f"Unknown agent_mode: {req.agent_mode}")

        # Create search with dynamic params
        search_params = {k.replace('search_', ''): v
                        for k, v in params.items() if k.startswith('search_')}

        if req.search_mode == "none":
            search = None
        elif req.search_mode == "mcts":
            search = MCTSSearch(
                config=MCTSConfig(
                    num_simulations=search_params.get('num_simulations', 50),
                    c_puct=search_params.get('c_puct', 2.0),
                    rollout_enabled=search_params.get('rollout_enabled', True),
                    rollout_depth=search_params.get('rollout_depth', 5),
                    dirichlet_epsilon=search_params.get('dirichlet_epsilon', 0.2),
                    dirichlet_concentration=search_params.get('dirichlet_concentration', 0.5),
                    temperature=search_params.get('temperature', 0.0),
                    pw_enabled=search_params.get('pw_enabled', False),
                    pw_c=search_params.get('pw_c', 1.5),
                    pw_alpha=search_params.get('pw_alpha', 0.5),
                    pw_min_children=search_params.get('pw_min_children', 1),
                    cache_decision_state=search_params.get('cache_decision_state', True),
                )
            )
        elif req.search_mode == "beam":
            search = BeamSearch(
                config=BeamConfig(
                    beam_width=search_params.get('beam_width', 8),
                    depth=search_params.get('depth', 5),
                    expansion_topk=search_params.get('expansion_topk', 16),
                    cache_decision_state=search_params.get('cache_decision_state', False),
                )
            )
        else:
            raise ValueError(f"Unknown search_mode: {req.search_mode}")

        ordering_mode = str(params.get("ordering_mode", "none"))
        if ordering_mode == "none":
            ordering_agent = None
        elif ordering_mode == "difficulty":
            ordering_agent = DifficultyOrderingAgent()
        else:
            raise ValueError(f"Unknown ordering_mode: {ordering_mode}")

        # Create Explorer (binds adapter and search internally)
        explorer = Explorer(engine, adapter, agent, search=search, ordering_agent=ordering_agent)
        explorer.reset(options=loaded.reset_kwargs)

        # Populate agent signal for initial candidates/scores
        if not explorer.current().terminal:
            explorer.predict_agent()

        session = Session(
            sid=sid,
            explorer=explorer,
            device=device,
            reset_kwargs=loaded.reset_kwargs,
        )

        async with self._lock:
            self._sessions[sid] = session

        return session

    async def get_session(self, sid: str) -> Session:
        """Get session by ID."""
        async with self._lock:
            if sid not in self._sessions:
                raise KeyError(f"Session not found: {sid}")
            return self._sessions[sid]

    async def delete_session(self, sid: str) -> None:
        """Delete a session."""
        async with self._lock:
            if sid in self._sessions:
                del self._sessions[sid]

    async def list_sessions(self) -> List[str]:
        """List all session IDs."""
        async with self._lock:
            return list(self._sessions.keys())


# Global session manager
manager = SessionManager()
