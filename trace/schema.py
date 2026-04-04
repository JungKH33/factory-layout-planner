"""Unified data structures for decision tracing and exploration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.action_space import ActionSpace
from envs.state import EnvState

# SearchOutput lives in search.base to avoid circular imports; re-export here.
from search.base import SearchOutput  # noqa: F401


# ---------------------------------------------------------------------------
# Signal — one source's recommendation at a decision point
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """One source's evaluation and recommendation at a decision point.

    Sources: ``"agent"``, ``"search:mcts"``, ``"search:beam"``, ``"human"``,
    ``"llm"``, etc.
    """

    source: str

    scores: np.ndarray
    """[N] normalised preference / probability per action."""

    recommended_action: int = -1
    recommended_value: float = 0.0

    values: Optional[np.ndarray] = None
    """[N] Q-values or expected returns (optional, source-dependent)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    # agent:  {"value_estimate": float}
    # search: {"algorithm": str, "iterations": int, "visits": [...], "top_k": [...]}
    # llm:    {"reasoning": str, "model": str}
    # human:  {"comment": str}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source": self.source,
            "scores": self.scores.tolist(),
            "recommended_action": self.recommended_action,
            "recommended_value": self.recommended_value,
        }
        if self.values is not None:
            d["values"] = self.values.tolist()
        if self.metadata:
            safe = {}
            for k, v in self.metadata.items():
                if isinstance(v, np.ndarray):
                    safe[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    safe[k] = v
                else:
                    safe[k] = str(v)
            d["metadata"] = safe
        return d


# ---------------------------------------------------------------------------
# Snapshot — engine + adapter state for restore
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """Restorable engine + adapter state at a decision point."""

    engine_state: EnvState
    adapter_state: Dict[str, object]
    action_space: Optional[ActionSpace] = None


# ---------------------------------------------------------------------------
# DecisionNode — one step in the placement decision tree
# ---------------------------------------------------------------------------

@dataclass
class DecisionNode:
    """A single decision point in the placement sequence."""

    id: int
    parent_id: Optional[int]
    step: int
    """Placement step index (0-based)."""

    group_id: Optional[str] = None
    """Facility group being placed at this node."""

    valid_actions: int = 0
    """Number of valid candidate positions."""

    # --- signals from various sources ---
    signals: Dict[str, Signal] = field(default_factory=dict)

    # --- chosen action ---
    chosen_action: Optional[int] = None
    chosen_by: Optional[str] = None
    reward: float = 0.0
    cum_reward: float = 0.0
    cost_after: Optional[float] = None

    terminal: bool = False

    # --- tree structure ---
    children: Dict[int, int] = field(default_factory=dict)
    """action_index → child node id."""

    _snapshot: Optional[Snapshot] = field(default=None, repr=False)
    """Lazily saved state for restoration (branch points / every node)."""


# ---------------------------------------------------------------------------
# DecisionTree — full branching decision history
# ---------------------------------------------------------------------------

@dataclass
class DecisionTree:
    """Complete tree of placement decisions with branching support."""

    nodes: Dict[int, DecisionNode] = field(default_factory=dict)
    root_id: int = 0
    active_id: int = 0
    branches: Dict[str, List[int]] = field(default_factory=dict)
    """Named paths: ``{"main": [0, 1, 2], "alt": [0, 1, 5, 6]}``."""

    _next_id: int = 0

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def active_node(self) -> DecisionNode:
        return self.nodes[self.active_id]

    def path_to(self, node_id: int) -> List[int]:
        """Return node ids from root to *node_id* (inclusive)."""
        path: List[int] = []
        nid: Optional[int] = node_id
        while nid is not None:
            path.append(nid)
            nid = self.nodes[nid].parent_id
        path.reverse()
        return path

    def terminal_nodes(self) -> List[DecisionNode]:
        """Return all terminal (leaf with no children) nodes."""
        return [n for n in self.nodes.values() if not n.children and n.terminal]


# ---------------------------------------------------------------------------
# TraceEvent — lightweight event for real-time streaming
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """Event emitted by Explorer for external consumers (WebUI, loggers)."""

    type: str
    """``"step"``, ``"undo"``, ``"signal_updated"``, ``"branch_created"``,
    ``"search_progress"``, ``"reset"``."""

    node_id: int
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "node_id": self.node_id, "data": self.data}
