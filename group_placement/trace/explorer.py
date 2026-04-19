"""Explorer — unified interactive decision explorer.

Replaces ``pipeline.py`` and ``webui/session.py``'s ``HistoryEntry`` with a
single class that owns: prediction, execution, navigation, branching, events.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from group_placement.agents.base import Agent, BaseAdapter, OrderingAgent
from group_placement.envs.action_space import ActionSpace
from group_placement.envs.env import FactoryLayoutEnv
from group_placement.envs.placement.base import GroupPlacement
from group_placement.envs.visualizer.data import extract_flow_edges_and_pairs
from group_placement.search.base import BaseSearch, BaseHierarchicalSearch

from group_placement.trace.schema import (
    DecisionNode,
    DecisionTree,
    FlowDelta,
    PhysicalContext,
    SearchOutput,
    Signal,
    Snapshot,
    TraceEvent,
)

logger = logging.getLogger(__name__)

EventCallback = Callable[[TraceEvent], None]


class Explorer:
    """Pipeline-level decision explorer with branching, undo, and multi-source signals."""

    #: Default top-K candidates captured per signal.
    candidates_top_k: int = 5

    def __init__(
        self,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        agent: Agent,
        search: Optional[BaseSearch] = None,
        ordering_agent: Optional[OrderingAgent] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.engine = engine
        self.adapter = adapter
        self.agent = agent
        self.search = search
        self.ordering_agent = ordering_agent
        self.llm: Optional[Any] = llm  # LLMAdvisor (optional, avoids import)

        self.tree = DecisionTree()

        # bind adapter/search to engine
        self.adapter.bind(engine)
        if self.search is not None:
            self.search.set_adapter(self.adapter)

        self._listeners: List[EventCallback] = []

        # redo stack for linear redo within a branch
        self._redo_stack: List[int] = []

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on(self, callback: EventCallback) -> None:
        self._listeners.append(callback)

    def off(self, callback: EventCallback) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _emit(self, event: TraceEvent) -> None:
        for cb in self._listeners:
            try:
                cb(event)
            except Exception:
                logger.warning("Event listener error", exc_info=True)

    # ------------------------------------------------------------------
    # Reset / state
    # ------------------------------------------------------------------

    def reset(self, *, options: Optional[Dict[str, Any]] = None) -> DecisionNode:
        """Reset environment and create the root decision node."""
        _obs, _info = self.engine.reset(options=options)

        self.tree = DecisionTree()
        self._redo_stack.clear()

        root = self._make_node(parent_id=None, step=0, save_snapshot=True)
        self.tree.nodes[root.id] = root
        self.tree.root_id = root.id
        self.tree.active_id = root.id

        self._emit(TraceEvent(type="reset", node_id=root.id))
        return root

    def rebase_to_current_state(self) -> DecisionNode:
        """Rebuild tree root from the current engine/adapter state."""
        self.tree = DecisionTree()
        self._redo_stack.clear()
        root = self._make_node(
            parent_id=None,
            step=len(self.engine.get_state().placed),
            save_snapshot=True,
        )
        self.tree.nodes[root.id] = root
        self.tree.root_id = root.id
        self.tree.active_id = root.id
        self._emit(TraceEvent(type="reset", node_id=root.id))
        return root

    def current(self) -> DecisionNode:
        return self.tree.active_node()

    def detail(
        self,
        node_id: Optional[int] = None,
        *,
        fallback_to_parent: bool = True,
    ) -> Dict[str, Any]:
        """Return the physical placement detail for *node_id* (default: active).

        Looks up the node, optionally walks to its parent when the node has no
        :class:`PhysicalContext` (typically a freshly-created post-step child
        that hasn't been stepped through yet). Returns a flat dict::

            {"node_id", "gid", "step", "terminal", "physical"}

        where ``physical`` is the result of
        :meth:`PhysicalContext.to_dict` or ``None``. Raises :class:`KeyError`
        for unknown ids.
        """
        target = self.tree.active_id if node_id is None else int(node_id)
        if target not in self.tree.nodes:
            raise KeyError(f"Node {target} not in tree")
        node = self.tree.nodes[target]

        if (
            fallback_to_parent
            and node.physical is None
            and node.parent_id is not None
        ):
            parent = self.tree.nodes.get(node.parent_id)
            if parent is not None and parent.physical is not None:
                node = parent

        return {
            "node_id": node.id,
            "gid": node.group_id,
            "step": node.step,
            "terminal": node.terminal,
            "physical": node.physical.to_dict() if node.physical else None,
        }

    def candidates(
        self,
        node_id: Optional[int] = None,
        *,
        source: str = "agent",
        top_k: Optional[int] = None,
        compute_if_missing: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return cached top-K candidates from *source*'s signal.

        Reads :attr:`Signal.metadata['candidates']` at *node_id* (default:
        active). When *compute_if_missing* is true and the signal is absent
        on the active node, triggers the matching predictor
        (:meth:`predict_agent` / :meth:`predict_search`). Remote nodes are
        never auto-predicted. Returns an empty list for terminal nodes or
        when the signal is unavailable. Each entry is the plain dict built
        by :meth:`_build_candidates`.
        """
        target = self.tree.active_id if node_id is None else int(node_id)
        if target not in self.tree.nodes:
            raise KeyError(f"Node {target} not in tree")
        node = self.tree.nodes[target]
        if node.terminal:
            return []
        sig = node.signals.get(source)
        if sig is None and compute_if_missing and target == self.tree.active_id:
            if source == "agent":
                sig = self.predict_agent()
            elif source.startswith("search") and self.search is not None:
                sig = self.predict_search()
        if sig is None:
            return []
        cands = list(sig.metadata.get("candidates", []))
        if top_k is not None and int(top_k) > 0:
            cands = cands[: int(top_k)]
        return cands

    def explain(self, node_id: Optional[int] = None) -> Dict[str, Any]:
        """Return a structured explanation of the decision at *node_id*.

        Pure read-only composition of ``physical.breakdown`` (per-reward
        contribution), ``Signal.metadata['candidates']`` (top-K considered),
        and the chosen signal into a single dict — intended for LLM tool
        responses and UI rendering. Defaults to the active node when
        *node_id* is ``None``. Raises :class:`KeyError` for unknown ids.
        """
        target = self.tree.active_id if node_id is None else int(node_id)
        if target not in self.tree.nodes:
            raise KeyError(f"Node {target} not in tree")
        node = self.tree.nodes[target]

        signals_out: Dict[str, Dict[str, Any]] = {}
        for name, sig in node.signals.items():
            other_meta = {k: v for k, v in sig.metadata.items() if k != "candidates"}
            signals_out[name] = {
                "recommended_action": int(sig.recommended_action),
                "recommended_value": float(sig.recommended_value),
                "candidates": list(sig.metadata.get("candidates", [])),
                "metadata": other_meta,
            }

        return {
            "node_id": node.id,
            "parent_id": node.parent_id,
            "step": node.step,
            "gid": node.group_id,
            "chosen_action": node.chosen_action,
            "chosen_by": node.chosen_by,
            "terminal": node.terminal,
            "reward": float(node.reward),
            "cum_reward": float(node.cum_reward),
            "cost_after": node.cost_after,
            "physical": node.physical.to_dict() if node.physical else None,
            "signals": signals_out,
        }

    def state_summary(self) -> Dict[str, Any]:
        state = self.engine.get_state()
        node = self.current()
        return {
            "step": node.step,
            "placed": [str(g) for g in state.placed],
            "remaining": [str(g) for g in state.remaining],
            "cost": float(self.engine.cost()),
            "node_id": node.id,
            "terminal": node.terminal,
            "tree_size": len(self.tree.nodes),
            "branches": list(self.tree.branches.keys()),
        }

    # ------------------------------------------------------------------
    # Predictions (separated from execution)
    # ------------------------------------------------------------------

    def predict_agent(self) -> Signal:
        """Get agent's policy, value, and recommended action for current state."""
        if self.ordering_agent is not None:
            self.ordering_agent.reorder(env=self.engine, obs={})

        obs = self.adapter.build_observation()
        action_space = self.adapter.build_action_space()
        valid = self.adapter.num_valid_actions(action_space)
        if valid <= 0:
            n = int(action_space.valid_mask.shape[0]) if action_space.valid_mask.numel() > 0 else 0
            return Signal(
                source="agent",
                scores=np.zeros(n, dtype=np.float32),
                recommended_action=-1,
                recommended_value=0.0,
            )

        scores_t = self.agent.policy(obs=obs, action_space=action_space)
        value = float(self.agent.value(obs=obs, action_space=action_space))
        action = int(self.agent.select_action(obs=obs, action_space=action_space))
        scores = scores_t.detach().cpu().numpy().astype(np.float32)

        # Top-K candidates: rank by delta cost (lower = better) when available,
        # else fall back to ranking by -score (higher score = better).
        action_costs = obs.get("action_costs", None)
        if isinstance(action_costs, torch.Tensor) and int(action_costs.numel()) == int(action_space.centers.shape[0]):
            candidates = self._build_candidates(
                action_space=action_space,
                rank_key=action_costs,
                delta_costs=action_costs,
                score=scores_t,
                chosen_action=action,
            )
        else:
            candidates = self._build_candidates(
                action_space=action_space,
                rank_key=-scores_t,
                score=scores_t,
                chosen_action=action,
            )

        signal = Signal(
            source="agent",
            scores=scores,
            recommended_action=action,
            recommended_value=value,
            metadata={"value_estimate": value, "candidates": candidates},
        )

        node = self.current()
        node.signals["agent"] = signal
        self._emit(TraceEvent(type="signal_updated", node_id=node.id, data={"source": "agent"}))
        return signal

    def predict_search(self, *, progress_interval: int = 10, **config_override: Any) -> Signal:
        """Run search from current state and return a Signal.

        Engine/adapter state is saved before search and restored after, so
        the current decision point is unchanged.
        """
        if self.search is None:
            raise RuntimeError("No search algorithm configured")

        if self.ordering_agent is not None:
            self.ordering_agent.reorder(env=self.engine, obs={})

        obs = self.adapter.build_observation()
        action_space = self.adapter.build_action_space()
        valid = self.adapter.num_valid_actions(action_space)
        if valid <= 0:
            n = int(action_space.valid_mask.shape[0]) if action_space.valid_mask.numel() > 0 else 0
            return Signal(
                source=f"search:{type(self.search).__name__}",
                scores=np.zeros(n, dtype=np.float32),
                recommended_action=-1,
            )

        # save state
        snap_engine = self.engine.get_state().copy()
        snap_adapter = self.adapter.get_state_copy()

        # bridge search progress → TraceEvent
        node = self.current()

        def _on_progress(
            iteration: int, total: int,
            visits: Any, values: Any,
            best_action: int, best_value: float,
        ) -> None:
            self._emit(TraceEvent(
                type="search_progress",
                node_id=node.id,
                data={
                    "iteration": iteration,
                    "total": total,
                    "visits": visits.tolist() if hasattr(visits, "tolist") else [],
                    "values": values.tolist() if hasattr(values, "tolist") else [],
                    "best_action": best_action,
                    "best_value": best_value,
                },
            ))

        try:
            output: SearchOutput = self.search.select(
                obs=obs, agent=self.agent, root_action_space=action_space,
                progress_fn=_on_progress, progress_interval=progress_interval,
            )
        finally:
            # restore state
            self.engine.set_state(snap_engine)
            self.adapter.set_state(snap_adapter)

        algo_name = type(self.search).__name__
        n = int(action_space.valid_mask.shape[0])

        visits = output.visits if output.visits is not None else np.zeros(n, dtype=np.float32)
        values = output.values if output.values is not None else np.zeros(n, dtype=np.float32)

        # normalise visits into scores
        visit_sum = float(visits.sum())
        scores = (visits / visit_sum).astype(np.float32) if visit_sum > 0 else np.zeros(n, dtype=np.float32)

        # Top-K candidates: rank by -visits (more visits = better).
        visits_t = torch.as_tensor(visits, dtype=torch.float32)
        values_t = torch.as_tensor(values, dtype=torch.float32)
        candidates = self._build_candidates(
            action_space=action_space,
            rank_key=-visits_t,
            score=values_t,
            chosen_action=output.action,
        )
        # Carry visits through explicitly alongside value per candidate.
        for entry in candidates:
            entry["visits"] = float(visits_t.view(-1)[entry["action"]].item())

        signal = Signal(
            source=f"search:{algo_name}",
            scores=scores,
            values=values,
            recommended_action=output.action,
            recommended_value=float(values[output.action]) if output.action >= 0 and output.action < len(values) else 0.0,
            metadata={
                "algorithm": algo_name,
                "iterations": output.iterations,
                "visits": visits.tolist(),
                "top_k": output.top_k,
                "worker_action": output.worker_action,
                "candidates": candidates,
            },
        )

        node.signals[f"search:{algo_name}"] = signal
        self._emit(TraceEvent(type="signal_updated", node_id=node.id, data={"source": signal.source}))
        return signal

    def predict_all(self) -> Dict[str, Signal]:
        """Collect signals from agent and search (if configured)."""
        signals: Dict[str, Signal] = {}
        signals["agent"] = self.predict_agent()
        if self.search is not None:
            sig = self.predict_search()
            signals[sig.source] = sig
        return signals

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, action_index: int, chosen_by: str = "human") -> DecisionNode:
        """Execute an action, create child node, advance active pointer."""
        node = self.current()
        if node.terminal:
            raise RuntimeError("Cannot step from a terminal node")

        # check if this action already has a child (revisiting a branch)
        if action_index in node.children:
            child_id = node.children[action_index]
            self._goto_node(child_id)
            return self.tree.nodes[child_id]

        # build action space if needed
        obs = self.adapter.build_observation()
        action_space = self.adapter.build_action_space()

        # resolve & step
        is_hierarchical = isinstance(self.search, BaseHierarchicalSearch) and self.adapter.supports_hierarchical
        worker_action = -1

        signal_meta = node.signals.get(chosen_by, Signal(source="", scores=np.array([]))).metadata
        if (
            is_hierarchical
            and chosen_by.startswith("search:")
            and int(signal_meta.get("worker_action", -1)) >= 0
        ):
            worker_action = int(signal_meta["worker_action"])
            worker_as = self.adapter.sub_action_space(action_index)
            placement = self.adapter.resolve_sub_action(worker_action, worker_as, parent_idx=action_index)
        else:
            placement = self.adapter.resolve_action(action_index, action_space)

        cost_before = float(self.engine.cost())
        records_before = self._snapshot_eval_records()

        _obs, reward, terminated, truncated, info = self.engine.step(placement)

        cost_after = float(self.engine.cost())
        records_after = self._snapshot_eval_records()

        # build PhysicalContext from the resolved placement
        physical = self._build_physical_context(
            placement, cost_before, cost_after,
            records_before=records_before,
            records_after=records_after,
        )

        # record on current node
        node.chosen_action = action_index
        node.chosen_by = chosen_by
        node.reward = float(reward)
        node.physical = physical

        # create child
        parent_cum = node.cum_reward
        child = self._make_node(
            parent_id=node.id,
            step=node.step + 1,
            save_snapshot=True,
        )
        child.cum_reward = parent_cum + float(reward)
        child.cost_after = cost_after
        child.terminal = bool(terminated or truncated)

        self.tree.nodes[child.id] = child
        node.children[action_index] = child.id
        self.tree.active_id = child.id
        self._redo_stack.clear()

        self._emit(TraceEvent(type="step", node_id=child.id, data={
            "action": action_index,
            "chosen_by": chosen_by,
            "reward": float(reward),
            "cost": child.cost_after,
            "terminal": child.terminal,
            "physical": physical.to_dict() if physical else None,
        }))
        return child

    def step_with(self, source: str) -> DecisionNode:
        """Execute the action recommended by *source* signal."""
        node = self.current()
        signal = node.signals.get(source)
        if signal is None:
            raise KeyError(f"No signal from source '{source}' on node {node.id}")
        if signal.recommended_action < 0:
            raise ValueError(f"Signal '{source}' has no valid recommended action")
        return self.step(signal.recommended_action, chosen_by=source)

    def auto_play(
        self,
        source: str = "agent",
        steps: int = -1,
    ) -> List[DecisionNode]:
        """Run *source*'s recommendation repeatedly until terminal or *steps* reached.

        Returns list of child nodes created.
        """
        results: List[DecisionNode] = []
        count = 0
        while True:
            node = self.current()
            if node.terminal:
                break
            if 0 <= steps <= count:
                break

            # predict
            if source == "agent":
                self.predict_agent()
            elif source.startswith("search:") or source == "search":
                self.predict_search()
                # use whatever search signal was generated
                if source == "search" and self.search is not None:
                    source = f"search:{type(self.search).__name__}"
            else:
                raise ValueError(f"Unknown auto_play source: {source}")

            child = self.step_with(source)
            results.append(child)
            count += 1
        return results

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def undo(self) -> Optional[DecisionNode]:
        """Move to parent node, restoring state."""
        node = self.current()
        if node.parent_id is None:
            return None
        self._redo_stack.append(node.id)
        self._goto_node(node.parent_id)
        self._emit(TraceEvent(type="undo", node_id=node.parent_id))
        return self.tree.nodes[node.parent_id]

    def redo(self) -> Optional[DecisionNode]:
        """Re-advance to the last undone child."""
        if not self._redo_stack:
            return None
        child_id = self._redo_stack.pop()
        self._goto_node(child_id)
        self._emit(TraceEvent(type="redo", node_id=child_id))
        return self.tree.nodes[child_id]

    def goto(self, node_id: int) -> DecisionNode:
        """Jump to any node in the tree, restoring engine state."""
        if node_id not in self.tree.nodes:
            raise KeyError(f"Node {node_id} not in tree")
        self._goto_node(node_id)
        self._redo_stack.clear()
        return self.tree.nodes[node_id]

    def path_to_here(self) -> List[DecisionNode]:
        """Return nodes from root to current (inclusive)."""
        ids = self.tree.path_to(self.tree.active_id)
        return [self.tree.nodes[nid] for nid in ids]

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(self, name: str) -> None:
        """Save current root→active path under *name*."""
        path = self.tree.path_to(self.tree.active_id)
        self.tree.branches[name] = path
        self._emit(TraceEvent(type="branch_created", node_id=self.tree.active_id, data={"name": name}))

    def list_branches(self) -> Dict[str, List[int]]:
        return dict(self.tree.branches)

    def compare(self, *branch_names: str) -> Dict[str, Dict[str, Any]]:
        """Compare terminal cost / reward across named branches."""
        result: Dict[str, Dict[str, Any]] = {}
        for name in branch_names:
            path = self.tree.branches.get(name)
            if not path:
                result[name] = {"error": "branch not found"}
                continue
            last = self.tree.nodes[path[-1]]
            result[name] = {
                "steps": len(path) - 1,
                "cost": last.cost_after,
                "cum_reward": last.cum_reward,
                "terminal": last.terminal,
            }
        return result

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def get_llm_context(self, max_tokens: int = 2000) -> str:
        """Build a text summary of current state for LLM consumption."""
        node = self.current()
        summary = self.state_summary()
        lines = [
            f"Step {node.step}: placing {node.group_id or '(done)'}",
            f"Placed: {summary['placed']}",
            f"Remaining: {summary['remaining']}",
            f"Current cost: {summary['cost']:.2f}",
            f"Node {node.id}, tree size {summary['tree_size']}",
        ]

        for src, sig in node.signals.items():
            top_k_actions = np.argsort(-sig.scores)[:5]
            actions_str = ", ".join(
                f"a{a}({sig.scores[a]:.3f})" for a in top_k_actions if sig.scores[a] > 0
            )
            lines.append(f"  [{src}] recommended=a{sig.recommended_action} top=[{actions_str}]")

        text = "\n".join(lines)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4]
        return text

    def apply_llm_decision(self, action_index: int, reasoning: str = "") -> DecisionNode:
        """Record an LLM-chosen action with reasoning."""
        node = self.current()
        n = node.valid_actions or 1
        scores = np.zeros(n, dtype=np.float32)
        if 0 <= action_index < n:
            scores[action_index] = 1.0
        node.signals["llm"] = Signal(
            source="llm",
            scores=scores,
            recommended_action=action_index,
            metadata={"reasoning": reasoning},
        )
        return self.step(action_index, chosen_by="llm")

    def llm_run(
        self,
        goal: str,
        on_step: Optional[Callable] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        mode: str = "agent",
    ) -> Any:
        """Run the agentic LLM loop with the given goal.

        Requires ``self.llm`` to be a :class:`~trace.llm_agent.ExplorerAgent`.

        Parameters
        ----------
        goal : str
            Free-text instruction for the LLM.
        on_step : callable, optional
            ``(event_type, text)`` callback for streaming output.
        messages : list[dict], optional
            Existing conversation history for multi-turn interactions.
        mode : str
            LLM operating mode (``"chat"``, ``"plan"``, ``"agent"``).

        Returns
        -------
        AgentResult from the LLM agent.
        """
        if self.llm is None:
            raise RuntimeError("No LLM agent configured")
        return self.llm.run(goal, self, on_step=on_step, messages=messages, mode=mode)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def export_tree(self) -> Dict[str, Any]:
        """JSON-serialisable tree structure (no engine state snapshots)."""
        nodes_out: Dict[str, Any] = {}
        for nid, node in self.tree.nodes.items():
            nd: Dict[str, Any] = {
                "id": node.id,
                "parent_id": node.parent_id,
                "step": node.step,
                "group_id": node.group_id,
                "valid_actions": node.valid_actions,
                "chosen_action": node.chosen_action,
                "chosen_by": node.chosen_by,
                "reward": node.reward,
                "cum_reward": node.cum_reward,
                "cost_after": node.cost_after,
                "terminal": node.terminal,
                "children": node.children,
                "signals": {k: v.to_dict() for k, v in node.signals.items()},
                "physical": node.physical.to_dict() if node.physical else None,
            }
            nodes_out[str(nid)] = nd
        return {
            "root_id": self.tree.root_id,
            "active_id": self.tree.active_id,
            "nodes": nodes_out,
            "branches": self.tree.branches,
        }

    def export_path(self, node_id: Optional[int] = None) -> Dict[str, Any]:
        """Export a single root→node path."""
        target = node_id if node_id is not None else self.tree.active_id
        path = self.tree.path_to(target)
        steps = []
        for nid in path:
            n = self.tree.nodes[nid]
            steps.append({
                "id": n.id,
                "step": n.step,
                "group_id": n.group_id,
                "chosen_action": n.chosen_action,
                "chosen_by": n.chosen_by,
                "cost_after": n.cost_after,
                "cum_reward": n.cum_reward,
                "physical": n.physical.to_dict() if n.physical else None,
            })
        return {"path": steps}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        *,
        action_space: ActionSpace,
        rank_key: torch.Tensor,
        delta_costs: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        chosen_action: int = -1,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-K candidates as plain dicts (sorted by *rank_key* ascending).

        Each dict carries ``{action, pos, rank, chosen}`` plus optional
        ``delta`` (from *delta_costs*), ``score`` (from *score*), and
        ``variant`` (from :attr:`ActionSpace.variant_indices`). Invalid actions
        are excluded.
        """
        k = int(k) if k is not None else int(self.candidates_top_k)
        if k <= 0 or int(action_space.centers.shape[0]) <= 0:
            return []
        valid = action_space.valid_mask.view(-1).cpu()
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) == 0:
            return []
        keys = rank_key.detach().to(dtype=torch.float32, device="cpu").view(-1)[valid_idx]
        order = torch.argsort(keys)
        picks = valid_idx[order[: k]]
        centers = action_space.centers.detach().cpu()
        variant_indices = action_space.variant_indices
        if variant_indices is not None:
            variant_indices = variant_indices.detach().cpu()
        if delta_costs is not None:
            delta_costs = delta_costs.detach().to(dtype=torch.float32, device="cpu")
        if score is not None:
            score = score.detach().to(dtype=torch.float32, device="cpu")
        out: List[Dict[str, Any]] = []
        for rank, idx_t in enumerate(picks.tolist()):
            idx = int(idx_t)
            pos = centers[idx].tolist()
            entry: Dict[str, Any] = {
                "action": idx,
                "pos": [float(pos[0]), float(pos[1])],
                "rank": int(rank),
                "chosen": idx == int(chosen_action),
            }
            if delta_costs is not None:
                entry["delta"] = float(delta_costs.view(-1)[idx].item())
            if score is not None:
                entry["score"] = float(score.view(-1)[idx].item())
            if variant_indices is not None:
                entry["variant"] = int(variant_indices.view(-1)[idx].item())
            out.append(entry)
        return out

    def _snapshot_eval_records(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot full per-reward records from :class:`EvalState`.

        Returns ``{name: record}`` where each record carries
        ``weighted_cost``, ``raw_cost``, ``weight``, and ``metadata``
        (verbatim from the reward component). Combines base and terminal
        phases; on name collision terminal is merged into base's weighted
        bucket. Empty dict if eval state is unavailable.
        """
        try:
            eval_state = self.engine.get_state().eval
        except AttributeError:
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for name, rec in eval_state.base_rewards.items():
            records[str(name)] = {
                "weighted_cost": float(rec.get("weighted_cost", 0.0)),
                "raw_cost": float(rec.get("raw_cost", 0.0)),
                "weight": float(rec.get("weight", 1.0)),
                "metadata": dict(rec.get("metadata", {}) or {}),
                "phase": "base",
            }
        for name, rec in eval_state.terminal_rewards.items():
            key = str(name)
            delta = float(rec.get("delta_cost", 0.0))
            metadata = dict(rec.get("metadata", {}) or {})
            if key in records:
                # Merge terminal delta into base bucket's weighted cost.
                records[key]["weighted_cost"] += delta
                records[key]["metadata"] = {
                    **records[key]["metadata"],
                    "_terminal": metadata,
                }
                records[key]["phase"] = "mixed"
            else:
                records[key] = {
                    "weighted_cost": delta,
                    "raw_cost": delta,
                    "weight": 1.0,
                    "metadata": metadata,
                    "phase": "terminal",
                }
        return records

    def _build_physical_context(
        self,
        placement: GroupPlacement,
        cost_before: float,
        cost_after: float,
        *,
        records_before: Optional[Dict[str, Dict[str, Any]]] = None,
        records_after: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[PhysicalContext]:
        """Extract PhysicalContext from a resolved placement after engine step."""
        try:
            gid = placement.group_id
            state = self.engine.get_state()
            p = state.placements.get(gid)
            if p is None:
                return None

            # geometry — use the stored placement which has x_bl, w, h, rotation
            x_bl = float(getattr(p, "x_bl", placement.min_x))
            y_bl = float(getattr(p, "y_bl", placement.min_y))
            w = float(getattr(p, "w", placement.max_x - placement.min_x))
            h = float(getattr(p, "h", placement.max_y - placement.min_y))
            rotation = int(getattr(p, "rotation", 0))
            variant_index = int(getattr(p, "variant_index", 0))

            # affected flow edges
            affected: list[FlowDelta] = []
            gid_key = str(gid)
            flow_edges, _ = extract_flow_edges_and_pairs(state.eval, phase="base")
            for (src, dst), edge in flow_edges.items():
                if src != gid_key and dst != gid_key:
                    continue
                weight = float(edge.get("weight", 0.0))
                total_dist = float(edge.get("distance", 0.0))
                affected.append(FlowDelta(
                    src=str(src), dst=str(dst),
                    weight=weight, distance=total_dist,
                ))

            breakdown: Dict[str, Dict[str, Any]] = {}
            if records_before is not None and records_after is not None:
                keys = set(records_before) | set(records_after)
                for key in keys:
                    rec_b = records_before.get(key) or {}
                    rec_a = records_after.get(key) or {}
                    w_before = float(rec_b.get("weighted_cost", 0.0))
                    w_after = float(rec_a.get("weighted_cost", 0.0))
                    delta = w_after - w_before
                    raw_before = float(rec_b.get("raw_cost", 0.0))
                    raw_after = float(rec_a.get("raw_cost", 0.0))
                    meta_before = dict(rec_b.get("metadata", {}) or {})
                    meta_after = dict(rec_a.get("metadata", {}) or {})
                    if (
                        delta == 0.0
                        and raw_before == raw_after
                        and not meta_before
                        and not meta_after
                    ):
                        continue
                    breakdown[key] = {
                        "delta": float(delta),
                        "raw_before": raw_before,
                        "raw_after": raw_after,
                        "weight": float(rec_a.get("weight", rec_b.get("weight", 1.0))),
                        "metadata_before": meta_before,
                        "metadata_after": meta_after,
                    }

            return PhysicalContext(
                gid=str(gid),
                x=x_bl, y=y_bl, w=w, h=h,
                rotation=rotation, variant_index=variant_index,
                x_center=float(placement.x_center),
                y_center=float(placement.y_center),
                entries=list(placement.entry_points),
                exits=list(placement.exit_points),
                delta_cost=cost_after - cost_before,
                cost_before=cost_before,
                cost_after=cost_after,
                affected_flows=affected,
                breakdown=breakdown,
            )
        except Exception:
            logger.debug("Failed to build PhysicalContext", exc_info=True)
            return None

    def _make_node(
        self,
        *,
        parent_id: Optional[int],
        step: int,
        save_snapshot: bool = True,
    ) -> DecisionNode:
        """Create a new DecisionNode, optionally capturing a state snapshot."""
        nid = self.tree.new_id()
        state = self.engine.get_state()
        gid = state.remaining[0] if state.remaining else None
        terminal = len(state.remaining) == 0

        snapshot = None
        if save_snapshot:
            snapshot = Snapshot(
                engine_state=state.copy(),
                adapter_state=self.adapter.get_state_copy(),
            )

        # count valid actions
        valid = 0
        if not terminal:
            try:
                obs = self.adapter.build_observation()
                action_space = self.adapter.build_action_space()
                valid = self.adapter.num_valid_actions(action_space)
                if snapshot is not None:
                    snapshot.action_space = action_space
            except Exception:
                pass

        return DecisionNode(
            id=nid,
            parent_id=parent_id,
            step=step,
            group_id=str(gid) if gid is not None else None,
            valid_actions=valid,
            terminal=terminal,
            _snapshot=snapshot,
        )

    def _goto_node(self, node_id: int) -> None:
        """Restore engine/adapter state to match *node_id*."""
        node = self.tree.nodes[node_id]
        if node._snapshot is not None:
            self.engine.set_state(node._snapshot.engine_state)
            self.adapter.set_state(node._snapshot.adapter_state)
        else:
            # replay from nearest ancestor with snapshot
            self._replay_to(node_id)
        self.tree.active_id = node_id

    def _replay_to(self, target_id: int) -> None:
        """Find nearest ancestor with a snapshot, restore it, then replay actions."""
        path = self.tree.path_to(target_id)
        # find deepest snapshot
        snap_idx = -1
        for i in range(len(path) - 1, -1, -1):
            if self.tree.nodes[path[i]]._snapshot is not None:
                snap_idx = i
                break
        if snap_idx < 0:
            raise RuntimeError(f"No snapshot found on path to node {target_id}")

        snap_node = self.tree.nodes[path[snap_idx]]
        self.engine.set_state(snap_node._snapshot.engine_state)
        self.adapter.set_state(snap_node._snapshot.adapter_state)

        # replay from snap_idx+1 to target
        for i in range(snap_idx, len(path) - 1):
            node = self.tree.nodes[path[i]]
            if node.chosen_action is None:
                break
            obs = self.adapter.build_observation()
            action_space = self.adapter.build_action_space()

            is_hierarchical = self.adapter.supports_hierarchical
            if is_hierarchical and node.chosen_by and node.chosen_by.startswith("search:"):
                sig = node.signals.get(node.chosen_by)
                wa = sig.metadata.get("worker_action", -1) if sig else -1
                if wa >= 0:
                    worker_as = self.adapter.sub_action_space(node.chosen_action)
                    placement = self.adapter.resolve_sub_action(wa, worker_as, parent_idx=node.chosen_action)
                else:
                    placement = self.adapter.resolve_action(node.chosen_action, action_space)
            else:
                placement = self.adapter.resolve_action(node.chosen_action, action_space)

            self.engine.step(placement)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from group_placement.envs.env_loader import load_env
    from group_placement.agents.registry import create as create_agent

    ENV_JSON = "group_placement/envs/env_configs/basic_01.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    agent, adapter = create_agent(method="greedyv3", agent="greedy", agent_kwargs={"prior_temperature": 1.0})

    exp = Explorer(engine, adapter, agent)
    exp.reset(options=loaded.reset_kwargs)

    t0 = time.perf_counter()
    results = exp.auto_play(source="agent")
    dt = time.perf_counter() - t0

    summary = exp.state_summary()
    print(f"auto_play finished: {len(results)} steps, cost={summary['cost']:.2f}, time={dt:.3f}s")
    print(f"tree size: {summary['tree_size']} nodes")

    # test undo/redo
    exp.undo()
    exp.undo()
    print(f"after 2x undo: step={exp.current().step}, node={exp.current().id}")
    exp.redo()
    print(f"after redo: step={exp.current().step}, node={exp.current().id}")

    # test branch
    exp.goto(exp.tree.root_id)
    exp.branch("first_run")
    print(f"branch 'first_run': {exp.list_branches()}")
    print(f"export_path: {exp.export_path()}")
