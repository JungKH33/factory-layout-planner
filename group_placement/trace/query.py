"""Analysis and LLM context generation over a DecisionTree."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from group_placement.trace.schema import DecisionNode, DecisionTree


class TraceQuery:
    """Read-only queries over a :class:`DecisionTree`."""

    def __init__(self, tree: DecisionTree) -> None:
        self.tree = tree

    # ------------------------------------------------------------------
    # Path queries
    # ------------------------------------------------------------------

    def best_path(self) -> Tuple[List[DecisionNode], float]:
        """Return (path, cost) for the terminal node with lowest cost."""
        terminals = self.tree.terminal_nodes()
        if not terminals:
            active = self.tree.active_node()
            return [active], active.cost_after or float("inf")

        best = min(terminals, key=lambda n: n.cost_after if n.cost_after is not None else float("inf"))
        path_ids = self.tree.path_to(best.id)
        return [self.tree.nodes[nid] for nid in path_ids], best.cost_after or float("inf")

    def top_k_paths(self, k: int = 5) -> List[Tuple[List[DecisionNode], float]]:
        """Return up to *k* terminal paths sorted by cost (ascending)."""
        terminals = self.tree.terminal_nodes()
        terminals.sort(key=lambda n: n.cost_after if n.cost_after is not None else float("inf"))
        results: List[Tuple[List[DecisionNode], float]] = []
        for node in terminals[:k]:
            path_ids = self.tree.path_to(node.id)
            path = [self.tree.nodes[nid] for nid in path_ids]
            results.append((path, node.cost_after or float("inf")))
        return results

    # ------------------------------------------------------------------
    # Signal analysis
    # ------------------------------------------------------------------

    def signal_agreement(self, node_id: int) -> Dict[str, float]:
        """Compute pairwise agreement between signals at a node.

        Agreement = 1.0 if both recommend the same action, else
        cosine similarity of their score vectors.
        """
        node = self.tree.nodes[node_id]
        sources = list(node.signals.keys())
        result: Dict[str, float] = {}
        for i, sa in enumerate(sources):
            for sb in sources[i + 1 :]:
                sig_a = node.signals[sa]
                sig_b = node.signals[sb]
                if sig_a.recommended_action == sig_b.recommended_action:
                    score = 1.0
                else:
                    va = sig_a.scores
                    vb = sig_b.scores
                    if len(va) == len(vb):
                        dot = float(np.dot(va, vb))
                        na = float(np.linalg.norm(va))
                        nb = float(np.linalg.norm(vb))
                        score = dot / max(na * nb, 1e-12)
                    else:
                        score = 0.0
                result[f"{sa}_vs_{sb}"] = round(score, 4)
        return result

    # ------------------------------------------------------------------
    # Branch analysis
    # ------------------------------------------------------------------

    def divergence_points(self, branch_a: str, branch_b: str) -> List[int]:
        """Return node IDs where two branches diverge."""
        pa = self.tree.branches.get(branch_a, [])
        pb = self.tree.branches.get(branch_b, [])
        divergences: List[int] = []
        for i in range(min(len(pa), len(pb))):
            if pa[i] != pb[i]:
                if i > 0:
                    divergences.append(pa[i - 1])
                break
        return divergences

    def best_branch(self) -> Optional[str]:
        """Return the branch name with lowest terminal cost."""
        best_name: Optional[str] = None
        best_cost = float("inf")
        for name, path in self.tree.branches.items():
            if not path:
                continue
            last = self.tree.nodes[path[-1]]
            c = last.cost_after if last.cost_after is not None else float("inf")
            if c < best_cost:
                best_cost = c
                best_name = name
        return best_name

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def depth_stats(self) -> Dict[str, Any]:
        """Per-depth statistics: node count, average branching factor."""
        by_depth: Dict[int, List[DecisionNode]] = {}
        for node in self.tree.nodes.values():
            by_depth.setdefault(node.step, []).append(node)

        stats: Dict[str, Any] = {}
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            branch_factors = [len(n.children) for n in nodes if n.children]
            stats[f"depth_{depth}"] = {
                "nodes": len(nodes),
                "avg_branching": float(np.mean(branch_factors)) if branch_factors else 0.0,
            }
        return stats

    # ------------------------------------------------------------------
    # Text output (for LLM / debug)
    # ------------------------------------------------------------------

    def summarize(self, max_depth: int = 2) -> str:
        """Text summary of the tree's top levels."""
        lines: List[str] = []
        root = self.tree.nodes[self.tree.root_id]
        self._summarize_node(root, lines, depth=0, max_depth=max_depth)
        return "\n".join(lines)

    def _summarize_node(
        self,
        node: DecisionNode,
        lines: List[str],
        depth: int,
        max_depth: int,
    ) -> None:
        indent = "  " * depth
        action_str = f"a{node.chosen_action}" if node.chosen_action is not None else "root"
        cost_str = f"cost={node.cost_after:.1f}" if node.cost_after is not None else ""
        phys_str = ""
        if node.physical is not None:
            p = node.physical
            phys_str = f" | {p.summary()}"
        lines.append(f"{indent}[{node.id}] step={node.step} {action_str} {cost_str} gid={node.group_id or '-'}{phys_str}")

        if depth >= max_depth:
            if node.children:
                lines.append(f"{indent}  ... {len(node.children)} children")
            return

        for _action_idx, child_id in sorted(node.children.items()):
            child = self.tree.nodes[child_id]
            self._summarize_node(child, lines, depth + 1, max_depth)

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Structured context string for LLM agents."""
        parts: List[str] = []
        parts.append("=== Decision Tree Summary ===")
        parts.append(f"Total nodes: {len(self.tree.nodes)}")
        parts.append(f"Branches: {list(self.tree.branches.keys())}")

        # best path
        path, cost = self.best_path()
        parts.append(f"\nBest path (cost={cost:.2f}, {len(path)} steps):")
        for n in path[:10]:
            phys = ""
            if n.physical is not None:
                p = n.physical
                phys = f" [{p.gid} at ({p.x_center:.0f},{p.y_center:.0f}) {p.w:.0f}x{p.h:.0f} delta={p.delta_cost:+.1f}]"
            parts.append(f"  step {n.step}: {n.group_id} → a{n.chosen_action} (by {n.chosen_by}){phys}")
        if len(path) > 10:
            parts.append(f"  ... {len(path) - 10} more steps")

        # current state
        active = self.tree.active_node()
        parts.append(f"\nCurrent: node {active.id}, step {active.step}, gid={active.group_id}")
        for src, sig in active.signals.items():
            parts.append(f"  [{src}] → a{sig.recommended_action} (val={sig.recommended_value:.3f})")

        text = "\n".join(parts)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4]
        return text
