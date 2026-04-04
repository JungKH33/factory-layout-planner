"""Interactive REPL for exploring factory layouts via Explorer.

Usage::

    python -m trace.repl                           # default: basic_01.json, greedyv3
    python -m trace.repl --env envs/env_configs/mixed_01.json
    python -m trace.repl --method greedyv3 --search mcts --sims 200
    python -m trace.repl --llm anthropic           # enable LLM critic/guided placement
"""
from __future__ import annotations

import argparse
import cmd
import logging
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agents.ordering import DifficultyOrderingAgent
from agents.registry import create as create_agent
from envs.env_loader import load_env
from search.beam import BeamConfig, BeamSearch
from search.best import BestFirstConfig, BestFirstSearch
from search.mcts import MCTSConfig, MCTSSearch
from search.hierarchical_beam import HierarchicalBeamConfig, HierarchicalBeamSearch
from search.hierarchical_best import HierarchicalBestFirstConfig, HierarchicalBestFirstSearch
from search.hierarchical_mcts import HierarchicalMCTSConfig, HierarchicalMCTSSearch

from trace.explorer import Explorer
from trace.query import TraceQuery
from trace.schema import Signal, TraceEvent


logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────

def _fmt_cost(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "?"


def _top_actions(scores: np.ndarray, n: int = 5) -> str:
    """Format top-N action indices with scores."""
    order = np.argsort(-scores)
    parts = []
    for idx in order[:n]:
        s = scores[idx]
        if s <= 0:
            break
        parts.append(f"{idx}({s:.3f})")
    return ", ".join(parts) if parts else "(none)"


def _node_line(node, prefix: str = "", physical: bool = True) -> str:
    gid = node.group_id or "(done)"
    cost = _fmt_cost(node.cost_after)
    by = node.chosen_by or ""
    action = f"a{node.chosen_action}" if node.chosen_action is not None else ""
    term = " [TERMINAL]" if node.terminal else ""
    phys = ""
    if physical and node.physical is not None:
        phys = f" | {node.physical.summary()}"
    return f"{prefix}[{node.id}] step={node.step} gid={gid} {action} cost={cost} by={by}{phys}{term}"


# ── REPL ─────────────────────────────────────────────────────────────────

class ExplorerREPL(cmd.Cmd):
    intro = ""
    prompt = "> "

    def __init__(self, exp: Explorer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exp = exp
        self.query = TraceQuery(exp.tree)
        self._last_agent_sig: Optional[Signal] = None
        self._last_search_sig: Optional[Signal] = None

    # ── prompt ───────────────────────────────────────────────────────

    def _update_prompt(self) -> None:
        node = self.exp.current()
        gid = node.group_id or "done"
        self.prompt = f"[step {node.step} | {gid} | node {node.id}] > "

    def preloop(self) -> None:
        self._show_status()
        self._update_prompt()

    def postcmd(self, stop: bool, line: str) -> bool:
        if not stop:
            self._update_prompt()
        return stop

    # ── display helpers ──────────────────────────────────────────────

    def _show_status(self) -> None:
        summary = self.exp.state_summary()
        node = self.exp.current()
        print()
        print(f"  Step:      {summary['step']}")
        print(f"  Facility:  {node.group_id or '(all placed)'}")
        print(f"  Cost:      {_fmt_cost(summary['cost'])}")
        print(f"  Placed:    {len(summary['placed'])} / {len(summary['placed']) + len(summary['remaining'])}")
        print(f"  Remaining: {', '.join(summary['remaining']) if summary['remaining'] else '(none)'}")
        print(f"  Node:      {node.id}  (tree: {summary['tree_size']} nodes)")
        if node.terminal:
            print(f"  ** TERMINAL **")
        if node.valid_actions > 0:
            print(f"  Candidates: {node.valid_actions}")
        print()

    def _show_step_result(self, parent_node, child_node) -> None:
        """Print step result with physical context."""
        phys = parent_node.physical
        if phys is not None:
            print(f"  -> {phys.summary()}")
            if phys.affected_flows:
                for fd in phys.affected_flows:
                    print(f"     flow {fd.src} -> {fd.dst}: w={fd.weight:.1f} dist={fd.distance:.1f}")
        else:
            print(f"  -> Stepped to node {child_node.id}, cost={_fmt_cost(child_node.cost_after)}")

    def _show_signal(self, sig: Signal, label: str = "") -> None:
        name = label or sig.source
        print(f"  [{name}]")
        print(f"    Recommended: action {sig.recommended_action}  (value={sig.recommended_value:.4f})")
        print(f"    Top actions: {_top_actions(sig.scores)}")
        if sig.metadata.get("iterations"):
            print(f"    Iterations:  {sig.metadata['iterations']}")

    # ── commands ─────────────────────────────────────────────────────

    def do_status(self, arg: str) -> None:
        """Show current state."""
        self._show_status()

    def do_candidates(self, arg: str) -> None:
        """Show candidate actions. Usage: candidates [N=10]"""
        node = self.exp.current()
        if node.terminal:
            print("  Terminal node -- no candidates.")
            return
        snapshot = node._snapshot
        if snapshot is None or snapshot.action_space is None:
            print("  No action space available.")
            return
        n = int(arg) if arg.strip() else 10
        aspace = snapshot.action_space
        mask = aspace.valid_mask.cpu().numpy()
        centers = aspace.centers.cpu().numpy()
        valid_indices = np.where(mask)[0]

        # if we have agent scores, sort by them
        scores = self._last_agent_sig.scores if self._last_agent_sig is not None else None

        if scores is not None and len(scores) == len(mask):
            order = np.argsort(-scores[valid_indices])
            valid_indices = valid_indices[order]

        total_valid = len(valid_indices)
        show = valid_indices[:n]
        print(f"  {total_valid} valid candidates (showing top {len(show)}):")
        print(f"  {'idx':>5}  {'x':>7}  {'y':>7}", end="")
        if scores is not None:
            print(f"  {'score':>8}", end="")
        print()
        for idx in show:
            cx, cy = centers[idx]
            line = f"  {idx:>5}  {cx:>7.1f}  {cy:>7.1f}"
            if scores is not None:
                line += f"  {scores[idx]:>8.4f}"
            print(line)
        if total_valid > n:
            print(f"  ... {total_valid - n} more (use 'candidates {total_valid}' to see all)")

    def do_agent(self, arg: str) -> None:
        """Get agent recommendation and step with it. Use 'agent predict' to only predict."""
        node = self.exp.current()
        if node.terminal:
            print("  Terminal node -- cannot step.")
            return
        sig = self.exp.predict_agent()
        self._last_agent_sig = sig
        self._show_signal(sig, "agent")
        if arg.strip() == "predict":
            return
        prev = self.exp.current()
        child = self.exp.step_with("agent")
        self._show_step_result(prev, child)

    def do_search(self, arg: str) -> None:
        """Run search and step with result. Use 'search predict' to only predict."""
        node = self.exp.current()
        if node.terminal:
            print("  Terminal node -- cannot search.")
            return
        if self.exp.search is None:
            print("  No search algorithm configured. Start with --search flag.")
            return

        # also get agent signal for comparison
        if self._last_agent_sig is None or "agent" not in node.signals:
            self._last_agent_sig = self.exp.predict_agent()

        print(f"  Running {type(self.exp.search).__name__}...")
        t0 = time.perf_counter()

        # simple progress callback for terminal
        last_report = [0.0]
        def _progress(iteration, total, visits, values, best_action, best_value):
            now = time.perf_counter()
            if now - last_report[0] >= 1.0 or iteration == total:
                elapsed = now - t0
                print(f"    [{iteration}/{total}] best=a{best_action} val={best_value:.4f} ({elapsed:.1f}s)", end="\r")
                last_report[0] = now

        self.exp.on(_progress_listener := lambda e: (
            _progress(e.data["iteration"], e.data["total"], e.data.get("visits"), e.data.get("values"), e.data["best_action"], e.data["best_value"])
            if e.type == "search_progress" else None
        ))

        try:
            sig = self.exp.predict_search()
        finally:
            self.exp.off(_progress_listener)

        dt = time.perf_counter() - t0
        print()  # clear \r line
        self._last_search_sig = sig
        self._show_signal(sig, f"search ({dt:.2f}s)")

        # show agreement with agent
        agent_sig = self._last_agent_sig
        if agent_sig and agent_sig.recommended_action >= 0:
            agree = "YES" if agent_sig.recommended_action == sig.recommended_action else "NO"
            print(f"    Agent agrees: {agree} (agent=a{agent_sig.recommended_action} vs search=a{sig.recommended_action})")

        if arg.strip() == "predict":
            return
        prev = self.exp.current()
        child = self.exp.step_with(sig.source)
        self._show_step_result(prev, child)

    def do_step(self, arg: str) -> None:
        """Step with a specific action index. Usage: step <action_index>"""
        node = self.exp.current()
        if node.terminal:
            print("  Terminal node -- cannot step.")
            return
        if not arg.strip():
            print("  Usage: step <action_index>")
            return
        try:
            action = int(arg.strip())
        except ValueError:
            print(f"  Invalid action index: {arg}")
            return
        prev = self.exp.current()
        try:
            child = self.exp.step(action, chosen_by="human")
        except Exception as e:
            print(f"  Error: {e}")
            return
        self._show_step_result(prev, child)
        self._last_agent_sig = None
        self._last_search_sig = None

    def do_auto(self, arg: str) -> None:
        """Auto-play to completion. Usage: auto [agent|search] [steps]"""
        parts = arg.strip().split()
        source = parts[0] if parts else "agent"
        steps = int(parts[1]) if len(parts) > 1 else -1

        if source == "search" and self.exp.search is None:
            print("  No search configured -- using agent.")
            source = "agent"

        print(f"  Auto-playing with source={source}...")
        t0 = time.perf_counter()
        results = self.exp.auto_play(source=source, steps=steps)
        dt = time.perf_counter() - t0

        if results:
            last = results[-1]
            print(f"  -> {len(results)} steps in {dt:.2f}s, final cost={_fmt_cost(last.cost_after)}")
        else:
            print("  -> No steps taken (already terminal?).")
        self._last_agent_sig = None
        self._last_search_sig = None

    def do_undo(self, arg: str) -> None:
        """Undo last step (move to parent node)."""
        result = self.exp.undo()
        if result is None:
            print("  Already at root -- nothing to undo.")
        else:
            print(f"  -> Back to node {result.id}, step={result.step}")
            self._last_agent_sig = None
            self._last_search_sig = None

    def do_redo(self, arg: str) -> None:
        """Redo previously undone step."""
        result = self.exp.redo()
        if result is None:
            print("  Nothing to redo.")
        else:
            print(f"  -> Forward to node {result.id}, step={result.step}")
            self._last_agent_sig = None
            self._last_search_sig = None

    def do_goto(self, arg: str) -> None:
        """Jump to a specific node. Usage: goto <node_id>"""
        if not arg.strip():
            print("  Usage: goto <node_id>")
            return
        try:
            nid = int(arg.strip())
            node = self.exp.goto(nid)
        except (ValueError, KeyError) as e:
            print(f"  Error: {e}")
            return
        print(f"  -> Jumped to node {node.id}, step={node.step}, gid={node.group_id}")
        self._last_agent_sig = None
        self._last_search_sig = None

    def do_branch(self, arg: str) -> None:
        """Save current path as named branch. Usage: branch <name>"""
        if not arg.strip():
            # list branches
            branches = self.exp.list_branches()
            if not branches:
                print("  No branches. Usage: branch <name>")
            else:
                print("  Branches:")
                for name, path in branches.items():
                    last = self.exp.tree.nodes[path[-1]]
                    print(f"    {name}: {len(path)} nodes, cost={_fmt_cost(last.cost_after)}")
            return
        self.exp.branch(arg.strip())
        print(f"  -> Branch '{arg.strip()}' saved.")

    def do_compare(self, arg: str) -> None:
        """Compare branches. Usage: compare <branch1> <branch2> ..."""
        names = arg.strip().split()
        if not names:
            names = list(self.exp.tree.branches.keys())
        if not names:
            print("  No branches to compare.")
            return
        result = self.exp.compare(*names)
        print(f"  {'Branch':<15} {'Steps':>6} {'Cost':>10} {'Reward':>10} {'Done':>6}")
        print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*6}")
        for name, info in result.items():
            if "error" in info:
                print(f"  {name:<15} {info['error']}")
            else:
                print(f"  {name:<15} {info['steps']:>6} {info['cost']:>10.2f} {info['cum_reward']:>10.3f} {'yes' if info['terminal'] else 'no':>6}")

    def do_tree(self, arg: str) -> None:
        """Show decision tree. Usage: tree [max_depth=3]"""
        depth = int(arg) if arg.strip() else 3
        print(self.query.summarize(max_depth=depth))

    def do_path(self, arg: str) -> None:
        """Show path from root to current node."""
        path = self.exp.path_to_here()
        print(f"  Path ({len(path)} nodes):")
        for node in path:
            print(f"  {_node_line(node, '  ')}")

    def do_query(self, arg: str) -> None:
        """Run analysis queries. Usage: query [best|topk|agreement|stats]"""
        subcmd = arg.strip().lower() if arg.strip() else "best"

        if subcmd == "best":
            path, cost = self.query.best_path()
            print(f"  Best path: {len(path)} nodes, cost={cost:.2f}")
            for n in path:
                print(f"    {_node_line(n)}")

        elif subcmd.startswith("top"):
            k = 5
            parts = subcmd.split()
            if len(parts) > 1:
                try:
                    k = int(parts[1])
                except ValueError:
                    pass
            results = self.query.top_k_paths(k=k)
            if not results:
                print("  No terminal paths found.")
            else:
                print(f"  Top-{len(results)} paths:")
                for i, (path, cost) in enumerate(results):
                    print(f"    #{i+1}: cost={cost:.2f}, {len(path)} nodes, final_node={path[-1].id}")

        elif subcmd == "agreement":
            node = self.exp.current()
            if len(node.signals) < 2:
                print("  Need at least 2 signals. Run 'agent predict' and 'search predict' first.")
                return
            result = self.query.signal_agreement(node.id)
            for pair, score in result.items():
                print(f"    {pair}: {score:.4f}")

        elif subcmd == "stats":
            stats = self.query.depth_stats()
            print(f"  {'Depth':<8} {'Nodes':>6} {'Avg Branch':>11}")
            for key, val in stats.items():
                depth = key.replace("depth_", "")
                print(f"  {depth:<8} {val['nodes']:>6} {val['avg_branching']:>11.2f}")

        else:
            print("  Usage: query [best|topk [k]|agreement|stats]")

    def do_detail(self, arg: str) -> None:
        """Show physical placement detail for current or specified node. Usage: detail [node_id]"""
        if arg.strip():
            try:
                nid = int(arg.strip())
                node = self.exp.tree.nodes[nid]
            except (ValueError, KeyError) as e:
                print(f"  Error: {e}")
                return
        else:
            node = self.exp.current()
            # if current node has no physical, check parent
            if node.physical is None and node.parent_id is not None:
                parent = self.exp.tree.nodes[node.parent_id]
                if parent.physical is not None:
                    node = parent

        phys = node.physical
        if phys is None:
            print(f"  Node {node.id} has no physical context (root or no step taken).")
            return

        print(f"  Node {node.id} placement:")
        print(f"    Facility:   {phys.gid}")
        print(f"    Position:   ({phys.x_center:.1f}, {phys.y_center:.1f}) center, ({phys.x:.1f}, {phys.y:.1f}) BL")
        print(f"    Size:       {phys.w:.0f} x {phys.h:.0f}")
        print(f"    Rotation:   {phys.rotation}")
        print(f"    Variant:    {phys.variant_index}")
        print(f"    Cost:       {phys.cost_before:.2f} -> {phys.cost_after:.2f} (delta={phys.delta_cost:+.2f})")
        if phys.entries:
            ent_str = ", ".join(f"({x:.0f},{y:.0f})" for x, y in phys.entries)
            print(f"    Entries:    {ent_str}")
        if phys.exits:
            ext_str = ", ".join(f"({x:.0f},{y:.0f})" for x, y in phys.exits)
            print(f"    Exits:      {ext_str}")
        if phys.affected_flows:
            print(f"    Flows ({len(phys.affected_flows)}):")
            for fd in phys.affected_flows:
                print(f"      {fd.src} -> {fd.dst}: weight={fd.weight:.1f}, avg_dist={fd.distance:.1f}")
        else:
            print(f"    Flows:      (none)")

    def do_signals(self, arg: str) -> None:
        """Show all signals on current node."""
        node = self.exp.current()
        if not node.signals:
            print("  No signals. Run 'agent predict' or 'search predict' first.")
            return
        for src, sig in node.signals.items():
            self._show_signal(sig, src)

    def do_context(self, arg: str) -> None:
        """Show LLM-formatted context for current state."""
        if arg.strip() == "v2":
            # show what the LLM tools would return
            from trace.llm_agent import _tool_status, _tool_candidates
            print(_tool_status(self.exp))
            print()
            if not self.exp.current().terminal:
                print(_tool_candidates(self.exp, top_k=8))
        else:
            print(self.exp.get_llm_context())

    def do_llm(self, arg: str) -> None:
        """Run LLM agent with a goal. Usage: llm <instruction>

        Examples:
          llm place this facility near factory_A
          llm place all remaining facilities optimizing for flow
          llm critique the current recommendations
          llm what would be a good position for this facility?
        """
        if self.exp.llm is None:
            print("  No LLM configured. Start with --llm flag (e.g. --llm anthropic).")
            return

        goal = arg.strip()
        if not goal:
            print("  Usage: llm <instruction>")
            print("  Examples:")
            print("    llm place this near factory_A")
            print("    llm place all remaining facilities")
            print("    llm what's the best position for this?")
            return

        def _on_step(event_type: str, text: str) -> None:
            if event_type == "thinking":
                for line in text.splitlines():
                    print(f"  [LLM] {line}")
            elif event_type == "tool_call":
                print(f"  [LLM] -> {text}")
            elif event_type == "tool_result":
                # indent tool results
                for line in text.splitlines():
                    print(f"         {line}")
            elif event_type == "done":
                print(f"  [LLM] Done: {text}")

        try:
            result = self.exp.llm_run(goal, on_step=_on_step)
            if result.final_text and not any(
                result.final_text in msg.get("content", "")
                for msg in result.messages
                if isinstance(msg.get("content"), str)
            ):
                print(f"  [LLM] {result.final_text}")
            print(f"  ({result.steps_taken} turns)")
        except Exception as e:
            print(f"  Error: {e}")

        self._last_agent_sig = None
        self._last_search_sig = None

    def do_reset(self, arg: str) -> None:
        """Reset the environment and tree."""
        self.exp.reset(options=self.exp.engine._reset_kwargs if hasattr(self.exp.engine, '_reset_kwargs') else None)
        self._last_agent_sig = None
        self._last_search_sig = None
        print("  -> Environment reset.")
        self._show_status()

    def do_help(self, arg: str) -> None:
        """Show available commands."""
        if arg:
            super().do_help(arg)
            return
        print("""
  Prediction:
    agent              Get agent recommendation and step
    agent predict      Get agent recommendation only (no step)
    search             Run search and step with result
    search predict     Run search only (no step)
    signals            Show all signals on current node

  Execution:
    step <idx>         Step with specific action index
    auto [src] [n]     Auto-play (src=agent|search, n=max steps)

  Navigation:
    undo               Go back one step
    redo               Re-advance after undo
    goto <node_id>     Jump to any node
    path               Show root -> current path

  Exploration:
    branch [name]      Save current path / list branches
    compare [b1 b2..]  Compare branch costs

  Analysis:
    candidates [n]     Show top-N candidate positions
    detail [node_id]   Show physical placement detail
    tree [depth]       Show decision tree
    query best         Best terminal path
    query topk [k]     Top-K terminal paths
    query agreement    Signal agreement at current node
    query stats        Depth statistics
    context            LLM-formatted state summary (v2 for physical)
    status             Current state summary

  LLM (requires --llm flag):
    llm <instruction>  Run LLM agent (e.g. "llm place near A")

  Other:
    reset              Reset environment
    quit / q           Exit
""")

    def do_quit(self, arg: str) -> bool:
        """Exit the REPL."""
        print("Bye!")
        return True

    do_q = do_quit
    do_EOF = do_quit

    def default(self, line: str) -> None:
        stripped = line.strip()
        if stripped:
            print(f"  Unknown command: {stripped}. Type 'help' for available commands.")

    def emptyline(self) -> None:
        pass


# ── CLI entry point ──────────────────────────────────────────────────────

def build_search(args):
    """Build search algorithm from CLI args."""
    mode = args.search
    if mode == "none":
        return None
    elif mode == "mcts":
        return MCTSSearch(config=MCTSConfig(
            num_simulations=args.sims,
            rollout_enabled=True,
            rollout_depth=args.rollout_depth,
            track_top_k=args.topk,
        ))
    elif mode == "beam":
        return BeamSearch(config=BeamConfig(
            beam_width=args.beam_width,
            depth=args.depth,
            expansion_topk=args.expansion_topk,
            track_top_k=args.topk,
        ))
    elif mode in ("best", "best_first"):
        return BestFirstSearch(config=BestFirstConfig(
            max_expansions=args.sims,
            depth=args.depth,
            expansion_topk=args.expansion_topk,
            track_top_k=args.topk,
        ))
    elif mode == "h_mcts":
        return HierarchicalMCTSSearch(config=HierarchicalMCTSConfig(
            num_simulations=args.sims,
            rollout_enabled=True,
            rollout_depth=args.rollout_depth,
            track_top_k=args.topk,
        ))
    elif mode == "h_beam":
        return HierarchicalBeamSearch(config=HierarchicalBeamConfig(
            beam_width=args.beam_width,
            depth=args.depth,
            track_top_k=args.topk,
        ))
    elif mode in ("h_best", "h_best_first"):
        return HierarchicalBestFirstSearch(config=HierarchicalBestFirstConfig(
            max_expansions=args.sims,
            depth=args.depth,
            track_top_k=args.topk,
        ))
    else:
        raise ValueError(f"Unknown search mode: {mode}")


def _build_llm_agent(provider_name: str, model_override: Optional[str] = None):
    """Create ExplorerAgent from CLI args."""
    from trace.llm_agent import ExplorerAgent

    if provider_name == "anthropic":
        from trace.llm_agent import AnthropicBackend
        kwargs = {}
        if model_override:
            kwargs["model"] = model_override
        return ExplorerAgent(backend=AnthropicBackend(**kwargs))
    elif provider_name == "openai":
        from trace.llm_agent import OpenAIBackend
        kwargs = {}
        if model_override:
            kwargs["model"] = model_override
        return ExplorerAgent(backend=OpenAIBackend(**kwargs))
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive factory layout explorer")
    parser.add_argument("--env", default="envs/env_configs/basic_01.json", help="Env config JSON path")
    parser.add_argument("--method", default="greedyv3", help="Adapter method")
    parser.add_argument("--agent", default="greedy", help="Agent type")
    parser.add_argument("--search", default="none",
                        choices=["none", "mcts", "beam", "best", "best_first", "h_mcts", "h_beam", "h_best", "h_best_first"],
                        help="Search algorithm")
    parser.add_argument("--ordering", default="none", choices=["none", "difficulty"])
    parser.add_argument("--sims", type=int, default=200, help="Simulations (MCTS) or max expansions (best-first)")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--depth", type=int, default=5, help="Search depth (beam/best)")
    parser.add_argument("--expansion-topk", type=int, default=16)
    parser.add_argument("--rollout-depth", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5, help="Track top-K results during search")
    parser.add_argument("--k", type=int, default=50, help="Candidate count for greedy adapters")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--llm", default="none",
                        choices=["none", "anthropic", "openai"],
                        help="LLM provider for critic/guided placement")
    parser.add_argument("--llm-model", default=None,
                        help="Override LLM model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.env} ...")
    loaded = load_env(args.env, device=device)
    engine = loaded.env
    engine.log = False

    adapter_kwargs: Dict[str, Any] = {"k": args.k, "quant_step": 10.0}
    agent, adapter = create_agent(
        method=args.method,
        agent=args.agent,
        agent_kwargs={"prior_temperature": 1.0},
        adapter_kwargs=adapter_kwargs,
    )

    search = build_search(args)
    ordering_agent = DifficultyOrderingAgent() if args.ordering == "difficulty" else None

    # LLM advisor (optional)
    llm_advisor = None
    if args.llm != "none":
        llm_advisor = _build_llm_agent(args.llm, args.llm_model)

    exp = Explorer(engine, adapter, agent, search=search,
                   ordering_agent=ordering_agent, llm=llm_advisor)
    exp.reset(options=loaded.reset_kwargs)

    # Store reset kwargs for later resets
    exp.engine._reset_kwargs = loaded.reset_kwargs  # type: ignore[attr-defined]

    header = f"Factory Layout Explorer - {args.env}"
    if search is not None:
        header += f" + {type(search).__name__}"
    if llm_advisor is not None:
        header += f" + LLM({args.llm})"
    print(header)
    print("Type 'help' for commands.\n")

    repl = ExplorerREPL(exp)
    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
