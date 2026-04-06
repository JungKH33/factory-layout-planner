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
import shlex
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rich.console import Console, Group
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

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


# ── Rich console ─────────────────────────────────────────────────────────

_THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "heading": "bold magenta",
    "muted": "dim",
    "accent": "bold cyan",
})
console = Console(theme=_THEME, highlight=False)

_BICHON_LOGO = Path(__file__).resolve().parent.parent / "assets" / "bichon-logo.png"


def _print_banner(subtitle: str = "") -> None:
    body = Text()
    body.append("  Welcome to Bichon Layout Planner\n", style="bold cyan")
    if subtitle:
        body.append(f"  {subtitle}\n", style="dim white")
    body.append("\n  Type ", style="dim")
    body.append("help", style="bold white")
    body.append(" for commands.", style="dim")

    inner: Any = body
    if _BICHON_LOGO.is_file():
        try:
            from rich_pixels import Pixels

            # Half-cell renderer: keep modest size so the banner fits typical terminals.
            logo = Pixels.from_image_path(_BICHON_LOGO, resize=(48, 24))
            inner = Group(logo, Text(), body)
        except Exception as exc:
            logger.debug("Could not render banner logo (%s): %s", _BICHON_LOGO, exc)

    console.print(Panel(inner, border_style="cyan", padding=(0, 2)))
    console.print()


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
    term = " [bold red]TERMINAL[/bold red]" if node.terminal else ""
    phys = ""
    if physical and node.physical is not None:
        phys = f" | {node.physical.summary()}"
    return f"{prefix}\\[{node.id}] step={node.step} gid={gid} {action} cost=[cyan]{cost}[/cyan] by={by}{phys}{term}"


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
        self._llm_mode: str = "agent"
        self._llm_messages: List[Dict[str, Any]] = []
        # Render state — single figure, layout + candidates overlay
        self._auto_render: bool = True
        self._fig: Any = None
        self._ax: Any = None
        self._cbar: Any = None  # colorbar instance (reused to avoid duplication)
        self._last_rendered_node: Optional[int] = None

    # ── prompt ───────────────────────────────────────────────────────

    def _update_prompt(self) -> None:
        node = self.exp.current()
        gid = node.group_id or "done"
        self.prompt = (
            f"\x1b[36m[step {node.step} | {gid} | node {node.id}]\x1b[0m"
            f" \x1b[33m>\x1b[0m "
        )

    def preloop(self) -> None:
        self._show_status()
        self._update_prompt()

    def postcmd(self, stop: bool, line: str) -> bool:
        if not stop:
            self._update_prompt()
            if self._auto_render:
                self._auto_render_update()
        return stop

    # ── display helpers ──────────────────────────────────────────────

    def _show_status(self) -> None:
        summary = self.exp.state_summary()
        node = self.exp.current()
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold cyan", width=12)
        table.add_column()
        table.add_row("Step", str(summary['step']))
        table.add_row("Facility", node.group_id or "[dim](all placed)[/dim]")
        table.add_row("Cost", _fmt_cost(summary['cost']))
        placed = summary['placed']
        remaining = summary['remaining']
        table.add_row("Placed", f"{len(placed)} / {len(placed) + len(remaining)}")
        rem_str = ", ".join(remaining) if remaining else "[dim](none)[/dim]"
        table.add_row("Remaining", rem_str)
        table.add_row("Node", f"{node.id}  [dim](tree: {summary['tree_size']} nodes)[/dim]")
        if node.terminal:
            table.add_row("", "[bold red]** TERMINAL **[/bold red]")
        if node.valid_actions > 0:
            table.add_row("Candidates", str(node.valid_actions))
        console.print()
        console.print(table)
        console.print()

    def _show_step_result(self, parent_node, child_node) -> None:
        """Print step result with physical context."""
        phys = parent_node.physical
        if phys is not None:
            console.print(f"  [success]\u2192[/success] {phys.summary()}")
            if phys.affected_flows:
                for fd in phys.affected_flows:
                    console.print(
                        f"    [muted]flow {fd.src} \u2192 {fd.dst}:"
                        f" w={fd.weight:.1f} dist={fd.distance:.1f}[/muted]"
                    )
        else:
            console.print(
                f"  [success]\u2192[/success] Stepped to node {child_node.id},"
                f" cost={_fmt_cost(child_node.cost_after)}"
            )

    def _show_signal(self, sig: Signal, label: str = "") -> None:
        name = label or sig.source
        console.print(f"  [heading]\\[{name}][/heading]")
        console.print(
            f"    Recommended: action [bold]{sig.recommended_action}[/bold]"
            f"  (value={sig.recommended_value:.4f})"
        )
        console.print(f"    Top actions: {_top_actions(sig.scores)}")
        if sig.metadata.get("iterations"):
            console.print(f"    Iterations:  {sig.metadata['iterations']}")

    # ── commands ─────────────────────────────────────────────────────

    def do_status(self, arg: str) -> None:
        """Show current state."""
        self._show_status()

    def do_candidates(self, arg: str) -> None:
        """Show candidate actions. Usage: candidates [N=10]"""
        node = self.exp.current()
        if node.terminal:
            console.print("  [warning]Terminal node -- no candidates.[/warning]")
            return
        snapshot = node._snapshot
        if snapshot is None or snapshot.action_space is None:
            console.print("  [muted]No action space available.[/muted]")
            return
        n = int(arg) if arg.strip() else 10
        aspace = snapshot.action_space
        mask = aspace.valid_mask.cpu().numpy()
        centers = aspace.centers.cpu().numpy()
        valid_indices = np.where(mask)[0]

        scores = self._last_agent_sig.scores if self._last_agent_sig is not None else None

        if scores is not None and len(scores) == len(mask):
            order = np.argsort(-scores[valid_indices])
            valid_indices = valid_indices[order]

        total_valid = len(valid_indices)
        show = valid_indices[:n]

        table = Table(
            title=f"{total_valid} valid candidates (top {len(show)})",
            title_style="bold",
            border_style="dim",
        )
        table.add_column("idx", justify="right", style="bold")
        table.add_column("x", justify="right")
        table.add_column("y", justify="right")
        if scores is not None:
            table.add_column("score", justify="right", style="cyan")
        for idx in show:
            cx, cy = centers[idx]
            row = [str(idx), f"{cx:.1f}", f"{cy:.1f}"]
            if scores is not None:
                row.append(f"{scores[idx]:.4f}")
            table.add_row(*row)
        console.print(table)
        if total_valid > n:
            console.print(
                f"  [muted]... {total_valid - n} more"
                f" (use 'candidates {total_valid}' to see all)[/muted]"
            )

    def do_agent(self, arg: str) -> None:
        """Get agent recommendation and step with it. Use 'agent predict' to only predict."""
        node = self.exp.current()
        if node.terminal:
            console.print("  [warning]Terminal node -- cannot step.[/warning]")
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
            console.print("  [warning]Terminal node -- cannot search.[/warning]")
            return
        if self.exp.search is None:
            console.print("  [error]No search algorithm configured. Start with --search flag.[/error]")
            return

        if self._last_agent_sig is None or "agent" not in node.signals:
            self._last_agent_sig = self.exp.predict_agent()

        search_name = type(self.exp.search).__name__
        console.print(f"  [info]Running {search_name}...[/info]")
        t0 = time.perf_counter()

        last_report = [0.0]

        def _progress(iteration, total, visits, values, best_action, best_value):
            now = time.perf_counter()
            if now - last_report[0] >= 1.0 or iteration == total:
                elapsed = now - t0
                print(
                    f"    [{iteration}/{total}] best=a{best_action}"
                    f" val={best_value:.4f} ({elapsed:.1f}s)",
                    end="\r",
                )
                last_report[0] = now

        self.exp.on(_progress_listener := lambda e: (
            _progress(
                e.data["iteration"], e.data["total"],
                e.data.get("visits"), e.data.get("values"),
                e.data["best_action"], e.data["best_value"],
            )
            if e.type == "search_progress" else None
        ))

        try:
            sig = self.exp.predict_search()
        finally:
            self.exp.off(_progress_listener)

        dt = time.perf_counter() - t0
        print()
        self._last_search_sig = sig
        self._show_signal(sig, f"search ({dt:.2f}s)")

        agent_sig = self._last_agent_sig
        if agent_sig and agent_sig.recommended_action >= 0:
            agree = agent_sig.recommended_action == sig.recommended_action
            style = "success" if agree else "warning"
            label = "YES" if agree else "NO"
            console.print(
                f"    Agent agrees: [{style}]{label}[/{style}]"
                f" (agent=a{agent_sig.recommended_action}"
                f" vs search=a{sig.recommended_action})"
            )

        if arg.strip() == "predict":
            return
        prev = self.exp.current()
        child = self.exp.step_with(sig.source)
        self._show_step_result(prev, child)

    def do_step(self, arg: str) -> None:
        """Step with a specific action index. Usage: step <action_index>"""
        node = self.exp.current()
        if node.terminal:
            console.print("  [warning]Terminal node -- cannot step.[/warning]")
            return
        if not arg.strip():
            console.print("  Usage: step <action_index>")
            return
        try:
            action = int(arg.strip())
        except ValueError:
            console.print(f"  [error]Invalid action index: {arg}[/error]")
            return
        prev = self.exp.current()
        try:
            child = self.exp.step(action, chosen_by="human")
        except Exception as e:
            console.print(f"  [error]Error: {e}[/error]")
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
            console.print("  [warning]No search configured -- using agent.[/warning]")
            source = "agent"

        console.print(f"  [info]Auto-playing with source={source}...[/info]")
        t0 = time.perf_counter()
        results = self.exp.auto_play(source=source, steps=steps)
        dt = time.perf_counter() - t0

        if results:
            last = results[-1]
            console.print(
                f"  [success]\u2192[/success] {len(results)} steps in {dt:.2f}s,"
                f" final cost={_fmt_cost(last.cost_after)}"
            )
        else:
            console.print("  [muted]\u2192 No steps taken (already terminal?).[/muted]")
        self._last_agent_sig = None
        self._last_search_sig = None

    def do_undo(self, arg: str) -> None:
        """Undo last step (move to parent node)."""
        result = self.exp.undo()
        if result is None:
            console.print("  [muted]Already at root -- nothing to undo.[/muted]")
        else:
            console.print(f"  [success]\u2192[/success] Back to node {result.id}, step={result.step}")
            self._last_agent_sig = None
            self._last_search_sig = None

    def do_redo(self, arg: str) -> None:
        """Redo previously undone step."""
        result = self.exp.redo()
        if result is None:
            console.print("  [muted]Nothing to redo.[/muted]")
        else:
            console.print(f"  [success]\u2192[/success] Forward to node {result.id}, step={result.step}")
            self._last_agent_sig = None
            self._last_search_sig = None

    def do_goto(self, arg: str) -> None:
        """Jump to a specific node. Usage: goto <node_id>"""
        if not arg.strip():
            console.print("  Usage: goto <node_id>")
            return
        try:
            nid = int(arg.strip())
            node = self.exp.goto(nid)
        except (ValueError, KeyError) as e:
            console.print(f"  [error]Error: {e}[/error]")
            return
        console.print(
            f"  [success]\u2192[/success] Jumped to node {node.id},"
            f" step={node.step}, gid={node.group_id}"
        )
        self._last_agent_sig = None
        self._last_search_sig = None

    def do_branch(self, arg: str) -> None:
        """Save current path as named branch. Usage: branch <name>"""
        if not arg.strip():
            branches = self.exp.list_branches()
            if not branches:
                console.print("  [muted]No branches. Usage: branch <name>[/muted]")
            else:
                table = Table(title="Branches", title_style="bold", border_style="dim")
                table.add_column("Name", style="bold")
                table.add_column("Nodes", justify="right")
                table.add_column("Cost", justify="right", style="cyan")
                for name, path in branches.items():
                    last = self.exp.tree.nodes[path[-1]]
                    table.add_row(name, str(len(path)), _fmt_cost(last.cost_after))
                console.print(table)
            return
        self.exp.branch(arg.strip())
        console.print(f"  [success]\u2192[/success] Branch '{arg.strip()}' saved.")

    def do_compare(self, arg: str) -> None:
        """Compare branches. Usage: compare <branch1> <branch2> ..."""
        names = arg.strip().split()
        if not names:
            names = list(self.exp.tree.branches.keys())
        if not names:
            console.print("  [muted]No branches to compare.[/muted]")
            return
        result = self.exp.compare(*names)
        table = Table(title="Branch Comparison", title_style="bold", border_style="dim")
        table.add_column("Branch", style="bold")
        table.add_column("Steps", justify="right")
        table.add_column("Cost", justify="right", style="cyan")
        table.add_column("Reward", justify="right")
        table.add_column("Done", justify="center")
        for name, info in result.items():
            if "error" in info:
                table.add_row(name, info["error"], "", "", "")
            else:
                done_style = "green" if info["terminal"] else "dim"
                table.add_row(
                    name,
                    str(info["steps"]),
                    f"{info['cost']:.2f}",
                    f"{info['cum_reward']:.3f}",
                    f"[{done_style}]{'yes' if info['terminal'] else 'no'}[/{done_style}]",
                )
        console.print(table)

    def do_tree(self, arg: str) -> None:
        """Show decision tree. Usage: tree [max_depth=3]"""
        depth = int(arg) if arg.strip() else 3
        console.print(self.query.summarize(max_depth=depth), markup=False)

    def do_path(self, arg: str) -> None:
        """Show path from root to current node."""
        path = self.exp.path_to_here()
        console.print(f"  [heading]Path ({len(path)} nodes):[/heading]")
        for node in path:
            console.print(f"  {_node_line(node, '  ')}")

    def do_query(self, arg: str) -> None:
        """Run analysis queries. Usage: query [best|topk|agreement|stats]"""
        subcmd = arg.strip().lower() if arg.strip() else "best"

        if subcmd == "best":
            path, cost = self.query.best_path()
            console.print(
                f"  [heading]Best path:[/heading] {len(path)} nodes,"
                f" cost=[cyan]{cost:.2f}[/cyan]"
            )
            for n in path:
                console.print(f"    {_node_line(n)}")

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
                console.print("  [muted]No terminal paths found.[/muted]")
            else:
                console.print(f"  [heading]Top-{len(results)} paths:[/heading]")
                for i, (path, cost) in enumerate(results):
                    console.print(
                        f"    [bold]#{i+1}[/bold]: cost=[cyan]{cost:.2f}[/cyan],"
                        f" {len(path)} nodes, final_node={path[-1].id}"
                    )

        elif subcmd == "agreement":
            node = self.exp.current()
            if len(node.signals) < 2:
                console.print(
                    "  [muted]Need at least 2 signals."
                    " Run 'agent predict' and 'search predict' first.[/muted]"
                )
                return
            result = self.query.signal_agreement(node.id)
            for pair, score in result.items():
                console.print(f"    {pair}: {score:.4f}")

        elif subcmd == "stats":
            stats = self.query.depth_stats()
            table = Table(
                title="Depth Statistics", title_style="bold", border_style="dim",
            )
            table.add_column("Depth", justify="right", style="bold")
            table.add_column("Nodes", justify="right")
            table.add_column("Avg Branch", justify="right", style="cyan")
            for key, val in stats.items():
                d = key.replace("depth_", "")
                table.add_row(d, str(val["nodes"]), f"{val['avg_branching']:.2f}")
            console.print(table)

        else:
            console.print("  Usage: query \\[best|topk \\[k]|agreement|stats]")

    def do_detail(self, arg: str) -> None:
        """Show physical placement detail for current or specified node. Usage: detail [node_id]"""
        if arg.strip():
            try:
                nid = int(arg.strip())
                node = self.exp.tree.nodes[nid]
            except (ValueError, KeyError) as e:
                console.print(f"  [error]Error: {e}[/error]")
                return
        else:
            node = self.exp.current()
            if node.physical is None and node.parent_id is not None:
                parent = self.exp.tree.nodes[node.parent_id]
                if parent.physical is not None:
                    node = parent

        phys = node.physical
        if phys is None:
            console.print(
                f"  [muted]Node {node.id} has no physical context"
                f" (root or no step taken).[/muted]"
            )
            return

        table = Table(
            show_header=False, box=None, padding=(0, 1),
            title=f"Node {node.id} Placement", title_style="bold",
        )
        table.add_column(style="bold cyan", width=12)
        table.add_column()
        table.add_row("Facility", phys.gid)
        table.add_row(
            "Position",
            f"({phys.x_center:.1f}, {phys.y_center:.1f}) center,"
            f" ({phys.x:.1f}, {phys.y:.1f}) BL",
        )
        table.add_row("Size", f"{phys.w:.0f} x {phys.h:.0f}")
        table.add_row("Rotation", str(phys.rotation))
        table.add_row("Variant", str(phys.variant_index))
        delta_style = "red" if phys.delta_cost > 0 else "green"
        table.add_row(
            "Cost",
            f"{phys.cost_before:.2f} \u2192 {phys.cost_after:.2f}"
            f" ([{delta_style}]{phys.delta_cost:+.2f}[/{delta_style}])",
        )
        if phys.entries:
            ent_str = ", ".join(f"({x:.0f},{y:.0f})" for x, y in phys.entries)
            table.add_row("Entries", ent_str)
        if phys.exits:
            ext_str = ", ".join(f"({x:.0f},{y:.0f})" for x, y in phys.exits)
            table.add_row("Exits", ext_str)
        console.print(table)

        if phys.affected_flows:
            flow_table = Table(
                title=f"Flows ({len(phys.affected_flows)})",
                title_style="bold", border_style="dim",
            )
            flow_table.add_column("Src", style="bold")
            flow_table.add_column("Dst", style="bold")
            flow_table.add_column("Weight", justify="right")
            flow_table.add_column("Avg Dist", justify="right", style="cyan")
            for fd in phys.affected_flows:
                flow_table.add_row(fd.src, fd.dst, f"{fd.weight:.1f}", f"{fd.distance:.1f}")
            console.print(flow_table)
        else:
            console.print("  [muted]Flows: (none)[/muted]")

    # ── render system ─────────────────────────────────────────────────

    def _render(self) -> None:
        """Render layout + candidate overlay in a single figure."""
        import matplotlib.pyplot as plt
        from envs.visualizer.data import extract_layout_data
        from envs.visualizer.mpl import _draw_layout_from_data

        node = self.exp.current()
        data = extract_layout_data(self.exp.engine)

        # Create or reuse figure
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 7))
            self._fig.canvas.manager.set_window_title("Bichon Layout")
            self._cbar = None

        ax = self._ax

        # Remove old colorbar before clearing
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        ax.clear()
        ax.set_xlim(0, data.grid_width)
        ax.set_ylim(0, data.grid_height)
        ax.set_aspect("equal")

        # Title
        gid = node.group_id or "done"
        ax.set_title(
            f"step {node.step}  |  placing: {gid}  |  "
            f"cost={data.cost:.1f}  |  {len(data.facilities)} placed"
        )

        # Draw layout (facilities, zones, flow, ports)
        _draw_layout_from_data(ax, data)

        # Overlay candidates if not terminal
        if not node.terminal:
            snapshot = node._snapshot
            if snapshot is not None and snapshot.action_space is not None:
                aspace = snapshot.action_space
                mask = aspace.valid_mask.cpu().numpy().astype(bool)
                centers = aspace.centers.cpu().numpy()
                valid_idx = np.where(mask)[0]

                if len(valid_idx) > 0:
                    agent_sig = self._last_agent_sig or node.signals.get("agent")
                    scores = agent_sig.scores if agent_sig is not None else None
                    recommended = agent_sig.recommended_action if agent_sig is not None else -1

                    xs = centers[valid_idx, 0]
                    ys = centers[valid_idx, 1]

                    if scores is not None and len(scores) == len(mask):
                        cs = scores[valid_idx]
                        sc = ax.scatter(
                            xs, ys, s=24, c=cs, cmap="RdYlGn", alpha=0.8,
                            linewidths=0.3, edgecolors="black", zorder=5,
                        )
                        self._cbar = self._fig.colorbar(
                            sc, ax=ax, label="score", shrink=0.6, pad=0.01,
                        )
                    else:
                        ax.scatter(
                            xs, ys, s=24, c="green", alpha=0.6,
                            linewidths=0.3, edgecolors="black", zorder=5,
                        )

                    # Highlight agent pick
                    if 0 <= recommended < len(mask) and mask[recommended]:
                        ax.scatter(
                            [centers[recommended, 0]], [centers[recommended, 1]],
                            s=180, facecolors="none", edgecolors="#1f77b4",
                            linewidths=2.5, zorder=6, label="agent pick",
                        )
                        ax.legend(loc="upper right", fontsize=8)

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._last_rendered_node = self.exp.tree.active_id

    def _auto_render_update(self) -> None:
        """Called by postcmd — re-render if figure is open and state changed."""
        import matplotlib.pyplot as plt
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            return
        if self.exp.tree.active_id == self._last_rendered_node:
            return
        # Ensure agent signal for candidate overlay
        node = self.exp.current()
        if not node.terminal and "agent" not in node.signals:
            self.exp.predict_agent()
            self._last_agent_sig = node.signals.get("agent")
        self._render()

    def do_render(self, arg: str) -> None:
        """Open/refresh the layout viewer, or toggle auto-render.

        Usage:
          render           Open/refresh layout + candidates
          render on        Enable auto-render (default when window is open)
          render off       Disable auto-render
          render save [p]  Save current layout to PNG (default: layout.png)
        """
        subcmd = arg.strip().lower()
        if subcmd == "on":
            self._auto_render = True
            console.print("  [success]Auto-render ON[/success]")
            return
        if subcmd == "off":
            self._auto_render = False
            console.print("  [muted]Auto-render OFF[/muted]")
            return
        if subcmd.startswith("save"):
            parts = arg.strip().split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else "layout.png"
            from envs.visualizer.data import extract_layout_data
            from envs.visualizer.mpl import MatplotlibBackend
            data = extract_layout_data(self.exp.engine)
            MatplotlibBackend().save_layout(data, path, show_flow=True, show_score=True)
            console.print(f"  [success]Saved to {path}[/success]")
            return
        # Default: render (ensures agent signal for candidates)
        node = self.exp.current()
        if not node.terminal and "agent" not in node.signals:
            self.exp.predict_agent()
            self._last_agent_sig = node.signals.get("agent")
        self._render()
        self._auto_render = True
        console.print("  [success]Rendered (auto-render ON).[/success]")

    def do_signals(self, arg: str) -> None:
        """Show all signals on current node."""
        node = self.exp.current()
        if not node.signals:
            console.print(
                "  [muted]No signals. Run 'agent predict' or 'search predict' first.[/muted]"
            )
            return
        for src, sig in node.signals.items():
            self._show_signal(sig, src)

    def do_summary(self, arg: str) -> None:
        """Show compact LLM-formatted summary for current state."""
        if arg.strip():
            console.print("  Usage: summary")
            return
        console.print(self.exp.get_llm_context(), markup=False)

    def do_context(self, arg: str) -> None:
        """Show detailed physical context (LLM tool-style output)."""
        if arg.strip():
            console.print("  Usage: context")
            return
        from trace.llm_agent import _tool_status, _tool_candidates
        console.print(_tool_status(self.exp), markup=False)
        console.print()
        if not self.exp.current().terminal:
            console.print(_tool_candidates(self.exp, top_k=8), markup=False)

    def do_llm(self, arg: str) -> None:
        """Run or configure LLM session.

        Examples:
          llm mode
          llm mode plan
          llm reset
          llm place this facility near factory_A
          llm place all remaining facilities optimizing for flow
          llm critique the current recommendations
          llm what would be a good position for this facility?
        """
        if self.exp.llm is None:
            console.print(
                "  [error]No LLM configured."
                " Start with --llm flag (e.g. --llm anthropic).[/error]"
            )
            return

        raw = arg.strip()
        if not raw:
            console.print("  Usage:")
            console.print("    llm mode \\[chat|plan|agent] \\[instruction]")
            console.print("    llm reset")
            console.print("    llm <instruction>")
            console.print(f"  Current mode: [bold]{self._llm_mode}[/bold]")
            console.print(f"  Session turns: [bold]{len(self._llm_messages)}[/bold] message(s)")
            return

        try:
            tokens = shlex.split(raw)
        except ValueError as e:
            console.print(f"  [error]Error: {e}[/error]")
            return

        if tokens[0] == "reset":
            self._llm_messages = []
            console.print("  [success]LLM session reset.[/success]")
            return

        goal = raw
        if tokens[0] == "mode":
            if len(tokens) == 1:
                console.print(f"  Current LLM mode: [bold]{self._llm_mode}[/bold]")
                return
            mode = tokens[1].strip().lower()
            if mode not in {"chat", "plan", "agent"}:
                console.print("  [error]Error: mode must be one of: chat, plan, agent[/error]")
                return
            self._llm_mode = mode
            if len(tokens) == 2:
                console.print(f"  LLM mode set to [bold]'{self._llm_mode}'[/bold].")
                return
            goal = " ".join(tokens[2:]).strip()
            if not goal:
                console.print(f"  LLM mode set to [bold]'{self._llm_mode}'[/bold].")
                return

        if not goal:
            console.print("  Usage: llm <instruction>")
            console.print("  Examples:")
            console.print("    llm place this near factory_A")
            console.print("    llm place all remaining facilities")
            console.print("    llm what's the best position for this?")
            return

        _last_tool_name = [None]

        def _on_step(event_type: str, text: str) -> None:
            if event_type == "thinking":
                for line in text.splitlines():
                    console.print(f"  [dim]\\[thinking][/dim] {line}")
            elif event_type == "tool_call":
                console.print(f"  [info]\\[tool][/info] {text}")
                _last_tool_name[0] = text.split("(")[0] if "(" in text else text
            elif event_type == "tool_result":
                for line in text.splitlines():
                    console.print(f"  [muted]\\[tool-result] {line}[/muted]")
                # Auto-update figure after state-changing tools
                if _last_tool_name[0] in ("place", "undo"):
                    try:
                        self._last_rendered_node = None
                        if self._auto_render:
                            self._auto_render_update()
                    except Exception:
                        pass
            elif event_type == "done":
                console.print(f"  [success]\\[done][/success] {text}")

        try:
            result = self.exp.llm_run(
                goal,
                on_step=_on_step,
                messages=self._llm_messages,
                mode=self._llm_mode,
            )
            self._llm_messages = result.messages
            if result.final_text and not any(
                result.final_text in msg.get("content", "")
                for msg in result.messages
                if isinstance(msg.get("content"), str)
            ):
                console.print()
                console.print(Panel(
                    RichMarkdown(result.final_text),
                    title="[bold]LLM Response[/bold]",
                    border_style="blue",
                    padding=(1, 2),
                ))
            console.print(
                f"  [muted]({result.steps_taken} turns,"
                f" stop={result.stop_reason}, mode={self._llm_mode})[/muted]"
            )
        except Exception as e:
            console.print(f"  [error]Error: {e}[/error]")

        self._last_agent_sig = None
        self._last_search_sig = None

    def do_reset(self, arg: str) -> None:
        """Reset the environment and tree."""
        self.exp.reset(
            options=self.exp.engine._reset_kwargs
            if hasattr(self.exp.engine, "_reset_kwargs") else None
        )
        self._last_agent_sig = None
        self._last_search_sig = None
        self._llm_messages = []
        console.print("  [success]\u2192 Environment reset.[/success]")
        self._show_status()

    def do_help(self, arg: str) -> None:
        """Show available commands."""
        if arg:
            super().do_help(arg)
            return

        sections = [
            ("Prediction", [
                ("agent", "Get agent recommendation and step"),
                ("agent predict", "Get agent recommendation only (no step)"),
                ("search", "Run search and step with result"),
                ("search predict", "Run search only (no step)"),
                ("signals", "Show all signals on current node"),
            ]),
            ("Execution", [
                ("step <idx>", "Step with specific action index"),
                ("auto [src] [n]", "Auto-play (src=agent|search, n=max steps)"),
            ]),
            ("Navigation", [
                ("undo", "Go back one step"),
                ("redo", "Re-advance after undo"),
                ("goto <node_id>", "Jump to any node"),
                ("path", "Show root \u2192 current path"),
            ]),
            ("Exploration", [
                ("branch [name]", "Save current path / list branches"),
                ("compare [b1 b2..]", "Compare branch costs"),
            ]),
            ("Analysis", [
                ("candidates [n]", "Show top-N candidate positions"),
                ("detail [node_id]", "Show physical placement detail"),
                ("tree [depth]", "Show decision tree"),
                ("query best", "Best terminal path"),
                ("query topk [k]", "Top-K terminal paths"),
                ("query agreement", "Signal agreement at current node"),
                ("query stats", "Depth statistics"),
                ("summary", "Compact LLM-formatted state summary"),
                ("context", "Physical context + top candidates"),
                ("status", "Current state summary"),
                ("render", "Open layout viewer (auto-updates on step)"),
                ("render on/off", "Toggle auto-render"),
                ("render save [path]", "Save layout to PNG"),
            ]),
            ("LLM (requires --llm)", [
                ("llm mode [m]", "Show/set mode (chat|plan|agent)"),
                ("llm reset", "Reset LLM conversation session"),
                ("llm <instruction>", "Continue session in current mode"),
            ]),
            ("Other", [
                ("reset", "Reset environment"),
                ("quit / q", "Exit"),
            ]),
        ]

        table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
        table.add_column(style="bold cyan", width=22)
        table.add_column(style="dim")

        for section_name, commands in sections:
            table.add_row(f"[bold magenta]{section_name}[/bold magenta]", "")
            for cmd_name, desc in commands:
                table.add_row(f"  {cmd_name}", desc)
            table.add_row("", "")

        console.print(Panel(table, title="[bold]Commands[/bold]", border_style="cyan", padding=(1, 1)))

    def do_quit(self, arg: str) -> bool:
        """Exit the REPL."""
        console.print("[info]Bye![/info]")
        return True

    do_q = do_quit
    do_EOF = do_quit

    def default(self, line: str) -> None:
        stripped = line.strip()
        if stripped:
            console.print(
                f"  [error]Unknown command: {stripped}[/error]."
                " Type [bold]'help'[/bold] for available commands."
            )

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
    parser.add_argument("--env", default="envs/env_configs/mixed_01.json", help="Env config JSON path")
    parser.add_argument("--method", default="greedyv4", help="Adapter method")
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

    console.print(f"[info]Loading {args.env} ...[/info]")
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

    llm_advisor = None
    if args.llm != "none":
        llm_advisor = _build_llm_agent(args.llm, args.llm_model)

    exp = Explorer(engine, adapter, agent, search=search,
                   ordering_agent=ordering_agent, llm=llm_advisor)
    exp.reset(options=loaded.reset_kwargs)

    exp.engine._reset_kwargs = loaded.reset_kwargs  # type: ignore[attr-defined]

    subtitle = args.env
    if search is not None:
        subtitle += f" + {type(search).__name__}"
    if llm_advisor is not None:
        subtitle += f" + LLM({args.llm})"
    _print_banner(subtitle)

    repl = ExplorerREPL(exp)
    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        console.print("\n[info]Bye![/info]")


if __name__ == "__main__":
    main()
