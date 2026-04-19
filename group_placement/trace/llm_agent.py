"""Agentic LLM advisor for interactive layout exploration via tool use.

The LLM interacts with the :class:`~trace.explorer.Explorer` through tools
(status, candidates, place, undo, ...) in a multi-turn loop — the same
operations a human performs in the REPL.

Provider-agnostic: :class:`BaseLLMBackend` defines the contract;
:class:`AnthropicBackend` and :class:`OpenAIBackend` handle format differences.

Usage::

    from group_placement.trace.llm_agent import ExplorerAgent, AnthropicBackend

    backend = AnthropicBackend()                  # needs ANTHROPIC_API_KEY
    agent = ExplorerAgent(backend=backend)
    result = agent.run("place everything optimising for flow", explorer)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Shared data structures
# =====================================================================

@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class BackendResponse:
    """Normalised LLM response (provider-agnostic)."""
    text: Optional[str]
    tool_calls: List[ToolCall]
    stop_reason: str  # "tool_use" | "end_turn" | "max_tokens"
    raw: Any = None  # original provider response for debugging


@dataclass
class ToolResult:
    """Result of a tool execution — text plus optional image."""
    text: str
    image_base64: Optional[str] = None
    image_media_type: str = "image/png"


@dataclass
class AgentResult:
    """Outcome of a single :meth:`ExplorerAgent.run` invocation."""
    messages: List[Dict[str, Any]]
    steps_taken: int
    final_text: str = ""
    stop_reason: str = "end_turn"


# =====================================================================
# Backend abstraction
# =====================================================================

class BaseLLMBackend(ABC):
    """Provider-agnostic interface for LLM chat with tool use."""

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> BackendResponse:
        """Send conversation with tool definitions, return normalised response."""
        ...

    @abstractmethod
    def make_tool_result_messages(
        self,
        tool_calls: List[ToolCall],
        results: List[ToolResult],
        raw_response: Any,
    ) -> Tuple[Dict[str, Any], Any]:
        """Build (assistant_msg, tool_results_msg) from executed tool calls.

        Returns two messages to append to the conversation:
        1. The assistant message (echoing back the raw response)
        2. The tool results message(s)
        """
        ...


# =====================================================================
# Anthropic backend
# =====================================================================

class AnthropicBackend(BaseLLMBackend):
    """Anthropic Messages API with tool use."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> BackendResponse:
        # Convert our tool format → Anthropic format
        api_tools = []
        for t in tools:
            api_tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })

        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=api_tools,
        )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in r.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return BackendResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=r.stop_reason or "end_turn",
            raw=r,
        )

    def make_tool_result_messages(
        self,
        tool_calls: List[ToolCall],
        results: List[ToolResult],
        raw_response: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Assistant message — echo back raw content blocks
        assistant_content = []
        for block in raw_response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input if isinstance(block.input, dict) else {},
                })
        assistant_msg = {"role": "assistant", "content": assistant_content}

        # Tool results — user message with tool_result blocks
        result_blocks = []
        for tc, result in zip(tool_calls, results):
            content: list = [{"type": "text", "text": result.text}]
            if result.image_base64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": result.image_media_type,
                        "data": result.image_base64,
                    },
                })
            result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": content,
            })
        tool_msg = {"role": "user", "content": result_blocks}

        return assistant_msg, tool_msg


# =====================================================================
# OpenAI backend
# =====================================================================

class OpenAIBackend(BaseLLMBackend):
    """OpenAI Chat Completions API with tool use."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def chat(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> BackendResponse:
        api_tools = []
        for t in tools:
            api_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })

        all_messages = [{"role": "system", "content": system}] + messages
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=all_messages,
            tools=api_tools if api_tools else None,
        )

        choice = r.choices[0]
        msg = choice.message

        text = msg.content or None
        tool_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        stop = "tool_use" if tool_calls else (choice.finish_reason or "end_turn")
        return BackendResponse(text=text, tool_calls=tool_calls, stop_reason=stop, raw=r)

    def make_tool_result_messages(
        self,
        tool_calls: List[ToolCall],
        results: List[ToolResult],
        raw_response: Any,
    ) -> Tuple[Dict[str, Any], Any]:
        choice = raw_response.choices[0]
        # Assistant message — include tool_calls
        assistant_msg = {"role": "assistant", "content": choice.message.content}
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        # Tool results — text only (OpenAI tool results don't support images)
        tool_msgs = []
        for tc, result in zip(tool_calls, results):
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.text,
            })

        return assistant_msg, tool_msgs


# =====================================================================
# Tool definitions
# =====================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "status",
        "description": (
            "Get current layout state: step number, placed facilities with "
            "positions, remaining facilities, current cost, and which facility "
            "is being placed next."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "candidates",
        "description": (
            "Get top-K placement candidates for the current facility. "
            "Returns position, delta_cost (cost increase if placed here), "
            "and agent score for each candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of candidates to return (default 8)",
                },
            },
        },
    },
    {
        "name": "explain",
        "description": (
            "Explain why a placement decision was made. Returns the chosen "
            "action, per-reward-component cost breakdown (e.g. how much flow "
            "vs. area contributed), the alternative top candidates considered, "
            "and signal info from each source (agent, search). Use this to "
            "answer 'why did you place X here?' or 'what were the other "
            "options?' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "Decision tree node ID. Omit for current node.",
                },
            },
        },
    },
    {
        "name": "detail",
        "description": (
            "Get physical placement detail for a specific step: facility ID, "
            "position, size, rotation, entry/exit ports, cost delta, and "
            "affected flow edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "Decision tree node ID. Omit for current node.",
                },
            },
        },
    },
    {
        "name": "flow_info",
        "description": (
            "Get flow connections for a facility: which other facilities it "
            "connects to, flow weights, and whether each neighbor is already "
            "placed (with position) or still remaining."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gid": {
                    "type": "string",
                    "description": "Facility group ID to query.",
                },
            },
            "required": ["gid"],
        },
    },
    {
        "name": "agent_recommend",
        "description": (
            "Get the greedy agent's recommended action for the current state. "
            "Returns recommended candidate index, value estimate, and top "
            "action scores. Does NOT execute the placement."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "place",
        "description": (
            "Place the current facility at the given candidate index. "
            "Returns the physical placement result (position, size, cost delta, "
            "affected flows)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {
                    "type": "integer",
                    "description": "Action index from the candidates list.",
                },
            },
            "required": ["candidate_index"],
        },
    },
    {
        "name": "undo",
        "description": "Undo the last placement, restoring the previous state.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "render",
        "description": (
            "Render the current layout as an image. Shows placed facilities "
            "(orange rectangles with IDs), forbidden zones (red), clearance "
            "halos (gray), flow arrows (blue), entry ports (green dots), exit "
            "ports (red dots), and the current cost. Use this to visually "
            "inspect the layout and verify placements."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that you are finished. Call this when the task is complete "
            "or you have nothing more to do."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was done.",
                },
            },
            "required": ["summary"],
        },
    },
]


# =====================================================================
# System prompt
# =====================================================================

SYSTEM_PROMPT = """\
You are a factory layout advisor. You help place facilities on a 2D grid to \
minimize total weighted material flow distance (Manhattan distance between \
entry/exit ports of connected facilities).

You have tools to inspect the current state, view candidates, check flow \
connections, render the layout visually, and execute placements. Use them \
to make informed decisions.

Key principles:
- Facilities with high flow weight should be placed close together.
- The flow distance is port-to-port (entry/exit), not center-to-center.
- Consider future placements — leave space for remaining high-flow neighbors.
- Use the agent_recommend tool to see what the greedy algorithm suggests, \
but you can override it if you have better reasoning.
- Use the render tool to visually inspect the layout — it shows facility \
positions, flow connections, and ports as an image.
- Call done() when finished.

Work step by step: check status, render to see the layout, look at candidates, \
reason about flow, then place."""


# =====================================================================
# Context helpers (reused from original implementation)
# =====================================================================

# =====================================================================
# Tool executors
# =====================================================================

def _tool_status(explorer) -> str:
    """Execute the 'status' tool."""
    summary = explorer.state_summary()
    node = explorer.current()
    state = explorer.engine.get_state()

    lines = [
        f"Step: {summary['step']}",
        f"Current facility: {node.group_id or '(all placed)'}",
        f"Cost: {summary['cost']:.1f}",
        f"Placed ({len(summary['placed'])}): {', '.join(summary['placed'])}",
        f"Remaining ({len(summary['remaining'])}): {', '.join(summary['remaining'])}",
        f"Terminal: {node.terminal}",
    ]

    # Include positions of placed facilities
    for gid in state.placed:
        p = state.placements.get(gid)
        if p is None:
            continue
        w = getattr(p, "w", 0)
        h = getattr(p, "h", 0)
        rot = getattr(p, "rotation", 0)
        rot_str = f" R{rot}" if rot else ""
        lines.append(f"  {gid}: ({p.x_center:.0f},{p.y_center:.0f}) {w:.0f}x{h:.0f}{rot_str}")

    return "\n".join(lines)


def _tool_candidates(explorer, top_k: int = 8) -> str:
    """Execute the 'candidates' tool."""
    node = explorer.current()
    if node.terminal:
        return "Terminal state — no candidates."

    prev_cap = explorer.candidates_top_k
    explorer.candidates_top_k = max(prev_cap, int(top_k))
    try:
        cands = explorer.candidates(top_k=top_k, compute_if_missing=True)
    finally:
        explorer.candidates_top_k = prev_cap
    if not cands:
        return "No valid candidates."

    state = explorer.engine.get_state()
    current_gid = state.remaining[0] if state.remaining else None
    spec = explorer.engine.group_specs.get(current_gid) if current_gid else None
    size_str = f" (size {int(spec.width)}x{int(spec.height)})" if spec else ""

    lines = [f"Candidates for {current_gid}{size_str}:"]
    for i, c in enumerate(cands):
        tag = " <-- agent-pick" if c.get("chosen") else ""
        pos = c.get("pos", [0.0, 0.0])
        parts = [
            f"idx={c['action']}",
            f"center=({pos[0]:.0f},{pos[1]:.0f})",
        ]
        if "delta" in c:
            parts.append(f"delta_cost={c['delta']:+.1f}")
        if "score" in c:
            parts.append(f"score={c['score']:.3f}")
        if "variant" in c:
            parts.append(f"variant={c['variant']}")
        lines.append(f"  #{i+1} " + " ".join(parts) + tag)
    return "\n".join(lines)


def _tool_explain(explorer, node_id: Optional[int] = None) -> str:
    """Execute the 'explain' tool — structured decision explanation."""
    try:
        info = explorer.explain(node_id)
    except KeyError:
        return f"Node {node_id} not found."

    lines: List[str] = [
        f"Node {info['node_id']} (step {info['step']}, gid={info['gid']}):",
    ]
    if info.get("chosen_by") is not None:
        lines.append(f"  chosen_by: {info['chosen_by']}  action={info['chosen_action']}")

    phys = info.get("physical") or {}
    if phys:
        lines.append(
            f"  placement: center=({phys['x_center']:.0f},{phys['y_center']:.0f}) "
            f"size={phys['w']:.0f}x{phys['h']:.0f} rot={phys['rotation']}"
        )
        lines.append(
            f"  cost: {phys['cost_before']:.1f} -> {phys['cost_after']:.1f} "
            f"(delta={phys['delta_cost']:+.2f})"
        )
        breakdown = phys.get("breakdown") or {}
        if breakdown:
            parts = [f"{name}={rec.get('delta', 0.0):+.2f}" for name, rec in breakdown.items()]
            lines.append(f"  breakdown: {', '.join(parts)}")
            for name, rec in breakdown.items():
                meta_after = rec.get("metadata_after") or {}
                if not meta_after:
                    continue
                try:
                    meta_json = json.dumps(
                        meta_after, default=str, ensure_ascii=False, separators=(",", ":")
                    )
                except (TypeError, ValueError):
                    meta_json = str(meta_after)
                if len(meta_json) > 400:
                    meta_json = meta_json[:397] + "..."
                lines.append(f"    {name} metadata: {meta_json}")

    signals = info.get("signals") or {}
    for src, sig in signals.items():
        lines.append(f"  [{src}] recommended_action={sig['recommended_action']}")
        for c in sig.get("candidates", [])[:5]:
            tag = " <-- chosen" if c.get("chosen") else ""
            parts = [f"idx={c['action']}", f"pos=({c['pos'][0]:.0f},{c['pos'][1]:.0f})"]
            if "delta" in c:
                parts.append(f"delta={c['delta']:+.2f}")
            if "visits" in c:
                parts.append(f"visits={int(c['visits'])}")
            if "score" in c:
                parts.append(f"score={c['score']:.3f}")
            lines.append(f"    rank {c['rank']}: " + " ".join(parts) + tag)

    if not phys and not signals:
        lines.append("  (no placement or signals recorded at this node)")
    return "\n".join(lines)


def _tool_detail(explorer, node_id: Optional[int] = None) -> str:
    """Execute the 'detail' tool."""
    try:
        info = explorer.detail(node_id)
    except KeyError:
        return f"Node {node_id} not found."

    phys = info.get("physical")
    if phys is None:
        return f"Node {info['node_id']} has no physical context."

    lines = [
        f"Node {info['node_id']} placement:",
        f"  Facility: {phys['gid']}",
        f"  Center: ({phys['x_center']:.1f}, {phys['y_center']:.1f})",
        f"  Bottom-left: ({phys['x']:.1f}, {phys['y']:.1f})",
        f"  Size: {phys['w']:.0f} x {phys['h']:.0f}",
        f"  Rotation: {phys['rotation']}",
        f"  Cost: {phys['cost_before']:.1f} -> {phys['cost_after']:.1f} "
        f"(delta={phys['delta_cost']:+.1f})",
    ]
    entries = phys.get("entries") or []
    if entries:
        lines.append("  Entries: " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in entries))
    exits_ = phys.get("exits") or []
    if exits_:
        lines.append("  Exits: " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in exits_))
    flows = phys.get("affected_flows") or []
    if flows:
        lines.append("  Affected flows:")
        for fd in flows:
            lines.append(
                f"    {fd['src']} -> {fd['dst']}: "
                f"weight={float(fd['weight']):.1f} dist={float(fd['distance']):.1f}"
            )
    return "\n".join(lines)


def _tool_flow_info(explorer, gid: str) -> str:
    """Execute the 'flow_info' tool."""
    engine = explorer.engine
    state = engine.get_state()
    gflow = engine.group_flow

    lines = [f"Flow connections for {gid}:"]

    # outgoing
    for dst, w in gflow.get(gid, {}).items():
        if dst in state.placed:
            p = state.placements[dst]
            lines.append(f"  {gid} -> {dst}: weight={w:.1f} (placed at {p.x_center:.0f},{p.y_center:.0f})")
        else:
            lines.append(f"  {gid} -> {dst}: weight={w:.1f} (not yet placed)")

    # incoming
    for src, dsts in gflow.items():
        if gid in dsts and src != gid:
            w = dsts[gid]
            if src in state.placed:
                p = state.placements[src]
                lines.append(f"  {src} -> {gid}: weight={w:.1f} (placed at {p.x_center:.0f},{p.y_center:.0f})")
            else:
                lines.append(f"  {src} -> {gid}: weight={w:.1f} (not yet placed)")

    if len(lines) == 1:
        lines.append("  (no flow connections)")
    return "\n".join(lines)


def _tool_agent_recommend(explorer) -> str:
    """Execute the 'agent_recommend' tool."""
    node = explorer.current()
    if node.terminal:
        return "Terminal state — no recommendation."

    sig = explorer.predict_agent()
    top_k = np.argsort(-sig.scores)[:5]
    lines = [
        f"Agent recommendation:",
        f"  Best action: idx={sig.recommended_action} (value={sig.recommended_value:.4f})",
        f"  Top actions:",
    ]
    for idx in top_k:
        if sig.scores[idx] <= 0:
            break
        lines.append(f"    idx={idx} score={sig.scores[idx]:.3f}")
    return "\n".join(lines)


def _tool_place(explorer, candidate_index: int) -> str:
    """Execute the 'place' tool."""
    node = explorer.current()
    if node.terminal:
        return "Error: terminal state — cannot place."

    if "agent" not in node.signals:
        explorer.predict_agent()

    try:
        child = explorer.step(candidate_index, chosen_by="llm")
    except Exception as e:
        return f"Error placing candidate {candidate_index}: {e}"

    parent = explorer.tree.nodes[child.parent_id]
    phys = parent.physical
    if phys:
        result = phys.summary()
        if phys.affected_flows:
            flow_lines = [f"  {fd.src}->{fd.dst} w={fd.weight:.1f} d={fd.distance:.1f}"
                          for fd in phys.affected_flows]
            result += "\nAffected flows:\n" + "\n".join(flow_lines)
        return result
    return f"Placed at node {child.id}, cost={child.cost_after}"


def _tool_render(explorer) -> ToolResult:
    """Render current layout as PNG and return base64 image."""
    import io
    import base64
    import matplotlib
    import matplotlib.pyplot as plt
    from group_placement.envs.visualizer.data import extract_layout_data
    from group_placement.envs.visualizer.mpl import _draw_layout_from_data

    # Use non-interactive backend for rendering to buffer
    prev_backend = matplotlib.get_backend()
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    try:
        data = extract_layout_data(explorer.engine)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, data.grid_width)
        ax.set_ylim(0, data.grid_height)
        ax.set_aspect("equal")
        ax.set_title(f"Layout (cost={data.cost:.1f}, {len(data.facilities)} placed)")
        _draw_layout_from_data(ax, data)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Text summary for non-vision backends
        fac_names = [f.group_id for f in data.facilities]
        text = (
            f"Layout rendered: grid {data.grid_width}x{data.grid_height}, "
            f"cost={data.cost:.1f}, "
            f"{len(data.facilities)} placed ({', '.join(fac_names)}), "
            f"{len(data.flow_arrows)} flow arrows"
        )
        return ToolResult(text=text, image_base64=img_b64)
    finally:
        try:
            matplotlib.use(prev_backend, force=True)
        except Exception:
            pass


def _tool_undo(explorer) -> str:
    """Execute the 'undo' tool."""
    result = explorer.undo()
    if result is None:
        return "Nothing to undo (at root)."
    return f"Undone. Now at node {result.id}, step={result.step}, gid={result.group_id or 'done'}"


# =====================================================================
# ExplorerAgent
# =====================================================================

StepCallback = Callable[[str, str], None]  # (event_type, text) → None

MODE_TOOL_NAMES: Dict[str, set[str]] = {
    # conversation-focused: inspect only
    "chat": {"status", "explain", "detail", "flow_info", "render", "done"},
    # planning: inspect + recommendations
    "plan": {"status", "candidates", "explain", "detail", "flow_info", "agent_recommend", "render", "done"},
    # execution: full tool access
    "agent": {t["name"] for t in TOOL_SCHEMAS},
}


class ExplorerAgent:
    """Agentic LLM that interacts with Explorer via tool calls.

    Parameters
    ----------
    backend : BaseLLMBackend
        Provider-specific LLM backend.
    system_prompt : str, optional
        Override the default system prompt.
    max_turns : int
        Maximum number of LLM ↔ tool round-trips.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_turns: int = 30,
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def run(
        self,
        goal: str,
        explorer: Any,
        *,
        on_step: Optional[StepCallback] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        mode: str = "agent",
    ) -> AgentResult:
        """Run the agentic loop until done or max_turns reached.

        Parameters
        ----------
        goal : str
            User's instruction (e.g. "place all facilities", "place B near A").
        explorer : Explorer
            The explorer instance to interact with.
        on_step : callable, optional
            ``(event_type, text)`` callback for streaming.
            event_type: ``"thinking"``, ``"tool_call"``, ``"tool_result"``,
            ``"done"``.
        """
        mode_name = mode.strip().lower() if mode else "agent"
        if mode_name not in MODE_TOOL_NAMES:
            raise ValueError(f"Unknown LLM mode: {mode_name}")

        allowed_tool_names = MODE_TOOL_NAMES[mode_name]
        tools = [t for t in TOOL_SCHEMAS if t["name"] in allowed_tool_names]

        history: List[Dict[str, Any]] = list(messages) if messages else []
        history.append({"role": "user", "content": goal})

        final_text = ""
        stop_reason = "max_turns"
        turns_taken = 0

        for turn in range(self.max_turns):
            turns_taken = turn + 1
            response = self.backend.chat(
                system=self.system_prompt,
                messages=history,
                tools=tools,
            )

            # Emit thinking text
            if response.text and on_step:
                on_step("thinking", response.text)
            if response.text:
                final_text = response.text

            # No tool calls → done
            if not response.tool_calls:
                stop_reason = "no_tool_calls"
                break

            # Execute tools
            results: List[ToolResult] = []
            for tc in response.tool_calls:
                if on_step:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
                    on_step("tool_call", f"{tc.name}({args_str})")

                if tc.name not in allowed_tool_names:
                    result = ToolResult(text=f"Tool '{tc.name}' is not allowed in mode '{mode_name}'.")
                else:
                    result = self._execute(tc, explorer)
                results.append(result)

                if on_step:
                    display = result.text if len(result.text) < 300 else result.text[:300] + "..."
                    on_step("tool_result", display)

            # Build messages for next turn
            assistant_msg, tool_result_msg = self.backend.make_tool_result_messages(
                response.tool_calls, results, response.raw,
            )
            history.append(assistant_msg)

            # tool_result_msg can be a single dict or a list (OpenAI)
            if isinstance(tool_result_msg, list):
                history.extend(tool_result_msg)
            else:
                history.append(tool_result_msg)

            # Check if 'done' tool was called
            if any(tc.name == "done" for tc in response.tool_calls):
                stop_reason = "done_tool"
                if on_step:
                    done_tc = next(tc for tc in response.tool_calls if tc.name == "done")
                    on_step("done", done_tc.arguments.get("summary", ""))
                break

        return AgentResult(
            messages=history,
            steps_taken=turns_taken,
            final_text=final_text,
            stop_reason=stop_reason,
        )

    def _execute(self, tc: ToolCall, explorer: Any) -> ToolResult:
        """Dispatch a tool call to the corresponding executor."""
        try:
            name = tc.name
            args = tc.arguments

            if name == "render":
                return _tool_render(explorer)

            # All other tools return plain text
            if name == "status":
                text = _tool_status(explorer)
            elif name == "candidates":
                text = _tool_candidates(explorer, top_k=args.get("top_k", 8))
            elif name == "explain":
                text = _tool_explain(explorer, node_id=args.get("node_id"))
            elif name == "detail":
                text = _tool_detail(explorer, node_id=args.get("node_id"))
            elif name == "flow_info":
                gid = args.get("gid", "")
                if not gid:
                    text = "Error: gid is required."
                else:
                    text = _tool_flow_info(explorer, gid=gid)
            elif name == "agent_recommend":
                text = _tool_agent_recommend(explorer)
            elif name == "place":
                idx = args.get("candidate_index")
                if idx is None:
                    text = "Error: candidate_index is required."
                else:
                    text = _tool_place(explorer, candidate_index=int(idx))
            elif name == "undo":
                text = _tool_undo(explorer)
            elif name == "done":
                text = args.get("summary", "Done.")
            else:
                text = f"Unknown tool: {name}"
            return ToolResult(text=text)
        except Exception as e:
            logger.warning("Tool execution error: %s", e, exc_info=True)
            return ToolResult(text=f"Error: {e}")
