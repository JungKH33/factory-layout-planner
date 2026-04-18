"""Plotly visualization backend.

Provides interactive, web-native visualization with hover, zoom, and click-to-toggle.
Requires: pip install plotly (optional dependency with lazy import).
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from group_placement.envs.visualizer.base import VisualizerBackend
from group_placement.envs.visualizer.data import LayoutData, StepFrame


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "Plotly backend requires plotly. Install with: pip install plotly"
        )


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_rect_trace(go, x0, y0, x1, y1, *, color, alpha, name, showlegend=False):
    """Create a filled rectangle as a Scatter trace."""
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        fill="toself",
        fillcolor=_hex_to_rgba(color, alpha),
        line=dict(color=_hex_to_rgba(color, min(1.0, alpha * 3)), width=1),
        mode="lines",
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        hoverinfo="name",
    )


def _build_polygon_trace(go, points, *, color, alpha, name, showlegend=False, line_dash=None):
    """Create a filled polygon as a Scatter trace."""
    xs = [p[0] for p in points] + [points[0][0]]
    ys = [p[1] for p in points] + [points[0][1]]
    line_kw = dict(color=color, width=1.2)
    if line_dash:
        line_kw["dash"] = line_dash
    return go.Scatter(
        x=xs, y=ys,
        fill="toself",
        fillcolor=_hex_to_rgba(color, alpha) if "#" in color else color,
        line=line_kw,
        mode="lines",
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        hoverinfo="name",
    )


class PlotlyBackend(VisualizerBackend):

    def plot_layout(self, data: LayoutData, **display_kwargs) -> Any:
        go = _require_plotly()
        fig = self._build_figure(go, data)
        fig.show()
        return fig

    def save_layout(
        self,
        data: LayoutData,
        save_path: str,
        *,
        show_masks: bool = True,
        show_flow: bool = False,
        show_score: bool = False,
        show_zones: bool = False,
    ) -> None:
        go = _require_plotly()
        fig = self._build_figure(
            go, data,
            show_masks=show_masks,
            show_flow=show_flow,
            show_score=show_score,
            show_zones=show_zones,
        )
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            try:
                fig.write_image(save_path, width=1300, height=650, scale=2)
            except Exception:
                # kaleido not installed — fall back to HTML
                html_path = save_path.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_path)

    def browse_steps(self, env: Any, *, frames: List[StepFrame], **kwargs) -> None:
        raise NotImplementedError(
            "browse_steps requires matplotlib (keyboard navigation). "
            "Use backend='matplotlib' for step browsing."
        )

    def plot_flow_graph(self, group_flow: dict, *, show_weights: bool = True) -> None:
        go = _require_plotly()
        import networkx as nx

        G = nx.DiGraph()
        for src, targets in group_flow.items():
            for dst, weight in targets.items():
                G.add_edge(src, dst, weight=weight)

        if G.number_of_nodes() == 0:
            return

        pos = nx.spring_layout(G, seed=42)

        # Edge traces
        edge_traces = []
        annotations = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.5, color="#888"),
                hoverinfo="none",
                showlegend=False,
            ))
            # Arrow annotation
            annotations.append(dict(
                ax=x0, ay=y0, axref="x", ayref="y",
                x=x1, y=y1, xref="x", yref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor="#555",
            ))
            if show_weights:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                annotations.append(dict(
                    x=mx, y=my, text=f"{d.get('weight', 1.0):.2f}",
                    showarrow=False, font=dict(size=10, color="#333"),
                    bgcolor="white", opacity=0.8,
                ))

        # Node trace
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [str(n) for n in G.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=30, color="#1f77b4", line=dict(width=2, color="white")),
            text=node_text, textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Flow Graph",
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=800, height=600,
        )
        fig.show()

    def draw_layout_on_axes(self, engine: Any, *, ax: Any, action_space: Any = None) -> Any:
        raise NotImplementedError(
            "draw_layout_on_axes is matplotlib-specific. "
            "Use backend='matplotlib' for Axes-level drawing."
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_figure(
        self,
        go,
        data: LayoutData,
        *,
        show_masks: bool = True,
        show_flow: bool = True,
        show_score: bool = True,
        show_zones: bool = True,
    ):
        fig = go.Figure()

        # --- Factory boundary ---
        W, H = data.grid_width, data.grid_height
        fig.add_trace(go.Scatter(
            x=[0, W, W, 0, 0],
            y=[0, 0, H, H, 0],
            mode="lines",
            line=dict(color="black", width=2),
            name="boundary",
            showlegend=False,
            hoverinfo="skip",
        ))

        # --- Constraint zone heatmaps ---
        if show_zones:
            for name in data.constraint_names:
                zone = data.constraint_zones.get(name)
                if zone is None:
                    continue
                legend_name = f"zone:{name}"
                first = True
                if zone.heatmap is not None:
                    fig.add_trace(go.Heatmap(
                        z=zone.heatmap,
                        x0=0, dx=1, y0=0, dy=1,
                        colorscale=[[0, _hex_to_rgba(zone.color, 0.1)],
                                    [1, _hex_to_rgba(zone.color, 0.5)]],
                        showscale=False,
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=True,
                        hoverinfo="z",
                        visible="legendonly",
                    ))
                    first = False
                for area in zone.rects:
                    rect = area.get("rect", None)
                    if rect is None:
                        continue
                    x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                    if x1 <= x0 or y1 <= y0:
                        continue
                    fig.add_trace(_build_rect_trace(
                        go, x0, y0, x1, y1,
                        color=zone.color, alpha=0.07,
                        name=legend_name, showlegend=first,
                    ))
                    if first:
                        first = False
                    # Label
                    val = area.get("value", None)
                    if val is not None:
                        fig.add_trace(go.Scatter(
                            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
                            mode="text",
                            text=[f"{name}{zone.op}{val}"],
                            textfont=dict(size=10, color=zone.color),
                            showlegend=False,
                            legendgroup=legend_name,
                            hoverinfo="skip",
                            visible="legendonly",
                        ))

        # --- Forbidden rects ---
        if show_masks and data.forbidden_rects:
            first_forb = True
            for x0, y0, x1, y1 in data.forbidden_rects:
                fig.add_trace(_build_rect_trace(
                    go, x0, y0, x1, y1,
                    color="#d62728", alpha=0.15,
                    name="forbidden", showlegend=first_forb,
                ))
                first_forb = False

        # --- Placed facilities ---
        first_fac = True
        for fac in data.facilities:
            if fac.body_polygon_abs:
                fig.add_trace(_build_polygon_trace(
                    go, fac.body_polygon_abs,
                    color="#ff8c00", alpha=0.6,
                    name="facilities", showlegend=first_fac,
                ))
                if fac.clearance_polygon_abs:
                    fig.add_trace(_build_polygon_trace(
                        go, fac.clearance_polygon_abs,
                        color="#999999", alpha=0.0,
                        name="clearance", showlegend=False,
                        line_dash="dash",
                    ))
            else:
                fig.add_trace(_build_rect_trace(
                    go, fac.x_bl, fac.y_bl, fac.x_bl + fac.w, fac.y_bl + fac.h,
                    color="#ff8c00", alpha=0.6,
                    name="facilities", showlegend=first_fac,
                ))
            first_fac = False
            # Label
            fig.add_trace(go.Scatter(
                x=[fac.x_center], y=[fac.y_center],
                mode="text",
                text=[str(fac.group_id)],
                textfont=dict(size=10, color="black"),
                showlegend=False,
                legendgroup="facilities",
                hoverinfo="text",
                hovertext=f"{fac.group_id} ({fac.w}x{fac.h})",
            ))

        # --- Action space candidates ---
        if data.candidates_xy:
            xs = [c[0] for c in data.candidates_xy]
            ys = [c[1] for c in data.candidates_xy]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(size=5, color="green", opacity=0.65),
                name="action_space",
                legendgroup="action_space",
                showlegend=True,
                hoverinfo="x+y",
            ))

        # --- Flow arrows ---
        if show_flow and data.flow_arrows:
            arrow_annotations = []
            first_flow = True
            for arrow in data.flow_arrows:
                fig.add_trace(go.Scatter(
                    x=[arrow.src_xy[0], arrow.dst_xy[0]],
                    y=[arrow.src_xy[1], arrow.dst_xy[1]],
                    mode="lines",
                    line=dict(color="rgba(65,105,225,0.7)", width=1.2),
                    name="flow",
                    legendgroup="flow",
                    showlegend=first_flow,
                    hoverinfo="text",
                    hovertext=f"weight={arrow.weight:.2f}",
                ))
                first_flow = False
                arrow_annotations.append(dict(
                    ax=arrow.src_xy[0], ay=arrow.src_xy[1],
                    axref="x", ayref="y",
                    x=arrow.dst_xy[0], y=arrow.dst_xy[1],
                    xref="x", yref="y",
                    showarrow=True, arrowhead=3, arrowsize=1,
                    arrowwidth=1.2, arrowcolor="rgba(65,105,225,0.7)",
                ))

            # Ports
            def _add_port_scatter(xys, *, color, alpha, name, first):
                if not xys:
                    return first
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in xys],
                    y=[p[1] for p in xys],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=alpha,
                                line=dict(width=0.8, color="white")),
                    name=name,
                    legendgroup="flow",
                    showlegend=False,
                    hoverinfo="x+y",
                ))
                return False

            first_flow = _add_port_scatter(data.ports.inactive_entries, color="#2ca02c", alpha=0.4, name="entry (inactive)", first=first_flow)
            first_flow = _add_port_scatter(data.ports.active_entries, color="#2ca02c", alpha=0.9, name="entry (active)", first=first_flow)
            first_flow = _add_port_scatter(data.ports.inactive_exits, color="#d62728", alpha=0.4, name="exit (inactive)", first=first_flow)
            _add_port_scatter(data.ports.active_exits, color="#d62728", alpha=0.9, name="exit (active)", first=first_flow)

            fig.update_layout(annotations=arrow_annotations)

        # --- Score annotation ---
        title_text = "FactoryLayoutEnv"
        if show_score:
            title_text += f"  (cost={data.cost:.3f})"

        fig.update_layout(
            title=title_text,
            xaxis=dict(
                range=[0, data.grid_width],
                scaleanchor="y", scaleratio=1,
                showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                range=[0, data.grid_height],
                showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            ),
            plot_bgcolor="white",
            width=1300, height=650,
            legend=dict(
                title="Layers (click to toggle)",
                x=1.02, y=1, xanchor="left",
            ),
        )

        return fig
