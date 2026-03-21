"""Visualization package with pluggable backends (matplotlib, plotly).

Usage:
    from envs.visualizer import plot_layout, save_layout
    plot_layout(env)                          # matplotlib (default)
    plot_layout(env, backend="plotly")        # plotly (interactive HTML)
"""
from __future__ import annotations

from typing import Any, Optional, List

from envs.visualizer.data import StepFrame, extract_layout_data

_DEFAULT_BACKEND = "matplotlib"


def _get_backend(backend: Optional[str] = None):
    name = (backend or _DEFAULT_BACKEND).lower()
    if name in ("matplotlib", "mpl"):
        from envs.visualizer.mpl import MatplotlibBackend
        return MatplotlibBackend()
    elif name in ("plotly",):
        from envs.visualizer.plotly import PlotlyBackend
        return PlotlyBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'matplotlib' or 'plotly'.")


def plot_layout(
    env: Any,
    *,
    action_space: Any = None,
    routes: Any = None,
    backend: Optional[str] = None,
) -> Any:
    """Interactive layout viewer.

    Args:
        env: FactoryLayoutEnv or wrapper
        action_space: Optional CandidateSet
        routes: Optional list of RouteResult
        backend: 'matplotlib' (default) or 'plotly'
    """
    engine = getattr(env, "engine", env)
    data = extract_layout_data(engine, action_space=action_space, routes=routes)
    return _get_backend(backend).plot_layout(data)


def save_layout(
    env: Any,
    *,
    save_path: str,
    show_masks: bool = True,
    show_flow: bool = False,
    show_score: bool = False,
    show_zones: bool = False,
    action_space: Any = None,
    backend: Optional[str] = None,
) -> None:
    """Save a static layout image.

    Args:
        env: FactoryLayoutEnv or wrapper
        save_path: Output file path (.png, .html, etc.)
        show_masks: Show forbidden area masks
        show_flow: Show flow arrows
        show_score: Show cost text
        show_zones: Show constraint zone rects
        action_space: Optional CandidateSet
        backend: 'matplotlib' (default) or 'plotly'
    """
    engine = getattr(env, "engine", env)
    data = extract_layout_data(engine, action_space=action_space)
    _get_backend(backend).save_layout(
        data, save_path,
        show_masks=show_masks,
        show_flow=show_flow,
        show_score=show_score,
        show_zones=show_zones,
    )


def browse_steps(
    env: Any,
    *,
    frames: List[StepFrame],
    backend: Optional[str] = None,
    **kwargs,
) -> None:
    """Browse step-by-step inference frames.

    Note: Only 'matplotlib' backend supports this (requires keyboard events).
    """
    _get_backend(backend).browse_steps(env, frames=frames, **kwargs)


def plot_flow_graph(env: Any, *, show_weights: bool = True, backend: Optional[str] = None) -> None:
    """Plot directed flow graph."""
    engine = getattr(env, "engine", env)
    _get_backend(backend).plot_flow_graph(engine.group_flow, show_weights=show_weights)


def draw_layout_layers(
    *,
    ax: Any,
    engine: Any,
    action_space: Any = None,
    routes: Any = None,
) -> Any:
    """Draw layout layers onto a pre-existing matplotlib Axes.

    This is always matplotlib — no backend parameter (callers manage their own fig/ax).
    """
    from envs.visualizer.mpl import MatplotlibBackend
    return MatplotlibBackend().draw_layout_on_axes(
        engine, ax=ax, action_space=action_space, routes=routes,
    )
