"""Abstract base class for visualization backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, List

from envs.visualizer.data import LayoutData, StepFrame


class VisualizerBackend(ABC):
    """Interface that all visualization backends must implement."""

    @abstractmethod
    def plot_layout(self, data: LayoutData, **display_kwargs) -> Any:
        """Show an interactive layout viewer."""
        ...

    @abstractmethod
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
        """Save a static layout image to disk."""
        ...

    @abstractmethod
    def browse_steps(
        self,
        env: Any,
        *,
        frames: List[StepFrame],
        **kwargs,
    ) -> None:
        """Browse step-by-step inference frames."""
        ...

    @abstractmethod
    def plot_flow_graph(self, group_flow: dict, *, show_weights: bool = True) -> None:
        """Plot directed flow graph."""
        ...

    @abstractmethod
    def draw_layout_on_axes(
        self,
        engine: Any,
        *,
        ax: Any,
        action_space: Any = None,
        routes: Any = None,
    ) -> Any:
        """Draw layout layers onto a pre-existing matplotlib Axes.

        This is matplotlib-specific. Other backends should raise NotImplementedError.
        """
        ...
