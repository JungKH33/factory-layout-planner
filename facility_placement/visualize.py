"""Facility-level placement visualization.

Independent from ``envs.visualizer`` but mirrors its API shape:

- **High-level** (mirror of ``envs.visualizer.plot_layout`` / ``save_layout``):
    * ``plot_facility_layout(state_dict, ...)`` — interactive viewer, returns
      ``(fig, ax)``.
    * ``save_facility_layout(state_dict, save_path=..., ...)`` — render and
      write to disk.

- **Low-level** (mirror of ``envs.visualizer.draw_layout_layers``):
    * ``draw_cluster_outlines(ax, cluster_placements, ...)`` — cluster bboxes
      in world mm.
    * ``draw_facility_rects(ax, facility_placements, ...)`` — facility
      rectangles + ports; supports both ``axes_units="world_mm"`` and
      ``"grid_cells"`` overlays.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Mapping, Optional, Tuple

import matplotlib.patches as mpatches

from .schema import FacilityPlacement

# Cluster bounding box (Phase 2): blue stroke, no fill.
# Ports: match ``mpl._scatter_ports`` (#2ca02c / #d62728, circles, white rim).
_CLUSTER_OUTLINE_COLOR = "steelblue"
_CLUSTER_OUTLINE_LINEWIDTH = 1.25
_GROUP_PATCH_LINEWIDTH = 1.2  # facility rect edges — match mpl group linewidth
_PORT_ENTRY_COLOR = "#2ca02c"
_PORT_EXIT_COLOR = "#d62728"
_PORT_SCATTER_S = 28.0
_PORT_EDGE_COLORS = "white"
_PORT_LINEWIDTHS = 0.8
_PORT_ALPHA = 0.9
_PORT_MARKER = "o"
_PORT_ZORDER = 4.0


# ---------------------------------------------------------------------------
# Low-level axes drawers (callers own the figure/axes)
# ---------------------------------------------------------------------------

def draw_cluster_outlines(
    ax: Any,
    cluster_placements: Iterable[Mapping[str, Any]],
    *,
    edgecolor: str = _CLUSTER_OUTLINE_COLOR,
    linewidth: float = _CLUSTER_OUTLINE_LINEWIDTH,
    linestyle: str = "-",
    facecolor: str = "none",
    zorder: float = 2.0,
    show_gid: bool = False,
    gid_fontsize: int = 8,
    gid_color: Optional[str] = None,
) -> List[Any]:
    """Draw cluster (group) bounding boxes in world mm — **outline only**.

    Default ``edgecolor`` is ``steelblue`` (no interior fill).

    ``cluster_placements`` entries match interchange ``export_group_placement`` items:
    ``x_bl_mm``, ``y_bl_mm``, ``cluster_w_mm``, ``cluster_h_mm``, optional ``gid``.
    """
    artists: List[Any] = []
    gc = gid_color if gid_color is not None else edgecolor

    for entry in cluster_placements:
        x = float(entry.get("x_bl_mm", 0.0))
        y = float(entry.get("y_bl_mm", 0.0))
        w = float(entry.get("cluster_w_mm", 0.0))
        h = float(entry.get("cluster_h_mm", 0.0))
        if w <= 0 or h <= 0:
            continue

        rect = mpatches.Rectangle(
            (x, y), w, h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        ax.add_patch(rect)
        artists.append(rect)

        if show_gid and entry.get("gid") is not None:
            t = ax.text(
                x + w / 2.0, y + h / 2.0, str(entry["gid"]),
                ha="center", va="center",
                fontsize=gid_fontsize, color=gc, zorder=zorder + 0.1,
            )
            artists.append(t)

    return artists


def draw_facility_rects(
    ax: Any,
    facility_placements: Iterable[FacilityPlacement],
    *,
    grid_size_mm: Optional[float] = None,
    axes_units: Literal["grid_cells", "world_mm"] = "world_mm",
    show_ids: bool = True,
    show_ports: bool = True,
    edgecolor: str = "black",
    linewidth: float = _GROUP_PATCH_LINEWIDTH,
    id_fontsize: int = 6,
    entry_color: str = _PORT_ENTRY_COLOR,
    exit_color: str = _PORT_EXIT_COLOR,
    port_marker_size: float = _PORT_SCATTER_S,
    port_edgecolors: str = _PORT_EDGE_COLORS,
    port_linewidths: float = _PORT_LINEWIDTHS,
    port_alpha: float = _PORT_ALPHA,
    port_marker: str = _PORT_MARKER,
    zorder: float = 3.0,
) -> List[Any]:
    """Draw facility rectangles and ports onto an existing Axes.

    Args:
        ax: A pre-created matplotlib ``Axes``.
        facility_placements: ``FacilityPlacement`` geometry in world mm.
        axes_units:
            ``world_mm`` (default) — use mm as-is.
            ``grid_cells`` — divide mm by ``grid_size_mm`` so axes match
            ``envs.visualizer`` cell coordinates (legacy overlay).
        grid_size_mm: Required when ``axes_units=="grid_cells"``; ignored for
            ``world_mm``.
        zorder: Facility-rect layer; ports use at least ``zorder+1`` (aligned with
            ``mpl`` port zorder 4).
        port_marker / port_edgecolors / port_linewidths / port_alpha: Match
            ``envs.visualizer.mpl._scatter_ports`` (circle markers, white rims).

    Returns:
        List of matplotlib artists created (useful for toggling/tests).
    """
    if axes_units == "grid_cells":
        if grid_size_mm is None or grid_size_mm <= 0:
            raise ValueError(
                f"grid_size_mm must be > 0 when axes_units='grid_cells', got {grid_size_mm!r}"
            )
        inv_scale = 1.0 / float(grid_size_mm)
    elif axes_units == "world_mm":
        inv_scale = 1.0
    else:
        raise ValueError(f"unknown axes_units: {axes_units!r}")

    artists: List[Any] = []

    entry_xs: List[float] = []
    entry_ys: List[float] = []
    exit_xs: List[float] = []
    exit_ys: List[float] = []

    for fp in facility_placements:
        x = float(fp.x_mm) * inv_scale
        y = float(fp.y_mm) * inv_scale
        w = float(fp.width_mm) * inv_scale
        h = float(fp.height_mm) * inv_scale

        rect = mpatches.Rectangle(
            (x, y), w, h,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(rect)
        artists.append(rect)

        if show_ids:
            txt = ax.text(
                x + w / 2.0, y + h / 2.0, str(fp.fid),
                ha="center", va="center",
                fontsize=id_fontsize, color=edgecolor,
                zorder=zorder + 0.1,
            )
            artists.append(txt)

        if show_ports:
            for px_mm, py_mm in fp.entry_points_abs_mm:
                entry_xs.append(float(px_mm) * inv_scale)
                entry_ys.append(float(py_mm) * inv_scale)
            for px_mm, py_mm in fp.exit_points_abs_mm:
                exit_xs.append(float(px_mm) * inv_scale)
                exit_ys.append(float(py_mm) * inv_scale)

    port_z = max(zorder + 1.0, _PORT_ZORDER)
    if show_ports and entry_xs:
        sc_in = ax.scatter(
            entry_xs, entry_ys,
            s=port_marker_size,
            c=entry_color,
            marker=port_marker,
            alpha=port_alpha,
            edgecolors=port_edgecolors,
            linewidths=port_linewidths,
            zorder=port_z,
        )
        artists.append(sc_in)
    if show_ports and exit_xs:
        sc_out = ax.scatter(
            exit_xs, exit_ys,
            s=port_marker_size,
            c=exit_color,
            marker=port_marker,
            alpha=port_alpha,
            edgecolors=port_edgecolors,
            linewidths=port_linewidths,
            zorder=port_z,
        )
        artists.append(sc_out)

    return artists


# ---------------------------------------------------------------------------
# High-level one-call API (mirrors envs.visualizer.plot_layout / save_layout)
# ---------------------------------------------------------------------------

def _build_facility_figure(
    state_dict: Mapping[str, Any],
    resolved: Mapping[str, List[FacilityPlacement]],
    *,
    figsize: Tuple[float, float],
    show_ids: bool,
    show_ports: bool,
    show_cluster_outlines: bool,
) -> Tuple[Any, Any]:
    """Build a world-mm figure with cluster outlines + facility rects."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    grid = state_dict["grid"]
    gw_mm = float(grid["width_cells"]) * float(grid["grid_size_mm"])
    gh_mm = float(grid["height_cells"]) * float(grid["grid_size_mm"])

    if show_cluster_outlines:
        draw_cluster_outlines(ax, state_dict["placements"])

    all_facs = [fp for lst in resolved.values() for fp in lst]
    draw_facility_rects(
        ax, all_facs,
        axes_units="world_mm",
        show_ids=show_ids,
        show_ports=show_ports,
    )

    ax.set_xlim(0, gw_mm)
    ax.set_ylim(0, gh_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    return fig, ax


def plot_facility_layout(
    state_dict: Mapping[str, Any],
    *,
    resolved: Optional[Mapping[str, List[FacilityPlacement]]] = None,
    on_missing: str = "warn",
    figsize: Tuple[float, float] = (12, 10),
    show_ids: bool = True,
    show_ports: bool = True,
    show_cluster_outlines: bool = True,
    show: bool = True,
) -> Optional[Tuple[Any, Any]]:
    """Interactive facility layout viewer (high-level).

    Resolves facilities from ``state_dict`` if ``resolved`` is not provided,
    builds a world-mm figure with cluster outlines + facility rects, and
    optionally calls ``plt.show()``.  Mirrors ``envs.visualizer.plot_layout``.

    Args:
        state_dict: Dict from ``export_group_placement``.
        resolved: Pre-computed ``resolve_facilities`` output; resolved inline
            when ``None``.
        on_missing: Forwarded to ``resolve_facilities`` when resolving inline.
        show: When ``True`` (default) ``plt.show()`` is called before returning.

    Returns:
        ``(fig, ax)`` tuple, or ``None`` if there are no resolvable facilities.
    """
    import matplotlib.pyplot as plt
    from .resolver import resolve_facilities

    if resolved is None:
        resolved = resolve_facilities(state_dict, on_missing=on_missing)
    if not any(resolved.values()):
        return None

    fig, ax = _build_facility_figure(
        state_dict, resolved,
        figsize=figsize,
        show_ids=show_ids,
        show_ports=show_ports,
        show_cluster_outlines=show_cluster_outlines,
    )
    if show:
        plt.show()
    return fig, ax


def save_facility_layout(
    state_dict: Mapping[str, Any],
    *,
    save_path: str,
    resolved: Optional[Mapping[str, List[FacilityPlacement]]] = None,
    on_missing: str = "warn",
    figsize: Tuple[float, float] = (12, 10),
    dpi: int = 150,
    show_ids: bool = True,
    show_ports: bool = True,
    show_cluster_outlines: bool = True,
    skip_if_empty: bool = True,
) -> bool:
    """Render the facility layout and save to ``save_path`` (high-level).

    Mirrors ``envs.visualizer.save_layout``: resolves facilities, builds a
    world-mm figure with cluster outlines + facility rects, writes the image,
    then closes the figure.

    Args:
        state_dict: Dict from ``export_group_placement``.
        save_path: Output file path (``.png`` / any format supported by
            ``Figure.savefig``).
        resolved: Pre-computed ``resolve_facilities`` output; resolved inline
            when ``None``.
        skip_if_empty: When ``True`` (default) the function returns ``False``
            without writing a file if no facilities could be resolved.

    Returns:
        ``True`` if the file was written, ``False`` if skipped because the
        resolved layout was empty.
    """
    import matplotlib.pyplot as plt
    from .resolver import resolve_facilities

    if resolved is None:
        resolved = resolve_facilities(state_dict, on_missing=on_missing)
    if skip_if_empty and not any(resolved.values()):
        return False

    fig, ax = _build_facility_figure(
        state_dict, resolved,
        figsize=figsize,
        show_ids=show_ids,
        show_ports=show_ports,
        show_cluster_outlines=show_cluster_outlines,
    )
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True
