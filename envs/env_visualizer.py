from __future__ import annotations
import torch

from dataclasses import dataclass
from typing import Optional, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RadioButtons
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx

from typing import Any

from envs.action_space import ActionSpace as CandidateSet
from envs.state import EnvState


@dataclass(frozen=True)
class StepFrame:
    """A single step frame for interactive browsing."""

    state: Any  # engine/wrapper state copy
    cost: float
    step_idx: int
    action_space: Optional[CandidateSet] = None
    scores: Optional["np.ndarray"] = None  # float [N] (same length as action_space.xyrot)
    selected_action: Optional[int] = None
    value: Optional[float] = None


def _set_visible_group(group: list[Any], v: bool) -> None:
    for a in group:
        try:
            a.set_visible(bool(v))
        except Exception:
            pass


def _default_layer_visibility() -> dict[str, bool]:
    # Match plot_layout defaults:
    # - zones ON
    # - forbidden masks ON
    # - engine-internal invalid/clearance OFF (toggle when debugging)
    # - flow/score/action_space ON
    # - routes ON (if provided)
    return {
        "forbidden_areas": True,
        "invalid_mask": False,
        "clearance_mask": False,
        "flow": True,
        "score": True,
        "action_space": True,
        "routes": True,
        "constraint_zones": True,
    }


def _apply_layer_visibility(groups: dict[str, list[Any]], vis: dict[str, bool]) -> None:
    for k, g in groups.items():
        if k not in vis:
            continue
        _set_visible_group(g, bool(vis[k]))


def _legend_proxies() -> list[Any]:
    # Keep consistent legend ordering/labels across viewers.
    return [
        patches.Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.15, label="forbidden_areas"),
        patches.Patch(facecolor="#8b0000", edgecolor="#8b0000", alpha=0.10, label="invalid_mask"),
        patches.Patch(facecolor="#ff6b6b", edgecolor="#ff6b6b", alpha=0.10, label="clearance_mask"),
        Line2D([0], [0], color="blue", lw=1.5, alpha=0.3, label="flow"),
        Line2D([0], [0], color="black", lw=0.0, marker="s", markersize=8, label="score"),
        Line2D([0], [0], color="green", lw=0.0, marker="o", markersize=6, alpha=0.65, label="action_space"),
        Line2D([0], [0], color="orange", lw=2.0, alpha=0.8, label="routes"),
        patches.Patch(facecolor="#1e90ff", edgecolor="#1e90ff", alpha=0.10, label="constraint_zones"),
    ]


def _install_click_legend(
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    groups: dict[str, list[Any]],
    vis: dict[str, bool],
    title: str = "Click legend to toggle",
    legend_ax: Optional[plt.Axes] = None,
    connect_once_state: Optional[dict[str, Any]] = None,
    on_toggle: Optional[Callable[[str, bool], None]] = None,
) -> dict[str, Any]:
    """Create a clickable legend that toggles artist visibility.

    If connect_once_state is provided, we reuse its pick handler and only update
    the internal (groups, mapping) references each render (important for browse_steps).
    """
    leg_host = legend_ax if legend_ax is not None else ax
    proxies = _legend_proxies()
    leg = leg_host.legend(handles=proxies, loc="upper left", title=title, framealpha=0.85)

    # Make both legend TEXT and HANDLE clickable. (Users often click the color box/line.)
    legend_artist_to_key: dict[Any, str] = {}
    keys = [str(p.get_label()) for p in proxies]

    # Text entries
    for i, text in enumerate(leg.get_texts()):
        k = str(text.get_text())
        text.set_picker(10)  # larger pick radius than default
        legend_artist_to_key[text] = k
        try:
            text.set_alpha(1.0 if bool(vis.get(k, True)) else 0.35)
        except Exception:
            pass

    # Handle entries (patch/line/etc)
    try:
        handles = list(getattr(leg, "legend_handles", []))
        for i, h in enumerate(handles):
            if i >= len(keys):
                break
            k = keys[i]
            try:
                h.set_picker(10)
            except Exception:
                try:
                    h.set_picker(True)
                except Exception:
                    pass
            legend_artist_to_key[h] = k
    except Exception:
        pass

    state = connect_once_state if connect_once_state is not None else {}
    state["groups"] = groups
    state["legend_artist_to_key"] = legend_artist_to_key
    state["vis"] = vis
    state["on_toggle"] = on_toggle

    if connect_once_state is None:
        def _on_pick(event) -> None:
            artist = getattr(event, "artist", None)
            key = state.get("legend_artist_to_key", {}).get(artist, None)
            if key is None:
                return
            group = state.get("groups", {}).get(key, [])
            # Toggle state even if group is empty, so the legend alpha stays consistent.
            cur = bool(state.get("vis", {}).get(key, True))
            state["vis"][key] = not cur
            _set_visible_group(group, not cur)
            try:
                artist.set_alpha(1.0 if not cur else 0.35)
            except Exception:
                pass
            cb = state.get("on_toggle", None)
            if callable(cb):
                cb(key, not cur)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", _on_pick)

    return state


_CONSTRAINT_BASE_COLORS = [
    "#1e90ff",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#bcbd22",
]


def _constraint_names(engine: Any) -> list[str]:
    constraints = getattr(engine, "zone_constraints", None)
    if not isinstance(constraints, dict):
        return []
    return [str(name) for name in constraints.keys()]


def _constraint_color(name: str, names: list[str]) -> str:
    idx = 0
    try:
        idx = names.index(str(name))
    except ValueError:
        idx = 0
    return _CONSTRAINT_BASE_COLORS[idx % len(_CONSTRAINT_BASE_COLORS)]


def _constraint_cmap(name: str, names: list[str]) -> LinearSegmentedColormap:
    base = _constraint_color(name, names)
    return LinearSegmentedColormap.from_list(
        f"constraint_{str(name)}",
        ["#ffffff", base],
    )


def _resolve_constraint_gid(engine: Any, action_space: Optional[CandidateSet]) -> Optional[Any]:
    if action_space is not None:
        gid = getattr(action_space, "gid", None)
        if gid is None:
            meta = getattr(action_space, "meta", None) or {}
            gid = meta.get("gid", None)
        if gid is not None:
            return gid
    state = engine.get_state()
    remaining = getattr(state, "remaining", [])
    if remaining:
        return remaining[0]
    return None


def _compare_constraint(src: torch.Tensor, v: Any, *, op: str) -> torch.Tensor:
    if op == "<":
        return src < v
    if op == "<=":
        return src <= v
    if op == ">":
        return src > v
    if op == ">=":
        return src >= v
    if op == "==":
        return src == v
    if op == "!=":
        return src != v
    raise ValueError(f"unsupported op: {op!r}")


def _plot_constraint_rects(
    ax: plt.Axes,
    *,
    cfg: dict[str, Any],
    color: str,
    label_values: bool,
) -> list[Any]:
    arts: list[Any] = []
    areas = cfg.get("areas", [])
    if not isinstance(areas, list):
        return arts
    for area in areas:
        if not isinstance(area, dict):
            continue
        rect = area.get("rect", None)
        if rect is None:
            continue
        x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        if w <= 0 or h <= 0:
            continue
        patch = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=0.07,
            linestyle="-",
            zorder=1.0,
        )
        ax.add_patch(patch)
        arts.append(patch)
        if label_values and area.get("value", None) is not None:
            label = ax.text(
                x0 + w / 2.0,
                y0 + h / 2.0,
                str(area["value"]),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, linewidth=0.0),
                zorder=1.1,
            )
            arts.append(label)
    return arts


def _plot_constraint_overlay(
    ax: plt.Axes,
    *,
    engine: Any,
    mode: str,
    target: str,
    constraint_gid: Optional[Any],
) -> tuple[list[Any], list[str], Optional[Any], Optional[str]]:
    mode_norm = str(mode).strip().lower()
    target_norm = str(target)
    names = _constraint_names(engine)
    constraints = getattr(engine, "zone_constraints", None)
    maps = getattr(engine.get_maps(), "constraint_maps", {})

    if not isinstance(constraints, dict) or not names:
        return [], ["constraint: none"], None, None
    if mode_norm == "off":
        return [], ["constraint: off"], None, None

    arts: list[Any] = []
    summary: list[str] = [f"mode: {mode_norm}"]

    if mode_norm == "outline":
        draw_names = names if target_norm == "All" else [target_norm]
        draw_names = [name for name in draw_names if name in constraints]
        if not draw_names:
            return [], ["constraint: outline", "target: none"], None, None
        summary.append(f"target: {'all' if target_norm == 'All' else target_norm}")
        for name in draw_names:
            cfg = constraints.get(name, {})
            color = _constraint_color(name, names)
            arts.extend(
                _plot_constraint_rects(
                    ax,
                    cfg=cfg,
                    color=color,
                    label_values=(target_norm != "All"),
                )
            )
            default_v = cfg.get("default", None)
            op = str(cfg.get("op", ""))
            area_count = len(cfg.get("areas", [])) if isinstance(cfg.get("areas", []), list) else 0
            summary.append(f"{name}: {op} {default_v} | areas={area_count}")
        return arts, summary, None, None

    if target_norm == "All":
        return [], [f"mode: {mode_norm}", "target: select a specific constraint"], None, None
    if target_norm not in constraints or target_norm not in maps:
        return [], [f"mode: {mode_norm}", f"target: {target_norm}", "constraint map missing"], None, None

    cfg = constraints[target_norm]
    color = _constraint_color(target_norm, names)
    cmap_tensor = maps[target_norm]
    default_v = cfg.get("default", None)
    dtype = str(cfg.get("dtype", ""))
    op = str(cfg.get("op", ""))
    summary.extend([
        f"constraint: {target_norm}",
        f"dtype: {dtype}",
        f"rule: cell {op} group_value",
        f"default: {default_v}",
    ])

    if mode_norm == "value":
        arr = cmap_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        im = ax.imshow(
            arr,
            origin="lower",
            extent=(0, engine.grid_width, 0, engine.grid_height),
            cmap=_constraint_cmap(target_norm, names),
            vmin=vmin,
            vmax=vmax,
            alpha=0.28,
            interpolation="nearest",
            zorder=0.2,
        )
        arts.append(im)
        arts.extend(_plot_constraint_rects(ax, cfg=cfg, color=color, label_values=True))
        summary.append(f"range: [{float(np.nanmin(arr)):.3f}, {float(np.nanmax(arr)):.3f}]")
        return arts, summary, im, f"{target_norm} value"

    if mode_norm == "check":
        if constraint_gid is None:
            return [], summary + ["gid: unavailable"], None, None
        spec = engine.group_specs.get(constraint_gid, None)
        zone_values = dict(getattr(spec, "zone_values", {}) or {}) if spec is not None else {}
        if target_norm not in zone_values:
            return [], summary + [f"gid: {constraint_gid}", "group value: missing"], None, None
        group_value = zone_values[target_norm]
        pass_mask = _compare_constraint(cmap_tensor, group_value, op=op)
        pass_np = pass_mask.detach().cpu().numpy().astype(bool, copy=False)
        rgba = np.zeros((pass_np.shape[0], pass_np.shape[1], 4), dtype=np.float32)
        rgba[..., 0] = np.where(pass_np, 0.0, 0.839)
        rgba[..., 1] = np.where(pass_np, 0.627, 0.153)
        rgba[..., 2] = np.where(pass_np, 0.172, 0.157)
        rgba[..., 3] = np.where(pass_np, 0.18, 0.22)
        im = ax.imshow(
            rgba,
            origin="lower",
            extent=(0, engine.grid_width, 0, engine.grid_height),
            interpolation="nearest",
            zorder=0.2,
        )
        arts.append(im)
        arts.extend(_plot_constraint_rects(ax, cfg=cfg, color=color, label_values=False))
        pass_ratio = float(pass_np.mean()) if pass_np.size > 0 else 0.0
        summary.extend([
            f"gid: {constraint_gid}",
            f"group value: {group_value}",
            f"pass cells: {pass_ratio * 100.0:.1f}%",
            "overlay: green=pass, red=fail",
        ])
        return arts, summary, None, None

    return [], [f"constraint mode unsupported: {mode_norm}"], None, None


def _draw_layout_layers(
    *,
    ax: plt.Axes,
    engine: Any,
    action_space: Optional[CandidateSet] = None,
    routes: Optional[list[Any]] = None,
    constraint_mode: str = "outline",
    constraint_target: str = "All",
    constraint_gid: Optional[Any] = None,
    constraint_meta_out: Optional[dict[str, Any]] = None,
) -> dict[str, list[Any]]:
    """Draw the same base layers used by plot_layout (zones/masks/layout/flow/score/action_space/routes)."""
    zone_artists: dict[str, list[Any]] = {"constraint": [], "forbidden": []}
    misc_artists: dict[str, list[Any]] = {
        "invalid_mask": [],
        "clearance_mask": [],
        "flow": [],
        "score": [],
        "action_space": [],
        "routes": [],
    }

    def _add_zone_rect(
        *,
        rect: list[int] | tuple[int, int, int, int],
        kind: str,
        edgecolor: str,
        facecolor: str,
        alpha: float,
        linestyle: str = "-",
        label: Optional[str] = None,
    ) -> None:
        x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        if w <= 0 or h <= 0:
            return
        p = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
        p.set_visible(True)
        ax.add_patch(p)
        zone_artists[kind].append(p)
        if label:
            t = ax.text(
                x0 + w / 2.0,
                y0 + h / 2.0,
                str(label),
                ha="center",
                va="center",
                fontsize=8,
                color=edgecolor,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
            )
            t.set_visible(True)
            zone_artists[kind].append(t)

    constraint_art, constraint_summary, constraint_mappable, constraint_cbar_label = _plot_constraint_overlay(
        ax,
        engine=engine,
        mode=constraint_mode,
        target=constraint_target,
        constraint_gid=constraint_gid,
    )
    zone_artists["constraint"].extend(constraint_art)
    if constraint_meta_out is not None:
        constraint_meta_out.clear()
        constraint_meta_out["summary_lines"] = list(constraint_summary)
        constraint_meta_out["mappable"] = constraint_mappable
        constraint_meta_out["colorbar_label"] = constraint_cbar_label

    # forbidden_areas
    if hasattr(engine, "forbidden_areas") and isinstance(getattr(engine, "forbidden_areas"), list):
        for a in getattr(engine, "forbidden_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            if rect is None:
                continue
            _add_zone_rect(
                rect=rect,
                kind="forbidden",
                edgecolor="#d62728",
                facecolor="#d62728",
                alpha=0.15,
                linestyle="-",
                label=None,
            )

    # engine internal masks (start hidden)
    maps = getattr(engine, "_maps", None)
    inv = maps.invalid if maps is not None else getattr(engine, "_invalid", None)
    if inv is not None:
        mi = _plot_mask(ax, inv, color="#8b0000", alpha=1)
        if mi is not None:
            mi.set_visible(False)
            misc_artists["invalid_mask"].append(mi)

    # clearance-only ring
    clr = maps.clear_invalid if maps is not None else getattr(engine, "_clear_invalid", None)
    occ = maps.occ_invalid if maps is not None else getattr(engine, "_occ_invalid", None)
    clr_vis = (clr & (~occ)) if (clr is not None and occ is not None) else clr
    if clr_vis is not None:
        mc2 = _plot_mask(ax, clr_vis, color="#ff6b6b", alpha=1)
        if mc2 is not None:
            mc2.set_visible(False)
            misc_artists["clearance_mask"].append(mc2)

    # placed rects/labels
    for gid in engine.get_state().placed:
        p = engine.get_state().placements[gid]
        rect = patches.Rectangle(
            (float(p.x_bl), float(p.y_bl)),
            float(p.w),
            float(p.h),
            linewidth=1.2,
            edgecolor="black",
            facecolor="orange",
            alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(p.cx, p.cy, str(gid), ha="center", va="center", fontsize=8)

    # action_space (optional; caller may render their own)
    if action_space is not None:
        meta = action_space.meta or {}
        gid = getattr(action_space, "gid", None)
        if gid is None:
            gid = meta.get("gid", None)
        xyrot = action_space.xyrot[action_space.mask]
        if int(xyrot.shape[0]) > 0:
            xs: list[float] = []
            ys: list[float] = []
            if gid is None:
                raise ValueError("CandidateSet must include `gid` (or meta['gid']) to convert BL->center for plotting.")
            for x_bl, y_bl, rot in xyrot.detach().cpu().tolist():
                cx, cy = _center_from_bl(engine, gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))
                xs.append(float(cx))
                ys.append(float(cy))
            sc = ax.scatter(xs, ys, s=18, c="green", alpha=0.65, linewidths=0.0)
            sc.set_visible(True)
            misc_artists["action_space"].append(sc)

    # flow overlay
    flow_art = _plot_flow_overlay(ax, engine)
    for a in flow_art:
        try:
            a.set_visible(True)
        except Exception:
            pass
    misc_artists["flow"].extend(flow_art)

    # score overlay
    score = engine.cost()
    score_text = ax.text(
        0.01,
        0.99,
        f"cost={score:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )
    score_text.set_visible(True)
    misc_artists["score"].append(score_text)

    # routes overlay (from postprocess.pathfinder)
    if routes is not None:
        route_arts = _plot_routes_overlay(ax, routes)
        for a in route_arts:
            try:
                a.set_visible(True)
            except Exception:
                pass
        misc_artists["routes"].extend(route_arts)

    return {
        "forbidden_areas": zone_artists["forbidden"],
        "invalid_mask": misc_artists["invalid_mask"],
        "clearance_mask": misc_artists["clearance_mask"],
        "flow": misc_artists["flow"],
        "score": misc_artists["score"],
        "action_space": misc_artists["action_space"],
        "routes": misc_artists["routes"],
        "constraint_zones": zone_artists["constraint"],
    }


def plot_layout(env: Any, *, action_space: Optional[CandidateSet] = None, routes: Optional[list[Any]] = None) -> None:
    """Interactive viewer (dynamic toggles only).

    - No save_path/show_* args here on purpose: use `save_layout(...)` for saving.
    - This function always opens a window and lets you toggle layers via the external legend.
    
    Args:
        env: FactoryLayoutEnv or wrapper
        action_space: Optional candidate set to display
        routes: Optional list of RouteResult from postprocess.pathfinder
    """
    # Support both engine (`FactoryLayoutEnv`) and wrapper envs by unwrapping.
    engine = getattr(env, "engine", env)

    # Keep legend / constraint controls outside the layout area.
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[20, 1, 7], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    cax_constraint = fig.add_subplot(gs[0, 1])
    cax_constraint.axis("off")
    right_gs = gs[0, 2].subgridspec(nrows=4, ncols=1, height_ratios=[5, 3, 5, 4], hspace=0.35)
    ax_leg = fig.add_subplot(right_gs[0, 0])
    ax_mode = fig.add_subplot(right_gs[1, 0])
    ax_target = fig.add_subplot(right_gs[2, 0])
    ax_constraint_info = fig.add_subplot(right_gs[3, 0])
    ax_leg.axis("off")
    ax_constraint_info.axis("off")
    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    layer_vis = _default_layer_visibility()
    legend_state: dict[str, Any] | None = None
    constraint_cbar = None
    constraint_meta: dict[str, Any] = {}
    constraint_names = _constraint_names(engine)
    mode_labels = ["Off", "Outline", "Value", "Check"]
    target_labels = ["All"] + constraint_names if constraint_names else ["All"]
    constraint_state = {
        "mode": "outline",
        "target": "All",
    }
    mode_radio = RadioButtons(ax_mode, mode_labels, active=1)
    target_radio = RadioButtons(ax_target, target_labels, active=0)
    ax_mode.set_title("Constraint", fontsize=10)
    ax_target.set_title("Target", fontsize=10)
    info_text = ax_constraint_info.text(
        0.01,
        0.99,
        "",
        transform=ax_constraint_info.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    def _render() -> None:
        nonlocal legend_state, constraint_cbar
        ax.clear()
        ax_leg.clear()
        ax_leg.axis("off")
        ax.set_xlim(0, engine.grid_width)
        ax.set_ylim(0, engine.grid_height)
        ax.set_aspect("equal")
        ax.set_title("FactoryLayoutEnv")

        constraint_gid = _resolve_constraint_gid(engine, action_space)
        groups = _draw_layout_layers(
            ax=ax,
            engine=engine,
            action_space=action_space,
            routes=routes,
            constraint_mode=constraint_state["mode"],
            constraint_target=constraint_state["target"],
            constraint_gid=constraint_gid,
            constraint_meta_out=constraint_meta,
        )
        _apply_layer_visibility(groups, layer_vis)
        legend_state = _install_click_legend(
            fig=fig,
            ax=ax,
            groups=groups,
            vis=layer_vis,
            legend_ax=ax_leg,
            connect_once_state=legend_state,
            on_toggle=lambda _key, _visible: _render(),
        )

        mappable = constraint_meta.get("mappable", None)
        if bool(layer_vis.get("constraint_zones", True)) and mappable is not None:
            cax_constraint.axis("on")
            if constraint_cbar is None:
                constraint_cbar = fig.colorbar(mappable, cax=cax_constraint)
            else:
                constraint_cbar.update_normal(mappable)
            label = constraint_meta.get("colorbar_label", None)
            if label:
                constraint_cbar.set_label(str(label))
        else:
            if constraint_cbar is not None:
                constraint_cbar.remove()
                constraint_cbar = None
            cax_constraint.clear()
            cax_constraint.axis("off")

        info_text.set_text("\n".join(constraint_meta.get("summary_lines", [])))
        fig.canvas.draw_idle()

    def _on_mode_clicked(label: str) -> None:
        constraint_state["mode"] = str(label).strip().lower()
        _render()

    def _on_target_clicked(label: str) -> None:
        constraint_state["target"] = str(label)
        _render()

    mode_radio.on_clicked(_on_mode_clicked)
    target_radio.on_clicked(_on_target_clicked)
    _render()
    plt.show()


def browse_steps(
    env: Any,
    *,
    frames: list[StepFrame],
    title: str = "Inference browser (←/→ to navigate, q to quit)",
    cmap: str = "RdYlGn",
    point_size: float = 18.0,
    selected_point_size: float = 90.0,
    selected_edge_width: float = 2.5,
    max_points: Optional[int] = None,
) -> None:
    """Browse step-by-step layouts with optional candidate-policy overlays.

    - If frame has action_space/scores/selected_action, draw policy scatter.
    - If not, browse layout states only.
    - Restores frame state via engine/wrapper set_state.
    """
    if not frames:
        raise ValueError("browse_steps: frames is empty")

    # Support wrapper envs by unwrapping engine.
    wrapper = env
    engine = getattr(env, "engine", env)
    if not hasattr(wrapper, "set_state"):
        raise ValueError("browse_steps requires a wrapper env with set_state(get_state_copy()) support.")

    # Fixed layout so axes don't shrink when navigating:
    # - main axis: layout + scatter
    # - side axes: policy/constraint colorbars
    # - far-right column: legend + constraint controls
    # - bottom axis: info panel (outside plot area)
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[12, 2],
        width_ratios=[20, 1, 1, 7],
        wspace=0.15,
        hspace=0.15,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax_policy = fig.add_subplot(gs[0, 1])
    cax_constraint = fig.add_subplot(gs[0, 2])
    right_gs = gs[0, 3].subgridspec(nrows=4, ncols=1, height_ratios=[5, 3, 5, 4], hspace=0.35)
    ax_leg = fig.add_subplot(right_gs[0, 0])
    ax_mode = fig.add_subplot(right_gs[1, 0])
    ax_target = fig.add_subplot(right_gs[2, 0])
    ax_constraint_info = fig.add_subplot(right_gs[3, 0])
    ax_leg.axis("off")
    ax_constraint_info.axis("off")
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title(title)

    # persistent artists (updated in-place)
    sc = None
    sc_sel = None
    policy_cbar = None
    constraint_cbar = None
    info_text = ax_info.text(
        0.01,
        0.5,
        "",
        transform=ax_info.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        family="monospace",
    )

    cur = {"idx": 0}
    layer_vis = _default_layer_visibility()
    legend_state: dict[str, Any] | None = None
    constraint_meta: dict[str, Any] = {}
    constraint_names = _constraint_names(engine)
    mode_labels = ["Off", "Outline", "Value", "Check"]
    target_labels = ["All"] + constraint_names if constraint_names else ["All"]
    constraint_state = {
        "mode": "outline",
        "target": "All",
    }
    mode_radio = RadioButtons(ax_mode, mode_labels, active=1)
    target_radio = RadioButtons(ax_target, target_labels, active=0)
    ax_mode.set_title("Constraint", fontsize=10)
    ax_target.set_title("Target", fontsize=10)
    constraint_info_text = ax_constraint_info.text(
        0.01,
        0.99,
        "",
        transform=ax_constraint_info.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    def _frame_to_xy(frame: StepFrame) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]]:
        cand = frame.action_space
        if cand is None or frame.scores is None or frame.selected_action is None:
            return None
        gid = cand.gid
        if gid is None:
            raise ValueError("CandidateSet.gid is required for BL->center conversion.")
        mask = cand.mask.detach().cpu().numpy().astype(bool)
        xyrot = cand.xyrot.detach().cpu().numpy()
        scores = np.asarray(frame.scores, dtype=np.float32)
        if scores.shape[0] != xyrot.shape[0]:
            raise ValueError(f"scores length {scores.shape[0]} != action_space {xyrot.shape[0]}")

        idxs = np.where(mask)[0]
        if max_points is not None and idxs.shape[0] > int(max_points):
            idxs = idxs[: int(max_points)]

        xs: list[float] = []
        ys: list[float] = []
        cs: list[float] = []
        for k in idxs.tolist():
            x_bl, y_bl, rot = xyrot[k].tolist()
            cx, cy = _center_from_bl(engine, gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))
            xs.append(float(cx))
            ys.append(float(cy))
            cs.append(float(scores[k]))

        # selected action point (even if invalid, we still try to draw it)
        sel = int(frame.selected_action)
        if sel < 0 or sel >= xyrot.shape[0]:
            sel_xy = (0.0, 0.0)
        else:
            x_bl, y_bl, rot = xyrot[sel].tolist()
            cx, cy = _center_from_bl(engine, gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))
            sel_xy = (float(cx), float(cy))

        return np.asarray(xs), np.asarray(ys), np.asarray(cs, dtype=np.float32), sel_xy

    def _render(idx: int) -> None:
        nonlocal sc, sc_sel, policy_cbar, constraint_cbar
        nonlocal legend_state
        idx = int(max(0, min(len(frames) - 1, idx)))
        cur["idx"] = idx
        f = frames[idx]

        # restore state for this frame
        frame_state = f.state
        if isinstance(frame_state, dict) and ("engine" in frame_state) and ("adapter" in frame_state):
            eng_state = frame_state.get("engine", None)
            adp_state = frame_state.get("adapter", None)
            if hasattr(engine, "set_state") and isinstance(eng_state, EnvState):
                engine.set_state(eng_state)  # type: ignore[attr-defined]
            if adp_state is not None:
                wrapper.set_state(adp_state)  # type: ignore[attr-defined]
        elif isinstance(frame_state, EnvState) and hasattr(engine, "set_state"):
            engine.set_state(frame_state)  # type: ignore[attr-defined]
        else:
            wrapper.set_state(frame_state)  # type: ignore[attr-defined]

        ax.clear()
        ax.set_xlim(0, engine.grid_width)
        ax.set_ylim(0, engine.grid_height)
        ax.set_aspect("equal")
        ax.set_title(title + f"  [step {f.step_idx}/{frames[-1].step_idx}]")

        ax_leg.clear()
        ax_leg.axis("off")

        # Draw the same base UI layers as plot_layout (zones/masks/layout/flow/score),
        # then overlay policy-colored action_space on top.
        constraint_gid = _resolve_constraint_gid(engine, f.action_space)
        groups = _draw_layout_layers(
            ax=ax,
            engine=engine,
            action_space=None,
            constraint_mode=constraint_state["mode"],
            constraint_target=constraint_state["target"],
            constraint_gid=constraint_gid,
            constraint_meta_out=constraint_meta,
        )
        xy_data = _frame_to_xy(f)
        if xy_data is not None:
            xs, ys, cs, (sx, sy) = xy_data
            sc = ax.scatter(xs, ys, s=float(point_size), c=cs, cmap=cmap, alpha=0.85, linewidths=0.0)
            sc_sel = ax.scatter(
                [sx],
                [sy],
                s=float(selected_point_size),
                facecolors="none",
                edgecolors="#1f77b4",  # blue
                linewidths=float(selected_edge_width),
            )
            groups["action_space"].extend([sc, sc_sel])
        _apply_layer_visibility(groups, layer_vis)
        legend_state = _install_click_legend(
            fig=fig,
            ax=ax,
            groups=groups,
            vis=layer_vis,
            legend_ax=ax_leg,
            connect_once_state=legend_state,
            on_toggle=lambda _key, _visible: _render(cur["idx"]),
        )

        # colorbar (fixed axis; no layout shrink). Create once, then update.
        if xy_data is not None and sc is not None:
            cax_policy.axis("on")
            if policy_cbar is None:
                policy_cbar = fig.colorbar(sc, cax=cax_policy)
                policy_cbar.set_label("policy score / prob")
            else:
                policy_cbar.update_normal(sc)
        else:
            if policy_cbar is not None:
                policy_cbar.remove()
                policy_cbar = None
            cax_policy.clear()
            cax_policy.axis("off")

        constraint_mappable = constraint_meta.get("mappable", None)
        if bool(layer_vis.get("constraint_zones", True)) and constraint_mappable is not None:
            cax_constraint.axis("on")
            if constraint_cbar is None:
                constraint_cbar = fig.colorbar(constraint_mappable, cax=cax_constraint)
            else:
                constraint_cbar.update_normal(constraint_mappable)
            label = constraint_meta.get("colorbar_label", None)
            if label:
                constraint_cbar.set_label(str(label))
        else:
            if constraint_cbar is not None:
                constraint_cbar.remove()
                constraint_cbar = None
            cax_constraint.clear()
            cax_constraint.axis("off")

        constraint_info_text.set_text("\n".join(constraint_meta.get("summary_lines", [])))

        # info panel outside the plot
        parts = [f"step={f.step_idx}", f"cost={f.cost:.3f}"]
        if f.value is not None:
            parts.append(f"value={float(f.value):.6f}")
        if f.action_space is not None and f.scores is not None and f.selected_action is not None:
            sel = int(f.selected_action)
            scores = np.asarray(f.scores, dtype=np.float32)
            sel_score = float(scores[sel]) if 0 <= sel < len(scores) else float("nan")
            parts.append(f"selected_action={sel}")
            parts.append(f"selected_policy={sel_score:.6f}")
            parts.append(f"valid={int(f.action_space.mask.to(torch.int64).sum().item())}")
        info_text.set_text(" | ".join(parts))

        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key in ("right", "d", "space"):
            _render(cur["idx"] + 1)
            return
        if event.key in ("left", "a"):
            _render(cur["idx"] - 1)
            return

    def _on_mode_clicked(label: str) -> None:
        constraint_state["mode"] = str(label).strip().lower()
        _render(cur["idx"])

    def _on_target_clicked(label: str) -> None:
        constraint_state["target"] = str(label)
        _render(cur["idx"])

    mode_radio.on_clicked(_on_mode_clicked)
    target_radio.on_clicked(_on_target_clicked)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _render(0)
    plt.show()
    plt.close(fig)


def save_layout(
    env: Any,
    *,
    show_masks: bool = True,
    show_flow: bool = False,
    show_score: bool = False,
    show_zones: bool = False,
    action_space: Optional[CandidateSet] = None,
    save_path: str,
) -> None:
    """Save a static layout image (no interactive toggles)."""
    engine = getattr(env, "engine", env)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title("FactoryLayoutEnv")

    if show_zones:
        constraints = getattr(engine, "zone_constraints", None)
        if isinstance(constraints, dict):
            for cname, cfg in constraints.items():
                if not isinstance(cfg, dict):
                    continue
                areas = cfg.get("areas", [])
                op = str(cfg.get("op", ""))
                if not isinstance(areas, list):
                    continue
                for a in areas:
                    if not isinstance(a, dict):
                        continue
                    rect = a.get("rect", None)
                    value = a.get("value", None)
                    if rect is None:
                        continue
                    x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                    ax.add_patch(
                        patches.Rectangle(
                            (x0, y0),
                            max(0, x1 - x0),
                            max(0, y1 - y0),
                            linewidth=1.2,
                            edgecolor="#1e90ff",
                            facecolor="#1e90ff",
                            alpha=0.10,
                            linestyle="-",
                        )
                    )
                    if value is not None:
                        ax.text(
                            x0 + (x1 - x0) / 2.0,
                            y0 + (y1 - y0) / 2.0,
                            f"{cname}{op}{value}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#1e90ff",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                        )

    if show_masks:
        # Draw forbidden_areas as rects
        if hasattr(engine, "forbidden_areas") and isinstance(engine.forbidden_areas, list):
            for a in engine.forbidden_areas:
                if not isinstance(a, dict) or "rect" not in a:
                    continue
                rect = a["rect"]
                x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                w, h = max(0, x1 - x0), max(0, y1 - y0)
                if w > 0 and h > 0:
                    p = patches.Rectangle((x0, y0), w, h, linewidth=1.2,
                                          edgecolor="#d62728", facecolor="#d62728", alpha=0.15)
                    ax.add_patch(p)

    for gid in engine.get_state().placed:
        p = engine.get_state().placements[gid]
        rect = patches.Rectangle(
            (float(p.x_bl), float(p.y_bl)),
            float(p.w),
            float(p.h),
            linewidth=1.2,
            edgecolor="black",
            facecolor="orange",
            alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(p.cx, p.cy, str(gid), ha="center", va="center", fontsize=8)

    if action_space is not None:
        meta = action_space.meta or {}
        gid = getattr(action_space, "gid", None)
        if gid is None:
            gid = meta.get("gid", None)
        xyrot = action_space.xyrot[action_space.mask]
        if int(xyrot.shape[0]) > 0:
            xs: list[float] = []
            ys: list[float] = []
            if gid is None:
                raise ValueError("CandidateSet must include `gid` (or meta['gid']) to convert BL->center for plotting.")
            for x_bl, y_bl, rot in xyrot.detach().cpu().tolist():
                cx, cy = _center_from_bl(engine, gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))
                xs.append(float(cx))
                ys.append(float(cy))
            ax.scatter(xs, ys, s=18, c="green", alpha=0.65, linewidths=0.0)

    if show_flow:
        _plot_flow_overlay(ax, engine)

    if show_score:
        score = engine.cost()
        ax.text(
            0.01,
            0.99,
            f"cost={score:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_flow_graph(env, *, show_weights: bool = True) -> None:
    """Plot directed flow graph from env.group_flow."""
    engine = getattr(env, "engine", env)
    G = nx.DiGraph()
    for src, targets in engine.group_flow.items():
        for dst, weight in targets.items():
            G.add_edge(src, dst, weight=weight)

    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.5,
        connectionstyle="arc3,rad=0.08",
    )
    if show_weights:
        edge_labels = {(u, v): f"{d.get('weight', 1.0):.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Flow Graph")
    plt.tight_layout()
    plt.show()


def _plot_flow_overlay(ax: plt.Axes, env) -> list[Any]:
    if not env.get_state().placed:
        return []
    arts: list[Any] = []
    # Use reward-computed argmin port pairs if available (env.get_state().flow.flow_port_pairs).
    port_pairs = env.get_state().flow.flow_port_pairs
    for src, targets in env.group_flow.items():
        if src not in env.get_state().placed:
            continue
        src_p = env.get_state().placements[src]
        for dst, weight in targets.items():
            if dst not in env.get_state().placed:
                continue
            dst_p = env.get_state().placements[dst]
            cached = port_pairs.get((src, dst))
            if cached is not None:
                (sx, sy), (dx, dy) = cached
            else:
                sx, sy = src_p.cx, src_p.cy
                dx, dy = dst_p.cx, dst_p.cy
            ann = ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=0.8, alpha=0.3),
            )
            arts.append(ann)
    arts.extend(_plot_port_overlay(ax, env, port_pairs=port_pairs))
    return arts


def _coord_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 6), round(float(y), 6))


def _plot_port_overlay(
    ax: plt.Axes,
    env,
    *,
    port_pairs: Optional[dict[Any, tuple[tuple[float, float], tuple[float, float]]]] = None,
) -> list[Any]:
    """Draw all placed ports as fixed-size circles.

    - entries: green circles
    - exits: red circles
    - active ports keep the same size and use higher alpha only
    """
    if not env.get_state().placed:
        return []

    pairs = port_pairs if port_pairs is not None else env.get_state().flow.flow_port_pairs
    active_entries: set[tuple[float, float]] = set()
    active_exits: set[tuple[float, float]] = set()
    for cached in pairs.values():
        if not isinstance(cached, tuple) or len(cached) != 2:
            continue
        exit_xy, entry_xy = cached
        active_exits.add(_coord_key(exit_xy[0], exit_xy[1]))
        active_entries.add(_coord_key(entry_xy[0], entry_xy[1]))

    inactive_entry_xs: list[float] = []
    inactive_entry_ys: list[float] = []
    active_entry_xs: list[float] = []
    active_entry_ys: list[float] = []
    inactive_exit_xs: list[float] = []
    inactive_exit_ys: list[float] = []
    active_exit_xs: list[float] = []
    active_exit_ys: list[float] = []

    for gid in env.get_state().placed:
        p = env.get_state().placements[gid]
        for x, y in getattr(p, "entries", []):
            if _coord_key(x, y) in active_entries:
                active_entry_xs.append(float(x))
                active_entry_ys.append(float(y))
            else:
                inactive_entry_xs.append(float(x))
                inactive_entry_ys.append(float(y))
        for x, y in getattr(p, "exits", []):
            if _coord_key(x, y) in active_exits:
                active_exit_xs.append(float(x))
                active_exit_ys.append(float(y))
            else:
                inactive_exit_xs.append(float(x))
                inactive_exit_ys.append(float(y))

    arts: list[Any] = []

    def _scatter(xs: list[float], ys: list[float], *, color: str, alpha: float) -> None:
        if not xs:
            return
        sc = ax.scatter(
            xs,
            ys,
            s=28,
            c=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.8,
            marker="o",
            zorder=4.0,
        )
        arts.append(sc)

    _scatter(inactive_entry_xs, inactive_entry_ys, color="#2ca02c", alpha=0.40)
    _scatter(active_entry_xs, active_entry_ys, color="#2ca02c", alpha=0.90)
    _scatter(inactive_exit_xs, inactive_exit_ys, color="#d62728", alpha=0.40)
    _scatter(active_exit_xs, active_exit_ys, color="#d62728", alpha=0.90)
    return arts


def _plot_routes_overlay(ax: plt.Axes, routes: list[Any]) -> list[Any]:
    """Draw routes from postprocess.pathfinder.RouteResult list.
    
    Args:
        ax: matplotlib Axes
        routes: List of RouteResult objects (from RoutePlanner.plan_all_routes())
    
    Returns:
        List of artist objects for layer toggling
    """
    arts: list[Any] = []
    
    # Color palette for different routes
    colors = ["#FF6B00", "#00CC66", "#9933FF", "#FF3366", "#00BFFF", "#FFD700", "#FF69B4", "#32CD32"]
    
    for i, route in enumerate(routes):
        if not route.success or route.path is None:
            continue
        
        path = route.path
        if len(path) < 2:
            continue
        
        color = colors[i % len(colors)]
        
        # Draw path as polyline
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        line, = ax.plot(xs, ys, color=color, lw=2.0, alpha=0.8, zorder=10)
        arts.append(line)
        
        # Draw start marker (exit point)
        start_marker = ax.scatter([xs[0]], [ys[0]], s=50, c=color, marker="o", edgecolors="white", linewidths=1, zorder=11)
        arts.append(start_marker)
        
        # Draw end marker (entry point) with arrow
        end_marker = ax.scatter([xs[-1]], [ys[-1]], s=80, c=color, marker=">", edgecolors="white", linewidths=1, zorder=11)
        arts.append(end_marker)
        
        # Label at midpoint
        mid_idx = len(path) // 2
        mid_x, mid_y = path[mid_idx]
        label = ax.text(
            mid_x, mid_y,
            f"{route.src_group}→{route.dst_group}",
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, linewidth=0),
            zorder=12,
        )
        arts.append(label)
    
    return arts


def _center_from_bl(engine: Any, *, gid: Any, x_bl: int, y_bl: int, rot: int) -> tuple[float, float]:
    """Compute (cx, cy) from bottom-left coord using group_specs._rotated_size."""
    spec = engine.group_specs[gid]
    w, h = spec._rotated_size(rot)
    return (float(x_bl) + w / 2.0, float(y_bl) + h / 2.0)



def _plot_mask(ax: plt.Axes, mask: Optional[object], *, color: str, alpha: float):
    if mask is None:
        return None
    # mask is torch.BoolTensor[H,W] where True means forbidden/invalid
    grid = np.asarray(mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask).astype(np.float32)
    if grid.ndim != 2 or grid.size == 0:
        return
    masked = np.ma.masked_where(grid == 0, grid)
    h, w = int(grid.shape[0]), int(grid.shape[1])
    ax.imshow(
        masked,
        origin="lower",
        extent=[0, w, 0, h],
        cmap=_single_color_cmap(color),
        alpha=alpha,
        interpolation="nearest",
    )
    return ax.images[-1] if ax.images else None


def _single_color_cmap(color: str):
    return plt.matplotlib.colors.ListedColormap([color])


def _as_mask_2d(mask_2d: object) -> np.ndarray:
    if isinstance(mask_2d, np.ndarray):
        m = mask_2d
    else:
        # torch.Tensor or other array-like
        m = np.asarray(mask_2d.detach().cpu().numpy() if hasattr(mask_2d, "detach") else mask_2d)
    if m.ndim != 2:
        raise ValueError(f"mask_2d must be 2D, got {m.ndim}D")
    return m.astype(bool)


if __name__ == "__main__":
    # Demo (updated):
    # - Uses generic zones.constraints + groups.*.zone_values
    # - Shows zone overlays and interactive toggles (show=True only).
    import torch

    from decision_adapters.alphachip import AlphaChipDecisionAdapter
    from decision_adapters.greedy import GreedyDecisionAdapter
    from envs.env import FactoryLayoutEnv
    from envs.placement.static import StaticSpec

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- groups (footprint width/height + zone_values) ---
    groups = {
        "A": StaticSpec(
            device=dev, id="A", width=20, height=10,
            entries_rel=[(10.0, 5.0)], exits_rel=[(10.0, 5.0)],
            clearance_left_rel=0, clearance_right_rel=0, clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, zone_values={"weight": 3.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "B": StaticSpec(
            device=dev, id="B", width=16, height=16,
            entries_rel=[(8.0, 8.0)], exits_rel=[(8.0, 8.0)],
            clearance_left_rel=0, clearance_right_rel=0, clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, zone_values={"weight": 4.0, "height": 2.0, "dry": 0.0, "placeable": 1},
        ),
        "C": StaticSpec(
            device=dev, id="C", width=18, height=12,
            entries_rel=[(9.0, 6.0)], exits_rel=[(9.0, 6.0)],
            clearance_left_rel=0, clearance_right_rel=0, clearance_bottom_rel=0, clearance_top_rel=0,
            rotatable=True, zone_values={"weight": 12.0, "height": 10.0, "dry": 2.0, "placeable": 1},
        ),
    }
    flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # --- forbidden areas ---
    forbidden_areas = [{"rect": [0, 0, 30, 20]}]

    # --- zones / constraints ---
    zone_constraints = {
        "weight": {"dtype": "float", "op": ">=", "default": 10.0, "areas": [{"rect": [60, 0, 120, 80], "value": 20.0}]},
        "height": {"dtype": "float", "op": ">=", "default": 20.0, "areas": [{"rect": [0, 60, 120, 80], "value": 5.0}]},
        "dry": {"dtype": "float", "op": "<=", "default": 0.0, "areas": [{"rect": [0, 40, 60, 80], "value": 2.0}]},
        "placeable": {"dtype": "int", "op": "==", "default": 0, "areas": [{"rect": [30, 20, 120, 80], "value": 1}]},
    }

    # Pre-place multiple groups to visualize non-empty layouts.
    # NOTE: reset() validates feasibility and raises ValueError if invalid.
    # NOTE: (x,y) are bottom-left (integer) coordinates of rotated AABB.
    initial_positions = {
        "A": (80, 15, 0),  # was center (90,20) for A(w=20,h=10)
        "B": (82, 32, 0),  # was center (90,40) for B(w=16,h=16)
    }
    # Force next gid to be "C" so constraint-driven mask is visible immediately after reset.
    remaining_order = ["C", "A", "B"]

    engine = FactoryLayoutEnv(
        grid_width=120,
        grid_height=80,
        group_specs=groups,
        group_flow=flow,
        forbidden_areas=forbidden_areas,
        zone_constraints=zone_constraints,
        device=dev,
        max_steps=10,
        log=False,
    )

    # ---- 1) Coarse wrapper demo ----
    # For coarse, we just visualize the current layout (candidate visualization via decode is omitted in demo).
    env1 = AlphaChipDecisionAdapter(engine=engine, coarse_grid=32, rot=0)
    _obs1, _ = env1.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    plot_layout(env1, action_space=None)
    plot_flow_graph(env1)

    # ---- 2) Greedy(TopK) wrapper demo ----
    env2 = GreedyDecisionAdapter(
        engine=engine,
        k=70,
        scan_step=5.0,
        quant_step=5.0,
        p_high=0.2,
        p_near=0.8,
        p_coarse=0.0,
        oversample_factor=2,
        random_seed=7,
    )
    _obs2, _ = env2.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    # Greedy(TopK) wrapper already builds action_space; use obs-provided (decoded) action_space for plotting.
    topk_obs, _ = env2.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    cand = None
    if isinstance(topk_obs, dict) and ("action_mask" in topk_obs) and ("action_xyrot" in topk_obs):
        cand = CandidateSet(xyrot=topk_obs["action_xyrot"], mask=topk_obs["action_mask"], gid=engine.get_state().remaining[0] if engine.get_state().remaining else None)
    plot_layout(env2, action_space=cand)
    plot_flow_graph(env2)
