from __future__ import annotations
import torch

from dataclasses import dataclass
from typing import Optional, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx

from typing import Any

from envs.wrappers.candidate_set import CandidateSet


@dataclass(frozen=True)
class StepFrame:
    """A single step frame for interactive browsing."""

    snapshot: dict[str, object]  # wrapper.get_snapshot()
    candidates: CandidateSet
    scores: "np.ndarray"  # float [N] (same length as candidates.xyrot)
    selected_action: int
    value: float
    cost: float
    step_idx: int


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
    # - flow/score/candidates ON
    # - routes ON (if provided)
    return {
        "forbidden_areas": True,
        "invalid_mask": False,
        "clearance_mask": False,
        "flow": True,
        "score": True,
        "candidates": True,
        "routes": True,
        "weight_zones": True,
        "dry_zones": True,
        "height_zones": True,
        "placement_zones": True,
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
        Line2D([0], [0], color="green", lw=0.0, marker="o", markersize=6, alpha=0.65, label="candidates"),
        Line2D([0], [0], color="orange", lw=2.0, alpha=0.8, label="routes"),
        patches.Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.08, label="weight_zones"),
        patches.Patch(facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.06, label="dry_zones"),
        patches.Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", alpha=0.04, label="height_zones"),
        patches.Patch(facecolor="#1e90ff", edgecolor="#1e90ff", alpha=0.12, label="placement_zones"),
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
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", _on_pick)

    return state


def _draw_layout_layers(
    *,
    ax: plt.Axes,
    engine: Any,
    candidate_set: Optional[CandidateSet] = None,
    routes: Optional[list[Any]] = None,
) -> dict[str, list[Any]]:
    """Draw the same base layers used by plot_layout (zones/masks/layout/flow/score/candidates/routes)."""
    zone_artists: dict[str, list[Any]] = {"weight": [], "dry": [], "height": [], "placement": [], "forbidden": []}
    misc_artists: dict[str, list[Any]] = {
        "invalid_mask": [],
        "clearance_mask": [],
        "flow": [],
        "score": [],
        "candidates": [],
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

    # zones
    if hasattr(engine, "weight_areas") and isinstance(getattr(engine, "weight_areas"), list):
        for a in getattr(engine, "weight_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            v = a.get("value", None)
            if rect is None:
                continue
            label = None if v is None else f"w≤{float(v):g}"
            _add_zone_rect(
                rect=rect,
                kind="weight",
                edgecolor="#1f77b4",
                facecolor="#1f77b4",
                alpha=0.08,
                linestyle="-",
                label=label,
            )

    if hasattr(engine, "dry_areas") and isinstance(getattr(engine, "dry_areas"), list):
        for a in getattr(engine, "dry_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            value = a.get("value", None)
            if rect is None:
                continue
            label = None if value is None else f"dry≥{float(value):g}"
            _add_zone_rect(
                rect=rect,
                kind="dry",
                edgecolor="#2ca02c",
                facecolor="#2ca02c",
                alpha=0.06,
                linestyle="--",
                label=label,
            )

    if hasattr(engine, "height_areas") and isinstance(getattr(engine, "height_areas"), list):
        for a in getattr(engine, "height_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            ch = a.get("value", None)
            if rect is None:
                continue
            label = None if ch is None else f"h≤{float(ch):g}"
            _add_zone_rect(
                rect=rect,
                kind="height",
                edgecolor="#7f7f7f",
                facecolor="#7f7f7f",
                alpha=0.04,
                linestyle=":",
                label=label,
            )

    # placement_areas: named areas that groups can be restricted to
    if hasattr(engine, "placement_areas") and isinstance(getattr(engine, "placement_areas"), list):
        for a in getattr(engine, "placement_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            aid = a.get("id", None)
            if rect is None:
                continue
            label = str(aid) if aid is not None else None
            _add_zone_rect(
                rect=rect,
                kind="placement",
                edgecolor="#1e90ff",
                facecolor="#1e90ff",
                alpha=0.12,
                linestyle="-.",
                label=label,
            )

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
    for gid in getattr(engine, "placed", []):
        p = engine.placements[gid]
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

    # candidates (optional; caller may render their own)
    if candidate_set is not None:
        meta = candidate_set.meta or {}
        gid = getattr(candidate_set, "gid", None)
        if gid is None:
            gid = meta.get("gid", None)
        xyrot = candidate_set.xyrot[candidate_set.mask]
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
            misc_artists["candidates"].append(sc)

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
        "candidates": misc_artists["candidates"],
        "routes": misc_artists["routes"],
        "weight_zones": zone_artists["weight"],
        "dry_zones": zone_artists["dry"],
        "height_zones": zone_artists["height"],
        "placement_zones": zone_artists["placement"],
    }


def plot_layout(env: Any, *, candidate_set: Optional[CandidateSet] = None, routes: Optional[list[Any]] = None) -> None:
    """Interactive viewer (dynamic toggles only).

    - No save_path/show_* args here on purpose: use `save_layout(...)` for saving.
    - This function always opens a window and lets you toggle layers via CheckButtons.
    
    Args:
        env: FactoryLayoutEnv or wrapper
        candidate_set: Optional candidate set to display
        routes: Optional list of RouteResult from postprocess.pathfinder
    """
    # Support both engine (`FactoryLayoutEnv`) and wrapper envs by unwrapping.
    engine = getattr(env, "engine", env)

    # Keep legend outside the layout area (separate axis on the right).
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[20, 5], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title("FactoryLayoutEnv")

    groups = _draw_layout_layers(ax=ax, engine=engine, candidate_set=candidate_set, routes=routes)
    layer_vis = _default_layer_visibility()
    _apply_layer_visibility(groups, layer_vis)
    _install_click_legend(fig=fig, ax=ax, groups=groups, vis=layer_vis, legend_ax=ax_leg)

    plt.tight_layout()
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
    """Browse step-by-step candidate policies with a consistent scatter visualization.

    - Candidates are plotted as scatter points colored by `scores`.
    - Selected action is highlighted with a bold outline.
    - Uses wrapper snapshot API: env.set_snapshot(frame.snapshot).
    """
    if not frames:
        raise ValueError("browse_steps: frames is empty")

    # Support wrapper envs by unwrapping engine.
    wrapper = env
    engine = getattr(env, "engine", env)
    if not hasattr(wrapper, "set_snapshot"):
        raise ValueError("browse_steps requires a wrapper env with set_snapshot(get_snapshot()) support.")

    # Fixed layout so axes don't shrink when navigating:
    # - main axis: layout + scatter
    # - right axis: colorbar
    # - far-right axis: clickable legend (outside layout)
    # - bottom axis: info panel (outside plot area)
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[12, 2],
        width_ratios=[20, 1, 5],
        wspace=0.15,
        hspace=0.15,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title(title)

    # persistent artists (updated in-place)
    sc = None
    sc_sel = None
    cbar = None
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

    def _frame_to_xy(frame: StepFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
        cand = frame.candidates
        gid = cand.gid
        if gid is None:
            raise ValueError("CandidateSet.gid is required for BL->center conversion.")
        mask = cand.mask.detach().cpu().numpy().astype(bool)
        xyrot = cand.xyrot.detach().cpu().numpy()
        scores = np.asarray(frame.scores, dtype=np.float32)
        if scores.shape[0] != xyrot.shape[0]:
            raise ValueError(f"scores length {scores.shape[0]} != candidates {xyrot.shape[0]}")

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
        nonlocal sc, sc_sel, cbar
        nonlocal legend_state
        idx = int(max(0, min(len(frames) - 1, idx)))
        cur["idx"] = idx
        f = frames[idx]

        # restore snapshot for this frame
        wrapper.set_snapshot(f.snapshot)  # type: ignore[attr-defined]

        ax.clear()
        ax.set_xlim(0, engine.grid_width)
        ax.set_ylim(0, engine.grid_height)
        ax.set_aspect("equal")
        ax.set_title(title + f"  [step {f.step_idx}/{frames[-1].step_idx}]")

        ax_leg.clear()
        ax_leg.axis("off")

        # Draw the same base UI layers as plot_layout (zones/masks/layout/flow/score),
        # then overlay policy-colored candidates on top.
        groups = _draw_layout_layers(ax=ax, engine=engine, candidate_set=None)
        xs, ys, cs, (sx, sy) = _frame_to_xy(f)
        sc = ax.scatter(xs, ys, s=float(point_size), c=cs, cmap=cmap, alpha=0.85, linewidths=0.0)
        sc_sel = ax.scatter(
            [sx],
            [sy],
            s=float(selected_point_size),
            facecolors="none",
            edgecolors="#1f77b4",  # blue
            linewidths=float(selected_edge_width),
        )
        groups["candidates"].extend([sc, sc_sel])
        _apply_layer_visibility(groups, layer_vis)
        legend_state = _install_click_legend(
            fig=fig,
            ax=ax,
            groups=groups,
            vis=layer_vis,
            legend_ax=ax_leg,
            connect_once_state=legend_state,
        )

        # colorbar (fixed axis; no layout shrink). Create once, then update.
        if cbar is None:
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label("policy score / prob")
        else:
            cbar.update_normal(sc)

        # info panel outside the plot
        sel_score = float(f.scores[int(f.selected_action)]) if 0 <= int(f.selected_action) < len(f.scores) else float("nan")
        info_text.set_text(
            " | ".join(
                [
                    f"step={f.step_idx}",
                    f"cost={f.cost:.3f}",
                    f"value={f.value:.6f}",
                    f"selected_action={int(f.selected_action)}",
                    f"selected_policy={sel_score:.6f}",
                    f"valid={int(f.candidates.mask.to(torch.int64).sum().item())}",
                ]
            )
        )

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
    candidate_set: Optional[CandidateSet] = None,
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
        # weight
        if hasattr(engine, "weight_areas") and isinstance(getattr(engine, "weight_areas"), list):
            for a in getattr(engine, "weight_areas"):
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
                        edgecolor="#1f77b4",
                        facecolor="#1f77b4",
                        alpha=0.08,
                    )
                )
                if value is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"w≤{float(value):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#1f77b4",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                    )
        # dry
        if hasattr(engine, "dry_areas") and isinstance(getattr(engine, "dry_areas"), list):
            for a in getattr(engine, "dry_areas"):
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
                        edgecolor="#2ca02c",
                        facecolor="#2ca02c",
                        alpha=0.06,
                        linestyle="--",
                    )
                )
                if value is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"dry≥{float(value):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#2ca02c",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                    )
        # height
        if hasattr(engine, "height_areas") and isinstance(getattr(engine, "height_areas"), list):
            for a in getattr(engine, "height_areas"):
                if not isinstance(a, dict):
                    continue
                rect = a.get("rect", None)
                ch = a.get("value", None)
                if rect is None:
                    continue
                x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        max(0, x1 - x0),
                        max(0, y1 - y0),
                        linewidth=1.2,
                        edgecolor="#7f7f7f",
                        facecolor="#7f7f7f",
                        alpha=0.04,
                        linestyle=":",
                    )
                )
                if ch is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"h≤{float(ch):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#7f7f7f",
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

    for gid in engine.placed:
        p = engine.placements[gid]
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

    if candidate_set is not None:
        meta = candidate_set.meta or {}
        gid = getattr(candidate_set, "gid", None)
        if gid is None:
            gid = meta.get("gid", None)
        xyrot = candidate_set.xyrot[candidate_set.mask]
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
    if not env.placed:
        return []
    arts: list[Any] = []
    # Use reward-computed argmin port pairs if available (env._flow_port_pairs).
    port_pairs = getattr(env, "_flow_port_pairs", {})
    for src, targets in env.group_flow.items():
        if src not in env.placed:
            continue
        src_p = env.placements[src]
        for dst, weight in targets.items():
            if dst not in env.placed:
                continue
            dst_p = env.placements[dst]
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
    # - Uses latest engine constraint fields:
    #   env.default_*, zones.*_areas[].value, groups.*.facility_*
    # - Shows zone overlays and interactive toggles (show=True only).
    import torch

    from envs.wrappers.alphachip import AlphaChipWrapperEnv
    from envs.wrappers.greedy import GreedyWrapperEnv
    from envs.env import FacilityGroup, FactoryLayoutEnv

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- groups (footprint width/height + constraint attributes) ---
    # Note: footprint height is still `height`; constraint vertical height is `facility_height`.
    groups = {
        "A": FacilityGroup(id="A", width=20, height=10, rotatable=True, facility_weight=3.0, facility_height=2.0, facility_dry=0.0),
        "B": FacilityGroup(id="B", width=16, height=16, rotatable=True, facility_weight=4.0, facility_height=2.0, facility_dry=0.0),
        # Make C "heavy + tall + dry-sensitive" to demonstrate zones affecting mask.
        "C": FacilityGroup(id="C", width=18, height=12, rotatable=True, facility_weight=12.0, facility_height=10.0, facility_dry=2.0),
    }
    flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # --- forbidden areas ---
    forbidden_areas = [{"rect": [0, 0, 30, 20]}]

    # --- zones / constraints ---
    # Unified schema: default_* + *_areas (rect,value)
    default_weight = 10.0
    weight_areas = [{"rect": (60, 0, 120, 80), "value": 20.0}]  # higher allowed weight on right half
    default_height = 20.0
    height_areas = [{"rect": (0, 60, 120, 80), "value": 5.0}]  # low ceiling strip
    default_dry = 0.0
    dry_areas = [{"rect": (0, 40, 60, 80), "value": 2.0}]  # higher dry requirement zone (reverse inequality)

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
        groups=groups,
        group_flow=flow,
        forbidden_areas=forbidden_areas,
        device=dev,
        max_steps=10,
        weight_areas=weight_areas,
        height_areas=height_areas,
        dry_areas=dry_areas,
        default_weight=default_weight,
        default_height=default_height,
        default_dry=default_dry,
        log=False,
    )

    # ---- 1) Coarse wrapper demo ----
    # For coarse, we just visualize the current layout (candidate visualization via decode is omitted in demo).
    env1 = AlphaChipWrapperEnv(engine=engine, coarse_grid=32, rot=0)
    _obs1, _ = env1.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    plot_layout(env1, candidate_set=None)
    plot_flow_graph(env1)

    # ---- 2) Greedy(TopK) wrapper demo ----
    env2 = GreedyWrapperEnv(
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
    # Greedy(TopK) wrapper already builds candidates; use obs-provided (decoded) candidates for plotting.
    topk_obs, _ = env2.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    cand = None
    if isinstance(topk_obs, dict) and ("action_mask" in topk_obs) and ("action_xyrot" in topk_obs):
        cand = CandidateSet(xyrot=topk_obs["action_xyrot"], mask=topk_obs["action_mask"], gid=engine.remaining[0] if engine.remaining else None)
    plot_layout(env2, candidate_set=cand)
    plot_flow_graph(env2)

