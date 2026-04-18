"""Matplotlib visualization backend.

Moved from envs/env_visualizer.py — all matplotlib-specific rendering lives here.
"""
from __future__ import annotations

from typing import Any, Optional, List, Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx

from group_placement.envs.visualizer.base import VisualizerBackend
from group_placement.envs.visualizer.data import (
    LayoutData, StepFrame, ConstraintZoneData,
    constraint_color, CONSTRAINT_BASE_COLORS,
)
from group_placement.envs.visualizer.layer import RewardLayer


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _single_color_cmap(color: str):
    return plt.matplotlib.colors.ListedColormap([color])


def _constraint_cmap(name: str, names: list) -> LinearSegmentedColormap:
    base = constraint_color(name, names)
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(base)
    light = (r + (1 - r) * 0.7, g + (1 - g) * 0.7, b + (1 - b) * 0.7)
    return LinearSegmentedColormap.from_list(f"constraint_{str(name)}", [light, base])


def _plot_mask(ax: plt.Axes, mask_np: Optional[np.ndarray], *, color: str, alpha: float):
    """Render a boolean mask as a colored overlay. Returns the image artist or None."""
    if mask_np is None:
        return None
    grid = mask_np.astype(np.float32)
    if grid.ndim != 2 or grid.size == 0:
        return None
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


def _set_visible_group(group: list, v: bool) -> None:
    for a in group:
        try:
            a.set_visible(bool(v))
        except Exception:
            pass


def _default_layer_visibility(
    constraint_names: list = (),
    reward_layers: dict = (),
) -> dict:
    vis = {
        "forbidden": True,
        "invalid_mask": False,
        "clearance_mask": True,
        "flow": True,
        "ports": True,
        "score": True,
        "action_space": True,
    }
    for name in constraint_names:
        vis[f"zone:{name}"] = False
    for key, layer in dict(reward_layers).items():
        if isinstance(layer, RewardLayer):
            vis[f"reward:{key}"] = layer.default_visible
        else:
            vis[f"reward:{key}"] = True
    return vis


def _apply_layer_visibility(groups: dict, vis: dict) -> None:
    for k, g in groups.items():
        if k not in vis:
            continue
        _set_visible_group(g, bool(vis[k]))


def _legend_proxies(constraint_names: list = (), reward_layers: dict = ()) -> list:
    proxies = [
        patches.Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.15, label="forbidden"),
        patches.Patch(facecolor="#8b0000", edgecolor="#8b0000", alpha=0.10, label="invalid_mask"),
        patches.Patch(facecolor="lightgray", edgecolor="gray", alpha=0.30, label="clearance_mask"),
        Line2D([0], [0], color="black", lw=0.0, marker="s", markersize=8, label="score"),
        Line2D([0], [0], color="green", lw=0.0, marker="o", markersize=6, alpha=0.65, label="action_space"),
        Line2D(
            [0], [0], color="#555555", lw=0.0, marker="x", markersize=7,
            markeredgewidth=1.3, linestyle="None", label="ports",
        ),
    ]
    for name in constraint_names:
        color = constraint_color(name, list(constraint_names))
        proxies.append(
            patches.Patch(facecolor=color, edgecolor=color, alpha=0.15, label=f"zone:{name}"),
        )
    # Per-reward-component proxies
    for key, layer in dict(reward_layers).items():
        if isinstance(layer, RewardLayer):
            ls = "--" if layer.style == "dashed" else "-"
            proxies.append(Line2D(
                [0], [0], color=layer.color, lw=1.6, alpha=0.8,
                linestyle=ls, label=f"reward:{key}",
            ))
        else:
            proxies.append(Line2D([0], [0], color="royalblue", lw=1.6, alpha=0.8,
                                  label=f"reward:{key}"))
    # Fallback: if no reward layers, keep classic "flow" entry
    if not dict(reward_layers):
        proxies.insert(3, Line2D([0], [0], color="royalblue", lw=1.8, alpha=0.7, label="flow"))
    return proxies


def _install_click_legend(
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    groups: dict,
    vis: dict,
    title: str = "Click legend to toggle",
    legend_ax: Optional[plt.Axes] = None,
    connect_once_state: Optional[dict] = None,
    on_toggle: Optional[Callable] = None,
    constraint_names: list = (),
    reward_layers: dict = (),
) -> dict:
    leg_host = legend_ax if legend_ax is not None else ax
    proxies = _legend_proxies(constraint_names=constraint_names, reward_layers=reward_layers)
    leg = leg_host.legend(handles=proxies, loc="upper left", title=title, framealpha=0.85)

    legend_artist_to_key: dict = {}
    keys = [str(p.get_label()) for p in proxies]

    for i, text in enumerate(leg.get_texts()):
        k = str(text.get_text())
        text.set_picker(10)
        legend_artist_to_key[text] = k
        try:
            text.set_alpha(1.0 if bool(vis.get(k, True)) else 0.35)
        except Exception:
            pass

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
            cur = bool(state.get("vis", {}).get(key, True))
            state["vis"][key] = not cur
            _set_visible_group(state.get("groups", {}).get(key, []), not cur)
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


# ---------------------------------------------------------------------------
# Core rendering from LayoutData
# ---------------------------------------------------------------------------

def _draw_layout_from_data(
    ax: plt.Axes,
    data: LayoutData,
) -> dict:
    """Render LayoutData onto an Axes, returning layer groups dict."""
    forbidden_artists: list = []
    misc_artists: dict = {
        "invalid_mask": [],
        "clearance_mask": [],
        "flow": [],
        "ports": [],
        "score": [],
        "action_space": [],
    }
    per_zone: dict = {}

    # --- constraint zone overlays ---
    for name in data.constraint_names:
        zone = data.constraint_zones.get(name)
        if zone is None:
            continue
        arts: list = []
        if zone.heatmap is not None:
            arr = zone.heatmap
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if abs(vmax - vmin) < 1e-9:
                vmax = vmin + 1.0
            im = ax.imshow(
                arr,
                origin="lower",
                extent=(0, data.grid_width, 0, data.grid_height),
                cmap=_constraint_cmap(name, data.constraint_names),
                vmin=vmin,
                vmax=vmax,
                alpha=0.45,
                interpolation="nearest",
                zorder=0.2,
            )
            arts.append(im)
        # constraint rects
        for area in zone.rects:
            rect = area.get("rect", None)
            if rect is None:
                continue
            x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            if w <= 0 or h <= 0:
                continue
            patch = patches.Rectangle(
                (x0, y0), w, h,
                linewidth=1.1, edgecolor=zone.color, facecolor=zone.color,
                alpha=0.07, linestyle="-", zorder=1.0,
            )
            ax.add_patch(patch)
            arts.append(patch)
            if area.get("value", None) is not None:
                label = ax.text(
                    x0 + w / 2.0, y0 + h / 2.0,
                    f"{name}{zone.op}{area['value']}",
                    ha="center", va="center", fontsize=8, color=zone.color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, linewidth=0.0),
                    zorder=1.1,
                )
                arts.append(label)
        per_zone[f"zone:{name}"] = arts

    # --- forbidden areas ---
    for x0, y0, x1, y1 in data.forbidden_rects:
        w, h = x1 - x0, y1 - y0
        p = patches.Rectangle(
            (x0, y0), w, h,
            linewidth=1.2, edgecolor="#d62728", facecolor="#d62728", alpha=0.15,
        )
        ax.add_patch(p)
        forbidden_artists.append(p)

    # --- masks (visibility follows _default_layer_visibility) ---
    layer_vis = _default_layer_visibility()
    if data.invalid_mask is not None:
        mi = _plot_mask(ax, data.invalid_mask, color="#8b0000", alpha=1)
        if mi is not None:
            mi.set_visible(layer_vis.get("invalid_mask", False))
            misc_artists["invalid_mask"].append(mi)

    if data.clearance_mask is not None:
        mc = _plot_mask(ax, data.clearance_mask, color="lightgray", alpha=0.3)
        if mc is not None:
            mc.set_visible(layer_vis.get("clearance_mask", False))
            misc_artists["clearance_mask"].append(mc)

    # --- placed facilities ---
    for fac in data.facilities:
        if fac.body_polygon_abs:
            patch = patches.Polygon(
                fac.body_polygon_abs, closed=True,
                linewidth=1.2, edgecolor="black", facecolor="orange", alpha=0.6,
                zorder=3,
            )
            if fac.clearance_polygon_abs:
                cl_patch = patches.Polygon(
                    fac.clearance_polygon_abs, closed=True,
                    linewidth=1, edgecolor="gray", facecolor="lightgray",
                    alpha=0.3, linestyle="--", zorder=2,
                )
                ax.add_patch(cl_patch)
        else:
            patch = patches.Rectangle(
                (fac.x_bl, fac.y_bl), fac.w, fac.h,
                linewidth=1.2, edgecolor="black", facecolor="orange", alpha=0.6,
                zorder=3,
            )
            has_cl = (fac.clearance_left or fac.clearance_right
                      or fac.clearance_bottom or fac.clearance_top)
            if has_cl:
                cl_x = fac.x_bl - fac.clearance_left
                cl_y = fac.y_bl - fac.clearance_bottom
                cl_w = fac.w + fac.clearance_left + fac.clearance_right
                cl_h = fac.h + fac.clearance_bottom + fac.clearance_top
                cl_rect = patches.Rectangle(
                    (cl_x, cl_y), cl_w, cl_h,
                    linewidth=1, edgecolor="gray", facecolor="lightgray",
                    alpha=0.3, linestyle="--", zorder=2,
                )
                ax.add_patch(cl_rect)
        ax.add_patch(patch)
        ax.text(fac.x_center, fac.y_center, str(fac.group_id), ha="center", va="center", fontsize=8)

    # --- action space candidates ---
    if data.candidates_xy:
        xs = [c[0] for c in data.candidates_xy]
        ys = [c[1] for c in data.candidates_xy]
        sc = ax.scatter(xs, ys, s=18, c="green", alpha=0.65, linewidths=0.0)
        sc.set_visible(True)
        misc_artists["action_space"].append(sc)

    # --- per-reward-component layers ---
    reward_artists: dict = {}
    base_ports_drawn = False

    if data.reward_layers:
        for key, layer in data.reward_layers.items():
            arts: list = []
            ls = "--" if layer.style == "dashed" else "-"
            for seg in layer.segments:
                if seg.polyline and len(seg.polyline) >= 2:
                    pl = seg.polyline
                    (line,) = ax.plot(
                        [p[0] for p in pl],
                        [p[1] for p in pl],
                        linestyle=ls, color=layer.color, lw=1.2, alpha=0.7, zorder=3.5,
                    )
                    arts.append(line)
                    ann = ax.annotate(
                        "",
                        xy=(float(pl[-1][0]), float(pl[-1][1])),
                        xytext=(float(pl[-2][0]), float(pl[-2][1])),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=layer.color,
                            linestyle=ls,
                            lw=1.0,
                            alpha=0.7,
                        ),
                        zorder=3.55,
                    )
                    arts.append(ann)
                else:
                    ann = ax.annotate(
                        "",
                        xy=seg.dst_xy,
                        xytext=seg.src_xy,
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=layer.color,
                            linestyle=ls,
                            lw=1.0,
                            alpha=0.7,
                        ),
                    )
                    arts.append(ann)

            # Draw ports only for the first base-phase layer (toggle: ports, not reward:*)
            if layer.phase == "base" and not base_ports_drawn:
                base_ports_drawn = True

                def _scatter_ports(xys, *, color, alpha):
                    if not xys:
                        return
                    xs_p = [pt[0] for pt in xys]
                    ys_p = [pt[1] for pt in xys]
                    sc = ax.scatter(
                        xs_p, ys_p, s=52, c=color, alpha=alpha,
                        marker="x", linewidths=1.15, zorder=4.0,
                    )
                    misc_artists["ports"].append(sc)

                for port in layer.ports:
                    if port.kind == "entry":
                        color, alpha = "#2ca02c", (0.90 if port.active else 0.40)
                    else:
                        color, alpha = "#d62728", (0.90 if port.active else 0.40)
                    _scatter_ports([port.xy], color=color, alpha=alpha)

            reward_artists[f"reward:{key}"] = arts

    # Fallback: use flow_arrows if no reward_layers (legacy path)
    if not data.reward_layers and data.flow_arrows:
        for arrow in data.flow_arrows:
            ann = ax.annotate(
                "",
                xy=arrow.dst_xy,
                xytext=arrow.src_xy,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="royalblue",
                    lw=1.0,
                    alpha=0.7,
                ),
            )
            misc_artists["flow"].append(ann)

        def _scatter_ports_legacy(xys, *, color, alpha):
            if not xys:
                return
            xs_p = [p[0] for p in xys]
            ys_p = [p[1] for p in xys]
            sc = ax.scatter(
                xs_p, ys_p, s=52, c=color, alpha=alpha,
                marker="x", linewidths=1.15, zorder=4.0,
            )
            misc_artists["ports"].append(sc)

        _scatter_ports_legacy(data.ports.inactive_entries, color="#2ca02c", alpha=0.40)
        _scatter_ports_legacy(data.ports.active_entries, color="#2ca02c", alpha=0.90)
        _scatter_ports_legacy(data.ports.inactive_exits, color="#d62728", alpha=0.40)
        _scatter_ports_legacy(data.ports.active_exits, color="#d62728", alpha=0.90)

    # --- score ---
    score_text = ax.text(
        0.01, 0.99, f"cost={data.cost:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )
    score_text.set_visible(True)
    misc_artists["score"].append(score_text)

    groups: dict = {
        "forbidden": forbidden_artists,
        "invalid_mask": misc_artists["invalid_mask"],
        "clearance_mask": misc_artists["clearance_mask"],
        "flow": misc_artists["flow"],
        "ports": misc_artists["ports"],
        "score": misc_artists["score"],
        "action_space": misc_artists["action_space"],
    }
    groups.update(per_zone)
    groups.update(reward_artists)
    return groups


# ---------------------------------------------------------------------------
# Legacy direct-engine rendering (for browse_steps / draw_layout_on_axes)
# ---------------------------------------------------------------------------

def _draw_layout_layers(
    *,
    ax: plt.Axes,
    engine: Any,
    action_space: Any = None,
) -> dict:
    """Draw layout layers directly from engine state.

    Used by browse_steps (which mutates engine state between frames) and
    draw_layout_on_axes (external callers needing to pass their own ax).
    """
    from group_placement.envs.visualizer.data import extract_layout_data
    data = extract_layout_data(engine, action_space=action_space)
    return _draw_layout_from_data(ax, data)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class MatplotlibBackend(VisualizerBackend):

    def plot_layout(self, data: LayoutData, **display_kwargs) -> Any:
        c_names = data.constraint_names

        fig = plt.figure(figsize=(13, 6.5))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[20, 7], wspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")
        ax.set_xlim(0, data.grid_width)
        ax.set_ylim(0, data.grid_height)
        ax.set_aspect("equal")
        layer_vis = _default_layer_visibility(c_names, data.reward_layers)
        legend_state: Optional[dict] = None

        def _render() -> None:
            nonlocal legend_state
            ax.clear()
            ax_leg.clear()
            ax_leg.axis("off")
            ax.set_xlim(0, data.grid_width)
            ax.set_ylim(0, data.grid_height)
            ax.set_aspect("equal")
            ax.set_title("FactoryLayoutEnv")

            groups = _draw_layout_from_data(ax, data)
            _apply_layer_visibility(groups, layer_vis)
            legend_state = _install_click_legend(
                fig=fig, ax=ax, groups=groups, vis=layer_vis,
                legend_ax=ax_leg, connect_once_state=legend_state,
                on_toggle=lambda _key, _visible: _render(),
                constraint_names=c_names,
                reward_layers=data.reward_layers,
            )
            fig.canvas.draw_idle()

        _render()
        plt.show()

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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, data.grid_width)
        ax.set_ylim(0, data.grid_height)
        ax.set_aspect("equal")
        ax.set_title("FactoryLayoutEnv")

        groups = _draw_layout_from_data(ax, data)

        if not show_masks:
            for a in groups.get("forbidden", []):
                a.set_visible(False)
            for a in groups.get("clearance_mask", []):
                a.set_visible(False)
        if not show_flow:
            for a in groups.get("flow", []):
                a.set_visible(False)
        if not show_score:
            for a in groups.get("score", []):
                a.set_visible(False)
        if not show_zones:
            for key, arts in groups.items():
                if key.startswith("zone:"):
                    for a in arts:
                        a.set_visible(False)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def browse_steps(
        self,
        env: Any,
        *,
        frames: List[StepFrame],
        title: str = "Inference browser (\u2190/\u2192 to navigate, q to quit)",
        cmap: str = "RdYlGn",
        point_size: float = 18.0,
        selected_point_size: float = 90.0,
        selected_edge_width: float = 2.5,
        max_points: Optional[int] = None,
    ) -> None:
        """Browse step-by-step layouts with keyboard navigation.

        This needs direct engine access because it restores state per-frame.
        """
        import torch
        from group_placement.envs.state import EnvState

        if not frames:
            raise ValueError("browse_steps: frames is empty")

        wrapper = env
        engine = getattr(env, "engine", env)
        if not hasattr(wrapper, "set_state"):
            raise ValueError("browse_steps requires a wrapper env with set_state() support.")

        from group_placement.envs.visualizer.data import _constraint_names, extract_layout_data
        c_names = _constraint_names(engine)

        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            height_ratios=[12, 2], width_ratios=[20, 1, 7],
            wspace=0.15, hspace=0.15,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax_policy = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[0, 2])
        ax_leg.axis("off")
        ax_info = fig.add_subplot(gs[1, :])
        ax_info.axis("off")

        ax.set_xlim(0, engine.grid_width)
        ax.set_ylim(0, engine.grid_height)
        ax.set_aspect("equal")
        ax.set_title(title)

        sc = None
        sc_sel = None
        policy_cbar = None
        info_text = ax_info.text(
            0.01, 0.5, "",
            transform=ax_info.transAxes, ha="left", va="center",
            fontsize=10, family="monospace",
        )

        cur = {"idx": 0}
        layer_vis = _default_layer_visibility(c_names)
        legend_state: Optional[dict] = None

        def _frame_to_xy(frame: StepFrame):
            cand = frame.action_space
            if cand is None or frame.scores is None or frame.selected_action is None:
                return None
            gid = cand.group_id
            if gid is None:
                raise ValueError("ActionSpace.group_id is required for BL->center conversion.")
            mask = cand.valid_mask.detach().cpu().numpy().astype(bool)
            poses = cand.centers.detach().cpu().numpy()
            scores = np.asarray(frame.scores, dtype=np.float32)
            if scores.shape[0] != poses.shape[0]:
                raise ValueError(f"scores length {scores.shape[0]} != action_space {poses.shape[0]}")

            idxs = np.where(mask)[0]
            if max_points is not None and idxs.shape[0] > int(max_points):
                idxs = idxs[:int(max_points)]

            xs, ys, cs = [], [], []
            for k in idxs.tolist():
                xs.append(float(poses[k, 0]))
                ys.append(float(poses[k, 1]))
                cs.append(float(scores[k]))

            sel = int(frame.selected_action)
            if sel < 0 or sel >= poses.shape[0]:
                sel_xy = (0.0, 0.0)
            else:
                sel_xy = (float(poses[sel, 0]), float(poses[sel, 1]))

            return np.asarray(xs), np.asarray(ys), np.asarray(cs, dtype=np.float32), sel_xy

        def _render(idx: int) -> None:
            nonlocal sc, sc_sel, policy_cbar, legend_state
            idx = int(max(0, min(len(frames) - 1, idx)))
            cur["idx"] = idx
            f = frames[idx]

            # Restore state
            frame_state = f.state
            if isinstance(frame_state, dict) and ("engine" in frame_state) and ("adapter" in frame_state):
                eng_state = frame_state.get("engine", None)
                adp_state = frame_state.get("adapter", None)
                if hasattr(engine, "set_state") and isinstance(eng_state, EnvState):
                    engine.set_state(eng_state)
                if adp_state is not None:
                    wrapper.set_state(adp_state)
            elif isinstance(frame_state, EnvState) and hasattr(engine, "set_state"):
                engine.set_state(frame_state)
            else:
                wrapper.set_state(frame_state)

            ax.clear()
            ax.set_xlim(0, engine.grid_width)
            ax.set_ylim(0, engine.grid_height)
            ax.set_aspect("equal")
            ax.set_title(title + f"  [step {f.step_idx}/{frames[-1].step_idx}]")

            ax_leg.clear()
            ax_leg.axis("off")

            groups = _draw_layout_layers(ax=ax, engine=engine, action_space=None)
            xy_data = _frame_to_xy(f)
            if xy_data is not None:
                xs, ys, cs, (sx, sy) = xy_data
                sc = ax.scatter(xs, ys, s=float(point_size), c=cs, cmap=cmap, alpha=0.85, linewidths=0.0)
                sc_sel = ax.scatter(
                    [sx], [sy],
                    s=float(selected_point_size),
                    facecolors="none", edgecolors="#1f77b4",
                    linewidths=float(selected_edge_width),
                )
                groups["action_space"].extend([sc, sc_sel])
            _apply_layer_visibility(groups, layer_vis)
            legend_state = _install_click_legend(
                fig=fig, ax=ax, groups=groups, vis=layer_vis,
                legend_ax=ax_leg, connect_once_state=legend_state,
                on_toggle=lambda _key, _visible: _render(cur["idx"]),
                constraint_names=c_names,
            )

            if xy_data is not None and sc is not None:
                cax_policy.axis("on")
                if policy_cbar is None:
                    policy_cbar = fig.colorbar(sc, cax=cax_policy)
                    policy_cbar.set_label("policy score / prob")
                else:
                    policy_cbar.update_normal(sc)
            else:
                policy_cbar = None
                cax_policy.clear()
                cax_policy.axis("off")

            parts = [f"step={f.step_idx}", f"cost={f.cost:.3f}"]
            if f.value is not None:
                parts.append(f"value={float(f.value):.6f}")
            if f.action_space is not None and f.scores is not None and f.selected_action is not None:
                sel = int(f.selected_action)
                scores = np.asarray(f.scores, dtype=np.float32)
                sel_score = float(scores[sel]) if 0 <= sel < len(scores) else float("nan")
                parts.append(f"selected_action={sel}")
                parts.append(f"selected_policy={sel_score:.6f}")
                parts.append(f"valid={int(f.action_space.valid_mask.to(torch.int64).sum().item())}")
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

        fig.canvas.mpl_connect("key_press_event", _on_key)
        _render(0)
        plt.show()
        plt.close(fig)

    def plot_flow_graph(self, group_flow: dict, *, show_weights: bool = True) -> None:
        G = nx.DiGraph()
        for src, targets in group_flow.items():
            for dst, weight in targets.items():
                G.add_edge(src, dst, weight=weight)

        if G.number_of_nodes() == 0:
            return

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=9)
        nx.draw_networkx_edges(
            G, pos, arrows=True, arrowstyle="-|>", arrowsize=16,
            width=1.5, connectionstyle="arc3,rad=0.08",
        )
        if show_weights:
            edge_labels = {(u, v): f"{d.get('weight', 1.0):.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Flow Graph")
        plt.tight_layout()
        plt.show()

    def draw_layout_on_axes(
        self,
        engine: Any,
        *,
        ax: Any,
        action_space: Any = None,
    ) -> dict:
        return _draw_layout_layers(ax=ax, engine=engine, action_space=action_space)
