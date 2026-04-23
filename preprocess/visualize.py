"""Visualize preprocess env JSON: one PNG of group × variant layouts.

Given the dict produced by :func:`preprocess.to_env.convert_to_env` (i.e. the
``env.json`` contents), :func:`save_variant_images` renders a single PNG in
which every ``(group, variant)`` combination occupies one subplot.

Layout of the figure:

* rows   = groups (in declaration order)
* cols   = variant index (padded with blank cells when a group has fewer
           variants than the widest row)

For each cell we draw:

* every facility from the variant's ``layout_ref`` as a filled rectangle
  (colored per-group so facilities are easy to group visually), and
* the cluster outline — the variant's ``body_polygon`` if present, otherwise
  the axis-aligned bounding box of its ``(width, height)``.

Styling follows the user's requirements:

* no black axes spine around each subplot,
* minimal whitespace between subplots (tight ``subplots_adjust``),
* tick marks removed; a small title above each cell gives ``gid · layout``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_slots(env: Mapping[str, Any], layout_ref: Optional[str]) -> List[Mapping[str, Any]]:
    if not layout_ref:
        return []
    layouts = env.get("facilities", {}).get("layouts", {})
    lay = layouts.get(layout_ref)
    if not isinstance(lay, dict):
        return []
    slots = lay.get("slots", [])
    return [s for s in slots if isinstance(s, dict)]


def _facility_dims(env: Mapping[str, Any], fid: str) -> Tuple[float, float]:
    facs = env.get("facilities", {}).get("facilities", {})
    fac = facs.get(fid, {})
    return float(fac.get("width", 0.0)), float(fac.get("height", 0.0))


def _variant_outline_world(
    variant: Mapping[str, Any], grid_size: float,
) -> Tuple[List[float], List[float]]:
    """Return ``(xs, ys)`` for the cluster outline in world coordinates.

    ``body_polygon`` — when present — is stored in scaled grid units, so we
    multiply by ``grid_size`` to bring it into the same frame as layout slots
    (which are stored in unscaled world units).
    """
    poly = variant.get("body_polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        xs = [float(p[0]) * grid_size for p in poly]
        ys = [float(p[1]) * grid_size for p in poly]
    else:
        w = float(variant.get("width", 0.0)) * grid_size
        h = float(variant.get("height", 0.0)) * grid_size
        xs = [0.0, w, w, 0.0]
        ys = [0.0, 0.0, h, h]
    return xs, ys


def _apply_slot_transform(
    slot: Mapping[str, Any], width: float, height: float,
) -> Tuple[float, float, float, float]:
    """Return ``(x_bl, y_bl, w, h)`` for a facility after slot rotation.

    Rotations are in degrees ``{0, 90, 180, 270}``. For the common 0° / no-mirror
    case this is a no-op. We only need axis-aligned footprints for drawing;
    rotation swaps width/height for 90° and 270°.
    """
    x = float(slot.get("x", 0.0))
    y = float(slot.get("y", 0.0))
    rot = int(slot.get("rotation", 0)) % 360
    if rot in (90, 270):
        return x, y, float(height), float(width)
    return x, y, float(width), float(height)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def save_variant_images(
    env: Mapping[str, Any],
    save_dir: str | Path,
    *,
    filename: str = "variants.png",
    cell_width_inch: float = 2.8,
    dpi: int = 150,
) -> str:
    """Render a single PNG of every group × variant layout.

    All subplot cells are the same physical size (``cell_width_inch`` wide,
    height automatically chosen from the overall content aspect ratio). Every
    variant is drawn at a shared world scale with ``aspect='equal'`` so wide
    and tall variants are visually comparable. Subplot frames (the default
    black border) are removed and inter-plot padding is kept tight.

    Args:
        env: The env dict returned by ``convert_to_env`` (i.e. the parsed
            contents of ``env.json``).
        save_dir: Directory to write the PNG into. Created if missing.
        filename: Output filename. Default ``variants.png``.
        cell_width_inch: Nominal width in inches reserved for each subplot.
            Cell height is derived automatically from content proportions.
        dpi: Rendering resolution.

    Returns:
        The absolute path of the written PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    groups: Dict[str, Any] = env.get("groups", {}) or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("env.groups is empty; nothing to visualize")

    group_ids: List[str] = list(groups.keys())
    grid_size = float(env.get("grid", {}).get("grid_size", 1.0))

    max_variants = max(
        (len(groups[gid].get("variants", []) or []) for gid in group_ids),
        default=0,
    )
    if max_variants == 0:
        raise ValueError("no variants found across groups; nothing to visualize")

    n_rows = len(group_ids)
    # Reserve the first column as a group-label-only cell.
    n_cols = max_variants + 1

    # Shared world scale: every variant uses the same inches-per-world-unit.
    # Cell dimensions are sized so that (a) the widest variant in the figure
    # fits a cell width, and (b) the tallest variant in the figure fits a
    # cell height. This keeps all variants at the same visual ruler while
    # reducing internal whitespace compared to square cells.
    max_w_units = 1.0
    max_h_units = 1.0
    for gid in group_ids:
        for v in (groups[gid].get("variants", []) or []):
            if not isinstance(v, dict):
                continue
            xs, ys = _variant_outline_world(v, grid_size)
            max_w_units = max(max_w_units, max(xs) - min(xs))
            max_h_units = max(max_h_units, max(ys) - min(ys))

    # 10% headroom inside each cell.
    inch_per_unit = (cell_width_inch * 0.9) / max_w_units
    cell_height_inch = max(max_h_units * inch_per_unit / 0.9, 0.7)

    fig_w = cell_width_inch * n_cols
    fig_h = cell_height_inch * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    # Tight outer margins; small gutter between subplots.
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.955, bottom=0.02,
        wspace=0.03, hspace=0.12,
    )

    cmap = plt.get_cmap("tab20")
    group_color = {gid: cmap(i % 20) for i, gid in enumerate(group_ids)}

    half_cell_w_units = (cell_width_inch / 2.0) / inch_per_unit
    half_cell_h_units = (cell_height_inch / 2.0) / inch_per_unit

    for ri, gid in enumerate(group_ids):
        g = groups[gid] if isinstance(groups[gid], dict) else {}
        variants: Sequence[Mapping[str, Any]] = g.get("variants", []) or []
        fill = group_color[gid]
        for ci in range(n_cols):
            ax = axes[ri][ci]
            # Remove the black subplot frame per user request.
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if ci == 0:
                ax.text(
                    0.5, 0.5, gid,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                continue

            variant_idx = ci - 1
            if variant_idx >= len(variants) or not isinstance(variants[variant_idx], dict):
                ax.set_axis_off()
                continue

            v = variants[variant_idx]
            slots = _resolve_slots(env, v.get("layout_ref"))

            # 1) facility rectangles (filled, subtle edge).
            for slot in slots:
                fid = slot.get("fid")
                if not fid:
                    continue
                w, h = _facility_dims(env, fid)
                if w <= 0 or h <= 0:
                    continue
                x_bl, y_bl, ww, hh = _apply_slot_transform(slot, w, h)
                rect = mpatches.Rectangle(
                    (x_bl, y_bl), ww, hh,
                    facecolor=fill,
                    edgecolor="#2b2b2b",
                    linewidth=0.6,
                    alpha=0.85,
                    zorder=2.0,
                )
                ax.add_patch(rect)

            # 2) cluster outline — drawn on top so facility edges don't hide it.
            xs, ys = _variant_outline_world(v, grid_size)
            out_xs = xs + [xs[0]]
            out_ys = ys + [ys[0]]
            ax.plot(out_xs, out_ys, color="black", linewidth=1.5, zorder=3.0)

            # Fixed cell size in world units centered on the variant: each
            # subplot shows exactly ``cell_width_inch × cell_height_inch`` at
            # the shared world scale, so every variant shares a visual ruler.
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            ax.set_xlim(cx - half_cell_w_units, cx + half_cell_w_units)
            ax.set_ylim(cy - half_cell_h_units, cy + half_cell_h_units)
            ax.set_aspect("equal")

    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(file_path)


__all__ = ["save_variant_images"]
