"""Phase 1 → Phase 2 interchange: export ``LoadedEnv`` to a JSON-serializable dict."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from group_placement.envs.env_loader import LoadedEnv


def export_group_placement(loaded: LoadedEnv) -> Dict[str, Any]:
    """Build the group-placement interchange dict (grid + placements + registries).

    ``placements`` rows follow ``placed_order``.  Integer grid fields preserve
    exact BL/dims for engine restore; mm fields remain for downstream consumers.
    """
    env = loaded.env
    grid_size_mm = float(loaded.grid_size_mm)
    state = env.get_state()

    placements_out: List[Dict[str, Any]] = []
    for gid in state.placed_nodes():
        placement = state.placements.get(gid)
        if placement is None:
            continue
        spec = env.group_specs[gid]
        variant_index = int(getattr(placement, "variant_index", 0))

        try:
            source_index = int(spec.variants[variant_index].source_index)
        except (IndexError, AttributeError):
            source_index = 0

        refs = loaded.group_variant_layout_refs.get(str(gid), [])
        layout_ref: Optional[str]
        if 0 <= source_index < len(refs):
            layout_ref = refs[source_index]
        else:
            layout_ref = None

        x_bl = int(getattr(placement, "x_bl", 0))
        y_bl = int(getattr(placement, "y_bl", 0))
        w_cells = int(getattr(placement, "w", 0))
        h_cells = int(getattr(placement, "h", 0))
        rotation = int(getattr(placement, "rotation", 0))
        mirror = bool(getattr(placement, "mirror", False))

        placements_out.append({
            "gid": str(gid),
            "x_bl_cells": x_bl,
            "y_bl_cells": y_bl,
            "cluster_w_cells": w_cells,
            "cluster_h_cells": h_cells,
            "x_bl_mm": float(x_bl) * grid_size_mm,
            "y_bl_mm": float(y_bl) * grid_size_mm,
            "cluster_w_mm": float(w_cells) * grid_size_mm,
            "cluster_h_mm": float(h_cells) * grid_size_mm,
            "rotation": rotation,
            "mirror": mirror,
            "variant_index": variant_index,
            "layout_ref": layout_ref,
        })

    return {
        "schema_version": 1,
        "grid": {
            "width_cells": int(env.grid_width),
            "height_cells": int(env.grid_height),
            "grid_size_mm": grid_size_mm,
        },
        "placed_order": [str(g) for g in state.placed_nodes()],
        "placements": placements_out,
        "facilities": dict(loaded.facilities_raw),
        "layouts": dict(loaded.layouts_raw),
    }


def save_group_placement(loaded: LoadedEnv, path: str) -> None:
    """Write ``export_group_placement(loaded)`` to ``path`` as UTF-8 JSON."""
    data = export_group_placement(loaded)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = [
    "export_group_placement",
    "save_group_placement",
]
