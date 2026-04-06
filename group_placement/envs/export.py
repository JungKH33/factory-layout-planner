"""Group-level placement export (Phase 1 → Phase 2 interchange).

Single unified export surface for ``envs``.  Produces the JSON-serializable
dict that ``facility_placement.resolve_facilities`` consumes for
facility-level unfolding:

- ``export_group_placement(loaded)`` resolves
  ``variant_index → source_index → layout_ref`` and converts cluster BL/W/H
  from grid cells to mm, packaging placements together with the top-level
  ``facilities`` / ``layouts`` registries into one dict.
- ``save_group_placement(loaded, path)`` writes that dict to disk so Phase 2
  can be rerun offline without a live env.

``facility_placement`` consumes only this dict and never touches any ``envs.*``
type.  ``FactoryLayoutEnv`` owns no export logic — callers go through this
module explicitly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from group_placement.envs.env_loader import LoadedEnv


def export_group_placement(loaded: LoadedEnv) -> Dict[str, Any]:
    """Flatten a Phase-1-completed env into the group-placement interchange dict.

    Responsibilities:
    - Walk ``env.get_state().placements`` and resolve each placement's
      ``variant_index`` → ``source_index`` via ``spec.variants[vi].source_index``.
    - Look up the source variant's ``layout_ref`` from
      ``loaded.group_variant_layout_refs`` (parsed from the JSON file).
    - Convert cluster BL / width / height from grid cells to world mm using
      ``loaded.grid_size_mm``.
    - Pass the top-level ``facilities`` and ``layouts`` registries through
      unchanged so the resolver has everything it needs in one dict.

    The returned dict is JSON-serializable — safe to ``json.dump()`` directly.
    """
    env = loaded.env
    grid_size_mm = float(loaded.grid_size_mm)
    state = env.get_state()

    placements_out: List[Dict[str, Any]] = []
    for gid, placement in state.placements.items():
        spec = env.group_specs[gid]
        variant_index = int(getattr(placement, "variant_index", 0))

        # spec.variants is a public property; each variant is a dataclass with source_index.
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

        x_bl = float(getattr(placement, "x_bl", 0.0))
        y_bl = float(getattr(placement, "y_bl", 0.0))
        w_cells = float(getattr(placement, "w", 0.0))
        h_cells = float(getattr(placement, "h", 0.0))
        rotation = int(getattr(placement, "rotation", 0))
        mirror = bool(getattr(placement, "mirror", False))

        placements_out.append({
            "gid":           str(gid),
            "x_bl_mm":       x_bl * grid_size_mm,
            "y_bl_mm":       y_bl * grid_size_mm,
            "cluster_w_mm":  w_cells * grid_size_mm,
            "cluster_h_mm":  h_cells * grid_size_mm,
            "rotation":      rotation,
            "mirror":        mirror,
            "variant_index": variant_index,
            "layout_ref":    layout_ref,
        })

    return {
        "schema_version": 1,
        "grid": {
            "width_cells":  int(env.grid_width),
            "height_cells": int(env.grid_height),
            "grid_size_mm": grid_size_mm,
        },
        "placed_order": [str(gid) for gid in state.placed_nodes()],
        "placements": placements_out,
        "facilities": dict(loaded.facilities_raw),
        "layouts":    dict(loaded.layouts_raw),
    }


def save_group_placement(loaded: LoadedEnv, path: str) -> None:
    """Write ``export_group_placement(loaded)`` to ``path`` as pretty-printed JSON.

    Useful for offline facility-level debugging: run cluster placement, save
    the interchange dict, then load it from disk and feed it to
    ``resolve_facilities`` without any running env.
    """
    data = export_group_placement(loaded)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = [
    "export_group_placement",
    "save_group_placement",
]
