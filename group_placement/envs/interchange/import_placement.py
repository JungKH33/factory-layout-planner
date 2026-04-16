"""Load interchange JSON and restore ``FactoryLayoutEnv`` state via ``reset``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

from group_placement.envs.env import FactoryLayoutEnv
from group_placement.envs.env_loader import LoadedEnv, load_env
from group_placement.envs.placement.base import GroupPlacement


def load_group_placement(path: Union[str, Path]) -> Dict[str, Any]:
    """Read interchange JSON from disk."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_group_id(env: FactoryLayoutEnv, gid_s: str) -> str | int:
    for k in env.group_specs:
        if str(k) == gid_s:
            return k
    raise KeyError(f"unknown group id {gid_s!r} (not in env.group_specs)")


def _grid_cells_from_entry(
    entry: Mapping[str, Any],
    *,
    grid_size_mm: float,
) -> tuple[int, int, int, int]:
    if all(k in entry for k in ("x_bl_cells", "y_bl_cells", "cluster_w_cells", "cluster_h_cells")):
        return (
            int(entry["x_bl_cells"]),
            int(entry["y_bl_cells"]),
            int(entry["cluster_w_cells"]),
            int(entry["cluster_h_cells"]),
        )
    gsm = float(grid_size_mm)
    if gsm <= 0:
        raise ValueError("grid_size_mm must be positive when inferring cells from mm fields")
    x_bl = int(round(float(entry.get("x_bl_mm", 0.0)) / gsm))
    y_bl = int(round(float(entry.get("y_bl_mm", 0.0)) / gsm))
    w = int(round(float(entry.get("cluster_w_mm", 0.0)) / gsm))
    h = int(round(float(entry.get("cluster_h_mm", 0.0)) / gsm))
    return x_bl, y_bl, w, h


def _validate_grid(env: FactoryLayoutEnv, grid_block: Mapping[str, Any]) -> None:
    w = int(grid_block["width_cells"])
    h = int(grid_block["height_cells"])
    if int(env.grid_width) != w or int(env.grid_height) != h:
        raise ValueError(
            f"interchange grid {w}x{h} does not match env {env.grid_width}x{env.grid_height}"
        )


def apply_interchange_to_loaded(
    loaded: LoadedEnv,
    data: Mapping[str, Any],
    *,
    strict: bool = True,
    check_grid: bool = True,
) -> None:
    """Replay interchange placements on ``loaded.env`` (in ``placed_order``).

    Requires ``load_env`` from the same env JSON the export was produced from
    (same ``groups`` / flow / zones).  Call ``loaded.env.reset`` with derived
    ``initial_placements`` and ``placement_order``.
    """
    env = loaded.env
    grid_block = data.get("grid") or {}
    if check_grid:
        if "width_cells" not in grid_block or "height_cells" not in grid_block:
            raise ValueError("interchange dict missing grid.width_cells / grid.height_cells")
        _validate_grid(env, grid_block)
    gsm_loaded = float(loaded.grid_size_mm)
    gsm_data = float(grid_block.get("grid_size_mm", gsm_loaded))
    if abs(gsm_data - gsm_loaded) > 1e-6:
        raise ValueError(
            f"interchange grid_size_mm={gsm_data} != loaded.grid_size_mm={gsm_loaded}"
        )

    entries: List[Mapping[str, Any]] = list(data.get("placements") or [])
    by_gid: Dict[str, Mapping[str, Any]] = {}
    for e in entries:
        by_gid[str(e["gid"])] = e

    order_s: List[str]
    if data.get("placed_order"):
        order_s = [str(x) for x in data["placed_order"]]
    else:
        order_s = [str(e["gid"]) for e in entries]

    if strict and order_s:
        if len(order_s) != len(set(order_s)):
            raise ValueError("placed_order contains duplicate gids")
        if set(order_s) != set(by_gid):
            raise ValueError(
                f"placed_order keys {set(order_s)!r} != placement gids {set(by_gid)!r}"
            )

    initial: Dict[str | int, GroupPlacement] = {}
    order_ids: List[str | int] = []

    for gid_s in order_s:
        if gid_s not in by_gid:
            if strict:
                raise KeyError(f"placed_order references missing placement for gid={gid_s!r}")
            continue
        entry = by_gid[gid_s]
        gid = _resolve_group_id(env, gid_s)
        variant_index = int(entry.get("variant_index", 0))
        spec = env.group_specs[gid]
        variants = spec.variants
        if variant_index < 0 or variant_index >= len(variants):
            raise IndexError(f"gid={gid_s!r}: variant_index={variant_index} out of range")
        vi = variants[variant_index]
        x_bl, y_bl, _w, _h = _grid_cells_from_entry(entry, grid_size_mm=gsm_loaded)
        x_c = float(x_bl) + float(vi.body_width) / 2.0
        y_c = float(y_bl) + float(vi.body_height) / 2.0
        placement = env.resolve_center_placement(
            group_id=gid,
            x_center=x_c,
            y_center=y_c,
            variant_index=variant_index,
        )
        if placement is None:
            if strict:
                raise RuntimeError(
                    "interchange placement resolve failed: "
                    f"gid={gid_s!r} variant_index={variant_index} x_center={x_c} y_center={y_c}"
                )
            continue
        initial[gid] = placement
        order_ids.append(gid)

    opts: Dict[str, Any] = {}
    if initial:
        opts["initial_placements"] = initial
        opts["placement_order"] = order_ids
        opts["strict_initial_placements"] = strict
    env.reset(options=opts)


def restore_loaded_from_files(
    env_json: Union[str, Path],
    interchange_json: Union[str, Path],
    *,
    strict: bool = True,
    check_grid: bool = True,
    **load_kw: Any,
) -> LoadedEnv:
    """``load_env`` + ``apply_interchange_to_loaded`` in one call."""
    loaded = load_env(str(Path(env_json)), **load_kw)
    data = load_group_placement(interchange_json)
    apply_interchange_to_loaded(loaded, data, strict=strict, check_grid=check_grid)
    return loaded


__all__ = [
    "apply_interchange_to_loaded",
    "load_group_placement",
    "restore_loaded_from_files",
]
