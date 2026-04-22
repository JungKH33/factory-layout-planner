"""Grid scaling utilities for factory layout env JSON.

Provides lossless integer downscaling: find the largest integer divisor
that keeps every geometry value exact, then divide all spatial fields by it
(and multiply ``grid_size`` accordingly).

Public API
----------
* ``autoscale_layout_env(env)`` -- one-call auto-scale (mutates *env* in-place).
* Lower-level helpers if you need finer control:
  ``collect_scale_basis_values``, ``max_lossless_integer_scale``,
  ``choose_grid_size``, ``apply_grid_scale``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

JsonDict = Dict[str, Any]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: Any) -> float:
    if not _is_number(v):
        raise ValueError(f"expected numeric value, got {v!r}")
    return float(v)


def _scaled_num(v: Any, scale: int) -> int | float:
    x = _num(v) / float(scale)
    xi = int(round(x))
    if abs(x - float(xi)) <= _EPS:
        return xi
    return x


# ---------------------------------------------------------------------------
# Basis-value collection  (used to determine max lossless scale)
# ---------------------------------------------------------------------------

def _append_basis(values: List[float], v: Any) -> None:
    if _is_number(v):
        values.append(float(v))


def _append_polygon_basis(values: List[float], polygon: Any) -> None:
    if not isinstance(polygon, list):
        return
    for p in polygon:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            _append_basis(values, p[0])
            _append_basis(values, p[1])


def _append_area_basis(values: List[float], area: Any) -> None:
    if not isinstance(area, dict):
        return
    rect = area.get("rect")
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        for r in rect:
            _append_basis(values, r)
    _append_polygon_basis(values, area.get("polygon"))


def collect_scale_basis_values(env: JsonDict) -> List[float]:
    """Collect geometry values that must stay exact under grid scaling."""
    vals: List[float] = []

    # -- grid --
    grid = env.get("grid", {})
    if isinstance(grid, dict):
        _append_basis(vals, grid.get("width"))
        _append_basis(vals, grid.get("height"))

    # -- groups --
    groups = env.get("groups", {})
    if isinstance(groups, dict):
        for g in groups.values():
            if not isinstance(g, dict):
                continue
            _append_basis(vals, g.get("width"))
            _append_basis(vals, g.get("height"))
            _append_basis(vals, g.get("clearance"))
            cl = g.get("clearance_lrtb")
            if isinstance(cl, (list, tuple)) and len(cl) == 4:
                for x in cl:
                    _append_basis(vals, x)
            _append_polygon_basis(vals, g.get("body_polygon"))
            _append_polygon_basis(vals, g.get("clearance_polygon"))

            variants = g.get("variants")
            if isinstance(variants, list):
                for v in variants:
                    if not isinstance(v, dict):
                        continue
                    _append_basis(vals, v.get("width"))
                    _append_basis(vals, v.get("height"))
                    _append_basis(vals, v.get("clearance"))
                    vcl = v.get("clearance_lrtb")
                    if isinstance(vcl, (list, tuple)) and len(vcl) == 4:
                        for x in vcl:
                            _append_basis(vals, x)
                    _append_polygon_basis(vals, v.get("body_polygon"))
                    _append_polygon_basis(vals, v.get("clearance_polygon"))

    # -- zones --
    zones = env.get("zones", {})
    if isinstance(zones, dict):
        forbidden = zones.get("forbidden")
        if isinstance(forbidden, list):
            for area in forbidden:
                _append_area_basis(vals, area)
        constraints = zones.get("constraints")
        if isinstance(constraints, dict):
            for cfg in constraints.values():
                if not isinstance(cfg, dict):
                    continue
                areas = cfg.get("areas")
                if isinstance(areas, list):
                    for area in areas:
                        _append_area_basis(vals, area)

    # -- reset / initial_placements --
    reset = env.get("reset")
    if isinstance(reset, dict):
        ip = reset.get("initial_placements")
        if isinstance(ip, dict):
            for pose in ip.values():
                if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                    _append_basis(vals, pose[0])
                    _append_basis(vals, pose[1])

    return vals


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

def max_lossless_integer_scale(values: List[float]) -> int:
    """Largest integer *s* where all basis values / *s* stay integer-like."""
    ints: List[int] = []
    for v in values:
        iv = int(round(v))
        if abs(v - float(iv)) > _EPS:
            return 1
        if iv != 0:
            ints.append(abs(iv))
    if not ints:
        return 1
    g = ints[0]
    for i in ints[1:]:
        g = math.gcd(g, i)
        if g == 1:
            break
    return max(1, int(g))


def _divisors(n: int) -> List[int]:
    out: List[int] = []
    r = int(math.isqrt(n))
    for i in range(1, r + 1):
        if n % i == 0:
            out.append(i)
            j = n // i
            if j != i:
                out.append(j)
    out.sort()
    return out


def is_lossless_for_scale(values: List[float], scale: int) -> bool:
    if scale == 1:
        return True
    for v in values:
        iv = int(round(v))
        if abs(v - float(iv)) > _EPS:
            return False
        if iv % scale != 0:
            return False
    return True


def choose_grid_size(
    base_grid_size: int,
    max_scale: int,
    auto_scale_up: bool,
) -> int:
    if not auto_scale_up:
        return base_grid_size
    divisors = _divisors(max_scale)
    candidates = [
        d for d in divisors if d >= base_grid_size and d % base_grid_size == 0
    ]
    if not candidates:
        return base_grid_size
    return max(candidates)


# ---------------------------------------------------------------------------
# In-place scaling helpers
# ---------------------------------------------------------------------------

def _scale_point_list(
    points: Any,
    scale: int,
    *,
    seen: set[int] | None = None,
) -> None:
    if not isinstance(points, list):
        return
    if seen is not None:
        obj_id = id(points)
        if obj_id in seen:
            return
        seen.add(obj_id)
    for p in points:
        if isinstance(p, list) and len(p) == 2 and _is_number(p[0]) and _is_number(p[1]):
            p[0] = _scaled_num(p[0], scale)
            p[1] = _scaled_num(p[1], scale)


def _scale_area(area: Any, scale: int) -> None:
    if not isinstance(area, dict):
        return
    rect = area.get("rect")
    if isinstance(rect, list) and len(rect) == 4:
        for i in range(4):
            if _is_number(rect[i]):
                rect[i] = _scaled_num(rect[i], scale)
    _scale_point_list(area.get("polygon"), scale)


def _scale_group_dict(g: JsonDict, scale: int) -> None:
    scaled_point_lists: set[int] = set()

    for k in ("width", "height", "_unit_width", "_unit_height", "clearance"):
        if k in g and _is_number(g[k]):
            g[k] = _scaled_num(g[k], scale)

    cl = g.get("clearance_lrtb")
    if isinstance(cl, list) and len(cl) == 4:
        for i in range(4):
            if _is_number(cl[i]):
                cl[i] = _scaled_num(cl[i], scale)

    _scale_point_list(g.get("entries_rel"), scale, seen=scaled_point_lists)
    _scale_point_list(g.get("exits_rel"), scale, seen=scaled_point_lists)
    _scale_point_list(g.get("body_polygon"), scale, seen=scaled_point_lists)
    _scale_point_list(g.get("clearance_polygon"), scale, seen=scaled_point_lists)

    variants = g.get("variants")
    if isinstance(variants, list):
        for v in variants:
            if not isinstance(v, dict):
                continue
            for k in ("width", "height", "clearance"):
                if k in v and _is_number(v[k]):
                    v[k] = _scaled_num(v[k], scale)
            vcl = v.get("clearance_lrtb")
            if isinstance(vcl, list) and len(vcl) == 4:
                for i in range(4):
                    if _is_number(vcl[i]):
                        vcl[i] = _scaled_num(vcl[i], scale)
            _scale_point_list(v.get("entries_rel"), scale, seen=scaled_point_lists)
            _scale_point_list(v.get("exits_rel"), scale, seen=scaled_point_lists)
            _scale_point_list(v.get("body_polygon"), scale, seen=scaled_point_lists)
            _scale_point_list(v.get("clearance_polygon"), scale, seen=scaled_point_lists)


# ---------------------------------------------------------------------------
# apply_grid_scale  --  mutate env dict in-place
# ---------------------------------------------------------------------------

def apply_grid_scale(env: JsonDict, scale: int) -> None:
    """Divide all spatial fields by *scale* and set ``grid_size`` accordingly."""
    if scale == 1:
        env["grid"]["grid_size"] = 1.0
        return

    grid = env.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("env.grid must be an object")

    # -- grid dimensions --
    if "width" in grid:
        grid["width"] = int(_scaled_num(grid["width"], scale))
    if "height" in grid:
        grid["height"] = int(_scaled_num(grid["height"], scale))
    grid["grid_size"] = float(scale)

    # -- groups --
    groups = env.get("groups", {})
    if isinstance(groups, dict):
        for g in groups.values():
            if isinstance(g, dict):
                _scale_group_dict(g, scale)

    # -- zones --
    zones = env.get("zones")
    if isinstance(zones, dict):
        forbidden = zones.get("forbidden")
        if isinstance(forbidden, list):
            for area in forbidden:
                _scale_area(area, scale)
        constraints = zones.get("constraints")
        if isinstance(constraints, dict):
            for cfg in constraints.values():
                if not isinstance(cfg, dict):
                    continue
                areas = cfg.get("areas")
                if isinstance(areas, list):
                    for area in areas:
                        _scale_area(area, scale)

    # -- reset / initial_placements --
    reset = env.get("reset")
    if isinstance(reset, dict):
        ip = reset.get("initial_placements")
        if isinstance(ip, dict):
            for gid, pose in ip.items():
                if (isinstance(pose, list) and len(pose) >= 2
                        and _is_number(pose[0]) and _is_number(pose[1])):
                    pose[0] = _scaled_num(pose[0], scale)
                    pose[1] = _scaled_num(pose[1], scale)
                    ip[gid] = pose


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def autoscale_layout_env(
    env: JsonDict,
    *,
    base_grid_size: int,
    auto_scale_up: bool,
) -> Dict[str, Any]:
    values = collect_scale_basis_values(env)
    max_scale = max_lossless_integer_scale(values)
    chosen_scale = choose_grid_size(
        base_grid_size=base_grid_size,
        max_scale=max_scale,
        auto_scale_up=auto_scale_up,
    )
    if not is_lossless_for_scale(values, chosen_scale):
        raise ValueError(
            "requested grid_size cannot be applied without loss: "
            f"base={base_grid_size}, chosen={chosen_scale}, max_lossless={max_scale}"
        )
    apply_grid_scale(env, chosen_scale)
    return {
        "base_grid_size": int(base_grid_size),
        "auto_scale_up": bool(auto_scale_up),
        "max_lossless_grid_size": int(max_scale),
        "applied_grid_size": int(chosen_scale),
    }


__all__ = [
    "autoscale_layout_env",
    "collect_scale_basis_values",
    "max_lossless_integer_scale",
    "choose_grid_size",
    "is_lossless_for_scale",
    "apply_grid_scale",
]
