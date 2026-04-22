"""Custom preprocess JSON -> group_placement env JSON.

Input schema (new, SMA-independent)::

    {
      "grid": { "width": 300, "height": 200, "grid_size": 1.0, "auto_scale_up": true },
      "env": {"max_candidates": 70},
      "groups": [
        {
          "id": "G1",
          "facility_count": 8,
          "unit_width": 10,
          "unit_height": 5,
          "clearance_lrtb": [1, 1, 1, 1],
          "variant_layouts": [[8, 1], [4, 2], "3x3"],
          "allow_irregular_on_remainder": true
        }
      ],
      "flow": [],
      "zones": {"constraints": {}},
      "reset": {},
      "facilities": {
        "obstacles": []
      }
    }

``auto_scale_up=true``이면 group/zone/grid 좌표를 기준으로 무손실 최대 정수 스케일을
자동 적용합니다. 예: 모든 길이가 100의 배수면 ``grid_size=100``으로 자동 상향.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .grid_scaler import autoscale_layout_env
except ImportError:  # pragma: no cover - allows direct script execution
    from grid_scaler import autoscale_layout_env

JsonDict = Dict[str, Any]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> JsonDict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("input JSON root must be an object")
    return data


def save_json(data: JsonDict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _as_int(v: Any, name: str) -> int:
    try:
        out = int(v)
    except Exception as e:
        raise ValueError(f"{name} must be an integer, got {v!r}") from e
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {out}")
    return out


def _as_float(v: Any, name: str) -> float:
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"{name} must be a number, got {v!r}") from e


def _as_positive_intlike(v: Any, name: str) -> int:
    f = _as_float(v, name)
    i = int(round(f))
    if abs(f - float(i)) > _EPS:
        raise ValueError(
            f"{name} must be an integer-like value (e.g., 1.0, 10.0), got {v!r}"
        )
    if i <= 0:
        raise ValueError(f"{name} must be > 0, got {i}")
    return i


def _parse_clearance(g: JsonDict) -> List[int]:
    if "clearance_lrtb" in g:
        raw = g["clearance_lrtb"]
        if not (isinstance(raw, list) and len(raw) == 4):
            raise ValueError("group.clearance_lrtb must be [left,right,bottom,top]")
        return [
            int(round(_as_float(raw[i], f"clearance_lrtb[{i}]")))
            for i in range(4)
        ]
    if "clearance" in g:
        n = int(round(_as_float(g["clearance"], "clearance")))
        return [n, n, n, n]
    return [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Layout pair helpers
# ---------------------------------------------------------------------------

def _normalize_layout_pair(item: Any) -> Tuple[int, int]:
    if isinstance(item, str):
        s = item.lower().replace(" ", "")
        if "x" not in s:
            raise ValueError(f"invalid layout pair string: {item!r} (expected '8x1')")
        a, b = s.split("x", 1)
        cols, rows = int(a), int(b)
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        cols, rows = int(item[0]), int(item[1])
    else:
        raise ValueError(
            f"invalid layout pair: {item!r} (expected [cols,rows] or 'CxR')"
        )
    if cols <= 0 or rows <= 0:
        raise ValueError(f"layout pair must be positive, got cols={cols}, rows={rows}")
    return cols, rows


def _default_layout_pairs(facility_count: int) -> List[Tuple[int, int]]:
    # Exact divisors only: 8 -> (8,1), (4,2)
    out: List[Tuple[int, int]] = []
    for r in range(1, int(math.sqrt(facility_count)) + 1):
        if facility_count % r != 0:
            continue
        c = facility_count // r
        if c >= r:
            out.append((c, r))
    out.sort(key=lambda x: (x[1], -x[0]))
    return out or [(facility_count, 1)]


def _ordered_layout_pairs(g: JsonDict, facility_count: int) -> List[Tuple[int, int]]:
    raw = g.get("variant_layouts")
    if raw is None:
        return _default_layout_pairs(facility_count)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(
            "group.variant_layouts must be a non-empty list when provided"
        )
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for item in raw:
        pair = _normalize_layout_pair(item)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_occupied_cells(
    count: int, cols: int, rows: int,
) -> List[Tuple[int, int]]:
    slots = cols * rows
    if count > slots:
        raise ValueError(
            f"facility_count={count} exceeds layout slots={slots} "
            f"for layout {cols}x{rows}"
        )
    cells: List[Tuple[int, int]] = []
    for idx in range(count):
        r = idx // cols
        c = idx % cols
        cells.append((c, r))
    return cells


def _toggle_edge(
    edges: set[Tuple[Tuple[int, int], Tuple[int, int]]],
    p0: Tuple[int, int],
    p1: Tuple[int, int],
) -> None:
    key = (p0, p1) if p0 <= p1 else (p1, p0)
    if key in edges:
        edges.remove(key)
    else:
        edges.add(key)


def _simplify_orthogonal_polygon(
    points: List[Tuple[float, float]],
) -> List[List[float]]:
    if len(points) <= 3:
        return [[float(x), float(y)] for x, y in points]
    simp: List[Tuple[float, float]] = []
    n = len(points)
    for i in range(n):
        px, py = points[(i - 1) % n]
        cx, cy = points[i]
        nx, ny = points[(i + 1) % n]
        collinear = (px == cx == nx) or (py == cy == ny)
        if not collinear:
            simp.append((cx, cy))
    return [[float(x), float(y)] for x, y in simp]


def _cells_to_polygon(
    cells: List[Tuple[int, int]],
    unit_w: float,
    unit_h: float,
) -> List[List[float]]:
    # Build outer boundary by cancelling shared edges on a unit grid.
    edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for c, r in cells:
        p00 = (c, r)
        p10 = (c + 1, r)
        p11 = (c + 1, r + 1)
        p01 = (c, r + 1)
        _toggle_edge(edges, p00, p10)
        _toggle_edge(edges, p10, p11)
        _toggle_edge(edges, p11, p01)
        _toggle_edge(edges, p01, p00)

    if not edges:
        raise ValueError("cannot build polygon from empty cells")

    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    bad_nodes = [k for k, v in adj.items() if len(v) != 2]
    if bad_nodes:
        raise ValueError(
            "invalid boundary graph while building irregular polygon"
        )

    start = min(adj.keys(), key=lambda p: (p[1], p[0]))
    nbrs = sorted(adj[start], key=lambda p: (p[1], p[0]))

    cur = start
    nxt = nbrs[-1]
    prev = None
    ordered: List[Tuple[int, int]] = [start]

    while True:
        ordered.append(nxt)
        prev, cur = cur, nxt
        if cur == start:
            break
        n0, n1 = adj[cur]
        nxt = n0 if n1 == prev else n1

    # Remove duplicated closing point and scale to world coords.
    if ordered[-1] == ordered[0]:
        ordered = ordered[:-1]

    scaled = [(x * unit_w, y * unit_h) for x, y in ordered]
    return _simplify_orthogonal_polygon(scaled)


def _slot_pitch(unit_size: float, clear_a: int, clear_b: int) -> float:
    # Keep at least one clearance distance between neighboring facilities.
    gap = float(max(0, int(clear_a), int(clear_b)))
    return float(unit_size) + gap


def _layout_extent(count: int, unit_size: float, pitch: float) -> float:
    if count <= 0:
        return 0.0
    return float(unit_size) + float(count - 1) * float(pitch)


def _ports_for_layout(
    unit_w: float,
    unit_h: float,
    cols: int,
    count: int,
    pitch_x: float,
    pitch_y: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    last = count - 1
    last_row = last // cols
    last_col = last % cols
    entries = [[0.0, unit_h / 2.0]]
    exits = [
        [
            float(last_col) * pitch_x + float(unit_w),
            float(last_row) * pitch_y + (float(unit_h) / 2.0),
        ]
    ]
    return entries, exits


# ---------------------------------------------------------------------------
# Facility / layout registry builder
# ---------------------------------------------------------------------------

def _build_facility_registry_for_group(
    *,
    gid: str,
    facility_count: int,
    unit_w: float,
    unit_h: float,
    clearance_lrtb: List[int],
    layout_pairs: List[Tuple[int, int]],
) -> Tuple[Dict[str, JsonDict], Dict[str, JsonDict], List[str]]:
    facilities: Dict[str, JsonDict] = {}
    layouts: Dict[str, JsonDict] = {}
    refs: List[str] = []

    fids = [f"{gid}__f{i + 1:03d}" for i in range(facility_count)]
    for fid in fids:
        facilities[fid] = {
            "width": float(unit_w),
            "height": float(unit_h),
            "entries_rel": [[float(unit_w) / 2.0, 0.0]],
            "exits_rel": [[float(unit_w) / 2.0, float(unit_h)]],
        }

    pitch_x = _slot_pitch(unit_w, clearance_lrtb[0], clearance_lrtb[1])
    pitch_y = _slot_pitch(unit_h, clearance_lrtb[2], clearance_lrtb[3])

    for i, (cols, rows) in enumerate(layout_pairs):
        layout_ref = f"L_{gid}_{cols}x{rows}_{i + 1}"
        refs.append(layout_ref)

        slots: List[JsonDict] = []
        for k in range(facility_count):
            r = k // cols
            c = k % cols
            slots.append(
                {
                    "fid": fids[k],
                    "x": float(c) * float(pitch_x),
                    "y": float(r) * float(pitch_y),
                    "rotation": 0,
                    "mirror": False,
                }
            )
        layouts[layout_ref] = {"slots": slots}

    return facilities, layouts, refs


# ---------------------------------------------------------------------------
# Group builder
# ---------------------------------------------------------------------------

def _build_group(
    g: JsonDict,
) -> Tuple[str, JsonDict, Dict[str, JsonDict], Dict[str, JsonDict]]:
    gid = str(g.get("id", "")).strip()
    if not gid:
        raise ValueError("each group must have non-empty string id")

    count = _as_int(g.get("facility_count"), f"groups[{gid}].facility_count")
    unit_w = _as_float(g.get("unit_width"), f"groups[{gid}].unit_width")
    unit_h = _as_float(g.get("unit_height"), f"groups[{gid}].unit_height")
    if unit_w <= 0 or unit_h <= 0:
        raise ValueError(f"groups[{gid}].unit_width/unit_height must be > 0")

    clearance_lrtb = _parse_clearance(g)
    rotatable = bool(g.get("rotatable", True))
    mirrorable = bool(g.get("mirrorable", False))
    allow_irregular = bool(g.get("allow_irregular_on_remainder", False))
    zone_values = (
        dict(g.get("zone_values", {}))
        if isinstance(g.get("zone_values", {}), dict)
        else {}
    )

    layout_pairs = _ordered_layout_pairs(g, count)

    facilities, layouts, layout_refs = _build_facility_registry_for_group(
        gid=gid,
        facility_count=count,
        unit_w=unit_w,
        unit_h=unit_h,
        clearance_lrtb=clearance_lrtb,
        layout_pairs=layout_pairs,
    )

    variants: List[JsonDict] = []
    any_remainder = False

    pitch_x = _slot_pitch(unit_w, clearance_lrtb[0], clearance_lrtb[1])
    pitch_y = _slot_pitch(unit_h, clearance_lrtb[2], clearance_lrtb[3])

    layout_meta: List[Tuple[int, int, int, float, float]] = []

    for i, (cols, rows) in enumerate(layout_pairs):
        slots = cols * rows
        remainder = slots - count
        if remainder < 0:
            raise ValueError(
                f"groups[{gid}] layout {cols}x{rows} has insufficient slots "
                f"({slots} < {count})"
            )

        width = _layout_extent(cols, unit_w, pitch_x)
        height = _layout_extent(rows, unit_h, pitch_y)
        entries, exits = _ports_for_layout(
            unit_w, unit_h, cols, count, pitch_x, pitch_y,
        )

        v: JsonDict = {
            "width": width,
            "height": height,
            "entries_rel": entries,
            "exits_rel": exits,
            "rotatable": rotatable,
            "mirrorable": mirrorable,
            "layout_ref": layout_refs[i],
            "_layout": f"{cols}x{rows}",
            "_empty_slots": int(remainder),
        }

        if remainder > 0:
            any_remainder = True
            if allow_irregular:
                cells = _build_occupied_cells(count, cols, rows)
                v["body_polygon"] = _cells_to_polygon(
                    cells, unit_w=pitch_x, unit_h=pitch_y,
                )

        variants.append(v)
        layout_meta.append((cols, rows, remainder, width, height))

    group_type = "irregular" if (allow_irregular and any_remainder) else "rect"

    if group_type == "irregular":
        # env_loader for irregular groups requires body_polygon in every variant.
        for i, v in enumerate(variants):
            if "body_polygon" in v:
                continue
            _cols, _rows, _rem, width, height = layout_meta[i]
            v["body_polygon"] = [
                [0.0, 0.0],
                [width, 0.0],
                [width, height],
                [0.0, height],
            ]

    out: JsonDict = {
        "type": group_type,
        "clearance_lrtb": clearance_lrtb,
        "rotatable": rotatable,
        "mirrorable": mirrorable,
        "variants": variants,
        "_facility_count": count,
        "_unit_width": unit_w,
        "_unit_height": unit_h,
    }

    if zone_values:
        out["zone_values"] = zone_values

    if group_type == "irregular":
        # Canonical polygon required if variants are later removed/edited.
        first_poly = variants[0].get("body_polygon")
        if first_poly is None:
            cols, rows = layout_pairs[0]
            width = _layout_extent(cols, unit_w, pitch_x)
            height = _layout_extent(rows, unit_h, pitch_y)
            # full rectangle polygon in canonical orientation
            first_poly = [
                [0.0, 0.0],
                [width, 0.0],
                [width, height],
                [0.0, height],
            ]
        out["body_polygon"] = first_poly

    return gid, out, facilities, layouts


# ---------------------------------------------------------------------------
# Top-level converter
# ---------------------------------------------------------------------------

def convert_to_env(
    input_path: str,
    output_path: str,
    visualize_dir: str | Path | None = None,
) -> JsonDict:
    spec = load_json(input_path)

    groups_in = spec.get("groups")
    if not isinstance(groups_in, list) or len(groups_in) == 0:
        raise ValueError("input.groups must be a non-empty list")

    groups_out: Dict[str, JsonDict] = {}
    facilities_out: Dict[str, JsonDict] = {}
    layouts_out: Dict[str, JsonDict] = {}

    for item in groups_in:
        if not isinstance(item, dict):
            raise ValueError("each item in input.groups must be an object")
        gid, gcfg, fac_map, lay_map = _build_group(item)
        if gid in groups_out:
            raise ValueError(f"duplicate group id: {gid}")
        groups_out[gid] = gcfg
        for fid, fcfg in fac_map.items():
            if fid in facilities_out:
                raise ValueError(f"duplicate generated facility id: {fid}")
            facilities_out[fid] = fcfg
        for lid, lcfg in lay_map.items():
            if lid in layouts_out:
                raise ValueError(f"duplicate generated layout_ref: {lid}")
            layouts_out[lid] = lcfg

    grid_in = spec.get("grid", {})
    if not isinstance(grid_in, dict):
        raise ValueError("input.grid must be an object")
    grid_w = _as_int(grid_in.get("width", 1), "grid.width")
    grid_h = _as_int(grid_in.get("height", 1), "grid.height")
    base_grid_size = _as_positive_intlike(
        grid_in.get("grid_size", 1.0), "grid.grid_size",
    )
    auto_scale_up = bool(grid_in.get("auto_scale_up", True))

    env_in = spec.get("env", {})
    if not isinstance(env_in, dict):
        raise ValueError("input.env must be an object")

    facilities_in = spec.get("facilities", {})
    if not isinstance(facilities_in, dict):
        raise ValueError("input.facilities must be an object")
    obstacles_in = facilities_in.get("obstacles", [])
    if not isinstance(obstacles_in, list):
        raise ValueError("input.facilities.obstacles must be a list")

    out: JsonDict = {
        "grid": {
            "width": grid_w,
            "height": grid_h,
            "grid_size": 1.0,
        },
        "env": dict(env_in),
        "groups": groups_out,
        "flow": spec.get("flow", []),
        "zones": spec.get("zones", {"constraints": {}}),
        "reset": spec.get("reset", {}),
        "facilities": {
            "obstacles": list(obstacles_in),
            "facilities": facilities_out,
            "layouts": layouts_out,
        },
    }

    scaling_meta = autoscale_layout_env(
        out,
        base_grid_size=base_grid_size,
        auto_scale_up=auto_scale_up,
    )
    out["_grid_scaling"] = scaling_meta

    save_json(out, output_path)

    if visualize_dir is not None:
        try:
            from .visualize import save_variant_images
        except ImportError:  # pragma: no cover - allows direct script execution
            from visualize import save_variant_images
        save_variant_images(out, visualize_dir)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert custom preprocess JSON into group_placement env JSON",
    )
    parser.add_argument("input", help="Input JSON path")
    parser.add_argument("output", help="Output env JSON path")
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="Optional output directory for per-group variant PNG files",
    )
    args = parser.parse_args()
    convert_to_env(
        args.input,
        args.output,
        visualize_dir=args.visualize_dir,
    )


if __name__ == "__main__":
    main()
