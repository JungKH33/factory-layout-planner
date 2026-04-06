"""Phase 2 resolver — unfold facilities from cluster placements.

Consumes ONLY the dict produced by ``envs.export.export_group_placement``; has
zero imports from ``envs`` / ``agents`` / ``search``.  Deterministic geometry
unfold — no search, no placeability re-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schema import FacilityPlacement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers — match envs/placement/static.py::StaticSpec._rotate_point
# ---------------------------------------------------------------------------

def _norm_rotation(rotation: int) -> int:
    r = int(rotation) % 360
    if r % 90 != 0:
        raise ValueError(f"rotation must be a multiple of 90, got {rotation!r}")
    return r


def _rotate_offset(dx: float, dy: float, rotation: int) -> Tuple[float, float]:
    """Rotate a center-relative offset.  Same convention as ``_rotate_point``
    in ``envs.placement.static``: 90° maps ``(dx, dy) -> (dy, -dx)``.
    """
    r = _norm_rotation(rotation)
    if r == 0:
        return (float(dx), float(dy))
    if r == 90:
        return (float(dy), -float(dx))
    if r == 180:
        return (-float(dx), -float(dy))
    # 270
    return (-float(dy), float(dx))


# ---------------------------------------------------------------------------
# Slot unfolding
# ---------------------------------------------------------------------------

def _unfold_slot(
    *,
    gid: str,
    slot: Mapping[str, Any],
    facility: Mapping[str, Any],
    cluster_cx_mm: float,
    cluster_cy_mm: float,
    cluster_w_src_mm: float,
    cluster_h_src_mm: float,
    cluster_rotation: int,
    cluster_mirror: bool,
) -> FacilityPlacement:
    """Unfold one slot into a world-frame ``FacilityPlacement``.

    All coordinates are in mm.

    Transform chain (per point):
      1. Point in facility BL frame → offset from facility center.
      2. Optional slot mirror, then slot rotation around facility center.
      3. Translate by facility BL in cluster *source* frame (given by slot.x/y).
      4. Offset from cluster source center → optional cluster mirror → cluster rotation.
      5. Translate by cluster world center → world mm.
    """
    fw = float(facility["width"])
    fh = float(facility["height"])

    slot_rotation = _norm_rotation(int(slot.get("rotation", 0)))
    slot_mirror = bool(slot.get("mirror", False))

    # Facility bbox dims after slot rotation.
    if slot_rotation in (90, 270):
        fw_slot, fh_slot = fh, fw
    else:
        fw_slot, fh_slot = fw, fh

    # Facility BL / center in cluster source frame (from slot definition).
    fbl_src_x = float(slot["x"])
    fbl_src_y = float(slot["y"])
    fc_src_x = fbl_src_x + fw_slot / 2.0
    fc_src_y = fbl_src_y + fh_slot / 2.0

    def _cluster_src_to_world(sx: float, sy: float) -> Tuple[float, float]:
        """Apply cluster-level mirror + rotation around cluster center."""
        dx = sx - cluster_w_src_mm / 2.0
        dy = sy - cluster_h_src_mm / 2.0
        if cluster_mirror:
            dx = -dx
        rdx, rdy = _rotate_offset(dx, dy, cluster_rotation)
        return (cluster_cx_mm + rdx, cluster_cy_mm + rdy)

    def _transform_port(px_bl: float, py_bl: float) -> Tuple[float, float]:
        """Facility BL-frame port → world mm."""
        pdx = px_bl - fw / 2.0
        pdy = py_bl - fh / 2.0
        if slot_mirror:
            pdx = -pdx
        rpdx, rpdy = _rotate_offset(pdx, pdy, slot_rotation)
        # Now in facility-center frame in the cluster source coordinate system.
        src_x = fc_src_x + rpdx
        src_y = fc_src_y + rpdy
        return _cluster_src_to_world(src_x, src_y)

    # World-frame facility center (from source-frame center).
    fc_world_x, fc_world_y = _cluster_src_to_world(fc_src_x, fc_src_y)

    # Total rotation after combining slot + cluster.  Determines the world bbox dims.
    total_rotation = (slot_rotation + cluster_rotation) % 360
    if total_rotation in (90, 270):
        fw_world, fh_world = fh, fw
    else:
        fw_world, fh_world = fw, fh

    fbl_world_x = fc_world_x - fw_world / 2.0
    fbl_world_y = fc_world_y - fh_world / 2.0

    entries_raw = facility.get("entries_rel", []) or []
    exits_raw = facility.get("exits_rel", []) or []
    entries_abs = tuple(_transform_port(float(p[0]), float(p[1])) for p in entries_raw)
    exits_abs = tuple(_transform_port(float(p[0]), float(p[1])) for p in exits_raw)

    return FacilityPlacement(
        gid=str(gid),
        fid=str(slot["fid"]),
        x_mm=float(fbl_world_x),
        y_mm=float(fbl_world_y),
        width_mm=float(fw_world),
        height_mm=float(fh_world),
        rotation=int(total_rotation),
        entry_points_abs_mm=entries_abs,
        exit_points_abs_mm=exits_abs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_ON_MISSING = {"warn", "silent", "error"}


def _handle_missing(on_missing: str, msg: str) -> None:
    if on_missing == "silent":
        return
    if on_missing == "warn":
        logger.warning(msg)
        return
    if on_missing == "error":
        raise KeyError(msg)
    raise ValueError(
        f"on_missing must be one of {sorted(_VALID_ON_MISSING)}, got {on_missing!r}"
    )


def resolve_facilities(
    state_dict: Mapping[str, Any],
    *,
    on_missing: str = "warn",
) -> Dict[str, List[FacilityPlacement]]:
    """Unfold cluster placements into per-cluster facility lists.

    Args:
        state_dict: Dict from ``envs.export.export_group_placement``.  Required
            keys: ``placements``, ``facilities``, ``layouts``.
        on_missing: ``"warn"`` (default), ``"silent"``, or ``"error"`` — how
            to report missing ``layout_ref`` or unknown ``fid`` references.

    Returns:
        Mapping from cluster ``gid`` to a list of ``FacilityPlacement``
        instances in world mm coordinates.  Clusters without a resolvable
        layout are represented by an empty list.
    """
    if on_missing not in _VALID_ON_MISSING:
        raise ValueError(
            f"on_missing must be one of {sorted(_VALID_ON_MISSING)}, got {on_missing!r}"
        )

    placements: Iterable[Mapping[str, Any]] = state_dict.get("placements", []) or []
    facilities: Mapping[str, Mapping[str, Any]] = state_dict.get("facilities", {}) or {}
    layouts: Mapping[str, Mapping[str, Any]] = state_dict.get("layouts", {}) or {}

    out: Dict[str, List[FacilityPlacement]] = {}

    for entry in placements:
        gid = str(entry.get("gid"))
        out.setdefault(gid, [])

        layout_ref: Optional[str] = entry.get("layout_ref")
        if layout_ref is None:
            _handle_missing(
                on_missing,
                f"[facility_placement] gid={gid!r} has no layout_ref; skipping",
            )
            continue

        layout = layouts.get(layout_ref)
        if layout is None:
            _handle_missing(
                on_missing,
                f"[facility_placement] gid={gid!r} references unknown layout_ref={layout_ref!r}; skipping",
            )
            continue

        slots = layout.get("slots", []) or []
        rotation = _norm_rotation(int(entry.get("rotation", 0)))
        mirror = bool(entry.get("mirror", False))
        cluster_w_mm = float(entry.get("cluster_w_mm", 0.0))
        cluster_h_mm = float(entry.get("cluster_h_mm", 0.0))
        x_bl_mm = float(entry.get("x_bl_mm", 0.0))
        y_bl_mm = float(entry.get("y_bl_mm", 0.0))

        # Cluster center is invariant under rotation.
        cluster_cx_mm = x_bl_mm + cluster_w_mm / 2.0
        cluster_cy_mm = y_bl_mm + cluster_h_mm / 2.0

        # Source-frame cluster dims: if cluster is rotated 90/270, source dims
        # are the swapped world dims.
        if rotation in (90, 270):
            cluster_w_src_mm = cluster_h_mm
            cluster_h_src_mm = cluster_w_mm
        else:
            cluster_w_src_mm = cluster_w_mm
            cluster_h_src_mm = cluster_h_mm

        for slot in slots:
            fid = str(slot.get("fid"))
            fac = facilities.get(fid)
            if fac is None:
                _handle_missing(
                    on_missing,
                    f"[facility_placement] gid={gid!r} layout={layout_ref!r} slot references unknown fid={fid!r}; skipping",
                )
                continue

            fp = _unfold_slot(
                gid=gid,
                slot=slot,
                facility=fac,
                cluster_cx_mm=cluster_cx_mm,
                cluster_cy_mm=cluster_cy_mm,
                cluster_w_src_mm=cluster_w_src_mm,
                cluster_h_src_mm=cluster_h_src_mm,
                cluster_rotation=rotation,
                cluster_mirror=mirror,
            )
            out[gid].append(fp)

    return out
