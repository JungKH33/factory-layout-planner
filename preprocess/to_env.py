"""SMA JSON -> Factory Layout Env JSON converter.

Usage:
    python -m converters.sma_to_env input.json output.json --grid-size 100
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_facility_dict(facilities: List[Dict]) -> Dict[str, Dict]:
    """facility id -> facility data mapping."""
    return {f["id"]: f for f in facilities}


def get_main_facility(
    group_id: str,
    facility_ids: List[str],
    facility_dict: Dict[str, Dict],
) -> Dict:
    """Get representative facility for a group.
    
    Priority:
    1. Facility with mainFacility=true
    2. First facility in list
    """
    for fid in facility_ids:
        fac = facility_dict.get(fid)
        if fac and fac.get("mainFacility", False):
            return fac
    
    first_id = facility_ids[0]
    fac = facility_dict.get(first_id)
    if fac is None:
        raise ValueError(f"Facility {first_id} not found for group {group_id}")
    return fac


def extract_clearance_from_variable_area(fac: Dict) -> Tuple[float, float, float, float]:
    """Extract clearance (UL, UR, UB, UA) from areaMap.VARIABLE_AREA.
    
    Based on notebook logic (lines 137-191):
    - UL: left clearance (negative x extension)
    - UR: right clearance (positive x extension beyond width)
    - UB: bottom clearance (negative y extension)
    - UA: top clearance (positive y extension beyond height)
    
    Returns: (UL, UR, UB, UA) in mm
    """
    area_map = fac.get("areaMap", {})
    variable_areas = area_map.get("VARIABLE_AREA", [])
    
    if not variable_areas:
        return (0.0, 0.0, 0.0, 0.0)
    
    fac_width = fac["width"]
    fac_height = fac["height"]
    
    # UL: min x (negative means left extension)
    min_x = float("inf")
    for va in variable_areas:
        # Notebook parity: only consider VARIABLE_AREA in (x<=0, y<=0) quadrant.
        # (Notebook condition was: -x>=0 and -y>=0)
        if va.get("x", 0) <= 0 and va.get("y", 0) <= 0:
            min_x = min(min_x, va.get("x", 0))
    UL = -min_x if min_x != float("inf") and min_x < 0 else 0.0
    
    # UR: max (x + width) beyond facility width
    max_x_right = -float("inf")
    for va in variable_areas:
        if va.get("x", 0) <= 0 and va.get("y", 0) <= 0:
            right_edge = va.get("x", 0) + va.get("width", 0)
            max_x_right = max(max_x_right, right_edge)
    UR = max(0.0, max_x_right - fac_width) if max_x_right != -float("inf") else 0.0
    
    # UB: min y (negative means bottom extension)
    min_y = float("inf")
    for va in variable_areas:
        if va.get("x", 0) <= 0 and va.get("y", 0) <= 0:
            min_y = min(min_y, va.get("y", 0))
    UB = -min_y if min_y != float("inf") and min_y < 0 else 0.0
    
    # UA: max (y + height) beyond facility height
    max_y_top = -float("inf")
    for va in variable_areas:
        if va.get("x", 0) <= 0 and va.get("y", 0) <= 0:
            top_edge = va.get("y", 0) + va.get("height", 0)
            max_y_top = max(max_y_top, top_edge)
    UA = max(0.0, max_y_top - fac_height) if max_y_top != -float("inf") else 0.0
    
    return (UL, UR, UB, UA)


def adjust_clearance_by_path(
    fac: Dict,
    UL: float, UR: float, UB: float, UA: float
) -> Tuple[float, float, float, float]:
    """Adjust clearance based on entrance/exit position and pathWidth.
    
    Based on notebook logic (lines 327-349):
    If entrance/exit is near edge, add pathWidth to clearance.
    """
    path_width = fac.get("pathWidth", 0)
    if path_width <= 0:
        return (UL, UR, UB, UA)
    
    width = fac["width"]
    height = fac["height"]
    ent_x = fac.get("entranceRelativeX", width / 2)
    ent_y = fac.get("entranceRelativeY", height / 2)
    exi_x = fac.get("exitRelativeX", width / 2)
    exi_y = fac.get("exitRelativeY", height / 2)
    
    # Right edge (>= 95% width)
    if ent_x >= 0.95 * width or exi_x >= 0.95 * width:
        UR = max(UR, path_width)
    # Left edge (<= 5% width)
    if ent_x <= 0.05 * width or exi_x <= 0.05 * width:
        UL = max(UL, path_width)
    # Top edge (>= 95% height)
    if ent_y >= 0.95 * height or exi_y >= 0.95 * height:
        UA = max(UA, path_width)
    # Bottom edge (<= 5% height)
    if ent_y <= 0.05 * height or exi_y <= 0.05 * height:
        UB = max(UB, path_width)
    
    return (UL, UR, UB, UA)


def calculate_cluster_clearance(
    n_cols: int,
    n_rows: int,
    UL: float, UR: float, UB: float, UA: float,
    fac_width: float,
    fac_height: float,
) -> Tuple[float, float, float, float]:
    """Calculate cluster clearance with buffer logic (노트북 로직).
    
    Based on notebook logic (lines 743-759):
    - buffer = min(1.0, (facility_count - 1) * 0.2)
    - 배열 방향에 따라 clearance 감소 적용
    
    Returns: (cluster_UL, cluster_UR, cluster_UB, cluster_UA)
    """
    facility_count = n_cols * n_rows
    buffer = min(1.0, (facility_count - 1) * 0.2)
    
    if fac_width < fac_height:  # 세로로 긴 경우 (vertical stacking)
        cluster_UL = round(buffer * UL)
        cluster_UR = round(buffer * UR)
        cluster_UB = UB
        cluster_UA = UA
    else:  # 가로로 긴 경우 (horizontal stacking)
        cluster_UL = UL
        cluster_UR = UR
        cluster_UB = round(buffer * UB)
        cluster_UA = round(buffer * UA)
    
    return (cluster_UL, cluster_UR, cluster_UB, cluster_UA)


def calculate_cluster_size_2d(
    fac_width: float,
    fac_height: float,
    facility_count: int,
    UL: float, UR: float, UB: float, UA: float,
    factory_width: float,
    factory_height: float,
    group_id: str = "",
) -> Tuple[float, float, int, int]:
    """Calculate cluster size with 2D arrangement to fit within factory.
    
    Based on notebook logic (lines 7525-7560):
    - Start with 1 row/column
    - If cluster exceeds factory, increase rows/columns
    - Recalculate until it fits or max iterations reached
    
    Returns: (cluster_width, cluster_height, n_cols, n_rows)
    """
    if facility_count <= 1:
        return (fac_width, fac_height, 1, 1)
    
    gap_x = max(UL, UR)
    gap_y = max(UB, UA)
    
    # Determine stacking direction based on facility shape
    # width >= height: prefer horizontal stacking (more columns)
    # width < height: prefer vertical stacking (more rows)
    stack_horizontal = fac_width >= fac_height
    
    # Start with 1 row/column arrangement
    if stack_horizontal:
        n_cols = facility_count
        n_rows = 1
    else:
        n_cols = 1
        n_rows = facility_count
    
    # Iteratively increase rows/cols until cluster fits
    # Check if cluster fits in at least ONE orientation (0° or 90°)
    max_iterations = 100
    for iteration in range(max_iterations):
        cluster_width = fac_width * n_cols + gap_x * max(0, n_cols - 1)
        cluster_height = fac_height * n_rows + gap_y * max(0, n_rows - 1)
        
        # Check if cluster fits in at least one orientation
        # Orientation 0°: width vs factory_width, height vs factory_height
        fits_0deg = (cluster_width <= factory_width * 0.95 and 
                     cluster_height <= factory_height * 0.95)
        # Orientation 90°: height vs factory_width, width vs factory_height
        fits_90deg = (cluster_height <= factory_width * 0.95 and 
                      cluster_width <= factory_height * 0.95)
        
        if fits_0deg or fits_90deg:
            break
        
        # Debug output for problematic cases
        if iteration < 5 or iteration % 10 == 0:
            logger.debug(
                "%s iter=%d: %dx%d, cluster=%.0fx%.0fmm, factory=%.0fx%.0fmm, fits_0=%s, fits_90=%s",
                group_id,
                iteration,
                n_cols,
                n_rows,
                cluster_width,
                cluster_height,
                factory_width,
                factory_height,
                fits_0deg,
                fits_90deg,
            )
        
        # Increase rows/cols to make cluster smaller
        # Try to balance the dimensions
        if cluster_width > cluster_height:
            # Width is larger: need more rows to reduce columns
            n_rows += 1
            n_cols = math.ceil(facility_count / n_rows)
        else:
            # Height is larger: need more columns to reduce rows
            n_cols += 1
            n_rows = math.ceil(facility_count / n_cols)
        
        # Safety check
        if n_rows > 50 or n_cols > 50:
            logger.warning("%s arrangement exceeded 50 rows/cols: %dx%d", group_id, n_cols, n_rows)
            break
    
    # Final calculation
    cluster_width = fac_width * n_cols + gap_x * max(0, n_cols - 1)
    cluster_height = fac_height * n_rows + gap_y * max(0, n_rows - 1)
    
    # Final warning only if doesn't fit in ANY orientation
    fits_0 = cluster_width <= factory_width and cluster_height <= factory_height
    fits_90 = cluster_height <= factory_width and cluster_width <= factory_height
    if not fits_0 and not fits_90:
        logger.warning(
            "%s cluster %.0fx%.0fmm doesn't fit in factory %.0fx%.0fmm in any orientation",
            group_id,
            cluster_width,
            cluster_height,
            factory_width,
            factory_height,
        )
    
    return (cluster_width, cluster_height, n_cols, n_rows)


def convert_io_to_relative(
    entrance_x: float,
    entrance_y: float,
    exit_x: float,
    exit_y: float,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Pass through BL-relative coords as-is.
    
    Source: entranceRelativeX/Y = position from facility bottom-left (0,0)
    Target: ent_rel_x/y = same (BL-relative). StaticSpec expects BL-relative.
    """
    return entrance_x, entrance_y, exit_x, exit_y


def build_groups(
    facility_groups: List[Dict],
    facility_dict: Dict[str, Dict],
    scale: float,
    factory_width: float,
    factory_height: float,
) -> Dict[str, Dict]:
    """Convert facilityGroups to env groups."""
    groups = {}
    
    for fg in facility_groups:
        group_id = fg["groupId"]
        facility_ids = fg["facilities"]
        facility_count = len(facility_ids)
        
        if not facility_ids:
            logger.warning("Group %s has no facilities, skipping", group_id)
            continue
        
        # Select representative facility (first one)
        main_fac = get_main_facility(group_id, facility_ids, facility_dict)
        
        # Basic properties (individual facility)
        fac_width = main_fac["width"]
        fac_height = main_fac["height"]
        
        # Check if this is a dock or storage facility (no clearance)
        is_dock = (main_fac.get("processGroup") == "dock" or 
                   main_fac.get("type") in ("dockIn", "dockOut"))
        is_storage = main_fac.get("type") == "storage"
        
        # Extract clearance from VARIABLE_AREA (dock/storage has 0 clearance)
        if is_dock or is_storage:
            UL, UR, UB, UA = 0, 0, 0, 0
        else:
            UL, UR, UB, UA = extract_clearance_from_variable_area(main_fac)
            # Adjust clearance by pathWidth
            UL, UR, UB, UA = adjust_clearance_by_path(main_fac, UL, UR, UB, UA)
        
        # Calculate cluster size with 2D arrangement
        cluster_width, cluster_height, n_cols, n_rows = calculate_cluster_size_2d(
            fac_width, fac_height, facility_count, UL, UR, UB, UA,
            factory_width, factory_height, group_id
        )
        
        # Check if cluster can rotate (fits in both orientations)
        fits_0deg = (cluster_width <= factory_width * 0.95 and 
                     cluster_height <= factory_height * 0.95)
        fits_90deg = (cluster_height <= factory_width * 0.95 and 
                      cluster_width <= factory_height * 0.95)
        can_rotate = fits_0deg and fits_90deg
        
        # If only 90° fits, swap width/height so stored dimensions work without rotation
        if fits_90deg and not fits_0deg:
            cluster_width, cluster_height = cluster_height, cluster_width
            # Also swap clearances for 90° CCW rotation:
            # new_L=old_T, new_R=old_B, new_B=old_L, new_T=old_R
            UL, UR, UB, UA = UA, UB, UL, UR
            logger.info("%s swapped to 90° orientation (only fits rotated)", group_id)
        elif not can_rotate and facility_count > 1:
            logger.info("%s rotatable=False (only fits in one orientation)", group_id)
        
        # Scale to grid units
        width = cluster_width / scale
        height = cluster_height / scale
        
        # IO coordinate conversion (based on individual facility, center-relative)
        ent_rel_x, ent_rel_y, exi_rel_x, exi_rel_y = convert_io_to_relative(
            main_fac.get("entranceRelativeX", fac_width / 2),
            main_fac.get("entranceRelativeY", 0),
            main_fac.get("exitRelativeX", fac_width / 2),
            main_fac.get("exitRelativeY", fac_height),
            fac_width,
            fac_height,
        )
        
        # Clearance with buffer logic (노트북 방식)
        cluster_UL, cluster_UR, cluster_UB, cluster_UA = calculate_cluster_clearance(
            n_cols, n_rows, UL, UR, UB, UA, fac_width, fac_height
        )
        
        group_data = {
            "width": width,
            "height": height,
            "rotatable": can_rotate,
            "ent_rel_x": ent_rel_x / scale,
            "ent_rel_y": ent_rel_y / scale,
            "exi_rel_x": exi_rel_x / scale,
            "exi_rel_y": exi_rel_y / scale,
            # Clearance (scaled, with buffer applied)
            "clearance_lrtb": [
                int(round(cluster_UL / scale)),
                int(round(cluster_UR / scale)),
                int(round(cluster_UB / scale)),
                int(round(cluster_UA / scale)),
            ],
        }
        
        # Generic zone values.
        zone_values: Dict[str, Any] = {}
        # storage는 height 제약 없음 (배치 가능 영역 어디든 가능)
        if "facilityHeightDouble" in main_fac and not is_storage:
            zone_values["height"] = main_fac["facilityHeightDouble"]
        if "facilityWeightDouble" in main_fac:
            zone_values["weight"] = main_fac["facilityWeightDouble"]
        if main_fac.get("dry") is not None:
            zone_values["dry"] = main_fac["dry"]
        placeable = main_fac.get("placeableAreaIds", [])
        if isinstance(placeable, list) and len(placeable) > 0:
            # placeable 제약은 == 1 형태로 사용.
            zone_values["placeable"] = 1
        if zone_values:
            group_data["zone_values"] = zone_values
        
        # Meta info (for debugging, prefixed with _)
        group_data["_facility_count"] = facility_count
        group_data["_n_cols"] = n_cols
        group_data["_n_rows"] = n_rows
        group_data["_individual_width"] = fac_width / scale
        group_data["_individual_height"] = fac_height / scale
        group_data["_main_facility_id"] = main_fac["id"]
        group_data["_type"] = main_fac.get("type", "unknown")
        group_data["_processGroup"] = main_fac.get("processGroup", "unknown")
        group_data["_pathWidth"] = main_fac.get("pathWidth", 0) / scale
        group_data["_raw_clearance"] = {
            "UL": UL / scale,
            "UR": UR / scale,
            "UB": UB / scale,
            "UA": UA / scale,
        }
        
        groups[group_id] = group_data
    
    return groups


def build_flow(
    facilities: List[Dict],
    valid_group_ids: set,
) -> List[List]:
    """Convert nextFacilityGroup to flow.
    
    TODO: Review flow weight determination (currently all 1.0)
    """
    flow_set = set()
    
    for fac in facilities:
        src_group = fac.get("facilityGroup")
        if src_group not in valid_group_ids:
            continue
        
        for dst_group in fac.get("nextFacilityGroup", []):
            if dst_group not in valid_group_ids:
                logger.warning("Flow target %s not in valid groups, skipping", dst_group)
                continue
            if src_group != dst_group:
                flow_set.add((src_group, dst_group))
    
    flow = [[src, dst, 1.0] for src, dst in sorted(flow_set)]
    return flow


def convert_rect(rect: Dict, scale: float) -> List[int]:
    """Convert rect dict to [x0, y0, x1, y1]."""
    x0 = rect["x"] / scale
    y0 = rect["y"] / scale
    x1 = (rect["x"] + rect["width"]) / scale
    y1 = (rect["y"] + rect["height"]) / scale
    return [int(x0), int(y0), int(x1), int(y1)]


def build_zones(
    placeable_areas: List[Dict],
    area_ceilings: List[Dict],
    area_weights: List[Dict],
    area_dry: List[Dict],
    forbidden_areas: List[Dict],
    column_areas: List[Dict],
    scale: float,
    *,
    default_height: float | None = None,
    default_weight: float | None = None,
    default_dry: float | None = None,
    default_placeable: int = 0,
) -> Dict[str, Any]:
    """Convert area constraints to zones.constraints schema."""
    constraints: Dict[str, Any] = {}

    def _require_default(cname: str, dv: float | None) -> float:
        if dv is None:
            raise ValueError(
                f"{cname} constraint requires default value. "
                f"Set factoryDimensions.{cname} in source JSON."
            )
        return float(dv)

    # Placeable (== 1)
    placeable_constraint_areas = []
    for area in placeable_areas:
        placeable_constraint_areas.append({
            "rect": convert_rect(area, scale),
            "value": 1,
        })
    if placeable_constraint_areas:
        constraints["placeable"] = {
            "dtype": "int",
            "op": "==",
            "default": int(default_placeable),
            "areas": placeable_constraint_areas,
        }

    # Height (<=)
    height_constraint_areas = []
    for area in area_ceilings:
        height_constraint_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("ceilingHeight", area.get("value", 0)),
        })
    if default_height is not None or height_constraint_areas:
        height_default = _require_default("ceilingHeight", default_height)
        constraints["height"] = {
            "dtype": "float",
            "op": "<=",
            "default": height_default,
            "areas": height_constraint_areas,
        }

    # Weight (<=)
    weight_constraint_areas = []
    for area in area_weights:
        weight_constraint_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("weight", area.get("value", 0)),
        })
    if default_weight is not None or weight_constraint_areas:
        weight_default = _require_default("weight", default_weight)
        constraints["weight"] = {
            "dtype": "float",
            "op": "<=",
            "default": weight_default,
            "areas": weight_constraint_areas,
        }

    # Dry (>=)
    dry_constraint_areas = []
    for area in area_dry:
        dry_constraint_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("dry", area.get("value", 0)),
        })
    if default_dry is not None or dry_constraint_areas:
        dry_default = _require_default("dry", default_dry)
        constraints["dry"] = {
            "dtype": "float",
            "op": ">=",
            "default": dry_default,
            "areas": dry_constraint_areas,
        }
    
    # Forbidden areas (from forbiddenAreas + columnAreas)
    forbidden_list = []
    for area in forbidden_areas:
        forbidden_list.append({"rect": convert_rect(area, scale)})
    for area in column_areas:
        forbidden_list.append({"rect": convert_rect(area, scale)})
    
    zones: Dict[str, Any] = {"constraints": constraints}

    # if forbidden_list:
    #     zones["forbidden_areas"] = forbidden_list

    return zones


def build_initial_placements(
    fixed_positions: List[Dict],
    facility_dict: Dict[str, Dict],
    groups: Dict[str, Dict],
    scale: float,
) -> Dict[str, List[float]]:
    """Convert fixedFacilityPositions to initial_placements [x_center, y_center, variant_index]."""
    placements: Dict[str, List[float]] = {}

    # rotation degrees → variant_index mapping
    _rot_to_oi = {0: 0, 90: 1, 180: 2, 270: 3}

    for fp in fixed_positions:
        fac_id = fp["facilityId"]
        fac = facility_dict.get(fac_id)
        if fac is None:
            logger.warning("Fixed position for unknown facility %s, skipping", fac_id)
            continue

        group_id = fac.get("facilityGroup")
        if group_id is None:
            logger.warning("Facility %s has no facilityGroup, skipping", fac_id)
            continue

        grp = groups.get(group_id)
        if grp is None:
            logger.warning("Group %s not in groups dict, skipping %s", group_id, fac_id)
            continue

        pos = fp["position"]
        x_bl = float(pos["x"]) / scale
        y_bl = float(pos["y"]) / scale
        rot_deg = int(fp.get("rotation", 0))
        oi = _rot_to_oi.get(rot_deg, 0)

        w = float(grp.get("width", 1))
        h = float(grp.get("height", 1))
        # swap dimensions for 90/270
        if rot_deg in (90, 270):
            w, h = h, w
        x_center = x_bl + w / 2.0
        y_center = y_bl + h / 2.0

        if group_id not in placements:
            placements[group_id] = [x_center, y_center, oi]
        else:
            logger.info("Group %s already has initial placement, skipping %s", group_id, fac_id)

    return placements


def recalculate_group_size(
    group: Dict[str, Any],
    n_cols: int,
    n_rows: int,
    scale: float = 1.0,
) -> None:
    """rows/cols 변경 시 그룹 크기와 clearance 재계산 (in-place).
    
    calculate_cluster_clearance 재사용.
    """
    fac_w = group["_individual_width"] * scale  # mm 단위로 복원
    fac_h = group["_individual_height"] * scale
    raw_cl = group["_raw_clearance"]
    UL, UR = raw_cl["UL"] * scale, raw_cl["UR"] * scale
    UB, UA = raw_cl["UB"] * scale, raw_cl["UA"] * scale
    
    gap_x = max(UL, UR)
    gap_y = max(UB, UA)
    
    # 크기 재계산
    new_width = fac_w * n_cols + gap_x * max(0, n_cols - 1)
    new_height = fac_h * n_rows + gap_y * max(0, n_rows - 1)
    
    # clearance 재계산 (buffer 적용)
    cluster_UL, cluster_UR, cluster_UB, cluster_UA = calculate_cluster_clearance(
        n_cols, n_rows, UL, UR, UB, UA, fac_w, fac_h
    )
    
    old_size = f"{group['width']:.1f}x{group['height']:.1f}"
    new_size = f"{new_width / scale:.1f}x{new_height / scale:.1f}"
    
    # 업데이트
    group["width"] = new_width / scale
    group["height"] = new_height / scale
    group["_n_cols"] = n_cols
    group["_n_rows"] = n_rows
    group["clearance_lrtb"] = [
        int(round(cluster_UL / scale)),
        int(round(cluster_UR / scale)),
        int(round(cluster_UB / scale)),
        int(round(cluster_UA / scale)),
    ]
    
    return old_size, new_size


def validate_and_adjust_groups(
    env_data: Dict[str, Any],
    scale: float = 1.0,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """env로 로드 후 placeable 검사, 필요시 rows/cols 조정.

    Args:
        env_data: 생성된 env 데이터
        scale: mm per grid unit
        max_iterations: 최대 조정 반복 횟수

    Returns:
        조정된 env_data
    """
    import sys
    import tempfile
    from pathlib import Path
    # 프로젝트 루트를 Python path에 추가
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from envs.env_loader import load_env
    
    for iteration in range(max_iterations):
        # 임시 JSON 파일로 저장 후 env 로드
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(env_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            loaded = load_env(temp_path)
            env = loaded.env
            env.reset(options=loaded.reset_kwargs)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        # placeable 검사 (remaining = 아직 배치 안 된 그룹)
        problematic_groups = []
        state = env.get_state()
        for gid in env.get_state().remaining:
            g = env.group_specs[gid]
            combined = g.placeable_map(state=state, gid=gid)
            count = int(combined.sum().item())
            
            if count == 0:
                problematic_groups.append(gid)
        
        if not problematic_groups:
            if iteration > 0:
                logger.info("All groups have placeable positions after %d adjustments", iteration)
            break
        
        logger.warning(
            "Iteration %d: %d groups have no placeable position",
            iteration,
            len(problematic_groups),
        )
        
        # rows/cols 조정
        for gid in problematic_groups:
            group = env_data["groups"][gid]
            n_cols = group["_n_cols"]
            n_rows = group["_n_rows"]
            facility_count = group["_facility_count"]
            
            # 더 긴 방향의 rows/cols 증가 (기존 calculate_cluster_size_2d 로직)
            if group["width"] > group["height"]:
                n_rows += 1
                n_cols = math.ceil(facility_count / n_rows)
            else:
                n_cols += 1
                n_rows = math.ceil(facility_count / n_cols)
            
            # 안전 체크
            if n_rows > 50 or n_cols > 50:
                logger.warning("%s arrangement exceeded 50: %dx%d, skipping", gid, n_cols, n_rows)
                continue
            
            # 재계산 (함수 재사용)
            old_size, new_size = recalculate_group_size(group, n_cols, n_rows, scale)
            logger.info(
                "Adjusted %s: %dx%d cols/rows, size: %s -> %s",
                gid,
                group["_n_cols"],
                group["_n_rows"],
                old_size,
                new_size,
            )
    
    return env_data


def convert_sma_to_env(
    input_path: str,
    output_path: str,
    grid_size: float = 100.0,
) -> Dict[str, Any]:
    """Convert SMA JSON to Env JSON.
    
    Args:
        input_path: Source SMA JSON file path
        output_path: Output Env JSON file path
        grid_size: mm per grid (default: 100mm = 1 grid)
    
    Returns:
        Converted env data
    """
    logger.info("Loading %s", input_path)
    data = load_json(input_path)
    
    scale = grid_size
    
    # Factory dimensions
    factory_dim = data["factoryDimensions"]
    factory_width = factory_dim["width"]
    factory_height = factory_dim["height"]
    grid_width = int(factory_width / scale)
    grid_height = int(factory_height / scale)
    logger.info("Factory: %sx%s mm -> %sx%s grid", factory_width, factory_height, grid_width, grid_height)
    
    # Facility dict
    facilities = data["facilities"]
    facility_dict = build_facility_dict(facilities)
    logger.info("Facilities: %d", len(facilities))
    
    # Groups (pass factory dimensions for 2D arrangement calculation)
    facility_groups = data.get("facilityGroups", [])
    groups = build_groups(
        facility_groups, facility_dict, scale,
        factory_width, factory_height
    )
    logger.info("Groups: %d", len(groups))
    
    # Print group summary
    for gid, g in groups.items():
        arrangement = f"{g['_n_cols']}x{g['_n_rows']}" if g['_facility_count'] > 1 else "1x1"
        logger.info(
            "  - %s: %d facilities (%s), cluster=%.1fx%.1f, type=%s",
            gid,
            g["_facility_count"],
            arrangement,
            g["width"],
            g["height"],
            g["_type"],
        )
    
    valid_group_ids = set(groups.keys())
    
    # Flow
    flow = build_flow(facilities, valid_group_ids)
    logger.info("Flow edges: %d", len(flow))
    
    # Zones (forbidden, placement, height, weight, dry)
    forbidden_areas = data.get("forbiddenAreas", [])
    column_areas = data.get("columnAreas", [])
    placeable_areas = data.get("placeableAreas", [])
    area_ceilings = data.get("areaCeilings", [])
    area_weights = data.get("areaWeights", [])
    area_dry = data.get("areaDry", [])
    zones = build_zones(
        placeable_areas, area_ceilings, area_weights, area_dry,
        forbidden_areas, column_areas, scale,
        default_height=factory_dim.get("ceilingHeight", None),
        default_weight=factory_dim.get("weight", None),
        default_dry=factory_dim.get("dry", None),
        default_placeable=0,
    )
    logger.info("Forbidden areas: %d", len(zones.get("forbidden_areas", [])))
    constraints = zones.get("constraints", {})
    logger.info("Constraints: %d", len(constraints) if isinstance(constraints, dict) else 0)
    if isinstance(constraints, dict):
        for cname, cfg in constraints.items():
            areas = cfg.get("areas", []) if isinstance(cfg, dict) else []
            logger.info("  - %s: areas=%d", cname, len(areas))
    
    # Initial positions
    fixed_positions = data.get("fixedFacilityPositions", [])
    initial_placements = build_initial_placements(fixed_positions, facility_dict, groups, scale)
    logger.info("Initial placements: %d", len(initial_placements))
    
    # Build output
    env_data = {
        "grid": {
            "width": grid_width,
            "height": grid_height,
            "grid_size": 1.0,
        },
        "env": {
            "max_candidates": 70,
        },
        "groups": groups,
        "flow": flow,
        "zones": zones,
        "reset": {},
    }
    
    if initial_placements:
        env_data["reset"]["initial_placements"] = initial_placements
    
    # Validate and adjust groups if needed
    logger.info("Validating placeable positions...")
    env_data = validate_and_adjust_groups(env_data, scale)
    
    # Save
    save_json(env_data, output_path)
    logger.info("Saved to %s", output_path)
    
    return env_data


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    parser = argparse.ArgumentParser(description="Convert SMA JSON to Factory Layout Env JSON")
    parser.add_argument("input", help="Input SMA JSON file path")
    parser.add_argument("output", help="Output Env JSON file path")
    parser.add_argument(
        "--grid-size",
        type=float,
        default=100.0,
        help="mm per grid unit (default: 100)",
    )
    
    args = parser.parse_args()
    convert_sma_to_env(args.input, args.output, args.grid_size)


if __name__ == "__main__":
    main()
