from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from .base import PlacementBase
from .static import StaticSpec


class GridMaps:
    """그리드 맵 텐서와 업데이트 로직 캡슐화.

    env가 직접 소유하던 맵 텐서 9개 + 업데이트 메서드 7개를 분리.

    Public tensors (env에서 읽기 전용):
        static_invalid  : [H,W] bool – forbidden + 영구 invalid
        occ_invalid     : [H,W] bool – 배치된 facility body
        clear_invalid   : [H,W] bool – 배치된 facility clearance
        zone_invalid    : [H,W] bool – next group의 zone/constraint
        invalid         : [H,W] bool – static | occ | zone (composite)
        weight_map      : [H,W] float
        height_map      : [H,W] float
        dry_map         : [H,W] float
        placement_area_masks : Dict[str, [H,W] bool]
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        device: torch.device,
        *,
        forbidden_areas: List[Dict[str, Any]],
        default_weight: float,
        weight_areas: List[Dict[str, Any]],
        default_height: float,
        height_areas: List[Dict[str, Any]],
        default_dry: float,
        dry_areas: List[Dict[str, Any]],
        placement_areas: List[Dict[str, Any]],
    ) -> None:
        self._H = grid_height
        self._W = grid_width
        self._device = device

        self.static_invalid = self._build_static_invalid(
            grid_height, grid_width, device, forbidden_areas
        )
        self.occ_invalid = torch.zeros((grid_height, grid_width), dtype=torch.bool, device=device)
        # Clearance layer: body padded by per-facility clearance of placed facilities.
        self.clear_invalid = torch.zeros((grid_height, grid_width), dtype=torch.bool, device=device)
        # Zone layer: next-group-dependent constraints (weight/height/dry/allowed_areas).
        self.zone_invalid = torch.zeros((grid_height, grid_width), dtype=torch.bool, device=device)
        # Composite: static | occ | zone.  (clear_invalid is queried separately.)
        self.invalid = self.static_invalid.clone()

        self.weight_map = self._build_value_map(
            grid_height, grid_width, device, default_weight, weight_areas
        )
        self.height_map = self._build_value_map(
            grid_height, grid_width, device, default_height, height_areas
        )
        self.dry_map = self._build_value_map(
            grid_height, grid_width, device, default_dry, dry_areas
        )
        self.placement_area_masks: Dict[str, torch.Tensor] = self._build_placement_area_masks(
            grid_height, grid_width, device, placement_areas
        )

    # ---- public update methods ----

    def paint(self, placement: PlacementBase) -> None:
        """배치된 facility의 body/clearance를 occ/clear 맵에 반영 후 recompute."""
        min_x = float(placement.min_x)
        min_y = float(placement.min_y)
        max_x = float(placement.max_x)
        max_y = float(placement.max_y)
        x0 = int(math.floor(min_x))
        y0 = int(math.floor(min_y))
        x1 = int(math.ceil(max_x))
        y1 = int(math.ceil(max_y))
        if x0 < x1 and y0 < y1:
            self.occ_invalid[y0:y1, x0:x1] = True
        # Use getattr for forward-compatibility with DynamicPlacement (no clearance fields).
        cl = int(getattr(placement, "clearance_left", 0))
        cr = int(getattr(placement, "clearance_right", 0))
        cb = int(getattr(placement, "clearance_bottom", 0))
        ct = int(getattr(placement, "clearance_top", 0))
        px0, py0 = x0 - cl, y0 - cb
        px1, py1 = x1 + cr, y1 + ct
        if px0 < px1 and py0 < py1:
            self.clear_invalid[py0:py1, px0:px1] = True
        self.recompute()

    def update_zone(self, geom: Optional[StaticSpec] = None) -> None:
        """zone_invalid를 geom 기준으로 갱신하고 invalid 재계산.

        geom=None이면 zone_invalid를 비우고 재계산 (remaining 없을 때).
        """
        self.zone_invalid.zero_()
        if geom is not None:
            self.zone_invalid |= (self.weight_map < float(geom.facility_weight))
            self.zone_invalid |= (self.height_map < float(geom.facility_height))
            self.zone_invalid |= (self.dry_map > float(geom.facility_dry))
            allowed = geom.allowed_areas
            if allowed:
                allowed_mask = torch.zeros(
                    (self._H, self._W), dtype=torch.bool, device=self._device
                )
                for aid in allowed:
                    if aid in self.placement_area_masks:
                        allowed_mask |= self.placement_area_masks[aid]
                self.zone_invalid |= (~allowed_mask)
        self.recompute()

    def zone_for_geom(self, geom: StaticSpec) -> torch.Tensor:
        """geom에 대한 zone_invalid 맵 반환 (상태 변경 없음)."""
        z = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        z |= (self.weight_map < float(geom.facility_weight))
        z |= (self.height_map < float(geom.facility_height))
        z |= (self.dry_map > float(geom.facility_dry))
        allowed = geom.allowed_areas
        if allowed:
            allowed_mask = torch.zeros(
                (self._H, self._W), dtype=torch.bool, device=self._device
            )
            for aid in allowed:
                if aid in self.placement_area_masks:
                    allowed_mask |= self.placement_area_masks[aid]
            z |= (~allowed_mask)
        return z

    def recompute(self) -> None:
        """invalid = static | occ | zone (in-place, 재할당 없음)."""
        self.invalid.copy_(self.static_invalid)
        self.invalid.logical_or_(self.occ_invalid)
        self.invalid.logical_or_(self.zone_invalid)

    def reset(self) -> None:
        """배치 초기화: occ/clear/zone 맵 리셋 후 invalid 재계산."""
        self.occ_invalid.zero_()
        self.clear_invalid.zero_()
        self.zone_invalid.zero_()
        self.recompute()

    # ---- static builders (private) ----

    @staticmethod
    def _build_static_invalid(
        H: int, W: int, device: torch.device, forbidden_areas: List[Dict[str, Any]]
    ) -> torch.Tensor:
        inv = torch.zeros((H, W), dtype=torch.bool, device=device)
        for area in forbidden_areas:
            if not isinstance(area, dict) or "rect" not in area:
                continue
            rect = area["rect"]
            x0 = max(0, min(W, int(rect[0])))
            x1 = max(0, min(W, int(rect[2])))
            y0 = max(0, min(H, int(rect[1])))
            y1 = max(0, min(H, int(rect[3])))
            if x1 > x0 and y1 > y0:
                inv[y0:y1, x0:x1] = True
        return inv

    @staticmethod
    def _build_value_map(
        H: int,
        W: int,
        device: torch.device,
        default_value: float,
        areas: List[Dict[str, Any]],
    ) -> torch.Tensor:
        m = torch.full((H, W), float(default_value), dtype=torch.float32, device=device)
        for a in areas:
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            v = a.get("value", None)
            if rect is None or v is None:
                continue
            x0, y0, x1, y1 = rect
            x0 = max(0, min(W, int(x0)))
            x1 = max(0, min(W, int(x1)))
            y0 = max(0, min(H, int(y0)))
            y1 = max(0, min(H, int(y1)))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = float(v)
        return m

    @staticmethod
    def _build_placement_area_masks(
        H: int, W: int, device: torch.device, placement_areas: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for area in placement_areas:
            if not isinstance(area, dict):
                continue
            aid = str(area.get("id", ""))
            if not aid:
                continue
            rect = area.get("rect", None)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            x0 = max(0, min(W, int(x0)))
            x1 = max(0, min(W, int(x1)))
            y0 = max(0, min(H, int(y0)))
            y1 = max(0, min(H, int(y1)))
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
            masks[aid] = m
        return masks
