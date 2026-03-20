from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class PlacementBase:
    """env가 placement 객체에서 읽는 최소 계약.

    Static / Dynamic 등 모든 placement 타입의 공통 base.
    모든 필드가 required — 기본값 없음 (dataclass 상속 규칙상 child에서
    non-default 필드를 추가하려면 parent에 default가 없어야 함).
    """

    x_bl: int
    y_bl: int
    rotation: int
    entries: List[Tuple[float, float]]
    exits: List[Tuple[float, float]]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    body_map: torch.Tensor
    clearance_map: torch.Tensor
    clearance_origin: Tuple[int, int]
    is_rectangular: bool
    clearance_left: int
    clearance_right: int
    clearance_bottom: int
    clearance_top: int

    def pose(self) -> Tuple[int, int, int]:
        """Return bottom-left pose tuple (x_bl, y_bl, rotation)."""
        return int(self.x_bl), int(self.y_bl), int(self.rotation)
