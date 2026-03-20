from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


class GroupSpec:
    """Common spec interface for group placement domains."""

    id: object
    device: torch.device
    zone_values: Dict[str, Any]

    @property
    def body_area(self) -> float:
        raise NotImplementedError(f"{type(self).__name__}.body_area is not implemented")

    def set_device(self, device: torch.device) -> None:
        raise NotImplementedError(f"{type(self).__name__}.set_device() is not implemented")

    def rotated_size(self, rotation: int):
        raise NotImplementedError(f"{type(self).__name__}.rotated_size() is not implemented")

    def build_placement(self, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.build_placement() is not implemented")

    def resolve(self, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.resolve() is not implemented")

    def placeable_map(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.placeable_map() is not implemented")

    def placeable_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.placeable_batch() is not implemented")

    def cost_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.cost_batch() is not implemented")


@dataclass
class PlacementBase:
    """env가 placement 객체에서 읽는 최소 계약.

    Center-based placement geometry shared by resolved placement objects.
    모든 필드가 required — 기본값 없음 (dataclass 상속 규칙상 child에서
    non-default 필드를 추가하려면 parent에 default가 없어야 함).
    """

    x_c: float
    y_c: float
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

    def position(self) -> Tuple[float, float]:
        """Return center position tuple (x_c, y_c)."""
        return float(self.x_c), float(self.y_c)
