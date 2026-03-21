from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Orientation — base protocol for (rotation, mirror) geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Orientation:
    """One (rotation, mirror) orientation with precomputed body geometry.

    All spec types populate this base with the minimal fields that search /
    pipeline need.  Spec-specific extras live in subclasses
    (e.g. ``StaticOrientation``).
    """

    rotation: int            # 0, 90, 180, 270
    mirror: bool
    body_w: int              # rotated body width  (grid cells)
    body_h: int              # rotated body height (grid cells)
    entry_offsets: Tuple[Tuple[float, float], ...]   # center-relative
    exit_offsets: Tuple[Tuple[float, float], ...]     # center-relative


class GroupSpec:
    """Common spec interface for group placement domains."""

    id: object
    device: torch.device
    zone_values: Dict[str, Any]

    @property
    def entry_port_mode(self) -> str:
        """How this facility's entry ports are aggregated: ``"min"`` or ``"mean"``."""
        return str(getattr(self, "_entry_port_mode", "min"))

    @property
    def exit_port_mode(self) -> str:
        """How this facility's exit ports are aggregated: ``"min"`` or ``"mean"``."""
        return str(getattr(self, "_exit_port_mode", "min"))

    @property
    def orientations(self) -> List[Orientation]:
        """All unique (rotation, mirror) orientations for this spec."""
        return []

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

    def placeable_center_map(self, *args, **kwargs):
        """Center-based validity map: ``map[y, x] = True`` means placing
        with **center** at approximately ``(x, y)`` is valid for at least
        one orientation.  Default delegates to ``placeable_map`` (BL-based).
        """
        raise NotImplementedError(f"{type(self).__name__}.placeable_center_map() is not implemented")

    def placeable_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.placeable_batch() is not implemented")

    def cost_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.cost_batch() is not implemented")


@dataclass
class GroupPlacement:
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
