from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# GroupVariant — one distinct placement form (source shape × rotation × mirror)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupVariant:
    """One placement variant with precomputed body geometry.

    A variant is a unique (source_shape, rotation, mirror) combination —
    every distinct way to place this facility.  Spec-specific extras live
    in subclasses (e.g. ``StaticVariant``).
    """

    source_index: int        # which original shape definition (0 for single-shape)
    rotation: int            # 0, 90, 180, 270
    mirror: bool
    body_width: int          # rotated body width  (grid cells)
    body_height: int         # rotated body height (grid cells)
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
    def variants(self) -> List[GroupVariant]:
        """All unique placement variants for this spec."""
        return []

    @property
    def body_area(self) -> float:
        raise NotImplementedError(f"{type(self).__name__}.body_area is not implemented")

    @property
    def body_widths(self) -> "torch.Tensor":
        """[V] float32 body width per variant."""
        raise NotImplementedError(f"{type(self).__name__}.body_widths is not implemented")

    @property
    def body_heights(self) -> "torch.Tensor":
        """[V] float32 body height per variant."""
        raise NotImplementedError(f"{type(self).__name__}.body_heights is not implemented")

    def set_device(self, device: torch.device) -> None:
        raise NotImplementedError(f"{type(self).__name__}.set_device() is not implemented")

    def rotated_size(self, rotation: int):
        raise NotImplementedError(f"{type(self).__name__}.rotated_size() is not implemented")

    def resolve(self, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.resolve() is not implemented")

    def placeable_map(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.placeable_map() is not implemented")

    def placeable_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.placeable_batch() is not implemented")

    def score_batch(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.score_batch() is not implemented")

    def build_placement(self, *, variant_index: int, x_bl: int, y_bl: int) -> "GroupPlacement":
        """Pure geometry — build placement without placeability check."""
        raise NotImplementedError(f"{type(self).__name__}.build_placement() is not implemented")


@dataclass
class GroupPlacement:
    """env가 placement 객체에서 읽는 최소 계약.

    Center-based placement geometry shared by resolved placement objects.
    모든 필드가 required — 기본값 없음 (dataclass 상속 규칙상 child에서
    non-default 필드를 추가하려면 parent에 default가 없어야 함).
    """

    x_center: float
    y_center: float
    entry_points: List[Tuple[float, float]]
    exit_points: List[Tuple[float, float]]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    body_mask: torch.Tensor
    clearance_mask: torch.Tensor
    clearance_origin: Tuple[int, int]
    is_rectangular: bool

    def position(self) -> Tuple[float, float]:
        """Return center position tuple (x_center, y_center)."""
        return float(self.x_center), float(self.y_center)
