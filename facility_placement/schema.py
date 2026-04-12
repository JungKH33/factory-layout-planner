"""Output schema for Phase 2 facility placement.

Only the output side is a dataclass — input (state dict, facilities, layouts,
slots) is handled as plain dicts matching ``export_group_placement`` interchange output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FacilityPlacement:
    """A single facility instance unfolded from a cluster's layout.

    Coordinates are in world **mm** (bottom-left origin), matching the mm
    coordinate system carried in the Phase 1 → Phase 2 interchange dict.
    """

    gid: str                                           # owning cluster
    fid: str                                           # facility registry key
    x_mm: float                                        # BL, world mm
    y_mm: float
    width_mm: float                                    # post-rotation bbox w
    height_mm: float                                   # post-rotation bbox h
    rotation: int                                      # 0/90/180/270
    entry_points_abs_mm: Tuple[Tuple[float, float], ...]
    exit_points_abs_mm:  Tuple[Tuple[float, float], ...]
