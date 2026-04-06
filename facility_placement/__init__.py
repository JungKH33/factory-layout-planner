"""Phase 2 facility-level placement (post-cluster unfolding).

Pure geometry — no search, no placeability re-validation.  Consumes the
state dict produced by ``envs.export.export_group_placement``; has no imports
from ``envs`` / ``agents`` / ``search``.
"""

from .schema import FacilityPlacement
from .resolver import resolve_facilities
from .visualize import (
    draw_cluster_outlines,
    draw_facility_rects,
    plot_facility_layout,
    save_facility_layout,
)

__all__ = [
    "FacilityPlacement",
    "resolve_facilities",
    "draw_cluster_outlines",
    "draw_facility_rects",
    "plot_facility_layout",
    "save_facility_layout",
]
