from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GroupPlacementConfig": ("pipeline.group_placement", "GroupPlacementConfig"),
    "run_group_placement": ("pipeline.group_placement", "run_group_placement"),
    "run_and_save_group_placement": ("pipeline.group_placement", "run_and_save_group_placement"),
    "GroupLaneGenerationConfig": ("pipeline.group_lane_generation", "GroupLaneGenerationConfig"),
    "run_group_lane_generation": ("pipeline.group_lane_generation", "run_group_lane_generation"),
    "run_and_save_group_lane_generation": ("pipeline.group_lane_generation", "run_and_save_group_lane_generation"),
    "FacilityPlacementConfig": ("pipeline.facility_placement", "FacilityPlacementConfig"),
    "run_facility_placement": ("pipeline.facility_placement", "run_facility_placement"),
    "run_and_save_facility_placement": ("pipeline.facility_placement", "run_and_save_facility_placement"),
}


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(name)
    module_name, obj_name = spec
    module = import_module(module_name)
    value = getattr(module, obj_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS.keys())
