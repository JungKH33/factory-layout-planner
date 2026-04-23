from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple


class LaneNetwork:
    def __init__(
        self,
        *,
        group_payload: Mapping[str, Any],
        lane_payload: Mapping[str, Any],
        default_capacity: int = 1,
    ):
        self.positions = self._load_positions(group_payload)
        self.route_lengths_m, self.route_capacity = self._load_routes(lane_payload, default_capacity)

    @staticmethod
    def _load_positions(group_payload: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for item in group_payload.get("placements", []) or []:
            if not isinstance(item, Mapping):
                continue
            gid = str(item.get("gid", ""))
            cx = float(item.get("x_bl_mm", 0.0)) + float(item.get("cluster_w_mm", 0.0)) * 0.5
            cy = float(item.get("y_bl_mm", 0.0)) + float(item.get("cluster_h_mm", 0.0)) * 0.5
            out[gid] = (cx * 1e-3, cy * 1e-3)
        return out

    @staticmethod
    def _load_routes(
        lane_payload: Mapping[str, Any],
        default_capacity: int,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], int]]:
        lengths: Dict[Tuple[str, str], float] = {}
        capacity: Dict[Tuple[str, str], int] = {}
        routes = lane_payload.get("routes", []) if isinstance(lane_payload, Mapping) else []
        for route in routes or []:
            if not isinstance(route, Mapping) or not bool(route.get("success", False)):
                continue
            src = str(route.get("src_gid", ""))
            dst = str(route.get("dst_gid", ""))
            key = (src, dst)
            cells = route.get("path_length", 0.0)
            # lane_generation path_length is in cells in current export path
            lengths[key] = max(0.1, float(cells))
            lane_width = float(route.get("lane_width", default_capacity))
            capacity[key] = max(1, int(round(lane_width)))
        return lengths, capacity

    def route_length_m(self, src_gid: str, dst_gid: str) -> float:
        key = (str(src_gid), str(dst_gid))
        if key in self.route_lengths_m:
            return max(0.1, float(self.route_lengths_m[key]))
        src = self.positions.get(str(src_gid))
        dst = self.positions.get(str(dst_gid))
        if src is None or dst is None:
            return 1.0
        return max(0.1, abs(src[0] - dst[0]) + abs(src[1] - dst[1]))

    def lane_capacity(self, src_gid: str, dst_gid: str) -> int:
        return int(self.route_capacity.get((str(src_gid), str(dst_gid)), 1))

    def lane_key(self, src_gid: str, dst_gid: str) -> str:
        return f"{src_gid}->{dst_gid}"

