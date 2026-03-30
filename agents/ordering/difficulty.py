from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from envs.env import FactoryLayoutEnv, GroupId


@dataclass(frozen=True)
class DifficultyOrderingAgent:
    """Sort remaining groups by static-zone difficulty."""

    def reorder(self, *, env: FactoryLayoutEnv, obs: dict[str, Any]) -> None:
        del obs
        if len(env.get_state().remaining) <= 1:
            return

        total_area = int(env.grid_width) * int(env.grid_height)
        static_only = env.get_maps().static_invalid
        ordering: list[tuple[float, float, GroupId]] = []
        for gid in env.get_state().remaining:
            spec = env.group_specs[gid]
            inv = static_only | env.get_maps().zone_for_geom(spec)
            invalid_area = int(inv.to(torch.int64).sum().item())
            free_area = max(1, int(total_area) - int(invalid_area))
            facility_area = float(spec.body_area)
            difficulty = float(facility_area) / float(free_area)
            ordering.append((difficulty, facility_area, gid))

        ordering.sort(key=lambda t: (-t[0], -t[1], str(t[2])))
        env.reorder_remaining([gid for _diff, _area, gid in ordering])
