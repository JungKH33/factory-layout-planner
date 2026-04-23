from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import simpy

from simulation.schema import TransportSpec


@dataclass
class FleetStats:
    transport_count: int = 0
    busy_time_sec: float = 0.0
    wait_time_sec: float = 0.0
    lane_wait_time_sec: float = 0.0


class TransportFleet:
    def __init__(self, env: simpy.Environment, spec: TransportSpec):
        self.env = env
        self.spec = spec
        self.vehicles = simpy.Resource(env, capacity=max(1, int(spec.fleet_size)))
        self._lane_resources: Dict[str, simpy.Resource] = {}
        self.stats = FleetStats()

    def _lane_resource(self, lane_key: str, capacity: int) -> simpy.Resource:
        resource = self._lane_resources.get(lane_key)
        if resource is None:
            resource = simpy.Resource(self.env, capacity=max(1, int(capacity)))
            self._lane_resources[lane_key] = resource
        return resource

    def move(
        self,
        *,
        lane_key: str,
        lane_capacity: int,
        distance_m: float,
    ):
        req_t0 = float(self.env.now)
        with self.vehicles.request() as vehicle_req:
            yield vehicle_req
            self.stats.wait_time_sec += max(0.0, float(self.env.now) - req_t0)

            lane_res = self._lane_resource(lane_key, lane_capacity)
            lane_t0 = float(self.env.now)
            with lane_res.request() as lane_req:
                yield lane_req
                self.stats.lane_wait_time_sec += max(0.0, float(self.env.now) - lane_t0)

                move_time = float(distance_m) / max(1e-6, float(self.spec.speed_mps))
                service = move_time + max(0.0, float(self.spec.load_unload_sec)) * 2.0
                t0 = float(self.env.now)
                yield self.env.timeout(service)
                self.stats.transport_count += 1
                self.stats.busy_time_sec += max(0.0, float(self.env.now) - t0)

