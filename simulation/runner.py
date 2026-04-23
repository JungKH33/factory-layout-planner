from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import simpy

from simulation.kpi import KPICollector
from simulation.models.facility import FacilityServer
from simulation.models.network import LaneNetwork
from simulation.models.transporter import TransportFleet
from simulation.schema import FlowEdge, SimulationInput, validate_input


@dataclass
class Lot:
    lot_id: int
    src_gid: str
    dst_gid: str
    created_at: float


class SimulationRunner:
    def __init__(self, sim_input: SimulationInput):
        validate_input(sim_input)
        self.data = sim_input
        self.env = simpy.Environment()
        self.rng = random.Random(int(sim_input.run_spec.seed))
        self.collector = KPICollector()
        self.network = LaneNetwork(
            group_payload=sim_input.group_placement,
            lane_payload=sim_input.lane_generation,
            default_capacity=1,
        )
        self.facilities = {
            gid: FacilityServer(self.env, spec) for gid, spec in sim_input.facility_specs.items()
        }
        self.fleet = TransportFleet(self.env, sim_input.transport_spec)
        self._lot_seq = 0
        self._wip = 0
        self._lane_counter: Dict[str, int] = {}

    def _next_lot(self, flow: FlowEdge) -> Lot:
        self._lot_seq += 1
        return Lot(
            lot_id=self._lot_seq,
            src_gid=flow.src_gid,
            dst_gid=flow.dst_gid,
            created_at=float(self.env.now),
        )

    def _interarrival_sec(self, flow: FlowEdge) -> float:
        # Keep edge arrivals stable while still reflecting flow weight.
        # weight=1 => 120sec, weight=10 => 12sec
        base = 120.0 / max(0.1, float(flow.weight))
        return max(1.0, base)

    def _sample_interval(self, mean_sec: float) -> float:
        return self.rng.expovariate(1.0 / max(1e-6, mean_sec))

    def _flow_generator(self, flow: FlowEdge):
        while float(self.env.now) < float(self.data.run_spec.horizon_sec):
            interval = self._sample_interval(self._interarrival_sec(flow))
            yield self.env.timeout(interval)
            lot = self._next_lot(flow)
            self.collector.on_created()
            self._wip += 1
            self.env.process(self._run_lot(lot))

    def _run_lot(self, lot: Lot):
        src = self.facilities.get(lot.src_gid)
        dst = self.facilities.get(lot.dst_gid)
        if src is None or dst is None:
            self.collector.on_dropped()
            self._wip = max(0, self._wip - 1)
            return

        yield self.env.process(src.enqueue(lot))
        _ = yield self.env.process(src.process_once())

        lane_key = self.network.lane_key(lot.src_gid, lot.dst_gid)
        lane_cap = self.network.lane_capacity(lot.src_gid, lot.dst_gid)
        distance_m = self.network.route_length_m(lot.src_gid, lot.dst_gid)
        self._lane_counter[lane_key] = self._lane_counter.get(lane_key, 0) + 1

        yield self.env.process(
            self.fleet.move(
                lane_key=lane_key,
                lane_capacity=lane_cap,
                distance_m=distance_m,
            )
        )

        yield self.env.process(dst.enqueue(lot))
        _ = yield self.env.process(dst.process_once())

        latency = float(self.env.now) - float(lot.created_at)
        if float(self.env.now) >= float(self.data.run_spec.warmup_sec):
            self.collector.on_completed(latency)
        self._wip = max(0, self._wip - 1)

    def _timeline_sampler(self):
        step = max(1e-3, float(self.data.run_spec.timeline_step_sec))
        horizon = float(self.data.run_spec.horizon_sec)
        while float(self.env.now) < horizon:
            self.collector.snapshot(sim_time=float(self.env.now), wip=int(self._wip))
            yield self.env.timeout(step)
        self.collector.snapshot(sim_time=float(horizon), wip=int(self._wip))

    def run(self) -> Dict[str, Any]:
        for flow in self.data.flows:
            self.env.process(self._flow_generator(flow))
        self.env.process(self._timeline_sampler())
        self.env.run(until=float(self.data.run_spec.horizon_sec))

        horizon = float(self.data.run_spec.horizon_sec)
        summary = self.collector.summary(
            horizon_sec=horizon,
            warmup_sec=float(self.data.run_spec.warmup_sec),
        )
        facility_util = {}
        for gid, server in self.facilities.items():
            capacity_time = max(1e-6, horizon * float(server.capacity_factor()))
            facility_util[gid] = {
                "started": int(server.stats.started),
                "completed": int(server.stats.completed),
                "queue_wait_sec": float(server.stats.queue_wait_sec),
                "blocked_in_sec": float(server.stats.blocked_in_sec),
                "busy_time_sec": float(server.stats.busy_time_sec),
                "processing_mode": str(server.spec.processing_mode),
                "utilization": float(server.stats.busy_time_sec / capacity_time),
            }

        agv_capacity_time = max(1e-6, horizon * float(self.data.transport_spec.fleet_size))
        transport_util = {
            "transport_count": int(self.fleet.stats.transport_count),
            "busy_time_sec": float(self.fleet.stats.busy_time_sec),
            "wait_time_sec": float(self.fleet.stats.wait_time_sec),
            "lane_wait_time_sec": float(self.fleet.stats.lane_wait_time_sec),
            "utilization": float(self.fleet.stats.busy_time_sec / agv_capacity_time),
        }

        top_facility_wait = sorted(
            (
                {"gid": gid, "queue_wait_sec": float(info["queue_wait_sec"])}
                for gid, info in facility_util.items()
            ),
            key=lambda x: x["queue_wait_sec"],
            reverse=True,
        )[:5]
        top_lane_hotspots = sorted(
            ({"lane": lane, "count": int(count)} for lane, count in self._lane_counter.items()),
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        return {
            "summary": summary,
            "utilization": {
                "facilities": facility_util,
                "transport": transport_util,
            },
            "bottlenecks": {
                "top_facility_queue_wait": top_facility_wait,
                "top_lane_hotspots": top_lane_hotspots,
            },
            "timeline": self.collector.timeline,
        }

