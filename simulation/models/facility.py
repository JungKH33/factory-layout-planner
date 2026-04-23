from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import simpy

if TYPE_CHECKING:
    pass

from simulation.schema import FacilitySpec, StageSpec


@dataclass
class FacilityStats:
    started: int = 0
    completed: int = 0
    busy_time_sec: float = 0.0
    queue_wait_sec: float = 0.0
    blocked_in_sec: float = 0.0


class FacilityServer:
    def __init__(self, env: "simpy.Environment", spec: FacilitySpec):
        self.env = env
        self.spec = spec
        self.resource = simpy.Resource(env, capacity=max(1, int(spec.parallel_slots)))
        self.input_queue = simpy.Store(env, capacity=max(1, int(spec.buffer_in)))
        self._mode = str(spec.processing_mode).strip().lower()
        self._batch_waiting: List[Tuple[object, simpy.Event]] = []
        self._batch_lock = simpy.Resource(env, capacity=1)
        self._stages: List[Tuple[StageSpec, simpy.Resource]] = [
            (
                stage,
                simpy.Resource(env, capacity=max(1, int(stage.parallel_slots))),
            )
            for stage in spec.stages
        ]
        self.stats = FacilityStats()

    def capacity_factor(self) -> float:
        if self._mode != "pipeline" or not self._stages:
            return float(max(1, int(self.spec.parallel_slots)))
        return float(sum(max(1, int(stage.parallel_slots)) for stage, _ in self._stages))

    def enqueue(self, lot: object):
        t0 = float(self.env.now)
        evt = self.input_queue.put({"lot": lot, "queued_at": float(self.env.now)})
        yield evt
        dt = float(self.env.now) - t0
        if dt > 0:
            self.stats.blocked_in_sec += dt

    def _process_serial_or_parallel(self, lot: object):
        with self.resource.request() as req:
            yield req
            self.stats.started += 1
            t0 = float(self.env.now)
            yield self.env.timeout(float(self.spec.cycle_time_sec))
            self.stats.busy_time_sec += max(0.0, float(self.env.now) - t0)
            self.stats.completed += 1
        return lot

    def _process_batch(self, lot: object):
        done_evt = self.env.event()
        batch: List[Tuple[object, simpy.Event]] = []
        leader = False

        with self._batch_lock.request() as lock_req:
            yield lock_req
            self._batch_waiting.append((lot, done_evt))
            if len(self._batch_waiting) >= int(self.spec.batch_size):
                batch = list(self._batch_waiting[: int(self.spec.batch_size)])
                self._batch_waiting = list(self._batch_waiting[int(self.spec.batch_size) :])
                leader = True

        if not leader:
            resolved = yield done_evt
            return resolved

        with self.resource.request() as req:
            yield req
            self.stats.started += int(len(batch))
            t0 = float(self.env.now)
            yield self.env.timeout(float(self.spec.cycle_time_sec))
            self.stats.busy_time_sec += max(0.0, float(self.env.now) - t0)
            self.stats.completed += int(len(batch))

        for batch_lot, evt in batch:
            if not evt.triggered:
                evt.succeed(batch_lot)
        return lot

    def _process_pipeline(self, lot: object):
        self.stats.started += 1
        busy = 0.0
        stages = self._stages
        if not stages:
            stages = [
                (
                    StageSpec(cycle_time_sec=float(self.spec.cycle_time_sec), parallel_slots=int(self.spec.parallel_slots)),
                    self.resource,
                )
            ]
        for stage, stage_res in stages:
            with stage_res.request() as req:
                yield req
                t0 = float(self.env.now)
                yield self.env.timeout(float(stage.cycle_time_sec))
                busy += max(0.0, float(self.env.now) - t0)
        self.stats.busy_time_sec += busy
        self.stats.completed += 1
        return lot

    def process_once(self):
        item = yield self.input_queue.get()
        queued_at = float(item.get("queued_at", float(self.env.now)))
        lot = item.get("lot")
        self.stats.queue_wait_sec += max(0.0, float(self.env.now) - queued_at)

        if self._mode == "batch":
            result = yield from self._process_batch(lot)
            return result
        if self._mode == "pipeline":
            result = yield from self._process_pipeline(lot)
            return result
        result = yield from self._process_serial_or_parallel(lot)
        return result

