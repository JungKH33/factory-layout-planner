from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LaneFlowSpec:
    src_gid: str
    dst_gid: str
    weight: float
    src_ports: Tuple[Tuple[int, int], ...]
    dst_ports: Tuple[Tuple[int, int], ...]


class LaneFlowGraph:
    """Runtime flow tensor bundle for lane-generation env."""

    def __init__(
        self,
        flows: Sequence[LaneFlowSpec],
        *,
        device: torch.device,
        ordering: str = "weight_desc",
    ) -> None:
        self._device = torch.device(device)
        self._flows = list(flows)
        self._ordering = str(ordering)

        f = len(self._flows)
        self.weights = torch.zeros((f,), dtype=torch.float32, device=self._device)
        self.src_gid: List[str] = []
        self.dst_gid: List[str] = []

        smax = max((len(x.src_ports) for x in self._flows), default=0)
        dmax = max((len(x.dst_ports) for x in self._flows), default=0)
        smax = max(1, int(smax))
        dmax = max(1, int(dmax))
        self.src_ports = torch.zeros((f, smax, 2), dtype=torch.long, device=self._device)
        self.dst_ports = torch.zeros((f, dmax, 2), dtype=torch.long, device=self._device)
        self.src_mask = torch.zeros((f, smax), dtype=torch.bool, device=self._device)
        self.dst_mask = torch.zeros((f, dmax), dtype=torch.bool, device=self._device)

        for i, spec in enumerate(self._flows):
            self.weights[i] = float(spec.weight)
            self.src_gid.append(str(spec.src_gid))
            self.dst_gid.append(str(spec.dst_gid))
            for j, (x, y) in enumerate(spec.src_ports):
                self.src_ports[i, j, 0] = int(x)
                self.src_ports[i, j, 1] = int(y)
                self.src_mask[i, j] = True
            for j, (x, y) in enumerate(spec.dst_ports):
                self.dst_ports[i, j, 0] = int(x)
                self.dst_ports[i, j, 1] = int(y)
                self.dst_mask[i, j] = True

        if self._ordering == "weight_desc":
            if f > 0:
                self.flow_order = torch.argsort(self.weights, descending=True)
            else:
                self.flow_order = torch.empty((0,), dtype=torch.long, device=self._device)
        elif self._ordering == "given":
            self.flow_order = torch.arange(f, dtype=torch.long, device=self._device)
        else:
            raise ValueError("ordering must be 'weight_desc' or 'given'")

        self.total_weight = float(self.weights.sum().item())

    def copy(self) -> "LaneFlowGraph":
        out = object.__new__(LaneFlowGraph)
        out._device = self._device
        out._flows = self._flows
        out._ordering = self._ordering
        out.weights = self.weights
        out.src_gid = self.src_gid
        out.dst_gid = self.dst_gid
        out.src_ports = self.src_ports
        out.dst_ports = self.dst_ports
        out.src_mask = self.src_mask
        out.dst_mask = self.dst_mask
        out.flow_order = self.flow_order
        out.total_weight = self.total_weight
        return out

    @property
    def flow_count(self) -> int:
        return int(self.weights.shape[0])

    def ordered_flow_index(self, step_count: int) -> Optional[int]:
        pos = int(step_count)
        if pos < 0 or pos >= int(self.flow_order.numel()):
            return None
        return int(self.flow_order[pos].item())

    def flow_pair(self, flow_index: int) -> Tuple[str, str, float]:
        i = int(flow_index)
        return self.src_gid[i], self.dst_gid[i], float(self.weights[i].item())

    def valid_ports(self, flow_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(flow_index)
        src = self.src_ports[i][self.src_mask[i]]
        dst = self.dst_ports[i][self.dst_mask[i]]
        return src, dst

    def remaining_weight(self, routed_mask: torch.Tensor) -> float:
        rem = self.weights.masked_fill(routed_mask.to(dtype=torch.bool, device=self._device), 0.0).sum()
        return float(rem.item())

    def remaining_weight_ratio(self, routed_mask: torch.Tensor) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.remaining_weight(routed_mask) / float(self.total_weight)
