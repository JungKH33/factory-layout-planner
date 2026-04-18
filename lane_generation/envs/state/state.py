"""Flat state object for the lane-generation env.

``LaneState`` flattens the directed-edge graph, the per-flow tensor bundle,
and the runtime placement bookkeeping into one object.

Lifecycle:

* :meth:`LaneState.build` constructs everything in one shot — edge tables
  and flow arrays.
* :meth:`LaneState.copy` shares static tensors/caches by reference and clones
  only runtime route state, so MCTS snapshots stay cheap.
* :meth:`LaneState.apply_route` records directed edges and lane slots on
  ``edge_map`` / ``edge_lane_mask``.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..action import LaneRoute


# ----------------------------------------------------------------------
# Flow spec (public)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LaneFlowSpec:
    src_gid: str
    dst_gid: str
    weight: float
    src_ports: Tuple[Tuple[int, int], ...]
    dst_ports: Tuple[Tuple[int, int], ...]
    reverse_allow: Optional[bool] = None
    merge_allow: Optional[bool] = None
    lane_width: float = 1.0


@dataclass(frozen=True)
class RoutingConfig:
    """Pathfinding configuration knobs for :class:`LaneState`."""

    selection: str = "static"  # static | benchmark
    algorithm: str = "wavefront"  # wavefront | dijkstra | astar
    max_wave_iters: int = 0
    max_path_steps: int = 0
    benchmark_warmup: int = 2
    benchmark_rounds: int = 6
    benchmark_max_flows_per_method: int = 6
    merge_allow: bool = True
    reverse_allow: bool = True
    capacity_epsilon: float = 1e-9


# ----------------------------------------------------------------------
# Directed-edge graph constants and builder (private)
# ----------------------------------------------------------------------


_DIR_TO_DXY = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}
_DXY_TO_DIR = {v: k for k, v in _DIR_TO_DXY.items()}
_REV_DIR = torch.tensor([2, 3, 0, 1], dtype=torch.long)


def _build_edge_tables(
    *,
    grid_height: int,
    grid_width: int,
    blocked_static: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Materialize the directed-edge graph for a (H, W) grid.

    Edge indexing: ``edge_id = (y * W + x) * 4 + dir`` with dir ∈
    {0:right, 1:down, 2:left, 3:up}. Each edge has a stable ``reverse_edge_lut``
    entry pointing at its flipped-direction companion.
    """
    h = int(grid_height)
    w = int(grid_width)
    dev = torch.device(device)
    blocked = blocked_static.to(device=dev, dtype=torch.bool)
    if tuple(blocked.shape) != (h, w):
        raise ValueError(f"blocked_static must be {(h, w)}, got {tuple(blocked.shape)}")

    n_cells = h * w
    n_edges = n_cells * 4
    cell = torch.arange(n_cells, dtype=torch.long, device=dev)
    cy = torch.div(cell, w, rounding_mode="floor")
    cx = cell % w

    edge_valid = torch.zeros((n_edges,), dtype=torch.bool, device=dev)
    reverse = torch.arange(n_edges, dtype=torch.long, device=dev)
    src = torch.empty((n_edges,), dtype=torch.long, device=dev)
    dst = torch.full((n_edges,), -1, dtype=torch.long, device=dev)

    for d in range(4):
        dx, dy = _DIR_TO_DXY[d]
        nx = cx + int(dx)
        ny = cy + int(dy)
        valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        eid = cell * 4 + int(d)
        src[eid] = cell
        edge_valid[eid] = valid
        dst_cell = ny * w + nx
        dst[eid[valid]] = dst_cell[valid]
        rev_dir = int(_REV_DIR[d].item())
        reverse[eid[valid]] = dst_cell[valid] * 4 + rev_dir

    return dict(
        blocked_static=blocked,
        edge_valid_flat=edge_valid,
        reverse_edge_lut=reverse,
        edge_src_cell=src,
        edge_dst_cell=dst,
        lane_dir_flat=torch.zeros((n_edges,), dtype=torch.bool, device=dev),
        edge_map=torch.zeros((h, w, 4), dtype=torch.int16, device=dev),
        edge_lane_mask=torch.zeros((h, w, 4), dtype=torch.int64, device=dev),
    )


# ----------------------------------------------------------------------
# Per-flow tensor bundle builder (private)
# ----------------------------------------------------------------------


def _build_flow_arrays(
    flows: Sequence[LaneFlowSpec],
    *,
    device: torch.device,
    ordering: str = "weight_desc",
) -> Dict[str, Any]:
    """Materialize the per-flow tensor bundle for :class:`LaneState`.

    Returns a dict containing all per-flow tensors plus the flat lists of
    src/dst gids and a python ``total_weight`` float. ``ordering`` controls
    the routing order returned in ``flow_order``.
    """
    dev = torch.device(device)
    f = len(flows)

    weights = torch.zeros((f,), dtype=torch.float32, device=dev)
    src_gid: List[str] = []
    dst_gid: List[str] = []

    smax = max((len(x.src_ports) for x in flows), default=0)
    dmax = max((len(x.dst_ports) for x in flows), default=0)
    smax = max(1, int(smax))
    dmax = max(1, int(dmax))
    src_ports = torch.zeros((f, smax, 2), dtype=torch.long, device=dev)
    dst_ports = torch.zeros((f, dmax, 2), dtype=torch.long, device=dev)
    src_mask = torch.zeros((f, smax), dtype=torch.bool, device=dev)
    dst_mask = torch.zeros((f, dmax), dtype=torch.bool, device=dev)

    for i, spec in enumerate(flows):
        weights[i] = float(spec.weight)
        src_gid.append(str(spec.src_gid))
        dst_gid.append(str(spec.dst_gid))
        for j, (x, y) in enumerate(spec.src_ports):
            src_ports[i, j, 0] = int(x)
            src_ports[i, j, 1] = int(y)
            src_mask[i, j] = True
        for j, (x, y) in enumerate(spec.dst_ports):
            dst_ports[i, j, 0] = int(x)
            dst_ports[i, j, 1] = int(y)
            dst_mask[i, j] = True

    if ordering == "weight_desc":
        if f > 0:
            flow_order = torch.argsort(weights, descending=True)
        else:
            flow_order = torch.empty((0,), dtype=torch.long, device=dev)
    elif ordering == "given":
        flow_order = torch.arange(f, dtype=torch.long, device=dev)
    else:
        raise ValueError("ordering must be 'weight_desc' or 'given'")

    return dict(
        weights=weights,
        src_ports=src_ports,
        dst_ports=dst_ports,
        src_mask=src_mask,
        dst_mask=dst_mask,
        flow_order=flow_order,
        src_gid=src_gid,
        dst_gid=dst_gid,
        total_weight=float(weights.sum().item()),
    )


# ----------------------------------------------------------------------
# Flat state (public)
# ----------------------------------------------------------------------


@dataclass
class LaneState:
    """Flat lane-generation env state.

    Field groups (in dataclass order):

    * Geometry / device.
    * Static directed-edge graph (shared by reference across copies).
    * Runtime placed-lane bits.
    * Per-flow tensor bundle (shared by reference across copies).
    * Per-flow python metadata (gids, original specs).
    * Episode progression.
    * Static pathfinding caches.
    """

    grid_height: int
    grid_width: int
    device: torch.device

    blocked_static: torch.Tensor
    edge_valid_flat: torch.Tensor
    reverse_edge_lut: torch.Tensor
    edge_src_cell: torch.Tensor
    edge_dst_cell: torch.Tensor
    lane_dir_flat: torch.Tensor
    edge_map: torch.Tensor
    edge_lane_mask: torch.Tensor

    weights: torch.Tensor
    src_ports: torch.Tensor
    dst_ports: torch.Tensor
    src_mask: torch.Tensor
    dst_mask: torch.Tensor
    flow_order: torch.Tensor
    total_weight: float
    src_gid: List[str]
    dst_gid: List[str]
    flow_specs: Tuple[LaneFlowSpec, ...]

    step_count: int
    routed_mask: torch.Tensor
    route_lane_slots_by_flow: Dict[int, Tuple[int, ...]]
    runtime_max_lane_slot: int
    edge_width_accum: torch.Tensor

    # Pathfinding algorithm configuration.
    _algorithm: str = "wavefront"
    _algorithm_multi_src: str = "wavefront"
    _max_wave_iters: int = 0
    _max_path_steps: int = 0
    _route_merge_allow: bool = True
    _route_reverse_allow: bool = True
    _route_capacity_epsilon: float = 1e-9
    _current_lane_width: float = 1.0
    _allocator: Optional[Any] = None  # LaneSlotAllocator (set by configure_routing)

    # Static/shared caches.
    _static_port_cells_by_gid: Optional[Dict[str, torch.Tensor]] = None
    _static_port_xy_set: Optional[Set[Tuple[int, int]]] = None
    _static_walkable_map: Optional[torch.Tensor] = None
    _cache_dist_map_by_port_targets: Optional[Dict[Tuple[Tuple[int, int], ...], torch.Tensor]] = None

    # Runtime/debug: last computed distance map.
    _runtime_last_dist_map: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        grid_height: int,
        grid_width: int,
        blocked_static: torch.Tensor,
        flows: Sequence[LaneFlowSpec],
        device: torch.device,
        flow_ordering: str = "weight_desc",
    ) -> "LaneState":
        dev = torch.device(device)
        edge = _build_edge_tables(
            grid_height=int(grid_height),
            grid_width=int(grid_width),
            blocked_static=blocked_static,
            device=dev,
        )
        flow_arrays = _build_flow_arrays(flows, device=dev, ordering=flow_ordering)
        f = len(flows)

        return cls(
            grid_height=int(grid_height),
            grid_width=int(grid_width),
            device=dev,
            blocked_static=edge["blocked_static"],
            edge_valid_flat=edge["edge_valid_flat"],
            reverse_edge_lut=edge["reverse_edge_lut"],
            edge_src_cell=edge["edge_src_cell"],
            edge_dst_cell=edge["edge_dst_cell"],
            lane_dir_flat=edge["lane_dir_flat"],
            edge_map=edge["edge_map"],
            edge_lane_mask=edge["edge_lane_mask"],
            weights=flow_arrays["weights"],
            src_ports=flow_arrays["src_ports"],
            dst_ports=flow_arrays["dst_ports"],
            src_mask=flow_arrays["src_mask"],
            dst_mask=flow_arrays["dst_mask"],
            flow_order=flow_arrays["flow_order"],
            total_weight=float(flow_arrays["total_weight"]),
            src_gid=list(flow_arrays["src_gid"]),
            dst_gid=list(flow_arrays["dst_gid"]),
            flow_specs=tuple(flows),
            step_count=0,
            routed_mask=torch.zeros((f,), dtype=torch.bool, device=dev),
            route_lane_slots_by_flow={},
            runtime_max_lane_slot=-1,
            edge_width_accum=torch.zeros(
                int(grid_height) * int(grid_width) * 4, dtype=torch.float32, device=dev
            ),
        )

    # ------------------------------------------------------------------
    # Copy / restore / reset
    # ------------------------------------------------------------------

    def copy(self) -> "LaneState":
        c = LaneState(
            grid_height=int(self.grid_height),
            grid_width=int(self.grid_width),
            device=self.device,
            blocked_static=self.blocked_static,
            edge_valid_flat=self.edge_valid_flat,
            reverse_edge_lut=self.reverse_edge_lut,
            edge_src_cell=self.edge_src_cell,
            edge_dst_cell=self.edge_dst_cell,
            lane_dir_flat=self.lane_dir_flat.clone(),
            edge_map=self.edge_map.clone(),
            edge_lane_mask=self.edge_lane_mask.clone(),
            weights=self.weights,
            src_ports=self.src_ports,
            dst_ports=self.dst_ports,
            src_mask=self.src_mask,
            dst_mask=self.dst_mask,
            flow_order=self.flow_order,
            total_weight=float(self.total_weight),
            src_gid=self.src_gid,
            dst_gid=self.dst_gid,
            flow_specs=self.flow_specs,
            step_count=int(self.step_count),
            routed_mask=self.routed_mask.clone(),
            route_lane_slots_by_flow=dict(self.route_lane_slots_by_flow),
            runtime_max_lane_slot=int(self.runtime_max_lane_slot),
            edge_width_accum=self.edge_width_accum.clone(),
        )
        c._algorithm = self._algorithm
        c._algorithm_multi_src = self._algorithm_multi_src
        c._max_wave_iters = self._max_wave_iters
        c._max_path_steps = self._max_path_steps
        c._route_merge_allow = self._route_merge_allow
        c._route_reverse_allow = self._route_reverse_allow
        c._route_capacity_epsilon = self._route_capacity_epsilon
        c._current_lane_width = self._current_lane_width
        c._allocator = self._allocator  # stateless, safe to share
        c._static_port_cells_by_gid = self._static_port_cells_by_gid
        c._static_port_xy_set = self._static_port_xy_set
        c._static_walkable_map = self._static_walkable_map
        c._cache_dist_map_by_port_targets = self._cache_dist_map_by_port_targets
        c._runtime_last_dist_map = self._runtime_last_dist_map
        return c

    def restore(self, src: "LaneState") -> None:
        if not isinstance(src, LaneState):
            raise TypeError(f"src must be LaneState, got {type(src).__name__}")
        if (self.grid_height, self.grid_width) != (src.grid_height, src.grid_width):
            raise ValueError("grid shape mismatch")
        self.lane_dir_flat.copy_(src.lane_dir_flat.to(device=self.device, dtype=torch.bool))
        self.edge_map.copy_(src.edge_map.to(device=self.device, dtype=torch.int16))
        self.edge_lane_mask.copy_(src.edge_lane_mask.to(device=self.device, dtype=torch.int64))
        self.routed_mask.copy_(src.routed_mask.to(device=self.device, dtype=torch.bool))
        self.step_count = int(src.step_count)
        self.route_lane_slots_by_flow = dict(src.route_lane_slots_by_flow)
        self.runtime_max_lane_slot = int(src.runtime_max_lane_slot)
        self.edge_width_accum.copy_(src.edge_width_accum.to(device=self.device, dtype=torch.float32))
        self._current_lane_width = float(src._current_lane_width)
        self._runtime_last_dist_map = None

    def reset_runtime(self) -> None:
        self.lane_dir_flat.zero_()
        self.edge_map.zero_()
        self.edge_lane_mask.zero_()
        self.routed_mask.zero_()
        self.step_count = 0
        self.route_lane_slots_by_flow = {}
        self.runtime_max_lane_slot = -1
        self.edge_width_accum.zero_()
        self._current_lane_width = 1.0
        self._runtime_last_dist_map = None

    # ------------------------------------------------------------------
    # Group port lookup & per-route free map
    # ------------------------------------------------------------------

    def _ensure_static_port_cells_by_gid(self) -> Dict[str, torch.Tensor]:
        if self._static_port_cells_by_gid is not None:
            return self._static_port_cells_by_gid
        cache: Dict[str, List[Tuple[int, int]]] = {}
        for spec in self.flow_specs:
            for gid, ports in ((spec.src_gid, spec.src_ports), (spec.dst_gid, spec.dst_ports)):
                cells = cache.setdefault(gid, [])
                for xy in ports:
                    cells.append((int(xy[0]), int(xy[1])))
        result: Dict[str, torch.Tensor] = {}
        for gid, cells in cache.items():
            t = torch.tensor(cells, dtype=torch.long, device=self.device)
            result[gid] = torch.unique(t, dim=0) if t.numel() > 0 else t
        self._static_port_cells_by_gid = result
        return result

    def _ensure_static_port_xy_set(self) -> Set[Tuple[int, int]]:
        if self._static_port_xy_set is not None:
            return self._static_port_xy_set
        cells_by_gid = self._ensure_static_port_cells_by_gid()
        out: Set[Tuple[int, int]] = set()
        for cells in cells_by_gid.values():
            for p in cells.detach().cpu().tolist():
                out.add((int(p[0]), int(p[1])))
        self._static_port_xy_set = out
        return out

    def group_port_cells(self, gid: str) -> torch.Tensor:
        """Return ``[N, 2]`` unique port cells (x, y) for group *gid*."""
        cache = self._ensure_static_port_cells_by_gid()
        return cache.get(gid, torch.empty((0, 2), dtype=torch.long, device=self.device))

    def _get_static_walkable_map(self) -> torch.Tensor:
        """Build a static walkable map with all known port cells unblocked.

        ``blocked_static`` marks placed facility bodies as blocked.  For lane
        routing we must treat port cells as walkable.  Since group placement
        is fixed during lane generation, this map is static and shared.
        """
        if self._static_walkable_map is not None:
            return self._static_walkable_map

        walkable = (~self.blocked_static).to(dtype=torch.bool, device=self.device)
        h, w = int(self.grid_height), int(self.grid_width)
        for pxy in self._ensure_static_port_xy_set():
            x, y = int(pxy[0]), int(pxy[1])
            if 0 <= y < h and 0 <= x < w:
                walkable[y, x] = True
        self._static_walkable_map = walkable
        return walkable

    # ------------------------------------------------------------------
    # Wavefront cache
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_target_points(targets_xy: torch.Tensor) -> Tuple[Tuple[int, int], ...]:
        pts = targets_xy.detach().to(dtype=torch.long, device="cpu").view(-1, 2).tolist()
        uniq = sorted({(int(p[0]), int(p[1])) for p in pts})
        return tuple(uniq)

    def _build_port_target_cache_key(self, targets_xy: torch.Tensor) -> Optional[Tuple[Tuple[int, int], ...]]:
        points = self._canonical_target_points(targets_xy)
        if len(points) == 0:
            return None
        h, w = int(self.grid_height), int(self.grid_width)
        port_set = self._ensure_static_port_xy_set()
        for x, y in points:
            if not (0 <= x < w and 0 <= y < h):
                return None
            if (int(x), int(y)) not in port_set:
                return None
        return points

    def get_dist_map_for_targets(
        self,
        *,
        targets_xy: torch.Tensor,
        allow_mask: Optional[torch.Tensor] = None,
        max_wave_iters: int = 0,
    ) -> torch.Tensor:
        """Return ``[H,W]`` distance map for *targets_xy*.

        Cache policy:
        - If *allow_mask* is None and all targets are known ports, cache by
          canonicalized target-port tuple.
        - Otherwise (non-port targets or directional constraints), compute
          without cache.
        """
        cache_key = None
        if allow_mask is None:
            cache_key = self._build_port_target_cache_key(targets_xy)
            if cache_key is not None:
                cache = self._cache_dist_map_by_port_targets
                if cache is not None:
                    hit = cache.get(cache_key, None)
                    if hit is not None:
                        self._runtime_last_dist_map = hit
                        return hit

        from .wavefront import wavefront_distance_field, _wavefront_with_mask

        walkable_map = self._get_static_walkable_map()
        mi = int(max_wave_iters)
        if allow_mask is not None:
            dist = _wavefront_with_mask(
                free_map=walkable_map,
                allow_4d=allow_mask,
                seeds_xy=targets_xy,
                max_iters=mi,
            )
        else:
            dist = wavefront_distance_field(
                free_map=walkable_map,
                seeds_xy=targets_xy,
                max_iters=mi,
            )
            if cache_key is not None:
                if self._cache_dist_map_by_port_targets is None:
                    self._cache_dist_map_by_port_targets = {}
                self._cache_dist_map_by_port_targets[cache_key] = dist
        self._runtime_last_dist_map = dist
        return dist

    @property
    def last_dist_map(self) -> Optional[torch.Tensor]:
        """Read-only access to the latest computed distance map."""
        return self._runtime_last_dist_map

    @property
    def static_walkable_map(self) -> torch.Tensor:
        """Static walkable grid [H,W] (facility blocked, known ports unblocked)."""
        return self._get_static_walkable_map()

    @property
    def cache_dist_map_by_port_targets(self) -> Dict[Tuple[Tuple[int, int], ...], torch.Tensor]:
        """Read-only accessor for cached port-target distance maps."""
        if self._cache_dist_map_by_port_targets is None:
            return {}
        return self._cache_dist_map_by_port_targets

    @property
    def route_merge_allow(self) -> bool:
        return bool(self._route_merge_allow)

    @property
    def route_reverse_allow(self) -> bool:
        return bool(self._route_reverse_allow)

    # ------------------------------------------------------------------
    # Static-graph helpers (formerly LaneMaps methods)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.grid_height), int(self.grid_width)

    @property
    def blocked_flat(self) -> torch.Tensor:
        return self.blocked_static.view(-1)

    @property
    def edge_count(self) -> int:
        return int(self.lane_dir_flat.shape[0])

    def edge_id(self, *, x: int, y: int, direction: int) -> int:
        return (int(y) * int(self.grid_width) + int(x)) * 4 + int(direction)

    def path_to_edge_ids_and_turns(
        self,
        path_xy: Sequence[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, int]:
        if len(path_xy) < 2:
            return torch.empty((0,), dtype=torch.long, device=self.device), 0

        eids: List[int] = []
        prev_dir: Optional[int] = None
        turns = 0
        h, w = self.shape

        for i in range(len(path_xy) - 1):
            x0, y0 = int(path_xy[i][0]), int(path_xy[i][1])
            x1, y1 = int(path_xy[i + 1][0]), int(path_xy[i + 1][1])
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                raise ValueError("path contains out-of-bound coordinates")
            dxy = (x1 - x0, y1 - y0)
            d = _DXY_TO_DIR.get(dxy)
            if d is None:
                raise ValueError(f"path has non-4-neighbor move: {dxy}")
            if prev_dir is not None and d != prev_dir:
                turns += 1
            prev_dir = d
            eid = self.edge_id(x=x0, y=y0, direction=d)
            if not bool(self.edge_valid_flat[eid].item()):
                raise ValueError("path includes invalid edge")
            eids.append(int(eid))

        if len(eids) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device), int(turns)

        seen: set = set()
        ordered_unique: List[int] = []
        for e in eids:
            if e in seen:
                continue
            seen.add(e)
            ordered_unique.append(e)
        return torch.tensor(ordered_unique, dtype=torch.long, device=self.device), int(turns)

    def _ensure_allocator(self):
        if self._allocator is None:
            from .slot_allocator import LaneSlotAllocator
            self._allocator = LaneSlotAllocator(
                capacity_epsilon=self._route_capacity_epsilon,
            )
        return self._allocator

    def _choose_lane_slot(self, *, dir_mask: int, rev_mask: int,
                          merge_allow: bool, reverse_allow: bool) -> int:
        """Compatibility shim — delegates to LaneSlotAllocator.choose_slot."""
        slot, _ = self._ensure_allocator().choose_slot(
            dir_mask=dir_mask, rev_mask=rev_mask,
            width_accum=0.0, lane_width=self._current_lane_width,
            merge_allow=merge_allow, reverse_allow=reverse_allow,
        )
        if slot is None:
            raise RuntimeError("no lane slot available (capacity exceeded or max_slots reached)")
        return int(slot)

    def _lane_slot_for_edge(self, edge_id: int, *,
                            merge_allow: bool, reverse_allow: bool) -> int:
        mask_flat = self.edge_lane_mask.view(-1)
        rev_flat = self.reverse_edge_lut.view(-1)
        accum_flat = self.edge_width_accum.view(-1)
        e = int(edge_id)
        if e < 0 or e >= int(mask_flat.shape[0]):
            raise ValueError(f"edge id out of range: {e}")
        r = int(rev_flat[e].item())
        n = int(mask_flat.shape[0])
        dir_mask = int(mask_flat[e].item())
        rev_mask = int(mask_flat[r].item()) if 0 <= r < n else 0
        leader = min(e, r) if 0 <= r < n else e
        wa = float(accum_flat[leader].item())
        slot, _ = self._ensure_allocator().choose_slot(
            dir_mask=dir_mask, rev_mask=rev_mask,
            width_accum=wa, lane_width=self._current_lane_width,
            merge_allow=merge_allow, reverse_allow=reverse_allow,
        )
        if slot is None:
            raise RuntimeError(f"no lane slot available for edge {e}")
        return int(slot)

    def preview_lane_slots_batch(
        self,
        *,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        merge_allow: Optional[bool] = None,
        reverse_allow: Optional[bool] = None,
    ) -> torch.Tensor:
        """Plan lane slots per candidate edge without mutating runtime state.

        Returns int16 tensor [K, L].  -1 = infeasible or padding.
        Delegates to :class:`LaneSlotAllocator` which tracks both mask and
        width-accum scratchpads per candidate.
        """
        ma = merge_allow if merge_allow is not None else self._route_merge_allow
        ra = reverse_allow if reverse_allow is not None else self._route_reverse_allow
        return self._ensure_allocator().preview_batch(
            self,
            candidate_edge_idx=candidate_edge_idx,
            candidate_edge_mask=candidate_edge_mask,
            lane_width=self._current_lane_width,
            merge_allow=ma, reverse_allow=ra,
        )

    def apply_edges(
        self,
        edge_indices: torch.Tensor,
        *,
        lane_slots: Optional[Sequence[int]] = None,
        lane_width: Optional[float] = None,
    ) -> None:
        if edge_indices.numel() == 0:
            return
        lw = float(lane_width) if lane_width is not None else float(self._current_lane_width)
        idx = edge_indices.to(device=self.device, dtype=torch.long).view(-1)
        mask_flat = self.edge_lane_mask.view(-1)
        count_flat = self.edge_map.view(-1)
        lane_flat = self.lane_dir_flat.view(-1)
        accum_flat = self.edge_width_accum.view(-1)
        rev_flat = self.reverse_edge_lut.view(-1)
        n = int(mask_flat.shape[0])

        if lane_slots is not None and len(lane_slots) != int(idx.numel()):
            raise ValueError("lane_slots length must match edge_indices length")

        for i in range(int(idx.numel())):
            e = int(idx[i].item())
            if e < 0 or e >= n:
                continue
            if lane_slots is None:
                slot = 0
            else:
                slot = int(lane_slots[i])
                if slot < 0 or slot > 63:
                    raise ValueError(f"lane slot out of range [0,63]: {slot}")
            old_mask = int(mask_flat[e].item())
            new_mask = old_mask | (1 << int(slot))
            mask_flat[e] = int(new_mask)
            count_flat[e] = int(int(new_mask).bit_count())
            lane_flat[e] = bool(new_mask != 0)
            if int(slot) > int(self.runtime_max_lane_slot):
                self.runtime_max_lane_slot = int(slot)
            # Update undirected width accumulator only for newly allocated bits
            if new_mask != old_mask:
                r = int(rev_flat[e].item())
                leader = min(e, r) if 0 <= r < n else e
                accum_flat[leader] = float(accum_flat[leader].item()) + lw
                if 0 <= r < n and r != leader:
                    accum_flat[r] = float(accum_flat[leader].item())

    @property
    def lane_map(self) -> torch.Tensor:
        """Dense lane-slot map ``[H,W,N,4]`` (bool)."""
        h, w = self.shape
        n = int(self.runtime_max_lane_slot) + 1
        if n <= 0:
            return torch.zeros((h, w, 0, 4), dtype=torch.bool, device=self.device)
        out = torch.zeros((h, w, n, 4), dtype=torch.bool, device=self.device)
        mask = self.edge_lane_mask
        for s in range(n):
            out[:, :, s, :] = ((mask >> int(s)) & 1).to(dtype=torch.bool)
        return out

    # ------------------------------------------------------------------
    # Flow helpers (formerly LaneFlowGraph methods)
    # ------------------------------------------------------------------

    @property
    def flow_count(self) -> int:
        return int(self.weights.shape[0])

    def ordered_flow_index(self, step_count: int) -> Optional[int]:
        pos = int(step_count)
        if pos < 0 or pos >= int(self.flow_order.numel()):
            return None
        return int(self.flow_order[pos].item())

    def current_flow_index(self) -> Optional[int]:
        return self.ordered_flow_index(self.step_count)

    def flow_pair(self, flow_index: int) -> Tuple[str, str, float]:
        i = int(flow_index)
        return self.src_gid[i], self.dst_gid[i], float(self.weights[i].item())

    def valid_ports(self, flow_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(flow_index)
        src = self.src_ports[i][self.src_mask[i]]
        dst = self.dst_ports[i][self.dst_mask[i]]
        return src, dst

    def flow_reverse_allow(self, flow_index: int) -> bool:
        """Per-flow reverse_allow (falls back to global RoutingConfig default)."""
        v = self.flow_specs[int(flow_index)].reverse_allow
        return bool(v) if v is not None else self._route_reverse_allow

    def flow_merge_allow(self, flow_index: int) -> bool:
        """Per-flow merge_allow (falls back to global RoutingConfig default)."""
        v = self.flow_specs[int(flow_index)].merge_allow
        return bool(v) if v is not None else self._route_merge_allow

    def remaining_weight(self) -> float:
        rem = self.weights.masked_fill(
            self.routed_mask.to(dtype=torch.bool, device=self.device), 0.0
        ).sum()
        return float(rem.item())

    def remaining_weight_ratio(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.remaining_weight() / float(self.total_weight)

    def remaining_flow_count(self) -> int:
        return int((~self.routed_mask).sum().item())

    @property
    def done(self) -> bool:
        return bool(self.step_count >= int(self.flow_count))

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def apply_route(self, route: LaneRoute) -> None:
        if self.done:
            raise RuntimeError("cannot apply route: already done")
        cur = self.current_flow_index()
        if cur is None:
            raise RuntimeError("current flow is None")
        # Stash the active flow's lane_width so allocator and apply_edges agree
        self._current_lane_width = float(self.flow_specs[int(cur)].lane_width)
        if int(route.flow_index) != int(cur):
            raise ValueError(
                f"route.flow_index={route.flow_index} does not match current flow={cur}"
            )
        e = route.edge_indices.to(device=self.device, dtype=torch.long).view(-1)
        lane_slots: List[int] = []
        planned = route.planned_lane_slots
        if isinstance(planned, torch.Tensor) and int(planned.numel()) == int(e.numel()):
            p = planned.to(device=self.device, dtype=torch.long).view(-1)
            for i in range(int(p.numel())):
                s = int(p[i].item())
                if s < 0:
                    raise ValueError("planned_lane_slots must be non-negative for active edges")
                lane_slots.append(s)
        else:
            ma = self.flow_merge_allow(int(cur))
            ra = self.flow_reverse_allow(int(cur))
            for i in range(int(e.numel())):
                lane_slots.append(int(self._lane_slot_for_edge(
                    int(e[i].item()), merge_allow=ma, reverse_allow=ra)))
            route.planned_lane_slots = torch.tensor(lane_slots, dtype=torch.long, device=self.device)
        self.apply_edges(e, lane_slots=lane_slots)
        self.route_lane_slots_by_flow[int(cur)] = tuple(lane_slots)
        self.routed_mask[int(cur)] = True
        self.step_count += 1

    def step(self, *, apply: bool, route: Optional[LaneRoute] = None) -> None:
        if not apply:
            return
        if route is None:
            raise ValueError("step(apply=True) requires route")
        self.apply_route(route)

    # ------------------------------------------------------------------
    # Routing configuration
    # ------------------------------------------------------------------

    def configure_routing(self, config: RoutingConfig) -> None:
        """Apply routing configuration (algorithm selection, limits).

        In ``benchmark`` mode, times each candidate algorithm on sample
        flows and picks the fastest per single/multi-src method.
        """
        self._max_wave_iters = int(config.max_wave_iters)
        self._max_path_steps = int(config.max_path_steps)
        self._route_merge_allow = bool(config.merge_allow)
        self._route_reverse_allow = bool(config.reverse_allow)
        self._route_capacity_epsilon = float(config.capacity_epsilon)

        from .slot_allocator import LaneSlotAllocator
        self._allocator = LaneSlotAllocator(
            capacity_epsilon=float(config.capacity_epsilon),
        )

        mode = (config.selection or "static").strip().lower()
        if mode == "benchmark":
            from .benchmark import pick_best_algorithms
            algos = pick_best_algorithms(self, config)
            self._algorithm = algos.get("single_src", config.algorithm or "wavefront")
            self._algorithm_multi_src = algos.get("multi_src", config.algorithm or "wavefront")
        elif mode == "static":
            algo = (config.algorithm or "wavefront").strip().lower()
            self._algorithm = algo
            self._algorithm_multi_src = algo
        else:
            raise ValueError(
                f"unknown routing selection {config.selection!r}; expected static|benchmark"
            )

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def allow_mask(self) -> torch.Tensor:
        """Per-cell-per-direction mask that forbids reverse-direction edges.

        ``result[y, x, d] == False`` iff the directed edge leaving cell
        ``(x, y)`` in direction ``d`` would walk against an already-placed lane.
        """
        h = int(self.grid_height)
        w = int(self.grid_width)
        eid_flat = torch.arange(h * w * 4, device=self.device, dtype=torch.long)
        rev = self.reverse_edge_lut[eid_flat]
        forbidden = self.lane_dir_flat[rev].view(h, w, 4)
        return ~forbidden

    def capacity_mask(
        self,
        lane_width: Optional[float] = None,
        *,
        merge_allow: Optional[bool] = None,
        reverse_allow: Optional[bool] = None,
    ) -> torch.Tensor:
        """Per-cell-per-direction feasibility mask for the given ``lane_width``.

        ``result[y, x, d] == True`` iff the directed edge can accept a lane of
        the requested width (either via slot reuse or remaining capacity).
        Delegates to :class:`LaneSlotAllocator`.

        Defaults to ``self._current_lane_width`` when *lane_width* is None.
        Per-flow *merge_allow* / *reverse_allow* override global defaults.
        """
        lw = float(lane_width) if lane_width is not None else float(self._current_lane_width)
        ma = merge_allow if merge_allow is not None else self._route_merge_allow
        ra = reverse_allow if reverse_allow is not None else self._route_reverse_allow
        return self._ensure_allocator().capacity_mask(
            self, lw, merge_allow=ma, reverse_allow=ra)

    def combined_mask(self, flow_index: int) -> Optional[torch.Tensor]:
        """Unified pathfind mask combining reverse-direction + capacity constraints.

        Returns ``None`` when no constraint is active (all edges feasible),
        otherwise ``bool [H, W, 4]``.
        """
        ra = self.flow_reverse_allow(flow_index)
        ma = self.flow_merge_allow(flow_index)
        am_cap = self.capacity_mask(merge_allow=ma, reverse_allow=ra)
        if not ra:
            am_opp = self.allow_mask()
            if am_cap.all():
                return am_opp
            return am_opp & am_cap
        if am_cap.all():
            return None
        return am_cap

    def pathfind(
        self,
        *,
        src_xy: Tuple[int, int],
        dst_xy: torch.Tensor,
        src_gid: Optional[str] = None,
        dst_gid: Optional[str] = None,
        allow_mask: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Find one shortest path from *src_xy* to any point in *dst_xy*.

        *src_gid* / *dst_gid* are kept for compatibility with existing call
        sites.  The walkable map is static for lane-generation and already
        includes all known port cells.
        """
        return self._dispatch_pathfind(
            algorithm=self._algorithm,
            src_xy=src_xy, dst_xy=dst_xy,
            src_gid=src_gid, dst_gid=dst_gid,
            allow_mask=allow_mask, rng=rng,
        )

    def pathfind_batch(
        self,
        *,
        src_xy_list: Sequence[Tuple[int, int]],
        dst_xy: torch.Tensor,
        src_gid: Optional[str] = None,
        dst_gid: Optional[str] = None,
        allow_mask: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> List[Optional[List[Tuple[int, int]]]]:
        """Find paths for multiple sources to the same destination set.

        Internally selects the multi-src algorithm variant when
        ``len(src_xy_list) > 1``.
        """
        algo = self._algorithm_multi_src if len(src_xy_list) > 1 else self._algorithm
        return [
            self._dispatch_pathfind(
                algorithm=algo,
                src_xy=s, dst_xy=dst_xy,
                src_gid=src_gid, dst_gid=dst_gid,
                allow_mask=allow_mask, rng=rng,
            )
            for s in src_xy_list
        ]

    def _dispatch_pathfind(
        self,
        *,
        algorithm: str,
        src_xy: Tuple[int, int],
        dst_xy: torch.Tensor,
        src_gid: Optional[str],
        dst_gid: Optional[str],
        allow_mask: Optional[torch.Tensor],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        algo = (algorithm or "wavefront").strip().lower()
        if algo == "wavefront":
            return self._pathfind_wavefront(
                src_xy=src_xy, dst_xy=dst_xy,
                src_gid=src_gid, dst_gid=dst_gid,
                allow_mask=allow_mask, rng=rng,
            )
        if algo in ("dijkstra", "astar"):
            return self._pathfind_graph_search(
                algorithm=algo, src_xy=src_xy, dst_xy=dst_xy,
                src_gid=src_gid, dst_gid=dst_gid,
                allow_mask=allow_mask, rng=rng,
            )
        raise ValueError(f"unknown routing algorithm {algo!r}; expected wavefront|dijkstra|astar")

    def _pathfind_wavefront(
        self,
        *,
        src_xy: Tuple[int, int],
        dst_xy: torch.Tensor,
        src_gid: Optional[str],
        dst_gid: Optional[str],
        allow_mask: Optional[torch.Tensor],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        from .wavefront import backtrace_shortest_path, _backtrace_with_mask
        _ = src_gid, dst_gid  # fixed static walkable map does not vary by src/dst

        dist = self.get_dist_map_for_targets(
            targets_xy=dst_xy,
            allow_mask=allow_mask,
            max_wave_iters=self._max_wave_iters,
        )
        if allow_mask is not None:
            dist_np = dist.detach().cpu().numpy()
            allow_np = allow_mask.detach().cpu().numpy()
            if rng is not None:
                seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
                py_rng = _random.Random(seed)
            else:
                py_rng = _random.Random()
            return _backtrace_with_mask(
                dist_np=dist_np,
                src_xy=src_xy,
                py_rng=py_rng,
                max_steps=self._max_path_steps,
                allow_4d_np=allow_np,
            )
        return backtrace_shortest_path(
            dist=dist, src_xy=src_xy, rng=rng, max_steps=self._max_path_steps,
        )

    def _pathfind_graph_search(
        self,
        *,
        algorithm: str,
        src_xy: Tuple[int, int],
        dst_xy: torch.Tensor,
        src_gid: Optional[str],
        dst_gid: Optional[str],
        allow_mask: Optional[torch.Tensor],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        from .dijkstra import _route_graph_search
        _ = src_gid, dst_gid  # fixed static walkable map does not vary by src/dst

        free_np = self._get_static_walkable_map().detach().cpu().numpy()
        allow_np = allow_mask.detach().cpu().numpy() if allow_mask is not None else None
        goals = [(int(p[0]), int(p[1])) for p in dst_xy.detach().cpu().tolist()]
        return _route_graph_search(
            free_np=free_np,
            src_xy=src_xy,
            goals_xy=goals,
            algorithm=algorithm,
            max_path_steps=self._max_path_steps,
            allow_np=allow_np,
            rng=rng,
        )
