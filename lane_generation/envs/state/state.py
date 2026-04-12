"""Flat state object for the lane-generation env.

``LaneState`` flattens the directed-edge graph, the per-flow tensor bundle,
and the runtime placement bookkeeping into one object.

Lifecycle:

* :meth:`LaneState.build` constructs everything in one shot — edge tables
  and flow arrays.
* :meth:`LaneState.copy` shares the static graph and flow arrays by reference;
  only ``lane_dir_flat`` and ``routed_mask`` are cloned, so MCTS snapshots
  stay cheap.
* :meth:`LaneState.apply_route` records the new lane on ``lane_dir_flat`` and
  bumps ``generation`` so any cached distance fields invalidate themselves on
  the next :meth:`ensure_wavefront_dist` call.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    forbid_opposite: bool = False


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
    * Cache invalidation generation counter.
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

    generation: int = 0

    # Pathfinding algorithm configuration.
    _algorithm: str = "wavefront"
    _algorithm_multi_src: str = "wavefront"
    _max_wave_iters: int = 0
    _max_path_steps: int = 0

    # Lazy group-id → port-cells lookup (static, shared on copy).
    _gid_port_cache: Optional[Dict[str, torch.Tensor]] = None

    # Wavefront BFS distance cache (shared by ref on copy).
    # Keyed on ``(generation, src_gid, dst_gid)``.
    _wavefront_dist: Optional[torch.Tensor] = None   # [H,W] int32
    _wavefront_dist_gen: int = -1
    _wavefront_dist_src_gid: Optional[str] = None
    _wavefront_dist_dst_gid: Optional[str] = None

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
            generation=0,
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
            generation=int(self.generation),
        )
        c._algorithm = self._algorithm
        c._algorithm_multi_src = self._algorithm_multi_src
        c._max_wave_iters = self._max_wave_iters
        c._max_path_steps = self._max_path_steps
        c._gid_port_cache = self._gid_port_cache
        c._wavefront_dist = self._wavefront_dist
        c._wavefront_dist_gen = self._wavefront_dist_gen
        c._wavefront_dist_src_gid = self._wavefront_dist_src_gid
        c._wavefront_dist_dst_gid = self._wavefront_dist_dst_gid
        return c

    def restore(self, src: "LaneState") -> None:
        if not isinstance(src, LaneState):
            raise TypeError(f"src must be LaneState, got {type(src).__name__}")
        if (self.grid_height, self.grid_width) != (src.grid_height, src.grid_width):
            raise ValueError("grid shape mismatch")
        self.lane_dir_flat.copy_(src.lane_dir_flat.to(device=self.device, dtype=torch.bool))
        self.routed_mask.copy_(src.routed_mask.to(device=self.device, dtype=torch.bool))
        self.step_count = int(src.step_count)
        self.generation = int(src.generation)

    def reset_runtime(self) -> None:
        self.lane_dir_flat.zero_()
        self.routed_mask.zero_()
        self.step_count = 0
        self.generation = 0
        self._wavefront_dist = None
        self._wavefront_dist_gen = -1
        self._wavefront_dist_src_gid = None
        self._wavefront_dist_dst_gid = None

    # ------------------------------------------------------------------
    # Group port lookup & per-route free map
    # ------------------------------------------------------------------

    def _ensure_gid_port_cache(self) -> Dict[str, torch.Tensor]:
        if self._gid_port_cache is not None:
            return self._gid_port_cache
        cache: Dict[str, List[Tuple[int, int]]] = {}
        for i, spec in enumerate(self.flow_specs):
            for gid, ports in ((spec.src_gid, spec.src_ports), (spec.dst_gid, spec.dst_ports)):
                cells = cache.setdefault(gid, [])
                for xy in ports:
                    cells.append((int(xy[0]), int(xy[1])))
        result: Dict[str, torch.Tensor] = {}
        for gid, cells in cache.items():
            t = torch.tensor(cells, dtype=torch.long, device=self.device)
            result[gid] = torch.unique(t, dim=0) if t.numel() > 0 else t
        self._gid_port_cache = result
        return result

    def group_port_cells(self, gid: str) -> torch.Tensor:
        """Return ``[N, 2]`` unique port cells (x, y) for group *gid*."""
        cache = self._ensure_gid_port_cache()
        return cache.get(gid, torch.empty((0, 2), dtype=torch.long, device=self.device))

    def _free_map(
        self,
        src_gid: Optional[str] = None,
        dst_gid: Optional[str] = None,
    ) -> torch.Tensor:
        """Build a free-cell map, unblocking port cells for the given groups.

        Facility bodies are marked as blocked in ``blocked_static``, but the
        port cells of the source/destination facilities must be walkable so
        the pathfinder can enter/exit them.
        """
        base = (~self.blocked_static).to(dtype=torch.bool, device=self.device)
        h, w = int(self.grid_height), int(self.grid_width)
        for gid in (src_gid, dst_gid):
            if gid is None:
                continue
            for p in self.group_port_cells(gid):
                x, y = int(p[0].item()), int(p[1].item())
                if 0 <= y < h and 0 <= x < w:
                    base[y, x] = True
        return base

    # ------------------------------------------------------------------
    # Wavefront cache
    # ------------------------------------------------------------------

    def ensure_wavefront_dist(
        self,
        *,
        targets_xy: torch.Tensor,
        src_gid: Optional[str] = None,
        dst_gid: Optional[str] = None,
        allow_mask: Optional[torch.Tensor] = None,
        max_wave_iters: int = 0,
    ) -> torch.Tensor:
        """Return a ``[H,W]`` BFS distance field, rebuilding if stale.

        The result is cached and keyed on ``(generation, src_gid, dst_gid)``
        so that each group pair gets its own distance field with the correct
        port cells unblocked.
        """
        gen = int(self.generation)
        if (self._wavefront_dist is not None
                and self._wavefront_dist_gen == gen
                and self._wavefront_dist_src_gid == src_gid
                and self._wavefront_dist_dst_gid == dst_gid):
            return self._wavefront_dist

        from .wavefront import wavefront_distance_field, _wavefront_with_mask

        free_map = self._free_map(src_gid=src_gid, dst_gid=dst_gid)
        mi = int(max_wave_iters)
        if allow_mask is not None:
            dist = _wavefront_with_mask(
                free_map=free_map,
                allow_4d=allow_mask,
                seeds_xy=targets_xy,
                max_iters=mi,
            )
        else:
            dist = wavefront_distance_field(
                free_map=free_map,
                seeds_xy=targets_xy,
                max_iters=mi,
            )
        self._wavefront_dist = dist
        self._wavefront_dist_gen = gen
        self._wavefront_dist_src_gid = src_gid
        self._wavefront_dist_dst_gid = dst_gid
        return dist

    @property
    def wavefront_dist(self) -> Optional[torch.Tensor]:
        """Read-only access to the current wavefront cache (may be ``None``)."""
        return self._wavefront_dist

    @property
    def wavefront_dist_gen(self) -> int:
        return int(self._wavefront_dist_gen)

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

    def apply_edges(self, edge_indices: torch.Tensor) -> None:
        if edge_indices.numel() == 0:
            return
        idx = edge_indices.to(device=self.device, dtype=torch.long).view(-1)
        self.lane_dir_flat[idx] = True

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

    def forbid_opposite(self, flow_index: int) -> bool:
        return bool(self.flow_specs[int(flow_index)].forbid_opposite)

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
        if int(route.flow_index) != int(cur):
            raise ValueError(
                f"route.flow_index={route.flow_index} does not match current flow={cur}"
            )
        self.apply_edges(route.edge_indices)
        self.routed_mask[int(cur)] = True
        self.step_count += 1
        self.generation += 1

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
        """Per-cell-per-direction "allow" mask for forbid_opposite.

        ``result[y, x, d] == False`` iff the directed edge leaving cell
        ``(x, y)`` in direction ``d`` would walk against an already-placed lane.
        """
        h = int(self.grid_height)
        w = int(self.grid_width)
        eid_flat = torch.arange(h * w * 4, device=self.device, dtype=torch.long)
        rev = self.reverse_edge_lut[eid_flat]
        forbidden = self.lane_dir_flat[rev].view(h, w, 4)
        return ~forbidden

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

        When *src_gid* / *dst_gid* are given, the corresponding facility
        port cells are unblocked in the free map so the pathfinder can
        enter and exit them.
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

        dist = self.ensure_wavefront_dist(
            targets_xy=dst_xy,
            src_gid=src_gid, dst_gid=dst_gid,
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

        free_np = self._free_map(src_gid=src_gid, dst_gid=dst_gid).detach().cpu().numpy()
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
