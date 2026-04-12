"""Lane adapters — candidate generation strategies.

Each adapter generates K route candidates for the current flow and packs
them into an :class:`ActionSpace`.  The :class:`BaseLaneAdapter` provides
the common scaffolding (scoring, tensor packing, action resolution); concrete
subclasses implement :meth:`_generate_candidates` with their own strategy.

Hierarchy mirrors group_placement ``BaseAdapter`` → concrete adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch

from .action import LaneRoute
from .action_space import ActionSpace

if TYPE_CHECKING:
    from .state import LaneState


@dataclass(frozen=True)
class LaneAdapterConfig:
    candidate_k: int = 8
    random_seed: int = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_candidates_with_builder(
    *,
    state: "LaneState",
    src_ports: torch.Tensor,
    k_target: int,
    build_path,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Run primary + diversity passes around a path-building closure."""
    candidates: List[torch.Tensor] = []
    turns_l: List[int] = []
    seen: Set[Tuple[int, ...]] = set()
    src_list = src_ports.detach().cpu().tolist()

    for p in src_list:
        path = build_path((int(p[0]), int(p[1])))
        if not path:
            continue
        edges, turns = state.path_to_edge_ids_and_turns(path)
        key = tuple(int(x) for x in edges.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(edges)
        turns_l.append(int(turns))
        if len(candidates) >= int(k_target):
            break

    retry = 0
    while len(candidates) < int(k_target) and retry < int(k_target) * 4 and len(src_list) > 0:
        p = src_list[retry % len(src_list)]
        path = build_path((int(p[0]), int(p[1])))
        retry += 1
        if not path:
            continue
        edges, turns = state.path_to_edge_ids_and_turns(path)
        key = tuple(int(x) for x in edges.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(edges)
        turns_l.append(int(turns))

    return candidates, turns_l


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseLaneAdapter(ABC):
    """Abstract base for lane adapters.

    Subclasses must implement :meth:`_generate_candidates`.  Scoring,
    tensor packing, and action resolution are handled here.
    """

    def __init__(self, *, config: Optional[LaneAdapterConfig] = None) -> None:
        self.config = config or LaneAdapterConfig()
        self.env = None
        self._rng = torch.Generator()
        self._rng.manual_seed(int(self.config.random_seed))

    def bind(self, env) -> None:
        self.env = env

    # ---- abstract --------------------------------------------------------

    @abstractmethod
    def _generate_candidates(
        self,
        *,
        state: "LaneState",
        flow_idx: int,
        k: int,
        rng: torch.Generator,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Return ``(edge_id_tensors, turn_counts)`` for up to *k* candidates."""

    # ---- concrete --------------------------------------------------------

    def _empty_action_space(self, *, flow_index: int, device: torch.device) -> ActionSpace:
        return ActionSpace(
            flow_index=int(flow_index),
            candidate_edge_idx=torch.zeros((0, 0), dtype=torch.long, device=device),
            candidate_edge_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
            valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    def build_action_space(self) -> ActionSpace:
        if self.env is None:
            raise RuntimeError("adapter is not bound to env")
        state = self.env.get_state()
        flow_idx = state.current_flow_index()
        if flow_idx is None:
            return self._empty_action_space(flow_index=-1, device=state.device)

        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        candidates, turns_l = self._generate_candidates(
            state=state,
            flow_idx=int(flow_idx),
            k=max(1, int(self.config.candidate_k)),
            rng=self._rng,
        )

        if len(candidates) == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        k = len(candidates)
        lmax = max(int(c.numel()) for c in candidates)
        edge_idx = torch.zeros((k, lmax), dtype=torch.long, device=state.device)
        edge_mask = torch.zeros((k, lmax), dtype=torch.bool, device=state.device)
        path_len = torch.zeros((k,), dtype=torch.float32, device=state.device)
        turns_t = torch.tensor(turns_l, dtype=torch.float32, device=state.device)

        for i, edges in enumerate(candidates):
            n = int(edges.numel())
            if n > 0:
                edge_idx[i, :n] = edges
                edge_mask[i, :n] = True
            path_len[i] = float(n)

        valid = edge_mask.any(dim=1)

        costs = self.env.reward_composer.delta_batch(
            state,
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            candidate_turns=turns_t,
        )

        return ActionSpace(
            flow_index=int(flow_idx),
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            valid_mask=valid,
            candidate_path_len=path_len,
            candidate_turns=turns_t,
            candidate_cost=costs,
        )

    def resolve_action(self, action_idx: int, action_space: ActionSpace) -> Optional[LaneRoute]:
        i = int(action_idx)
        k = int(action_space.valid_mask.shape[0])
        if i < 0 or i >= k:
            return None
        if not bool(action_space.valid_mask[i].item()):
            return None

        edges = action_space.candidate_edge_idx[i][action_space.candidate_edge_mask[i]]
        turns = int(action_space.candidate_turns[i].item()) if action_space.candidate_turns is not None else 0
        path_len = (
            float(action_space.candidate_path_len[i].item())
            if action_space.candidate_path_len is not None
            else float(edges.numel())
        )
        return LaneRoute(
            flow_index=int(action_space.flow_index),
            candidate_index=i,
            edge_indices=edges,
            path_length=path_len,
            turns=turns,
        )


# ---------------------------------------------------------------------------
# Concrete: direct shortest-path candidates
# ---------------------------------------------------------------------------


class DirectRouteAdapter(BaseLaneAdapter):
    """Generate K shortest-path candidates via :meth:`LaneState.pathfind`.

    Calls :meth:`pathfind` for each source port through
    :func:`_build_candidates_with_builder` (primary + diversity passes).
    """

    def _generate_candidates(self, *, state, flow_idx, k, rng):
        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return [], []

        src_gid, dst_gid, _ = state.flow_pair(flow_idx)
        allow_mask = state.allow_mask() if state.forbid_opposite(flow_idx) else None

        def build_path(src_xy):
            return state.pathfind(
                src_xy=src_xy,
                dst_xy=dst_ports,
                src_gid=src_gid,
                dst_gid=dst_gid,
                allow_mask=allow_mask,
                rng=rng,
            )

        return _build_candidates_with_builder(
            state=state,
            src_ports=src_ports,
            k_target=int(k),
            build_path=build_path,
        )


class HananAdapter(BaseLaneAdapter):
    """Generate candidates by selecting one Hanan point per action.

    Each candidate index corresponds to one feasible Hanan point.  The adapter
    resolves that point into a concrete route by stitching:

    - ``src_port -> hanan_point``
    - ``hanan_point -> any dst_port``

    Notes:
        - Hanan points that fall on blocked static cells are excluded.
        - This adapter still emits one concrete route per env step (single-flow).
        - TODO(B): to support true multi-flow Steiner-style placement in one
          action, extend action/env schema to commit a batch of routes at once.
    """

    @staticmethod
    def _build_hanan_points(*, state: "LaneState", src_ports: torch.Tensor, dst_ports: torch.Tensor) -> List[Tuple[int, int]]:
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return []
        ports = torch.cat([src_ports, dst_ports], dim=0).to(dtype=torch.long, device=state.device)
        xs = torch.unique(ports[:, 0]).detach().cpu().tolist()
        ys = torch.unique(ports[:, 1]).detach().cpu().tolist()

        h = int(state.grid_height)
        w = int(state.grid_width)
        blocked = state.blocked_static
        points: List[Tuple[int, int]] = []
        for yv in ys:
            y = int(yv)
            if y < 0 or y >= h:
                continue
            for xv in xs:
                x = int(xv)
                if x < 0 or x >= w:
                    continue
                # User request: blocked Hanan points are discarded.
                if bool(blocked[y, x].item()):
                    continue
                points.append((x, y))
        return points

    def _generate_candidates(self, *, state, flow_idx, k, rng):
        src_ports, dst_ports = state.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return [], []

        src_gid, dst_gid, _ = state.flow_pair(flow_idx)
        allow_mask = state.allow_mask() if state.forbid_opposite(flow_idx) else None
        hanan_points = self._build_hanan_points(state=state, src_ports=src_ports, dst_ports=dst_ports)
        if len(hanan_points) == 0:
            return [], []

        # Keep reproducible but diverse candidate order under the adapter RNG.
        if len(hanan_points) > 1:
            perm = torch.randperm(len(hanan_points), generator=rng).detach().cpu().tolist()
            ordered_points = [hanan_points[int(i)] for i in perm]
        else:
            ordered_points = hanan_points

        src_list = [(int(p[0]), int(p[1])) for p in src_ports.detach().cpu().tolist()]
        seen: Set[Tuple[int, ...]] = set()
        packed: List[Tuple[torch.Tensor, int, int]] = []  # (edges, turns, edge_count)

        for hx, hy in ordered_points:
            hp_xy = (int(hx), int(hy))
            hp_t = torch.tensor([[hx, hy]], dtype=torch.long, device=state.device)

            p2 = state.pathfind(
                src_xy=hp_xy,
                dst_xy=dst_ports,
                dst_gid=dst_gid,
                allow_mask=allow_mask,
                rng=rng,
            )
            if not p2:
                continue

            best_path = None
            best_cost = None
            for sxy in src_list:
                p1 = state.pathfind(
                    src_xy=sxy,
                    dst_xy=hp_t,
                    src_gid=src_gid,
                    allow_mask=allow_mask,
                    rng=rng,
                )
                if not p1:
                    continue
                merged = p1 + p2[1:]
                if len(merged) < 2:
                    continue
                edge_est = len(merged) - 1
                if best_path is None or edge_est < int(best_cost):
                    best_path = merged
                    best_cost = edge_est

            if best_path is None:
                continue

            edges, turns = state.path_to_edge_ids_and_turns(best_path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            packed.append((edges, int(turns), int(edges.numel())))

        if len(packed) == 0:
            return [], []

        # Prefer shorter and straighter routes first.
        packed.sort(key=lambda it: (int(it[2]), int(it[1])))
        top = packed[: max(1, int(k))]
        return [it[0] for it in top], [int(it[1]) for it in top]


# backward compatibility
LaneAdapter = DirectRouteAdapter
