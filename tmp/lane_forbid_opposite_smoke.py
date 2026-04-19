"""Smoke test for forbid_opposite routing across all three strategies.

Setup:
  * 16x16 empty grid
  * Flow A: (0, 0) -> (15, 15), forbid_opposite=True
  * Flow B: (15, 15) -> (0, 0), forbid_opposite=True

Procedure (per strategy):
  1. Build a fresh LaneState with the chosen routing strategy.
  2. Route flow A, commit it via state.apply_route(...).
  3. Route flow B and check that no candidate of B uses any directed edge
     that is the reverse of an edge already placed by flow A.
  4. Verify state.copy() preserves the placed-lane bookkeeping (a copied
     state with a new flow committed leaves the parent's lane_dir_flat
     untouched).

The "forbid" check is the load-bearing one: for every edge id ``e`` in
flow B's first candidate, ``state.lane_dir_flat[reverse_edge_lut[e]]``
must be False.

Run:
    python -m tmp.lane_forbid_opposite_smoke
"""
from __future__ import annotations

from typing import List

import torch

from lane_generation.envs.action import LaneRoute
from lane_generation.envs.routing import (
    AStarStrategy,
    DijkstraStrategy,
    RoutingConfig,
    RoutingStrategy,
    WavefrontStrategy,
)
from lane_generation.envs.state import LaneFlowSpec, LaneState, PortSelector, PortSpec


GRID = 16


def _build_state(algorithm: str) -> LaneState:
    ports = {
        "A_src.ex.0": PortSpec(port_id="A_src.ex.0", gid="A_src", xy=(0, 0), kind="exit"),
        "A_dst.en.0": PortSpec(port_id="A_dst.en.0", gid="A_dst", xy=(GRID - 1, GRID - 1), kind="entry"),
        "B_src.ex.0": PortSpec(port_id="B_src.ex.0", gid="B_src", xy=(GRID - 1, GRID - 1), kind="exit"),
        "B_dst.en.0": PortSpec(port_id="B_dst.en.0", gid="B_dst", xy=(0, 0), kind="entry"),
    }
    flows = [
        LaneFlowSpec(
            src=PortSelector(gid="A_src", port_ids=("A_src.ex.0",)),
            dst=PortSelector(gid="A_dst", port_ids=("A_dst.en.0",)),
            weight=1.0,
        ),
        LaneFlowSpec(
            src=PortSelector(gid="B_src", port_ids=("B_src.ex.0",)),
            dst=PortSelector(gid="B_dst", port_ids=("B_dst.en.0",)),
            weight=0.9,
        ),
    ]
    blocked = torch.zeros((GRID, GRID), dtype=torch.bool)
    return LaneState.build(
        grid_height=GRID,
        grid_width=GRID,
        blocked_static=blocked,
        flows=flows,
        port_specs=ports,
        port_groups={},
        device=torch.device("cpu"),
        flow_ordering="given",
    )


def _route_first(state: LaneState, flow_idx: int) -> torch.Tensor:
    cands, _ = state.routing.route_flow(
        state=state, flow_idx=flow_idx, k=4, rng=None,
    )
    if not cands:
        raise AssertionError(f"flow={flow_idx}: no candidates returned")
    return cands[0]


def _no_opposite(state: LaneState, edge_ids: torch.Tensor) -> bool:
    if edge_ids.numel() == 0:
        return True
    rev = state.reverse_edge_lut[edge_ids]
    return not bool(state.lane_dir_flat[rev].any().item())


def _check_one_strategy(name: str, expected_cls: type) -> None:
    print(f"[{name}]")
    state = _build_state(name)
    if not isinstance(state.routing, expected_cls):
        raise AssertionError(
            f"{name}: expected {expected_cls.__name__}, got {type(state.routing).__name__}"
        )

    edges_a = _route_first(state, flow_idx=0)
    print(f"  flow A edges: {int(edges_a.numel())}")

    state.apply_route(LaneRoute(
        flow_index=0,
        candidate_index=0,
        edge_indices=edges_a,
        path_length=float(edges_a.numel()),
    ))

    cands_b, _ = state.routing.route_flow(
        state=state, flow_idx=1, k=4, rng=None,
    )
    if not cands_b:
        raise AssertionError(f"{name}: flow B has no candidates after committing A")
    print(f"  flow B candidates: {len(cands_b)} (lengths: {[int(c.numel()) for c in cands_b]})")

    for i, cb in enumerate(cands_b):
        if not _no_opposite(state, cb):
            raise AssertionError(
                f"{name}: flow B candidate {i} contains an edge whose reverse "
                f"was placed by flow A (forbid_opposite violated)"
            )
    print(f"  forbid_opposite respected on all {len(cands_b)} B candidates")

    snap = state.copy()
    snap.apply_route(LaneRoute(
        flow_index=1,
        candidate_index=0,
        edge_indices=cands_b[0],
        path_length=float(cands_b[0].numel()),
    ))
    if bool(state.routed_mask[1].item()):
        raise AssertionError(f"{name}: parent state contaminated by snapshot apply_route")
    print("  state.copy() isolation OK\n")


def main() -> int:
    _check_one_strategy("wavefront", WavefrontStrategy)
    _check_one_strategy("dijkstra", DijkstraStrategy)
    _check_one_strategy("astar", AStarStrategy)

    print("All forbid_opposite smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
