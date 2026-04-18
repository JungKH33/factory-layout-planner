"""Dijkstra / A* pathfinding primitive (sequential host-side heap search).

Both algorithms share the same search loop; the only difference is whether
a Manhattan heuristic is applied (A*) or not (Dijkstra).
"""
from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Set, Tuple

import torch


def _route_graph_search(
    *,
    free_np,
    src_xy: Tuple[int, int],
    goals_xy: Sequence[Tuple[int, int]],
    algorithm: str,
    max_path_steps: int,
    allow_np,
    rng: Optional[torch.Generator],
) -> Optional[List[Tuple[int, int]]]:
    """Sequential dijkstra/A* on a numpy ``free_np`` grid.

    ``allow_np[y, x, d]`` (optional) gates outgoing direction ``d`` from
    cell ``(x, y)``: when False the relax is skipped, enforcing
    reverse-direction constraints at search time.
    """
    h, w = int(free_np.shape[0]), int(free_np.shape[1])
    sx, sy = int(src_xy[0]), int(src_xy[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return None
    if not bool(free_np[sy, sx]):
        return None

    goals: Set[Tuple[int, int]] = {
        (int(x), int(y))
        for x, y in goals_xy
        if 0 <= int(x) < w and 0 <= int(y) < h and bool(free_np[int(y), int(x)])
    }
    if not goals:
        return None

    base_dirs: List[Tuple[int, int, int]] = [
        (1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3),
    ]
    if rng is not None:
        perm = torch.randperm(4, generator=rng).tolist()
        dirs = [base_dirs[int(i)] for i in perm]
    else:
        dirs = base_dirs

    use_h = (algorithm == "astar")

    def heuristic(x: int, y: int) -> int:
        if not use_h:
            return 0
        return min(abs(gx - int(x)) + abs(gy - int(y)) for gx, gy in goals)

    g_best = {(sx, sy): 0}
    prev = {(sx, sy): None}
    pq: List[Tuple[int, int, int, int]] = [(heuristic(sx, sy), 0, sx, sy)]
    cap = int(max_path_steps)

    while pq:
        _f, g, x, y = heapq.heappop(pq)
        if g != g_best.get((x, y), None):
            continue
        if (x, y) in goals:
            path: List[Tuple[int, int]] = []
            cur: Optional[Tuple[int, int]] = (x, y)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path

        ng = int(g) + 1
        if cap > 0 and ng > cap:
            continue

        for dx, dy, d in dirs:
            nx, ny = int(x + dx), int(y + dy)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if not free_np[ny, nx]:
                continue
            if allow_np is not None and not bool(allow_np[y, x, d]):
                continue

            old = g_best.get((nx, ny), None)
            if old is not None and int(ng) >= int(old):
                continue

            g_best[(nx, ny)] = int(ng)
            prev[(nx, ny)] = (x, y)
            nf = int(ng) + int(heuristic(nx, ny))
            heapq.heappush(pq, (nf, int(ng), nx, ny))

    return None
