from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def wavefront_distance_field(
    *,
    free_map: torch.Tensor,
    seeds_xy: torch.Tensor,
    max_iters: int = 0,
) -> torch.Tensor:
    """Compute shortest-path distance field on 4-neighbor grid.

    Returns int32 [H,W], unreachable=-1.
    """
    if free_map.dim() != 2:
        raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")
    h, w = int(free_map.shape[0]), int(free_map.shape[1])
    dist = torch.full((h, w), -1, dtype=torch.int32, device=free_map.device)

    if seeds_xy.numel() == 0:
        return dist

    seeds = seeds_xy.to(device=free_map.device, dtype=torch.long).view(-1, 2)
    sx = seeds[:, 0]
    sy = seeds[:, 1]
    inb = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    if not bool(inb.any().item()):
        return dist

    frontier = torch.zeros((h, w), dtype=torch.bool, device=free_map.device)
    frontier[sy[inb], sx[inb]] = True
    frontier &= free_map
    if not bool(frontier.any().item()):
        return dist

    cap = int(max_iters)
    step = 0
    while bool(frontier.any().item()):
        dist[frontier] = int(step)
        nxt = torch.zeros_like(frontier)
        nxt[1:, :] |= frontier[:-1, :]
        nxt[:-1, :] |= frontier[1:, :]
        nxt[:, 1:] |= frontier[:, :-1]
        nxt[:, :-1] |= frontier[:, 1:]
        nxt &= free_map
        nxt &= (dist < 0)
        frontier = nxt
        step += 1
        if cap > 0 and step >= cap:
            break

    return dist


def backtrace_shortest_path(
    *,
    dist: torch.Tensor,
    src_xy: Tuple[int, int],
    rng: Optional[torch.Generator] = None,
    max_steps: int = 0,
) -> Optional[List[Tuple[int, int]]]:
    """Backtrace one shortest path from src to nearest zero-distance seed."""
    if dist.dim() != 2:
        raise ValueError("dist must be [H,W]")
    h, w = int(dist.shape[0]), int(dist.shape[1])
    x, y = int(src_xy[0]), int(src_xy[1])
    if not (0 <= x < w and 0 <= y < h):
        return None
    d0 = int(dist[y, x].item())
    if d0 < 0:
        return None

    path: List[Tuple[int, int]] = [(x, y)]
    cur = d0
    prev_dir: Optional[int] = None
    steps = 0

    neighbors = [
        (1, 0, 0),
        (0, 1, 1),
        (-1, 0, 2),
        (0, -1, 3),
    ]

    while cur > 0:
        if int(max_steps) > 0 and steps >= int(max_steps):
            return None

        options: List[Tuple[int, int, int]] = []
        for dx, dy, d in neighbors:
            nx = x + int(dx)
            ny = y + int(dy)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if int(dist[ny, nx].item()) == cur - 1:
                options.append((nx, ny, int(d)))
        if not options:
            return None

        # Prefer straight continuation; random tie-break on same class.
        if prev_dir is not None:
            straight = [o for o in options if o[2] == prev_dir]
            if straight:
                options = straight
        if len(options) > 1:
            perm = torch.randperm(len(options), generator=rng)
            pick = int(perm[0].item())
        else:
            pick = 0

        nx, ny, nd = options[pick]
        path.append((nx, ny))
        x, y = nx, ny
        prev_dir = nd
        cur -= 1
        steps += 1

    return path
