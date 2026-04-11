from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch


_WAVEFRONT_SYNC_EVERY = 16


def wavefront_distance_field(
    *,
    free_map: torch.Tensor,
    seeds_xy: torch.Tensor,
    max_iters: int = 0,
) -> torch.Tensor:
    """Compute shortest-path distance field on 4-neighbor grid.

    Returns int32 [H,W], unreachable=-1.

    Runs entirely on ``free_map.device`` (CPU or CUDA). Termination is checked
    once every ``_WAVEFRONT_SYNC_EVERY`` iterations (amortized device sync)
    instead of every step, and the absolute iteration cap is ``H*W`` — enough
    to handle maze-shaped grids where BFS depth can exceed ``H+W``.
    """
    if free_map.dim() != 2:
        raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")
    h, w = int(free_map.shape[0]), int(free_map.shape[1])
    device = free_map.device
    dist = torch.full((h, w), -1, dtype=torch.int32, device=device)

    if seeds_xy.numel() == 0:
        return dist

    seeds = seeds_xy.to(device=device, dtype=torch.long).view(-1, 2)
    sx = seeds[:, 0]
    sy = seeds[:, 1]
    inb = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sx = sx[inb]
    sy = sy[inb]
    if sx.numel() == 0:
        return dist

    frontier = torch.zeros((h, w), dtype=torch.bool, device=device)
    frontier[sy, sx] = True
    frontier &= free_map

    cap = int(max_iters) if int(max_iters) > 0 else (h * w)
    step = 0
    while step < cap:
        dist.masked_fill_(frontier, step)

        nxt = torch.zeros_like(frontier)
        nxt[1:, :] |= frontier[:-1, :]
        nxt[:-1, :] |= frontier[1:, :]
        nxt[:, 1:] |= frontier[:, :-1]
        nxt[:, :-1] |= frontier[:, 1:]
        nxt &= free_map
        nxt &= (dist < 0)
        frontier = nxt
        step += 1

        if step % _WAVEFRONT_SYNC_EVERY == 0:
            if not bool(frontier.any().item()):
                break

    return dist


def wavefront_distance_field_batched(
    *,
    free_map: torch.Tensor,
    seeds_xy: torch.Tensor,
    seeds_mask: torch.Tensor,
    max_iters: int = 0,
) -> torch.Tensor:
    """Batched 4-neighbor BFS distance fields, one per flow.

    Args:
        free_map: ``[H,W]`` bool. True = walkable. Shared across all flows.
        seeds_xy: ``[M,K,2]`` long. ``[m,k]`` is the (x,y) of the k-th seed
            for flow m. Padding entries are tolerated; ``seeds_mask`` selects
            real ones.
        seeds_mask: ``[M,K]`` bool. True where ``seeds_xy[m,k]`` is a real
            seed for flow m.
        max_iters: hard cap on BFS iterations. ``0`` means ``H*W``.

    Returns:
        ``[M,H,W]`` int32 distance field. Unreachable cells = -1. Flows with
        zero valid seeds get an all -1 row.

    All M frontiers run in a single PyTorch loop, so the per-iteration cost
    is one set of 3D shifts on a ``[M,H,W]`` bool tensor instead of M
    independent 2D loops. Memory: ``M*H*W*5`` bytes (int32 dist + bool
    frontier + scratch).
    """
    if free_map.dim() != 2:
        raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")
    if seeds_xy.dim() != 3 or seeds_xy.shape[-1] != 2:
        raise ValueError(f"seeds_xy must be [M,K,2], got {tuple(seeds_xy.shape)}")
    if seeds_mask.dim() != 2 or seeds_mask.shape[:2] != seeds_xy.shape[:2]:
        raise ValueError(
            f"seeds_mask must be [M,K] matching seeds_xy, got {tuple(seeds_mask.shape)}"
        )

    device = free_map.device
    h, w = int(free_map.shape[0]), int(free_map.shape[1])
    m = int(seeds_xy.shape[0])
    dist = torch.full((m, h, w), -1, dtype=torch.int32, device=device)
    if m == 0:
        return dist

    xs = seeds_xy[..., 0].to(device=device, dtype=torch.long)
    ys = seeds_xy[..., 1].to(device=device, dtype=torch.long)
    inb = (
        seeds_mask.to(device=device, dtype=torch.bool)
        & (xs >= 0) & (xs < w)
        & (ys >= 0) & (ys < h)
    )

    frontier = torch.zeros((m, h, w), dtype=torch.bool, device=device)
    if bool(inb.any()):
        flow_idx = (
            torch.arange(m, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand_as(inb)
        )
        sel_m = flow_idx[inb]
        sel_y = ys[inb]
        sel_x = xs[inb]
        frontier[sel_m, sel_y, sel_x] = True
    free_b = free_map.unsqueeze(0)
    frontier &= free_b

    cap = int(max_iters) if int(max_iters) > 0 else (h * w)
    step = 0
    while step < cap:
        dist.masked_fill_(frontier, step)

        nxt = torch.zeros_like(frontier)
        nxt[:, 1:, :] |= frontier[:, :-1, :]
        nxt[:, :-1, :] |= frontier[:, 1:, :]
        nxt[:, :, 1:] |= frontier[:, :, :-1]
        nxt[:, :, :-1] |= frontier[:, :, 1:]
        nxt &= free_b
        nxt &= (dist < 0)
        frontier = nxt
        step += 1

        if step % _WAVEFRONT_SYNC_EVERY == 0:
            if not bool(frontier.any().item()):
                break

    return dist


_NEIGHBOR_DX = (1, 0, -1, 0)
_NEIGHBOR_DY = (0, 1, 0, -1)


def backtrace_shortest_path(
    *,
    dist: torch.Tensor,
    src_xy: Tuple[int, int],
    rng: Optional[torch.Generator] = None,
    max_steps: int = 0,
) -> Optional[List[Tuple[int, int]]]:
    """Backtrace one shortest path from src to nearest zero-distance seed.

    Sequential walk: reads one cell per step, picks a neighbor that is one
    step closer, and repeats. This can't be vectorized on GPU, so ``dist`` is
    copied to host memory once up front and the loop runs on plain Python/
    numpy scalars (no per-step device sync).
    """
    if dist.dim() != 2:
        raise ValueError("dist must be [H,W]")
    h, w = int(dist.shape[0]), int(dist.shape[1])
    x, y = int(src_xy[0]), int(src_xy[1])
    if not (0 <= x < w and 0 <= y < h):
        return None

    dist_np = dist.detach().cpu().numpy()
    d0 = int(dist_np[y, x])
    if d0 < 0:
        return None

    # Derive a deterministic python RNG from the torch generator (if given) so
    # we can avoid torch.randperm inside the hot loop.
    if rng is not None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        py_rng = random.Random(seed)
    else:
        py_rng = random.Random()

    path: List[Tuple[int, int]] = [(x, y)]
    cur = d0
    prev_dir: Optional[int] = None
    steps = 0
    cap = int(max_steps)

    while cur > 0:
        if cap > 0 and steps >= cap:
            return None

        target = cur - 1
        options: List[Tuple[int, int, int]] = []
        for d in range(4):
            nx = x + _NEIGHBOR_DX[d]
            ny = y + _NEIGHBOR_DY[d]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if int(dist_np[ny, nx]) == target:
                options.append((nx, ny, d))
        if not options:
            return None

        if prev_dir is not None:
            straight = [o for o in options if o[2] == prev_dir]
            if straight:
                options = straight
        if len(options) > 1:
            pick = py_rng.randrange(len(options))
        else:
            pick = 0

        nx, ny, nd = options[pick]
        path.append((nx, ny))
        x, y = nx, ny
        prev_dir = nd
        cur = target
        steps += 1

    return path
