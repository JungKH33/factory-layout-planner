from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


_DIR_TO_DXY = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}
_DXY_TO_DIR = {v: k for k, v in _DIR_TO_DXY.items()}
_REV_DIR = torch.tensor([2, 3, 0, 1], dtype=torch.long)


@dataclass
class LaneMaps:
    """Static + runtime lane maps.

    Directed edge indexing:
    - edge_id = ((y * W + x) * 4 + dir)
    - dir: 0:right, 1:down, 2:left, 3:up
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

    @classmethod
    def build(
        cls,
        *,
        grid_height: int,
        grid_width: int,
        blocked_static: torch.Tensor,
        device: torch.device,
    ) -> "LaneMaps":
        h = int(grid_height)
        w = int(grid_width)
        dev = torch.device(device)
        blocked = blocked_static.to(device=dev, dtype=torch.bool)
        if tuple(blocked.shape) != (h, w):
            raise ValueError(f"blocked_static must be {(h,w)}, got {tuple(blocked.shape)}")

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

        return cls(
            grid_height=h,
            grid_width=w,
            device=dev,
            blocked_static=blocked,
            edge_valid_flat=edge_valid,
            reverse_edge_lut=reverse,
            edge_src_cell=src,
            edge_dst_cell=dst,
            lane_dir_flat=torch.zeros((n_edges,), dtype=torch.bool, device=dev),
        )

    def copy(self) -> "LaneMaps":
        return LaneMaps(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            device=self.device,
            blocked_static=self.blocked_static,
            edge_valid_flat=self.edge_valid_flat,
            reverse_edge_lut=self.reverse_edge_lut,
            edge_src_cell=self.edge_src_cell,
            edge_dst_cell=self.edge_dst_cell,
            lane_dir_flat=self.lane_dir_flat.clone(),
        )

    def restore(self, src: "LaneMaps") -> None:
        if not isinstance(src, LaneMaps):
            raise TypeError(f"src must be LaneMaps, got {type(src).__name__}")
        if (self.grid_height, self.grid_width) != (src.grid_height, src.grid_width):
            raise ValueError("grid shape mismatch")
        self.lane_dir_flat.copy_(src.lane_dir_flat.to(device=self.device, dtype=torch.bool))

    def reset_runtime(self) -> None:
        self.lane_dir_flat.zero_()

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

    def path_to_edge_ids_and_turns(self, path_xy: Sequence[Tuple[int, int]]) -> Tuple[torch.Tensor, int]:
        if len(path_xy) < 2:
            return torch.empty((0,), dtype=torch.long, device=self.device), 0

        eids: List[int] = []
        prev_dir: int | None = None
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

        # ordered unique edges to avoid duplicate-length inflation
        seen = set()
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
