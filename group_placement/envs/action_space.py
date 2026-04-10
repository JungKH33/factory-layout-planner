from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .action import GroupId


@dataclass(frozen=True)
class ActionSpace:
    """Factory layout action space.

    Core fields (always present):
      - centers: [N, 2] float32 candidate poses (x_center, y_center) — center coordinates.
        The engine resolves variant (rotation/mirror/shape) at step time.
      - valid_mask: [N] bool validity mask (True = valid)

    Geometric features (used for reward computation, Optional):
      - entry_points / exit_points:    [N, Emax/Xmax, 2] IO port absolute coordinates
      - entry_mask / exit_mask:        [N, Emax] / [N, Xmax]
      - min_x / max_x / min_y / max_y: [N] bounding box
    """

    # -- core (required) --
    centers: torch.Tensor     # [N, 2] float32 (x_center, y_center)
    valid_mask: torch.Tensor  # bool [N]
    group_id: Optional[GroupId] = None
    variant_indices: Optional[torch.Tensor] = None  # int64 [N] — per-action variant index

    # -- geometric features (Optional, for reward) --
    entry_points: Optional[torch.Tensor] = None    # [N, Emax, 2]
    exit_points: Optional[torch.Tensor] = None     # [N, Xmax, 2]
    entry_mask: Optional[torch.Tensor] = None      # [N, Emax]
    exit_mask: Optional[torch.Tensor] = None       # [N, Xmax]
    min_x: Optional[torch.Tensor] = None           # [N]
    max_x: Optional[torch.Tensor] = None           # [N]
    min_y: Optional[torch.Tensor] = None           # [N]
    max_y: Optional[torch.Tensor] = None           # [N]

    def __post_init__(self) -> None:
        if not isinstance(self.centers, torch.Tensor) or not isinstance(self.valid_mask, torch.Tensor):
            raise TypeError("ActionSpace.centers and ActionSpace.valid_mask must be torch.Tensor")
        if self.centers.ndim != 2 or int(self.centers.shape[-1]) != 2:
            raise ValueError(f"ActionSpace.centers must have shape [N,2], got {tuple(self.centers.shape)}")
        if self.valid_mask.ndim != 1 or int(self.valid_mask.shape[0]) != int(self.centers.shape[0]):
            raise ValueError(
                f"ActionSpace.valid_mask must have shape [N], got {tuple(self.valid_mask.shape)} for N={int(self.centers.shape[0])}"
            )
        if self.valid_mask.dtype != torch.bool:
            raise TypeError(f"ActionSpace.valid_mask must be torch.bool, got {self.valid_mask.dtype}")
        if self.variant_indices is not None:
            if not isinstance(self.variant_indices, torch.Tensor):
                raise TypeError("ActionSpace.variant_indices must be torch.Tensor or None")
            if self.variant_indices.ndim != 1 or int(self.variant_indices.shape[0]) != int(self.centers.shape[0]):
                raise ValueError(
                    f"ActionSpace.variant_indices must have shape [N], "
                    f"got {tuple(self.variant_indices.shape)} for N={int(self.centers.shape[0])}"
                )
