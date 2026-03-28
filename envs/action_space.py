from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .action import GroupId


@dataclass(frozen=True)
class ActionSpace:
    """Factory layout 환경의 표준 action space.

    Core fields (항상 존재):
      - poses: [N, 2] float32 후보 포즈 (x_c, y_c) — center coordinates.
        The engine resolves variant (rotation/mirror/shape) at step time.
      - mask:  [N] bool validity mask (True = valid)

    Geometric features (reward 계산 시 사용, Optional):
      - entries / exits:             [N, Emax/Xmax, 2] IO 포트 절대 좌표
      - entries_mask / exits_mask:   [N, Emax] / [N, Xmax]
      - min_x / max_x / min_y / max_y: [N] 바운딩 박스
    """

    # -- core (필수) --
    poses: torch.Tensor   # [N, 2] float32 (x_c, y_c)
    mask: torch.Tensor    # bool [N]
    gid: Optional[GroupId] = None
    variant_indices: Optional[torch.Tensor] = None  # int64 [N] — per-action variant index

    # -- geometric features (Optional, reward용) --
    entries: Optional[torch.Tensor] = None       # [N, Emax, 2]
    exits: Optional[torch.Tensor] = None         # [N, Xmax, 2]
    entries_mask: Optional[torch.Tensor] = None  # [N, Emax]
    exits_mask: Optional[torch.Tensor] = None    # [N, Xmax]
    min_x: Optional[torch.Tensor] = None         # [N]
    max_x: Optional[torch.Tensor] = None         # [N]
    min_y: Optional[torch.Tensor] = None         # [N]
    max_y: Optional[torch.Tensor] = None         # [N]

    def __post_init__(self) -> None:
        if not isinstance(self.poses, torch.Tensor) or not isinstance(self.mask, torch.Tensor):
            raise TypeError("ActionSpace.poses and ActionSpace.mask must be torch.Tensor")
        if self.poses.ndim != 2 or int(self.poses.shape[-1]) != 2:
            raise ValueError(f"ActionSpace.poses must have shape [N,2], got {tuple(self.poses.shape)}")
        if self.mask.ndim != 1 or int(self.mask.shape[0]) != int(self.poses.shape[0]):
            raise ValueError(
                f"ActionSpace.mask must have shape [N], got {tuple(self.mask.shape)} for N={int(self.poses.shape[0])}"
            )
        if self.mask.dtype != torch.bool:
            raise TypeError(f"ActionSpace.mask must be torch.bool, got {self.mask.dtype}")
        if self.variant_indices is not None:
            if not isinstance(self.variant_indices, torch.Tensor):
                raise TypeError("ActionSpace.variant_indices must be torch.Tensor or None")
            if self.variant_indices.ndim != 1 or int(self.variant_indices.shape[0]) != int(self.poses.shape[0]):
                raise ValueError(
                    f"ActionSpace.variant_indices must have shape [N], "
                    f"got {tuple(self.variant_indices.shape)} for N={int(self.poses.shape[0])}"
                )
