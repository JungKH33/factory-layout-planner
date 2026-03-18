from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .action import GroupId


@dataclass(frozen=True)
class ActionSpace:
    """Factory layout 환경의 표준 action space.

    Core fields (항상 존재):
      - poses: [N, 3] 후보 포즈 (x_bl, y_bl, orient)
        orient is a coarse orientation class (0 or 1), not concrete rot.
      - mask:  [N] bool validity mask (True = valid)

    Geometric features (reward 계산 시 사용, Optional):
      - entries / exits:             [N, Emax/Xmax, 2] IO 포트 절대 좌표
      - entries_mask / exits_mask:   [N, Emax] / [N, Xmax]
      - min_x / max_x / min_y / max_y: [N] 바운딩 박스

    Adapter/policy 전용:
      - meta: Dict for additional adapter-specific data (action_delta 등)
    """

    # -- core (필수) --
    poses: torch.Tensor   # [N, 3]
    mask: torch.Tensor    # bool [N]
    gid: Optional[GroupId] = None

    # -- geometric features (Optional, reward용) --
    entries: Optional[torch.Tensor] = None       # [N, Emax, 2]
    exits: Optional[torch.Tensor] = None         # [N, Xmax, 2]
    entries_mask: Optional[torch.Tensor] = None  # [N, Emax]
    exits_mask: Optional[torch.Tensor] = None    # [N, Xmax]
    min_x: Optional[torch.Tensor] = None         # [N]
    max_x: Optional[torch.Tensor] = None         # [N]
    min_y: Optional[torch.Tensor] = None         # [N]
    max_y: Optional[torch.Tensor] = None         # [N]

    # -- adapter/policy 전용 --
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.poses, torch.Tensor) or not isinstance(self.mask, torch.Tensor):
            raise TypeError("ActionSpace.poses and ActionSpace.mask must be torch.Tensor")
        if self.poses.ndim != 2 or int(self.poses.shape[-1]) != 3:
            raise ValueError(f"ActionSpace.poses must have shape [N,3], got {tuple(self.poses.shape)}")
        if self.mask.ndim != 1 or int(self.mask.shape[0]) != int(self.poses.shape[0]):
            raise ValueError(
                f"ActionSpace.mask must have shape [N], got {tuple(self.mask.shape)} for N={int(self.poses.shape[0])}"
            )
        if self.mask.dtype != torch.bool:
            raise TypeError(f"ActionSpace.mask must be torch.bool, got {self.mask.dtype}")
