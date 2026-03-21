from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Placement action: center coordinates (orient-free).

    - gid is required.
    - x_c/y_c are center coordinates (float, orient-independent).
    - orientation_index: if None the engine tries all orientations and
      picks the cheapest placeable one via GroupSpec.resolve().
      If set, the engine uses that specific orientation directly.
    """

    gid: GroupId
    x_c: float
    y_c: float
    orientation_index: Optional[int] = None
