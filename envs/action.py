from __future__ import annotations

from dataclasses import dataclass
from typing import Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Placement action: center coordinates (orient-free).

    - gid is required.
    - x_c/y_c are center coordinates (float, orient-independent).
    - The engine resolves to concrete rotation/mirror via
      StaticSpec.resolve().
    """

    gid: GroupId
    x_c: float
    y_c: float
