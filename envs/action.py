from __future__ import annotations

from dataclasses import dataclass
from typing import Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Normalized placement action payload.

    - gid is required.
    - x/y are bottom-left integer grid coordinates.
    - rot is in degrees (multiples of 90 expected by env).
    """

    gid: GroupId
    x: int
    y: int
    rot: int
