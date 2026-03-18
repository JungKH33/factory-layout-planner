from __future__ import annotations

from dataclasses import dataclass
from typing import Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Coarse placement action (intent).

    - gid is required.
    - x/y are bottom-left integer grid coordinates.
    - orient is a coarse orientation class: 0 = {R0, R180} family,
      1 = {R90, R270} family.  The engine resolves this to a concrete
      rot/mirror via StaticSpec.resolve().
    """

    gid: GroupId
    x: int
    y: int
    orient: int
