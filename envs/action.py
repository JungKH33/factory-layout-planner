from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Placement action: center coordinates.

    - gid is required.
    - x_c/y_c are center coordinates (float).
    - variant_index: if None the engine tries all variants and
      picks the cheapest placeable one via GroupSpec.resolve().
      If set, the engine uses that specific variant directly.
    - source_index: if set, only variants from this source shape are
      considered during resolve(). Ignored when variant_index is set.
    """

    gid: GroupId
    x_c: float
    y_c: float
    variant_index: Optional[int] = None
    source_index: Optional[int] = None
