from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Placement action: center coordinates + optional variant pin.

    - group_id is required.
    - x_center/y_center are float center coordinates (adapter/agent coordinate space).
    - variant_index: if None the engine tries all variants and
      picks the cheapest placeable one.
      If set, the engine uses that specific variant directly.
    - source_index: if set, only variants from this source shape are
      considered during resolve(). Ignored when variant_index is set.
    """

    group_id: GroupId
    x_center: float
    y_center: float
    variant_index: Optional[int] = None
    source_index: Optional[int] = None
