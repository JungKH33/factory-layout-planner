"""Unified port model and runtime policy for lane generation.

This module introduces three small dataclasses and one stateful policy
object that together describe:

* **``PortSpec``** — a single physical port identified by ``port_id`` with an
  owning facility ``gid``, a grid cell ``xy``, and an optional ``max_flows``
  capacity.
* **``PortGroup``** — a named set of ``port_id``s with a selection policy
  (``one_of`` / ``all_of`` / ``up_to_k``).  A single ``port_id`` may appear in
  multiple groups.
* **``PortSelector``** — per-flow-side spec that names the eligible ports
  either directly via ``port_ids`` or via ``port_group_ids``.
* **``PortPolicy``** — runtime object that owns the catalog plus the
  per-port ``port_flow_count`` counter and per-group ``port_group_chosen`` /
  ``port_group_choice_mask`` tensors.  It resolves a flow's candidate port
  ``xy`` set under the current runtime constraints (capacity + group locks)
  and commits the final selection once a route is applied.

``LaneState`` owns a single ``PortPolicy`` instance.  The static catalog is
shared by reference on ``LaneState.copy``; runtime counters are cloned so
MCTS snapshots stay cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


# ----------------------------------------------------------------------
# Static data model
# ----------------------------------------------------------------------


_ALLOWED_KINDS = ("entry", "exit", "bidir")
_ALLOWED_SELECTIONS = ("one_of", "all_of", "up_to_k")


@dataclass(frozen=True)
class PortSpec:
    """A single physical port.

    ``max_flows == 0`` means unlimited.  ``max_flows >= 1`` caps the number
    of flows that may simultaneously use the port.
    """

    port_id: str
    gid: str
    xy: Tuple[int, int]
    kind: str = "bidir"
    max_flows: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"PortSpec.kind must be one of {_ALLOWED_KINDS}, got {self.kind!r}"
            )
        if int(self.max_flows) < 0:
            raise ValueError(f"PortSpec.max_flows must be >= 0, got {self.max_flows!r}")


@dataclass(frozen=True)
class PortGroup:
    """A named set of ports with a selection policy.

    ``selection``:

    * ``one_of`` — at most one member may be picked across the whole episode;
      once a flow uses member ``p``, subsequent flows that reference this
      group are restricted to ``p``.
    * ``all_of`` — every member must be used (across flows) before the group
      is considered satisfied; individual flows may pick any subset.
    * ``up_to_k`` — at most ``k`` distinct members may be used across the
      episode.
    """

    group_id: str
    port_ids: Tuple[str, ...]
    selection: str = "one_of"
    k: int = 1

    def __post_init__(self) -> None:
        if self.selection not in _ALLOWED_SELECTIONS:
            raise ValueError(
                f"PortGroup.selection must be one of {_ALLOWED_SELECTIONS}, got {self.selection!r}"
            )
        if int(self.k) < 1:
            raise ValueError(f"PortGroup.k must be >= 1, got {self.k!r}")
        if len(self.port_ids) == 0:
            raise ValueError(f"PortGroup.port_ids must not be empty (group_id={self.group_id!r})")


@dataclass(frozen=True)
class PortSelector:
    """Per-flow-side port eligibility spec.

    Either ``port_ids`` or ``port_group_ids`` (or both) must be non-empty —
    the resolved candidate set is the union, after group-selection and
    capacity filtering.
    """

    gid: str
    port_ids: Tuple[str, ...] = ()
    port_group_ids: Tuple[str, ...] = ()
    # TODO(span): add ``span: int = 1`` once the adapter supports selecting
    # multiple ports per endpoint for a single flow (mirrors the
    # ``c_k`` / ``t_k`` knobs in group_placement.FlowReward).

    def __post_init__(self) -> None:
        if not self.port_ids and not self.port_group_ids:
            raise ValueError(
                f"PortSelector(gid={self.gid!r}) must specify at least one of "
                "port_ids / port_group_ids"
            )


# ----------------------------------------------------------------------
# Runtime policy
# ----------------------------------------------------------------------


@dataclass
class PortPolicy:
    """Runtime port policy owned by :class:`LaneState`.

    Field groups:

    * Static catalog (shared by reference across :meth:`LaneState.copy`):
      ``port_specs``, ``port_groups``, ``_port_to_groups``, ``_port_index``,
      ``_group_index``, ``_group_member_index``.
    * Runtime counters (cloned across :meth:`LaneState.copy`):
      ``port_flow_count``, ``port_group_chosen``, ``port_group_choice_mask``.
    """

    device: torch.device

    # Static catalog
    port_specs: Dict[str, PortSpec]
    port_groups: Dict[str, PortGroup]
    _port_to_groups: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    _port_index: Dict[str, int] = field(default_factory=dict)
    _group_index: Dict[str, int] = field(default_factory=dict)
    _group_member_index: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Runtime counters
    port_flow_count: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    port_group_chosen: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    port_group_choice_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def build(
        cls,
        *,
        port_specs: Mapping[str, PortSpec],
        port_groups: Mapping[str, PortGroup],
        device: torch.device,
    ) -> "PortPolicy":
        dev = torch.device(device)
        specs = dict(port_specs)
        groups = dict(port_groups)

        # Validate group membership references only known ports.
        for gid, grp in groups.items():
            for pid in grp.port_ids:
                if pid not in specs:
                    raise ValueError(
                        f"PortGroup {gid!r} references unknown port_id {pid!r}"
                    )

        port_index = {pid: i for i, pid in enumerate(specs.keys())}
        group_index = {gid: i for i, gid in enumerate(groups.keys())}
        group_member_index: Dict[str, Dict[str, int]] = {}
        for gid, grp in groups.items():
            group_member_index[gid] = {pid: j for j, pid in enumerate(grp.port_ids)}

        port_to_groups: Dict[str, List[str]] = {pid: [] for pid in specs.keys()}
        for gid, grp in groups.items():
            for pid in grp.port_ids:
                port_to_groups[pid].append(gid)

        P = len(specs)
        G = len(groups)
        P_max = max((len(g.port_ids) for g in groups.values()), default=0)

        policy = cls(
            device=dev,
            port_specs=specs,
            port_groups=groups,
            _port_to_groups={pid: tuple(gs) for pid, gs in port_to_groups.items()},
            _port_index=port_index,
            _group_index=group_index,
            _group_member_index=group_member_index,
            port_flow_count=torch.zeros((P,), dtype=torch.int32, device=dev),
            port_group_chosen=torch.full((G,), -1, dtype=torch.int16, device=dev),
            port_group_choice_mask=torch.zeros((G, max(P_max, 1)), dtype=torch.bool, device=dev),
        )
        return policy

    # ------------------------------------------------------------------
    # Copy / restore / reset
    # ------------------------------------------------------------------

    def copy(self) -> "PortPolicy":
        return PortPolicy(
            device=self.device,
            port_specs=self.port_specs,
            port_groups=self.port_groups,
            _port_to_groups=self._port_to_groups,
            _port_index=self._port_index,
            _group_index=self._group_index,
            _group_member_index=self._group_member_index,
            port_flow_count=self.port_flow_count.clone(),
            port_group_chosen=self.port_group_chosen.clone(),
            port_group_choice_mask=self.port_group_choice_mask.clone(),
        )

    def restore(self, src: "PortPolicy") -> None:
        if not isinstance(src, PortPolicy):
            raise TypeError(f"src must be PortPolicy, got {type(src).__name__}")
        self.port_flow_count.resize_(src.port_flow_count.shape)
        self.port_flow_count.copy_(src.port_flow_count.to(device=self.device, dtype=torch.int32))
        self.port_group_chosen.resize_(src.port_group_chosen.shape)
        self.port_group_chosen.copy_(src.port_group_chosen.to(device=self.device, dtype=torch.int16))
        self.port_group_choice_mask.resize_(src.port_group_choice_mask.shape)
        self.port_group_choice_mask.copy_(src.port_group_choice_mask.to(device=self.device, dtype=torch.bool))

    def reset_runtime(self) -> None:
        self.port_flow_count.zero_()
        self.port_group_chosen.fill_(-1)
        self.port_group_choice_mask.zero_()

    # ------------------------------------------------------------------
    # Catalog lookups
    # ------------------------------------------------------------------

    @property
    def port_count(self) -> int:
        return len(self.port_specs)

    @property
    def group_count(self) -> int:
        return len(self.port_groups)

    def port_ids_at(self, xy: Tuple[int, int]) -> Tuple[str, ...]:
        """Return every ``port_id`` whose ``xy`` equals the given cell.

        One physical cell may host multiple ``PortSpec`` entries when the
        same coordinate participates in several groups.
        """
        x = int(xy[0])
        y = int(xy[1])
        out: List[str] = []
        for pid, spec in self.port_specs.items():
            if int(spec.xy[0]) == x and int(spec.xy[1]) == y:
                out.append(pid)
        return tuple(out)

    # ------------------------------------------------------------------
    # Candidate resolution
    # ------------------------------------------------------------------

    def _selector_port_ids(self, selector: PortSelector) -> List[str]:
        """Expand a selector's direct + group references into a port_id list.

        Duplicates are removed while preserving first-seen order.
        """
        seen: Dict[str, None] = {}
        for pid in selector.port_ids:
            if pid not in self.port_specs:
                raise KeyError(f"PortSelector references unknown port_id {pid!r}")
            seen.setdefault(pid, None)
        for gid in selector.port_group_ids:
            grp = self.port_groups.get(gid)
            if grp is None:
                raise KeyError(f"PortSelector references unknown port_group_id {gid!r}")
            for pid in grp.port_ids:
                seen.setdefault(pid, None)
        return list(seen.keys())

    def _allowed_by_groups(self, port_id: str) -> bool:
        """Check ``port_id`` against every group it belongs to.

        - ``one_of``: if the group has a locked choice, only that port is allowed.
        - ``up_to_k``: if the group has already reached ``k`` distinct chosen
          members and ``port_id`` is not among them, the port is disallowed.
        - ``all_of``: no restriction on eligibility (all members must
          eventually be used; individual picks are free).
        """
        for gid in self._port_to_groups.get(port_id, ()):
            grp = self.port_groups[gid]
            gi = self._group_index[gid]
            if grp.selection == "one_of":
                chosen = int(self.port_group_chosen[gi].item())
                if chosen < 0:
                    continue
                locked_pid = grp.port_ids[chosen]
                if locked_pid != port_id:
                    return False
            elif grp.selection == "up_to_k":
                mask_row = self.port_group_choice_mask[gi, : len(grp.port_ids)]
                already = int(mask_row.sum().item())
                if already >= int(grp.k):
                    member_idx = self._group_member_index[gid].get(port_id, -1)
                    if member_idx < 0 or not bool(mask_row[member_idx].item()):
                        return False
            # all_of: no restriction
        return True

    def resolve_port_ids(self, selector: PortSelector) -> List[str]:
        """Return the runtime-eligible ``port_id`` list for *selector*.

        Filters out ports whose ``max_flows`` cap is already saturated and
        ports blocked by a group-selection lock.
        """
        eligible: List[str] = []
        for pid in self._selector_port_ids(selector):
            spec = self.port_specs[pid]
            if int(spec.max_flows) > 0:
                idx = self._port_index[pid]
                if int(self.port_flow_count[idx].item()) >= int(spec.max_flows):
                    continue
            if not self._allowed_by_groups(pid):
                continue
            eligible.append(pid)
        return eligible

    def resolve_xy(self, selector: PortSelector) -> torch.Tensor:
        """Materialize eligible ports as an ``[N, 2]`` long tensor of cells."""
        pids = self.resolve_port_ids(selector)
        if not pids:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        xs: List[List[int]] = []
        for pid in pids:
            spec = self.port_specs[pid]
            xs.append([int(spec.xy[0]), int(spec.xy[1])])
        return torch.tensor(xs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Selection commit
    # ------------------------------------------------------------------

    def commit_xy(
        self,
        selector: PortSelector,
        xy: Tuple[int, int],
    ) -> Tuple[str, ...]:
        """Record that the flow used ``xy`` on this selector's side.

        The runtime counters are updated for **every** ``port_id`` owned by
        this selector whose ``xy`` matches (since one physical cell may be
        referenced by several port_ids via different groups).  Returns the
        touched ``port_id``s.
        """
        owned = set(self._selector_port_ids(selector))
        if not owned:
            return ()
        x = int(xy[0])
        y = int(xy[1])
        touched: List[str] = []
        for pid in owned:
            spec = self.port_specs[pid]
            if int(spec.xy[0]) == x and int(spec.xy[1]) == y:
                touched.append(pid)
        if not touched:
            return ()
        for pid in touched:
            idx = self._port_index[pid]
            self.port_flow_count[idx] = self.port_flow_count[idx] + 1
            for gid in self._port_to_groups.get(pid, ()):
                grp = self.port_groups[gid]
                gi = self._group_index[gid]
                mem_idx = self._group_member_index[gid].get(pid, -1)
                if mem_idx < 0:
                    continue
                if grp.selection == "one_of":
                    cur = int(self.port_group_chosen[gi].item())
                    if cur < 0:
                        self.port_group_chosen[gi] = int(mem_idx)
                # For all_of / up_to_k we mark the member as used.
                self.port_group_choice_mask[gi, mem_idx] = True
        return tuple(touched)
