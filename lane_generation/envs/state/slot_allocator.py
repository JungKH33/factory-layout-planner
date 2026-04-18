"""Lane slot allocation and single-grid capacity policy.

``LaneSlotAllocator`` is a stateless helper that owns slot-selection
algorithms and physical-width capacity checks.  All mutable tensor state
(``edge_lane_mask``, ``edge_width_accum``) lives on ``LaneState``; the
allocator only reads and previews against that state.

Physical model
--------------
One directed edge corresponds to a crossing through a corridor that is at
most one grid-cell wide.  ``GRID_CAPACITY = 1.0`` is the total available
physical width in grid-cell units.

* ``lane_width <= 1.0``  — supported.  A new slot is granted only when
  ``edge_width_accum[edge] + lane_width <= GRID_CAPACITY``.
* ``lane_width > 1.0``   — unsupported in this revision (TODO: multi-cell
  lane support, which requires the path itself to span multiple grid rows).

Slot reuse (merge_allow / reverse_allow) does **not** consume additional
physical width because the lane already exists in the corridor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .state import LaneState


class LaneSlotAllocator:
    """Stateless slot-selection + single-grid capacity policy."""

    GRID_CAPACITY: float = 1.0

    def __init__(
        self,
        *,
        capacity_epsilon: float = 1e-9,
        max_slots: int = 63,
    ) -> None:
        self._epsilon = float(capacity_epsilon)
        self._max_slots = int(max_slots)

    # ------------------------------------------------------------------
    # Slot-primitive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_set_slot(mask: int) -> Optional[int]:
        if int(mask) == 0:
            return None
        lsb = int(mask) & -int(mask)
        return int(lsb.bit_length() - 1)

    def _first_free_slot(self, forbidden_mask: int) -> Optional[int]:
        m = int(forbidden_mask)
        for s in range(self._max_slots + 1):
            if ((m >> s) & 1) == 0:
                return s
        return None

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def choose_slot(
        self,
        *,
        dir_mask: int,
        rev_mask: int,
        width_accum: float,
        lane_width: float,
        merge_allow: bool = True,
        reverse_allow: bool = True,
    ) -> Tuple[Optional[int], bool]:
        """Select a lane slot for a flow of the given ``lane_width``.

        Returns ``(slot_id, is_new)``:

        * ``is_new=False``  — slot is reused from an existing lane; no
          physical width is consumed.
        * ``is_new=True``   — a brand-new slot is allocated; the caller must
          increment ``edge_width_accum`` by ``lane_width``.
        * ``(None, False)`` — edge is infeasible for this flow (capacity
          exceeded or no free slot within ``max_slots``).

        lane_width > GRID_CAPACITY is always infeasible (TODO: multi-cell).
        """
        lw = float(lane_width)

        if lw > self.GRID_CAPACITY + self._epsilon:
            return (None, False)

        if merge_allow:
            reuse_dir = self._first_set_slot(int(dir_mask))
            if reuse_dir is not None:
                return (int(reuse_dir), False)
            if reverse_allow:
                reuse_rev = self._first_set_slot(int(rev_mask))
                if reuse_rev is not None:
                    return (int(reuse_rev), False)
            if float(width_accum) + lw > self.GRID_CAPACITY + self._epsilon:
                return (None, False)
            forbidden = int(rev_mask) if not reverse_allow else 0
            slot = self._first_free_slot(forbidden)
            return (slot, slot is not None)

        if reverse_allow:
            shared = int(rev_mask) & ~int(dir_mask)
            reuse_shared = self._first_set_slot(shared)
            if reuse_shared is not None:
                return (int(reuse_shared), False)
            if float(width_accum) + lw > self.GRID_CAPACITY + self._epsilon:
                return (None, False)
            slot = self._first_free_slot(int(dir_mask))
            return (slot, slot is not None)

        # Fully disjoint: always a new slot
        if float(width_accum) + lw > self.GRID_CAPACITY + self._epsilon:
            return (None, False)
        slot = self._first_free_slot(int(dir_mask) | int(rev_mask))
        return (slot, slot is not None)

    # ------------------------------------------------------------------
    # Batch preview (no state mutation)
    # ------------------------------------------------------------------

    def preview_batch(
        self,
        state: "LaneState",
        *,
        candidate_edge_idx: torch.Tensor,
        candidate_edge_mask: torch.Tensor,
        lane_width: float,
        merge_allow: bool = True,
        reverse_allow: bool = True,
    ) -> torch.Tensor:
        """Plan slots per candidate path without mutating state.

        Returns ``int16 [K, L]``.  ``-1`` means infeasible or padding.
        Active entries that receive ``-1`` mark the whole candidate as
        infeasible; the upstream adapter checks this via
        ``((planned >= 0) | ~edge_mask).all(dim=1)``.

        Two per-candidate scratchpads accumulate hypothetical state so that
        sequential edges within the same candidate see each other:

        * ``mask_ov`` — hypothetical ``edge_lane_mask`` overrides per edge id.
        * ``accum_ov`` — hypothetical ``edge_width_accum`` overrides per
          undirected leader (``leader = min(edge_id, reverse_edge_id)``).
        """
        idx = candidate_edge_idx.to(device=state.device, dtype=torch.long)
        edge_m = candidate_edge_mask.to(device=state.device, dtype=torch.bool)
        if idx.shape != edge_m.shape:
            raise ValueError(
                f"shape mismatch: candidate_edge_idx {tuple(idx.shape)} vs "
                f"candidate_edge_mask {tuple(edge_m.shape)}"
            )

        k = int(idx.shape[0])
        l = int(idx.shape[1]) if idx.dim() == 2 else 0
        out = torch.full((k, l), -1, dtype=torch.int16, device=state.device)
        if k == 0 or l == 0:
            return out

        mask_flat = state.edge_lane_mask.view(-1)
        rev_flat = state.reverse_edge_lut.view(-1)
        valid_flat = state.edge_valid_flat.view(-1)
        accum_flat = state.edge_width_accum.view(-1)
        n_edges = int(mask_flat.shape[0])
        lw = float(lane_width)

        def _get_mask(eid: int, overrides: Dict[int, int]) -> int:
            ov = overrides.get(eid)
            return int(ov) if ov is not None else int(mask_flat[eid].item())

        def _get_accum(leader: int, overrides: Dict[int, float]) -> float:
            ov = overrides.get(leader)
            return float(ov) if ov is not None else float(accum_flat[leader].item())

        for ci in range(k):
            mask_ov: Dict[int, int] = {}
            accum_ov: Dict[int, float] = {}

            for li in range(l):
                if not bool(edge_m[ci, li].item()):
                    continue
                e = int(idx[ci, li].item())
                if e < 0 or e >= n_edges or not bool(valid_flat[e].item()):
                    continue

                r = int(rev_flat[e].item())
                dir_m = _get_mask(e, mask_ov)
                rev_m = _get_mask(r, mask_ov) if 0 <= r < n_edges else 0
                leader = min(e, r) if 0 <= r < n_edges else e
                wa = _get_accum(leader, accum_ov)

                slot, is_new = self.choose_slot(
                    dir_mask=dir_m, rev_mask=rev_m,
                    width_accum=wa, lane_width=lw,
                    merge_allow=merge_allow, reverse_allow=reverse_allow,
                )
                if slot is None:
                    # Candidate is infeasible; remaining entries stay -1
                    break

                out[ci, li] = int(slot)
                mask_ov[e] = int(dir_m | (1 << slot))
                if is_new:
                    accum_ov[leader] = wa + lw

        return out

    # ------------------------------------------------------------------
    # Pathfinding feasibility mask
    # ------------------------------------------------------------------

    def capacity_mask(
        self,
        state: "LaneState",
        lane_width: float,
        *,
        merge_allow: bool = True,
        reverse_allow: bool = True,
    ) -> torch.Tensor:
        """Return ``bool [H, W, 4]`` — True if the edge can accept a lane of ``lane_width``.

        An edge is feasible when at least one holds:

        * Can reuse an existing same-direction slot (merge_allow and dir_mask != 0).
        * Can reuse a reverse slot (depending on merge_allow / reverse_allow
          combination — mirrors ``choose_slot`` reuse branches).
        * Has remaining physical capacity (accum + lane_width <= GRID_CAPACITY).

        lane_width > GRID_CAPACITY returns all-False (multi-cell — TODO).
        """
        lw = float(lane_width)
        h, w = int(state.grid_height), int(state.grid_width)
        dev = state.device

        if lw > self.GRID_CAPACITY + self._epsilon:
            return torch.zeros((h, w, 4), dtype=torch.bool, device=dev)

        mask_flat = state.edge_lane_mask.view(-1)          # [E] int64
        rev_flat = state.reverse_edge_lut.view(-1)          # [E] long
        valid_flat = state.edge_valid_flat.view(-1)          # [E] bool
        accum_flat = state.edge_width_accum.view(-1)         # [E] float32

        rev_mask_flat = mask_flat[rev_flat]                  # [E] int64

        dir_nonzero = mask_flat != 0                         # [E] bool
        rev_nonzero = rev_mask_flat != 0                     # [E] bool
        rev_excl_dir = (rev_mask_flat & ~mask_flat) != 0     # [E] bool

        can_reuse = torch.zeros(h * w * 4, dtype=torch.bool, device=dev)
        if merge_allow:
            can_reuse |= dir_nonzero
            if reverse_allow:
                can_reuse |= rev_nonzero
        elif reverse_allow:
            can_reuse |= rev_excl_dir

        has_cap = (accum_flat + lw) <= (self.GRID_CAPACITY + self._epsilon)

        return (valid_flat & (can_reuse | has_cap)).view(h, w, 4)


__all__ = ["LaneSlotAllocator"]
