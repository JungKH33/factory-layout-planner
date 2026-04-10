from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .base import (
    PORT_SPAN_ALL,
    GroupSpec,
    GroupPlacement,
    GroupVariant,
    normalize_port_span,
)

if TYPE_CHECKING:
    from ..state.base import EnvState
    from ..reward.core import RewardComposer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticVariant(GroupVariant):
    """Static-spec variant: extends GroupVariant with clearance and cache keys."""

    cL: int = 0
    cR: int = 0
    cB: int = 0
    cT: int = 0
    clearance_origin: Tuple[int, int] = (0, 0)
    is_rectangular: bool = True
    shape_key: tuple = ()
    cost_key: tuple = ()


# Backwards-compat alias (will be removed)


# ---------------------------------------------------------------------------
# Placement result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StaticRectPlacement(GroupPlacement):
    """Resolved absolute placement for a rectangular static group."""

    x_bl: int
    y_bl: int
    rotation: int
    w: float
    h: float
    clearance_left: int
    clearance_right: int
    clearance_bottom: int
    clearance_top: int
    mirror: bool = False
    variant_index: int = 0


@dataclass
class StaticIrregularPlacement(GroupPlacement):
    """Resolved absolute placement for an irregular static group."""

    x_bl: int
    y_bl: int
    rotation: int
    w: float
    h: float
    mirror: bool = False
    variant_index: int = 0
    body_polygon_abs: Optional[List[Tuple[float, float]]] = None
    clearance_polygon_abs: Optional[List[Tuple[float, float]]] = None


# ---------------------------------------------------------------------------
# StaticSpec — shared base for StaticRectSpec / StaticIrregularSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticSpec(GroupSpec):
    """Common spec for static facilities (rect or irregular).

    Provides:
    - Variant storage and lookup
    - Rotation / mirror / clearance helpers
    - Port coordinate transforms (single + batch)
    - placeable_map / placeable_batch / score_batch
    - resolve() — tries every variant and picks the best
    - shape_tensors() — returns (body_mask, clearance_mask, origin, is_rect) by key

    Subclasses must implement ``__post_init__`` which is responsible for
    calling ``_store_variants(variants, shape_tensors_by_key)`` to populate
    the internal caches.
    """

    device: torch.device
    id: object
    width: int
    height: int
    entries_rel: List[Tuple[float, float]]
    exits_rel: List[Tuple[float, float]]
    rotatable: bool = True
    mirrorable: bool = False
    zone_values: Dict[str, Any] = field(default_factory=dict)
    _entry_port_span: int = 1
    _exit_port_span: int = 1
    clearance_lrtb_rel: Optional[Tuple[int, int, int, int]] = None
    _variant_defs: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    # ----- variant storage (populated by _store_variants) -----

    def _store_variants(
        self,
        variants: List[StaticVariant],
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]],
    ) -> None:
        self._variants: List[StaticVariant] = variants
        self._variants_by_shape: Dict[tuple, List[StaticVariant]] = {}
        for vi in variants:
            self._variants_by_shape.setdefault(vi.shape_key, []).append(vi)
        self._shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = shape_tensors_by_key
        self._variant_entry_off_t: List[torch.Tensor] = []
        self._variant_exit_off_t: List[torch.Tensor] = []
        for vi in variants:
            if vi.entry_offsets:
                et = torch.tensor(list(vi.entry_offsets), dtype=torch.float32, device=self.device)
            else:
                et = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            if vi.exit_offsets:
                xt = torch.tensor(list(vi.exit_offsets), dtype=torch.float32, device=self.device)
            else:
                xt = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self._variant_entry_off_t.append(et)
            self._variant_exit_off_t.append(xt)

        V = len(variants)
        P_ent = max((int(t.shape[0]) for t in self._variant_entry_off_t), default=0)
        P_ext = max((int(t.shape[0]) for t in self._variant_exit_off_t), default=0)
        self._all_entry_offsets = torch.zeros((V, P_ent, 2), dtype=torch.float32, device=self.device)
        self._all_entry_mask = torch.zeros((V, P_ent), dtype=torch.bool, device=self.device)
        self._all_exit_offsets = torch.zeros((V, P_ext, 2), dtype=torch.float32, device=self.device)
        self._all_exit_mask = torch.zeros((V, P_ext), dtype=torch.bool, device=self.device)
        for i, (et, xt) in enumerate(zip(self._variant_entry_off_t, self._variant_exit_off_t)):
            p_e = int(et.shape[0])
            if p_e > 0:
                self._all_entry_offsets[i, :p_e] = et
                self._all_entry_mask[i, :p_e] = True
            p_x = int(xt.shape[0])
            if p_x > 0:
                self._all_exit_offsets[i, :p_x] = xt
                self._all_exit_mask[i, :p_x] = True
        self._all_body_width = torch.tensor(
            [float(vi.body_width) for vi in variants],
            dtype=torch.float32, device=self.device,
        )
        self._all_body_height = torch.tensor(
            [float(vi.body_height) for vi in variants],
            dtype=torch.float32, device=self.device,
        )
        # source index mapping: flat variant idx → source_index
        self._variant_to_source: List[int] = [vi.source_index for vi in variants]
        # source_index → (start, end) range in _variants
        src_ranges: Dict[int, List[int]] = {}
        for idx, vi in enumerate(variants):
            src_ranges.setdefault(vi.source_index, []).append(idx)
        n_sources = max(src_ranges.keys()) + 1 if src_ranges else 0
        self._source_ranges: List[Tuple[int, int]] = []
        for s in range(n_sources):
            idxs = src_ranges.get(s, [])
            if idxs:
                self._source_ranges.append((min(idxs), max(idxs) + 1))
            else:
                self._source_ranges.append((0, 0))
        self._num_sources: int = n_sources

        # shape_key → variant flat indices (tensor, for vectorised broadcast)
        self._shape_variant_indices: Dict[tuple, torch.Tensor] = {}
        for sk in self._variants_by_shape:
            idxs = [i for i, v in enumerate(self._variants) if v.shape_key == sk]
            self._shape_variant_indices[sk] = torch.tensor(idxs, dtype=torch.long, device=self.device)

    def _resolve_port_span_all(self) -> None:
        """Replace ``PORT_SPAN_ALL`` sentinel with actual max port count across variants.

        Must be called after ``_store_variants``. After this, ``_entry_port_span``
        and ``_exit_port_span`` are always concrete integers ``>= 1``, so no
        downstream code needs to know about the sentinel.
        """
        max_entries = max((len(vi.entry_offsets) for vi in self._variants), default=0)
        max_exits = max((len(vi.exit_offsets) for vi in self._variants), default=0)
        if int(self._entry_port_span) == int(PORT_SPAN_ALL):
            self._entry_port_span = max(1, int(max_entries))
        if int(self._exit_port_span) == int(PORT_SPAN_ALL):
            self._exit_port_span = max(1, int(max_exits))

    @property
    def num_sources(self) -> int:
        """Number of distinct source shape definitions."""
        return getattr(self, "_num_sources", 1)

    @property
    def variants(self) -> List[GroupVariant]:
        """All unique placement variants for this spec."""
        return list(self._variants)

    @property
    def body_widths(self) -> torch.Tensor:
        """[V] float32 body width per variant."""
        return self._all_body_width

    @property
    def body_heights(self) -> torch.Tensor:
        """[V] float32 body height per variant."""
        return self._all_body_height

    # ----- rotation / clearance helpers -----

    @staticmethod
    def _norm_rotation(rotation: int) -> int:
        r = int(rotation) % 360
        if r % 90 != 0:
            raise ValueError(f"rotation must be a multiple of 90 degrees, got {rotation!r}")
        return r

    def _resolve_rotation(self, rotation: int) -> int:
        r = self._norm_rotation(rotation)
        if not bool(self.rotatable):
            return 0
        return r

    def _rotated_size(self, rotation: int) -> Tuple[float, float]:
        r = self._resolve_rotation(rotation)
        if r in (90, 270):
            return (float(self.height), float(self.width))
        return (float(self.width), float(self.height))

    @staticmethod
    def _rotate_point(dx: float, dy: float, rotation: int) -> Tuple[float, float]:
        r = StaticSpec._norm_rotation(rotation)
        if r == 0:
            return (float(dx), float(dy))
        if r == 90:
            return (float(dy), -float(dx))
        if r == 180:
            return (-float(dx), -float(dy))
        if r == 270:
            return (-float(dy), float(dx))
        return (float(dx), float(dy))

    def rotated_size(self, rotation: int) -> Tuple[int, int]:
        r = self._resolve_rotation(rotation)
        w, h = self._rotated_size(r)
        return int(w), int(h)

    # ----- clearance helpers -----

    @staticmethod
    def _clearance_lrtb_for_rotation_raw(
        rotation: int,
        *,
        mirror: bool = False,
        clearance_lrtb_rel: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int, int, int]:
        """Return (L, R, B, T) clearance for a given rotation/mirror of a LRTB tuple."""
        r = StaticSpec._norm_rotation(rotation)
        if clearance_lrtb_rel is not None:
            cL, cR, cB, cT = (int(clearance_lrtb_rel[0]), int(clearance_lrtb_rel[1]),
                               int(clearance_lrtb_rel[2]), int(clearance_lrtb_rel[3]))
        else:
            cL = cR = cB = cT = 0
        if bool(mirror):
            cL, cR = cR, cL
        if r == 90:
            return (cB, cT, cR, cL)
        if r == 180:
            return (cR, cL, cT, cB)
        if r == 270:
            return (cT, cB, cL, cR)
        return (cL, cR, cB, cT)

    def _clearance_lrtb_for_rotation(self, rotation: int, *, mirror: bool = False) -> Tuple[int, int, int, int]:
        """Return (L, R, B, T) clearance for a given rotation/mirror of the canonical LRTB."""
        return self._clearance_lrtb_for_rotation_raw(
            rotation, mirror=mirror, clearance_lrtb_rel=self.clearance_lrtb_rel,
        )

    @staticmethod
    def _dilate_body_map(
        body_mask: torch.Tensor,
        L: int, R: int, B: int, T: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Dilate body_mask by (L, R, B, T) cells.

        Returns ``(clearance_mask, clearance_origin)`` where
        *clearance_origin* = ``(L, B)`` is the offset of the body BL
        inside the clearance_mask.
        """
        if L == 0 and R == 0 and B == 0 and T == 0:
            return body_mask.clone(), (0, 0)
        bH, bW = int(body_mask.shape[0]), int(body_mask.shape[1])
        cH = bH + B + T
        cW = bW + L + R
        clearance = torch.zeros((cH, cW), dtype=torch.bool, device=body_mask.device)
        ys, xs = torch.nonzero(body_mask, as_tuple=True)
        for yv, xv in zip(ys.tolist(), xs.tolist()):
            r0 = int(yv)
            r1 = int(yv) + B + T + 1
            c0 = int(xv)
            c1 = int(xv) + L + R + 1
            clearance[r0:r1, c0:c1] = True
        return clearance, (L, B)

    def _resolve_clearance(self, body_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Compute ``(clearance_mask, clearance_origin)`` from clearance fields."""
        if self.clearance_lrtb_rel is not None:
            L, R, B, T = (int(self.clearance_lrtb_rel[0]), int(self.clearance_lrtb_rel[1]),
                          int(self.clearance_lrtb_rel[2]), int(self.clearance_lrtb_rel[3]))
            return self._dilate_body_map(body_mask, L, R, B, T)
        return body_mask.clone(), (0, 0)

    # ----- port coordinate helpers -----

    def _ports_from_center(
        self,
        x_center: float,
        y_center: float,
        rotation: int,
        ports_rel: List[Tuple[float, float]],
        *,
        mirror: bool = False,
    ) -> List[Tuple[float, float]]:
        r = self._resolve_rotation(rotation)
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        out = []
        for dx_bl, dy_bl in ports_rel:
            dx = dx_bl - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = self._rotate_point(dx, dy_bl - half_h, r)
            out.append((x_center + rdx, y_center + rdy))
        return out

    def _entries_from_center(self, x_center: float, y_center: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_center, y_center, rotation, self.entries_rel, mirror=mirror)

    def _exits_from_center(self, x_center: float, y_center: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_center, y_center, rotation, self.exits_rel, mirror=mirror)

    # ----- shape key / tensors -----

    @staticmethod
    def _make_shape_key(
        *,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> tuple:
        if bool(is_rectangular):
            return (
                "rect",
                int(body_mask.shape[0]),
                int(body_mask.shape[1]),
                int(clearance_mask.shape[0]),
                int(clearance_mask.shape[1]),
                int(clearance_origin[0]),
                int(clearance_origin[1]),
            )
        body_cpu = body_mask.to(device="cpu", dtype=torch.bool).contiguous()
        clear_cpu = clearance_mask.to(device="cpu", dtype=torch.bool).contiguous()
        body_sig = hashlib.sha1(body_cpu.numpy().tobytes()).hexdigest()
        clear_sig = hashlib.sha1(clear_cpu.numpy().tobytes()).hexdigest()
        return (
            "mask",
            int(body_cpu.shape[0]),
            int(body_cpu.shape[1]),
            body_sig,
            int(clear_cpu.shape[0]),
            int(clear_cpu.shape[1]),
            clear_sig,
            int(clearance_origin[0]),
            int(clearance_origin[1]),
        )

    def shape_tensors(self, shape_key: tuple) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]:
        out = self._shape_tensors_by_key.get(shape_key, None)
        if out is None:
            raise KeyError(f"unknown shape_key for spec {self.id!r}: {shape_key!r}")
        return out

    # ----- device migration -----

    def set_device(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.device = device
        for i, t in enumerate(self._variant_entry_off_t):
            self._variant_entry_off_t[i] = t.to(device)
        for i, t in enumerate(self._variant_exit_off_t):
            self._variant_exit_off_t[i] = t.to(device)
        self._all_entry_offsets = self._all_entry_offsets.to(device)
        self._all_entry_mask = self._all_entry_mask.to(device)
        self._all_exit_offsets = self._all_exit_offsets.to(device)
        self._all_exit_mask = self._all_exit_mask.to(device)
        self._all_body_width = self._all_body_width.to(device)
        self._all_body_height = self._all_body_height.to(device)
        self._shape_variant_indices = {
            k: v.to(device) for k, v in self._shape_variant_indices.items()
        }
        moved: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        for key, (body_mask, clearance_mask, clearance_origin, is_rectangular) in self._shape_tensors_by_key.items():
            moved[key] = (
                body_mask.to(device=device, dtype=torch.bool),
                clearance_mask.to(device=device, dtype=torch.bool),
                clearance_origin,
                bool(is_rectangular),
            )
        self._shape_tensors_by_key = moved

    # ----- variant-level port offset computation -----

    @staticmethod
    def _compute_port_offsets_raw(
        rotation: int,
        mirror: bool,
        width: int,
        height: int,
        entries_rel: List[Tuple[float, float]],
        exits_rel: List[Tuple[float, float]],
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
        """Compute center-relative port offsets for a given rotation/mirror and source shape."""
        half_w = width / 2.0
        half_h = height / 2.0
        eo: List[Tuple[float, float]] = []
        for dx_bl, dy_bl in entries_rel:
            dx = float(dx_bl) - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = StaticSpec._rotate_point(dx, float(dy_bl) - half_h, rotation)
            eo.append((round(rdx, 6), round(rdy, 6)))
        xo: List[Tuple[float, float]] = []
        for dx_bl, dy_bl in exits_rel:
            dx = float(dx_bl) - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = StaticSpec._rotate_point(dx, float(dy_bl) - half_h, rotation)
            xo.append((round(rdx, 6), round(rdy, 6)))
        return tuple(eo), tuple(xo)

    def _compute_port_offsets(
        self, rotation: int, mirror: bool,
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
        """Compute center-relative port offsets for a given rotation/mirror."""
        return self._compute_port_offsets_raw(
            rotation, mirror, self.width, self.height, self.entries_rel, self.exits_rel,
        )

    # ----- resolve (variant selection) -----

    def _make_placement(
        self,
        vi: StaticVariant,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
        flat_variant_index: int = 0,
    ) -> GroupPlacement:
        """Construct a concrete placement object. Subclasses override for their type."""
        raise NotImplementedError

    def resolve(
        self,
        *,
        x_bl: int,
        y_bl: int,
        variant_index: int,
        is_placeable_fn: Callable[..., bool],
    ) -> 'GroupPlacement | None':
        """Single-variant resolve from BL coordinates.

        Caller is responsible for iterating variants / picking the best
        (see ``env.resolve_action``).
        """
        vi = self._variants[variant_index]
        body_mask, clearance_mask, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
        if not is_placeable_fn(
            x_bl, y_bl,
            body_mask, clearance_mask, clearance_origin, is_rectangular,
        ):
            return None
        x_c = float(x_bl) + float(vi.body_width) / 2.0
        y_c = float(y_bl) + float(vi.body_height) / 2.0
        return self._make_placement(
            vi, x_c, y_c, x_bl, y_bl,
            body_mask, clearance_mask, clearance_origin, is_rectangular,
            variant_index,
        )

    def build_placement(self, *, variant_index: int, x_bl: int, y_bl: int) -> GroupPlacement:
        """Pure geometry — no placeability check. Caller already verified."""
        vi = self._variants[variant_index]
        body_mask, clearance_mask, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
        x_c = float(x_bl) + float(vi.body_width) / 2.0
        y_c = float(y_bl) + float(vi.body_height) / 2.0
        return self._make_placement(
            vi, x_c, y_c, x_bl, y_bl,
            body_mask, clearance_mask, clearance_origin, is_rectangular,
            variant_index,
        )

    # ----- placeable / cost batch API -----

    def placeable_map(self, state: "EnvState", gid: object) -> torch.Tensor:
        result = None
        for shape_key in self._variants_by_shape:
            body_mask, clearance_mask, clearance_origin, is_rectangular = self.shape_tensors(shape_key)
            m = state.placeable_map(
                gid=gid,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                is_rectangular=is_rectangular,
            )
            if result is None:
                result = m
            else:
                result = result | m
        if result is None:
            H, W = state.maps.shape
            return torch.zeros((H, W), dtype=torch.bool, device=self.device)
        return result

    def placeable_batch(
        self,
        state: "EnvState",
        gid: object,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        per_variant: bool = False,
    ) -> torch.Tensor:
        """BL-only batch placeability.  Loops S unique shape_keys.

        Args:
            x_bl, y_bl: ``[N, V]`` int64 — per-variant BL coordinates
                (adapter computes via ``_centers_to_bl``).

        Returns:
            per_variant=False: ``[N]`` bool — True if ANY variant placeable.
            per_variant=True:  ``[N, V]`` bool — per-variant result.
        """
        N = int(x_bl.shape[0])
        V = len(self._variants)
        if per_variant:
            result = torch.zeros((N, V), dtype=torch.bool, device=self.device)
            for sk, vi_t in self._shape_variant_indices.items():
                body_mask, clearance_mask, clearance_origin, is_rect = self.shape_tensors(sk)
                first = int(vi_t[0].item())
                ok_s = state.placeable_batch(
                    gid=gid,
                    x_bl=x_bl[:, first], y_bl=y_bl[:, first],
                    body_mask=body_mask, clearance_mask=clearance_mask,
                    clearance_origin=clearance_origin, is_rectangular=is_rect,
                )
                result[:, vi_t] = ok_s.unsqueeze(1).expand(-1, vi_t.shape[0])
            return result
        # per_variant=False: OR across shape_keys
        ok = torch.zeros(N, dtype=torch.bool, device=self.device)
        for sk, vi_t in self._shape_variant_indices.items():
            body_mask, clearance_mask, clearance_origin, is_rect = self.shape_tensors(sk)
            first = int(vi_t[0].item())
            ok |= state.placeable_batch(
                gid=gid,
                x_bl=x_bl[:, first], y_bl=y_bl[:, first],
                body_mask=body_mask, clearance_mask=clearance_mask,
                clearance_origin=clearance_origin, is_rectangular=is_rect,
            )
        return ok

    def score_batch(
        self,
        *,
        gid: object,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        state: "EnvState",
        reward: "RewardComposer",
        per_variant: bool = False,
    ) -> torch.Tensor:
        """Vectorized incremental cost for batch of BL positions.

        Two-phase approach:
          Phase 1 — placeability loop over S unique shape_keys (broadcast
                    to V variant columns).
          Phase 2 — flatten all valid (position, variant) pairs, build
                    features via gather, single ``delta_batch`` call.

        Args:
            x_bl, y_bl: ``[N, V]`` int64 — per-variant BL coordinates.

        Returns:
            per_variant=False: ``[N]`` float — min cost across variants.
            per_variant=True:  ``[N, V]`` float — per-variant cost (inf
                where not placeable).
        """
        N = int(x_bl.shape[0])
        V = len(self._variants)
        if N == 0:
            if per_variant:
                return torch.zeros((0, V), dtype=torch.float32, device=self.device)
            return torch.zeros((0,), dtype=torch.float32, device=self.device)

        # Phase 1: placeability — S shape_key iterations (not V)
        ok = torch.zeros((N, V), dtype=torch.bool, device=self.device)
        for sk, vi_t in self._shape_variant_indices.items():
            body_mask, clearance_mask, clearance_origin, is_rect = self.shape_tensors(sk)
            first = int(vi_t[0].item())
            ok_s = state.placeable_batch(
                gid=gid,
                x_bl=x_bl[:, first], y_bl=y_bl[:, first],
                body_mask=body_mask, clearance_mask=clearance_mask,
                clearance_origin=clearance_origin, is_rectangular=is_rect,
            )
            ok[:, vi_t] = ok_s.unsqueeze(1).expand(-1, vi_t.shape[0])

        result = torch.full((N, V), float('inf'), dtype=torch.float32, device=self.device)
        valid_n, valid_v = torch.where(ok)
        M = int(valid_n.shape[0])

        if M > 0:
            # Phase 2: vectorized feature build (BL direct, no snap-back)
            needed = reward.required()
            bw = self._all_body_width[valid_v]
            bh = self._all_body_height[valid_v]
            x_bl_f = x_bl[valid_n, valid_v].float()
            y_bl_f = y_bl[valid_n, valid_v].float()
            x_c = x_bl_f + bw / 2.0
            y_c = y_bl_f + bh / 2.0

            features: Dict[str, torch.Tensor] = {}
            if "min_x" in needed:
                features["min_x"] = x_bl_f
            if "max_x" in needed:
                features["max_x"] = x_bl_f + bw
            if "min_y" in needed:
                features["min_y"] = y_bl_f
            if "max_y" in needed:
                features["max_y"] = y_bl_f + bh
            if "entry_points" in needed or "exit_points" in needed:
                c = torch.stack([x_c, y_c], dim=1)
            if "entry_points" in needed:
                if self._all_entry_offsets.shape[1] > 0:
                    features["entry_points"] = c[:, None, :] + self._all_entry_offsets[valid_v]
                    features["entry_mask"] = self._all_entry_mask[valid_v]
                else:
                    features["entry_points"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                    features["entry_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)
            if "exit_points" in needed:
                if self._all_exit_offsets.shape[1] > 0:
                    features["exit_points"] = c[:, None, :] + self._all_exit_offsets[valid_v]
                    features["exit_mask"] = self._all_exit_mask[valid_v]
                else:
                    features["exit_points"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                    features["exit_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)

            scores = reward.delta_batch(state, gid=gid, **features)
            result[valid_n, valid_v] = scores

        if per_variant:
            return result
        return result.min(dim=1).values


# ---------------------------------------------------------------------------
# StaticRectSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticRectSpec(StaticSpec):
    """Static rectangular facility spec."""

    def __post_init__(self) -> None:
        self._entry_port_span = normalize_port_span(self._entry_port_span, name="entry_port_span")
        self._exit_port_span = normalize_port_span(self._exit_port_span, name="exit_port_span")
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticRectSpec {self.id!r} must have positive width/height, "
                f"got width={self.width}, height={self.height}"
            )

        # Build source shape list: either from _variant_defs or single top-level shape
        if self._variant_defs:
            source_shapes = self._variant_defs
        else:
            source_shapes = [
                {
                    "width": self.width,
                    "height": self.height,
                    "entries_rel": self.entries_rel,
                    "exits_rel": self.exits_rel,
                    "rotatable": self.rotatable,
                    "mirrorable": self.mirrorable,
                    "clearance_lrtb_rel": self.clearance_lrtb_rel,
                },
            ]

        seen_cost: set = set()
        variants_list: List[StaticVariant] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_variant_count = 0

        for src_idx, src in enumerate(source_shapes):
            src_w = int(src["width"])
            src_h = int(src["height"])
            src_entries = list(src["entries_rel"])
            src_exits = list(src["exits_rel"])
            src_rotatable = bool(src.get("rotatable", self.rotatable))
            src_mirrorable = bool(src.get("mirrorable", self.mirrorable))
            src_cl = src.get("clearance_lrtb_rel", self.clearance_lrtb_rel)

            if self._entry_port_span != PORT_SPAN_ALL and self._entry_port_span > len(src_entries):
                logger.warning(
                    "StaticRectSpec %r source[%d] entry_port_span=%d exceeds available=%d; using %d",
                    self.id,
                    src_idx,
                    int(self._entry_port_span),
                    len(src_entries),
                    len(src_entries),
                )
            if self._exit_port_span != PORT_SPAN_ALL and self._exit_port_span > len(src_exits):
                logger.warning(
                    "StaticRectSpec %r source[%d] exit_port_span=%d exceeds available=%d; using %d",
                    self.id,
                    src_idx,
                    int(self._exit_port_span),
                    len(src_exits),
                    len(src_exits),
                )

            if src_w <= 0 or src_h <= 0:
                raise ValueError(
                    f"StaticRectSpec {self.id!r} source[{src_idx}] must have positive "
                    f"width/height, got width={src_w}, height={src_h}"
                )

            # Validate ports
            invalid_ports: List[str] = []
            bounds = f"[0, {src_w}] x [0, {src_h}]"
            for port_type, ports in (("entry", src_entries), ("exit", src_exits)):
                for idx, port in enumerate(ports):
                    px, py = float(port[0]), float(port[1])
                    if px < 0.0 or px > float(src_w) or py < 0.0 or py > float(src_h):
                        invalid_ports.append(f"{port_type}[{idx}]=({px}, {py})")
            if invalid_ports:
                logger.warning(
                    f"StaticRectSpec {self.id!r} source[{src_idx}] has ports outside "
                    f"local bounds {bounds}: " + ", ".join(invalid_ports)
                )

            rotations = (0, 90, 180, 270) if src_rotatable else (0,)
            mirrors = (False, True) if src_mirrorable else (False,)

            for rot in rotations:
                rr = self._norm_rotation(rot)
                for m in mirrors:
                    raw_variant_count += 1
                    w, h = (src_w, src_h) if rr in (0, 180) else (src_h, src_w)
                    body_width, body_height = int(w), int(h)
                    cL, cR, cB, cT = self._clearance_lrtb_for_rotation_raw(
                        rr, mirror=m, clearance_lrtb_rel=src_cl)
                    eo_t, xo_t = self._compute_port_offsets_raw(
                        rr, m, src_w, src_h, src_entries, src_exits)

                body_mask = torch.ones((body_height, body_width), dtype=torch.bool, device=self.device)
                clearance_mask = torch.ones(
                    (body_height + cB + cT, body_width + cL + cR),
                    dtype=torch.bool, device=self.device,
                )
                clearance_origin = (int(cL), int(cB))
                is_rectangular = True
                pk = self._make_shape_key(
                    body_mask=body_mask, clearance_mask=clearance_mask,
                    clearance_origin=clearance_origin, is_rectangular=is_rectangular,
                )
                ck = (body_width, body_height, cL, cR, cB, cT, eo_t, xo_t)
                if ck in seen_cost:
                    continue
                seen_cost.add(ck)

                vi = StaticVariant(
                    source_index=src_idx, rotation=rr, mirror=m,
                    body_width=body_width, body_height=body_height,
                    cL=cL, cR=cR, cB=cB, cT=cT,
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t, exit_offsets=xo_t,
                    shape_key=pk, cost_key=ck,
                )
                variants_list.append(vi)
                shape_tensors_by_key[pk] = (body_mask, clearance_mask, clearance_origin, is_rectangular)

        self._store_variants(variants_list, shape_tensors_by_key)
        self._resolve_port_span_all()

        logger.info(
            "StaticRectSpec %r variant summary: sources=%d, raw=%d, unique=%d, unique_shapes=%d",
            self.id, len(source_shapes), raw_variant_count,
            len(self._variants), len(self._variants_by_shape),
        )

    @property
    def body_area(self) -> float:
        return float(self.width) * float(self.height)

    def _make_placement(
        self,
        vi: StaticVariant,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
        flat_variant_index: int = 0,
    ) -> StaticRectPlacement:
        entry_points = [(x_c_s + dx, y_c_s + dy) for dx, dy in vi.entry_offsets]
        exit_points = [(x_c_s + dx, y_c_s + dy) for dx, dy in vi.exit_offsets]
        w = float(vi.body_width)
        h = float(vi.body_height)
        return StaticRectPlacement(
            group_id=self.id,
            x_center=x_c_s, y_center=y_c_s,
            entry_points=entry_points, exit_points=exit_points,
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_mask=body_mask, clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl, y_bl=y_bl, rotation=vi.rotation,
            w=w, h=h,
            clearance_left=vi.cL, clearance_right=vi.cR,
            clearance_bottom=vi.cB, clearance_top=vi.cT,
            mirror=vi.mirror,
            variant_index=flat_variant_index,
        )


# ---------------------------------------------------------------------------
# StaticIrregularSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticIrregularSpec(StaticSpec):
    """Static irregular facility spec backed by polygon definitions.

    ``body_polygon`` is the list of (x, y) vertices in body-local BL coords.
    ``clearance_polygon`` optionally defines the clearance outline in the
    same coordinate system (may extend to negative coords).  When *None*,
    ``clearance_lrtb_rel`` from the base class is used
    to auto-generate the clearance map via dilation.
    """

    body_polygon: List[Tuple[float, float]] = field(default_factory=list)
    clearance_polygon: Optional[List[Tuple[float, float]]] = None

    body_map_canonical: Any = field(init=False, default=None, repr=False)
    clearance_map_canonical: Any = field(init=False, default=None, repr=False)
    clearance_origin_canonical: Tuple[int, int] = field(init=False, default=(0, 0), repr=False)

    def __post_init__(self) -> None:
        self._entry_port_span = normalize_port_span(self._entry_port_span, name="entry_port_span")
        self._exit_port_span = normalize_port_span(self._exit_port_span, name="exit_port_span")

        if not self.body_polygon:
            raise ValueError(f"StaticIrregularSpec {self.id!r} requires a non-empty body_polygon")

        canonical = self._rasterize_polygon(self.body_polygon, self.device)
        self.body_map_canonical = canonical
        self.height = int(canonical.shape[0])
        self.width = int(canonical.shape[1])
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} body_polygon rasterized to empty map "
                f"({self.height}x{self.width})"
            )

        if self.clearance_polygon is not None:
            cm, co = self._rasterize_clearance_polygon(
                self.clearance_polygon, self.body_polygon, self.device)
            self.clearance_map_canonical = cm
            self.clearance_origin_canonical = co
        else:
            cm, co = self._resolve_clearance(canonical)
            self.clearance_map_canonical = cm
            self.clearance_origin_canonical = co

        cH, cW = int(self.clearance_map_canonical.shape[0]), int(self.clearance_map_canonical.shape[1])
        bH, bW = self.height, self.width
        ox, oy = self.clearance_origin_canonical
        if cH < bH or cW < bW:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} clearance_mask ({cH}x{cW}) must be "
                f">= body_mask ({bH}x{bW})"
            )
        if ox < 0 or oy < 0 or ox + bW > cW or oy + bH > cH:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} body does not fit inside clearance_mask "
                f"at origin ({ox},{oy})"
            )

        if self._entry_port_span != PORT_SPAN_ALL and self._entry_port_span > len(self.entries_rel):
            logger.warning(
                "StaticIrregularSpec %r entry_port_span=%d exceeds available=%d; using %d",
                self.id,
                int(self._entry_port_span),
                len(self.entries_rel),
                len(self.entries_rel),
            )
        if self._exit_port_span != PORT_SPAN_ALL and self._exit_port_span > len(self.exits_rel):
            logger.warning(
                "StaticIrregularSpec %r exit_port_span=%d exceeds available=%d; using %d",
                self.id,
                int(self._exit_port_span),
                len(self.exits_rel),
                len(self.exits_rel),
            )

        self._validate_body_bbox()
        self._validate_ports()
        self._build_variants()

    # ----- rasterisation helpers -----

    @staticmethod
    def _rasterize_polygon(
        vertices: List[Tuple[float, float]],
        device: torch.device,
    ) -> torch.Tensor:
        """Rasterize a closed polygon to a ``bool[H, W]`` tensor.

        Vertices are in local BL coordinates.  The polygon's bounding box
        (snapped to integer grid) determines the tensor dimensions.  Cell
        ``(y, x)`` is ``True`` when its center ``(x+0.5, y+0.5)`` lies
        inside the polygon.
        """
        import numpy as np
        from matplotlib.path import Path as MplPath

        verts = np.asarray(vertices, dtype=np.float64)
        x_min, y_min = float(verts[:, 0].min()), float(verts[:, 1].min())
        x_max, y_max = float(verts[:, 0].max()), float(verts[:, 1].max())

        W = int(round(x_max - x_min))
        H = int(round(y_max - y_min))
        if W <= 0 or H <= 0:
            raise ValueError(f"polygon bbox must have positive area, got W={W}, H={H}")

        shifted = verts - np.array([x_min, y_min])
        cx = np.arange(W, dtype=np.float64) + 0.5
        cy = np.arange(H, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(cx, cy)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        mask = MplPath(shifted).contains_points(pts).reshape(H, W)
        return torch.as_tensor(mask, dtype=torch.bool, device=device).contiguous()

    @staticmethod
    def _rasterize_clearance_polygon(
        clearance_vertices: List[Tuple[float, float]],
        body_vertices: List[Tuple[float, float]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Rasterize clearance polygon and compute clearance_origin.

        Both polygons share the same body-local coordinate system.
        ``clearance_origin`` is the offset of the body BL ``(0,0)``
        inside the clearance_mask.
        """
        import numpy as np
        from matplotlib.path import Path as MplPath

        cverts = np.asarray(clearance_vertices, dtype=np.float64)
        bverts = np.asarray(body_vertices, dtype=np.float64)
        cx_min, cy_min = float(cverts[:, 0].min()), float(cverts[:, 1].min())
        cx_max, cy_max = float(cverts[:, 0].max()), float(cverts[:, 1].max())
        bx_min, by_min = float(bverts[:, 0].min()), float(bverts[:, 1].min())

        cW = int(round(cx_max - cx_min))
        cH = int(round(cy_max - cy_min))
        if cW <= 0 or cH <= 0:
            raise ValueError(f"clearance polygon bbox must have positive area, got W={cW}, H={cH}")

        shifted = cverts - np.array([cx_min, cy_min])
        px = np.arange(cW, dtype=np.float64) + 0.5
        py = np.arange(cH, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(px, py)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        mask = MplPath(shifted).contains_points(pts).reshape(cH, cW)
        clearance_mask = torch.as_tensor(mask, dtype=torch.bool, device=device).contiguous()

        ox = int(round(bx_min - cx_min))
        oy = int(round(by_min - cy_min))
        return clearance_mask, (ox, oy)

    @staticmethod
    def _transform_polygon_to_world(
        polygon: List[Tuple[float, float]],
        x_bl: float,
        y_bl: float,
        rotation: int,
        mirror: bool,
        body_width: int,
        body_height: int,
    ) -> List[Tuple[float, float]]:
        """Transform canonical polygon vertices to world coordinates."""
        half_w = body_width / 2.0
        half_h = body_height / 2.0
        r = rotation % 360
        if r in (90, 270):
            rw, rh = float(body_height), float(body_width)
        else:
            rw, rh = float(body_width), float(body_height)
        cx = x_bl + rw / 2.0
        cy = y_bl + rh / 2.0
        result: List[Tuple[float, float]] = []
        for px, py in polygon:
            dx = px - half_w
            dy = py - half_h
            if mirror:
                dx = -dx
            if r == 90:
                dx, dy = dy, -dx
            elif r == 180:
                dx, dy = -dx, -dy
            elif r == 270:
                dx, dy = -dy, dx
            result.append((cx + dx, cy + dy))
        return result

    # ----- properties / device migration -----

    @property
    def body_area(self) -> float:
        return float(self.body_map_canonical.to(dtype=torch.int32).sum().item())

    def set_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.device == device:
            return
        self.body_map_canonical = self.body_map_canonical.to(device=device, dtype=torch.bool)
        self.clearance_map_canonical = self.clearance_map_canonical.to(device=device, dtype=torch.bool)
        super().set_device(device)

    # ----- placement builders -----

    def _make_placement(
        self,
        vi: StaticVariant,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
        flat_variant_index: int = 0,
    ) -> StaticIrregularPlacement:
        entry_points = [(x_c_s + dx, y_c_s + dy) for dx, dy in vi.entry_offsets]
        exit_points = [(x_c_s + dx, y_c_s + dy) for dx, dy in vi.exit_offsets]
        w = float(vi.body_width)
        h = float(vi.body_height)
        si = vi.source_index
        src_bp = self._source_body_polygons[si]
        src_cp = self._source_clearance_polygons[si]
        src_w, src_h = self._source_canonical_sizes[si]
        bp_abs = (self._transform_polygon_to_world(
            src_bp, x_bl, y_bl, vi.rotation, vi.mirror,
            src_w, src_h) if src_bp else None)
        cp_abs = (self._transform_polygon_to_world(
            src_cp, x_bl, y_bl, vi.rotation, vi.mirror,
            src_w, src_h) if src_cp else None)
        return StaticIrregularPlacement(
            group_id=self.id,
            x_center=x_c_s, y_center=y_c_s,
            entry_points=entry_points, exit_points=exit_points,
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_mask=body_mask, clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl, y_bl=y_bl, rotation=vi.rotation,
            w=w, h=h,
            mirror=vi.mirror,
            variant_index=flat_variant_index,
            body_polygon_abs=bp_abs,
            clearance_polygon_abs=cp_abs,
        )

    # ----- irregular-specific helpers -----

    def _validate_body_bbox(self) -> None:
        bm = self.body_map_canonical
        if not bool(bm[0, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must touch the bottom boundary")
        if not bool(bm[-1, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must touch the top boundary")
        if not bool(bm[:, 0].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must touch the left boundary")
        if not bool(bm[:, -1].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must touch the right boundary")

    def _validate_ports(self) -> None:
        invalid_bounds: List[str] = []
        bounds = f"[0, {self.width}] x [0, {self.height}]"
        for port_type, ports in (("entry", self.entries_rel), ("exit", self.exits_rel)):
            for idx, port in enumerate(ports):
                x, y = float(port[0]), float(port[1])
                if x < 0.0 or x > float(self.width) or y < 0.0 or y > float(self.height):
                    invalid_bounds.append(f"{port_type}[{idx}]=({x}, {y})")
        if invalid_bounds:
            logger.warning(
                f"StaticIrregularSpec {self.id!r} has ports outside local bounds {bounds}: "
                + ", ".join(invalid_bounds)
            )

        import numpy as np
        from matplotlib.path import Path as MplPath

        poly = MplPath(np.asarray(self.body_polygon, dtype=np.float64))
        invalid_interior: List[str] = []
        for port_type, ports in (("entry", self.entries_rel), ("exit", self.exits_rel)):
            for idx, port in enumerate(ports):
                x, y = float(port[0]), float(port[1])
                if x < 0.0 or x > float(self.width) or y < 0.0 or y > float(self.height):
                    continue
                if not poly.contains_point((x, y)):
                    invalid_interior.append(f"{port_type}[{idx}]=({x}, {y})")
        if invalid_interior:
            logger.warning(
                f"StaticIrregularSpec {self.id!r} has ports outside body_polygon interior: "
                + ", ".join(invalid_interior)
            )

    @staticmethod
    def _transform_clearance_origin(
        origin: Tuple[int, int],
        body_size: Tuple[int, int],
        clearance_size: Tuple[int, int],
        rotation: int,
        mirror: bool,
    ) -> Tuple[int, int]:
        """Compute clearance_origin after rotation/mirror.

        ``origin`` is (ox, oy) of the body inside the clearance map in
        canonical orientation.  ``body_size`` is (bH, bW), ``clearance_size``
        is (cH, cW) — both in canonical orientation.
        """
        ox, oy = origin
        bH, bW = body_size
        _cH, cW = clearance_size
        cH = _cH
        if mirror:
            ox = cW - ox - bW
        if rotation == 0:
            return (ox, oy)
        if rotation == 90:
            return (oy, cW - ox - bW)
        if rotation == 180:
            return (cW - ox - bW, cH - oy - bH)
        if rotation == 270:
            return (cH - oy - bH, ox)
        raise ValueError(f"rotation must be a multiple of 90 degrees, got {rotation!r}")

    def _build_variants(self) -> None:
        # Build source list: multi-source from _variant_defs or single canonical
        if self._variant_defs:
            source_shapes = self._variant_defs
        else:
            source_shapes = [
                {
                    "width": self.width,
                    "height": self.height,
                    "entries_rel": self.entries_rel,
                    "exits_rel": self.exits_rel,
                    "rotatable": self.rotatable,
                    "mirrorable": self.mirrorable,
                    "clearance_lrtb_rel": self.clearance_lrtb_rel,
                    "body_map_canonical": self.body_map_canonical,
                    "clearance_map_canonical": self.clearance_map_canonical,
                    "clearance_origin_canonical": self.clearance_origin_canonical,
                    "body_polygon": self.body_polygon,
                    "clearance_polygon": self.clearance_polygon,
                },
            ]

        # Per-source polygon data for _make_placement
        self._source_body_polygons: List[List[Tuple[float, float]]] = []
        self._source_clearance_polygons: List[Optional[List[Tuple[float, float]]]] = []
        self._source_canonical_sizes: List[Tuple[int, int]] = []

        seen_cost: set = set()
        variants_list: List[StaticVariant] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_variant_count = 0

        for src_idx, src in enumerate(source_shapes):
            src_entries = list(src["entries_rel"])
            src_exits = list(src["exits_rel"])
            src_rotatable = bool(src.get("rotatable", self.rotatable))
            src_mirrorable = bool(src.get("mirrorable", self.mirrorable))

            if self._entry_port_span != PORT_SPAN_ALL and self._entry_port_span > len(src_entries):
                logger.warning(
                    "StaticIrregularSpec %r source[%d] entry_port_span=%d exceeds available=%d; using %d",
                    self.id,
                    src_idx,
                    int(self._entry_port_span),
                    len(src_entries),
                    len(src_entries),
                )
            if self._exit_port_span != PORT_SPAN_ALL and self._exit_port_span > len(src_exits):
                logger.warning(
                    "StaticIrregularSpec %r source[%d] exit_port_span=%d exceeds available=%d; using %d",
                    self.id,
                    src_idx,
                    int(self._exit_port_span),
                    len(src_exits),
                    len(src_exits),
                )

            # Get or rasterize canonical maps for this source
            if "body_map_canonical" in src:
                src_body_canon = src["body_map_canonical"]
                src_clear_canon = src["clearance_map_canonical"]
                src_clear_origin = src["clearance_origin_canonical"]
                src_w = int(src["width"])
                src_h = int(src["height"])
            else:
                # Rasterize from body_polygon
                src_bp = src["body_polygon"]
                src_body_canon = self._rasterize_polygon(src_bp, self.device)
                src_h = int(src_body_canon.shape[0])
                src_w = int(src_body_canon.shape[1])
                src_cp = src.get("clearance_polygon")
                if src_cp is not None:
                    src_clear_canon, src_clear_origin = self._rasterize_clearance_polygon(
                        src_cp, src_bp, self.device)
                else:
                    src_cl = src.get("clearance_lrtb_rel", self.clearance_lrtb_rel)
                    if src_cl is not None:
                        src_clear_canon, src_clear_origin = self._dilate_body_map(
                            src_body_canon, int(src_cl[0]), int(src_cl[1]),
                            int(src_cl[2]), int(src_cl[3]))
                    else:
                        src_clear_canon = src_body_canon.clone()
                        src_clear_origin = (0, 0)

            bH_canon, bW_canon = int(src_body_canon.shape[0]), int(src_body_canon.shape[1])
            cH_canon, cW_canon = int(src_clear_canon.shape[0]), int(src_clear_canon.shape[1])

            self._source_body_polygons.append(list(src.get("body_polygon", self.body_polygon)))
            self._source_clearance_polygons.append(src.get("clearance_polygon", self.clearance_polygon))
            self._source_canonical_sizes.append((src_w, src_h))

            rotations = (0, 90, 180, 270) if src_rotatable else (0,)
            mirrors = (False, True) if src_mirrorable else (False,)

            for rot in rotations:
                rr = self._norm_rotation(rot)
                for m in mirrors:
                    raw_variant_count += 1
                    body_mask = self._transform_body_map(src_body_canon, rr, mirror=m)
                    clearance_mask = self._transform_body_map(src_clear_canon, rr, mirror=m)
                    body_height = int(body_mask.shape[0])
                    body_width = int(body_mask.shape[1])
                    clearance_origin = self._transform_clearance_origin(
                        src_clear_origin,
                        (bH_canon, bW_canon), (cH_canon, cW_canon),
                        rr, m,
                    )
                    cL, cB = clearance_origin
                    cR = int(clearance_mask.shape[1]) - body_width - cL
                    cT = int(clearance_mask.shape[0]) - body_height - cB
                    is_rectangular = bool(body_mask.all().item()) and bool(clearance_mask.all().item())
                    eo_t, xo_t = self._compute_port_offsets_raw(
                        rr, m, src_w, src_h, src_entries, src_exits)

                    pk = self._make_shape_key(
                        body_mask=body_mask, clearance_mask=clearance_mask,
                        clearance_origin=clearance_origin, is_rectangular=is_rectangular,
                    )
                    ck = (pk, eo_t, xo_t)
                    if ck in seen_cost:
                        continue
                    seen_cost.add(ck)

                    vi = StaticVariant(
                        source_index=src_idx, rotation=rr, mirror=bool(m),
                        body_width=body_width, body_height=body_height,
                        cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT),
                        clearance_origin=clearance_origin,
                        is_rectangular=is_rectangular,
                        entry_offsets=eo_t, exit_offsets=xo_t,
                        shape_key=pk, cost_key=ck,
                    )
                    variants_list.append(vi)
                    shape_tensors_by_key[pk] = (body_mask, clearance_mask, clearance_origin, is_rectangular)

        self._store_variants(variants_list, shape_tensors_by_key)
        self._resolve_port_span_all()

        logger.info(
            "StaticIrregularSpec %r variant summary: sources=%d, raw=%d, unique=%d, unique_shapes=%d",
            self.id, len(source_shapes), raw_variant_count,
            len(self._variants), len(self._variants_by_shape),
        )

    def _coerce_body_map(self, body_mask: Any) -> torch.Tensor:
        t = torch.as_tensor(body_mask, dtype=torch.bool, device=self.device)
        if t.ndim != 2:
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must be 2D, got ndim={t.ndim}")
        if t.numel() == 0 or int(t.shape[0]) <= 0 or int(t.shape[1]) <= 0:
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must be non-empty")
        if not bool(t.any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_mask must contain at least one occupied cell")
        return t.contiguous()

    def _transform_body_map(self, body_mask: torch.Tensor, rotation: int, *, mirror: bool = False) -> torch.Tensor:
        src = body_mask.to(device=self.device, dtype=torch.bool).contiguous()
        H, W = int(src.shape[0]), int(src.shape[1])
        if mirror:
            src = torch.flip(src, dims=(1,))
        ys, xs = torch.nonzero(src, as_tuple=True)
        if rotation == 0:
            out_h, out_w = H, W
            out_x, out_y = xs, ys
        elif rotation == 90:
            out_h, out_w = W, H
            out_x, out_y = ys, (W - 1) - xs
        elif rotation == 180:
            out_h, out_w = H, W
            out_x, out_y = (W - 1) - xs, (H - 1) - ys
        elif rotation == 270:
            out_h, out_w = W, H
            out_x, out_y = (H - 1) - ys, xs
        else:
            raise ValueError(f"rotation must be a multiple of 90 degrees, got {rotation!r}")
        out = torch.zeros((out_h, out_w), dtype=torch.bool, device=self.device)
        out[out_y, out_x] = True
        return out



# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)

    H, W = 64, 64
    geom = StaticRectSpec(
        id="demo",
        device=device,
        width=8,
        height=4,
        entries_rel=[(0.0, 2.0), (8.0, 2.0)],
        exits_rel=[(4.0, 4.0)],
        clearance_lrtb_rel=(1, 1, 0, 0),
    )
    print(f"variants={len(geom._variants)}, unique_shapes={len(geom._variants_by_shape)}")
    for vi in geom._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_width},{vi.body_height}) "
              f"clear=({vi.cL},{vi.cR},{vi.cB},{vi.cT}) "
              f"entry_points={len(vi.entry_offsets)} exit_points={len(vi.exit_offsets)}")

    # resolve with variant_index=1 (90° rotation)
    def _always_true(*args):
        return True
    # resolve with variant_index=1 (90° rotation) — BL coords
    vi1 = geom._variants[1]
    _x_bl = int(round(14.0 - float(vi1.body_width) / 2.0))
    _y_bl = int(round(7.0 - float(vi1.body_height) / 2.0))
    resolved = geom.resolve(x_bl=_x_bl, y_bl=_y_bl, is_placeable_fn=_always_true, variant_index=1)
    print(
        f"\nresolve(x_bl={_x_bl},y_bl={_y_bl},variant_index=1) -> "
        f"w={resolved.w}, h={resolved.h}, entry_points={len(resolved.entry_points)}, exit_points={len(resolved.exit_points)}"
    )

    # --- Multi-source rect demo ---
    multi = StaticRectSpec(
        id="multi_rect",
        device=device,
        width=8,  # representative (first source)
        height=4,
        entries_rel=[(0.0, 2.0)],
        exits_rel=[(8.0, 2.0)],
        _variant_defs=[
            {"width": 8, "height": 4, "entries_rel": [(0.0, 2.0)], "exits_rel": [(8.0, 2.0)],
             "rotatable": True, "mirrorable": False, "clearance_lrtb_rel": (1, 1, 0, 0)},
            {"width": 10, "height": 3, "entries_rel": [(0.0, 1.5)], "exits_rel": [(10.0, 1.5)],
             "rotatable": True, "mirrorable": False, "clearance_lrtb_rel": (1, 1, 0, 0)},
        ],
    )
    print(f"\nmulti-source rect: sources={multi.num_sources}, variants={len(multi._variants)}, "
          f"unique_shapes={len(multi._variants_by_shape)}")
    for vi in multi._variants:
        print(f"  src={vi.source_index} rot={vi.rotation} body=({vi.body_width},{vi.body_height})")
    # resolve source_index=1 — pick first variant of that source
    _src_s, _src_e = multi._source_ranges[1]
    _vi_s = multi._variants[_src_s]
    _x_bl_s = int(round(20.0 - float(_vi_s.body_width) / 2.0))
    _y_bl_s = int(round(10.0 - float(_vi_s.body_height) / 2.0))
    r0 = multi.resolve(x_bl=_x_bl_s, y_bl=_y_bl_s, is_placeable_fn=_always_true, variant_index=_src_s)
    print(f"  resolve(source_index=1, vi={_src_s}) -> w={r0.w}, h={r0.h}, vi={r0.variant_index}")

    # --- Irregular demo (L-shape via polygon + auto clearance) ---
    irr = StaticIrregularSpec(
        id="L_demo",
        device=device,
        width=6,
        height=4,
        entries_rel=[(0.0, 1.0)],
        exits_rel=[(6.0, 1.0)],
        body_polygon=[(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)],
        clearance_lrtb_rel=(1, 1, 1, 1),
    )
    print(f"\nirregular variants={len(irr._variants)}, unique_shapes={len(irr._variants_by_shape)}")
    for vi in irr._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_width},{vi.body_height}) "
              f"rect={vi.is_rectangular} origin={vi.clearance_origin}")
    _vi_irr = irr._variants[0]
    _x_bl_i = int(round(20.0 - float(_vi_irr.body_width) / 2.0))
    _y_bl_i = int(round(10.0 - float(_vi_irr.body_height) / 2.0))
    ip = irr.resolve(x_bl=_x_bl_i, y_bl=_y_bl_i, is_placeable_fn=_always_true, variant_index=0)
    print(f"  resolve -> w={ip.w}, h={ip.h}, body_mask={tuple(ip.body_mask.shape)}")
    print(f"  body_polygon_abs={ip.body_polygon_abs}")
    print("OK")
