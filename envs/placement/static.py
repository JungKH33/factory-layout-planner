from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .base import PlacementBase

if TYPE_CHECKING:
    from ..state.base import EnvState
    from ..reward.core import RewardComposer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantInfo:
    """One unique (rotation, mirror) variant with precomputed geometry."""
    rotation: int
    mirror: bool
    body_w: int
    body_h: int
    cL: int
    cR: int
    cB: int
    cT: int
    clearance_origin: Tuple[int, int]
    is_rectangular: bool
    entry_offsets: Tuple[Tuple[float, float], ...]   # center-relative
    exit_offsets: Tuple[Tuple[float, float], ...]     # center-relative
    shape_key: tuple
    cost_key: tuple  # (body_w, body_h, cL, cR, cB, cT, entry_offsets, exit_offsets)

@dataclass
class StaticPlacement(PlacementBase):
    """Resolved absolute placement for a placed static group (world/placed orientation).

    Inherits the common placement contract from PlacementBase and adds
    geometry fields (w, h, x_c, y_c) that are specific to static (fixed-size) facilities.
    ``rotation`` is the concrete rotation (0/90/180/270) and ``mirror`` indicates
    whether a local x-axis flip was applied before rotation.

    Center coordinates use ``x_c``/``y_c`` naming, matching ``EnvAction``.
    """
    # TODO(modularization): move single-port selection (including empty -> center fallback)
    # from env into placement-level API so env does not handle entries/exits shape branching.
    w: float
    h: float
    x_c: float
    y_c: float
    mirror: bool = False


@dataclass
class StaticSpec:
    """Static facility spec and placement helper for grid-aligned groups.

    All fields are defined in the local frame before rotation:
    - width/height are the base footprint (unrotated)
    - entries_rel/exits_rel are offsets from the **bottom-left corner** of the
      unrotated facility (pre-rotation local frame).  (0,0) = BL corner,
      (width, height/2) = right edge midpoint, etc.
      At runtime the offset is converted to center-relative before rotation,
      because rotation is applied around the facility center.
    - clearance_*_rel are local L/R/B/T values
    """

    device: torch.device
    id: object
    width: int
    height: int
    entries_rel: List[Tuple[float, float]]
    exits_rel: List[Tuple[float, float]]
    clearance_left_rel: int
    clearance_right_rel: int
    clearance_bottom_rel: int
    clearance_top_rel: int
    rotatable: bool = True
    mirrorable: bool = False
    zone_values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate spec dimensions and flag ports outside the local pre-rotation box.
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticSpec {self.id!r} must have positive width/height, "
                f"got width={self.width}, height={self.height}"
            )

        invalid_ports: List[str] = []
        bounds = f"[0, {self.width}] x [0, {self.height}]"
        for port_type, ports in (("entry", self.entries_rel), ("exit", self.exits_rel)):
            for idx, port in enumerate(ports):
                x, y = float(port[0]), float(port[1])
                if x < 0.0 or x > float(self.width) or y < 0.0 or y > float(self.height):
                    invalid_ports.append(f"{port_type}[{idx}]=({x}, {y})")

        if invalid_ports:
            logger.warning(
                f"StaticSpec {self.id!r} has ports outside local bounds {bounds}: "
                + ", ".join(invalid_ports)
            )

        # --- Variant precomputation ---
        rotations = (0, 90, 180, 270) if self.rotatable else (0,)
        mirrors = (False, True) if self.mirrorable else (False,)

        half_w = self.width / 2.0
        half_h = self.height / 2.0

        seen_cost: set = set()
        variants: List[VariantInfo] = []
        variants_by_shape: Dict[tuple, List[VariantInfo]] = {}
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_variant_count = 0

        for rot in rotations:
            rr = self._resolve_rotation(rot)
            for m in mirrors:
                raw_variant_count += 1
                w, h = self._rotated_size(rr)
                body_w, body_h = int(w), int(h)
                cL, cR, cB, cT = self._clearance_lrtb(rr, mirror=m)

                eo: List[Tuple[float, float]] = []
                for dx_bl, dy_bl in self.entries_rel:
                    dx = float(dx_bl) - half_w
                    if bool(m):
                        dx = -dx
                    rdx, rdy = self._rotate_point(dx, float(dy_bl) - half_h, rr)
                    eo.append((round(rdx, 6), round(rdy, 6)))

                xo: List[Tuple[float, float]] = []
                for dx_bl, dy_bl in self.exits_rel:
                    dx = float(dx_bl) - half_w
                    if bool(m):
                        dx = -dx
                    rdx, rdy = self._rotate_point(dx, float(dy_bl) - half_h, rr)
                    xo.append((round(rdx, 6), round(rdy, 6)))

                eo_t = tuple(eo)
                xo_t = tuple(xo)
                body_map = torch.ones((body_h, body_w), dtype=torch.bool, device=self.device)
                clearance_map = torch.ones(
                    (body_h + cB + cT, body_w + cL + cR),
                    dtype=torch.bool,
                    device=self.device,
                )
                clearance_origin = (int(cL), int(cB))
                is_rectangular = True
                pk = self._make_shape_key(
                    body_map=body_map,
                    clearance_map=clearance_map,
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                )
                ck = (body_w, body_h, cL, cR, cB, cT, eo_t, xo_t)

                if ck in seen_cost:
                    continue
                seen_cost.add(ck)

                vi = VariantInfo(
                    rotation=rr, mirror=m,
                    body_w=body_w, body_h=body_h,
                    cL=cL, cR=cR, cB=cB, cT=cT,
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t, exit_offsets=xo_t,
                    shape_key=pk, cost_key=ck,
                )
                variants.append(vi)
                shape_tensors_by_key[pk] = (body_map, clearance_map, clearance_origin, is_rectangular)
                if pk not in variants_by_shape:
                    variants_by_shape[pk] = []
                variants_by_shape[pk].append(vi)

        self._variants: List[VariantInfo] = variants
        self._variants_by_shape: Dict[tuple, List[VariantInfo]] = variants_by_shape
        self._shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = shape_tensors_by_key

        # Precompute offset tensors per variant
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

        logger.info(
            "StaticSpec %r variant summary: raw=%d, unique_variants=%d, unique_shapes=%d",
            self.id,
            raw_variant_count,
            len(self._variants),          # ck 기준 unique
            len(self._variants_by_shape),  # shape 기준 unique
        )

    def set_device(self, device: torch.device) -> None:
        """Move precomputed tensors to *device* in-place."""
        if self.device == device:
            return
        self.device = device
        for i, t in enumerate(self._variant_entry_off_t):
            self._variant_entry_off_t[i] = t.to(device)
        for i, t in enumerate(self._variant_exit_off_t):
            self._variant_exit_off_t[i] = t.to(device)
        moved: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        for key, (body_map, clearance_map, clearance_origin, is_rectangular) in self._shape_tensors_by_key.items():
            moved[key] = (
                body_map.to(device=device, dtype=torch.bool),
                clearance_map.to(device=device, dtype=torch.bool),
                clearance_origin,
                bool(is_rectangular),
            )
        self._shape_tensors_by_key = moved

    @staticmethod
    def _make_shape_key(
        *,
        body_map: torch.Tensor,
        clearance_map: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> tuple:
        if bool(is_rectangular):
            return (
                "rect",
                int(body_map.shape[0]),
                int(body_map.shape[1]),
                int(clearance_map.shape[0]),
                int(clearance_map.shape[1]),
                int(clearance_origin[0]),
                int(clearance_origin[1]),
            )
        body_cpu = body_map.to(device="cpu", dtype=torch.bool).contiguous()
        clear_cpu = clearance_map.to(device="cpu", dtype=torch.bool).contiguous()
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


    @staticmethod
    def _norm_rotation(rotation: int) -> int:
        """Normalize rotation to {0,90,180,270} and validate."""
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
        """Return rotated (w,h) for 90-degree-multiple rotations."""
        r = self._resolve_rotation(rotation)
        if r in (90, 270):
            return (float(self.height), float(self.width))
        return (float(self.width), float(self.height))

    @staticmethod
    def _rotate_point(dx: float, dy: float, rotation: int) -> Tuple[float, float]:
        """Rotate a local point (dx,dy) CCW by multiples of 90 degrees."""
        r = StaticSpec._norm_rotation(rotation)
        if r == 0:
            return (float(dx), float(dy))
        if r == 90:
            return (float(dy), -float(dx))
        if r == 180:
            return (-float(dx), -float(dy))
        if r == 270:
            return (-float(dy), float(dx))
        # unreachable
        return (float(dx), float(dy))

    def _rotation_idx(self, rotation: torch.Tensor) -> torch.Tensor:
        r = torch.remainder(rotation, 360)
        if torch.any((r % 90) != 0):
            raise ValueError("rotation must be multiples of 90")
        if not bool(self.rotatable):
            return torch.zeros_like(r, dtype=torch.long)
        return (r // 90).to(dtype=torch.long)

    def _wh_for_rotation(self, rot_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_choices = torch.tensor(
            [float(self.width), float(self.height), float(self.width), float(self.height)],
            dtype=torch.float32,
            device=self.device,
        )
        h_choices = torch.tensor(
            [float(self.height), float(self.width), float(self.height), float(self.width)],
            dtype=torch.float32,
            device=self.device,
        )
        return w_choices[rot_idx], h_choices[rot_idx]

    @staticmethod
    def _offsets_rotation(base: torch.Tensor) -> torch.Tensor:
        dx = base[:, 0]
        dy = base[:, 1]
        r0 = torch.stack([dx, dy], dim=1)
        r90 = torch.stack([dy, -dx], dim=1)
        r180 = torch.stack([-dx, -dy], dim=1)
        r270 = torch.stack([-dy, dx], dim=1)
        return torch.stack([r0, r90, r180, r270], dim=0)  # [4,N,2]

    def _clearance_lrtb(self, rotation: int, *, mirror: bool = False) -> Tuple[int, int, int, int]:
        """Return clearance (cL, cR, cB, cT) after optional mirror + rotation."""
        r = self._resolve_rotation(rotation)
        cL, cR, cB, cT = (
            int(self.clearance_left_rel),
            int(self.clearance_right_rel),
            int(self.clearance_bottom_rel),
            int(self.clearance_top_rel),
        )
        if bool(mirror):
            cL, cR = cR, cL
        if r == 90:
            return (cB, cT, cR, cL)
        if r == 180:
            return (cR, cL, cT, cB)
        if r == 270:
            return (cT, cB, cL, cR)
        return (cL, cR, cB, cT)

    def rotated_size(self, rotation: int) -> Tuple[int, int]:
        """Return (body_w, body_h) for the given concrete rotation (0/90/180/270)."""
        r = self._resolve_rotation(rotation)
        w, h = self._rotated_size(r)
        return int(w), int(h)

    def _ports_from_center(
        self,
        x_c: float,
        y_c: float,
        rotation: int,
        ports_rel: List[Tuple[float, float]],
        *,
        mirror: bool = False,
    ) -> List[Tuple[float, float]]:
        """Convert BL-relative port offsets to absolute world coordinates.

        ports_rel are BL-relative offsets → center-relative (subtract half_w, half_h)
        → mirror+rotate → add world center (x_c, y_c).
        """
        r = self._resolve_rotation(rotation)
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        out = []
        for dx_bl, dy_bl in ports_rel:
            dx = dx_bl - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = self._rotate_point(dx, dy_bl - half_h, r)
            out.append((x_c + rdx, y_c + rdy))
        return out

    def _entries_from_center(self, x_c: float, y_c: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_c, y_c, rotation, self.entries_rel, mirror=mirror)

    def _exits_from_center(self, x_c: float, y_c: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_c, y_c, rotation, self.exits_rel, mirror=mirror)

    def _ports_batch(
        self,
        ports_rel: List[Tuple[float, float]],
        center: torch.Tensor,   # [M, 2]
        rot_idx: torch.Tensor,  # [M]
        *,
        mirror: bool = False,
    ) -> torch.Tensor:          # [M, N, 2]
        """Batch port computation from center coordinates.

        ports_rel (BL-relative offsets) → center-relative (subtract half_wh)
        → optional mirror (negate dx) → _offsets_rotation → add world center.
        """
        M = int(center.shape[0])
        if not ports_rel:
            return torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
        base = torch.tensor(ports_rel, dtype=torch.float32, device=self.device)  # [N, 2]
        half_wh = torch.tensor(
            [self.width / 2.0, self.height / 2.0],
            dtype=torch.float32, device=self.device,
        )
        center_offsets = base - half_wh                     # [N, 2]  BL → center-relative
        if mirror:
            center_offsets = center_offsets.clone()
            center_offsets[:, 0] = -center_offsets[:, 0]
        rot_offsets = self._offsets_rotation(center_offsets)     # [4, N, 2]
        return center[:, None, :] + rot_offsets[rot_idx]    # [M, N, 2]

    def _entries_batch(self, center: torch.Tensor, rot_idx: torch.Tensor, *, mirror: bool = False) -> torch.Tensor:
        return self._ports_batch(self.entries_rel, center, rot_idx, mirror=mirror)

    def _exits_batch(self, center: torch.Tensor, rot_idx: torch.Tensor, *, mirror: bool = False) -> torch.Tensor:
        return self._ports_batch(self.exits_rel, center, rot_idx, mirror=mirror)

    def build_placement(
        self,
        *,
        x_c: float,
        y_c: float,
        rotation: int = 0,
        mirror: bool = False,
    ) -> StaticPlacement:
        """Build a StaticPlacement from center coords + concrete rotation.

        Used as a force-place fallback when resolve() returns None.
        """
        r = self._resolve_rotation(rotation)
        w, h = self._rotated_size(r)
        x_bl = int(round(x_c - w / 2.0))
        y_bl = int(round(y_c - h / 2.0))
        x_c_s = float(x_bl) + w / 2.0
        y_c_s = float(y_bl) + h / 2.0
        entries = list(self._entries_from_center(x_c_s, y_c_s, r, mirror=mirror))
        exits = list(self._exits_from_center(x_c_s, y_c_s, r, mirror=mirror))
        cL, cR, cB, cT = self._clearance_lrtb(r, mirror=mirror)
        body_map = torch.ones((int(h), int(w)), dtype=torch.bool, device=self.device)
        clearance_map = torch.ones(
            (int(h) + int(cB) + int(cT), int(w) + int(cL) + int(cR)),
            dtype=torch.bool,
            device=self.device,
        )
        clearance_origin = (int(cL), int(cB))
        return StaticPlacement(
            entries=entries,
            exits=exits,
            min_x=float(x_bl),
            max_x=float(x_bl) + w,
            min_y=float(y_bl),
            max_y=float(y_bl) + h,
            body_map=body_map,
            clearance_map=clearance_map,
            clearance_origin=clearance_origin,
            is_rectangular=True,
            x_bl=x_bl,
            y_bl=y_bl,
            rotation=int(r),
            w=float(w),
            h=float(h),
            x_c=x_c_s,
            y_c=y_c_s,
            clearance_left=int(cL),
            clearance_right=int(cR),
            clearance_bottom=int(cB),
            clearance_top=int(cT),
            mirror=bool(mirror),
        )

    def resolve(
        self,
        *,
        x_c: float,
        y_c: float,
        is_placeable_fn: Callable[..., bool],
        score_fn: Optional[Callable] = None,
    ) -> 'StaticPlacement | None':
        """Resolve center coordinates to the best concrete placement, or None.

        Tries all (rotation, mirror) variants, derives BL from center per variant,
        filters by placeability, dedupes by geometry signature, and picks the one
        with lowest score (delta_cost).

        Callbacks:
          is_placeable_fn(x_bl, y_bl, body_map, clearance_map, clearance_origin, is_rectangular) -> bool
          score_fn(placements: List[StaticPlacement]) -> Tensor[N]
        If *score_fn* is None, picks the first feasible variant.
        """
        placeable: List[StaticPlacement] = []
        for vi in self._variants:
            w = float(vi.body_w)
            h = float(vi.body_h)
            x_bl = int(round(x_c - w / 2.0))
            y_bl = int(round(y_c - h / 2.0))
            x_c_s = float(x_bl) + w / 2.0
            y_c_s = float(y_bl) + h / 2.0
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)

            if not is_placeable_fn(
                x_bl,
                y_bl,
                body_map,
                clearance_map,
                clearance_origin,
                is_rectangular,
            ):
                continue

            entries = list(self._entries_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
            exits = list(self._exits_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
            placeable.append(StaticPlacement(
                entries=entries, exits=exits,
                min_x=float(x_bl), max_x=float(x_bl) + w,
                min_y=float(y_bl), max_y=float(y_bl) + h,
                body_map=body_map,
                clearance_map=clearance_map,
                clearance_origin=clearance_origin,
                is_rectangular=bool(is_rectangular),
                x_bl=x_bl, y_bl=y_bl, rotation=vi.rotation,
                w=w, h=h, x_c=x_c_s, y_c=y_c_s,
                clearance_left=vi.cL, clearance_right=vi.cR,
                clearance_bottom=vi.cB, clearance_top=vi.cT,
                mirror=vi.mirror,
            ))

        if not placeable:
            return None
        if score_fn is None or len(placeable) == 1:
            return placeable[0]

        scores = score_fn(placeable).to(dtype=torch.float32, device=self.device).view(-1)
        return placeable[int(torch.argmin(scores).item())]

    # ------------------------------------------------------------------
    # Unified placeable / cost API (variant-aware)
    # ------------------------------------------------------------------

    def placeable_map(self, state: "EnvState", gid: object) -> torch.Tensor:
        """Return [H,W] bool BL-indexed placeable map, OR'd across all unique shapes."""
        result = None
        for shape_key in self._variants_by_shape:
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(shape_key)
            m = state.is_placeable_map(
                gid=gid,
                body_map=body_map,
                clearance_map=clearance_map,
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
        x_c: torch.Tensor,
        y_c: torch.Tensor,
    ) -> torch.Tensor:
        """Check if center coordinates are placeable in ANY variant. Returns [N] bool."""
        N = int(x_c.shape[0])
        ok = torch.zeros(N, dtype=torch.bool, device=self.device)
        for shape_key in self._variants_by_shape:
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(shape_key)
            body_h, body_w = int(body_map.shape[0]), int(body_map.shape[1])
            x_bl = torch.round(x_c - body_w / 2.0).to(torch.long)
            y_bl = torch.round(y_c - body_h / 2.0).to(torch.long)
            shape_ok = state.is_placeable_batch(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_map=body_map,
                clearance_map=clearance_map,
                clearance_origin=clearance_origin,
                is_rectangular=is_rectangular,
            )
            ok = ok | shape_ok
        return ok

    def cost_batch(
        self,
        *,
        gid: object,
        poses: torch.Tensor,
        state: "EnvState",
        reward: "RewardComposer",
    ) -> torch.Tensor:
        """[N] float — min delta_cost across PLACEABLE variants. inf if no variant fits."""
        N = int(poses.shape[0])
        if N == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)

        best = torch.full((N,), float('inf'), dtype=torch.float32, device=self.device)
        x_c = poses[:, 0]
        y_c = poses[:, 1]
        needed = reward.required()

        for i, vi in enumerate(self._variants):
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
            x_bl = torch.round(x_c - vi.body_w / 2.0).to(torch.long)
            y_bl = torch.round(y_c - vi.body_h / 2.0).to(torch.long)
            ok = state.is_placeable_batch(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_map=body_map,
                clearance_map=clearance_map,
                clearance_origin=clearance_origin,
                is_rectangular=is_rectangular,
            )
            if not ok.any():
                continue
            idx = torch.where(ok)[0]
            features = self._build_variant_features(vi, i, poses[idx], needed)
            scores = reward.delta_batch(state, gid=gid, **features)
            best[idx] = torch.min(best[idx], scores)

        return best

    def _build_variant_features(
        self,
        vi: VariantInfo,
        vi_idx: int,
        center: torch.Tensor,
        needed: set,
    ) -> Dict[str, torch.Tensor]:
        """Build feature dict for a single variant at given center positions."""
        M = int(center.shape[0])
        out: Dict[str, torch.Tensor] = {}

        x_c = center[:, 0]
        y_c = center[:, 1]

        # Snap center → BL (integer) → snapped center
        x_bl = torch.round(x_c - vi.body_w / 2.0).to(torch.long)
        y_bl = torch.round(y_c - vi.body_h / 2.0).to(torch.long)
        x_c_s = x_bl.to(torch.float32) + vi.body_w / 2.0
        y_c_s = y_bl.to(torch.float32) + vi.body_h / 2.0

        if "min_x" in needed:
            out["min_x"] = x_bl.to(torch.float32)
        if "max_x" in needed:
            out["max_x"] = x_bl.to(torch.float32) + float(vi.body_w)
        if "min_y" in needed:
            out["min_y"] = y_bl.to(torch.float32)
        if "max_y" in needed:
            out["max_y"] = y_bl.to(torch.float32) + float(vi.body_h)

        if "entries" in needed:
            eo = self._variant_entry_off_t[vi_idx]  # [E, 2]
            if eo.shape[0] > 0:
                c = torch.stack([x_c_s, y_c_s], dim=1)  # [M, 2]
                out["entries"] = c[:, None, :] + eo[None, :, :]  # [M, E, 2]
                out["entries_mask"] = torch.ones(
                    (M, eo.shape[0]), dtype=torch.bool, device=self.device,
                )
            else:
                out["entries"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                out["entries_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)

        if "exits" in needed:
            xo = self._variant_exit_off_t[vi_idx]  # [X, 2]
            if xo.shape[0] > 0:
                c = torch.stack([x_c_s, y_c_s], dim=1)  # [M, 2]
                out["exits"] = c[:, None, :] + xo[None, :, :]  # [M, X, 2]
                out["exits_mask"] = torch.ones(
                    (M, xo.shape[0]), dtype=torch.bool, device=self.device,
                )
            else:
                out["exits"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                out["exits_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)

        return out

    # build_candidate_features removed — replaced by _build_variant_features + cost_batch


if __name__ == "__main__":
    import time

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)

    H, W = 64, 64
    geom = StaticSpec(
        id="demo",
        device=device,
        width=8,
        height=4,
        entries_rel=[(0.0, 2.0), (8.0, 2.0)],
        exits_rel=[(4.0, 4.0)],
        clearance_left_rel=1,
        clearance_right_rel=1,
        clearance_bottom_rel=0,
        clearance_top_rel=0,
    )

    # --- Variant precomputation ---
    print(f"variants={len(geom._variants)}, unique_shapes={len(geom._variants_by_shape)}")
    for vi in geom._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"clear=({vi.cL},{vi.cR},{vi.cB},{vi.cT}) "
              f"entries={len(vi.entry_offsets)} exits={len(vi.exit_offsets)}")

    # --- build_placement ---
    scalar = geom.build_placement(x_c=14.0, y_c=7.0, rotation=90)
    print(
        f"\nbuild_placement(x_c=14.0,y_c=7.0,rotation=90) -> "
        f"w={scalar.w}, h={scalar.h}, entries={len(scalar.entries)}, exits={len(scalar.exits)}"
    )

    # --- Mirrorable spec ---
    geom_m = StaticSpec(
        id="mirror_demo",
        device=device,
        width=8,
        height=4,
        entries_rel=[(0.0, 2.0)],
        exits_rel=[(8.0, 2.0)],
        clearance_left_rel=2,
        clearance_right_rel=1,
        clearance_bottom_rel=0,
        clearance_top_rel=0,
        mirrorable=True,
    )
    print(f"\nmirrorable: variants={len(geom_m._variants)}, unique_shapes={len(geom_m._variants_by_shape)}")
    for vi in geom_m._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"clear=({vi.cL},{vi.cR},{vi.cB},{vi.cT})")
    print("OK")
