from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .base import GroupSpec, GroupPlacement

if TYPE_CHECKING:
    from ..state.base import EnvState
    from ..reward.core import RewardComposer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

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
    entry_offsets: Tuple[Tuple[float, float], ...]
    exit_offsets: Tuple[Tuple[float, float], ...]
    shape_key: tuple
    cost_key: tuple


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


@dataclass
class StaticIrregularPlacement(GroupPlacement):
    """Resolved absolute placement for an irregular static group."""

    x_bl: int
    y_bl: int
    rotation: int
    w: float
    h: float
    mirror: bool = False


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
    - placeable_map / placeable_batch / cost_batch
    - resolve() — tries every variant and picks the best
    - shape_tensors() — returns (body_map, clearance_map, origin, is_rect) by key

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
    _entry_port_mode: str = "min"
    _exit_port_mode: str = "min"

    # ----- variant storage (populated by _store_variants) -----

    def _store_variants(
        self,
        variants: List[VariantInfo],
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]],
    ) -> None:
        self._variants: List[VariantInfo] = variants
        self._variants_by_shape: Dict[tuple, List[VariantInfo]] = {}
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

    # ----- port coordinate helpers -----

    def _ports_from_center(
        self,
        x_c: float,
        y_c: float,
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
            out.append((x_c + rdx, y_c + rdy))
        return out

    def _entries_from_center(self, x_c: float, y_c: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_c, y_c, rotation, self.entries_rel, mirror=mirror)

    def _exits_from_center(self, x_c: float, y_c: float, rotation: int, *, mirror: bool = False) -> List[Tuple[float, float]]:
        return self._ports_from_center(x_c, y_c, rotation, self.exits_rel, mirror=mirror)

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
            dtype=torch.float32, device=self.device,
        )
        h_choices = torch.tensor(
            [float(self.height), float(self.width), float(self.height), float(self.width)],
            dtype=torch.float32, device=self.device,
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

    def _ports_batch(
        self,
        ports_rel: List[Tuple[float, float]],
        center: torch.Tensor,
        rot_idx: torch.Tensor,
        *,
        mirror: bool = False,
    ) -> torch.Tensor:
        M = int(center.shape[0])
        if not ports_rel:
            return torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
        base = torch.tensor(ports_rel, dtype=torch.float32, device=self.device)
        half_wh = torch.tensor(
            [self.width / 2.0, self.height / 2.0],
            dtype=torch.float32, device=self.device,
        )
        center_offsets = base - half_wh
        if mirror:
            center_offsets = center_offsets.clone()
            center_offsets[:, 0] = -center_offsets[:, 0]
        rot_offsets = self._offsets_rotation(center_offsets)
        return center[:, None, :] + rot_offsets[rot_idx]

    def _entries_batch(self, center: torch.Tensor, rot_idx: torch.Tensor, *, mirror: bool = False) -> torch.Tensor:
        return self._ports_batch(self.entries_rel, center, rot_idx, mirror=mirror)

    def _exits_batch(self, center: torch.Tensor, rot_idx: torch.Tensor, *, mirror: bool = False) -> torch.Tensor:
        return self._ports_batch(self.exits_rel, center, rot_idx, mirror=mirror)

    # ----- shape key / tensors -----

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

    # ----- device migration -----

    def set_device(self, device: torch.device) -> None:
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

    # ----- variant-level port offset computation -----

    def _compute_port_offsets(
        self, rotation: int, mirror: bool,
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
        """Compute center-relative port offsets for a given rotation/mirror."""
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        eo: List[Tuple[float, float]] = []
        for dx_bl, dy_bl in self.entries_rel:
            dx = float(dx_bl) - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = self._rotate_point(dx, float(dy_bl) - half_h, rotation)
            eo.append((round(rdx, 6), round(rdy, 6)))
        xo: List[Tuple[float, float]] = []
        for dx_bl, dy_bl in self.exits_rel:
            dx = float(dx_bl) - half_w
            if bool(mirror):
                dx = -dx
            rdx, rdy = self._rotate_point(dx, float(dy_bl) - half_h, rotation)
            xo.append((round(rdx, 6), round(rdy, 6)))
        return tuple(eo), tuple(xo)

    # ----- resolve (variant selection) -----

    def _make_placement(
        self,
        vi: VariantInfo,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_map: torch.Tensor,
        clearance_map: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> GroupPlacement:
        """Construct a concrete placement object. Subclasses override for their type."""
        raise NotImplementedError

    def resolve(
        self,
        *,
        x_c: float,
        y_c: float,
        is_placeable_fn: Callable[..., bool],
        score_fn: Optional[Callable] = None,
    ) -> 'GroupPlacement | None':
        placeable: List[GroupPlacement] = []
        for vi in self._variants:
            w = float(vi.body_w)
            h = float(vi.body_h)
            x_bl = int(round(x_c - w / 2.0))
            y_bl = int(round(y_c - h / 2.0))
            x_c_s = float(x_bl) + w / 2.0
            y_c_s = float(y_bl) + h / 2.0
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)

            if not is_placeable_fn(
                x_bl, y_bl,
                body_map, clearance_map, clearance_origin, is_rectangular,
            ):
                continue

            placeable.append(self._make_placement(
                vi, x_c_s, y_c_s, x_bl, y_bl,
                body_map, clearance_map, clearance_origin, is_rectangular,
            ))

        if not placeable:
            return None
        if score_fn is None or len(placeable) == 1:
            return placeable[0]
        scores = score_fn(placeable).to(dtype=torch.float32, device=self.device).view(-1)
        return placeable[int(torch.argmin(scores).item())]

    # ----- placeable / cost batch API -----

    def placeable_map(self, state: "EnvState", gid: object) -> torch.Tensor:
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
        N = int(x_c.shape[0])
        ok = torch.zeros(N, dtype=torch.bool, device=self.device)
        for shape_key in self._variants_by_shape:
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(shape_key)
            body_h, body_w = int(body_map.shape[0]), int(body_map.shape[1])
            x_bl = torch.round(x_c - body_w / 2.0).to(torch.long)
            y_bl = torch.round(y_c - body_h / 2.0).to(torch.long)
            shape_ok = state.is_placeable_batch(
                gid=gid,
                x_bl=x_bl, y_bl=y_bl,
                body_map=body_map, clearance_map=clearance_map,
                clearance_origin=clearance_origin, is_rectangular=is_rectangular,
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
                x_bl=x_bl, y_bl=y_bl,
                body_map=body_map, clearance_map=clearance_map,
                clearance_origin=clearance_origin, is_rectangular=is_rectangular,
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
        M = int(center.shape[0])
        out: Dict[str, torch.Tensor] = {}
        x_c = center[:, 0]
        y_c = center[:, 1]
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
            eo = self._variant_entry_off_t[vi_idx]
            if eo.shape[0] > 0:
                c = torch.stack([x_c_s, y_c_s], dim=1)
                out["entries"] = c[:, None, :] + eo[None, :, :]
                out["entries_mask"] = torch.ones(
                    (M, eo.shape[0]), dtype=torch.bool, device=self.device,
                )
            else:
                out["entries"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                out["entries_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)
        if "exits" in needed:
            xo = self._variant_exit_off_t[vi_idx]
            if xo.shape[0] > 0:
                c = torch.stack([x_c_s, y_c_s], dim=1)
                out["exits"] = c[:, None, :] + xo[None, :, :]
                out["exits_mask"] = torch.ones(
                    (M, xo.shape[0]), dtype=torch.bool, device=self.device,
                )
            else:
                out["exits"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                out["exits_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)
        return out


# ---------------------------------------------------------------------------
# StaticRectSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticRectSpec(StaticSpec):
    """Static rectangular facility spec."""

    clearance_left_rel: int = 0
    clearance_right_rel: int = 0
    clearance_bottom_rel: int = 0
    clearance_top_rel: int = 0

    def _clearance_lrtb(self, rotation: int, *, mirror: bool = False) -> Tuple[int, int, int, int]:
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

    def __post_init__(self) -> None:
        _valid_modes = ("min", "mean")
        if self._entry_port_mode not in _valid_modes:
            raise ValueError(f"entry_port_mode must be one of {_valid_modes}, got {self._entry_port_mode!r}")
        if self._exit_port_mode not in _valid_modes:
            raise ValueError(f"exit_port_mode must be one of {_valid_modes}, got {self._exit_port_mode!r}")
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticRectSpec {self.id!r} must have positive width/height, "
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
                f"StaticRectSpec {self.id!r} has ports outside local bounds {bounds}: "
                + ", ".join(invalid_ports)
            )

        rotations = (0, 90, 180, 270) if self.rotatable else (0,)
        mirrors = (False, True) if self.mirrorable else (False,)

        seen_cost: set = set()
        variants: List[VariantInfo] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_variant_count = 0

        for rot in rotations:
            rr = self._resolve_rotation(rot)
            for m in mirrors:
                raw_variant_count += 1
                w, h = self._rotated_size(rr)
                body_w, body_h = int(w), int(h)
                cL, cR, cB, cT = self._clearance_lrtb(rr, mirror=m)
                eo_t, xo_t = self._compute_port_offsets(rr, m)

                body_map = torch.ones((body_h, body_w), dtype=torch.bool, device=self.device)
                clearance_map = torch.ones(
                    (body_h + cB + cT, body_w + cL + cR),
                    dtype=torch.bool, device=self.device,
                )
                clearance_origin = (int(cL), int(cB))
                is_rectangular = True
                pk = self._make_shape_key(
                    body_map=body_map, clearance_map=clearance_map,
                    clearance_origin=clearance_origin, is_rectangular=is_rectangular,
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

        self._store_variants(variants, shape_tensors_by_key)

        logger.info(
            "StaticRectSpec %r variant summary: raw=%d, unique_variants=%d, unique_shapes=%d",
            self.id, raw_variant_count,
            len(self._variants), len(self._variants_by_shape),
        )

    @property
    def body_area(self) -> float:
        return float(self.width) * float(self.height)

    def _make_placement(
        self,
        vi: VariantInfo,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_map: torch.Tensor,
        clearance_map: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> StaticRectPlacement:
        entries = list(self._entries_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
        exits = list(self._exits_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
        w = float(vi.body_w)
        h = float(vi.body_h)
        return StaticRectPlacement(
            x_c=x_c_s, y_c=y_c_s,
            entries=entries, exits=exits,
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_map=body_map, clearance_map=clearance_map,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl, y_bl=y_bl, rotation=vi.rotation,
            w=w, h=h,
            clearance_left=vi.cL, clearance_right=vi.cR,
            clearance_bottom=vi.cB, clearance_top=vi.cT,
            mirror=vi.mirror,
        )

    def build_placement(
        self,
        *,
        x_c: float,
        y_c: float,
        rotation: int = 0,
        mirror: bool = False,
    ) -> StaticRectPlacement:
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
            dtype=torch.bool, device=self.device,
        )
        clearance_origin = (int(cL), int(cB))
        return StaticRectPlacement(
            x_c=x_c_s, y_c=y_c_s,
            entries=entries, exits=exits,
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_map=body_map, clearance_map=clearance_map,
            clearance_origin=clearance_origin, is_rectangular=True,
            x_bl=x_bl, y_bl=y_bl, rotation=int(r),
            w=float(w), h=float(h),
            clearance_left=int(cL), clearance_right=int(cR),
            clearance_bottom=int(cB), clearance_top=int(cT),
            mirror=bool(mirror),
        )


# ---------------------------------------------------------------------------
# StaticIrregularSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticIrregularSpec(StaticSpec):
    """Static irregular facility spec backed by a canonical occupancy mask.

    Clearance is defined directly via ``clearance_map_canonical`` (a 2-D boolean
    tensor that is >= body_map in size).  ``clearance_origin_canonical`` is the
    (x, y) offset where the body sits inside the clearance map.  If
    ``clearance_map_canonical`` is *None*, clearance equals the body (no extra
    clearance zone).
    """

    body_map_canonical: Any = None
    clearance_map_canonical: Any = None
    clearance_origin_canonical: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        _valid_modes = ("min", "mean")
        if self._entry_port_mode not in _valid_modes:
            raise ValueError(f"entry_port_mode must be one of {_valid_modes}, got {self._entry_port_mode!r}")
        if self._exit_port_mode not in _valid_modes:
            raise ValueError(f"exit_port_mode must be one of {_valid_modes}, got {self._exit_port_mode!r}")
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} must have positive width/height, "
                f"got width={self.width}, height={self.height}"
            )

        canonical = self._coerce_body_map(self.body_map_canonical)
        if int(canonical.shape[1]) != self.width or int(canonical.shape[0]) != self.height:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} body_map shape must match "
                f"(height={self.height}, width={self.width}), got {tuple(canonical.shape)!r}"
            )
        self.body_map_canonical = canonical

        if self.clearance_map_canonical is None:
            self.clearance_map_canonical = canonical.clone()
            self.clearance_origin_canonical = (0, 0)
        else:
            cm = self._coerce_body_map(self.clearance_map_canonical)
            self.clearance_map_canonical = cm
            ox, oy = int(self.clearance_origin_canonical[0]), int(self.clearance_origin_canonical[1])
            self.clearance_origin_canonical = (ox, oy)
            cH, cW = int(cm.shape[0]), int(cm.shape[1])
            bH, bW = self.height, self.width
            if cH < bH or cW < bW:
                raise ValueError(
                    f"StaticIrregularSpec {self.id!r} clearance_map ({cH}x{cW}) must be "
                    f">= body_map ({bH}x{bW})"
                )
            if ox < 0 or oy < 0 or ox + bW > cW or oy + bH > cH:
                raise ValueError(
                    f"StaticIrregularSpec {self.id!r} body does not fit inside clearance_map "
                    f"at origin ({ox},{oy})"
                )

        self._validate_body_bbox()
        self._validate_ports()
        self._build_variants()

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

    def _make_placement(
        self,
        vi: VariantInfo,
        x_c_s: float,
        y_c_s: float,
        x_bl: int,
        y_bl: int,
        body_map: torch.Tensor,
        clearance_map: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> StaticIrregularPlacement:
        entries = list(self._entries_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
        exits = list(self._exits_from_center(x_c_s, y_c_s, vi.rotation, mirror=vi.mirror))
        w = float(vi.body_w)
        h = float(vi.body_h)
        return StaticIrregularPlacement(
            x_c=x_c_s, y_c=y_c_s,
            entries=entries, exits=exits,
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_map=body_map, clearance_map=clearance_map,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl, y_bl=y_bl, rotation=vi.rotation,
            w=w, h=h,
            mirror=vi.mirror,
        )

    def build_placement(
        self,
        *,
        x_c: float,
        y_c: float,
        rotation: int = 0,
        mirror: bool = False,
    ) -> StaticIrregularPlacement:
        r = self._resolve_rotation(rotation)
        vi = self._variant_for_pose(r, mirror=mirror)
        w = float(vi.body_w)
        h = float(vi.body_h)
        x_bl = int(round(x_c - w / 2.0))
        y_bl = int(round(y_c - h / 2.0))
        x_c_s = float(x_bl) + w / 2.0
        y_c_s = float(y_bl) + h / 2.0
        body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
        return StaticIrregularPlacement(
            x_c=x_c_s, y_c=y_c_s,
            entries=list(self._entries_from_center(x_c_s, y_c_s, r, mirror=mirror)),
            exits=list(self._exits_from_center(x_c_s, y_c_s, r, mirror=mirror)),
            min_x=float(x_bl), max_x=float(x_bl) + w,
            min_y=float(y_bl), max_y=float(y_bl) + h,
            body_map=body_map, clearance_map=clearance_map,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl, y_bl=y_bl, rotation=int(r),
            w=w, h=h,
            mirror=bool(mirror),
        )

    def _variant_for_pose(self, rotation: int, *, mirror: bool) -> VariantInfo:
        rr = self._resolve_rotation(rotation)
        for vi in self._variants:
            if vi.rotation == rr and bool(vi.mirror) == bool(mirror):
                return vi
        raise ValueError(
            f"StaticIrregularSpec {self.id!r} does not have a variant for rotation={rr}, mirror={bool(mirror)}"
        )

    # ----- irregular-specific helpers -----

    def _validate_body_bbox(self) -> None:
        bm = self.body_map_canonical
        if not bool(bm[0, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the bottom boundary")
        if not bool(bm[-1, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the top boundary")
        if not bool(bm[:, 0].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the left boundary")
        if not bool(bm[:, -1].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the right boundary")

    def _validate_ports(self) -> None:
        invalid_ports: List[str] = []
        bounds = f"[0, {self.width}] x [0, {self.height}]"
        for port_type, ports in (("entry", self.entries_rel), ("exit", self.exits_rel)):
            for idx, port in enumerate(ports):
                x, y = float(port[0]), float(port[1])
                if x < 0.0 or x > float(self.width) or y < 0.0 or y > float(self.height):
                    invalid_ports.append(f"{port_type}[{idx}]=({x}, {y})")
        if invalid_ports:
            logger.warning(
                f"StaticIrregularSpec {self.id!r} has ports outside local bounds {bounds}: "
                + ", ".join(invalid_ports)
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
        rotations = (0, 90, 180, 270) if self.rotatable else (0,)
        mirrors = (False, True) if self.mirrorable else (False,)

        seen_cost: set = set()
        variants: List[VariantInfo] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_variant_count = 0
        bH_canon, bW_canon = self.height, self.width
        cH_canon = int(self.clearance_map_canonical.shape[0])
        cW_canon = int(self.clearance_map_canonical.shape[1])

        for rot in rotations:
            rr = self._resolve_rotation(rot)
            for m in mirrors:
                raw_variant_count += 1
                body_map = self._transform_body_map(self.body_map_canonical, rr, mirror=m)
                clearance_map = self._transform_body_map(self.clearance_map_canonical, rr, mirror=m)
                body_h = int(body_map.shape[0])
                body_w = int(body_map.shape[1])
                clearance_origin = self._transform_clearance_origin(
                    self.clearance_origin_canonical,
                    (bH_canon, bW_canon), (cH_canon, cW_canon),
                    rr, m,
                )
                cL, cB = clearance_origin
                cR = int(clearance_map.shape[1]) - body_w - cL
                cT = int(clearance_map.shape[0]) - body_h - cB
                is_rectangular = bool(body_map.all().item()) and bool(clearance_map.all().item())
                eo_t, xo_t = self._compute_port_offsets(rr, m)

                pk = self._make_shape_key(
                    body_map=body_map, clearance_map=clearance_map,
                    clearance_origin=clearance_origin, is_rectangular=is_rectangular,
                )
                ck = (pk, eo_t, xo_t)
                if ck in seen_cost:
                    continue
                seen_cost.add(ck)

                vi = VariantInfo(
                    rotation=rr, mirror=bool(m),
                    body_w=body_w, body_h=body_h,
                    cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT),
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t, exit_offsets=xo_t,
                    shape_key=pk, cost_key=ck,
                )
                variants.append(vi)
                shape_tensors_by_key[pk] = (body_map, clearance_map, clearance_origin, is_rectangular)

        self._store_variants(variants, shape_tensors_by_key)

        logger.info(
            "StaticIrregularSpec %r variant summary: raw=%d, unique_variants=%d, unique_shapes=%d",
            self.id, raw_variant_count,
            len(self._variants), len(self._variants_by_shape),
        )

    def _coerce_body_map(self, body_map: Any) -> torch.Tensor:
        t = torch.as_tensor(body_map, dtype=torch.bool, device=self.device)
        if t.ndim != 2:
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must be 2D, got ndim={t.ndim}")
        if t.numel() == 0 or int(t.shape[0]) <= 0 or int(t.shape[1]) <= 0:
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must be non-empty")
        if not bool(t.any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must contain at least one occupied cell")
        return t.contiguous()

    def _transform_body_map(self, body_map: torch.Tensor, rotation: int, *, mirror: bool = False) -> torch.Tensor:
        src = body_map.to(device=self.device, dtype=torch.bool).contiguous()
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
        clearance_left_rel=1,
        clearance_right_rel=1,
        clearance_bottom_rel=0,
        clearance_top_rel=0,
    )
    print(f"variants={len(geom._variants)}, unique_shapes={len(geom._variants_by_shape)}")
    for vi in geom._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"clear=({vi.cL},{vi.cR},{vi.cB},{vi.cT}) "
              f"entries={len(vi.entry_offsets)} exits={len(vi.exit_offsets)}")

    scalar = geom.build_placement(x_c=14.0, y_c=7.0, rotation=90)
    print(
        f"\nbuild_placement(x_c=14.0,y_c=7.0,rotation=90) -> "
        f"w={scalar.w}, h={scalar.h}, entries={len(scalar.entries)}, exits={len(scalar.exits)}"
    )

    # --- Irregular demo ---
    body = torch.zeros((4, 6), dtype=torch.bool, device=device)
    body[0:2, :] = True
    body[2:4, 0:2] = True
    # Build clearance_map with 1-cell padding around each occupied cell
    bH, bW = 4, 6
    cL, cR, cB, cT = 1, 1, 1, 1
    clear_h, clear_w = bH + cB + cT, bW + cL + cR
    clearance = torch.zeros((clear_h, clear_w), dtype=torch.bool, device=device)
    ys, xs = torch.nonzero(body, as_tuple=True)
    for x, y in zip(xs.tolist(), ys.tolist()):
        clearance[y : y + cB + cT + 1, x : x + cL + cR + 1] = True

    irr = StaticIrregularSpec(
        id="L_demo",
        device=device,
        width=6,
        height=4,
        entries_rel=[(0.0, 1.0)],
        exits_rel=[(6.0, 1.0)],
        body_map_canonical=body,
        clearance_map_canonical=clearance,
        clearance_origin_canonical=(cL, cB),
    )
    print(f"\nirregular variants={len(irr._variants)}, unique_shapes={len(irr._variants_by_shape)}")
    for vi in irr._variants:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"rect={vi.is_rectangular} origin={vi.clearance_origin}")
    ip = irr.build_placement(x_c=20.0, y_c=10.0, rotation=0)
    print(f"  build_placement -> w={ip.w}, h={ip.h}, body_map={tuple(ip.body_map.shape)}")
    print("OK")
