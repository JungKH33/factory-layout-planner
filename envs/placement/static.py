from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .base import GroupSpec, GroupPlacement, Orientation

if TYPE_CHECKING:
    from ..state.base import EnvState
    from ..reward.core import RewardComposer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticOrientation(Orientation):
    """Static-spec orientation: extends Orientation with clearance and cache keys."""

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


@dataclass
class StaticIrregularPlacement(GroupPlacement):
    """Resolved absolute placement for an irregular static group."""

    x_bl: int
    y_bl: int
    rotation: int
    w: float
    h: float
    mirror: bool = False
    body_polygon_abs: Optional[List[Tuple[float, float]]] = None
    clearance_polygon_abs: Optional[List[Tuple[float, float]]] = None


# ---------------------------------------------------------------------------
# StaticSpec — shared base for StaticRectSpec / StaticIrregularSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticSpec(GroupSpec):
    """Common spec for static facilities (rect or irregular).

    Provides:
    - Orientation storage and lookup
    - Rotation / mirror / clearance helpers
    - Port coordinate transforms (single + batch)
    - placeable_map / placeable_batch / cost_batch
    - resolve() — tries every orientation and picks the best
    - shape_tensors() — returns (body_map, clearance_map, origin, is_rect) by key

    Subclasses must implement ``__post_init__`` which is responsible for
    calling ``_store_orientations(orientations, shape_tensors_by_key)`` to populate
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
    clearance_lrtb_rel: Optional[Tuple[int, int, int, int]] = None

    # ----- orientation storage (populated by _store_orientations) -----

    def _store_orientations(
        self,
        orientations: List[StaticOrientation],
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]],
    ) -> None:
        self._orientations: List[StaticOrientation] = orientations
        self._orientations_by_shape: Dict[tuple, List[StaticOrientation]] = {}
        for vi in orientations:
            self._orientations_by_shape.setdefault(vi.shape_key, []).append(vi)
        self._shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = shape_tensors_by_key
        self._orient_entry_off_t: List[torch.Tensor] = []
        self._orient_exit_off_t: List[torch.Tensor] = []
        for vi in orientations:
            if vi.entry_offsets:
                et = torch.tensor(list(vi.entry_offsets), dtype=torch.float32, device=self.device)
            else:
                et = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            if vi.exit_offsets:
                xt = torch.tensor(list(vi.exit_offsets), dtype=torch.float32, device=self.device)
            else:
                xt = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self._orient_entry_off_t.append(et)
            self._orient_exit_off_t.append(xt)

        V = len(orientations)
        P_ent = max((int(t.shape[0]) for t in self._orient_entry_off_t), default=0)
        P_ext = max((int(t.shape[0]) for t in self._orient_exit_off_t), default=0)
        self._all_entry_offsets = torch.zeros((V, P_ent, 2), dtype=torch.float32, device=self.device)
        self._all_entry_mask = torch.zeros((V, P_ent), dtype=torch.bool, device=self.device)
        self._all_exit_offsets = torch.zeros((V, P_ext, 2), dtype=torch.float32, device=self.device)
        self._all_exit_mask = torch.zeros((V, P_ext), dtype=torch.bool, device=self.device)
        for i, (et, xt) in enumerate(zip(self._orient_entry_off_t, self._orient_exit_off_t)):
            p_e = int(et.shape[0])
            if p_e > 0:
                self._all_entry_offsets[i, :p_e] = et
                self._all_entry_mask[i, :p_e] = True
            p_x = int(xt.shape[0])
            if p_x > 0:
                self._all_exit_offsets[i, :p_x] = xt
                self._all_exit_mask[i, :p_x] = True
        self._all_body_w = torch.tensor(
            [float(vi.body_w) for vi in orientations],
            dtype=torch.float32, device=self.device,
        )
        self._all_body_h = torch.tensor(
            [float(vi.body_h) for vi in orientations],
            dtype=torch.float32, device=self.device,
        )

    @property
    def orientations(self) -> List[Orientation]:
        """All unique (rotation, mirror) orientations for this spec."""
        return list(self._orientations)

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

    def _clearance_lrtb_for_orientation(self, rotation: int, *, mirror: bool = False) -> Tuple[int, int, int, int]:
        """Return (L, R, B, T) clearance for a given rotation/mirror of the canonical LRTB."""
        r = self._resolve_rotation(rotation)
        if self.clearance_lrtb_rel is not None:
            cL, cR, cB, cT = (int(self.clearance_lrtb_rel[0]), int(self.clearance_lrtb_rel[1]),
                               int(self.clearance_lrtb_rel[2]), int(self.clearance_lrtb_rel[3]))
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

    @staticmethod
    def _dilate_body_map(
        body_map: torch.Tensor,
        L: int, R: int, B: int, T: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Dilate body_map by (L, R, B, T) cells.

        Returns ``(clearance_map, clearance_origin)`` where
        *clearance_origin* = ``(L, B)`` is the offset of the body BL
        inside the clearance_map.
        """
        if L == 0 and R == 0 and B == 0 and T == 0:
            return body_map.clone(), (0, 0)
        bH, bW = int(body_map.shape[0]), int(body_map.shape[1])
        cH = bH + B + T
        cW = bW + L + R
        clearance = torch.zeros((cH, cW), dtype=torch.bool, device=body_map.device)
        ys, xs = torch.nonzero(body_map, as_tuple=True)
        for yv, xv in zip(ys.tolist(), xs.tolist()):
            r0 = int(yv)
            r1 = int(yv) + B + T + 1
            c0 = int(xv)
            c1 = int(xv) + L + R + 1
            clearance[r0:r1, c0:c1] = True
        return clearance, (L, B)

    def _resolve_clearance(self, body_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Compute ``(clearance_map, clearance_origin)`` from clearance fields."""
        if self.clearance_lrtb_rel is not None:
            L, R, B, T = (int(self.clearance_lrtb_rel[0]), int(self.clearance_lrtb_rel[1]),
                          int(self.clearance_lrtb_rel[2]), int(self.clearance_lrtb_rel[3]))
            return self._dilate_body_map(body_map, L, R, B, T)
        return body_map.clone(), (0, 0)

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
        for i, t in enumerate(self._orient_entry_off_t):
            self._orient_entry_off_t[i] = t.to(device)
        for i, t in enumerate(self._orient_exit_off_t):
            self._orient_exit_off_t[i] = t.to(device)
        self._all_entry_offsets = self._all_entry_offsets.to(device)
        self._all_entry_mask = self._all_entry_mask.to(device)
        self._all_exit_offsets = self._all_exit_offsets.to(device)
        self._all_exit_mask = self._all_exit_mask.to(device)
        self._all_body_w = self._all_body_w.to(device)
        self._all_body_h = self._all_body_h.to(device)
        moved: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        for key, (body_map, clearance_map, clearance_origin, is_rectangular) in self._shape_tensors_by_key.items():
            moved[key] = (
                body_map.to(device=device, dtype=torch.bool),
                clearance_map.to(device=device, dtype=torch.bool),
                clearance_origin,
                bool(is_rectangular),
            )
        self._shape_tensors_by_key = moved

    # ----- orientation-level port offset computation -----

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

    # ----- resolve (orientation selection) -----

    def _make_placement(
        self,
        vi: StaticOrientation,
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
        orientation_index: Optional[int] = None,
    ) -> 'GroupPlacement | None':
        if orientation_index is not None:
            orientations_to_try = [self._orientations[orientation_index]]
        else:
            orientations_to_try = self._orientations

        placeable: List[GroupPlacement] = []
        for vi in orientations_to_try:
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
        for shape_key in self._orientations_by_shape:
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

    def placeable_center_map(self, state: "EnvState", gid: object) -> torch.Tensor:
        """Center-based validity map (OR of all orientation shapes shifted to center coords)."""
        H, W = state.maps.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        for shape_key in self._orientations_by_shape:
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(shape_key)
            bl_map = state.is_placeable_map(
                gid=gid,
                body_map=body_map,
                clearance_map=clearance_map,
                clearance_origin=clearance_origin,
                is_rectangular=is_rectangular,
            )
            bh, bw = int(body_map.shape[0]), int(body_map.shape[1])
            dx = bw // 2
            dy = bh // 2
            src_h = min(H - dy, int(bl_map.shape[0]))
            src_w = min(W - dx, int(bl_map.shape[1]))
            if src_h > 0 and src_w > 0:
                result[dy:dy + src_h, dx:dx + src_w] |= bl_map[:src_h, :src_w]
        return result

    def placeable_batch(
        self,
        state: "EnvState",
        gid: object,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        per_orientation: bool = False,
    ) -> torch.Tensor:
        """Check placeability for batch of center positions.

        Returns:
            per_orientation=False (default): ``[N]`` bool — True if ANY
                orientation is placeable.
            per_orientation=True: ``[N, V]`` bool — per-orientation result
                (V = len(self._orientations)).
        """
        N = int(x_c.shape[0])
        V = len(self._orientations)
        if per_orientation:
            result = torch.zeros((N, V), dtype=torch.bool, device=self.device)
            for i, vi in enumerate(self._orientations):
                body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
                body_h, body_w = int(body_map.shape[0]), int(body_map.shape[1])
                x_bl = torch.round(x_c - body_w / 2.0).to(torch.long)
                y_bl = torch.round(y_c - body_h / 2.0).to(torch.long)
                result[:, i] = state.is_placeable_batch(
                    gid=gid,
                    x_bl=x_bl, y_bl=y_bl,
                    body_map=body_map, clearance_map=clearance_map,
                    clearance_origin=clearance_origin, is_rectangular=is_rectangular,
                )
            return result
        # Default: OR across all shapes (original behaviour)
        ok = torch.zeros(N, dtype=torch.bool, device=self.device)
        for shape_key in self._orientations_by_shape:
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
        per_orientation: bool = False,
    ) -> torch.Tensor:
        """Vectorized incremental cost for batch of center positions.

        Two-phase approach:
          Phase 1 — per-orientation placeability loop (shape-dependent, unavoidable).
          Phase 2 — flatten all valid (position, orientation) pairs, build features
                    via gather on padded offset tables, single ``delta_batch`` call.

        Returns:
            per_orientation=False (default): ``[N]`` float — min cost across
                all orientations (inf where nothing is placeable).
            per_orientation=True: ``[N, V]`` float — per-orientation cost
                (V = len(self._orientations), inf where not placeable).
        """
        N = int(poses.shape[0])
        V = len(self._orientations)
        if N == 0:
            if per_orientation:
                return torch.zeros((0, V), dtype=torch.float32, device=self.device)
            return torch.zeros((0,), dtype=torch.float32, device=self.device)

        x_c = poses[:, 0]
        y_c = poses[:, 1]

        # Phase 1: placeability check per orientation
        ok = torch.zeros((N, V), dtype=torch.bool, device=self.device)
        for i, vi in enumerate(self._orientations):
            body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
            x_bl = torch.round(x_c - vi.body_w / 2.0).to(torch.long)
            y_bl = torch.round(y_c - vi.body_h / 2.0).to(torch.long)
            ok[:, i] = state.is_placeable_batch(
                gid=gid,
                x_bl=x_bl, y_bl=y_bl,
                body_map=body_map, clearance_map=clearance_map,
                clearance_origin=clearance_origin, is_rectangular=is_rectangular,
            )

        result = torch.full((N, V), float('inf'), dtype=torch.float32, device=self.device)
        valid_n, valid_v = torch.where(ok)
        M = int(valid_n.shape[0])

        if M > 0:
            # Phase 2: vectorized feature build + single delta_batch
            needed = reward.required()
            bw = self._all_body_w[valid_v]
            bh = self._all_body_h[valid_v]
            cx = poses[valid_n, 0]
            cy = poses[valid_n, 1]
            x_bl_f = torch.round(cx - bw / 2.0)
            y_bl_f = torch.round(cy - bh / 2.0)
            x_c_s = x_bl_f + bw / 2.0
            y_c_s = y_bl_f + bh / 2.0

            features: Dict[str, torch.Tensor] = {}
            if "min_x" in needed:
                features["min_x"] = x_bl_f
            if "max_x" in needed:
                features["max_x"] = x_bl_f + bw
            if "min_y" in needed:
                features["min_y"] = y_bl_f
            if "max_y" in needed:
                features["max_y"] = y_bl_f + bh
            if "entries" in needed or "exits" in needed:
                c = torch.stack([x_c_s, y_c_s], dim=1)
            if "entries" in needed:
                if self._all_entry_offsets.shape[1] > 0:
                    features["entries"] = c[:, None, :] + self._all_entry_offsets[valid_v]
                    features["entries_mask"] = self._all_entry_mask[valid_v]
                else:
                    features["entries"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                    features["entries_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)
            if "exits" in needed:
                if self._all_exit_offsets.shape[1] > 0:
                    features["exits"] = c[:, None, :] + self._all_exit_offsets[valid_v]
                    features["exits_mask"] = self._all_exit_mask[valid_v]
                else:
                    features["exits"] = torch.empty((M, 0, 2), dtype=torch.float32, device=self.device)
                    features["exits_mask"] = torch.empty((M, 0), dtype=torch.bool, device=self.device)

            scores = reward.delta_batch(state, gid=gid, **features)
            result[valid_n, valid_v] = scores

        if per_orientation:
            return result
        return result.min(dim=1).values


# ---------------------------------------------------------------------------
# StaticRectSpec
# ---------------------------------------------------------------------------

@dataclass
class StaticRectSpec(StaticSpec):
    """Static rectangular facility spec."""

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
        orients: List[StaticOrientation] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_orientation_count = 0

        for rot in rotations:
            rr = self._resolve_rotation(rot)
            for m in mirrors:
                raw_orientation_count += 1
                w, h = self._rotated_size(rr)
                body_w, body_h = int(w), int(h)
                cL, cR, cB, cT = self._clearance_lrtb_for_orientation(rr, mirror=m)
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

                vi = StaticOrientation(
                    rotation=rr, mirror=m,
                    body_w=body_w, body_h=body_h,
                    cL=cL, cR=cR, cB=cB, cT=cT,
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t, exit_offsets=xo_t,
                    shape_key=pk, cost_key=ck,
                )
                orients.append(vi)
                shape_tensors_by_key[pk] = (body_map, clearance_map, clearance_origin, is_rectangular)

        self._store_orientations(orients, shape_tensors_by_key)

        logger.info(
            "StaticRectSpec %r orientation summary: raw=%d, unique=%d, unique_shapes=%d",
            self.id, raw_orientation_count,
            len(self._orientations), len(self._orientations_by_shape),
        )

    @property
    def body_area(self) -> float:
        return float(self.width) * float(self.height)

    def _make_placement(
        self,
        vi: StaticOrientation,
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
        _valid_modes = ("min", "mean")
        if self._entry_port_mode not in _valid_modes:
            raise ValueError(f"entry_port_mode must be one of {_valid_modes}, got {self._entry_port_mode!r}")
        if self._exit_port_mode not in _valid_modes:
            raise ValueError(f"exit_port_mode must be one of {_valid_modes}, got {self._exit_port_mode!r}")

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
        self._build_orientations()

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
        inside the clearance_map.
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
        clearance_map = torch.as_tensor(mask, dtype=torch.bool, device=device).contiguous()

        ox = int(round(bx_min - cx_min))
        oy = int(round(by_min - cy_min))
        return clearance_map, (ox, oy)

    @staticmethod
    def _transform_polygon_to_world(
        polygon: List[Tuple[float, float]],
        x_bl: float,
        y_bl: float,
        rotation: int,
        mirror: bool,
        body_w: int,
        body_h: int,
    ) -> List[Tuple[float, float]]:
        """Transform canonical polygon vertices to world coordinates."""
        half_w = body_w / 2.0
        half_h = body_h / 2.0
        r = rotation % 360
        if r in (90, 270):
            rw, rh = float(body_h), float(body_w)
        else:
            rw, rh = float(body_w), float(body_h)
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
        vi: StaticOrientation,
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
        bp_abs = (self._transform_polygon_to_world(
            self.body_polygon, x_bl, y_bl, vi.rotation, vi.mirror,
            self.width, self.height) if self.body_polygon else None)
        cp_abs = (self._transform_polygon_to_world(
            self.clearance_polygon, x_bl, y_bl, vi.rotation, vi.mirror,
            self.width, self.height) if self.clearance_polygon else None)
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
            body_polygon_abs=bp_abs,
            clearance_polygon_abs=cp_abs,
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

    def _build_orientations(self) -> None:
        rotations = (0, 90, 180, 270) if self.rotatable else (0,)
        mirrors = (False, True) if self.mirrorable else (False,)

        seen_cost: set = set()
        orients: List[StaticOrientation] = []
        shape_tensors_by_key: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]] = {}
        raw_orientation_count = 0
        bH_canon, bW_canon = self.height, self.width
        cH_canon = int(self.clearance_map_canonical.shape[0])
        cW_canon = int(self.clearance_map_canonical.shape[1])

        for rot in rotations:
            rr = self._resolve_rotation(rot)
            for m in mirrors:
                raw_orientation_count += 1
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

                vi = StaticOrientation(
                    rotation=rr, mirror=bool(m),
                    body_w=body_w, body_h=body_h,
                    cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT),
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t, exit_offsets=xo_t,
                    shape_key=pk, cost_key=ck,
                )
                orients.append(vi)
                shape_tensors_by_key[pk] = (body_map, clearance_map, clearance_origin, is_rectangular)

        self._store_orientations(orients, shape_tensors_by_key)

        logger.info(
            "StaticIrregularSpec %r orientation summary: raw=%d, unique=%d, unique_shapes=%d",
            self.id, raw_orientation_count,
            len(self._orientations), len(self._orientations_by_shape),
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
        clearance_lrtb_rel=(1, 1, 0, 0),
    )
    print(f"orientations={len(geom._orientations)}, unique_shapes={len(geom._orientations_by_shape)}")
    for vi in geom._orientations:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"clear=({vi.cL},{vi.cR},{vi.cB},{vi.cT}) "
              f"entries={len(vi.entry_offsets)} exits={len(vi.exit_offsets)}")

    # resolve with orientation_index=1 (90° rotation)
    def _always_true(*args):
        return True
    resolved = geom.resolve(x_c=14.0, y_c=7.0, is_placeable_fn=_always_true, orientation_index=1)
    print(
        f"\nresolve(x_c=14.0,y_c=7.0,orientation_index=1) -> "
        f"w={resolved.w}, h={resolved.h}, entries={len(resolved.entries)}, exits={len(resolved.exits)}"
    )

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
    print(f"\nirregular orientations={len(irr._orientations)}, unique_shapes={len(irr._orientations_by_shape)}")
    for vi in irr._orientations:
        print(f"  rot={vi.rotation} mirror={vi.mirror} body=({vi.body_w},{vi.body_h}) "
              f"rect={vi.is_rectangular} origin={vi.clearance_origin}")
    ip = irr.resolve(x_c=20.0, y_c=10.0, is_placeable_fn=_always_true, orientation_index=0)
    print(f"  resolve -> w={ip.w}, h={ip.h}, body_map={tuple(ip.body_map.shape)}")
    print(f"  body_polygon_abs={ip.body_polygon_abs}")
    print("OK")
