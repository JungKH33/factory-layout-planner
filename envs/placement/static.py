from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .base import PlacementBase

logger = logging.getLogger(__name__)

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

    def variant_geometries(self, rotation: int) -> List[Tuple[int, int, int, int, int, int]]:
        """Return unique (body_w, body_h, cL, cR, cB, cT) for the given rotation across mirrors.

        Public API for env.placeable_map.
        """
        r = self._resolve_rotation(rotation)
        mirrors = (False, True) if self.mirrorable else (False,)
        seen: set = set()
        result: List[Tuple[int, int, int, int, int, int]] = []
        for m in mirrors:
            w, h = self._rotated_size(r)
            cL, cR, cB, cT = self._clearance_lrtb(r, mirror=m)
            geom = (int(w), int(h), int(cL), int(cR), int(cB), int(cT))
            if geom not in seen:
                seen.add(geom)
                result.append(geom)
        return result

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
        return StaticPlacement(
            entries=entries,
            exits=exits,
            min_x=float(x_bl),
            max_x=float(x_bl) + w,
            min_y=float(y_bl),
            max_y=float(y_bl) + h,
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
          is_placeable_fn(x_bl, y_bl, body_w, body_h, cL, cR, cB, cT) -> bool
          score_fn(placements: List[StaticPlacement]) -> Tensor[N]
        If *score_fn* is None, picks the first feasible variant.
        """
        rotations = (0, 90, 180, 270) if self.rotatable else (0,)
        mirrors = (False, True) if self.mirrorable else (False,)

        seen_sig: set = set()
        placeable: List[StaticPlacement] = []
        for rot in rotations:
            for m in mirrors:
                rr = self._resolve_rotation(rot)
                w, h = self._rotated_size(rr)
                x_bl = int(round(x_c - w / 2.0))
                y_bl = int(round(y_c - h / 2.0))
                x_c_s = float(x_bl) + w / 2.0
                y_c_s = float(y_bl) + h / 2.0
                cL, cR, cB, cT = self._clearance_lrtb(rr, mirror=m)

                entries = list(self._entries_from_center(x_c_s, y_c_s, rr, mirror=m))
                exits = list(self._exits_from_center(x_c_s, y_c_s, rr, mirror=m))
                sig = (
                    (round(w, 6), round(h, 6)),
                    (cL, cR, cB, cT),
                    tuple((round(ex, 6), round(ey, 6)) for ex, ey in entries),
                    tuple((round(ex, 6), round(ey, 6)) for ex, ey in exits),
                )
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)

                if not is_placeable_fn(x_bl, y_bl, int(w), int(h), cL, cR, cB, cT):
                    continue

                placeable.append(StaticPlacement(
                    entries=entries, exits=exits,
                    min_x=float(x_bl), max_x=float(x_bl) + w,
                    min_y=float(y_bl), max_y=float(y_bl) + h,
                    x_bl=x_bl, y_bl=y_bl, rotation=int(rr),
                    w=float(w), h=float(h), x_c=x_c_s, y_c=y_c_s,
                    clearance_left=cL, clearance_right=cR,
                    clearance_bottom=cB, clearance_top=cT,
                    mirror=bool(m),
                ))

        if not placeable:
            return None
        if score_fn is None or len(placeable) == 1:
            return placeable[0]

        scores = score_fn(placeable).to(dtype=torch.float32, device=self.device).view(-1)
        return placeable[int(torch.argmin(scores).item())]

    def build_candidate_features(
        self,
        *,
        x_c: object,
        y_c: object,
        rotation: object,
        mirror: bool = False,
        needed: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Build candidate feature tensors for center-coordinate pose vectors.

        ``x_c``/``y_c`` are center coordinates (float); ``rotation`` is
        concrete 0/90/180/270 (int or tensor).
        When ``mirror=True``, port dx offsets are flipped before rotation.
        """
        supported = {
            "entries",
            "exits",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
        }
        unknown = set(needed) - supported
        if unknown:
            raise ValueError(f"build_candidate_features: unknown feature keys: {sorted(unknown)}")

        scalar_input = not (torch.is_tensor(x_c) or torch.is_tensor(y_c))
        if scalar_input:
            x_c = torch.tensor([float(x_c)], dtype=torch.float32, device=self.device)
            y_c = torch.tensor([float(y_c)], dtype=torch.float32, device=self.device)

        x_c = x_c.to(device=self.device, dtype=torch.float32).view(-1)
        y_c = y_c.to(device=self.device, dtype=torch.float32).view(-1)
        M = int(x_c.numel())
        if int(y_c.numel()) != M:
            raise ValueError("build_candidate_features: x_c,y_c must have same length")

        if not torch.is_tensor(rotation):
            rotation = torch.full((M,), int(rotation), dtype=torch.long, device=self.device)
        rot_idx = self._rotation_idx(rotation.to(device=self.device, dtype=torch.long).view(-1))

        out: Dict[str, torch.Tensor] = {}

        # Snap center → BL (integer) → snapped center, matching actual grid placement.
        w, h = self._wh_for_rotation(rot_idx)
        x_bl = torch.round(x_c - w * 0.5).to(torch.long)
        y_bl = torch.round(y_c - h * 0.5).to(torch.long)
        x_c_s = x_bl.to(torch.float32) + w * 0.5
        y_c_s = y_bl.to(torch.float32) + h * 0.5

        if {"min_x", "max_x", "min_y", "max_y"} & needed:
            min_x = x_bl.to(dtype=torch.float32)
            min_y = y_bl.to(dtype=torch.float32)
            if "min_x" in needed:
                out["min_x"] = min_x
            if "max_x" in needed:
                out["max_x"] = min_x + w
            if "min_y" in needed:
                out["min_y"] = min_y
            if "max_y" in needed:
                out["max_y"] = min_y + h

        if {"entries", "exits"} & needed:
            center = torch.stack([x_c_s, y_c_s], dim=1)  # [M, 2]
            if "entries" in needed:
                out["entries"] = self._entries_batch(center, rot_idx, mirror=mirror)
            if "exits" in needed:
                out["exits"] = self._exits_batch(center, rot_idx, mirror=mirror)

        return out


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
        entries_rel=[(0.0, 2.0), (8.0, 2.0)],  # BL-relative: 왼쪽 끝 / 오른쪽 끝 (height/2=2)
        exits_rel=[(4.0, 4.0)],                # BL-relative: 윗면 중간 (width/2=4, height=4)
        clearance_left_rel=1,
        clearance_right_rel=1,
        clearance_bottom_rel=0,
        clearance_top_rel=0,
    )

    invalid = (torch.rand((H, W), device=device) < 0.05)
    clear_invalid = (torch.rand((H, W), device=device) < 0.02)

    M = 20000
    x = torch.randint(0, W, (M,), device=device)
    y = torch.randint(0, H, (M,), device=device)

    # build_placement vs build_candidate_features
    scalar = geom.build_placement(x_c=14.0, y_c=7.0, rotation=90)
    print(
        "build_placement(x_c=14.0,y_c=7.0,rotation=90) -> "
        f"w={scalar.w}, h={scalar.h}, entries={len(scalar.entries)}, exits={len(scalar.exits)}"
    )

    rot_vals = torch.randint(0, 4, (M,), device=device) * 90  # 0/90/180/270

    t0 = time.perf_counter()
    for i in range(M):
        _ = geom.build_placement(x_c=float(x[i].item()) + 4.0, y_c=float(y[i].item()) + 2.0, rotation=int(rot_vals[i].item()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    needed = {"entries", "exits", "min_x"}
    t2 = time.perf_counter()
    x_c_batch = x.to(torch.float32) + 4.0
    y_c_batch = y.to(torch.float32) + 2.0
    batch = geom.build_candidate_features(x_c=x_c_batch, y_c=y_c_batch, rotation=rot_vals, needed=needed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    scalar_ms = (t1 - t0) * 1000.0
    batch_ms = (t3 - t2) * 1000.0
    print(
        f"build_candidate_features(M={M}) -> "
        f"entries={tuple(batch['entries'].shape)}, "
        f"exits={tuple(batch['exits'].shape)}, "
        f"min_x={tuple(batch['min_x'].shape)}"
    )
    print(f"build_placement loop: {scalar_ms:.2f} ms total, {scalar_ms / M:.4f} ms/elem")
    print(f"build_candidate_features: {batch_ms:.2f} ms total, {batch_ms / M:.6f} ms/elem")
