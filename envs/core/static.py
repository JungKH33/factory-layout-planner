from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import PlacementBase

RectI = Tuple[int, int, int, int]


@dataclass
class StaticPlacement(PlacementBase):
    """Resolved absolute placement for a placed static group (world/placed orientation).

    Inherits the common placement contract from PlacementBase and adds
    geometry fields (w, h, cx, cy) that are specific to static (fixed-size) facilities.
    """
    # TODO(modularization): move single-port selection (including empty -> center fallback)
    # from env into placement-level API so env does not handle entries/exits shape branching.
    w: float
    h: float
    cx: float
    cy: float


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
    allowed_areas: Optional[List[str]] = None

    # Map-based zone requirements (used by env zone-invalid logic).
    facility_height: float = float("-inf")
    facility_weight: float = float("-inf")
    facility_dry: float = float("inf")

    @staticmethod
    def _norm_rot(rot: int) -> int:
        """Normalize rotation to {0,90,180,270} and validate."""
        r = int(rot) % 360
        if r % 90 != 0:
            raise ValueError(f"rot must be a multiple of 90 degrees, got {rot!r}")
        return r

    def _resolve_rot(self, rot: int) -> int:
        r = self._norm_rot(rot)
        if not bool(self.rotatable):
            return 0
        return r

    def _rotated_size(self, rot: int) -> Tuple[float, float]:
        """Return rotated (w,h) for 90-degree-multiple rotations."""
        r = self._resolve_rot(rot)
        if r in (90, 270):
            return (float(self.height), float(self.width))
        return (float(self.width), float(self.height))

    @staticmethod
    def _rotate_point(dx: float, dy: float, rot: int) -> Tuple[float, float]:
        """Rotate a local point (dx,dy) CCW by multiples of 90 degrees."""
        r = StaticSpec._norm_rot(rot)
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

    def _rot_idx(self, rot: torch.Tensor) -> torch.Tensor:
        r = torch.remainder(rot, 360)
        if torch.any((r % 90) != 0):
            raise ValueError("rot must be multiples of 90")
        if not bool(self.rotatable):
            return torch.zeros_like(r, dtype=torch.long)
        return (r // 90).to(dtype=torch.long)

    def _wh_for_rot(self, rot_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def _offsets_rot(base: torch.Tensor) -> torch.Tensor:
        dx = base[:, 0]
        dy = base[:, 1]
        r0 = torch.stack([dx, dy], dim=1)
        r90 = torch.stack([dy, -dx], dim=1)
        r180 = torch.stack([-dx, -dy], dim=1)
        r270 = torch.stack([-dy, dx], dim=1)
        return torch.stack([r0, r90, r180, r270], dim=0)  # [4,N,2]

    def _clearance_lrtb(self, rot: int) -> Tuple[int, int, int, int]:
        """Return clearance (cL, cR, cB, cT) rotated with the body."""
        r = self._resolve_rot(rot)
        cL, cR, cB, cT = (
            int(self.clearance_left_rel),
            int(self.clearance_right_rel),
            int(self.clearance_bottom_rel),
            int(self.clearance_top_rel),
        )
        if r == 90:
            return (cB, cT, cR, cL)
        if r == 180:
            return (cR, cL, cT, cB)
        if r == 270:
            return (cT, cB, cL, cR)
        return (cL, cR, cB, cT)

    def _center_from_bl(self, x_bl: float, y_bl: float, rot: int) -> Tuple[float, float]:
        w, h = self._rotated_size(rot)
        return (float(x_bl) + float(w) / 2.0, float(y_bl) + float(h) / 2.0)

    def _ports_from_bl(
        self,
        x_bl: float,
        y_bl: float,
        rot: int,
        ports_rel: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Convert BL-relative port offsets to absolute world coordinates.

        BL-relative → center-relative (subtract half_w, half_h) → rotate → add world center.
        """
        r = self._resolve_rot(rot)
        cx, cy = self._center_from_bl(x_bl, y_bl, r)
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        out = []
        for dx_bl, dy_bl in ports_rel:
            rdx, rdy = self._rotate_point(dx_bl - half_w, dy_bl - half_h, r)
            out.append((cx + rdx, cy + rdy))
        return out

    def _entries_from_bl(self, x_bl: float, y_bl: float, rot: int) -> List[Tuple[float, float]]:
        return self._ports_from_bl(x_bl, y_bl, rot, self.entries_rel)

    def _exits_from_bl(self, x_bl: float, y_bl: float, rot: int) -> List[Tuple[float, float]]:
        return self._ports_from_bl(x_bl, y_bl, rot, self.exits_rel)

    def _body_rect_bl(self, x_bl: int, y_bl: int, rot: int) -> RectI:
        w, h = self._rotated_size(rot)
        x0 = int(x_bl)
        y0 = int(y_bl)
        return (x0, y0, x0 + int(w), y0 + int(h))

    def _center_from_bl_batch(
        self,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        rot_idx: torch.Tensor,
    ) -> torch.Tensor:
        w, h = self._wh_for_rot(rot_idx)
        cx = x_bl.to(dtype=torch.float32) + w * 0.5
        cy = y_bl.to(dtype=torch.float32) + h * 0.5
        return torch.stack([cx, cy], dim=1)

    def _ports_from_bl_batch(
        self,
        ports_rel: List[Tuple[float, float]],
        center: torch.Tensor,   # [M, 2]
        rot_idx: torch.Tensor,  # [M]
    ) -> torch.Tensor:          # [M, N, 2]
        """Batch version of _ports_from_bl.

        BL-relative offsets → center-relative (subtract half_wh) → _offsets_rot → add world center.
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
        rot_offsets = self._offsets_rot(center_offsets)     # [4, N, 2]
        return center[:, None, :] + rot_offsets[rot_idx]    # [M, N, 2]

    def _entries_from_bl_batch(self, center: torch.Tensor, rot_idx: torch.Tensor) -> torch.Tensor:
        return self._ports_from_bl_batch(self.entries_rel, center, rot_idx)

    def _exits_from_bl_batch(self, center: torch.Tensor, rot_idx: torch.Tensor) -> torch.Tensor:
        return self._ports_from_bl_batch(self.exits_rel, center, rot_idx)

    @staticmethod
    def _pad_rect_i(rect: RectI, *, cL: int, cR: int, cB: int, cT: int) -> RectI:
        x0, y0, x1, y1 = rect
        return (x0 - int(cL), y0 - int(cB), x1 + int(cR), y1 + int(cT))

    @staticmethod
    def _recti_hits_invalid(rect: RectI, invalid: torch.Tensor) -> bool:
        x0, y0, x1, y1 = rect
        H, W = invalid.shape
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            return True
        if x0 >= x1 or y0 >= y1:
            return False
        return bool(torch.any(invalid[y0:y1, x0:x1]).item())

    def is_placeable(
        self,
        *,
        x_bl: int,
        y_bl: int,
        rot: int,
        invalid: torch.Tensor,
        clear_invalid: torch.Tensor,
    ) -> bool:
        r = self._resolve_rot(rot)
        rect = self._body_rect_bl(x_bl, y_bl, r)
        cL, cR, cB, cT = self._clearance_lrtb(r)
        rect_pad = self._pad_rect_i(rect, cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT))

        if self._recti_hits_invalid(rect, invalid):
            return False
        if self._recti_hits_invalid(rect, clear_invalid):
            return False
        if self._recti_hits_invalid(rect_pad, invalid):
            return False
        return True

    def is_placeable_mask(
        self,
        *,
        rot: int,
        invalid: torch.Tensor,
        clear_invalid: torch.Tensor,
    ) -> torch.Tensor:
        r = self._resolve_rot(rot)
        w, h = self._rotated_size(r)
        w, h = int(w), int(h)
        cL, cR, cB, cT = self._clearance_lrtb(r)
        kw = w + cL + cR
        kh = h + cB + cT

        H, W = invalid.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        if kh > H or kw > W or h <= 0 or w <= 0:
            return result

        inv_f = invalid.float().unsqueeze(0).unsqueeze(0)
        clear_f = clear_invalid.float().unsqueeze(0).unsqueeze(0)

        body_kernel = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
        pad_kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=torch.float32)

        body_inv = F.conv2d(inv_f, body_kernel, padding=0).squeeze()
        body_clear = F.conv2d(clear_f, body_kernel, padding=0).squeeze()
        pad_inv = F.conv2d(inv_f, pad_kernel, padding=0).squeeze()

        valid_h = H - kh + 1
        valid_w = W - kw + 1
        if valid_h <= 0 or valid_w <= 0:
            return result

        body_inv_slice = body_inv[cB : cB + valid_h, cL : cL + valid_w]
        body_clear_slice = body_clear[cB : cB + valid_h, cL : cL + valid_w]

        valid_mask = (body_inv_slice == 0) & (body_clear_slice == 0) & (pad_inv == 0)
        result[:valid_h, :valid_w] = valid_mask
        return result

    def is_placeable_batch(
        self,
        *,
        x: object,
        y: object,
        rot: object,
        invalid: torch.Tensor,
        clear_invalid: torch.Tensor,
    ) -> torch.Tensor:
        """Batch placeable check; scalar or tensor inputs supported."""
        scalar_input = not (torch.is_tensor(x) or torch.is_tensor(y) or torch.is_tensor(rot))
        if scalar_input:
            x = torch.tensor([int(x)], dtype=torch.long, device=self.device)
            y = torch.tensor([int(y)], dtype=torch.long, device=self.device)
            rot = torch.tensor([int(rot)], dtype=torch.long, device=self.device)

        x = x.to(device=self.device, dtype=torch.long).view(-1)
        y = y.to(device=self.device, dtype=torch.long).view(-1)
        rot = rot.to(device=self.device, dtype=torch.long).view(-1)

        H, W = invalid.shape

        mask_0 = self.is_placeable_mask(rot=0, invalid=invalid, clear_invalid=clear_invalid)
        mask_90 = self.is_placeable_mask(rot=90, invalid=invalid, clear_invalid=clear_invalid)

        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        rot_norm = torch.remainder(rot, 360)
        if torch.any((rot_norm % 90) != 0):
            raise ValueError("rot must be multiples of 90")
        if not bool(self.rotatable):
            rot_norm = torch.zeros_like(rot_norm)
        is_rot0 = (rot_norm == 0) | (rot_norm == 180)

        x_clamped = x.clamp(0, W - 1)
        y_clamped = y.clamp(0, H - 1)

        result_0 = mask_0[y_clamped, x_clamped]
        result_90 = mask_90[y_clamped, x_clamped]

        result = torch.where(is_rot0, result_0, result_90)
        result = result & in_bounds
        return result

    def build_placement(self, *, x_bl: int, y_bl: int, rot: int) -> StaticPlacement:
        """Build a placed-instance snapshot with absolute/rotated values."""
        r = self._resolve_rot(rot)
        w, h = self._rotated_size(r)
        cx, cy = self._center_from_bl(x_bl, y_bl, r)
        entries = list(self._entries_from_bl(x_bl, y_bl, r))
        exits = list(self._exits_from_bl(x_bl, y_bl, r))
        cL, cR, cB, cT = self._clearance_lrtb(r)
        min_x = float(x_bl)
        max_x = float(x_bl) + float(w)
        min_y = float(y_bl)
        max_y = float(y_bl) + float(h)
        return StaticPlacement(
            entries=entries,
            exits=exits,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            x_bl=int(x_bl),
            y_bl=int(y_bl),
            rot=int(r),
            w=float(w),
            h=float(h),
            cx=float(cx),
            cy=float(cy),
            clearance_left=int(cL),
            clearance_right=int(cR),
            clearance_bottom=int(cB),
            clearance_top=int(cT),
        )

    def build_candidate_features(
        self,
        *,
        x_bl: object,
        y_bl: object,
        rot: object,
        needed: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Build candidate feature tensors for given pose vectors (only requested keys)."""
        supported = {
            "entries",
            "exits",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "x_bl",
            "y_bl",
            "rot",
            "w",
            "h",
            "cx",
            "cy",
            "clearance_left",
            "clearance_right",
            "clearance_bottom",
            "clearance_top",
        }
        unknown = set(needed) - supported
        if unknown:
            raise ValueError(f"build_candidate_features: unknown feature keys: {sorted(unknown)}")

        scalar_input = not (torch.is_tensor(x_bl) or torch.is_tensor(y_bl) or torch.is_tensor(rot))
        if scalar_input:
            x_bl = torch.tensor([int(x_bl)], dtype=torch.long, device=self.device)
            y_bl = torch.tensor([int(y_bl)], dtype=torch.long, device=self.device)
            rot = torch.tensor([int(rot)], dtype=torch.long, device=self.device)

        x_bl = x_bl.to(device=self.device, dtype=torch.long).view(-1)
        y_bl = y_bl.to(device=self.device, dtype=torch.long).view(-1)
        rot = rot.to(device=self.device, dtype=torch.long).view(-1)
        M = int(x_bl.numel())
        if int(y_bl.numel()) != M or int(rot.numel()) != M:
            raise ValueError("build_candidate_features: x_bl,y_bl,rot must have same length")

        rot_idx = self._rot_idx(rot)
        out: Dict[str, torch.Tensor] = {}

        if {"x_bl", "y_bl", "rot"} & needed:
            if "x_bl" in needed:
                out["x_bl"] = x_bl.to(dtype=torch.float32)
            if "y_bl" in needed:
                out["y_bl"] = y_bl.to(dtype=torch.float32)
            if "rot" in needed:
                out["rot"] = rot.to(dtype=torch.float32)

        if {"w", "h", "min_x", "max_x", "min_y", "max_y", "cx", "cy"} & needed:
            w, h = self._wh_for_rot(rot_idx)
            if "w" in needed:
                out["w"] = w
            if "h" in needed:
                out["h"] = h
            if "min_x" in needed or "max_x" in needed:
                min_x = x_bl.to(dtype=torch.float32)
                if "min_x" in needed:
                    out["min_x"] = min_x
                if "max_x" in needed:
                    out["max_x"] = min_x + w
            if "min_y" in needed or "max_y" in needed:
                min_y = y_bl.to(dtype=torch.float32)
                if "min_y" in needed:
                    out["min_y"] = min_y
                if "max_y" in needed:
                    out["max_y"] = min_y + h
            if "cx" in needed or "cy" in needed:
                center = self._center_from_bl_batch(x_bl, y_bl, rot_idx)
                if "cx" in needed:
                    out["cx"] = center[:, 0]
                if "cy" in needed:
                    out["cy"] = center[:, 1]

        if {"entries", "exits"} & needed:
            center = self._center_from_bl_batch(x_bl, y_bl, rot_idx)
            entry_xy = self._entries_from_bl_batch(center, rot_idx)
            exit_xy = self._exits_from_bl_batch(center, rot_idx)
            if "entries" in needed:
                out["entries"] = entry_xy
            if "exits" in needed:
                out["exits"] = exit_xy

        if {"clearance_left", "clearance_right", "clearance_bottom", "clearance_top"} & needed:
            r = torch.remainder(rot, 360)
            if not bool(self.rotatable):
                r = torch.zeros_like(r)
            cL, cR, cB, cT = (
                int(self.clearance_left_rel),
                int(self.clearance_right_rel),
                int(self.clearance_bottom_rel),
                int(self.clearance_top_rel),
            )
            cL0 = torch.full((M,), cL, device=self.device, dtype=torch.float32)
            cR0 = torch.full((M,), cR, device=self.device, dtype=torch.float32)
            cB0 = torch.full((M,), cB, device=self.device, dtype=torch.float32)
            cT0 = torch.full((M,), cT, device=self.device, dtype=torch.float32)

            r0 = (r == 0)
            r90 = (r == 90)
            r180 = (r == 180)
            r270 = (r == 270)

            cL_t = torch.where(r0, cL0, torch.where(r90, cB0, torch.where(r180, cR0, cT0)))
            cR_t = torch.where(r0, cR0, torch.where(r90, cT0, torch.where(r180, cL0, cB0)))
            cB_t = torch.where(r0, cB0, torch.where(r90, cR0, torch.where(r180, cT0, cL0)))
            cT_t = torch.where(r0, cT0, torch.where(r90, cL0, torch.where(r180, cB0, cR0)))

            if "clearance_left" in needed:
                out["clearance_left"] = cL_t
            if "clearance_right" in needed:
                out["clearance_right"] = cR_t
            if "clearance_bottom" in needed:
                out["clearance_bottom"] = cB_t
            if "clearance_top" in needed:
                out["clearance_top"] = cT_t

        return out

    def build_placement_batch(
        self,
        *,
        x_bl: object,
        y_bl: object,
        rot: object,
        needed: set[str],
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias for build_candidate_features()."""
        return self.build_candidate_features(x_bl=x_bl, y_bl=y_bl, rot=rot, needed=needed)


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
    rot = torch.randint(0, 4, (M,), device=device) * 90

    # build_placement vs build_placement_batch
    scalar = geom.build_placement(x_bl=10, y_bl=5, rot=90)
    print(
        "build_placement(x_bl=10,y_bl=5,rot=90) -> "
        f"w={scalar.w}, h={scalar.h}, entries={len(scalar.entries)}, exits={len(scalar.exits)}"
    )

    t0 = time.perf_counter()
    for i in range(M):
        _ = geom.build_placement(x_bl=int(x[i].item()), y_bl=int(y[i].item()), rot=int(rot[i].item()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    needed = {"entries", "exits", "min_x"}
    t2 = time.perf_counter()
    batch = geom.build_placement_batch(x_bl=x, y_bl=y, rot=rot, needed=needed)
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

    # is_placeable vs is_placeable_batch
    sample_ok = geom.is_placeable(x_bl=10, y_bl=5, rot=90, invalid=invalid, clear_invalid=clear_invalid)
    print(f"is_placeable(x_bl=10,y_bl=5,rot=90) -> {sample_ok}")

    t4 = time.perf_counter()
    for i in range(M):
        _ = geom.is_placeable(
            x_bl=int(x[i].item()),
            y_bl=int(y[i].item()),
            rot=int(rot[i].item()),
            invalid=invalid,
            clear_invalid=clear_invalid,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    mask_batch = geom.is_placeable_batch(
        x=x,
        y=y,
        rot=rot,
        invalid=invalid,
        clear_invalid=clear_invalid,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t7 = time.perf_counter()

    scalar_ms = (t5 - t4) * 1000.0
    batch_ms = (t7 - t6) * 1000.0
    print(f"is_placeable_batch(M={M}) -> true={int(mask_batch.sum().item())}")
    print(f"is_placeable loop: {scalar_ms:.2f} ms total, {scalar_ms / M:.4f} ms/elem")
    print(f"is_placeable_batch: {batch_ms:.2f} ms total, {batch_ms / M:.6f} ms/elem")

    # is_placeable_mask
    t8 = time.perf_counter()
    mask = geom.is_placeable_mask(rot=0, invalid=invalid, clear_invalid=clear_invalid)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t9 = time.perf_counter()
    print(
        f"is_placeable_mask(rot=0) -> shape={tuple(mask.shape)}, true={int(mask.sum().item())}"
    )
    print(f"is_placeable_mask: {(t9 - t8) * 1000.0:.2f} ms total")
