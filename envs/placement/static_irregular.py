from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple

import torch

from .base import PlacementBase
from .static_rect import StaticRectSpec, VariantInfo

logger = logging.getLogger(__name__)


@dataclass
class StaticIrregularPlacement(PlacementBase):
    """Resolved absolute placement for an irregular static group."""

    x_bl: int
    y_bl: int
    rotation: int
    w: float
    h: float
    # TODO(placement-contract): remove duplicated clearance scalars after
    # all placement consumers rely on clearance_map/clearance_origin only.
    clearance_left: int
    clearance_right: int
    clearance_bottom: int
    clearance_top: int
    mirror: bool = False


@dataclass
class StaticIrregularSpec(StaticRectSpec):
    """Static spec backed by a canonical occupancy mask in the local bbox."""

    body_map: Any = None

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} must have positive width/height, "
                f"got width={self.width}, height={self.height}"
            )

        canonical_body_map = self._coerce_body_map(self.body_map)
        if int(canonical_body_map.shape[1]) != self.width or int(canonical_body_map.shape[0]) != self.height:
            raise ValueError(
                f"StaticIrregularSpec {self.id!r} body_map shape must match "
                f"(height={self.height}, width={self.width}), got {tuple(canonical_body_map.shape)!r}"
            )
        self.body_map = canonical_body_map
        self._validate_body_bbox()
        self._validate_ports()
        self._build_variants()

    @property
    def body_area(self) -> float:
        return float(self.body_map.to(dtype=torch.int32).sum().item())

    def set_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.device == device:
            return
        self.body_map = self.body_map.to(device=device, dtype=torch.bool)
        super().set_device(device)

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
        entries = list(self._entries_from_center(x_c_s, y_c_s, r, mirror=mirror))
        exits = list(self._exits_from_center(x_c_s, y_c_s, r, mirror=mirror))
        body_map, clearance_map, clearance_origin, is_rectangular = self.shape_tensors(vi.shape_key)
        return StaticIrregularPlacement(
            x_c=x_c_s,
            y_c=y_c_s,
            entries=entries,
            exits=exits,
            min_x=float(x_bl),
            max_x=float(x_bl) + w,
            min_y=float(y_bl),
            max_y=float(y_bl) + h,
            body_map=body_map,
            clearance_map=clearance_map,
            clearance_origin=clearance_origin,
            is_rectangular=bool(is_rectangular),
            x_bl=x_bl,
            y_bl=y_bl,
            rotation=int(r),
            w=w,
            h=h,
            clearance_left=int(vi.cL),
            clearance_right=int(vi.cR),
            clearance_bottom=int(vi.cB),
            clearance_top=int(vi.cT),
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

    def _validate_body_bbox(self) -> None:
        if not bool(self.body_map[0, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the bottom boundary")
        if not bool(self.body_map[-1, :].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the top boundary")
        if not bool(self.body_map[:, 0].any().item()):
            raise ValueError(f"StaticIrregularSpec {self.id!r} body_map must touch the left boundary")
        if not bool(self.body_map[:, -1].any().item()):
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

    def _build_variants(self) -> None:
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
                cL, cR, cB, cT = self._clearance_lrtb(rr, mirror=m)
                body_map = self._transform_body_map(self.body_map, rr, mirror=m)
                body_h = int(body_map.shape[0])
                body_w = int(body_map.shape[1])
                clearance_map = self._build_clearance_map(body_map, cL=cL, cR=cR, cB=cB, cT=cT)
                clearance_origin = (int(cL), int(cB))
                is_rectangular = bool(body_map.all().item()) and bool(clearance_map.all().item())

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
                pk = self._make_shape_key(
                    body_map=body_map,
                    clearance_map=clearance_map,
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                )
                ck = (pk, eo_t, xo_t)
                if ck in seen_cost:
                    continue
                seen_cost.add(ck)

                vi = VariantInfo(
                    rotation=rr,
                    mirror=bool(m),
                    body_w=body_w,
                    body_h=body_h,
                    cL=int(cL),
                    cR=int(cR),
                    cB=int(cB),
                    cT=int(cT),
                    clearance_origin=clearance_origin,
                    is_rectangular=is_rectangular,
                    entry_offsets=eo_t,
                    exit_offsets=xo_t,
                    shape_key=pk,
                    cost_key=ck,
                )
                variants.append(vi)
                shape_tensors_by_key[pk] = (body_map, clearance_map, clearance_origin, is_rectangular)
                if pk not in variants_by_shape:
                    variants_by_shape[pk] = []
                variants_by_shape[pk].append(vi)

        self._variants = variants
        self._variants_by_shape = variants_by_shape
        self._shape_tensors_by_key = shape_tensors_by_key
        self._variant_entry_off_t = []
        self._variant_exit_off_t = []
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
            "StaticIrregularSpec %r variant summary: raw=%d, unique_variants=%d, unique_shapes=%d",
            self.id,
            raw_variant_count,
            len(self._variants),
            len(self._variants_by_shape),
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

    def _build_clearance_map(
        self,
        body_map: torch.Tensor,
        *,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> torch.Tensor:
        body_h, body_w = int(body_map.shape[0]), int(body_map.shape[1])
        clear_h = body_h + int(cB) + int(cT)
        clear_w = body_w + int(cL) + int(cR)
        out = torch.zeros((clear_h, clear_w), dtype=torch.bool, device=self.device)
        ys, xs = torch.nonzero(body_map, as_tuple=True)
        span_h = int(cB) + int(cT) + 1
        span_w = int(cL) + int(cR) + 1
        for x, y in zip(xs.tolist(), ys.tolist()):
            out[int(y) : int(y) + span_h, int(x) : int(x) + span_w] = True
        return out
