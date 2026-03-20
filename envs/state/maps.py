from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..action import GroupId
from ..placement.static import StaticSpec

RectI = Tuple[int, int, int, int]


class GridMaps:
    """Runtime map state + static constraint maps.

    Runtime tensors are copied on `copy()`.
    Static maps/cache are shared by reference for efficiency when device matches.
    """

    def __init__(
        self,
        *,
        grid_height: int,
        grid_width: int,
        device: torch.device,
        forbidden_areas: List[Dict[str, Any]],
        zone_constraints: Dict[str, Dict[str, Any]],
        collision_check: str = "auto",
    ) -> None:
        self._H = int(grid_height)
        self._W = int(grid_width)
        self._device = torch.device(device)
        self._collision_check = str(collision_check).lower()
        self._resolved_collision_check = self._resolve_collision_check(
            self._collision_check,
            self._device,
        )

        self._static_invalid = self._build_static_invalid(
            self._H,
            self._W,
            self._device,
            forbidden_areas,
        )
        self._zone_constraints = dict(zone_constraints or {})
        (
            self._constraint_maps,
            self._constraint_ops,
            self._constraint_dtypes,
        ) = self._build_constraint_maps(
            self._H,
            self._W,
            self._device,
            self._zone_constraints,
        )

        # Runtime fields.
        self.occ_invalid = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        self.clear_invalid = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        self.has_bbox = False
        self.bbox_min_x = 0.0
        self.bbox_max_x = 0.0
        self.bbox_min_y = 0.0
        self.bbox_max_y = 0.0

        # Static caches.
        self._group_specs: Dict[GroupId, StaticSpec] = {}
        self._zone_invalid_by_gid: Dict[GroupId, torch.Tensor] = {}
        self._static_invalid_ps: Optional[torch.Tensor] = None
        self._zone_invalid_ps_by_gid: Dict[GroupId, torch.Tensor] = {}
        self._rect_linear_offsets_cache: Dict[
            Tuple[int, int, int, int, int, int],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}

        # Runtime prefix caches (prefixsum backend only).
        self._occ_invalid_ps: Optional[torch.Tensor] = None
        self._clear_invalid_ps: Optional[torch.Tensor] = None

        if self._uses_prefixsum:
            self._static_invalid_ps = self._build_prefix(self._static_invalid)
            self._rebuild_runtime_prefix_cache()

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self._H), int(self._W)

    @property
    def grid_width(self) -> int:
        return int(self._W)

    @property
    def grid_height(self) -> int:
        return int(self._H)

    @property
    def collision_check(self) -> str:
        return str(self._resolved_collision_check)

    @property
    def static_invalid(self) -> torch.Tensor:
        return self._static_invalid

    @property
    def invalid(self) -> torch.Tensor:
        # Runtime composite invalid map (no cached tensor):
        # static forbidden + occupied body area.
        return self._static_invalid | self.occ_invalid

    @property
    def constraint_maps(self) -> Dict[str, torch.Tensor]:
        return self._constraint_maps

    @property
    def _uses_prefixsum(self) -> bool:
        return self._resolved_collision_check == "prefixsum"

    @staticmethod
    def _resolve_collision_check(mode: str, device: torch.device) -> str:
        m = str(mode).lower()
        if m not in {"conv", "prefixsum", "auto"}:
            raise ValueError(f"collision_check must be 'conv'|'prefixsum'|'auto', got {mode!r}")
        if m != "auto":
            return m
        return "conv" if torch.device(device).type == "cuda" else "prefixsum"

    @staticmethod
    def _rect_hits_invalid(rect: RectI, src: torch.Tensor) -> bool:
        x0, y0, x1, y1 = rect
        h, w = src.shape
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return True
        if x0 >= x1 or y0 >= y1:
            return False
        return bool(torch.any(src[y0:y1, x0:x1]).item())

    @staticmethod
    def _build_prefix(src: torch.Tensor) -> torch.Tensor:
        if src.dim() != 2:
            raise ValueError(f"prefix source must be [H,W], got {tuple(src.shape)}")
        src_i = src.to(dtype=torch.int32)
        ps = torch.cumsum(torch.cumsum(src_i, dim=0), dim=1)
        out = torch.zeros(
            (int(src_i.shape[0]) + 1, int(src_i.shape[1]) + 1),
            dtype=torch.int32,
            device=src_i.device,
        )
        out[1:, 1:] = ps
        return out

    @staticmethod
    def _build_prefix_batch(src: torch.Tensor) -> torch.Tensor:
        if src.dim() != 3:
            raise ValueError(f"prefix batch source must be [N,H,W], got {tuple(src.shape)}")
        src_i = src.to(dtype=torch.int32)
        ps = torch.cumsum(torch.cumsum(src_i, dim=1), dim=2)
        n, h, w = src_i.shape
        out = torch.zeros((int(n), int(h) + 1, int(w) + 1), dtype=torch.int32, device=src_i.device)
        out[:, 1:, 1:] = ps
        return out

    @staticmethod
    def _window_sum(prefix: torch.Tensor, height: int, width: int) -> torch.Tensor:
        h = int(height)
        w = int(width)
        if h <= 0 or w <= 0:
            raise ValueError(f"window size must be positive, got ({h},{w})")
        return prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]

    @staticmethod
    def _rect_sum_batch(
        prefix: torch.Tensor,
        *,
        x0: torch.Tensor,
        y0: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
    ) -> torch.Tensor:
        if prefix.dim() != 2:
            raise ValueError(f"prefix must be [H+1,W+1], got {tuple(prefix.shape)}")
        if x0.shape != y0.shape or x0.shape != x1.shape or x0.shape != y1.shape:
            raise ValueError("rectangle corner tensors must share the same shape")
        stride = int(prefix.shape[1])
        flat = prefix.reshape(-1)
        x0 = x0.to(dtype=torch.long)
        y0 = y0.to(dtype=torch.long)
        x1 = x1.to(dtype=torch.long)
        y1 = y1.to(dtype=torch.long)
        idx00 = y0 * stride + x0
        idx01 = y0 * stride + x1
        idx10 = y1 * stride + x0
        idx11 = y1 * stride + x1
        return flat[idx11] - flat[idx01] - flat[idx10] + flat[idx00]

    def _get_rect_linear_offsets(
        self,
        *,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(body_w), int(body_h), int(cL), int(cR), int(cB), int(cT))
        cached = self._rect_linear_offsets_cache.get(key, None)
        if cached is not None:
            return cached

        body_x = torch.arange(int(body_w), dtype=torch.long, device=self._device)
        body_y = torch.arange(int(body_h), dtype=torch.long, device=self._device)
        body_offsets = (body_y[:, None] * int(self._W) + body_x[None, :]).reshape(-1)

        pad_x = torch.arange(-int(cL), int(body_w) + int(cR), dtype=torch.long, device=self._device)
        pad_y = torch.arange(-int(cB), int(body_h) + int(cT), dtype=torch.long, device=self._device)
        pad_offsets = (pad_y[:, None] * int(self._W) + pad_x[None, :]).reshape(-1)

        out = (body_offsets, pad_offsets)
        self._rect_linear_offsets_cache[key] = out
        return out

    def _ensure_static_prefix_cache(self) -> torch.Tensor:
        prefix = self._static_invalid_ps
        if isinstance(prefix, torch.Tensor):
            return prefix
        prefix = self._build_prefix(self._static_invalid)
        self._static_invalid_ps = prefix
        return prefix

    def _ensure_runtime_prefix_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        occ_ps = self._occ_invalid_ps
        clear_ps = self._clear_invalid_ps
        if isinstance(occ_ps, torch.Tensor) and isinstance(clear_ps, torch.Tensor):
            return occ_ps, clear_ps
        self._rebuild_runtime_prefix_cache()
        if self._occ_invalid_ps is None or self._clear_invalid_ps is None:
            raise RuntimeError("runtime prefix cache rebuild failed")
        return self._occ_invalid_ps, self._clear_invalid_ps

    def _rebuild_runtime_prefix_cache(self) -> None:
        if not self._uses_prefixsum:
            self._occ_invalid_ps = None
            self._clear_invalid_ps = None
            return
        stacked = torch.stack([self.occ_invalid, self.clear_invalid], dim=0)
        prefix = self._build_prefix_batch(stacked)
        self._occ_invalid_ps = prefix[0]
        self._clear_invalid_ps = prefix[1]

    def _build_zone_invalid_cache(self) -> None:
        zone_invalid_by_gid: Dict[GroupId, torch.Tensor] = {}
        for gid, spec in self._group_specs.items():
            z = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
            zone_values = dict(getattr(spec, "zone_values", {}) or {})
            for cname, cmap in self._constraint_maps.items():
                if cname not in zone_values:
                    continue
                op = self._constraint_ops[cname]
                dtype = self._constraint_dtypes[cname]
                gval = self._coerce_group_value(zone_values[cname], dtype=dtype, cname=cname)
                # op = facility requirement: facility_value op zone_value (e.g. height<=30)
                pass_mask = self._compare_constraint(gval, cmap, op=op)
                z |= (~pass_mask)
            zone_invalid_by_gid[gid] = z
        self._zone_invalid_by_gid = zone_invalid_by_gid

        if not self._uses_prefixsum:
            self._zone_invalid_ps_by_gid = {}
            return
        if not zone_invalid_by_gid:
            self._zone_invalid_ps_by_gid = {}
            return

        gids = list(zone_invalid_by_gid.keys())
        stacked = torch.stack([zone_invalid_by_gid[gid] for gid in gids], dim=0)
        prefix = self._build_prefix_batch(stacked)
        self._zone_invalid_ps_by_gid = {
            gid: prefix[idx]
            for idx, gid in enumerate(gids)
        }

    def _get_zone_invalid(self, gid: GroupId) -> torch.Tensor:
        if gid not in self._group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        z = self._zone_invalid_by_gid.get(gid, None)
        if isinstance(z, torch.Tensor):
            return z
        raise RuntimeError(
            f"zone invalid cache miss for gid={gid!r}; "
            "call bind_group_specs()/_build_zone_invalid_cache() before placement checks"
        )

    def _get_zone_invalid_ps(self, gid: GroupId) -> torch.Tensor:
        if gid not in self._group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        z = self._zone_invalid_ps_by_gid.get(gid, None)
        if isinstance(z, torch.Tensor):
            return z
        raise RuntimeError(
            f"zone invalid prefix cache miss for gid={gid!r}; "
            "call bind_group_specs()/_build_zone_invalid_cache() before placement checks"
        )

    def is_placeable(
        self,
        *,
        gid: GroupId,
        body_rect: RectI,
        pad_rect: RectI,
    ) -> bool:
        invalid = self._static_invalid | self.occ_invalid | self._get_zone_invalid(gid)
        if self._rect_hits_invalid(body_rect, invalid):
            return False
        if self._rect_hits_invalid(body_rect, self.clear_invalid):
            return False
        if self._rect_hits_invalid(pad_rect, invalid):
            return False
        return True

    def _is_placeable_map_conv(
        self,
        *,
        gid: GroupId,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
        valid_h: int,
        valid_w: int,
    ) -> torch.Tensor:
        invalid = self._static_invalid | self.occ_invalid | self._get_zone_invalid(gid)
        kw = int(body_w + cL + cR)
        kh = int(body_h + cB + cT)

        inv_f = invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        clear_f = self.clear_invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        body_kernel = torch.ones((1, 1, body_h, body_w), device=self._device, dtype=torch.float32)
        pad_kernel = torch.ones((1, 1, kh, kw), device=self._device, dtype=torch.float32)

        body_inv = F.conv2d(inv_f, body_kernel, padding=0).squeeze(0).squeeze(0)
        body_clear = F.conv2d(clear_f, body_kernel, padding=0).squeeze(0).squeeze(0)
        pad_inv = F.conv2d(inv_f, pad_kernel, padding=0).squeeze(0).squeeze(0)

        body_inv_slice = body_inv[cB: cB + valid_h, cL: cL + valid_w]
        body_clear_slice = body_clear[cB: cB + valid_h, cL: cL + valid_w]
        return (body_inv_slice == 0) & (body_clear_slice == 0) & (pad_inv == 0)

    def _is_placeable_map_prefix(
        self,
        *,
        gid: GroupId,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
        valid_h: int,
        valid_w: int,
    ) -> torch.Tensor:
        static_ps = self._ensure_static_prefix_cache()
        occ_ps, clear_ps = self._ensure_runtime_prefix_cache()

        kw = int(body_w + cL + cR)
        kh = int(body_h + cB + cT)
        zone_ps = self._get_zone_invalid_ps(gid)

        pad_static = self._window_sum(static_ps, kh, kw)
        pad_occ = self._window_sum(occ_ps, kh, kw)
        pad_zone = self._window_sum(zone_ps, kh, kw)
        pad_hit = pad_static + pad_occ + pad_zone

        body_static = self._window_sum(static_ps, body_h, body_w)
        body_occ = self._window_sum(occ_ps, body_h, body_w)
        body_zone = self._window_sum(zone_ps, body_h, body_w)
        body_clear = self._window_sum(clear_ps, body_h, body_w)

        body_hit = (
            body_static[cB: cB + valid_h, cL: cL + valid_w]
            + body_occ[cB: cB + valid_h, cL: cL + valid_w]
            + body_zone[cB: cB + valid_h, cL: cL + valid_w]
            + body_clear[cB: cB + valid_h, cL: cL + valid_w]
        )
        return (pad_hit == 0) & (body_hit == 0)

    def _is_placeable_batch_prefix(
        self,
        *,
        gid: GroupId,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> torch.Tensor:
        """Direct rectangle query path for sparse batch checks.

        This stays rectangle-specific so future irregular shapes can add a
        separate mask-based query path without rewriting the full-map backends.
        """
        if x_bl.shape != y_bl.shape:
            raise ValueError(
                f"x_bl and y_bl must share shape, got {tuple(x_bl.shape)} vs {tuple(y_bl.shape)}"
            )

        result = torch.zeros_like(x_bl, dtype=torch.bool, device=self._device)
        if result.numel() == 0:
            return result

        bw = int(body_w)
        bh = int(body_h)
        kw = int(bw + cL + cR)
        kh = int(bh + cB + cT)
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return result

        x_bl = x_bl.to(device=self._device, dtype=torch.long)
        y_bl = y_bl.to(device=self._device, dtype=torch.long)

        body_x0 = x_bl
        body_y0 = y_bl
        body_x1 = x_bl + bw
        body_y1 = y_bl + bh

        pad_x0 = x_bl - int(cL)
        pad_y0 = y_bl - int(cB)
        pad_x1 = x_bl + bw + int(cR)
        pad_y1 = y_bl + bh + int(cT)

        in_bounds = (
            (body_x0 >= 0)
            & (body_y0 >= 0)
            & (body_x1 <= self._W)
            & (body_y1 <= self._H)
            & (pad_x0 >= 0)
            & (pad_y0 >= 0)
            & (pad_x1 <= self._W)
            & (pad_y1 <= self._H)
        )
        if not bool(in_bounds.any().item()):
            return result

        static_ps = self._ensure_static_prefix_cache()
        occ_ps, clear_ps = self._ensure_runtime_prefix_cache()
        zone_ps = self._get_zone_invalid_ps(gid)

        body_static = self._rect_sum_batch(
            static_ps,
            x0=body_x0[in_bounds],
            y0=body_y0[in_bounds],
            x1=body_x1[in_bounds],
            y1=body_y1[in_bounds],
        )
        body_occ = self._rect_sum_batch(
            occ_ps,
            x0=body_x0[in_bounds],
            y0=body_y0[in_bounds],
            x1=body_x1[in_bounds],
            y1=body_y1[in_bounds],
        )
        body_zone = self._rect_sum_batch(
            zone_ps,
            x0=body_x0[in_bounds],
            y0=body_y0[in_bounds],
            x1=body_x1[in_bounds],
            y1=body_y1[in_bounds],
        )
        body_clear = self._rect_sum_batch(
            clear_ps,
            x0=body_x0[in_bounds],
            y0=body_y0[in_bounds],
            x1=body_x1[in_bounds],
            y1=body_y1[in_bounds],
        )

        pad_static = self._rect_sum_batch(
            static_ps,
            x0=pad_x0[in_bounds],
            y0=pad_y0[in_bounds],
            x1=pad_x1[in_bounds],
            y1=pad_y1[in_bounds],
        )
        pad_occ = self._rect_sum_batch(
            occ_ps,
            x0=pad_x0[in_bounds],
            y0=pad_y0[in_bounds],
            x1=pad_x1[in_bounds],
            y1=pad_y1[in_bounds],
        )
        pad_zone = self._rect_sum_batch(
            zone_ps,
            x0=pad_x0[in_bounds],
            y0=pad_y0[in_bounds],
            x1=pad_x1[in_bounds],
            y1=pad_y1[in_bounds],
        )

        result[in_bounds] = (
            (body_static + body_occ + body_zone == 0)
            & (body_clear == 0)
            & (pad_static + pad_occ + pad_zone == 0)
        )
        return result

    def _is_placeable_batch_gather(
        self,
        *,
        gid: GroupId,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> torch.Tensor:
        if x_bl.shape != y_bl.shape:
            raise ValueError(
                f"x_bl and y_bl must share shape, got {tuple(x_bl.shape)} vs {tuple(y_bl.shape)}"
            )

        result = torch.zeros(x_bl.shape, dtype=torch.bool, device=self._device)
        if result.numel() == 0:
            return result

        bw = int(body_w)
        bh = int(body_h)
        kw = int(bw + cL + cR)
        kh = int(bh + cB + cT)
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return result

        x_flat = x_bl.to(device=self._device, dtype=torch.long).reshape(-1)
        y_flat = y_bl.to(device=self._device, dtype=torch.long).reshape(-1)

        body_x0 = x_flat
        body_y0 = y_flat
        body_x1 = x_flat + bw
        body_y1 = y_flat + bh

        pad_x0 = x_flat - int(cL)
        pad_y0 = y_flat - int(cB)
        pad_x1 = x_flat + bw + int(cR)
        pad_y1 = y_flat + bh + int(cT)

        in_bounds = (
            (body_x0 >= 0)
            & (body_y0 >= 0)
            & (body_x1 <= self._W)
            & (body_y1 <= self._H)
            & (pad_x0 >= 0)
            & (pad_y0 >= 0)
            & (pad_x1 <= self._W)
            & (pad_y1 <= self._H)
        )
        if not bool(in_bounds.any().item()):
            return result

        valid_idx = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
        x_valid = x_flat[valid_idx]
        y_valid = y_flat[valid_idx]
        base = y_valid * int(self._W) + x_valid

        body_offsets, pad_offsets = self._get_rect_linear_offsets(
            body_w=bw,
            body_h=bh,
            cL=int(cL),
            cR=int(cR),
            cB=int(cB),
            cT=int(cT),
        )
        static_f = self._static_invalid.reshape(-1)
        occ_f = self.occ_invalid.reshape(-1)
        zone_f = self._get_zone_invalid(gid).reshape(-1)
        clear_f = self.clear_invalid.reshape(-1)

        offset_count = max(int(body_offsets.numel()), int(pad_offsets.numel()), 1)
        chunk_size = max(1, 1_048_576 // offset_count)
        flat_result = result.reshape(-1)

        for start in range(0, int(valid_idx.numel()), chunk_size):
            stop = min(start + chunk_size, int(valid_idx.numel()))
            base_chunk = base[start:stop]
            body_idx = base_chunk[:, None] + body_offsets[None, :]
            pad_idx = base_chunk[:, None] + pad_offsets[None, :]

            body_hit = static_f[body_idx].any(dim=1)
            body_hit |= occ_f[body_idx].any(dim=1)
            body_hit |= zone_f[body_idx].any(dim=1)
            body_clear_hit = clear_f[body_idx].any(dim=1)

            pad_hit = static_f[pad_idx].any(dim=1)
            pad_hit |= occ_f[pad_idx].any(dim=1)
            pad_hit |= zone_f[pad_idx].any(dim=1)

            flat_result[valid_idx[start:stop]] = (~body_hit) & (~body_clear_hit) & (~pad_hit)
        return result

    def is_placeable_batch(
        self,
        *,
        gid: GroupId,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_w: int,
        body_h: int,
        cL: int = 0,
        cR: int = 0,
        cB: int = 0,
        cT: int = 0,
    ) -> torch.Tensor:
        """Vectorized placeability check. Returns [N] bool."""
        if self._resolved_collision_check == "conv":
            result = self._is_placeable_batch_gather(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_w=int(body_w),
                body_h=int(body_h),
                cL=int(cL),
                cR=int(cR),
                cB=int(cB),
                cT=int(cT),
            )
        elif self._resolved_collision_check == "prefixsum":
            result = self._is_placeable_batch_prefix(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_w=int(body_w),
                body_h=int(body_h),
                cL=int(cL),
                cR=int(cR),
                cB=int(cB),
                cT=int(cT),
            )
        else:
            raise RuntimeError(f"unsupported collision backend: {self._resolved_collision_check!r}")
        return result

    def is_placeable_map(
        self,
        *,
        gid: GroupId,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> torch.Tensor:
        w = int(body_w)
        h = int(body_h)
        cL = int(cL)
        cR = int(cR)
        cB = int(cB)
        cT = int(cT)

        H, W = self.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self._device)
        if h <= 0 or w <= 0:
            return result

        kw = int(w + cL + cR)
        kh = int(h + cB + cT)
        if kh > H or kw > W or kh <= 0 or kw <= 0:
            return result

        valid_h = H - kh + 1
        valid_w = W - kw + 1
        if valid_h <= 0 or valid_w <= 0:
            return result

        if self._resolved_collision_check == "conv":
            valid_mask = self._is_placeable_map_conv(
                gid=gid,
                body_w=w,
                body_h=h,
                cL=cL,
                cR=cR,
                cB=cB,
                cT=cT,
                valid_h=valid_h,
                valid_w=valid_w,
            )
        elif self._resolved_collision_check == "prefixsum":
            valid_mask = self._is_placeable_map_prefix(
                gid=gid,
                body_w=w,
                body_h=h,
                cL=cL,
                cR=cR,
                cB=cB,
                cT=cT,
                valid_h=valid_h,
                valid_w=valid_w,
            )
        else:
            raise RuntimeError(f"unsupported collision backend: {self._resolved_collision_check!r}")

        result[cB:cB + valid_h, cL:cL + valid_w] = valid_mask
        return result

    def bind_group_specs(self, group_specs: Dict[GroupId, StaticSpec]) -> None:
        self._group_specs = group_specs
        self._build_zone_invalid_cache()

    def copy(self) -> "GridMaps":
        out = object.__new__(GridMaps)
        out._H = self._H
        out._W = self._W
        out._device = self._device
        out._collision_check = self._collision_check
        out._resolved_collision_check = self._resolved_collision_check

        out.occ_invalid = self.occ_invalid.clone()
        out.clear_invalid = self.clear_invalid.clone()
        out.has_bbox = bool(self.has_bbox)
        out.bbox_min_x = float(self.bbox_min_x)
        out.bbox_max_x = float(self.bbox_max_x)
        out.bbox_min_y = float(self.bbox_min_y)
        out.bbox_max_y = float(self.bbox_max_y)

        out._static_invalid = self._static_invalid
        out._zone_constraints = self._zone_constraints
        out._constraint_maps = self._constraint_maps
        out._constraint_ops = self._constraint_ops
        out._constraint_dtypes = self._constraint_dtypes
        out._group_specs = self._group_specs
        out._zone_invalid_by_gid = self._zone_invalid_by_gid
        out._static_invalid_ps = self._static_invalid_ps
        out._zone_invalid_ps_by_gid = self._zone_invalid_ps_by_gid
        out._rect_linear_offsets_cache = self._rect_linear_offsets_cache
        if self._uses_prefixsum:
            out._occ_invalid_ps = self._occ_invalid_ps
            out._clear_invalid_ps = self._clear_invalid_ps
        else:
            out._occ_invalid_ps = None
            out._clear_invalid_ps = None
        return out

    def restore(self, src: "GridMaps") -> None:
        """In-place restore of runtime map fields."""
        if not isinstance(src, GridMaps):
            raise TypeError(f"src must be GridMaps, got {type(src).__name__}")
        if int(self._H) != int(src._H) or int(self._W) != int(src._W):
            raise ValueError(
                f"grid shape mismatch: source=({int(src._H)},{int(src._W)}), "
                f"target=({int(self._H)},{int(self._W)})"
            )
        self.occ_invalid.copy_(src.occ_invalid.to(device=self._device, dtype=torch.bool))
        self.clear_invalid.copy_(src.clear_invalid.to(device=self._device, dtype=torch.bool))
        self.has_bbox = bool(src.has_bbox)
        self.bbox_min_x = float(src.bbox_min_x)
        self.bbox_max_x = float(src.bbox_max_x)
        self.bbox_min_y = float(src.bbox_min_y)
        self.bbox_max_y = float(src.bbox_max_y)
        if self._uses_prefixsum:
            self._occ_invalid_ps = src._occ_invalid_ps
            self._clear_invalid_ps = src._clear_invalid_ps
        else:
            self._occ_invalid_ps = None
            self._clear_invalid_ps = None

    def reset_runtime(self) -> None:
        self.occ_invalid.zero_()
        self.clear_invalid.zero_()
        self.has_bbox = False
        self.bbox_min_x = 0.0
        self.bbox_max_x = 0.0
        self.bbox_min_y = 0.0
        self.bbox_max_y = 0.0
        if self._uses_prefixsum:
            self._rebuild_runtime_prefix_cache()

    def update_rects(
        self,
        *,
        bbox_min_x: float,
        bbox_max_x: float,
        bbox_min_y: float,
        bbox_max_y: float,
        body_rect: RectI,
        clear_rect: RectI,
    ) -> None:
        min_x = float(bbox_min_x)
        min_y = float(bbox_min_y)
        max_x = float(bbox_max_x)
        max_y = float(bbox_max_y)
        if not self.has_bbox:
            self.has_bbox = True
            self.bbox_min_x = float(min_x)
            self.bbox_max_x = float(max_x)
            self.bbox_min_y = float(min_y)
            self.bbox_max_y = float(max_y)
        else:
            self.bbox_min_x = min(float(self.bbox_min_x), float(min_x))
            self.bbox_max_x = max(float(self.bbox_max_x), float(max_x))
            self.bbox_min_y = min(float(self.bbox_min_y), float(min_y))
            self.bbox_max_y = max(float(self.bbox_max_y), float(max_y))

        x0, y0, x1, y1 = body_rect
        bx0 = max(0, min(self._W, int(x0)))
        bx1 = max(0, min(self._W, int(x1)))
        by0 = max(0, min(self._H, int(y0)))
        by1 = max(0, min(self._H, int(y1)))
        if bx0 < bx1 and by0 < by1:
            self.occ_invalid[by0:by1, bx0:bx1] = True

        px0, py0, px1, py1 = clear_rect
        cx0 = max(0, min(self._W, int(px0)))
        cx1 = max(0, min(self._W, int(px1)))
        cy0 = max(0, min(self._H, int(py0)))
        cy1 = max(0, min(self._H, int(py1)))
        if cx0 < cx1 and cy0 < cy1:
            self.clear_invalid[cy0:cy1, cx0:cx1] = True

        if self._uses_prefixsum:
            self._rebuild_runtime_prefix_cache()

    def paint_rects(
        self,
        *,
        bbox_min_x: float,
        bbox_max_x: float,
        bbox_min_y: float,
        bbox_max_y: float,
        body_rect: RectI,
        clear_rect: RectI,
    ) -> None:
        self.update_rects(
            bbox_min_x=bbox_min_x,
            bbox_max_x=bbox_max_x,
            bbox_min_y=bbox_min_y,
            bbox_max_y=bbox_max_y,
            body_rect=body_rect,
            clear_rect=clear_rect,
        )

    def placed_bbox(self) -> Tuple[float, float, float, float]:
        if not self.has_bbox:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(self.bbox_min_x),
            float(self.bbox_max_x),
            float(self.bbox_min_y),
            float(self.bbox_max_y),
        )

    @staticmethod
    def _build_static_invalid(
        H: int,
        W: int,
        device: torch.device,
        forbidden_areas: List[Dict[str, Any]],
    ) -> torch.Tensor:
        inv = torch.zeros((H, W), dtype=torch.bool, device=device)
        for area in forbidden_areas:
            if not isinstance(area, dict) or "rect" not in area:
                continue
            rect = area["rect"]
            x0 = max(0, min(W, int(rect[0])))
            x1 = max(0, min(W, int(rect[2])))
            y0 = max(0, min(H, int(rect[1])))
            y1 = max(0, min(H, int(rect[3])))
            if x1 > x0 and y1 > y0:
                inv[y0:y1, x0:x1] = True
        return inv

    def _build_constraint_maps(
        self,
        H: int,
        W: int,
        device: torch.device,
        constraints: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, str]]:
        maps: Dict[str, torch.Tensor] = {}
        ops: Dict[str, str] = {}
        dtypes: Dict[str, str] = {}

        for cname, cfg in constraints.items():
            if not isinstance(cfg, dict):
                continue
            dtype = str(cfg.get("dtype", "float")).lower()
            op = str(cfg.get("op", "=="))
            areas = cfg.get("areas", [])
            if dtype not in {"float", "int", "bool"}:
                raise ValueError(f"invalid dtype for constraint {cname!r}: {dtype!r}")
            if op not in {"<", "<=", ">", ">=", "==", "!="}:
                raise ValueError(f"invalid op for constraint {cname!r}: {op!r}")
            if dtype == "bool" and op not in {"==", "!="}:
                raise ValueError(f"bool constraint {cname!r} supports only == or !=")
            if not isinstance(areas, list):
                raise ValueError(f"constraint {cname!r}.areas must be a list")
            if "default" not in cfg:
                raise ValueError(f"constraint {cname!r}.default is required")

            default_v = self._coerce_constraint_value(
                cfg["default"],
                dtype=dtype,
                cname=str(cname),
                ctx="default",
            )
            if dtype == "float":
                m = torch.full((H, W), float(default_v), dtype=torch.float32, device=device)
            elif dtype == "int":
                m = torch.full((H, W), int(default_v), dtype=torch.int64, device=device)
            else:
                m = torch.full((H, W), bool(default_v), dtype=torch.bool, device=device)

            for area in areas:
                if not isinstance(area, dict):
                    continue
                rect = area.get("rect", None)
                value = area.get("value", None)
                if rect is None or value is None:
                    continue
                x0, y0, x1, y1 = rect
                x0 = max(0, min(W, int(x0)))
                x1 = max(0, min(W, int(x1)))
                y0 = max(0, min(H, int(y0)))
                y1 = max(0, min(H, int(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                v = self._coerce_constraint_value(
                    value,
                    dtype=dtype,
                    cname=str(cname),
                    ctx="area",
                )
                m[y0:y1, x0:x1] = v

            name = str(cname)
            maps[name] = m
            ops[name] = op
            dtypes[name] = dtype

        return maps, ops, dtypes

    def _coerce_group_value(self, v: Any, *, dtype: str, cname: str) -> Any:
        return self._coerce_constraint_value(v, dtype=dtype, cname=cname, ctx="group")

    def _coerce_constraint_value(self, v: Any, *, dtype: str, cname: str, ctx: str) -> Any:
        if dtype == "float":
            return float(v)
        if dtype == "int":
            vf = float(v)
            rv = round(vf)
            if abs(vf - rv) > 1e-6:
                raise ValueError(
                    f"{ctx} value for {cname!r} must be int-like, got {v!r}"
                )
            return int(rv)
        if dtype == "bool":
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and float(v) in (0.0, 1.0):
                return bool(int(v))
            raise ValueError(
                f"{ctx} value for {cname!r} must be bool (or 0/1), got {v!r}"
            )
        raise ValueError(f"unknown dtype {dtype!r} for {cname!r} ({ctx})")

    @staticmethod
    def _compare_constraint(src: torch.Tensor, v: Any, *, op: str) -> torch.Tensor:
        if op == "<":
            return src < v
        if op == "<=":
            return src <= v
        if op == ">":
            return src > v
        if op == ">=":
            return src >= v
        if op == "==":
            return src == v
        if op == "!=":
            return src != v
        raise ValueError(f"unsupported op: {op!r}")
