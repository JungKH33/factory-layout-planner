from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..placement.base import GroupSpec

logger = logging.getLogger(__name__)


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
        forbidden: List[Dict[str, Any]],
        zone_constraints: Dict[str, Dict[str, Any]],
        backend_selection: str = "static",
    ) -> None:
        self._H = int(grid_height)
        self._W = int(grid_width)
        self._device = torch.device(device)
        self._backend_selection = backend_selection  # "static" | "benchmark"
        # Populated by _resolve_backends_static / _resolve_backends_benchmark
        # key: (operation, spec_class_type) → backend name
        self._backends: Dict[Tuple[str, type], str] = {}

        self._static_invalid = self._build_static_invalid(
            self._H,
            self._W,
            self._device,
            forbidden,
        )
        self._zone_constraints = dict(zone_constraints or {})
        (
            self._constraint_maps,
            self._constraint_ops,
            self._constraint_dtypes,
            self._constraint_exclusive,
            self._constraint_id_maps,
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
        self._group_specs: Dict[str | int, GroupSpec] = {}
        self._zone_invalid_by_gid: Dict[str | int, torch.Tensor] = {}
        self._static_invalid_ps: torch.Tensor = self._build_prefix(self._static_invalid)
        self._zone_invalid_ps_by_gid: Dict[str | int, torch.Tensor] = {}
        self._mask_linear_offsets_cache: Dict[
            Tuple[int, int, int, int, int],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._body_edge_offsets_cache: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._has_exclusive_edges = False
        self._exclusive_edge_x: Optional[torch.Tensor] = None
        self._exclusive_edge_y: Optional[torch.Tensor] = None
        self._exclusive_edge_x_ps: Optional[torch.Tensor] = None
        self._exclusive_edge_y_ps: Optional[torch.Tensor] = None
        self._build_exclusive_edge_cache()

        # Runtime prefix caches are always maintained for rectangular fast paths.
        self._occ_invalid_ps: torch.Tensor = torch.zeros((self._H + 1, self._W + 1), dtype=torch.int32, device=self._device)
        self._clear_invalid_ps: torch.Tensor = torch.zeros((self._H + 1, self._W + 1), dtype=torch.int32, device=self._device)
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

    @staticmethod
    def _rect_sum(
        prefix: torch.Tensor,
        *,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> int:
        if prefix.dim() != 2:
            raise ValueError(f"prefix must be [H+1,W+1], got {tuple(prefix.shape)}")
        stride = int(prefix.shape[1])
        flat = prefix.reshape(-1)
        idx00 = int(y0) * stride + int(x0)
        idx01 = int(y0) * stride + int(x1)
        idx10 = int(y1) * stride + int(x0)
        idx11 = int(y1) * stride + int(x1)
        return int((flat[idx11] - flat[idx01] - flat[idx10] + flat[idx00]).item())

    def _get_mask_linear_offsets(
        self,
        *,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        key = (
            int(body_mask.data_ptr()),
            int(clearance_mask.data_ptr()),
            int(pad_left),
            int(pad_bottom),
            int(self._W),
        )
        cached = self._mask_linear_offsets_cache.get(key, None)
        if cached is not None:
            return cached

        body_mask = body_mask.to(device=self._device, dtype=torch.bool)
        clearance_mask = clearance_mask.to(device=self._device, dtype=torch.bool)
        body_y, body_x = torch.where(body_mask)
        clear_y, clear_x = torch.where(clearance_mask)
        body_offsets = body_y.to(dtype=torch.long) * int(self._W) + body_x.to(dtype=torch.long)
        clear_offsets = clear_y.to(dtype=torch.long) * int(self._W) + clear_x.to(dtype=torch.long)

        out = (body_offsets, clear_offsets)
        self._mask_linear_offsets_cache[key] = out
        return out

    def _get_body_edge_linear_offsets(self, *, body_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return local edge indices for adjacent occupied body-cell pairs.

        - x_offsets index into flattened edge_x map [H, W-1]
        - y_offsets index into flattened edge_y map [H-1, W]
        """
        h = int(body_mask.shape[0])
        w = int(body_mask.shape[1])
        key = (
            int(body_mask.data_ptr()),
            h,
            w,
            int(self._W),
        )
        cached = self._body_edge_offsets_cache.get(key, None)
        if cached is not None:
            return cached

        body_mask = body_mask.to(device=self._device, dtype=torch.bool)
        if w > 1:
            h_pairs = body_mask[:, :-1] & body_mask[:, 1:]
            hy, hx = torch.where(h_pairs)
            x_offsets = hy.to(dtype=torch.long) * int(self._W - 1) + hx.to(dtype=torch.long)
        else:
            x_offsets = torch.empty((0,), dtype=torch.long, device=self._device)

        if h > 1:
            v_pairs = body_mask[:-1, :] & body_mask[1:, :]
            vy, vx = torch.where(v_pairs)
            y_offsets = vy.to(dtype=torch.long) * int(self._W) + vx.to(dtype=torch.long)
        else:
            y_offsets = torch.empty((0,), dtype=torch.long, device=self._device)

        out = (x_offsets, y_offsets)
        self._body_edge_offsets_cache[key] = out
        return out

    def _build_exclusive_edge_cache(self) -> None:
        active = [name for name, enabled in self._constraint_exclusive.items() if bool(enabled)]
        if not active:
            self._has_exclusive_edges = False
            self._exclusive_edge_x = None
            self._exclusive_edge_y = None
            self._exclusive_edge_x_ps = None
            self._exclusive_edge_y_ps = None
            return

        edge_x = torch.zeros((self._H, max(self._W - 1, 0)), dtype=torch.bool, device=self._device)
        edge_y = torch.zeros((max(self._H - 1, 0), self._W), dtype=torch.bool, device=self._device)
        for cname in active:
            id_map = self._constraint_id_maps.get(cname, None)
            if not isinstance(id_map, torch.Tensor):
                continue
            if self._W > 1:
                edge_x |= (id_map[:, 1:] != id_map[:, :-1])
            if self._H > 1:
                edge_y |= (id_map[1:, :] != id_map[:-1, :])

        self._exclusive_edge_x = edge_x
        self._exclusive_edge_y = edge_y
        self._exclusive_edge_x_ps = self._build_prefix(edge_x) if int(edge_x.shape[1]) > 0 else None
        self._exclusive_edge_y_ps = self._build_prefix(edge_y) if int(edge_y.shape[0]) > 0 else None
        self._has_exclusive_edges = bool(
            (int(edge_x.numel()) > 0 and bool(edge_x.any().item()))
            or (int(edge_y.numel()) > 0 and bool(edge_y.any().item()))
        )
        # TODO: If disconnected-body specs become common, add a class-moment
        # (XOR-generalized) fallback path for strict same-partition checks.

    def _ensure_static_prefix_cache(self) -> torch.Tensor:
        return self._static_invalid_ps

    def _ensure_runtime_prefix_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._occ_invalid_ps, self._clear_invalid_ps

    def _rebuild_runtime_prefix_cache(self) -> None:
        stacked = torch.stack([self.occ_invalid, self.clear_invalid], dim=0)
        prefix = self._build_prefix_batch(stacked)
        self._occ_invalid_ps = prefix[0]
        self._clear_invalid_ps = prefix[1]

    def _build_zone_invalid_cache(self) -> None:
        zone_invalid_by_gid: Dict[str | int, torch.Tensor] = {}
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

    def _get_zone_invalid(self, gid: str | int) -> torch.Tensor:
        if gid not in self._group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        z = self._zone_invalid_by_gid.get(gid, None)
        if isinstance(z, torch.Tensor):
            return z
        raise RuntimeError(
            f"zone invalid cache miss for gid={gid!r}; "
            "call bind_group_specs()/_build_zone_invalid_cache() before placement checks"
        )

    def _get_zone_invalid_ps(self, gid: str | int) -> torch.Tensor:
        if gid not in self._group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        z = self._zone_invalid_ps_by_gid.get(gid, None)
        if isinstance(z, torch.Tensor):
            return z
        raise RuntimeError(
            f"zone invalid prefix cache miss for gid={gid!r}; "
            "call bind_group_specs()/_build_zone_invalid_cache() before placement checks"
        )

    def _select_map_backend(self, *, gid: str | int) -> str:
        spec_type = type(self._group_specs[gid])
        return self._backends[("map", spec_type)]

    def _select_batch_backend(self, *, gid: str | int) -> str:
        spec_type = type(self._group_specs[gid])
        return self._backends[("batch", spec_type)]

    def placeable(
        self,
        *,
        gid: str | int,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> bool:
        body_mask = body_mask.to(device=self._device, dtype=torch.bool)
        clearance_mask = clearance_mask.to(device=self._device, dtype=torch.bool)
        backend = self._select_batch_backend(gid=gid)
        if backend == "prefixsum":
            result = self._is_placeable_prefixsum(
                gid=gid,
                x_bl=int(x_bl),
                y_bl=int(y_bl),
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
            )
        else:
            result = self._is_placeable_gather(
                gid=gid,
                x_bl=int(x_bl),
                y_bl=int(y_bl),
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
            )
        return result

    def _is_placeable_prefixsum(
        self,
        *,
        gid: str | int,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
    ) -> bool:
        bw = int(body_mask.shape[1])
        bh = int(body_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return False

        body_x0 = int(x_bl)
        body_y0 = int(y_bl)
        body_x1 = body_x0 + bw
        body_y1 = body_y0 + bh
        pad_x0 = body_x0 - pad_left
        pad_y0 = body_y0 - pad_bottom
        pad_x1 = pad_x0 + kw
        pad_y1 = pad_y0 + kh

        if (
            body_x0 < 0
            or body_y0 < 0
            or body_x1 > self._W
            or body_y1 > self._H
            or pad_x0 < 0
            or pad_y0 < 0
            or pad_x1 > self._W
            or pad_y1 > self._H
        ):
            return False

        static_ps = self._ensure_static_prefix_cache()
        occ_ps, clear_ps = self._ensure_runtime_prefix_cache()
        zone_ps = self._get_zone_invalid_ps(gid)

        body_hit = (
            self._rect_sum(static_ps, x0=body_x0, y0=body_y0, x1=body_x1, y1=body_y1)
            + self._rect_sum(occ_ps, x0=body_x0, y0=body_y0, x1=body_x1, y1=body_y1)
            + self._rect_sum(zone_ps, x0=body_x0, y0=body_y0, x1=body_x1, y1=body_y1)
        )
        if body_hit != 0:
            return False
        if self._has_exclusive_edges:
            edge_hit = 0
            if bw > 1 and isinstance(self._exclusive_edge_x_ps, torch.Tensor):
                edge_hit += self._rect_sum(
                    self._exclusive_edge_x_ps,
                    x0=body_x0,
                    y0=body_y0,
                    x1=body_x1 - 1,
                    y1=body_y1,
                )
            if bh > 1 and isinstance(self._exclusive_edge_y_ps, torch.Tensor):
                edge_hit += self._rect_sum(
                    self._exclusive_edge_y_ps,
                    x0=body_x0,
                    y0=body_y0,
                    x1=body_x1,
                    y1=body_y1 - 1,
                )
            if edge_hit != 0:
                return False
        body_clear_hit = self._rect_sum(clear_ps, x0=body_x0, y0=body_y0, x1=body_x1, y1=body_y1)
        if body_clear_hit != 0:
            return False
        pad_hit = (
            self._rect_sum(static_ps, x0=pad_x0, y0=pad_y0, x1=pad_x1, y1=pad_y1)
            + self._rect_sum(occ_ps, x0=pad_x0, y0=pad_y0, x1=pad_x1, y1=pad_y1)
            + self._rect_sum(zone_ps, x0=pad_x0, y0=pad_y0, x1=pad_x1, y1=pad_y1)
        )
        return pad_hit == 0

    def _is_placeable_gather(
        self,
        *,
        gid: str | int,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
    ) -> bool:
        bw = int(body_mask.shape[1])
        bh = int(body_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return False

        body_x0 = int(x_bl)
        body_y0 = int(y_bl)
        body_x1 = body_x0 + bw
        body_y1 = body_y0 + bh
        pad_x0 = body_x0 - pad_left
        pad_y0 = body_y0 - pad_bottom
        pad_x1 = pad_x0 + kw
        pad_y1 = pad_y0 + kh

        if (
            body_x0 < 0
            or body_y0 < 0
            or body_x1 > self._W
            or body_y1 > self._H
            or pad_x0 < 0
            or pad_y0 < 0
            or pad_x1 > self._W
            or pad_y1 > self._H
        ):
            return False

        base_body = int(body_y0) * int(self._W) + int(body_x0)
        base_clear = int(pad_y0) * int(self._W) + int(pad_x0)
        body_offsets, pad_offsets = self._get_mask_linear_offsets(
            body_mask=body_mask,
            clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
        )
        body_idx = base_body + body_offsets
        pad_idx = base_clear + pad_offsets

        static_f = self._static_invalid.reshape(-1)
        occ_f = self.occ_invalid.reshape(-1)
        zone_f = self._get_zone_invalid(gid).reshape(-1)
        clear_f = self.clear_invalid.reshape(-1)

        if bool(static_f[body_idx].any().item()) or bool(occ_f[body_idx].any().item()) or bool(zone_f[body_idx].any().item()):
            return False
        if self._has_exclusive_edges:
            x_offsets, y_offsets = self._get_body_edge_linear_offsets(body_mask=body_mask)
            if int(x_offsets.numel()) > 0 and isinstance(self._exclusive_edge_x, torch.Tensor):
                edge_x_f = self._exclusive_edge_x.reshape(-1)
                base_x = int(body_y0) * int(self._W - 1) + int(body_x0)
                if bool(edge_x_f[base_x + x_offsets].any().item()):
                    return False
            if int(y_offsets.numel()) > 0 and isinstance(self._exclusive_edge_y, torch.Tensor):
                edge_y_f = self._exclusive_edge_y.reshape(-1)
                base_y = int(body_y0) * int(self._W) + int(body_x0)
                if bool(edge_y_f[base_y + y_offsets].any().item()):
                    return False
        if bool(clear_f[body_idx].any().item()):
            return False
        if bool(static_f[pad_idx].any().item()) or bool(occ_f[pad_idx].any().item()) or bool(zone_f[pad_idx].any().item()):
            return False
        return True

    def _is_placeable_map_conv(
        self,
        *,
        gid: str | int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        valid_h: int,
        valid_w: int,
    ) -> torch.Tensor:
        invalid = self._static_invalid | self.occ_invalid | self._get_zone_invalid(gid)
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        body_kernel = body_mask.to(device=self._device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        clearance_kernel = clearance_mask.to(device=self._device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        inv_f = invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        clear_f = self.clear_invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        body_inv = F.conv2d(inv_f, body_kernel, padding=0).squeeze(0).squeeze(0)
        body_clear = F.conv2d(clear_f, body_kernel, padding=0).squeeze(0).squeeze(0)
        pad_inv = F.conv2d(inv_f, clearance_kernel, padding=0).squeeze(0).squeeze(0)
        body_inv_slice = body_inv[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
        body_clear_slice = body_clear[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
        if not self._has_exclusive_edges:
            return (body_inv_slice == 0) & (body_clear_slice == 0) & (pad_inv == 0)

        body_exclusive = torch.zeros_like(body_inv)
        if self._has_exclusive_edges:
            if int(body_mask.shape[1]) > 1 and isinstance(self._exclusive_edge_x, torch.Tensor):
                pair_x = (body_mask[:, :-1] & body_mask[:, 1:]).to(device=self._device, dtype=torch.float32)
                pair_x_kernel = pair_x.unsqueeze(0).unsqueeze(0)
                edge_x_f = self._exclusive_edge_x.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                body_exclusive = body_exclusive + F.conv2d(edge_x_f, pair_x_kernel, padding=0).squeeze(0).squeeze(0)
            if int(body_mask.shape[0]) > 1 and isinstance(self._exclusive_edge_y, torch.Tensor):
                pair_y = (body_mask[:-1, :] & body_mask[1:, :]).to(device=self._device, dtype=torch.float32)
                pair_y_kernel = pair_y.unsqueeze(0).unsqueeze(0)
                edge_y_f = self._exclusive_edge_y.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                body_exclusive = body_exclusive + F.conv2d(edge_y_f, pair_y_kernel, padding=0).squeeze(0).squeeze(0)
        body_exclusive_slice = body_exclusive[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
        return (body_inv_slice == 0) & (body_clear_slice == 0) & (body_exclusive_slice == 0) & (pad_inv == 0)

    def _is_placeable_map_prefixsum(
        self,
        *,
        gid: str | int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        valid_h: int,
        valid_w: int,
    ) -> torch.Tensor:
        static_ps = self._ensure_static_prefix_cache()
        occ_ps, clear_ps = self._ensure_runtime_prefix_cache()
        body_height = int(body_mask.shape[0])
        body_width = int(body_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        zone_ps = self._get_zone_invalid_ps(gid)

        pad_static = self._window_sum(static_ps, kh, kw)
        pad_occ = self._window_sum(occ_ps, kh, kw)
        pad_zone = self._window_sum(zone_ps, kh, kw)
        pad_hit = pad_static + pad_occ + pad_zone

        body_static = self._window_sum(static_ps, body_height, body_width)
        body_occ = self._window_sum(occ_ps, body_height, body_width)
        body_zone = self._window_sum(zone_ps, body_height, body_width)
        body_clear = self._window_sum(clear_ps, body_height, body_width)
        body_hit = (
            body_static[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
            + body_occ[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
            + body_zone[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
            + body_clear[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
        )
        if not self._has_exclusive_edges:
            return (pad_hit == 0) & (body_hit == 0)

        body_exclusive = torch.zeros_like(body_static)
        if self._has_exclusive_edges:
            if body_width > 1 and isinstance(self._exclusive_edge_x_ps, torch.Tensor):
                body_exclusive = body_exclusive + self._window_sum(
                    self._exclusive_edge_x_ps,
                    body_height,
                    body_width - 1,
                )
            if body_height > 1 and isinstance(self._exclusive_edge_y_ps, torch.Tensor):
                body_exclusive = body_exclusive + self._window_sum(
                    self._exclusive_edge_y_ps,
                    body_height - 1,
                    body_width,
                )
        body_hit = body_hit + body_exclusive[pad_bottom: pad_bottom + valid_h, pad_left: pad_left + valid_w]
        return (pad_hit == 0) & (body_hit == 0)

    def _is_placeable_batch_prefixsum(
        self,
        *,
        gid: str | int,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
    ) -> torch.Tensor:
        if x_bl.shape != y_bl.shape:
            raise ValueError(
                f"x_bl and y_bl must share shape, got {tuple(x_bl.shape)} vs {tuple(y_bl.shape)}"
            )

        result = torch.zeros_like(x_bl, dtype=torch.bool, device=self._device)
        if result.numel() == 0:
            return result

        bw = int(body_mask.shape[1])
        bh = int(body_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return result

        x_bl = x_bl.to(device=self._device, dtype=torch.long)
        y_bl = y_bl.to(device=self._device, dtype=torch.long)

        body_x0 = x_bl
        body_y0 = y_bl
        body_x1 = x_bl + bw
        body_y1 = y_bl + bh

        pad_x0 = x_bl - pad_left
        pad_y0 = y_bl - pad_bottom
        pad_x1 = pad_x0 + kw
        pad_y1 = pad_y0 + kh

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

        body_ok = (body_static + body_occ + body_zone == 0) & (body_clear == 0)
        if self._has_exclusive_edges:
            body_exclusive = torch.zeros_like(body_static)
            if bw > 1 and isinstance(self._exclusive_edge_x_ps, torch.Tensor):
                body_exclusive = body_exclusive + self._rect_sum_batch(
                    self._exclusive_edge_x_ps,
                    x0=body_x0[in_bounds],
                    y0=body_y0[in_bounds],
                    x1=body_x1[in_bounds] - 1,
                    y1=body_y1[in_bounds],
                )
            if bh > 1 and isinstance(self._exclusive_edge_y_ps, torch.Tensor):
                body_exclusive = body_exclusive + self._rect_sum_batch(
                    self._exclusive_edge_y_ps,
                    x0=body_x0[in_bounds],
                    y0=body_y0[in_bounds],
                    x1=body_x1[in_bounds],
                    y1=body_y1[in_bounds] - 1,
                )
            body_ok &= (body_exclusive == 0)
        result[in_bounds] = body_ok & (pad_static + pad_occ + pad_zone == 0)
        return result

    def _is_placeable_batch_gather(
        self,
        *,
        gid: str | int,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
    ) -> torch.Tensor:
        if x_bl.shape != y_bl.shape:
            raise ValueError(
                f"x_bl and y_bl must share shape, got {tuple(x_bl.shape)} vs {tuple(y_bl.shape)}"
            )

        result = torch.zeros(x_bl.shape, dtype=torch.bool, device=self._device)
        if result.numel() == 0:
            return result

        bw = int(body_mask.shape[1])
        bh = int(body_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])
        if bw <= 0 or bh <= 0 or kw <= 0 or kh <= 0:
            return result

        x_flat = x_bl.to(device=self._device, dtype=torch.long).reshape(-1)
        y_flat = y_bl.to(device=self._device, dtype=torch.long).reshape(-1)

        body_x0 = x_flat
        body_y0 = y_flat
        body_x1 = x_flat + bw
        body_y1 = y_flat + bh

        pad_x0 = x_flat - pad_left
        pad_y0 = y_flat - pad_bottom
        pad_x1 = pad_x0 + kw
        pad_y1 = pad_y0 + kh

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
        base_body = y_valid * int(self._W) + x_valid
        base_clear = (y_valid - pad_bottom) * int(self._W) + (x_valid - pad_left)

        body_offsets, pad_offsets = self._get_mask_linear_offsets(
            body_mask=body_mask,
            clearance_mask=clearance_mask,
            clearance_origin=clearance_origin,
        )
        static_f = self._static_invalid.reshape(-1)
        occ_f = self.occ_invalid.reshape(-1)
        zone_f = self._get_zone_invalid(gid).reshape(-1)
        clear_f = self.clear_invalid.reshape(-1)
        edge_x_offsets = torch.empty((0,), dtype=torch.long, device=self._device)
        edge_y_offsets = torch.empty((0,), dtype=torch.long, device=self._device)
        edge_x_f = None
        edge_y_f = None
        if self._has_exclusive_edges:
            edge_x_offsets, edge_y_offsets = self._get_body_edge_linear_offsets(body_mask=body_mask)
            if int(edge_x_offsets.numel()) > 0 and isinstance(self._exclusive_edge_x, torch.Tensor):
                edge_x_f = self._exclusive_edge_x.reshape(-1)
            if int(edge_y_offsets.numel()) > 0 and isinstance(self._exclusive_edge_y, torch.Tensor):
                edge_y_f = self._exclusive_edge_y.reshape(-1)

        offset_count = max(
            int(body_offsets.numel()),
            int(pad_offsets.numel()),
            int(edge_x_offsets.numel()),
            int(edge_y_offsets.numel()),
            1,
        )
        chunk_size = max(1, 1_048_576 // offset_count)
        flat_result = result.reshape(-1)

        for start in range(0, int(valid_idx.numel()), chunk_size):
            stop = min(start + chunk_size, int(valid_idx.numel()))
            body_idx = base_body[start:stop, None] + body_offsets[None, :]
            pad_idx = base_clear[start:stop, None] + pad_offsets[None, :]

            body_hit = static_f[body_idx].any(dim=1)
            body_hit |= occ_f[body_idx].any(dim=1)
            body_hit |= zone_f[body_idx].any(dim=1)
            body_clear_hit = clear_f[body_idx].any(dim=1)

            pad_hit = static_f[pad_idx].any(dim=1)
            pad_hit |= occ_f[pad_idx].any(dim=1)
            pad_hit |= zone_f[pad_idx].any(dim=1)
            edge_hit = torch.zeros((stop - start,), dtype=torch.bool, device=self._device)
            if edge_x_f is not None and int(edge_x_offsets.numel()) > 0:
                base_edge_x = y_valid[start:stop] * int(self._W - 1) + x_valid[start:stop]
                edge_x_idx = base_edge_x[:, None] + edge_x_offsets[None, :]
                edge_hit |= edge_x_f[edge_x_idx].any(dim=1)
            if edge_y_f is not None and int(edge_y_offsets.numel()) > 0:
                base_edge_y = y_valid[start:stop] * int(self._W) + x_valid[start:stop]
                edge_y_idx = base_edge_y[:, None] + edge_y_offsets[None, :]
                edge_hit |= edge_y_f[edge_y_idx].any(dim=1)

            flat_result[valid_idx[start:stop]] = (~body_hit) & (~body_clear_hit) & (~pad_hit) & (~edge_hit)
        return result

    def placeable_batch(
        self,
        *,
        gid: str | int,
        x_bl: torch.Tensor,
        y_bl: torch.Tensor,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> torch.Tensor:
        """Vectorized placeability check. Returns [N] bool."""
        backend = self._select_batch_backend(gid=gid)
        if backend == "prefixsum":
            result = self._is_placeable_batch_prefixsum(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
            )
        else:
            result = self._is_placeable_batch_gather(
                gid=gid,
                x_bl=x_bl,
                y_bl=y_bl,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
            )
        return result

    def placeable_map(
        self,
        *,
        gid: str | int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
    ) -> torch.Tensor:
        body_mask = body_mask.to(device=self._device, dtype=torch.bool)
        clearance_mask = clearance_mask.to(device=self._device, dtype=torch.bool)
        h = int(body_mask.shape[0])
        w = int(body_mask.shape[1])
        kh = int(clearance_mask.shape[0])
        kw = int(clearance_mask.shape[1])
        pad_left = int(clearance_origin[0])
        pad_bottom = int(clearance_origin[1])

        H, W = self.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self._device)
        if h <= 0 or w <= 0 or kh <= 0 or kw <= 0:
            return result
        if kh > H or kw > W:
            return result

        valid_h = H - kh + 1
        valid_w = W - kw + 1
        if valid_h <= 0 or valid_w <= 0:
            return result

        backend = self._select_map_backend(gid=gid)
        if backend == "prefixsum":
            valid_mask = self._is_placeable_map_prefixsum(
                gid=gid,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                valid_h=valid_h,
                valid_w=valid_w,
            )
        else:
            valid_mask = self._is_placeable_map_conv(
                gid=gid,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                valid_h=valid_h,
                valid_w=valid_w,
            )

        result[pad_bottom:pad_bottom + valid_h, pad_left:pad_left + valid_w] = valid_mask
        return result

    def bind_group_specs(self, group_specs: Dict[str | int, GroupSpec]) -> None:
        self._group_specs = group_specs
        self._build_zone_invalid_cache()
        if self._backend_selection == "benchmark":
            self._resolve_backends_benchmark()
        else:
            self._resolve_backends_static()

    # ------------------------------------------------------------------
    # Backend resolve (static / benchmark)
    # ------------------------------------------------------------------

    _STATIC_BACKEND_DEFAULTS: Dict[str, Dict[str, str]] = {
        "StaticRectSpec": {"map_cpu": "prefixsum", "map_cuda": "conv", "batch": "prefixsum"},
        "StaticIrregularSpec": {"map_cpu": "conv", "map_cuda": "conv", "batch": "gather"},
    }

    # Valid backend candidates per spec class.
    # prefixsum assumes body fills entire bounding rect — unsafe for irregular shapes.
    _BACKEND_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
        "StaticRectSpec": {"map": ["conv", "prefixsum"], "batch": ["prefixsum", "gather"]},
        "StaticIrregularSpec": {"map": ["conv"], "batch": ["gather"]},
    }

    def _discover_spec_types(self) -> Dict[type, Tuple[str | int, GroupSpec]]:
        """Collect unique spec class types with a representative (gid, spec) each."""
        found: Dict[type, Tuple[str | int, GroupSpec]] = {}
        for gid, spec in self._group_specs.items():
            st = type(spec)
            if st not in found:
                found[st] = (gid, spec)
        return found

    def _resolve_backends_static(self) -> None:
        spec_types = self._discover_spec_types()
        logger.info("=== collision backend selection (mode=static) ===")
        logger.info("  device=%s  grid=%dx%d", self._device.type, self._H, self._W)
        logger.info("  scanning group specs for shape types ...")
        for st, (gid, _spec) in spec_types.items():
            logger.info("    found %-24s (e.g. gid=%r)", st.__name__, gid)

        logger.info("  using predefined backend rules (no benchmark)")

        map_key = "map_cuda" if self._device.type == "cuda" else "map_cpu"
        self._backends = {}
        for st in spec_types:
            name = st.__name__
            defaults = self._STATIC_BACKEND_DEFAULTS.get(name)
            if defaults is None:
                raise ValueError(
                    f"no static backend defaults for spec type {name!r}; "
                    "add an entry to _STATIC_BACKEND_DEFAULTS or use backend_selection='benchmark'"
                )
            self._backends[("map", st)] = defaults[map_key]
            self._backends[("batch", st)] = defaults["batch"]

        self._log_selected_backends()
        logger.info("collision backend selection done")

    def _get_candidates(self, spec_type: type) -> Dict[str, List[str]]:
        """Return valid backend candidates for a spec class type."""
        name = spec_type.__name__
        candidates = self._BACKEND_CANDIDATES.get(name)
        if candidates is not None:
            return candidates
        # Unknown spec type: fall back to safest backends only.
        return {"map": ["conv"], "batch": ["gather"]}

    def _resolve_backends_benchmark(self) -> None:
        spec_types = self._discover_spec_types()
        warmup = 3
        rounds = 10

        logger.info("=== collision backend selection (mode=benchmark) ===")
        logger.info("  device=%s  grid=%dx%d", self._device.type, self._H, self._W)
        logger.info("  scanning group specs for shape types ...")
        for st, (gid, _spec) in spec_types.items():
            logger.info("    found %-24s (e.g. gid=%r)", st.__name__, gid)

        logger.info(
            "  benchmarking each backend per shape type (warmup=%d, rounds=%d) ...",
            warmup, rounds,
        )

        # results: {(op, spec_type): {backend: (mean, std)}}
        results: Dict[Tuple[str, type], Dict[str, Tuple[float, float]]] = {}
        self._backends = {}

        for st, (gid, spec) in spec_types.items():
            candidates = self._get_candidates(st)
            map_candidates = candidates["map"]
            batch_candidates = candidates["batch"]

            body_mask, clearance_mask, clearance_origin, _is_rect = self._get_bench_tensors(spec)
            kh, kw = int(clearance_mask.shape[0]), int(clearance_mask.shape[1])
            valid_h = self._H - kh + 1
            valid_w = self._W - kw + 1

            # --- map backends ---
            map_results: Dict[str, Tuple[float, float]] = {}
            if len(map_candidates) == 1:
                # Single valid backend — no benchmark needed.
                self._backends[("map", st)] = map_candidates[0]
            elif valid_h > 0 and valid_w > 0:
                for backend in map_candidates:
                    times = self._bench_map(
                        gid=gid,
                        body_mask=body_mask,
                        clearance_mask=clearance_mask,
                        clearance_origin=clearance_origin,
                        valid_h=valid_h,
                        valid_w=valid_w,
                        backend=backend,
                        warmup=warmup,
                        rounds=rounds,
                    )
                    map_results[backend] = (float(np.mean(times)), float(np.std(times)))
            results[("map", st)] = map_results

            # --- batch backends ---
            batch_results: Dict[str, Tuple[float, float]] = {}
            if len(batch_candidates) == 1:
                # Single valid backend — no benchmark needed.
                self._backends[("batch", st)] = batch_candidates[0]
            else:
                n_candidates = min(valid_h * valid_w, 2000) if valid_h > 0 and valid_w > 0 else 100
                for backend in batch_candidates:
                    times = self._bench_batch(
                        gid=gid,
                        body_mask=body_mask,
                        clearance_mask=clearance_mask,
                        clearance_origin=clearance_origin,
                        n_candidates=n_candidates,
                        backend=backend,
                        warmup=warmup,
                        rounds=rounds,
                    )
                    batch_results[backend] = (float(np.mean(times)), float(np.std(times)))
            results[("batch", st)] = batch_results

        # Log results and select best
        logger.info("  benchmark results - mean (+-std) ms:")
        for (op, st), backend_times in sorted(results.items(), key=lambda x: (x[0][0], x[0][1].__name__)):
            if not backend_times:
                # Single candidate — already assigned, no benchmark data.
                chosen = self._backends.get((op, st), "?")
                logger.info("    %-5s %-24s | %s (only valid backend)", op, st.__name__, chosen)
                continue
            parts = []
            for b, (mean, std) in backend_times.items():
                parts.append(f"{b} {mean:.3f} (+-{std:.3f})")
            logger.info("    %-5s %-24s | %s", op, st.__name__, " | ".join(parts))
            best = min(backend_times, key=lambda b: backend_times[b][0])
            self._backends[(op, st)] = best

        self._log_selected_backends()
        logger.info("collision backend selection done")

    def _log_selected_backends(self) -> None:
        logger.info("  selected collision backends:")
        for (op, st) in sorted(self._backends, key=lambda k: (k[0], k[1].__name__)):
            logger.info("    %-5s %-24s | %s", op, st.__name__, self._backends[(op, st)])

    def _get_bench_tensors(
        self, spec: GroupSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], bool]:
        """Get a representative (body_mask, clearance_mask, clearance_origin, is_rectangular) from spec."""
        vi = spec.variants[0]
        shape_key = getattr(vi, "shape_key", None)
        if shape_key is not None:
            return spec.shape_tensors(shape_key)
        # Fallback: create dummy rect tensors
        w = getattr(spec, "width", 10)
        h = getattr(spec, "height", 10)
        body = torch.ones((int(h), int(w)), dtype=torch.bool, device=self._device)
        return body, body.clone(), (0, 0), True

    def _bench_map(
        self,
        *,
        gid: str | int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        valid_h: int,
        valid_w: int,
        backend: str,
        warmup: int,
        rounds: int,
    ) -> List[float]:
        fn = self._is_placeable_map_conv if backend == "conv" else self._is_placeable_map_prefixsum
        kwargs = dict(
            gid=gid,
            body_mask=body_mask.to(device=self._device, dtype=torch.bool),
            clearance_mask=clearance_mask.to(device=self._device, dtype=torch.bool),
            clearance_origin=clearance_origin,
            valid_h=valid_h,
            valid_w=valid_w,
        )
        is_cuda = self._device.type == "cuda"
        for _ in range(warmup):
            fn(**kwargs)
            if is_cuda:
                torch.cuda.synchronize()
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn(**kwargs)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    def _bench_batch(
        self,
        *,
        gid: str | int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        n_candidates: int,
        backend: str,
        warmup: int,
        rounds: int,
    ) -> List[float]:
        fn = self._is_placeable_batch_prefixsum if backend == "prefixsum" else self._is_placeable_batch_gather
        # Generate random valid positions within grid
        bh, bw = int(body_mask.shape[0]), int(body_mask.shape[1])
        pad_left, pad_bottom = int(clearance_origin[0]), int(clearance_origin[1])
        kh, kw = int(clearance_mask.shape[0]), int(clearance_mask.shape[1])
        max_x = max(1, self._W - bw - (kw - bw) + 1)
        max_y = max(1, self._H - bh - (kh - bh) + 1)
        x_bl = torch.randint(pad_left, max(pad_left + 1, max_x), (n_candidates,), device=self._device)
        y_bl = torch.randint(pad_bottom, max(pad_bottom + 1, max_y), (n_candidates,), device=self._device)
        kwargs = dict(
            gid=gid,
            x_bl=x_bl,
            y_bl=y_bl,
            body_mask=body_mask.to(device=self._device, dtype=torch.bool),
            clearance_mask=clearance_mask.to(device=self._device, dtype=torch.bool),
            clearance_origin=clearance_origin,
        )
        is_cuda = self._device.type == "cuda"
        for _ in range(warmup):
            fn(**kwargs)
            if is_cuda:
                torch.cuda.synchronize()
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn(**kwargs)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    def copy(self) -> "GridMaps":
        out = object.__new__(GridMaps)
        out._H = self._H
        out._W = self._W
        out._device = self._device

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
        out._constraint_exclusive = self._constraint_exclusive
        out._constraint_id_maps = self._constraint_id_maps
        out._group_specs = self._group_specs
        out._zone_invalid_by_gid = self._zone_invalid_by_gid
        out._static_invalid_ps = self._static_invalid_ps
        out._zone_invalid_ps_by_gid = self._zone_invalid_ps_by_gid
        out._mask_linear_offsets_cache = self._mask_linear_offsets_cache
        out._body_edge_offsets_cache = self._body_edge_offsets_cache
        out._has_exclusive_edges = self._has_exclusive_edges
        out._exclusive_edge_x = self._exclusive_edge_x
        out._exclusive_edge_y = self._exclusive_edge_y
        out._exclusive_edge_x_ps = self._exclusive_edge_x_ps
        out._exclusive_edge_y_ps = self._exclusive_edge_y_ps
        out._occ_invalid_ps = self._occ_invalid_ps
        out._clear_invalid_ps = self._clear_invalid_ps
        out._backend_selection = self._backend_selection
        out._backends = self._backends
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
        self._occ_invalid_ps = src._occ_invalid_ps
        self._clear_invalid_ps = src._clear_invalid_ps

    def reset_runtime(self) -> None:
        self.occ_invalid.zero_()
        self.clear_invalid.zero_()
        self.has_bbox = False
        self.bbox_min_x = 0.0
        self.bbox_max_x = 0.0
        self.bbox_min_y = 0.0
        self.bbox_max_y = 0.0
        self._rebuild_runtime_prefix_cache()

    def _paint_mask(
        self,
        *,
        dst: torch.Tensor,
        mask: torch.Tensor,
        x0: int,
        y0: int,
        is_rectangular: bool,
    ) -> None:
        mask = mask.to(device=self._device, dtype=torch.bool)
        h = int(mask.shape[0])
        w = int(mask.shape[1])
        if h <= 0 or w <= 0:
            return
        x1 = int(x0) + w
        y1 = int(y0) + h
        cx0 = max(0, min(self._W, int(x0)))
        cy0 = max(0, min(self._H, int(y0)))
        cx1 = max(0, min(self._W, int(x1)))
        cy1 = max(0, min(self._H, int(y1)))
        if cx0 >= cx1 or cy0 >= cy1:
            return
        mx0 = cx0 - int(x0)
        my0 = cy0 - int(y0)
        mx1 = mx0 + (cx1 - cx0)
        my1 = my0 + (cy1 - cy0)
        if bool(is_rectangular):
            dst[cy0:cy1, cx0:cx1] = True
            return
        dst[cy0:cy1, cx0:cx1] |= mask[my0:my1, mx0:mx1]

    def paint_placement(
        self,
        *,
        bbox_min_x: float,
        bbox_max_x: float,
        bbox_min_y: float,
        bbox_max_y: float,
        x_bl: int,
        y_bl: int,
        body_mask: torch.Tensor,
        clearance_mask: torch.Tensor,
        clearance_origin: Tuple[int, int],
        is_rectangular: bool,
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

        self._paint_mask(
            dst=self.occ_invalid,
            mask=body_mask,
            x0=int(x_bl),
            y0=int(y_bl),
            is_rectangular=is_rectangular,
        )
        self._paint_mask(
            dst=self.clear_invalid,
            mask=clearance_mask,
            x0=int(x_bl) - int(clearance_origin[0]),
            y0=int(y_bl) - int(clearance_origin[1]),
            is_rectangular=is_rectangular,
        )
        self._rebuild_runtime_prefix_cache()

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
        forbidden: List[Dict[str, Any]],
    ) -> torch.Tensor:
        inv = torch.zeros((H, W), dtype=torch.bool, device=device)
        for i, area in enumerate(forbidden):
            if not isinstance(area, dict):
                continue
            area_mask = GridMaps._build_area_mask(
                H=H,
                W=W,
                device=device,
                area=area,
                path=f"forbidden[{i}]",
            )
            inv |= area_mask
        return inv

    @staticmethod
    def _build_area_mask(
        *,
        H: int,
        W: int,
        device: torch.device,
        area: Dict[str, Any],
        path: str,
    ) -> torch.Tensor:
        """Rasterize one area config into a bool mask over the full grid."""
        mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        if "shape_type" not in area:
            raise ValueError(f"{path}.shape_type is required ('rect' or 'irregular')")
        area_type = str(area.get("shape_type", "")).strip().lower()
        if area_type == "rect":
            rect = area.get("rect", None)
            if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
                raise ValueError(f"{path}.rect must be [x0,y0,x1,y1]")
            try:
                x0 = max(0, min(W, int(float(rect[0]))))
                x1 = max(0, min(W, int(float(rect[2]))))
                y0 = max(0, min(H, int(float(rect[1]))))
                y1 = max(0, min(H, int(float(rect[3]))))
            except Exception as e:
                raise ValueError(f"{path}.rect must contain numeric coordinates") from e
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
            return mask

        if area_type == "irregular":
            polygon = area.get("polygon", None)
            if not (isinstance(polygon, (list, tuple)) and len(polygon) >= 3):
                raise ValueError(f"{path}.polygon must be [[x,y], ...] with at least 3 points")

            points: list[tuple[float, float]] = []
            for pi, p in enumerate(polygon):
                if not (isinstance(p, (list, tuple)) and len(p) == 2):
                    raise ValueError(f"{path}.polygon[{pi}] must be [x,y]")
                try:
                    points.append((float(p[0]), float(p[1])))
                except Exception as e:
                    raise ValueError(f"{path}.polygon[{pi}] must contain numeric x,y") from e

            verts = np.asarray(points, dtype=np.float64)
            x_min = max(0, min(W, int(np.floor(float(verts[:, 0].min())))))
            x_max = max(0, min(W, int(np.ceil(float(verts[:, 0].max())))))
            y_min = max(0, min(H, int(np.floor(float(verts[:, 1].min())))))
            y_max = max(0, min(H, int(np.ceil(float(verts[:, 1].max())))))
            if x_max <= x_min or y_max <= y_min:
                return mask

            from matplotlib.path import Path as MplPath

            xs = np.arange(x_min, x_max, dtype=np.float64) + 0.5
            ys = np.arange(y_min, y_max, dtype=np.float64) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            inside = MplPath(verts).contains_points(pts).reshape(y_max - y_min, x_max - x_min)
            if np.any(inside):
                mask[y_min:y_max, x_min:x_max] = torch.as_tensor(
                    inside, dtype=torch.bool, device=device
                )
            return mask

        raise ValueError(f"{path}.shape_type must be 'rect' or 'irregular', got {area_type!r}")

    def _build_constraint_maps(
        self,
        H: int,
        W: int,
        device: torch.device,
        constraints: Dict[str, Dict[str, Any]],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, str],
        Dict[str, str],
        Dict[str, bool],
        Dict[str, torch.Tensor],
    ]:
        maps: Dict[str, torch.Tensor] = {}
        ops: Dict[str, str] = {}
        dtypes: Dict[str, str] = {}
        exclusive: Dict[str, bool] = {}
        id_maps: Dict[str, torch.Tensor] = {}

        for cname, cfg in constraints.items():
            if not isinstance(cfg, dict):
                continue
            dtype = str(cfg.get("dtype", "float")).lower()
            op = str(cfg.get("op", "=="))
            areas = cfg.get("areas", [])
            ex_raw = cfg.get("exclusive", False)
            if isinstance(ex_raw, str):
                ex = ex_raw.strip().lower()
                if ex in {"true", "1", "yes", "on", "body"}:
                    ex_enabled = True
                elif ex in {"false", "0", "no", "off", "none"}:
                    ex_enabled = False
                else:
                    raise ValueError(f"invalid exclusive for constraint {cname!r}: {ex_raw!r}")
            else:
                ex_enabled = bool(ex_raw)
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

            default_raw = cfg["default"]
            default_id = cfg.get("default_id", None)
            if isinstance(default_raw, dict):
                if "value" not in default_raw:
                    raise ValueError(f"constraint {cname!r}.default object must contain key 'value'")
                if default_id is None:
                    default_id = default_raw.get("id", None)
                default_value = default_raw["value"]
            else:
                default_value = default_raw
            if ex_enabled and default_id is None:
                raise ValueError(
                    f"constraint {cname!r}: exclusive=true requires default id "
                    f"(use default={{'value':..., 'id':...}} or default_id)"
                )
            default_v = self._coerce_constraint_value(
                default_value,
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
            id_map = None
            id_code: Dict[str, int] = {}
            if ex_enabled:
                id_map = torch.zeros((H, W), dtype=torch.int64, device=device)
                default_label = self._coerce_constraint_id(default_id, cname=str(cname), ctx="default")
                id_code[default_label] = 0
                id_map.fill_(0)

            for i, area in enumerate(areas):
                if not isinstance(area, dict):
                    continue
                value = area.get("value", None)
                if value is None:
                    continue
                area_mask = self._build_area_mask(
                    H=H,
                    W=W,
                    device=device,
                    area=area,
                    path=f"constraints.{cname}.areas[{i}]",
                )
                if not bool(area_mask.any().item()):
                    continue
                v = self._coerce_constraint_value(
                    value,
                    dtype=dtype,
                    cname=str(cname),
                    ctx="area",
                )
                m[area_mask] = v
                if ex_enabled:
                    area_id = area.get("id", None)
                    if area_id is None:
                        raise ValueError(
                            f"constraint {cname!r}.areas requires key 'id' when exclusive=true"
                        )
                    area_label = self._coerce_constraint_id(area_id, cname=str(cname), ctx="area")
                    if area_label not in id_code:
                        id_code[area_label] = int(len(id_code))
                    id_map[area_mask] = int(id_code[area_label])

            name = str(cname)
            maps[name] = m
            ops[name] = op
            dtypes[name] = dtype
            exclusive[name] = bool(ex_enabled)
            if ex_enabled and isinstance(id_map, torch.Tensor):
                id_maps[name] = id_map

        return maps, ops, dtypes, exclusive, id_maps

    @staticmethod
    def _coerce_constraint_id(v: Any, *, cname: str, ctx: str) -> str:
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                raise ValueError(f"{ctx} id for {cname!r} must not be empty")
            return s
        if isinstance(v, (int, float, bool)):
            return str(v)
        raise ValueError(f"{ctx} id for {cname!r} must be scalar (str/int/float/bool), got {v!r}")

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
