from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..action import GroupId

FlowPortPair = Tuple[Tuple[float, float], Tuple[float, float]]
FlowPortPairs = Dict[Tuple[GroupId, GroupId], List[FlowPortPair]]


class FlowGraph:
    """Runtime flow helper state.

    - `_group_flow` is shared/static input.
    - `flow_port_pairs` and score/delta tensor caches are runtime fields.
    """

    def __init__(self, group_flow: Dict[GroupId, Dict[GroupId, float]], *, device: torch.device) -> None:
        self._group_flow = group_flow
        self._device = torch.device(device)
        self.flow_port_pairs: FlowPortPairs = {}
        self._flow_port_pairs_nodes_key: Tuple[GroupId, ...] = tuple()
        self._io_nodes_key: Tuple[GroupId, ...] = tuple()
        self._row_by_gid: Dict[GroupId, int] = {}
        self.placed_entries = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
        self.placed_exits = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
        self.placed_entries_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
        self.placed_exits_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
        self._flow_w_cache = torch.empty((0, 0), dtype=torch.float32, device=self._device)
        self._flow_w_nodes_key: Tuple[GroupId, ...] = tuple()
        self.delta_w_out = torch.empty((0,), dtype=torch.float32, device=self._device)
        self.delta_w_in = torch.empty((0,), dtype=torch.float32, device=self._device)
        self._delta_key: Tuple[Optional[GroupId], Tuple[GroupId, ...]] = (None, tuple())

    def copy(self) -> "FlowGraph":
        out = object.__new__(FlowGraph)
        out._group_flow = self._group_flow
        out._device = self._device
        out.flow_port_pairs = dict(self.flow_port_pairs)
        out._flow_port_pairs_nodes_key = tuple(self._flow_port_pairs_nodes_key)
        out._io_nodes_key = tuple(self._io_nodes_key)
        out._row_by_gid = dict(self._row_by_gid)
        out.placed_entries = self.placed_entries.clone()
        out.placed_exits = self.placed_exits.clone()
        out.placed_entries_mask = self.placed_entries_mask.clone()
        out.placed_exits_mask = self.placed_exits_mask.clone()
        out._flow_w_cache = self._flow_w_cache.clone()
        out._flow_w_nodes_key = self._flow_w_nodes_key
        out.delta_w_out = self.delta_w_out.clone()
        out.delta_w_in = self.delta_w_in.clone()
        out._delta_key = self._delta_key
        return out

    def restore(self, src: "FlowGraph") -> None:
        """In-place restore of runtime flow fields."""
        if not isinstance(src, FlowGraph):
            raise TypeError(f"src must be FlowGraph, got {type(src).__name__}")
        self.flow_port_pairs.clear()
        self.flow_port_pairs.update(src.flow_port_pairs)
        self._flow_port_pairs_nodes_key = tuple(src._flow_port_pairs_nodes_key)
        self._io_nodes_key = tuple(src._io_nodes_key)
        self._row_by_gid = dict(src._row_by_gid)
        src_entries = src.placed_entries.to(device=self._device, dtype=torch.float32)
        self.placed_entries.resize_(src_entries.shape)
        self.placed_entries.copy_(src_entries)
        src_exits = src.placed_exits.to(device=self._device, dtype=torch.float32)
        self.placed_exits.resize_(src_exits.shape)
        self.placed_exits.copy_(src_exits)
        src_entries_mask = src.placed_entries_mask.to(device=self._device, dtype=torch.bool)
        self.placed_entries_mask.resize_(src_entries_mask.shape)
        self.placed_entries_mask.copy_(src_entries_mask)
        src_exits_mask = src.placed_exits_mask.to(device=self._device, dtype=torch.bool)
        self.placed_exits_mask.resize_(src_exits_mask.shape)
        self.placed_exits_mask.copy_(src_exits_mask)
        src_flow_w = src._flow_w_cache.to(device=self._device, dtype=torch.float32)
        self._flow_w_cache.resize_(src_flow_w.shape)
        self._flow_w_cache.copy_(src_flow_w)
        self._flow_w_nodes_key = tuple(src._flow_w_nodes_key)
        src_w_out = src.delta_w_out.to(device=self._device, dtype=torch.float32)
        self.delta_w_out.resize_(src_w_out.shape)
        self.delta_w_out.copy_(src_w_out)
        src_w_in = src.delta_w_in.to(device=self._device, dtype=torch.float32)
        self.delta_w_in.resize_(src_w_in.shape)
        self.delta_w_in.copy_(src_w_in)
        self._delta_key = src._delta_key

    def _invalidate_flow_cache(self) -> None:
        self._flow_w_nodes_key = tuple()
        self._flow_w_cache = torch.empty((0, 0), dtype=torch.float32, device=self._device)

    def _invalidate_delta_cache(self) -> None:
        self._delta_key = (None, tuple())
        self.delta_w_out = torch.empty((0,), dtype=torch.float32, device=self._device)
        self.delta_w_in = torch.empty((0,), dtype=torch.float32, device=self._device)

    def invalidate_on_nodes_changed(self) -> None:
        # flow/delta tensors are keyed by (gid,nodes) and remain valid for the old key.
        # Keep them so append paths can reuse previous tensors incrementally.
        return

    def clear_flow_port_pairs(self) -> None:
        self.flow_port_pairs = {}
        self._flow_port_pairs_nodes_key = tuple()

    def set_flow_port_pairs(
        self,
        pairs: FlowPortPairs,
        *,
        nodes: Optional[List[GroupId]] = None,
    ) -> None:
        self.flow_port_pairs = dict(pairs)
        self._flow_port_pairs_nodes_key = tuple(nodes) if nodes is not None else tuple()

    @property
    def flow_port_pairs_nodes_key(self) -> Tuple[GroupId, ...]:
        return self._flow_port_pairs_nodes_key

    def reset_runtime(self) -> None:
        self.clear_flow_port_pairs()
        self._io_nodes_key = tuple()
        self._row_by_gid = {}
        self.placed_entries = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
        self.placed_exits = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
        self.placed_entries_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
        self.placed_exits_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
        self._invalidate_flow_cache()
        self._invalidate_delta_cache()

    @staticmethod
    def _ports_from_placement(placement: object, *, kind: str) -> List[Tuple[float, float]]:
        x_center = float(getattr(placement, "x_center", 0.0))
        y_center = float(getattr(placement, "y_center", 0.0))
        fallback = (x_center, y_center)
        src = getattr(placement, kind, None)
        if torch.is_tensor(src):
            t = src.to(dtype=torch.float32).view(-1, 2)
            if int(t.shape[0]) <= 0:
                return [fallback]
            return [(float(t[k, 0].item()), float(t[k, 1].item())) for k in range(int(t.shape[0]))]
        if isinstance(src, (list, tuple)) and len(src) > 0:
            return [(float(pt[0]), float(pt[1])) for pt in src]
        return [fallback]

    def _ensure_io_capacity(self, *, rows: int, emax: int, xmax: int) -> None:
        cur_rows = int(self.placed_entries.shape[0])
        cur_emax = int(self.placed_entries.shape[1]) if cur_rows > 0 else int(self.placed_entries_mask.shape[1])
        cur_xmax = int(self.placed_exits.shape[1]) if cur_rows > 0 else int(self.placed_exits_mask.shape[1])
        tgt_rows = max(int(rows), cur_rows)
        tgt_emax = max(int(emax), max(1, cur_emax))
        tgt_xmax = max(int(xmax), max(1, cur_xmax))
        if cur_rows == tgt_rows and cur_emax == tgt_emax and cur_xmax == tgt_xmax:
            return

        new_entries = torch.zeros((tgt_rows, tgt_emax, 2), dtype=torch.float32, device=self._device)
        new_exits = torch.zeros((tgt_rows, tgt_xmax, 2), dtype=torch.float32, device=self._device)
        new_entries_mask = torch.zeros((tgt_rows, tgt_emax), dtype=torch.bool, device=self._device)
        new_exits_mask = torch.zeros((tgt_rows, tgt_xmax), dtype=torch.bool, device=self._device)

        if cur_rows > 0:
            new_entries[:cur_rows, :cur_emax] = self.placed_entries[:, :cur_emax]
            new_exits[:cur_rows, :cur_xmax] = self.placed_exits[:, :cur_xmax]
            new_entries_mask[:cur_rows, :cur_emax] = self.placed_entries_mask[:, :cur_emax]
            new_exits_mask[:cur_rows, :cur_xmax] = self.placed_exits_mask[:, :cur_xmax]

        self.placed_entries = new_entries
        self.placed_exits = new_exits
        self.placed_entries_mask = new_entries_mask
        self.placed_exits_mask = new_exits_mask

    def _reindex_io_to_nodes(self, nodes: List[GroupId]) -> None:
        nodes_key = tuple(nodes)
        if nodes_key == self._io_nodes_key:
            return
        if len(nodes_key) == 0:
            self._io_nodes_key = tuple()
            self._row_by_gid = {}
            self.placed_entries = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
            self.placed_exits = torch.empty((0, 1, 2), dtype=torch.float32, device=self._device)
            self.placed_entries_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
            self.placed_exits_mask = torch.zeros((0, 1), dtype=torch.bool, device=self._device)
            return
        idx: List[int] = []
        for gid in nodes_key:
            row = self._row_by_gid.get(gid, None)
            if row is None:
                raise KeyError(f"io row missing for gid={gid!r}")
            idx.append(int(row))
        rows = torch.tensor(idx, dtype=torch.long, device=self._device)
        self.placed_entries = self.placed_entries.index_select(0, rows)
        self.placed_exits = self.placed_exits.index_select(0, rows)
        self.placed_entries_mask = self.placed_entries_mask.index_select(0, rows)
        self.placed_exits_mask = self.placed_exits_mask.index_select(0, rows)
        self._row_by_gid = {gid: i for i, gid in enumerate(nodes_key)}
        self._io_nodes_key = nodes_key

    def upsert_io(self, *, placement: object, nodes: List[GroupId]) -> None:
        gid = placement.group_id
        nodes_key = tuple(nodes)
        if gid not in nodes_key:
            raise ValueError(f"upsert_io: gid={gid!r} not in nodes")
        entry_points = self._ports_from_placement(placement, kind="entry_points")
        exit_points = self._ports_from_placement(placement, kind="exit_points")

        row = self._row_by_gid.get(gid, None)
        if row is None:
            row = len(self._row_by_gid)
            self._row_by_gid[gid] = int(row)
        self._ensure_io_capacity(rows=len(nodes_key), emax=len(entry_points), xmax=len(exit_points))

        r = int(row)
        self.placed_entries[r].zero_()
        self.placed_exits[r].zero_()
        self.placed_entries_mask[r].zero_()
        self.placed_exits_mask[r].zero_()
        for j, (x, y) in enumerate(entry_points):
            self.placed_entries[r, j, 0] = float(x)
            self.placed_entries[r, j, 1] = float(y)
            self.placed_entries_mask[r, j] = True
        for j, (x, y) in enumerate(exit_points):
            self.placed_exits[r, j, 0] = float(x)
            self.placed_exits[r, j, 1] = float(y)
            self.placed_exits_mask[r, j] = True
        if nodes_key == self._io_nodes_key:
            return

        prev_key = self._io_nodes_key
        # Fast path: placed_nodes_order only appends in normal env progression.
        if (
            len(nodes_key) == len(prev_key) + 1
            and nodes_key[:-1] == prev_key
            and nodes_key[-1] == gid
            and int(self._row_by_gid.get(gid, -1)) == len(prev_key)
        ):
            self._io_nodes_key = nodes_key
            return
        self._reindex_io_to_nodes(list(nodes_key))

    def io_tensors(self, nodes: List[GroupId]) -> Tuple[List[GroupId], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._reindex_io_to_nodes(nodes)
        return (
            list(nodes),
            self.placed_entries,
            self.placed_exits,
            self.placed_entries_mask,
            self.placed_exits_mask,
        )

    def build_flow_w(self, nodes: List[GroupId]) -> torch.Tensor:
        nodes_key = tuple(nodes)
        if self._flow_w_nodes_key == nodes_key:
            return self._flow_w_cache

        prev_key = self._flow_w_nodes_key
        prev_n = len(prev_key)
        if (
            len(nodes_key) == prev_n + 1
            and nodes_key[:-1] == prev_key
            and int(self._flow_w_cache.shape[0]) == prev_n
        ):
            p = len(nodes_key)
            flow_w = torch.zeros((p, p), dtype=torch.float32, device=self._device)
            if prev_n > 0:
                flow_w[:prev_n, :prev_n] = self._flow_w_cache
            new_gid = nodes_key[-1]
            out_edges_new = self._group_flow.get(new_gid, {})
            for j, dst in enumerate(nodes_key):
                flow_w[p - 1, j] = float(out_edges_new.get(dst, 0.0))
            for i, src in enumerate(prev_key):
                flow_w[i, p - 1] = float(self._group_flow.get(src, {}).get(new_gid, 0.0))
            self._flow_w_nodes_key = nodes_key
            self._flow_w_cache = flow_w
            return flow_w

        p = len(nodes)
        flow_w = torch.zeros((p, p), dtype=torch.float32, device=self._device)
        for i, src in enumerate(nodes):
            out_edges = self._group_flow.get(src, {})
            for j, dst in enumerate(nodes):
                flow_w[i, j] = float(out_edges.get(dst, 0.0))
        self._flow_w_nodes_key = nodes_key
        self._flow_w_cache = flow_w
        return flow_w

    def build_delta_flow_weights(
        self,
        current_gid: Optional[GroupId],
        nodes: List[GroupId],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes_key = tuple(nodes)
        key = (current_gid, nodes_key)
        if self._delta_key == key:
            return self.delta_w_out, self.delta_w_in

        prev_gid, prev_nodes_key = self._delta_key
        prev_n = len(prev_nodes_key)
        if (
            prev_gid == current_gid
            and len(nodes_key) == prev_n + 1
            and nodes_key[:-1] == prev_nodes_key
            and int(self.delta_w_out.shape[0]) == prev_n
            and int(self.delta_w_in.shape[0]) == prev_n
        ):
            p = len(nodes_key)
            w_out = torch.zeros((p,), dtype=torch.float32, device=self._device)
            w_in = torch.zeros((p,), dtype=torch.float32, device=self._device)
            if prev_n > 0:
                w_out[:prev_n] = self.delta_w_out
                w_in[:prev_n] = self.delta_w_in
            if current_gid is not None:
                new_gid = nodes_key[-1]
                out_edges = self._group_flow.get(current_gid, {})
                w_out[p - 1] = float(out_edges.get(new_gid, 0.0))
                w_in[p - 1] = float(self._group_flow.get(new_gid, {}).get(current_gid, 0.0))
            self.delta_w_out = w_out
            self.delta_w_in = w_in
            self._delta_key = key
            return w_out, w_in

        p = len(nodes)
        w_out = torch.zeros((p,), dtype=torch.float32, device=self._device)
        w_in = torch.zeros((p,), dtype=torch.float32, device=self._device)
        if current_gid is not None:
            out_edges = self._group_flow.get(current_gid, {})
            for i, pgid in enumerate(nodes):
                w_out[i] = float(out_edges.get(pgid, 0.0))
                w_in[i] = float(self._group_flow.get(pgid, {}).get(current_gid, 0.0))

        self.delta_w_out = w_out
        self.delta_w_in = w_in
        self._delta_key = key
        return w_out, w_in
