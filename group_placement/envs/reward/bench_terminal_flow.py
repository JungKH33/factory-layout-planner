"""Benchmark: TerminalFlowReward.exact_flow_cost overhead from recent changes.

Measures three paths against an inline "before" baseline:
  (A) return_metadata=False       — has_valid fix only (torch.where on small [m,t])
  (B) return_metadata=True span=1 — return_detail=True + metadata loop, single pair
  (C) return_metadata=True span>1 — same + multi-pair polylines

Run:
    python -m group_placement.envs.reward.bench_terminal_flow
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch

ROOT = Path(__file__).resolve().parents[3]
CFGS = {
    "multiport (5 nodes, span>1, 300x300)": ROOT / "group_placement/envs/env_configs/multiport_01.json",
    "basic_20 (20 nodes, span=1, 800x800)":  ROOT / "group_placement/envs/env_configs/basic_20.json",
}


# ---------------------------------------------------------------------------
# Inline "before" baseline (pre-fix exact_flow_cost metadata path)
# ---------------------------------------------------------------------------

def _before_metadata_loop(
    *,
    m: int,
    t: int,
    flow_w_cpu: torch.Tensor,
    reduced_mt: torch.Tensor,
    has_valid_cpu: torch.Tensor,
    c_idx_cpu: Optional[torch.Tensor],
    p_idx_cpu: Optional[torch.Tensor],
    target_uid_cpu: torch.Tensor,
    exits_clamped_cpu: torch.Tensor,
    dist_batch: torch.Tensor,
    placed_nodes: list,
    unreachable_thresh: float,
    h: int,
    w: int,
    backtrack_fn,
) -> dict:
    from group_placement.envs.reward.flow import FlowReward
    edges: dict = {}
    for i in range(m):
        for j in range(t):
            fij = float(flow_w_cpu[i, j].item())
            if fij == 0.0:
                continue
            if not bool(has_valid_cpu[i, j].item()):
                continue
            dist_val = float(reduced_mt[i, j].item())
            if dist_val >= unreachable_thresh:
                continue
            c_best = int(c_idx_cpu[i, j].item()) if c_idx_cpu is not None else 0
            p_best = int(p_idx_cpu[i, j].item()) if p_idx_cpu is not None else 0
            src_gid = str(placed_nodes[i])
            dst_gid = str(placed_nodes[j])
            terminal_model: dict = {"pair_indices": [[c_best, p_best]]}
            if c_idx_cpu is not None and p_idx_cpu is not None:
                uid = int(target_uid_cpu[j, p_best].item())
                if uid >= 0 and uid < int(dist_batch.shape[0]):
                    ex_x = int(exits_clamped_cpu[i, c_best, 0].item())
                    ex_y = int(exits_clamped_cpu[i, c_best, 1].item())
                    pl = backtrack_fn(dist_batch[uid], (ex_x, ex_y), h, w)
                    terminal_model["polylines"] = [pl]
            edges[f"{src_gid}->{dst_gid}"] = {
                "weight": fij, "distance": dist_val,
                "pair_count": 1, "models": {"terminal": terminal_model},
            }
    return edges


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _auto_place(env, loaded) -> None:
    """Place all facilities with the greedy adapter."""
    from group_placement.agents.placement.greedy.adapter_v3 import GreedyV3Adapter
    from group_placement.agents.placement.greedy.agent import GreedyAgent

    adapter = GreedyV3Adapter(k=80)
    agent = GreedyAgent()
    adapter.bind(env)

    env.reset(**loaded.reset_kwargs)
    while env.get_state().remaining:
        obs = adapter.build_observation()
        action_space = adapter.build_action_space()
        if int(action_space.valid_mask.sum()) == 0:
            break
        action_idx = agent.select_action(obs=obs, action_space=action_space)
        placement = adapter.resolve_action(action_idx, action_space)
        env.step(placement)


def _build_terminal_reward(loaded):
    """Extract or create a TerminalFlowReward from the loaded env."""
    from group_placement.envs.reward.terminal import TerminalFlowReward
    env = loaded.env
    # Prefer what's configured on the env
    tc = getattr(env, "terminal_reward_composer", None)
    if tc is not None:
        for name, comp in tc.components.items():
            if isinstance(comp, TerminalFlowReward):
                return comp
    # Fallback: plain TerminalFlowReward (span from group_specs if available)
    group_specs = getattr(env, "_group_specs", None) or getattr(env, "group_specs", None)
    return TerminalFlowReward(group_specs=group_specs)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench(label: str, fn, n: int = 200, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"  {label:<55s} {ms:8.3f} ms")
    return ms


def run_scenario(cfg_label: str, cfg_path: Path, n: int) -> None:
    from group_placement.envs.env_loader import load_env
    from group_placement.envs.reward.flow import FlowReward

    print(f"\n{'='*72}")
    print(f"  {cfg_label}")
    print(f"{'='*72}")

    loaded = load_env(str(cfg_path), device=torch.device("cpu"))
    env = loaded.env
    _auto_place(env, loaded)

    if not env.get_state().placed:
        print("  [SKIP] no facilities placed")
        return

    state = env.get_state()
    maps = env.get_maps()
    tfr = _build_terminal_reward(loaded)

    # ---------- pre-compute shared wavefront (reused across bench calls) ----------
    # We benchmark exact_flow_cost repeatedly; it recomputes wavefront each call.
    # That's the real cost — we don't cache it.

    n_placed = len(state.placed)
    n_flow   = int((env.get_state().build_flow_w() != 0).sum().item())
    print(f"  placed={n_placed}  active_flow_edges={n_flow}")

    # ---- (A) return_metadata=False ----
    t_no_meta = bench(
        f"(A) return_metadata=False",
        lambda: tfr.exact_flow_cost(state=state, maps=maps, return_metadata=False),
        n=n,
    )

    # ---- (B) return_metadata=True, current code ----
    t_meta_new = bench(
        f"(B) return_metadata=True  [new: return_detail+multi-polyline]",
        lambda: tfr.exact_flow_cost(state=state, maps=maps, return_metadata=True),
        n=n,
    )

    # ---- (C) "before" baseline: return_detail=False + single-pair loop ----
    # Replicate the pre-fix call: no return_detail, no multi-pair.
    # We call _masked_pair_reduce directly, then run the old metadata loop.
    (
        placed_nodes, placed_entries, placed_exits,
        placed_entries_mask, placed_exits_mask,
    ) = state.io_tensors()
    device = placed_entries.device

    # Build cost tensor once (the wavefront part) — we exclude it from the
    # baseline timing so we're comparing apples to apples (metadata loop only).
    result_full = tfr.exact_flow_cost(state=state, maps=maps, return_metadata=True)
    # We can't easily split the old vs new metadata loop without re-running
    # the full wavefront; instead, time the *full* call with the old loop inlined.

    def _before_full():
        """Replicate the pre-change exact_flow_cost with single-pair metadata."""
        from group_placement.envs.reward.terminal import (
            _wavefront_distance_field_batched,
            _wavefront_distance_field,
            TerminalFlowReward,
        )
        # --- same setup as exact_flow_cost ---
        (pn, pe, px, pem, pxm) = state.io_tensors()
        dev = pe.device
        fw = state.build_flow_w().to(dev, torch.float32)
        h_g = int(maps.static_invalid.shape[0])
        w_g = int(maps.static_invalid.shape[1])
        blocked = maps.static_invalid.to(dev, torch.bool) | maps.occ_invalid.to(dev, torch.bool)
        walkable = ~blocked
        _, anch_en, en_valid = tfr._anchor_ports_by_group(
            state=state, placed_nodes=pn, ports_xy=pe,
            ports_mask=pem, grid_h=h_g, grid_w=w_g,
        )
        _, anch_ex, ex_valid = tfr._anchor_ports_by_group(
            state=state, placed_nodes=pn, ports_xy=px,
            ports_mask=pxm, grid_h=h_g, grid_w=w_g,
        )
        if bool(en_valid.any().item()) or bool(ex_valid.any().item()):
            all_p = torch.cat([anch_en[en_valid], anch_ex[ex_valid]], dim=0)
            ppx, ppy = all_p[:, 0], all_p[:, 1]
            inb = (ppx >= 0) & (ppx < w_g) & (ppy >= 0) & (ppy < h_g)
            if bool(inb.any().item()):
                walkable[ppy[inb], ppx[inb]] = True
        t_dim = int(anch_en.shape[0]); p_dim = int(anch_en.shape[1])
        m_dim = int(anch_ex.shape[0]); c_dim = int(anch_ex.shape[1])
        if m_dim == 0 or c_dim == 0 or t_dim == 0 or p_dim == 0:
            return
        target_uid = torch.full((t_dim, p_dim), -1, dtype=torch.long, device=dev)
        tgt_cells = anch_en[en_valid]
        if int(tgt_cells.numel()) == 0:
            return
        unique_tgts, inverse = torch.unique(tgt_cells, dim=0, return_inverse=True)
        target_uid[en_valid] = inverse
        chunk = max(1, int(tfr.batch_chunk_size))
        parts = []
        for s in range(0, int(unique_tgts.shape[0]), chunk):
            part = unique_tgts[s:s+chunk]
            mm = int(part.shape[0])
            parts.append(_wavefront_distance_field_batched(
                free_map=walkable,
                seeds_xy=part.view(mm, 1, 2),
                seeds_mask=torch.ones((mm, 1), dtype=torch.bool, device=dev),
                max_iters=int(tfr.max_wave_iters),
            ))
        dist_batch = torch.cat(parts, dim=0) if parts else torch.empty((0, h_g, w_g), dtype=torch.int32, device=dev)
        anch_ex_cl = anch_ex.clamp_min(0)
        anch_ex_cl[..., 0] = anch_ex_cl[..., 0].clamp_max(w_g - 1)
        anch_ex_cl[..., 1] = anch_ex_cl[..., 1].clamp_max(h_g - 1)
        src_x = anch_ex_cl[..., 0]; src_y = anch_ex_cl[..., 1]
        raw_en = torch.round(pe).to(torch.long)
        raw_ex = torch.round(px).to(torch.long)
        ex_extra = ((raw_ex[..., 0]-anch_ex[..., 0]).abs()+(raw_ex[..., 1]-anch_ex[..., 1]).abs()).float()
        en_extra = ((raw_en[..., 0]-anch_en[..., 0]).abs()+(raw_en[..., 1]-anch_en[..., 1]).abs()).float()
        valid_mask = ex_valid[:, None, :, None] & en_valid[None, :, None, :]
        cost = torch.full((m_dim, t_dim, c_dim, p_dim), float(tfr.unreachable_cost), dtype=torch.float32, device=dev)
        tp_valid = en_valid & (target_uid >= 0)
        tp_idx = torch.nonzero(tp_valid, as_tuple=False)
        if int(tp_idx.shape[0]) > 0:
            tp_ti = tp_idx[:, 0]; tp_pj = tp_idx[:, 1]
            tp_uid = target_uid[tp_ti, tp_pj]
            gd = dist_batch[tp_uid][:, src_y, src_x].float()
            gd = torch.where(gd >= 0.0, gd, torch.full_like(gd, float(tfr.unreachable_cost)))
            gd = gd + ex_extra.unsqueeze(0) + en_extra[tp_ti, tp_pj].view(-1, 1, 1)
            for k in range(int(tp_idx.shape[0])):
                cost[:, int(tp_ti[k]), :, int(tp_pj[k])] = gd[k]
        # OLD: no return_detail, no has_valid fix
        reduced_mt, c_idx, p_idx = FlowReward._masked_pair_reduce(cost, valid_mask, None, None)
        # OLD: total without has_valid fix
        total = (reduced_mt * fw).sum()  # noqa
        # OLD: metadata loop (single pair, no multi-polyline)
        hv = valid_mask.any(dim=3).any(dim=2)
        uth = float(tfr.unreachable_cost) * 0.9
        fw_cpu = fw.cpu(); red_cpu = reduced_mt.cpu(); hv_cpu = hv.cpu()
        ci_cpu = c_idx.cpu(); pi_cpu = p_idx.cpu()
        tuid_cpu = target_uid.cpu(); exc_cpu = anch_ex_cl.cpu()
        for i in range(m_dim):
            for j in range(t_dim):
                if float(fw_cpu[i, j]) == 0.0: continue
                if not bool(hv_cpu[i, j]): continue
                if float(red_cpu[i, j]) >= uth: continue
                c_b = int(ci_cpu[i, j]); p_b = int(pi_cpu[i, j])
                uid_v = int(tuid_cpu[j, p_b])
                if uid_v >= 0 and uid_v < int(dist_batch.shape[0]):
                    ex_x = int(exc_cpu[i, c_b, 0]); ex_y = int(exc_cpu[i, c_b, 1])
                    tfr._backtrack_polyline(dist_batch[uid_v], (ex_x, ex_y), h_g, w_g)

    t_meta_before = bench(
        f"(C) return_metadata=True  [before: single-pair, no return_detail]",
        _before_full,
        n=n,
    )

    print()
    overhead_no_meta = (t_no_meta / t_meta_new * 100) if t_meta_new > 0 else 0
    overhead_vs_before = ((t_meta_new - t_meta_before) / t_meta_before * 100) if t_meta_before > 0 else 0
    print(f"  (A) vs (B)  no-meta is {t_no_meta/t_meta_new*100:.0f}% of meta cost")
    print(f"  (B) vs (C)  new metadata is {overhead_vs_before:+.1f}% vs old metadata")


def run() -> None:
    N_SMALL = 50
    N_LARGE = 20

    print("\nTerminalFlowReward.exact_flow_cost  overhead benchmark")

    for label, path in CFGS.items():
        if not path.exists():
            print(f"\n[SKIP] {label}: {path} not found")
            continue
        n = N_SMALL if "multiport" in str(path) else N_LARGE
        run_scenario(label, path, n=n)

    print()


if __name__ == "__main__":
    run()
