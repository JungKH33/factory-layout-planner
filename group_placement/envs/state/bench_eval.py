"""Benchmark: old cost-dict (8f8d78b) vs new EvalState.

Compares the actual data-structure operations used in EnvState copy/restore/update.

OLD baseline (8f8d78b):
  state.cost = {
      "base": {"total": X, "flow": X, "area": X, ...},
      "terminal": {"delta": {name: X, ...}, "total": X},
      "final": {"total": X},
      "finalized": bool,
  }
  copy   → _clone_cost_dict (shallow dict ops)
  restore → assign cloned dict
  refresh → mutate cs["base"], cs["terminal"], cs["final"] directly

NEW current:
  state.eval = EvalState(reward_scale)
  copy    → EvalState.copy()
  restore → EvalState.restore()
  update  → record_base_delta() / set_base_snapshot()
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .eval import EvalState

# ---------------------------------------------------------------------------
# Old baseline helpers (reproduced from 8f8d78b)
# ---------------------------------------------------------------------------

def _old_empty_cost() -> Dict[str, Any]:
    return {
        "base": {"total": 0.0},
        "terminal": {"delta": {}, "total": 0.0},
        "final": {"total": 0.0},
        "finalized": False,
    }


def _old_clone_cost(src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(src, dict):
        return _old_empty_cost()
    base = dict(src.get("base", {}) or {})
    terminal = dict(src.get("terminal", {}) or {})
    terminal_delta = dict(terminal.get("delta", {}) or {})
    terminal["delta"] = terminal_delta
    final = dict(src.get("final", {}) or {})
    finalized = bool(src.get("finalized", False))
    return {"base": base, "terminal": terminal, "final": final, "finalized": finalized}


def _old_refresh_base(cs: Dict[str, Any], scores: Dict[str, float]) -> None:
    """Equivalent to env._refresh_base_cost_snapshot (dict ops only, no reward recompute)."""
    base: Dict[str, float] = {str(k): float(v) for k, v in scores.items()}
    if "total" not in base:
        base["total"] = sum(v for k, v in base.items() if k != "total")
    cs["base"] = base
    cs["terminal"] = {"delta": {}, "total": 0.0}
    cs["final"] = {"total": float(base["total"])}
    cs["finalized"] = False


def _old_finalize_terminal(cs: Dict[str, Any], delta: Dict[str, float]) -> None:
    base_total = float(cs.get("base", {}).get("total", 0.0))
    delta_norm = {str(k): float(v) for k, v in delta.items()}
    delta_total = sum(delta_norm.values())
    cs["terminal"] = {"delta": delta_norm, "total": delta_total}
    cs["final"] = {"total": base_total + delta_total}
    cs["finalized"] = True


def _old_recompute(cs: Dict[str, Any]) -> float:
    base_total = float(cs.get("base", {}).get("total", 0.0))
    terminal_total = float(cs.get("terminal", {}).get("total", 0.0))
    return base_total + terminal_total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_COMPS_SIMPLE = 2   # flow + area
N_COMPS_RICH   = 25  # many reward components


def _make_old_simple() -> Dict[str, Any]:
    cs = _old_empty_cost()
    _old_refresh_base(cs, {"flow": 80.0, "area": 20.0})
    return cs


def _make_old_rich() -> Dict[str, Any]:
    cs = _old_empty_cost()
    scores = {f"comp_{i}": float(i * 3) for i in range(N_COMPS_RICH)}
    _old_refresh_base(cs, scores)
    delta = {f"delta_{i}": float(i * 0.5) for i in range(N_COMPS_RICH)}
    _old_finalize_terminal(cs, delta)
    return cs


def _make_new_simple() -> EvalState:
    ev = EvalState.empty(100.0)
    ev.record_base_delta(
        reward_delta_by_name={"flow": 80.0, "area": 20.0},
        reward_weights_by_name={"flow": 1.0, "area": 1.0},
    )
    return ev


def _make_new_rich() -> EvalState:
    ev = EvalState.empty(100.0)
    names = [f"comp_{i}" for i in range(N_COMPS_RICH)]
    deltas = {n: float(i * 3) for i, n in enumerate(names)}
    weights = {n: 1.0 for n in names}
    ev.record_base_delta(reward_delta_by_name=deltas, reward_weights_by_name=weights)
    terminal = {f"delta_{i}": {"delta_cost": float(i * 0.5)} for i in range(N_COMPS_RICH)}
    ev.set_terminal_snapshot(terminal)
    return ev


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench(label: str, fn, n: int = 10_000) -> float:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    us = elapsed / n * 1e6
    print(f"  {label:<45s} {us:8.2f} us")
    return us


def run() -> None:
    N = 10_000

    old_s = _make_old_simple()
    old_r = _make_old_rich()
    new_s = _make_new_simple()
    new_r = _make_new_rich()

    print(f"\nEvalState benchmark vs old cost-dict  ({N:,} iterations)")
    print("=" * 70)

    # ---- copy ----
    print("\n[copy]")
    t_old_copy_s  = bench("old  copy (simple)", lambda: _old_clone_cost(old_s), N)
    t_new_copy_s  = bench("new  copy (simple)", lambda: new_s.copy(),            N)
    t_old_copy_r  = bench(f"old  copy (rich/{N_COMPS_RICH} comps)", lambda: _old_clone_cost(old_r), N)
    t_new_copy_r  = bench(f"new  copy (rich/{N_COMPS_RICH} comps)", lambda: new_r.copy(),            N)

    # ---- restore ----
    print("\n[restore]")
    snap_old_s = _old_clone_cost(old_s)
    snap_old_r = _old_clone_cost(old_r)
    snap_new_s = new_s.copy()
    snap_new_r = new_r.copy()

    dst_new_s = new_s.copy()
    dst_new_r = new_r.copy()

    def old_restore_s():
        nonlocal old_s
        old_s = _old_clone_cost(snap_old_s)  # old: overwrite reference

    def old_restore_r():
        nonlocal old_r
        old_r = _old_clone_cost(snap_old_r)

    t_old_rst_s = bench("old  restore (simple)", old_restore_s, N)
    t_new_rst_s = bench("new  restore (simple)", lambda: dst_new_s.restore(snap_new_s), N)
    t_old_rst_r = bench(f"old  restore (rich/{N_COMPS_RICH} comps)", old_restore_r, N)
    t_new_rst_r = bench(f"new  restore (rich/{N_COMPS_RICH} comps)", lambda: dst_new_r.restore(snap_new_r), N)

    # ---- base update ----
    print("\n[base update / refresh]")
    scores_s = {"flow": 80.0, "area": 20.0}
    scores_r = {f"comp_{i}": float(i * 3) for i in range(N_COMPS_RICH)}
    weights_r = {n: 1.0 for n in scores_r}

    tmp_old_s = _old_empty_cost()
    tmp_old_r = _old_empty_cost()
    tmp_new_s = EvalState.empty(100.0)
    tmp_new_r = EvalState.empty(100.0)

    t_old_ref_s = bench("old  refresh_base (simple)", lambda: _old_refresh_base(tmp_old_s, scores_s), N)
    t_new_ref_s = bench("new  record_base_delta (simple)",
                        lambda: tmp_new_s.record_base_delta(
                            reward_delta_by_name=scores_s,
                            reward_weights_by_name={"flow": 1.0, "area": 1.0}), N)
    t_old_ref_r = bench(f"old  refresh_base (rich/{N_COMPS_RICH} comps)", lambda: _old_refresh_base(tmp_old_r, scores_r), N)
    t_new_ref_r = bench(f"new  record_base_delta (rich/{N_COMPS_RICH} comps, +weights)",
                        lambda: tmp_new_r.record_base_delta(
                            reward_delta_by_name=scores_r,
                            reward_weights_by_name=weights_r), N)
    tmp_new_r2 = _make_new_rich()
    bench(f"new  record_base_delta (rich/{N_COMPS_RICH} comps, no weights) [PROD PATH]",
          lambda: tmp_new_r2.record_base_delta(reward_delta_by_name=scores_r), N)

    # ---- recompute objective ----
    print("\n[recompute_objective]")
    bench("old  recompute (simple)", lambda: _old_recompute(old_s), N)
    bench("new  recompute_objective (simple)", lambda: new_s.recompute_objective(finalized=False), N)
    bench(f"old  recompute (rich/{N_COMPS_RICH} comps)", lambda: _old_recompute(old_r), N)
    bench(f"new  recompute_objective (rich/{N_COMPS_RICH} comps)", lambda: new_r.recompute_objective(finalized=False), N)

    # ---- summary table ----
    print("\n" + "=" * 70)
    print("Summary (speedup = old / new):")
    print(f"  copy simple:   old={t_old_copy_s:.2f}us  new={t_new_copy_s:.2f}us  "
          f"  {'NEW' if t_new_copy_s < t_old_copy_s else 'OLD'} faster "
          f"x{max(t_old_copy_s,t_new_copy_s)/min(t_old_copy_s,t_new_copy_s):.1f}")
    print(f"  copy rich:     old={t_old_copy_r:.2f}us  new={t_new_copy_r:.2f}us  "
          f"  {'NEW' if t_new_copy_r < t_old_copy_r else 'OLD'} faster "
          f"x{max(t_old_copy_r,t_new_copy_r)/min(t_old_copy_r,t_new_copy_r):.1f}")
    print(f"  restore simple:old={t_old_rst_s:.2f}us  new={t_new_rst_s:.2f}us  "
          f"  {'NEW' if t_new_rst_s < t_old_rst_s else 'OLD'} faster "
          f"x{max(t_old_rst_s,t_new_rst_s)/min(t_old_rst_s,t_new_rst_s):.1f}")
    print(f"  restore rich:  old={t_old_rst_r:.2f}us  new={t_new_rst_r:.2f}us  "
          f"  {'NEW' if t_new_rst_r < t_old_rst_r else 'OLD'} faster "
          f"x{max(t_old_rst_r,t_new_rst_r)/min(t_old_rst_r,t_new_rst_r):.1f}")


if __name__ == "__main__":
    run()
