from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalState:
    """Runtime evaluation snapshot owned by EnvState.

    Base and terminal evaluation are stored separately.
    This prevents stale terminal artifacts (e.g., failure penalty) from leaking
    into future non-terminal states.
    """

    reward_scale: float
    layout_rev: int = 0
    objective: Dict[str, Any] = field(default_factory=dict)
    base_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    terminal_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def empty(cls, reward_scale: float) -> "EvalState":
        rs = float(reward_scale)
        if rs <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {rs}")
        return cls(
            reward_scale=rs,
            layout_rev=0,
            objective={
                "base_total": 0.0,
                "terminal_total": 0.0,
                "cost_total": 0.0,
                "reward_total": 0.0,
                "reward_scale": rs,
                "finalized": False,
            },
            base_components={},
            terminal_components={},
        )

    def copy(self) -> "EvalState":
        return EvalState(
            reward_scale=float(self.reward_scale),
            layout_rev=int(self.layout_rev),
            objective=deepcopy(self.objective),
            base_components=deepcopy(self.base_components),
            terminal_components=deepcopy(self.terminal_components),
        )

    def restore(self, src: "EvalState") -> None:
        if not isinstance(src, EvalState):
            raise TypeError(f"src must be EvalState, got {type(src).__name__}")
        self.reward_scale = float(src.reward_scale)
        self.layout_rev = int(src.layout_rev)
        self.objective = deepcopy(src.objective)
        self.base_components = deepcopy(src.base_components)
        self.terminal_components = deepcopy(src.terminal_components)

    def reset_runtime(self) -> None:
        self.layout_rev = 0
        self.objective = {
            "base_total": 0.0,
            "terminal_total": 0.0,
            "cost_total": 0.0,
            "reward_total": 0.0,
            "reward_scale": float(self.reward_scale),
            "finalized": False,
        }
        self.base_components = {}
        self.terminal_components = {}

    @staticmethod
    def _mark_stale_in_meta(meta: Any, *, layout_rev: int) -> None:
        if isinstance(meta, dict):
            if "status" in meta and "layout_rev" in meta and meta["status"] == "ok":
                meta["status"] = "stale"
                meta["layout_rev"] = int(layout_rev)
            for v in meta.values():
                EvalState._mark_stale_in_meta(v, layout_rev=layout_rev)
            return
        if isinstance(meta, list):
            for v in meta:
                EvalState._mark_stale_in_meta(v, layout_rev=layout_rev)

    def begin_layout_revision(self) -> None:
        self.layout_rev = int(self.layout_rev) + 1
        self.objective["finalized"] = False
        for rec in self.base_components.values():
            meta = rec.get("meta", None)
            self._mark_stale_in_meta(meta, layout_rev=self.layout_rev)
        # New placement invalidates all terminal deltas by definition.
        self.terminal_components = {}
        self.recompute_objective(finalized=False)

    @staticmethod
    def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if (
                isinstance(v, dict)
                and isinstance(dst.get(k), dict)
            ):
                EvalState._deep_merge_dict(dst[k], v)
            else:
                dst[k] = deepcopy(v)

    @staticmethod
    def _parse_edge_key(raw_key: object) -> Optional[Tuple[str, str]]:
        if isinstance(raw_key, str) and "->" in raw_key:
            lhs, rhs = raw_key.split("->", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs and rhs:
                return lhs, rhs
            return None
        if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
            return str(raw_key[0]), str(raw_key[1])
        return None

    @staticmethod
    def _extract_model_port_pairs(edge_meta: Dict[str, Any], *, model_priority: Tuple[str, ...]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        models = edge_meta.get("models", None)
        if not isinstance(models, dict):
            return []
        raw_pairs: object = None
        for model_name in model_priority:
            model = models.get(str(model_name), None)
            if isinstance(model, dict) and isinstance(model.get("port_pairs", None), list):
                raw_pairs = model["port_pairs"]
                break
        if raw_pairs is None:
            return []

        out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for row in raw_pairs:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                continue
            ex, en = row
            if not isinstance(ex, (list, tuple)) or not isinstance(en, (list, tuple)):
                continue
            if len(ex) != 2 or len(en) != 2:
                continue
            out.append((
                (float(ex[0]), float(ex[1])),
                (float(en[0]), float(en[1])),
            ))
        return out

    def set_base_snapshot(self, components: Dict[str, Dict[str, Any]]) -> None:
        out: Dict[str, Dict[str, Any]] = {}
        for name, rec_in in dict(components or {}).items():
            rec = dict(rec_in or {})
            raw = float(rec.get("raw_cost", 0.0))
            weight = float(rec.get("weight", 1.0))
            out[str(name)] = {
                "raw_cost": raw,
                "weight": weight,
                "weighted_cost": raw * weight,
                "meta": dict(rec.get("meta", {}) or {}),
            }
        self.base_components = out

    def set_terminal_snapshot(self, components: Dict[str, Dict[str, Any]]) -> None:
        out: Dict[str, Dict[str, Any]] = {}
        for name, rec_in in dict(components or {}).items():
            rec = dict(rec_in or {})
            out[str(name)] = {
                "delta_cost": float(rec.get("delta_cost", 0.0)),
                "meta": dict(rec.get("meta", {}) or {}),
            }
        self.terminal_components = out

    def clear_terminal(self) -> None:
        self.terminal_components = {}
        self.recompute_objective(finalized=False)

    def merge_component_meta(
        self,
        *,
        name: str,
        patch: Dict[str, Any],
        phase: str = "base",
    ) -> None:
        phase_key = str(phase)
        if phase_key == "base":
            target = self.base_components
            defaults: Dict[str, Any] = {
                "raw_cost": 0.0,
                "weight": 1.0,
                "weighted_cost": 0.0,
                "meta": {},
            }
        elif phase_key == "terminal":
            target = self.terminal_components
            defaults = {
                "delta_cost": 0.0,
                "meta": {},
            }
        else:
            raise ValueError(f"phase must be 'base' or 'terminal', got {phase!r}")
        key = str(name)
        rec = target.get(key)
        if not isinstance(rec, dict):
            rec = dict(defaults)
            target[key] = rec
        cur = rec.get("meta", None)
        if not isinstance(cur, dict):
            cur = {}
            rec["meta"] = cur
        self._deep_merge_dict(cur, dict(patch or {}))

    def edge_metadata(self, *, phase: str = "base") -> Dict[Tuple[str, str], Dict[str, Any]]:
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        phase_key = str(phase)
        if phase_key == "base":
            sources = [("base", self.base_components)]
        elif phase_key == "terminal":
            sources = [("terminal", self.terminal_components)]
        elif phase_key == "merged":
            sources = [
                ("base", self.base_components),
                ("terminal", self.terminal_components),
            ]
        else:
            raise ValueError(f"phase must be 'base'|'terminal'|'merged', got {phase!r}")

        for phase_name, comp_map in sources:
            for comp_name, rec in comp_map.items():
                if not isinstance(rec, dict):
                    continue
                meta = rec.get("meta", None)
                if not isinstance(meta, dict):
                    continue
                edges = meta.get("edges", None)
                if not isinstance(edges, dict):
                    continue
                for raw_key, raw_edge in edges.items():
                    key = self._parse_edge_key(raw_key)
                    if key is None:
                        continue
                    src, dst = key
                    edge_data = dict(raw_edge or {}) if isinstance(raw_edge, dict) else {}
                    merged = out.get((src, dst))
                    if merged is None:
                        merged = {}
                        out[(src, dst)] = merged
                    self._deep_merge_dict(merged, edge_data)
                    contrib = merged.get("components", None)
                    if not isinstance(contrib, list):
                        contrib = []
                        merged["components"] = contrib
                    if str(comp_name) not in contrib:
                        contrib.append(str(comp_name))
                    phase_src = merged.get("phases", None)
                    if not isinstance(phase_src, list):
                        phase_src = []
                        merged["phases"] = phase_src
                    if phase_name not in phase_src:
                        phase_src.append(phase_name)
        return out

    def edge_port_pairs(
        self,
        *,
        phase: str = "base",
        model_priority: Tuple[str, ...] = ("routed", "estimated"),
    ) -> Dict[Tuple[str, str], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        out: Dict[Tuple[str, str], List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
        for key, edge_meta in self.edge_metadata(phase=phase).items():
            pairs = self._extract_model_port_pairs(edge_meta, model_priority=model_priority)
            if pairs:
                out[key] = pairs
        return out

    def recompute_objective(self, *, finalized: bool) -> float:
        base_total = 0.0
        for rec in self.base_components.values():
            base_total += float(rec.get("weighted_cost", 0.0))
        terminal_total = 0.0
        for rec in self.terminal_components.values():
            terminal_total += float(rec.get("delta_cost", 0.0))
        total = float(base_total + terminal_total)
        self.objective["base_total"] = float(base_total)
        self.objective["terminal_total"] = float(terminal_total)
        self.objective["cost_total"] = float(total)
        self.objective["reward_total"] = -float(total) / float(self.reward_scale)
        self.objective["reward_scale"] = float(self.reward_scale)
        self.objective["finalized"] = bool(finalized)
        return float(total)

    def finalize(self, *, finalized: bool) -> float:
        return self.recompute_objective(finalized=finalized)
