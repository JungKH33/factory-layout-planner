from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class EvalState:
    """Runtime evaluation snapshot owned by EnvState.

    Internal storage is split into:
    - static-like: reward_scale, _reward_weights_by_name
    - dynamic: _base_raw / _base_metadata, _terminal_delta / _terminal_metadata
    - running totals: _base_total, _terminal_total  (incremental, O(1) recompute)
    - cache: _objective (O(1) sync), _base_rewards_cache / _terminal_rewards_cache

    Invariants maintained across all mutator returns:
      _base_total     == sum(raw * weight for each key in _base_raw)
      _terminal_total == sum(_terminal_delta.values())

    Metadata dicts stored internally are always replaced on write (never mutated
    in-place), so shallow copies of the outer container are safe. Properties that
    expose metadata return per-entry shallow copies to prevent callers from
    accidentally polluting internal state.
    """

    __slots__ = (
        "reward_scale",
        "layout_rev",
        "_reward_weights_by_name",
        "_base_raw",
        "_base_metadata",
        "_terminal_delta",
        "_terminal_metadata",
        "_base_total",
        "_terminal_total",
        "_objective",
        "_base_rewards_cache",
        "_terminal_rewards_cache",
    )

    def __init__(self, reward_scale: float) -> None:
        rs = float(reward_scale)
        if rs <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {rs}")
        self.reward_scale = rs
        self.layout_rev = 0

        self._reward_weights_by_name: Dict[str, float] = {}
        self._base_raw: Dict[str, float] = {}
        self._base_metadata: Dict[str, Dict[str, Any]] = {}
        self._terminal_delta: Dict[str, float] = {}
        self._terminal_metadata: Dict[str, Dict[str, Any]] = {}

        self._base_total: float = 0.0
        self._terminal_total: float = 0.0

        self._objective: Dict[str, Any] = {
            "base_total": 0.0,
            "terminal_total": 0.0,
            "cost_total": 0.0,
            "reward_total": 0.0,
            "reward_scale": rs,
            "finalized": False,
        }
        self._base_rewards_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._terminal_rewards_cache: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def empty(cls, reward_scale: float) -> "EvalState":
        return cls(reward_scale=float(reward_scale))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_json_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): EvalState._clone_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [EvalState._clone_json_value(v) for v in value]
        return value

    @staticmethod
    def _merge_json_dict(
        base: Mapping[str, Any],
        metadata_update: Mapping[str, Any],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(base)
        for key, update_value in metadata_update.items():
            old_value = base.get(key, None)
            if isinstance(old_value, dict) and isinstance(update_value, dict):
                out[str(key)] = EvalState._merge_json_dict(old_value, update_value)
            else:
                out[str(key)] = EvalState._clone_json_value(update_value)
        return out

    def _invalidate_reward_cache(self) -> None:
        self._base_rewards_cache = None
        self._terminal_rewards_cache = None

    def _sync_objective(self, *, finalized: bool) -> float:
        """Write running totals into _objective dict. O(1)."""
        cost = self._base_total + self._terminal_total
        self._objective["base_total"] = self._base_total
        self._objective["terminal_total"] = self._terminal_total
        self._objective["cost_total"] = cost
        self._objective["reward_total"] = -cost / self.reward_scale
        self._objective["reward_scale"] = self.reward_scale
        self._objective["finalized"] = finalized
        return cost

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def objective(self) -> Dict[str, Any]:
        return self._objective

    @property
    def base_rewards(self) -> Dict[str, Dict[str, Any]]:
        if self._base_rewards_cache is not None:
            return self._base_rewards_cache
        out: Dict[str, Dict[str, Any]] = {}
        for name, raw in self._base_raw.items():
            w = self._reward_weights_by_name.get(name, 1.0)
            out[name] = {
                "raw_cost": raw,
                "weight": w,
                "weighted_cost": raw * w,
                "metadata": dict(self._base_metadata.get(name, {})),
            }
        self._base_rewards_cache = out
        return out

    @property
    def terminal_rewards(self) -> Dict[str, Dict[str, Any]]:
        if self._terminal_rewards_cache is not None:
            return self._terminal_rewards_cache
        out: Dict[str, Dict[str, Any]] = {}
        for name, delta in self._terminal_delta.items():
            out[name] = {
                "delta_cost": delta,
                "metadata": dict(self._terminal_metadata.get(name, {})),
            }
        self._terminal_rewards_cache = out
        return out

    # ------------------------------------------------------------------
    # copy / restore
    # ------------------------------------------------------------------

    def copy(self) -> "EvalState":
        out = EvalState(reward_scale=self.reward_scale)
        out.layout_rev = self.layout_rev

        out._reward_weights_by_name = dict(self._reward_weights_by_name)
        out._base_raw = dict(self._base_raw)
        # Shallow copy of outer dict: inner dicts are safe to share because all
        # write paths (set_base_snapshot, record_base_delta, merge_metadata) always
        # replace entries via _merge_json_dict / _clone_json_value rather than
        # mutating them in-place. External callers receive copies via base_rewards.
        out._base_metadata = dict(self._base_metadata)
        out._terminal_delta = dict(self._terminal_delta)
        out._terminal_metadata = dict(self._terminal_metadata)

        out._base_total = self._base_total
        out._terminal_total = self._terminal_total
        out._objective = dict(self._objective)
        # caches are not copied — regenerated lazily on first access
        return out

    def restore(self, src: "EvalState") -> None:
        if not isinstance(src, EvalState):
            raise TypeError(f"src must be EvalState, got {type(src).__name__}")

        self.reward_scale = src.reward_scale
        self.layout_rev = src.layout_rev

        self._reward_weights_by_name = dict(src._reward_weights_by_name)
        self._base_raw = dict(src._base_raw)
        # Same shallow-copy contract as copy() above.
        self._base_metadata = dict(src._base_metadata)
        self._terminal_delta = dict(src._terminal_delta)
        self._terminal_metadata = dict(src._terminal_metadata)

        self._base_total = src._base_total
        self._terminal_total = src._terminal_total
        self._objective = dict(src._objective)
        self._base_rewards_cache = None
        self._terminal_rewards_cache = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_runtime(self) -> None:
        self.layout_rev = 0
        self._reward_weights_by_name = {}
        self._base_raw = {}
        self._base_metadata = {}
        self._terminal_delta = {}
        self._terminal_metadata = {}
        self._base_total = 0.0
        self._terminal_total = 0.0
        self._objective = {
            "base_total": 0.0,
            "terminal_total": 0.0,
            "cost_total": 0.0,
            "reward_total": 0.0,
            "reward_scale": self.reward_scale,
            "finalized": False,
        }
        self._base_rewards_cache = None
        self._terminal_rewards_cache = None

    def start_revision(self) -> None:
        self.layout_rev += 1
        self._terminal_delta = {}
        self._terminal_metadata = {}
        self._terminal_total = 0.0
        self._invalidate_reward_cache()
        self._sync_objective(finalized=False)

    # ------------------------------------------------------------------
    # Base snapshot mutators
    # ------------------------------------------------------------------

    def set_base_snapshot(self, rewards: Dict[str, Dict[str, Any]]) -> None:
        weights: Dict[str, float] = {}
        raws: Dict[str, float] = {}
        metadatas: Dict[str, Dict[str, Any]] = {}
        base_total = 0.0

        for name, rec_in in dict(rewards or {}).items():
            key = str(name)
            rec = dict(rec_in or {})
            w = float(rec.get("weight", 1.0))
            r = float(rec.get("raw_cost", 0.0))
            weights[key] = w
            raws[key] = r
            base_total += r * w
            metadata = dict(rec.get("metadata", {}) or {})
            metadatas[key] = self._clone_json_value(metadata)

        self._reward_weights_by_name = weights
        self._base_raw = raws
        self._base_metadata = metadatas
        self._base_total = base_total
        self._invalidate_reward_cache()
        self._sync_objective(finalized=self._objective["finalized"])

    def record_base_delta(
        self,
        *,
        reward_delta_by_name: Mapping[str, float],
        metadata_by_reward: Optional[Mapping[str, Mapping[str, Any]]] = None,
        reward_weights_by_name: Optional[Mapping[str, float]] = None,
    ) -> None:
        metadata_by_reward_dict = dict(metadata_by_reward or {})
        weights = dict(reward_weights_by_name or {})

        if not weights:
            # Common path (production): weights are fixed, only raw deltas arrive.
            _raws = self._base_raw
            _wts = self._reward_weights_by_name
            _meta = self._base_metadata
            dt = 0.0
            for name, delta in dict(reward_delta_by_name or {}).items():
                key = str(name)
                if key not in _wts:
                    raise KeyError(f"missing weight for reward {key!r}")
                fdelta = float(delta)
                old_raw = _raws.get(key, 0.0)
                _raws[key] = old_raw + fdelta
                dt += fdelta * _wts[key]

                if key in metadata_by_reward_dict:
                    metadata_value = metadata_by_reward_dict[key]
                    if not isinstance(metadata_value, Mapping):
                        raise TypeError(f"metadata_by_reward[{key!r}] must be a mapping")
                    metadata_update = dict(metadata_value)
                    if metadata_update:
                        _meta[key] = self._merge_json_dict(_meta.get(key, {}), metadata_update)
                    elif key not in _meta:
                        _meta[key] = {}
                elif key not in _meta:
                    _meta[key] = {}
            self._base_total += dt
        else:
            # Full path: weights may change alongside raw deltas.
            for name, delta in dict(reward_delta_by_name or {}).items():
                key = str(name)

                old_raw = self._base_raw.get(key, 0.0)
                old_w = self._reward_weights_by_name.get(key, 0.0)

                if key in weights:
                    new_w = float(weights[key])
                    self._reward_weights_by_name[key] = new_w
                elif key not in self._reward_weights_by_name:
                    raise KeyError(f"missing weight for reward {key!r}")
                else:
                    new_w = old_w

                new_raw = old_raw + float(delta)
                self._base_raw[key] = new_raw
                self._base_total += new_raw * new_w - old_raw * old_w

                if key in metadata_by_reward_dict:
                    metadata_value = metadata_by_reward_dict[key]
                    if not isinstance(metadata_value, Mapping):
                        raise TypeError(f"metadata_by_reward[{key!r}] must be a mapping")
                    metadata_update = dict(metadata_value)
                    if metadata_update:
                        current_metadata = self._base_metadata.get(key, {})
                        self._base_metadata[key] = self._merge_json_dict(current_metadata, metadata_update)
                    elif key not in self._base_metadata:
                        self._base_metadata[key] = {}
                elif key not in self._base_metadata:
                    self._base_metadata[key] = {}

        self._invalidate_reward_cache()
        self._sync_objective(finalized=False)

    # ------------------------------------------------------------------
    # Terminal snapshot mutators
    # ------------------------------------------------------------------

    def set_terminal_snapshot(self, rewards: Dict[str, Dict[str, Any]]) -> None:
        deltas: Dict[str, float] = {}
        metadatas: Dict[str, Dict[str, Any]] = {}
        terminal_total = 0.0

        for name, rec_in in dict(rewards or {}).items():
            key = str(name)
            rec = dict(rec_in or {})
            d = float(rec.get("delta_cost", 0.0))
            deltas[key] = d
            terminal_total += d
            metadata = dict(rec.get("metadata", {}) or {})
            metadatas[key] = self._clone_json_value(metadata)

        self._terminal_delta = deltas
        self._terminal_metadata = metadatas
        self._terminal_total = terminal_total
        self._invalidate_reward_cache()
        self._sync_objective(finalized=self._objective["finalized"])

    def clear_terminal(self) -> None:
        self._terminal_delta = {}
        self._terminal_metadata = {}
        self._terminal_total = 0.0
        self._invalidate_reward_cache()
        self._sync_objective(finalized=False)

    # ------------------------------------------------------------------
    # Metadata merge (no total change)
    # ------------------------------------------------------------------

    def merge_metadata(
        self,
        *,
        name: str,
        metadata: Dict[str, Any],
        phase: str = "base",
    ) -> None:
        phase_key = str(phase)
        key = str(name)
        metadata_update = dict(metadata or {})

        if phase_key == "base":
            if key not in self._reward_weights_by_name:
                self._reward_weights_by_name[key] = 1.0
            if key not in self._base_raw:
                self._base_raw[key] = 0.0
            if metadata_update:
                current_metadata = self._base_metadata.get(key, {})
                self._base_metadata[key] = self._merge_json_dict(current_metadata, metadata_update)
            elif key not in self._base_metadata:
                self._base_metadata[key] = {}
            self._invalidate_reward_cache()
            return

        if phase_key == "terminal":
            if key not in self._terminal_delta:
                self._terminal_delta[key] = 0.0
            if metadata_update:
                current_metadata = self._terminal_metadata.get(key, {})
                self._terminal_metadata[key] = self._merge_json_dict(current_metadata, metadata_update)
            elif key not in self._terminal_metadata:
                self._terminal_metadata[key] = {}
            self._invalidate_reward_cache()
            return

        raise ValueError(f"phase must be 'base' or 'terminal', got {phase!r}")

    # ------------------------------------------------------------------
    # Objective computation (public API — now O(1))
    # ------------------------------------------------------------------

    def recompute_objective(self, *, finalized: bool) -> float:
        return self._sync_objective(finalized=finalized)

    def finalize(self, *, finalized: bool) -> float:
        return self.recompute_objective(finalized=finalized)
