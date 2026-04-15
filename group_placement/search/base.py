from __future__ import annotations

import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from group_placement.agents.base import Agent, BaseAdapter
from group_placement.envs.env import FactoryLayoutEnv
from group_placement.envs.state import EnvState
from group_placement.envs.action_space import ActionSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SearchOutput — structured return from select()
# ---------------------------------------------------------------------------

@dataclass
class SearchOutput:
    """Structured result from a search algorithm's ``select()``."""

    action: int
    """Primary action index (manager action for hierarchical searches)."""

    worker_action: int = -1
    """Worker action index for hierarchical searches (-1 if flat)."""

    visits: Optional[np.ndarray] = None
    """[N] visit counts per root action."""

    values: Optional[np.ndarray] = None
    """[N] Q-values / scores per root action."""

    iterations: int = 0
    """Number of iterations / simulations / expansions performed."""

    top_k: Optional[List[Dict[str, Any]]] = None
    """Best terminal solutions found during search.
    Each dict: ``{"cost", "cum_reward", "positions", "engine_state"}``."""


# ---------------------------------------------------------------------------
# Progress callback type — simple callable, no wrapper class
# ---------------------------------------------------------------------------

ProgressFn = Callable[[int, int, np.ndarray, np.ndarray, int, float], None]
"""(iteration, total, visits, values, best_action, best_value)"""


# ---------------------------------------------------------------------------
# Top-K terminal tracking — stateless utility functions
# ---------------------------------------------------------------------------

def track_terminal(
    heap: list,
    counter: int,
    engine: FactoryLayoutEnv,
    cum_reward: float,
    max_k: int,
) -> int:
    """Record a terminal engine state in a local min-heap.

    Returns the updated counter.  Does nothing if *max_k* <= 0.
    """
    if max_k <= 0:
        return counter
    cost = engine.cost()
    # Deduplicate by cost
    if any(-neg_c == cost for neg_c, _, _ in heap):
        return counter
    positions = {
        str(gid): p.position()
        for gid, p in engine.get_state().placements.items()
    }
    entry = (
        -cost,   # negated for max-heap behaviour (worst-cost evicted first)
        counter,
        {
            "cost": cost,
            "cum_reward": cum_reward,
            "positions": positions,
            "engine_state": engine.get_state().copy(),
        },
    )
    if len(heap) < max_k:
        heapq.heappush(heap, entry)
    elif -cost > heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def collect_top_k(heap: list) -> Optional[List[Dict[str, Any]]]:
    """Convert a local heap into a cost-ascending list of result dicts."""
    if not heap:
        return None
    return sorted([item[2] for item in heap], key=lambda r: r["cost"])


# ---------------------------------------------------------------------------
# Snapshot / cache helpers
# ---------------------------------------------------------------------------

@dataclass
class SearchSnapshot:
    """Restorable engine + adapter state for a searched node."""
    engine_state: EnvState
    adapter_state: Dict[str, object]


@dataclass
class DecisionCache:
    """Cached decision artifacts for one concrete environment state."""
    snapshot: SearchSnapshot
    obs: Dict[str, Any]
    action_space: ActionSpace


@dataclass(frozen=True)
class BaseSearchConfig:
    """Common search config fields shared by all search algorithms."""
    pass


# ---------------------------------------------------------------------------
# BaseSearch
# ---------------------------------------------------------------------------

class BaseSearch(ABC):
    """Base class for search algorithms.

    Agent output contract used by all searches:
    - policy(obs, action_space) -> [N] non-negative scores (higher is better),
      not necessarily normalized.
    - value(obs, action_space) -> scalar utility (single float), not probability.
    """

    def __init__(self):
        self.adapter: Optional[BaseAdapter] = None

    def set_adapter(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter

    # ---- snapshot helpers ----

    def _capture_snapshot(
        self,
        *,
        engine,
        adapter: BaseAdapter,
    ) -> SearchSnapshot:
        return SearchSnapshot(
            engine_state=engine.get_state().copy(),
            adapter_state=adapter.get_state_copy(),
        )

    def _restore_snapshot(
        self,
        *,
        engine,
        adapter: BaseAdapter,
        snapshot: SearchSnapshot,
    ) -> None:
        engine.set_state(snapshot.engine_state)
        adapter.set_state(snapshot.adapter_state)

    def _capture_decision_cache(
        self,
        *,
        engine,
        adapter: BaseAdapter,
        obs: Dict[str, Any],
        action_space: ActionSpace,
    ) -> DecisionCache:
        return DecisionCache(
            snapshot=self._capture_snapshot(engine=engine, adapter=adapter),
            obs=dict(obs),
            action_space=action_space,
        )

    def _empty_action_space(self, *, device: torch.device) -> ActionSpace:
        return ActionSpace(
            centers=torch.zeros((0, 2), dtype=torch.float32, device=device),
            valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    def _capture_terminal_cache(
        self,
        *,
        engine,
        adapter: BaseAdapter,
    ) -> DecisionCache:
        return DecisionCache(
            snapshot=SearchSnapshot(
                engine_state=engine.get_state().copy(),
                adapter_state={},
            ),
            obs={},
            action_space=self._empty_action_space(device=adapter.device),
        )

    # ---- action execution helper ----

    def _apply_action_index(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        action: int,
        action_space: ActionSpace,
    ):
        try:
            placement = adapter.resolve_action(int(action), action_space)
            _, reward, terminated, truncated, info = engine.step_placement(
                placement,
            )
        except IndexError:
            return float(engine.failure_penalty()), False, True, {"reason": "action_out_of_range"}
        except ValueError as e:
            reason = "no_valid_actions" if str(e) == "no_valid_actions" else "masked_action"
            return float(engine.failure_penalty()), False, True, {"reason": reason}
        return float(reward), bool(terminated), bool(truncated), info

    # ---- batched policy / value helpers ----

    def _policy_many(
        self,
        *,
        agent: Agent,
        obs_batch: List[dict],
        action_space_batch: List[ActionSpace],
        device: torch.device,
    ) -> List[torch.Tensor]:
        n = int(len(obs_batch))
        if n != int(len(action_space_batch)):
            raise ValueError("obs_batch and action_space_batch length mismatch")
        if n <= 0:
            return []

        out_raw: Optional[List[torch.Tensor]] = None
        policy_batch_fn = getattr(agent, "policy_batch", None)
        if callable(policy_batch_fn):
            try:
                maybe = policy_batch_fn(
                    obs_batch=obs_batch,
                    action_space_batch=action_space_batch,
                )
            except TypeError:
                maybe = policy_batch_fn(obs_batch, action_space_batch)
            except Exception:
                maybe = None
            if isinstance(maybe, (list, tuple)) and int(len(maybe)) == n:
                out_raw = list(maybe)

        if out_raw is None:
            out_raw = [
                agent.policy(obs=obs, action_space=action_space)
                for obs, action_space in zip(obs_batch, action_space_batch)
            ]

        out: List[torch.Tensor] = []
        for item in out_raw:
            if isinstance(item, torch.Tensor):
                out.append(item.to(dtype=torch.float32, device=device).view(-1))
            else:
                out.append(torch.tensor(item, dtype=torch.float32, device=device).view(-1))
        return out

    def _value_many(
        self,
        *,
        agent: Agent,
        obs_batch: List[dict],
        action_space_batch: List[ActionSpace],
    ) -> List[float]:
        n = int(len(obs_batch))
        if n != int(len(action_space_batch)):
            raise ValueError("obs_batch and action_space_batch length mismatch")
        if n <= 0:
            return []

        values_raw: Optional[List[float]] = None
        value_batch_fn = getattr(agent, "value_batch", None)
        if callable(value_batch_fn):
            try:
                maybe = value_batch_fn(
                    obs_batch=obs_batch,
                    action_space_batch=action_space_batch,
                )
            except TypeError:
                maybe = value_batch_fn(obs_batch, action_space_batch)
            except Exception:
                maybe = None
            if isinstance(maybe, (list, tuple)) and int(len(maybe)) == n:
                values_raw = list(maybe)

        if values_raw is None:
            values_raw = [
                agent.value(obs=obs, action_space=action_space)
                for obs, action_space in zip(obs_batch, action_space_batch)
            ]

        out: List[float] = []
        for v in values_raw:
            if isinstance(v, torch.Tensor):
                out.append(float(v.view(-1)[0].item()) if v.numel() > 0 else 0.0)
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(0.0)
        return out

    # ---- value / fallback helpers (shared by flat searches) ----

    def _safe_value(
        self,
        *,
        agent: Agent,
        obs: dict,
        action_space: ActionSpace,
    ) -> float:
        """Call agent.value() and coerce to float, returning 0.0 on failure."""
        try:
            value = agent.value(obs=obs, action_space=action_space)
        except Exception:
            return 0.0
        try:
            return float(value)
        except Exception:
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return float(value.view(-1)[0].item())
            return 0.0

    def _fallback_action(
        self,
        *,
        agent: Agent,
        obs: dict,
        root_action_space: ActionSpace,
        device: torch.device,
    ) -> int:
        """Pick a valid action via agent policy argmax; first valid index on failure."""
        valid = root_action_space.valid_mask.to(dtype=torch.bool, device=device).view(-1)
        valid_idx = torch.where(valid)[0]
        if int(valid_idx.numel()) <= 0:
            return -1
        try:
            scores = agent.policy(obs=obs, action_space=root_action_space).to(dtype=torch.float32, device=device).view(-1)
            scores = scores.masked_fill(~valid, float("-inf"))
            if bool(torch.isfinite(scores).any().item()):
                return int(torch.argmax(scores).item())
        except Exception:
            pass
        return int(valid_idx[0].item())

    # ---- abstract API ----

    @abstractmethod
    def select(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
        progress_fn: Optional[ProgressFn] = None,
        progress_interval: int = 10,
    ) -> SearchOutput:
        """Select an action from action_space."""
        ...


# ---------------------------------------------------------------------------
# BaseHierarchicalSearch
# ---------------------------------------------------------------------------

class BaseHierarchicalSearch(BaseSearch):
    """Base class for hierarchical (manager/worker) search algorithms.

    Subclasses implement ``select()`` directly, returning a ``SearchOutput``
    whose ``worker_action`` field carries the worker-level choice.
    """

    def set_adapter(self, adapter: BaseAdapter) -> None:
        if not adapter.supports_hierarchical:
            raise TypeError(
                f"{type(self).__name__} requires adapter with "
                f"supports_hierarchical=True, got {type(adapter).__name__}"
            )
        super().set_adapter(adapter)
