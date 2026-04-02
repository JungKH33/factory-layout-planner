from __future__ import annotations

import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base import Agent, BaseAdapter
from envs.env import FactoryLayoutEnv
from envs.state import EnvState
from envs.action_space import ActionSpace

logger = logging.getLogger(__name__)


@dataclass
class SearchProgress:
    """Progress update during search."""
    iteration: int              # Current iteration (simulation/beam step/etc)
    total: int                  # Total iterations
    visits: np.ndarray          # Visit counts per candidate [N]
    values: np.ndarray          # Q-values or scores per candidate [N]
    best_action: int            # Current best action
    best_value: float           # Best value
    extra: Dict[str, Any] = field(default_factory=dict)  # Algorithm-specific data


# Type alias for progress callback
ProgressCallback = Callable[[SearchProgress], None]


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


class BaseSearch(ABC):
    """Base class for search algorithms with progress callback support.

    Agent output contract used by all searches:
    - policy(obs, action_space) -> [N] non-negative scores (higher is better),
      not necessarily normalized.
    - value(obs, action_space) -> scalar utility (single float), not probability.
    """

    def __init__(self):
        self.top_tracker: Optional[TopKTracker] = None
        self._progress_callback: Optional[ProgressCallback] = None
        self._progress_interval: int = 10
        self.adapter: Optional[BaseAdapter] = None

    def set_adapter(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter
    
    def set_progress_callback(
        self,
        callback: Optional[ProgressCallback],
        interval: int = 10,
    ) -> None:
        """Set callback for progress updates during search.
        
        Args:
            callback: Function called with SearchProgress at intervals
            interval: Call callback every N iterations
        """
        self._progress_callback = callback
        self._progress_interval = max(1, interval)
    
    def _emit_progress(self, progress: SearchProgress) -> None:
        """Emit progress update if callback is set."""
        if self._progress_callback is not None:
            self._progress_callback(progress)

    def _capture_snapshot(
        self,
        *,
        engine,
        adapter: BaseAdapter,
    ) -> SearchSnapshot:
        """Capture engine + adapter state after a decision has been built."""
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
        """Restore engine + adapter state captured by ``_capture_snapshot``."""
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
        """Capture a search state together with its built observation/action space."""
        return DecisionCache(
            snapshot=self._capture_snapshot(engine=engine, adapter=adapter),
            obs=dict(obs),
            action_space=action_space,
        )

    def _empty_action_space(self, *, device: torch.device) -> ActionSpace:
        """Return an empty action space for terminal search nodes."""
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
        """Capture terminal engine state with an empty decision payload."""
        return DecisionCache(
            snapshot=SearchSnapshot(
                engine_state=engine.get_state().copy(),
                adapter_state={},
            ),
            obs={},
            action_space=self._empty_action_space(device=adapter.device),
        )
    
    def _apply_action_index(
        self,
        *,
        engine: FactoryLayoutEnv,
        adapter: BaseAdapter,
        action: int,
        action_space: ActionSpace,
    ):
        """Apply discrete action index via adapter → engine.
        """
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

    def _track_terminal(self, *, engine: FactoryLayoutEnv, cum_reward: float) -> None:
        """Record a terminal state in the top-K tracker (if enabled)."""
        if self.top_tracker is None:
            return
        cost = engine.total_cost()
        positions = {str(gid): p.position() for gid, p in engine.get_state().placements.items()}
        self.top_tracker.add(SearchResult(
            cost=cost,
            cum_reward=cum_reward,
            positions=positions,
            engine_state=engine.get_state().copy(),
        ))

    def _policy_many(
        self,
        *,
        agent: Agent,
        obs_batch: List[dict],
        action_space_batch: List[ActionSpace],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Compute policy scores for multiple nodes.

        Preferred path:
        - agent.policy_batch(obs_batch=..., action_space_batch=...)

        Fallback path:
        - per-item agent.policy(...)

        Output semantics are identical to Agent.policy contract:
        [N] non-negative scores, larger = better.
        """
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
        """Compute value estimates for multiple nodes.

        Preferred path:
        - agent.value_batch(obs_batch=..., action_space_batch=...)

        Fallback path:
        - per-item agent.value(...)

        Output semantics are identical to Agent.value contract:
        single scalar utility per node, larger = better.
        """
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

    @abstractmethod
    def select(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
    ) -> int:
        """Select an action from action_space."""
        ...


class BaseHierarchicalSearch(BaseSearch):
    """Base class for hierarchical (manager/worker) search algorithms."""

    def set_adapter(self, adapter: BaseAdapter) -> None:
        if not adapter.supports_hierarchical:
            raise TypeError(
                f"{type(self).__name__} requires adapter with "
                f"supports_hierarchical=True, got {type(adapter).__name__}"
            )
        super().set_adapter(adapter)

    @abstractmethod
    def select_h(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
    ) -> Tuple[int, int]:
        """Select hierarchical action as (manager_action, worker_action)."""
        ...

    def select(
        self,
        *,
        obs: dict,
        agent: Agent,
        root_action_space: ActionSpace,
    ) -> int:
        manager_action, _worker_action = self.select_h(
            obs=obs,
            agent=agent,
            root_action_space=root_action_space,
        )
        return int(manager_action)


@dataclass
class SearchResult:
    """하나의 완료된 배치 결과"""
    cost: float                                   # cost() 값 (낮을수록 좋음)
    cum_reward: float                             # 누적 reward
    positions: Dict[str, Tuple[float, float]]     # {gid: (x_center, y_center)}
    engine_state: EnvState                        # env 복원용


class TopKTracker:
    """Search 중 최고 결과 K개 추적 (cost 기준 오름차순 - 낮을수록 좋음)"""

    def __init__(self, k: int = 5, verbose: bool = False):
        self.k = k
        self.verbose = verbose  # True면 리스트 변경 시 print
        self._heap: List[Tuple[float, int, SearchResult]] = []  # max-heap (negated cost)
        self._counter = 0

    def add(self, result: SearchResult) -> bool:
        """결과 추가. cost가 낮을수록 좋은 결과로 간주. 리스트 변경 시 True 반환."""
        if any(-neg_c == result.cost for neg_c, _, _ in self._heap):
            return False

        entry = (-result.cost, self._counter, result)
        self._counter += 1
        changed = False

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            changed = True
        elif -result.cost > self._heap[0][0]:  # 새 cost가 현재 worst보다 낮으면
            heapq.heapreplace(self._heap, entry)
            changed = True

        if changed and self.verbose:
            worst_cost = -self._heap[0][0] if self._heap else float("inf")
            best_cost = self.best_cost()
            logger.info(
                "TopK New result: cost=%.2f | Top-%d range: [%.2f ~ %.2f]",
                result.cost,
                len(self._heap),
                best_cost,
                worst_cost,
            )

        return changed

    def get_results(self) -> List[SearchResult]:
        """cost 오름차순 (best first)으로 반환"""
        return sorted([item[2] for item in self._heap], key=lambda r: r.cost)

    def best_cost(self) -> float:
        """현재 best cost 반환"""
        if not self._heap:
            return float("inf")
        results = self.get_results()
        return results[0].cost if results else float("inf")

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0
