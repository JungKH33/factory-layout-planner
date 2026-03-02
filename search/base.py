from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from envs.wrappers.base import BaseWrapper

from agents.base import Agent
from envs.wrappers.candidate_set import CandidateSet


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


class SearchStrategy(Protocol):
    """Choose an action index among candidates for the current state."""

    def select(
        self,
        *,
        env: BaseWrapper,
        obs: dict,
        agent: Agent,
        root_candidates: CandidateSet,
    ) -> int: ...


class BaseSearch(ABC):
    """Base class for search algorithms with progress callback support."""
    
    def __init__(self):
        self.top_tracker: Optional[TopKTracker] = None
        self._progress_callback: Optional[ProgressCallback] = None
        self._progress_interval: int = 10
    
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
    
    @abstractmethod
    def select(
        self,
        *,
        env: BaseWrapper,
        obs: dict,
        agent: Agent,
        root_candidates: CandidateSet,
    ) -> int:
        """Select an action from candidates."""
        ...


@dataclass
class SearchResult:
    """하나의 완료된 배치 결과"""
    cost: float                                   # cost() 값 (낮을수록 좋음)
    cum_reward: float                             # 누적 reward
    positions: Dict[str, Tuple[int, int, int]]    # {gid: (x, y, rot)}
    snapshot: Dict[str, Any]                      # env 복원용


class TopKTracker:
    """Search 중 최고 결과 K개 추적 (cost 기준 오름차순 - 낮을수록 좋음)"""

    def __init__(self, k: int = 5, verbose: bool = False):
        self.k = k
        self.verbose = verbose  # True면 리스트 변경 시 print
        self._heap: List[Tuple[float, int, SearchResult]] = []  # max-heap (negated cost)
        self._counter = 0

    def add(self, result: SearchResult) -> bool:
        """결과 추가. cost가 낮을수록 좋은 결과로 간주. 리스트 변경 시 True 반환."""
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
            print(
                f"[TopK] New result: cost={result.cost:.2f} | "
                f"Top-{len(self._heap)} range: [{best_cost:.2f} ~ {worst_cost:.2f}]"
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

