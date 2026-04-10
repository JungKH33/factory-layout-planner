from __future__ import annotations

from dataclasses import dataclass
import heapq
import logging
import statistics
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from .action import LaneRoute
from .action_space import ActionSpace
from .wavefront import backtrace_shortest_path, wavefront_distance_field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaneAdapterConfig:
    candidate_k: int = 8
    max_wave_iters: int = 0
    max_backtrace_steps: int = 0
    random_seed: int = 0
    route_algorithm: str = "wavefront"  # wavefront | astar | dijkstra | auto
    route_algorithm_selection: str = "static"  # static | benchmark
    benchmark_warmup: int = 2
    benchmark_rounds: int = 6
    benchmark_max_flows_per_method: int = 6
    # Backward-compatible aliases (deprecated).
    algorithm: Optional[str] = None
    algorithm_selection: Optional[str] = None


class LaneAdapter:
    """Build lane candidates for current flow and resolve action index to route."""

    _METHOD_SINGLE = "single_src"
    _METHOD_MULTI = "multi_src"
    _ALL_METHODS = (_METHOD_SINGLE, _METHOD_MULTI)

    def __init__(self, *, config: Optional[LaneAdapterConfig] = None) -> None:
        self.config = config or LaneAdapterConfig()
        self.env = None
        self._rng = torch.Generator()
        self._rng.manual_seed(int(self.config.random_seed))
        self._selected_algorithms: Dict[str, str] = {}

    def bind(self, env) -> None:
        self.env = env
        self._resolve_algorithms()

    def _empty_action_space(self, *, flow_index: int, device: torch.device) -> ActionSpace:
        return ActionSpace(
            flow_index=int(flow_index),
            candidate_edge_idx=torch.zeros((0, 0), dtype=torch.long, device=device),
            candidate_edge_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
            valid_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _normalize_algorithm(name: str) -> str:
        algo = str(name).strip().lower()
        if algo not in {"wavefront", "astar", "dijkstra", "auto"}:
            raise ValueError(f"invalid lane algorithm {name!r}; expected wavefront|astar|dijkstra|auto")
        return algo

    @staticmethod
    def _normalize_selection_mode(name: str) -> str:
        mode = str(name).strip().lower()
        if mode not in {"static", "benchmark"}:
            raise ValueError(f"invalid route_algorithm_selection {name!r}; expected static|benchmark")
        return mode

    def _config_route_algorithm(self) -> str:
        raw = self.config.algorithm if self.config.algorithm is not None else self.config.route_algorithm
        return self._normalize_algorithm(raw)

    def _config_route_algorithm_selection(self) -> str:
        raw = (
            self.config.algorithm_selection
            if self.config.algorithm_selection is not None
            else self.config.route_algorithm_selection
        )
        return self._normalize_selection_mode(raw)

    def _method_key(self, *, src_ports: torch.Tensor) -> str:
        src_n = int(src_ports.shape[0]) if src_ports.dim() >= 1 else 0
        return self._METHOD_MULTI if src_n > 1 else self._METHOD_SINGLE

    def _candidate_algorithms(self, *, method: str) -> List[str]:
        algo_cfg = self._config_route_algorithm()
        if algo_cfg != "auto":
            return [algo_cfg]
        if method == self._METHOD_SINGLE:
            # Single-source flow often benefits from directed search.
            return ["astar", "wavefront", "dijkstra"]
        return ["wavefront", "astar", "dijkstra"]

    def _route_wavefront(
        self,
        *,
        dist: torch.Tensor,
        src_xy: Tuple[int, int],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        return backtrace_shortest_path(
            dist=dist,
            src_xy=src_xy,
            rng=rng,
            max_steps=int(self.config.max_backtrace_steps),
        )

    def _route_graph_search(
        self,
        *,
        free_map: torch.Tensor,
        src_xy: Tuple[int, int],
        goals_xy: Sequence[Tuple[int, int]],
        algorithm: str,
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        if free_map.dim() != 2:
            raise ValueError(f"free_map must be [H,W], got {tuple(free_map.shape)}")

        algo = self._normalize_algorithm(algorithm)
        if algo not in {"astar", "dijkstra"}:
            raise ValueError(f"algorithm must be 'astar' or 'dijkstra', got {algorithm!r}")

        h, w = int(free_map.shape[0]), int(free_map.shape[1])
        sx, sy = int(src_xy[0]), int(src_xy[1])
        if not (0 <= sx < w and 0 <= sy < h):
            return None
        if not bool(free_map[sy, sx].item()):
            return None

        goals: Set[Tuple[int, int]] = {
            (int(x), int(y))
            for x, y in goals_xy
            if 0 <= int(x) < w and 0 <= int(y) < h and bool(free_map[int(y), int(x)].item())
        }
        if not goals:
            return None

        base_dirs: List[Tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if rng is not None:
            perm = torch.randperm(4, generator=rng).tolist()
            dirs = [base_dirs[int(i)] for i in perm]
        else:
            dirs = base_dirs

        def heuristic(x: int, y: int) -> int:
            if algo != "astar":
                return 0
            return min(abs(gx - int(x)) + abs(gy - int(y)) for gx, gy in goals)

        g_best = {(sx, sy): 0}
        prev = {(sx, sy): None}
        pq: List[Tuple[int, int, int, int]] = [(heuristic(sx, sy), 0, sx, sy)]
        cap = int(self.config.max_backtrace_steps)

        while pq:
            _f, g, x, y = heapq.heappop(pq)
            if g != g_best.get((x, y), None):
                continue
            if (x, y) in goals:
                path: List[Tuple[int, int]] = []
                cur: Optional[Tuple[int, int]] = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path

            ng = int(g) + 1
            if cap > 0 and ng > cap:
                continue

            for dx, dy in dirs:
                nx, ny = int(x + dx), int(y + dy)
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if not bool(free_map[ny, nx].item()):
                    continue

                old = g_best.get((nx, ny), None)
                if old is not None and int(ng) >= int(old):
                    continue

                g_best[(nx, ny)] = int(ng)
                prev[(nx, ny)] = (x, y)
                nf = int(ng) + int(heuristic(nx, ny))
                heapq.heappush(pq, (nf, int(ng), nx, ny))

        return None

    def _route_astar(
        self,
        *,
        free_map: torch.Tensor,
        src_xy: Tuple[int, int],
        goals_xy: Sequence[Tuple[int, int]],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        return self._route_graph_search(
            free_map=free_map,
            src_xy=src_xy,
            goals_xy=goals_xy,
            algorithm="astar",
            rng=rng,
        )

    def _route_dijkstra(
        self,
        *,
        free_map: torch.Tensor,
        src_xy: Tuple[int, int],
        goals_xy: Sequence[Tuple[int, int]],
        rng: Optional[torch.Generator],
    ) -> Optional[List[Tuple[int, int]]]:
        return self._route_graph_search(
            free_map=free_map,
            src_xy=src_xy,
            goals_xy=goals_xy,
            algorithm="dijkstra",
            rng=rng,
        )

    def _discover_flow_samples(self) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        samples: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {m: [] for m in self._ALL_METHODS}
        if self.env is None:
            return samples

        state = self.env.get_state()
        max_per_method = max(1, int(self.config.benchmark_max_flows_per_method))
        for flow_idx in range(int(state.flow.flow_count)):
            src_ports, dst_ports = state.flow.valid_ports(flow_idx)
            if src_ports.numel() == 0 or dst_ports.numel() == 0:
                continue
            method = self._method_key(src_ports=src_ports)
            if len(samples[method]) >= max_per_method:
                continue
            samples[method].append(
                (
                    src_ports.detach().clone(),
                    dst_ports.detach().clone(),
                )
            )
            if all(len(samples[m]) >= max_per_method for m in self._ALL_METHODS):
                break
        return samples

    def _build_candidates(
        self,
        *,
        state,
        free_map: torch.Tensor,
        src_ports: torch.Tensor,
        dst_ports: torch.Tensor,
        k_target: int,
        algorithm: str,
        rng: Optional[torch.Generator],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        candidates: List[torch.Tensor] = []
        turns_l: List[int] = []
        seen = set()
        src_list = src_ports.detach().cpu().tolist()
        dst_list = [(int(p[0]), int(p[1])) for p in dst_ports.detach().cpu().tolist()]
        algo = self._normalize_algorithm(algorithm)
        if algo == "auto":
            raise ValueError("internal error: _build_candidates requires concrete algorithm")

        if algo == "wavefront":
            dist = wavefront_distance_field(
                free_map=free_map,
                seeds_xy=dst_ports,
                max_iters=int(self.config.max_wave_iters),
            )

            def build_path(src_xy: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
                return self._route_wavefront(
                    dist=dist,
                    src_xy=src_xy,
                    rng=rng,
                )
        elif algo == "astar":
            def build_path(src_xy: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
                return self._route_astar(
                    free_map=free_map,
                    src_xy=src_xy,
                    goals_xy=dst_list,
                    rng=rng,
                )
        else:
            def build_path(src_xy: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
                return self._route_dijkstra(
                    free_map=free_map,
                    src_xy=src_xy,
                    goals_xy=dst_list,
                    rng=rng,
                )

        # Primary pass: one route per source port.
        for p in src_list:
            path = build_path((int(p[0]), int(p[1])))
            if not path:
                continue
            edges, turns = state.maps.path_to_edge_ids_and_turns(path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(edges)
            turns_l.append(int(turns))
            if len(candidates) >= int(k_target):
                break

        # Diversity pass: stochastic tie-break retries.
        retry = 0
        while len(candidates) < int(k_target) and retry < int(k_target) * 4 and len(src_list) > 0:
            p = src_list[retry % len(src_list)]
            path = build_path((int(p[0]), int(p[1])))
            retry += 1
            if not path:
                continue
            edges, turns = state.maps.path_to_edge_ids_and_turns(path)
            key = tuple(int(x) for x in edges.detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(edges)
            turns_l.append(int(turns))

        return candidates, turns_l

    def _bench_method_algorithm(
        self,
        *,
        algorithm: str,
        samples: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        warmup: int,
        rounds: int,
    ) -> Tuple[float, float]:
        if self.env is None:
            return float("inf"), 0.0
        state = self.env.get_state()
        free_map = (~state.maps.blocked_static).to(dtype=torch.bool, device=state.device)
        k_bench = max(1, min(int(self.config.candidate_k), 4))

        for _ in range(max(0, int(warmup))):
            for src_ports, dst_ports in samples:
                self._build_candidates(
                    state=state,
                    free_map=free_map,
                    src_ports=src_ports,
                    dst_ports=dst_ports,
                    k_target=k_bench,
                    algorithm=algorithm,
                    rng=None,
                )

        times: List[float] = []
        for _ in range(max(1, int(rounds))):
            t0 = time.perf_counter()
            for src_ports, dst_ports in samples:
                self._build_candidates(
                    state=state,
                    free_map=free_map,
                    src_ports=src_ports,
                    dst_ports=dst_ports,
                    k_target=k_bench,
                    algorithm=algorithm,
                    rng=None,
                )
            times.append((time.perf_counter() - t0) * 1000.0)

        if not times:
            return float("inf"), 0.0
        mean = float(statistics.mean(times))
        std = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
        return mean, std

    def _resolve_algorithms_static(self, *, active_methods: Sequence[str]) -> None:
        self._selected_algorithms = {}
        for method in active_methods:
            self._selected_algorithms[method] = self._candidate_algorithms(method=method)[0]

        logger.info("=== lane route algorithm selection (mode=static) ===")
        for method in active_methods:
            logger.info("  %-10s -> %s", method, self._selected_algorithms[method])
        logger.info("lane route algorithm selection done")

    def _resolve_algorithms_benchmark(
        self,
        *,
        active_methods: Sequence[str],
        samples: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        warmup = max(0, int(self.config.benchmark_warmup))
        rounds = max(1, int(self.config.benchmark_rounds))
        self._selected_algorithms = {}

        logger.info("=== lane route algorithm selection (mode=benchmark) ===")
        logger.info("  benchmarking per method (warmup=%d, rounds=%d)", warmup, rounds)

        for method in active_methods:
            method_samples = samples.get(method, [])
            candidates = self._candidate_algorithms(method=method)
            if len(candidates) == 1:
                self._selected_algorithms[method] = candidates[0]
                logger.info("  %-10s | %s (single candidate)", method, candidates[0])
                continue
            if len(method_samples) == 0:
                self._selected_algorithms[method] = candidates[0]
                logger.info("  %-10s | %s (no sample flow)", method, candidates[0])
                continue

            perf: Dict[str, Tuple[float, float]] = {}
            for algo in candidates:
                mean_ms, std_ms = self._bench_method_algorithm(
                    algorithm=algo,
                    samples=method_samples,
                    warmup=warmup,
                    rounds=rounds,
                )
                perf[algo] = (mean_ms, std_ms)

            parts = [f"{a} {perf[a][0]:.3f}(+-{perf[a][1]:.3f})" for a in candidates]
            logger.info("  %-10s | %s", method, " | ".join(parts))
            best = min(candidates, key=lambda a: perf[a][0])
            self._selected_algorithms[method] = best

        logger.info("  selected lane algorithms:")
        for method in active_methods:
            logger.info("  %-10s -> %s", method, self._selected_algorithms[method])
        logger.info("lane route algorithm selection done")

    def _resolve_algorithms(self) -> None:
        if self.env is None:
            self._selected_algorithms = {}
            return
        mode = self._config_route_algorithm_selection()
        algo_cfg = self._config_route_algorithm()
        samples = self._discover_flow_samples()
        active_methods = [m for m in self._ALL_METHODS if len(samples[m]) > 0]
        if not active_methods:
            active_methods = list(self._ALL_METHODS)

        if mode == "benchmark" and algo_cfg == "auto":
            self._resolve_algorithms_benchmark(active_methods=active_methods, samples=samples)
            return

        # Explicit algorithm does not need benchmark; auto+static uses rule-based defaults.
        self._resolve_algorithms_static(active_methods=active_methods)

    def _algorithm_for_flow(self, *, src_ports: torch.Tensor) -> str:
        method = self._method_key(src_ports=src_ports)
        algo = self._selected_algorithms.get(method, None)
        if algo is not None:
            return algo
        algo = self._candidate_algorithms(method=method)[0]
        self._selected_algorithms[method] = algo
        return algo

    def build_action_space(self) -> ActionSpace:
        if self.env is None:
            raise RuntimeError("LaneAdapter is not bound to env")
        state = self.env.get_state()
        flow_idx = state.current_flow_index()
        if flow_idx is None:
            return self._empty_action_space(flow_index=-1, device=state.device)

        src_ports, dst_ports = state.flow.valid_ports(flow_idx)
        if src_ports.numel() == 0 or dst_ports.numel() == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        free_map = (~state.maps.blocked_static).to(dtype=torch.bool, device=state.device)
        k_target = max(1, int(self.config.candidate_k))
        algorithm = self._algorithm_for_flow(src_ports=src_ports)
        candidates, turns_l = self._build_candidates(
            state=state,
            free_map=free_map,
            src_ports=src_ports,
            dst_ports=dst_ports,
            k_target=k_target,
            algorithm=algorithm,
            rng=self._rng,
        )

        if len(candidates) == 0:
            return self._empty_action_space(flow_index=flow_idx, device=state.device)

        k = len(candidates)
        lmax = max(int(c.numel()) for c in candidates)
        edge_idx = torch.zeros((k, lmax), dtype=torch.long, device=state.device)
        edge_mask = torch.zeros((k, lmax), dtype=torch.bool, device=state.device)
        path_len = torch.zeros((k,), dtype=torch.float32, device=state.device)
        turns_t = torch.tensor(turns_l, dtype=torch.float32, device=state.device)

        for i, edges in enumerate(candidates):
            n = int(edges.numel())
            if n > 0:
                edge_idx[i, :n] = edges
                edge_mask[i, :n] = True
            path_len[i] = float(n)

        rev = state.maps.reverse_edge_lut[edge_idx]
        reverse_hit = (state.maps.lane_dir_flat[rev] & edge_mask).any(dim=1)
        edge_ok = (state.maps.edge_valid_flat[edge_idx] | (~edge_mask)).all(dim=1)
        valid = (~reverse_hit) & edge_ok & (edge_mask.any(dim=1))

        costs = self.env.reward_composer.delta_batch(
            state,
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            candidate_turns=turns_t,
        )

        return ActionSpace(
            flow_index=int(flow_idx),
            candidate_edge_idx=edge_idx,
            candidate_edge_mask=edge_mask,
            valid_mask=valid,
            candidate_path_len=path_len,
            candidate_turns=turns_t,
            candidate_cost=costs,
        )

    def resolve_action(self, action_idx: int, action_space: ActionSpace) -> Optional[LaneRoute]:
        i = int(action_idx)
        k = int(action_space.valid_mask.shape[0])
        if i < 0 or i >= k:
            return None
        if not bool(action_space.valid_mask[i].item()):
            return None

        edges = action_space.candidate_edge_idx[i][action_space.candidate_edge_mask[i]]
        turns = int(action_space.candidate_turns[i].item()) if action_space.candidate_turns is not None else 0
        path_len = float(action_space.candidate_path_len[i].item()) if action_space.candidate_path_len is not None else float(edges.numel())
        return LaneRoute(
            flow_index=int(action_space.flow_index),
            candidate_index=i,
            edge_indices=edges,
            path_length=path_len,
            turns=turns,
        )
