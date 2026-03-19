"""Route planning module using Dijkstra/A* algorithm.

배치 완료된 env를 기반으로 설비 간 물류 동선을 계산합니다.

사용법:
    planner = RoutePlanner(env)
    planner = RoutePlanner(env, algorithm="astar")  # A* 사용
    result = planner.plan("A", "B")
    results = planner.plan_all()
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from envs.action import EnvAction


@dataclass
class RouteResult:
    """단일 경로 탐색 결과."""
    src_group: str
    dst_group: str
    src_exit: Tuple[int, int]      # (x, y) 출발점 (src의 exit)
    dst_entry: Tuple[int, int]     # (x, y) 도착점 (dst의 entry)
    path: Optional[List[Tuple[int, int]]]  # 경로 좌표 리스트, None이면 경로 없음
    cost: float                    # 경로 비용 (거리)
    success: bool                  # 경로 탐색 성공 여부


def _find_path(
    occ_blocked,
    static_blocked,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = False,
    algorithm: str = "dijkstra",
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """Dijkstra 또는 A* 알고리즘으로 경로 탐색 (내부 구현용).
    
    Args:
        occ_blocked: tensor, 설비 본체 (시작/끝점 허용, 중간 회피)
        static_blocked: tensor, forbidden_areas (항상 회피)
        start: (x, y) 시작점
        goal: (x, y) 도착점
        allow_diagonal: 대각선 이동 허용 여부
        algorithm: "dijkstra" 또는 "astar"
    
    Returns:
        (path, cost) 튜플
    """
    H, W = occ_blocked.shape
    
    sx, sy = start
    gx, gy = goal
    
    # 휴리스틱 함수 (A*용)
    def heuristic(x: int, y: int) -> float:
        if algorithm == "astar":
            if allow_diagonal:
                # 유클리드 거리 (대각선 허용 시)
                return ((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
            else:
                # 맨해튼 거리
                return abs(gx - x) + abs(gy - y)
        return 0.0  # Dijkstra는 휴리스틱 없음
    
    # 범위 체크
    if not (0 <= sx < W and 0 <= sy < H):
        return None, float('inf')
    if not (0 <= gx < W and 0 <= gy < H):
        return None, float('inf')
    
    # 시작/도착점이 forbidden_area면 실패 (설비 본체는 허용)
    if static_blocked[sy, sx].item():
        return None, float('inf')
    if static_blocked[gy, gx].item():
        return None, float('inf')
    
    # 이동 방향 정의
    if allow_diagonal:
        directions = [
            (0, 1, 1.0), (0, -1, 1.0), (-1, 0, 1.0), (1, 0, 1.0),
            (-1, 1, 1.414), (1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414),
        ]
    else:
        directions = [(0, 1, 1.0), (0, -1, 1.0), (-1, 0, 1.0), (1, 0, 1.0)]
    
    # Dijkstra / A*
    # pq: (f, g, x, y) where f = g + h (A*) or f = g (Dijkstra)
    g_start = 0.0
    f_start = g_start + heuristic(sx, sy)
    pq = [(f_start, g_start, sx, sy)]
    dist = {(sx, sy): 0.0}  # g 값 (실제 비용)
    prev = {(sx, sy): None}
    
    while pq:
        f, g, x, y = heapq.heappop(pq)
        
        if (x, y) == (gx, gy):
            # 경로 복원
            path = []
            cur = (gx, gy)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, g  # 실제 비용 반환
        
        if g > dist.get((x, y), float('inf')):
            continue
        
        for dx, dy, move_cost in directions:
            nx, ny = x + dx, y + dy
            
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            
            # forbidden_area는 항상 회피
            if static_blocked[ny, nx].item():
                continue
            
            # 설비 본체 회피 (단, 시작점과 도착점은 예외)
            if (nx, ny) != (gx, gy) and (nx, ny) != (sx, sy) and occ_blocked[ny, nx].item():
                continue
            
            new_g = g + move_cost
            if new_g < dist.get((nx, ny), float('inf')):
                dist[(nx, ny)] = new_g
                prev[(nx, ny)] = (x, y)
                new_f = new_g + heuristic(nx, ny)
                heapq.heappush(pq, (new_f, new_g, nx, ny))
    
    return None, float('inf')


class RoutePlanner:
    """물류 동선 계획 클래스.
    
    사용법:
        planner = RoutePlanner(env)
        planner = RoutePlanner(env, algorithm="astar")  # A* 사용
        result = planner.plan("A", "B")
        results = planner.plan_all()
    """
    
    def __init__(self, env, allow_diagonal: bool = False, algorithm: str = "dijkstra"):
        """
        Args:
            env: FactoryLayoutEnv 인스턴스 (배치 완료 상태)
            allow_diagonal: 대각선 이동 허용 여부
            algorithm: "dijkstra" 또는 "astar"
        """
        self.env = env
        self.allow_diagonal = allow_diagonal
        self.algorithm = algorithm
        
        # 충돌맵 캐싱
        self._occ_blocked = env.get_maps().occ_invalid
        self._static_blocked = env.get_maps().static_invalid
    
    def _get_io_coords(self, gid: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """설비의 entry/exit 좌표 계산.
        
        Returns:
            (entry, exit) 튜플, 각각 (x, y)
        """
        if gid not in self.env.get_state().placed:
            raise ValueError(f"Group {gid} is not placed")

        p = self.env.get_state().placements.get(gid, None)
        if p is None:
            raise ValueError(f"placement missing for gid={gid!r}")

        entries = list(getattr(p, "entries", []))
        exits = list(getattr(p, "exits", []))
        if entries:
            entry = (int(round(float(entries[0][0]))), int(round(float(entries[0][1]))))
        else:
            cx = 0.5 * (float(getattr(p, "min_x")) + float(getattr(p, "max_x")))
            cy = 0.5 * (float(getattr(p, "min_y")) + float(getattr(p, "max_y")))
            entry = (int(round(cx)), int(round(cy)))
        if exits:
            exit_ = (int(round(float(exits[0][0]))), int(round(float(exits[0][1]))))
        else:
            cx = 0.5 * (float(getattr(p, "min_x")) + float(getattr(p, "max_x")))
            cy = 0.5 * (float(getattr(p, "min_y")) + float(getattr(p, "max_y")))
            exit_ = (int(round(cx)), int(round(cy)))

        return entry, exit_
    
    def plan(self, src_gid: str, dst_gid: str) -> RouteResult:
        """단일 경로 계획.
        
        Args:
            src_gid: 출발 그룹 ID
            dst_gid: 도착 그룹 ID
        
        Returns:
            RouteResult 객체
        """
        # IO 좌표 계산
        try:
            _, src_exit = self._get_io_coords(src_gid)
            dst_entry, _ = self._get_io_coords(dst_gid)
        except ValueError:
            return RouteResult(
                src_group=src_gid,
                dst_group=dst_gid,
                src_exit=(0, 0),
                dst_entry=(0, 0),
                path=None,
                cost=float('inf'),
                success=False,
            )
        
        # 경로 탐색
        path, cost = _find_path(
            self._occ_blocked,
            self._static_blocked,
            src_exit,
            dst_entry,
            self.allow_diagonal,
            self.algorithm,
        )
        
        return RouteResult(
            src_group=src_gid,
            dst_group=dst_gid,
            src_exit=src_exit,
            dst_entry=dst_entry,
            path=path,
            cost=cost,
            success=(path is not None),
        )
    
    def plan_all(self) -> List[RouteResult]:
        """env의 모든 flow에 대해 경로 계획.
        
        Returns:
            RouteResult 리스트
        """
        results = []
        for src_gid, dst_dict in self.env.group_flow.items():
            for dst_gid, weight in dst_dict.items():
                result = self.plan(src_gid, dst_gid)
                results.append(result)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """경로 계획 결과 요약."""
        results = self.plan_all()
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count
        total_cost = sum(r.cost for r in results if r.success)
        failed_flows = [(r.src_group, r.dst_group) for r in results if not r.success]
        
        return {
            "total_flows": len(results),
            "success_count": success_count,
            "fail_count": fail_count,
            "total_cost": total_cost,
            "failed_flows": failed_flows,
        }


if __name__ == "__main__":
    """사용 예시."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("postprocess", 1)[0])
    
    from envs.env_loader import load_env
    from envs.env_visualizer import plot_layout
    from postprocess.output import print_summary
    
    # 1. env 로드
    config_path = "envs/env_configs/clearance_03.json"
    print(f"Loading: {config_path}")
    env = load_env(config_path).env
    
    # 2. 설비 배치
    print("\n[2] Placing facilities...")
    env.reset()
    step = 0
    while env.get_state().remaining:
        gid = env.get_state().remaining[0]
        cx = env.grid_width // 2 + step * 50
        cy = env.grid_height // 2 + step * 30
        
        placed = False
        for dx in range(-200, 201, 10):
            for dy in range(-200, 201, 10):
                x, y = cx + dx, cy + dy
                _spec = env.group_specs[gid]
                _w0, _h0 = _spec.rotated_size(0)
                _x_c = float(x) + float(_w0) / 2.0
                _y_c = float(y) + float(_h0) / 2.0
                _gid_r, _pl_r = env.resolve_action(EnvAction(gid=gid, x_c=_x_c, y_c=_y_c))
                if _pl_r is None:
                    continue
                _obs, _reward, _terminated, _truncated, info = env.step_action(
                    EnvAction(gid=gid, x_c=_x_c, y_c=_y_c)
                )
                if info.get("reason") == "placed":
                    print(f"    {gid} at ({x}, {y})")
                    placed = True
                    break
            if placed:
                break
        if not placed:
            print(f"    Failed: {gid}")
            break
        step += 1
    
    # 3. 경로 계획
    print("\n[3] Planning routes...")
    planner = RoutePlanner(env, algorithm="astar")
    results = planner.plan_all()
    
    for r in results:
        status = "OK" if r.success else "FAIL"
        path_len = len(r.path) if r.path else 0
        print(f"    {r.src_group}->{r.dst_group}: {status}, cost={r.cost:.1f}, len={path_len}")
    
    # 4. 요약
    print("\n[4] Summary:")
    print_summary(planner)
    
    # 5. 시각화
    print("\n[5] Visualizing...")
    plot_layout(env, routes=results)
