"""Route planning output utilities.

경로 계획 결과를 JSON, 시각화용 데이터로 출력합니다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pathfinder import RouteResult, RoutePlanner

logger = logging.getLogger(__name__)


def routes_to_dict(results: List[RouteResult]) -> List[Dict[str, Any]]:
    """RouteResult 리스트를 dict 리스트로 변환.
    
    Args:
        results: RouteResult 리스트
    
    Returns:
        JSON 직렬화 가능한 dict 리스트
    """
    output = []
    for r in results:
        output.append({
            "src_group": r.src_group,
            "dst_group": r.dst_group,
            "src_exit": list(r.src_exit),
            "dst_entry": list(r.dst_entry),
            "path": [list(p) for p in r.path] if r.path else None,
            "cost": r.cost if r.cost != float('inf') else None,
            "success": r.success,
        })
    return output


def save_routes_json(
    results: List[RouteResult],
    output_path: str,
    indent: int = 2,
) -> None:
    """경로 결과를 JSON 파일로 저장.
    
    Args:
        results: RouteResult 리스트
        output_path: 출력 파일 경로
        indent: JSON 들여쓰기
    """
    data = {
        "routes": routes_to_dict(results),
        "summary": {
            "total": len(results),
            "success": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_cost": sum(r.cost for r in results if r.success and r.cost != float('inf')),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def routes_to_polylines(results: List[RouteResult]) -> List[Dict[str, Any]]:
    """시각화용 polyline 데이터 생성.
    
    Args:
        results: RouteResult 리스트
    
    Returns:
        [
            {
                "id": "A->B",
                "points": [[x1,y1], [x2,y2], ...],
                "success": True/False,
            },
            ...
        ]
    """
    polylines = []
    for r in results:
        polyline = {
            "id": f"{r.src_group}->{r.dst_group}",
            "src_group": r.src_group,
            "dst_group": r.dst_group,
            "points": [list(p) for p in r.path] if r.path else [],
            "success": r.success,
        }
        polylines.append(polyline)
    return polylines


def print_summary(planner: RoutePlanner) -> None:
    """경로 계획 결과 요약 출력.
    
    Args:
        planner: RoutePlanner 인스턴스
    """
    summary = planner.get_summary()

    logger.info("=" * 50)
    logger.info("Route Planning Summary")
    logger.info("=" * 50)
    logger.info("Total flows: %s", summary["total_flows"])
    logger.info("Success: %s", summary["success_count"])
    logger.info("Failed: %s", summary["fail_count"])
    logger.info("Total cost: %.2f", summary["total_cost"])

    if summary['failed_flows']:
        logger.info("Failed flows:")
        for src, dst in summary['failed_flows']:
            logger.info("  - %s -> %s", src, dst)
    logger.info("=" * 50)
