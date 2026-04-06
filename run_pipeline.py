from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-stage layout pipeline")
    p.add_argument("--env-json", type=str, required=True)

    p.add_argument("--group-json", type=str, default="results/pipeline/group_placement.json")
    p.add_argument("--lane-json", type=str, default="results/pipeline/group_lane_generation.json")
    p.add_argument("--facility-json", type=str, default="results/pipeline/facility_placement.json")

    p.add_argument("--skip-group", action="store_true")
    p.add_argument("--skip-lane", action="store_true")
    p.add_argument("--skip-facility", action="store_true")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--backend-selection", type=str, default="benchmark")

    p.add_argument("--wrapper-mode", type=str, default="greedyv3")
    p.add_argument("--agent-mode", type=str, default="greedy")
    p.add_argument("--search-mode", type=str, default="mcts")
    p.add_argument("--ordering-mode", type=str, default="none")
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--mcts-sims", type=int, default=1000)
    p.add_argument("--rollout-enabled", action="store_true", default=True)
    p.add_argument("--no-rollout", action="store_false", dest="rollout_enabled")
    p.add_argument("--rollout-depth", type=int, default=10)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--search-depth", type=int, default=5)
    p.add_argument("--expansion-topk", type=int, default=16)
    p.add_argument("--max-expansions", type=int, default=200)

    p.add_argument("--lane-algorithm", type=str, default="astar")
    p.add_argument("--lane-diagonal", action="store_true")
    p.add_argument("--facility-on-missing", type=str, default="warn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    group_json = str(Path(args.group_json))
    lane_json = str(Path(args.lane_json))
    facility_json = str(Path(args.facility_json))

    if not args.skip_group:
        from pipeline.group_placement import GroupPlacementConfig, run_and_save_group_placement

        gp_cfg = GroupPlacementConfig(
            env_json=str(args.env_json),
            device=args.device,
            backend_selection=str(args.backend_selection),
            wrapper_mode=str(args.wrapper_mode),
            agent_mode=str(args.agent_mode),
            search_mode=str(args.search_mode),
            ordering_mode=str(args.ordering_mode),
            max_decisions=int(args.max_decisions),
            mcts_sims=int(args.mcts_sims),
            rollout_enabled=bool(args.rollout_enabled),
            rollout_depth=int(args.rollout_depth),
            beam_width=int(args.beam_width),
            search_depth=int(args.search_depth),
            expansion_topk=int(args.expansion_topk),
            max_expansions=int(args.max_expansions),
        )
        gp_artifact = run_and_save_group_placement(gp_cfg, group_json)
        logger.info("group_placement saved: %s", group_json)
        logger.info("group metrics: %s", gp_artifact.metrics)
    elif not Path(group_json).exists():
        raise FileNotFoundError(f"--skip-group was set, but group artifact does not exist: {group_json}")

    if not args.skip_lane:
        from pipeline.group_lane_generation import GroupLaneGenerationConfig, run_and_save_group_lane_generation

        lane_cfg = GroupLaneGenerationConfig(
            group_placement_json=group_json,
            output_json=lane_json,
            env_json=str(args.env_json),
            device=args.device,
            backend_selection=str(args.backend_selection),
            algorithm=str(args.lane_algorithm),
            allow_diagonal=bool(args.lane_diagonal),
        )
        lane_artifact = run_and_save_group_lane_generation(lane_cfg)
        logger.info("group_lane_generation saved: %s", lane_json)
        logger.info("lane metrics: %s", lane_artifact.metrics)

    if not args.skip_facility:
        from pipeline.facility_placement import FacilityPlacementConfig, run_and_save_facility_placement

        facility_cfg = FacilityPlacementConfig(
            group_placement_json=group_json,
            output_json=facility_json,
            env_json=str(args.env_json),
            on_missing=str(args.facility_on_missing),
        )
        facility_artifact = run_and_save_facility_placement(facility_cfg)
        logger.info("facility_placement saved: %s", facility_json)
        logger.info("facility metrics: %s", facility_artifact.metrics)


if __name__ == "__main__":
    main()
