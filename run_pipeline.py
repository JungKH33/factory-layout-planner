from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-stage layout pipeline")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--workspace", type=str, default="results")

    p.add_argument("--skip-group", action="store_true")
    p.add_argument("--skip-lane", action="store_true")
    p.add_argument("--skip-facility", action="store_true")
    p.add_argument("--skip-simulation", action="store_true")

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
    p.add_argument("--group-top-n", type=int, default=1)

    p.add_argument("--lane-algorithm", type=str, default="astar")
    p.add_argument("--lane-diagonal", action="store_true")
    p.add_argument("--facility-on-missing", type=str, default="warn")
    p.add_argument("--sim-horizon", type=float, default=3600.0)
    p.add_argument("--sim-warmup", type=float, default=300.0)
    p.add_argument("--sim-seed", type=int, default=0)
    p.add_argument("--sim-timeline-step", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    input_path = Path(args.input)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.workspace) / f"{input_path.stem}_{ts}"
    preprocess_dir = run_dir / "preprocess"
    group_dir = run_dir / "group_placement"
    lane_dir = run_dir / "lane_generation"
    facility_dir = run_dir / "facility_placement"
    simulation_dir = run_dir / "simulation"
    for d in (preprocess_dir, group_dir, lane_dir, facility_dir, simulation_dir):
        d.mkdir(parents=True, exist_ok=True)

    group_json = str(group_dir / "group_placement.json")
    logger.info("run workspace: %s", run_dir)
    logger.info(
        "stage output dirs: preprocess=%s, group_placement=%s, lane_generation=%s, facility_placement=%s, simulation=%s",
        preprocess_dir,
        group_dir,
        lane_dir,
        facility_dir,
        simulation_dir,
    )

    from pipeline.preprocess import PreprocessConfig, run_and_save_preprocess

    def _run_stage(stage_name: str, fn):
        logger.info("[stage=%s] START", stage_name)
        start = time.perf_counter()
        try:
            result = fn()
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception("[stage=%s] FAILED elapsed_sec=%.3f", stage_name, elapsed)
            raise
        elapsed = time.perf_counter() - start
        logger.info("[stage=%s] DONE elapsed_sec=%.3f", stage_name, elapsed)
        return result

    pp_cfg = PreprocessConfig(
        input_json=str(args.input),
        output_dir=str(preprocess_dir),
        visualize_dir=str(preprocess_dir),
    )
    pp_artifact = _run_stage("preprocess", lambda: run_and_save_preprocess(pp_cfg))
    env_json_path = Path(pp_artifact.env_json)

    if not args.skip_group:
        from pipeline.group_placement import GroupPlacementConfig, run_and_save_group_placement

        gp_cfg = GroupPlacementConfig(
            env_json=str(env_json_path),
            output_dir=str(group_dir),
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
            top_n=max(1, int(args.group_top_n)),
        )
        _run_stage("group_placement", lambda: run_and_save_group_placement(gp_cfg))
    elif (not args.skip_lane or not args.skip_facility) and not Path(group_json).exists():
        raise FileNotFoundError(
            f"--skip-group was set, but group artifact does not exist: {group_json}"
        )

    if not args.skip_lane:
        from pipeline.lane_generation import LaneGenerationConfig, run_and_save_lane_generation

        lane_cfg = LaneGenerationConfig(
            group_placement_json=group_json,
            output_dir=str(lane_dir),
            env_json=str(env_json_path),
            device=args.device,
            backend_selection=str(args.backend_selection),
            algorithm=str(args.lane_algorithm),
            allow_diagonal=bool(args.lane_diagonal),
        )
        _run_stage("lane_generation", lambda: run_and_save_lane_generation(lane_cfg))

    if not args.skip_facility:
        from pipeline.facility_placement import FacilityPlacementConfig, run_and_save_facility_placement

        facility_cfg = FacilityPlacementConfig(
            group_placement_json=group_json,
            output_dir=str(facility_dir),
            env_json=str(env_json_path),
            on_missing=str(args.facility_on_missing),
        )
        _run_stage("facility_placement", lambda: run_and_save_facility_placement(facility_cfg))

    if not args.skip_simulation:
        from pipeline.simulation import SimulationConfig, run_and_save_simulation

        sim_cfg = SimulationConfig(
            group_placement_json=group_json,
            output_dir=str(simulation_dir),
            env_json=str(env_json_path),
            facility_placement_json=str(facility_dir / "facility_placement.json"),
            lane_generation_json=str(lane_dir / "lane_generation.json"),
            horizon_sec=float(args.sim_horizon),
            warmup_sec=float(args.sim_warmup),
            seed=int(args.sim_seed),
            timeline_step_sec=float(args.sim_timeline_step),
        )
        _run_stage("simulation", lambda: run_and_save_simulation(sim_cfg))


if __name__ == "__main__":
    main()
