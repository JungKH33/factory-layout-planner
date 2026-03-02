# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Reinforcement learning system for optimizing factory facility layouts on a 2D grid. Agents place facilities one-by-one (sequential MDP) to minimize material flow distance while respecting spatial constraints (clearances, forbidden zones, weight/height/dry areas, placement-area restrictions).

## Commands

```bash
# Inference (edit module-level constants at top of file to configure)
python inference.py

# WebUI (interactive placement at http://localhost:8000)
python run_webui.py

# Training
python train.py --mode maskplace --env-json env_configs/basic_01.json --device cuda
python train_torchrl.py --env-json env_configs/basic_01.json --device cuda

# Preprocess raw facility JSON → env config
python -m preprocess.to_env input.json output.json

# Quick smoke test for env_new
python -m envs.env_new          # runs __main__ timing demo
python -m envs.static           # runs __main__ batch timing demo

# Quick module import check
python -c "from envs.env_new import FactoryLayoutEnv, PlacementBase, GridMaps; print('OK')"
```

There are no automated tests or a lint config. Validation is done by running the demo `__main__` blocks in individual modules.

## Architecture Overview

### Layered stack (bottom → top)

```
env_configs/*.json          ← problem definition (grid, facilities, flow, zones)
       ↓  json_loader.py
envs/env.py                 ← FactoryLayoutEnv (Gymnasium env, primary production env)
envs/env_new.py             ← refactored env (in-progress replacement for env.py)
       ↓
envs/wrappers/greedyv3.py   ← action-space wrapper: Discrete(K) over Top-K candidates
       ↓
agents/greedy.py            ← policy: argmin(Δcost) over candidates
search/mcts.py              ← optional tree search wrapping the wrapper env
       ↓
pipeline.py                 ← DecisionPipeline(agent, search) — ties it all together
       ↓
inference.py                ← episode loop, visualization, output
```

### Two env files: `env.py` vs `env_new.py`

- **`env.py`** is the production environment used by wrappers, training, and most of the codebase.
- **`env_new.py`** is a refactored version (in progress). It is *not* yet wired into wrappers or training. Both expose the same public API (`step_action`, `reset`, `estimate_delta_obj`, `cal_obj`, `get_snapshot`/`set_snapshot`).
- `env_new.py` introduces `PlacementBase` (common placement contract) and `GridMaps` (encapsulates all map tensors and update logic that was previously spread across 7 methods in `FactoryLayoutEnv`).

### Env internal map layers

The engine maintains four boolean `[H,W]` tensors composed into a single `invalid` mask:

| Tensor | Meaning |
|---|---|
| `static_invalid` | forbidden areas (permanent) |
| `occ_invalid` | body footprints of placed facilities |
| `clear_invalid` | clearance halos of placed facilities (queried separately from invalid) |
| `zone_invalid` | constraint zones for the *next* group to be placed (changes each step) |
| `invalid` | `static \| occ \| zone` — what `is_placeable` checks |

`zone_invalid` is recomputed after every placement based on `remaining[0]`'s `facility_weight/height/dry` and `allowed_areas` fields.

### Reward signal

Step reward = `-(cost_new - cost_prev) / cost_scale`

`cal_obj()` computes weighted L1 (Manhattan) flow distance between facility entry/exit ports + compactness (HPWL). `estimate_delta_obj()` computes the *incremental* cost change for a batch of candidate placements using tensor ops (used by wrappers and greedy agent without actually stepping the env).

### Wrapper contract

Every wrapper (`GreedyWrapperV3Env`, `AlphaChipWrapperEnv`, `MaskPlaceWrapperEnv`) must produce:
- `obs["action_mask"]`: `BoolTensor[K]` — True = valid action
- `obs["action_xyrot"]`: `LongTensor[K,3]` — `(x_bl, y_bl, rot)` for each candidate

`pipeline.py` reads these to build a `CandidateSet` and route it through `agent.select_action()` or `search.select()`.

### Search (MCTS / Beam)

Search operates at the **wrapper** level, not the engine level. Each MCTS node stores a `get_snapshot()`/`set_snapshot()` copy of the engine state. The agent provides priors (softmax of `-Δcost`) and leaf values (`estimate_terminal_reward()`). Rollouts use the greedy agent to completion.

### `env_configs/*.json` schema

Key fields:
- `grid_width`, `grid_height`, `grid_size` (meters per cell)
- `groups`: dict of facility specs with `width`, `height`, `rotatable`, `ent_rel_x/y`, `exi_rel_x/y`, `clearance_*`, `allowed_areas`, `facility_weight/height/dry`
- `flow_edges`: `[[src, dst, weight], ...]` — directed material flow graph
- `forbidden_areas`, `weight_areas`, `height_areas`, `dry_areas`, `placement_areas`: list of `{"rect": [x0,y0,x1,y1], ...}`

`json_loader.load_env(path, device)` returns `LoadedEnv(env, reset_kwargs)` where `reset_kwargs` carries `initial_positions` if the config has pre-placed facilities.

### Postprocessing

After inference, `postprocess/` handles:
- **`RoutePlanner`** (`pathfinder.py`): Dijkstra/A* on the grid, routing material flows between placed facilities while avoiding bodies but allowing clearance traversal.
- **`DynamicPlanner`** (`dynamic.py` in `envs/`): Frontier BFS expansion for "dynamic" (storage rack) groups whose footprint is not fixed but grows to fill a required capacity. Uses Conv2D-based validity checking and flow-penalty-guided expansion.

### WebUI session model

`webui/session.py` manages per-session `(env, wrapper, agent, search, history)`. History is a stack of `get_snapshot()` dicts enabling undo. The FastAPI backend exposes REST for step/undo/search and WebSocket for streaming search progress.

## Key Conventions

- **Coordinate system**: bottom-left origin `(x_bl, y_bl)`, rotation in `{0, 90, 180, 270}` degrees CCW.
- **Grid units**: integer cells. `grid_size` (meters/cell) is only for display/output — the engine works in pure integer cells.
- **Device ownership**: `StaticGeom` owns a `device` and all its tensor ops run on it. The env inherits device from the geom specs via `json_loader`.
- **`inference.py` config**: controlled by module-level constants at the top (`ENV_JSON`, `WRAPPER_MODE`, `AGENT_MODE`, `SEARCH_MODE`, etc.) — no CLI args.
- **`env_new.py` vs `env.py`**: When extending the refactored env, use `env_new.py`. When touching training/wrappers/inference, use `env.py` until migration is complete.
