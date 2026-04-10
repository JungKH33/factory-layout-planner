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
python train.py --mode maskplace --env-json envs/env_configs/basic_01.json --device cuda
python train_torchrl.py --env-json envs/env_configs/basic_01.json --device cuda

# Preprocess raw facility JSON → env config
python -m preprocess.to_env input.json output.json

# Quick smoke tests (individual modules have __main__ blocks)
python -m envs.env
python -m envs.placement.static
python -m agents.placement.greedy.adapter_v3
```

There are no automated tests or a lint config. Validation is done by running the demo `__main__` blocks in individual modules.

## Architecture Overview

### Layered stack (bottom → top)

```
envs/env_configs/*.json                      ← problem definition (grid, facilities, flow, zones)
       ↓  env_loader.py
envs/env.py                                  ← FactoryLayoutEnv (single production Gymnasium env)
       ↓
agents/placement/greedy/adapter_v3.py        ← generates ActionSpace(poses[K,3], mask[K]) from engine state
agents/placement/greedy/agent.py             ← policy: argmin(Δcost) over candidates
search/mcts.py                               ← optional tree search over adapter
       ↓
trace/explorer.py                            ← Explorer(engine, adapter, agent, search) — unified decision explorer
       ↓
inference.py                                 ← episode loop, visualization, output (uses agents.registry)
```

### Agents package (`agents/`)

Each method lives in its own subpackage with its agent + adapter bundled together:

```
agents/
  base.py              ← Agent(Protocol), OrderingAgent(Protocol), BaseAdapter(ABC)
  registry.py          ← create(method, agent=...) factory + compatibility matrix
  placement/
    greedy/            ← GreedyAgent + GreedyAdapter/V2/V3
      agent.py, adapter.py, adapter_v2.py, adapter_v3.py
    maskplace/         ← MaskPlaceAgent + MaskPlaceAdapter + MaskPlaceModel
      agent.py, adapter.py, model.py
    alphachip/         ← AlphaChipAgent + AlphaChipAdapter + model/gnn
      agent.py, adapter.py, model.py, gnn.py
  ordering/            ← DifficultyOrderingAgent
    difficulty.py
```

**Registry**: `agents.registry.create(method, agent=..., agent_kwargs=..., adapter_kwargs=...)` creates valid (agent, adapter) pairs. GreedyAgent is compatible with all adapters; MaskPlace/AlphaChip agents require their specific adapters.

**Observation ownership**: Each adapter defines its own `build_observation()` format. The engine does NOT build observations — `step_action()` and `reset()` return `{}` for obs. Call order: `build_observation()` → `build_action_space()` (mask is created in obs, reused in action_space).

**Adapters** are **not** Gymnasium envs. They are pure stateless-ish adapters that:
- `build_observation()` → model-specific dict (greedy returns `{}`, AlphaChip returns graph tensors, etc.)
- `build_action_space()` → `ActionSpace(poses, mask)` from current engine state
- `decode_action(index, action_space)` → `EnvAction(gid, x, y, orient)`

The pipeline calls `engine.step_action(action)` directly — adapters never step the env themselves.

For RL training, `AdapterGymEnv` (`gym_env.py`) wraps engine + adapter into a `gym.Env` with proper `step()`/`reset()` that returns adapter observations with `action_mask`.

### Env internal state

`FactoryLayoutEnv` owns an `EnvState` (from `envs/state/`) containing:

- **`GridMaps`** (`envs/state/maps.py`): all `[H,W]` boolean/float tensors

| Tensor | Meaning |
|---|---|
| `static_invalid` | forbidden areas (permanent, shared on copy) |
| `occ_invalid` | body footprints of placed facilities |
| `clear_invalid` | clearance halos of placed facilities |
| `zone_invalid` | constraint zones for the *next* group (weight/height/dry/allowed_areas) |
| `invalid` | `static \| occ \| zone` — what `is_placeable` checks |

- **`FlowGraph`** (`envs/state/flow.py`): IO port caches, flow weight matrix (incrementally updated)
- **`GroupPlacement`** (`envs/placement/base.py`): placed facility geometry (x_bl, y_bl, rot, ports, clearances)

State copy/restore: `engine.get_state().copy()` / `engine.set_state(state)`. Static tensors are shared by reference on copy (cheap for MCTS snapshots); only runtime tensors (occ, clear, zone, invalid) are cloned.

### Reward signal

Step reward = `-(delta_cost) / reward_scale`

`RewardComposer` (`envs/reward/`) holds named components:
- **FlowReward**: weighted L1 (Manhattan) distance between IO ports. `delta()` uses `[M,1,C,1,2] - [1,T,1,P,2]` broadcasting for batch candidate evaluation.
- **AreaReward**: HPWL compactness `0.5 * ((max_x-min_x) + (max_y-min_y))`
- **TerminalReward**: failure penalty = `penalty_weight * remaining_area_ratio / reward_scale`

`spec.cost_batch(gid, poses, state, reward, per_variant)` is the vectorized incremental cost used by adapters and agents (never full `cost()` per candidate).

### Search (MCTS / Beam)

Search operates at the **adapter** level. Each MCTS node stores an `EnvState` copy via `engine.get_state().copy()`. The agent provides priors (softmax of `-Δcost`) and leaf values. Rollouts use the greedy agent to a configurable depth.

**Variant expansion**: when `expand_variants=True` on an adapter (BaseAdapter param), each center candidate is expanded into `(center, variant_index)` pairs. `max_variants` limits how many variants per center are kept. The adapter's `_apply_variant_expansion()` computes per-variant costs in a single `cost_batch(per_variant=True)` call and selects the top-K `(center, variant)` pairs by cost. `ActionSpace.variant_indices` carries the per-action variant index; `decode_action()` produces `EnvAction` with explicit `variant_index`. Search algorithms (MCTS/Beam) treat these as regular actions with no special variant branching.

Top-K terminal tracking uses stateless utility functions (`track_terminal`, `collect_top_k` in `search/base.py`) with local heaps per `select()` call. `SearchOutput` carries the final action, visits, values, and top-K results.

### trace/ — Unified Decision Explorer

```
trace/
  schema.py        ← Signal, Snapshot, DecisionNode, DecisionTree, TraceEvent
  explorer.py      ← Explorer: predict, step, undo/redo, branch, auto_play, events
  query.py         ← TraceQuery: best_path, top_k_paths, signal_agreement, summarize
```

`Explorer` replaces the old `DecisionPipeline` and WebUI `HistoryEntry`:

```python
exp = Explorer(engine, adapter, agent, search, ordering_agent)
exp.reset(options=loaded.reset_kwargs)

# Predictions (no side-effects on engine state)
agent_signal = exp.predict_agent()       # → Signal with scores, value
search_signal = exp.predict_search()     # → Signal with visits, values, top_k

# Execution
child = exp.step(action_index, chosen_by="human")
child = exp.step_with("agent")           # step using agent's recommendation
results = exp.auto_play(source="agent")  # run to completion

# Navigation (tree-based undo/redo)
exp.undo()
exp.redo()
exp.goto(node_id)
exp.branch("name")

# Events (for WebUI streaming)
exp.on(callback)   # callback receives TraceEvent
```

Search algorithms return `SearchOutput` (action, visits, values, iterations, top_k). `select()` accepts `progress_fn` callback for real-time streaming.

### `envs/env_configs/*.json` schema

```json
{
  "grid": { "width": 500, "height": 500, "grid_size": 1.0 },
  "env": { "default_weight": 10.0, "default_height": 20.0, "default_dry": 0.0,
           "reward_scale": 100.0, "penalty_weight": 50000.0 },
  "groups": { "<gid>": { "width", "height", "rotatable", "ent_rel_x/y", "exi_rel_x/y",
                          "facility_clearance_*", "facility_weight/height/dry", "allowed_areas",
                          "variants": [{"width", "height", "entries_rel", "exits_rel", ...}, ...] } },
  "flow": [["<src>", "<dst>", <weight>], ...],
  "zones": { "forbidden", "weight_areas", "height_areas", "dry_areas", "placement_areas" },
  "reset": { "initial_placements": {"<gid>": [x_c, y_c, variant_index, source_index]}, "remaining_order": [...] }
}
```

Zone constraint logic: op is facility requirement (e.g. `height<=30` → facility height must be ≤ zone value; `weight>=10` → facility weight must be ≥ zone value). Valid when `facility_value op zone_value`.

`env_loader.load_env(path, device)` returns `LoadedEnv(env, reset_kwargs)`.

### Postprocessing

- **`RoutePlanner`** (`postprocess/pathfinder.py`): Dijkstra/A* on grid, routing flows between placed facilities.
- **`DynamicGroupGenerator`** (`postprocess/dynamic_group.py`): BFS frontier expansion for storage racks with variable footprint.
- **`DynamicPlanner`** (`envs/placement/dynamic.py`): Conv2D-based validity expansion for dynamic groups.

### WebUI session model

`webui/session.py` manages per-session `Explorer` instances. Each `Session` wraps an `Explorer` that handles state management, undo/redo (via `DecisionTree`), and search execution. FastAPI backend exposes REST for step/undo/search and WebSocket for streaming search progress via `Explorer` events.

## Key Conventions

- **Coordinate system**: bottom-left origin `(x_bl, y_bl)`, rotation in `{0, 90, 180, 270}` degrees CCW. Tensor indexing is `tensor[y, x]`. Port coords (`ent_rel_x/y`, `exi_rel_x/y`) are BL-relative.
- **Variant model**: `Variant` (base, `envs/placement/base.py`) → `StaticVariant` (static impl, `envs/placement/static.py`). `GroupSpec.variants` returns all placement variants (rotation/mirror/shape combinations). Each variant has a `source_index` tracking which original shape definition it came from (0 for single-shape groups). Multi-shape groups define `"variants"` array in JSON — each entry is a distinct source shape with its own width/height/ports/clearance. All source shapes × rotation × mirror combinations are flattened into a single `_variants` list. `_source_ranges` maps `source_index → (start, end)` for filtering. Adapters work with center coordinates only; the engine resolves variant at step time. `EnvAction.variant_index` can pin a specific variant; `EnvAction.source_index` filters to variants from a given source shape.
- **Grid units**: integer cells. `grid_size` (meters/cell) is only for display/output.
- **Device ownership**: engine's `device` is set at construction; all tensors follow it. Adapters inherit device on `bind()`.
- **`inference.py` config**: module-level constants (`ENV_JSON`, `WRAPPER_MODE`, `AGENT_MODE`, `SEARCH_MODE`, etc.) — no CLI args.
- **Current facility selection**: adapters usually build candidates for `remaining[0]` (ordering agents can reorder this list), but `EnvAction` itself now requires explicit `gid`.
- **Delta pattern**: all candidate evaluation uses `cost_batch()` (vectorized incremental) rather than full `cost()` per candidate. `cost_batch(per_variant=True)` and `placeable_batch(per_variant=True)` return `[N, V]` per-variant results.
- **Static-sharing on copy**: `GridMaps.copy()` shares static tensors by reference, clones only runtime tensors. Makes MCTS snapshots cheap.
