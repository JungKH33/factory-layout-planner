<p align="center">
  <img src="assets/logo.png" alt="Bichon layout planner" width="620" />
</p>

# Factory Layout Optimizer

A reinforcement learning system for optimizing factory facility layouts on a 2D grid. An RL agent places facilities one-by-one (sequential MDP) to minimize material flow distance while respecting spatial constraints such as clearances, forbidden zones, weight/height limits, dry-area requirements, and placement-area restrictions.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Environment Configuration (JSON)](#environment-configuration-json)
  - [Grid Section](#grid-section)
  - [Env Section](#env-section)
  - [Groups Section](#groups-section)
  - [Flow Section](#flow-section)
  - [Zones Section](#zones-section)
  - [Reset Section](#reset-section)
- [Agents and Adapters](#agents-and-adapters)
- [Search Algorithms](#search-algorithms)
- [Training](#training)
- [Inference](#inference)
- [Web UI](#web-ui)
- [Postprocessing](#postprocessing)
- [Key Concepts Explained](#key-concepts-explained)

---

## Quick Start

```bash
# Run inference (configure module-level constants in inference.py)
python inference.py

# Launch interactive Web UI at http://localhost:8000
python run_webui.py

# Train with PPO (Tianshou backend)
python train.py --mode maskplace --env-json envs/env_configs/basic_01.json --device cuda

# Train with PPO (TorchRL backend)
python train_torchrl.py --env-json envs/env_configs/basic_01.json --device cuda

# Preprocess raw facility JSON into env config
python -m preprocess.to_env input.json output.json

# Smoke tests (individual modules have __main__ blocks)
python -m envs.env
python -m envs.placement.static
python -m agents.placement.greedy.adapter_v3
```

---

## Architecture Overview

```
envs/env_configs/*.json                      <-- problem definition
       |  env_loader.py
envs/env.py                                  <-- FactoryLayoutEnv (Gymnasium env)
       |
agents/placement/greedy/adapter_v3.py        <-- ActionSpace(poses[K,3], mask[K])
agents/placement/greedy/agent.py             <-- policy: argmin(delta_cost)
search/mcts.py                               <-- optional tree search
       |
pipeline.py                                  <-- DecisionPipeline(adapter, agent, search)
       |
inference.py                                 <-- episode loop, visualization, output
```

### Pipeline Flow

```
1. ordering_agent.reorder(env)             # optional: reorder remaining facilities
2. adapter.build_observation()             # model-specific observation
3. adapter.build_action_space()            # ActionSpace with candidate poses + validity mask
4. search.select(obs, agent, action_space) # or agent.select_action(obs, action_space)
5. adapter.decode_action(index)            # index -> EnvAction(gid, x, y, orient)
6. engine.step_action(action)              # place facility on grid
```

---

## Environment Configuration (JSON)

All problem instances are defined as JSON files in `envs/env_configs/`. A config has these top-level sections:

```json
{
  "grid": { ... },
  "env": { ... },
  "groups": { ... },
  "flow": [ ... ],
  "zones": { ... },
  "reset": { ... }
}
```

### Grid Section

Defines the 2D grid dimensions.

| Field | Type | Required | Description |
|---|---|---|---|
| `width` | int | yes | Grid width in cells |
| `height` | int | yes | Grid height in cells |
| `grid_size` | float | no | Meters per cell (display/export only, default 1.0) |

```json
"grid": { "width": 500, "height": 500, "grid_size": 1.0 }
```

### Env Section

Global environment parameters.

| Field | Type | Default | Description |
|---|---|---|---|
| `reward_scale` | float | 100.0 | Divides raw cost to normalize rewards |
| `penalty_weight` | float | 50000.0 | Terminal failure penalty multiplier |
| `log` | bool | false | Enable detailed logging |

**Reward formula:**
- Step reward: `-(delta_cost) / reward_scale`
- Failure penalty: `-(penalty_weight * remaining_area_ratio) / reward_scale`

### Groups Section

Each key is a facility ID (string). Each value defines geometry, ports, clearance, and zone requirements.

#### Basic Geometry

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"rect"` | `"rect"` for rectangle, `"irregular"` for polygon |
| `width` | int | required | Bounding box width in cells |
| `height` | int | required | Bounding box height in cells |
| `rotatable` | bool | true | Allow 90-degree rotations (0/90/180/270) |
| `mirrorable` | bool | false | Allow horizontal mirroring |

**Example - rectangular facility:**
```json
"A": { "width": 160, "height": 80, "rotatable": true }
```

#### Irregular Facilities (Polygons)

For non-rectangular shapes, set `type` to `"irregular"` and provide `body_polygon`.

| Field | Type | Description |
|---|---|---|
| `body_polygon` | `[[x,y], ...]` | Polygon vertices in bottom-left relative coords |

**Example - L-shaped facility:**
```json
"C": {
  "type": "irregular",
  "width": 60, "height": 40,
  "rotatable": true,
  "body_polygon": [
    [0, 0], [60, 0], [60, 20],
    [20, 20], [20, 40], [0, 40]
  ]
}
```

This defines an L-shape within a 60x40 bounding box:

```
     +-------+
     |       |
     |  C    |   <- upper part: 20 wide x 20 tall
     |       |
+----+       |
|            |   <- lower part: 60 wide x 20 tall
|            |
+------------+
```

#### IO Ports

Ports define where material enters/exits a facility. Coordinates are relative to the bottom-left corner at rotation 0.

**Multi-port format (recommended):**

| Field | Type | Default | Description |
|---|---|---|---|
| `entries_rel` | `[[x,y], ...]` | center | Entry port positions (BL-relative) |
| `exits_rel` | `[[x,y], ...]` | center | Exit port positions (BL-relative) |
| `entry_port_span` | int or string | `1` | Number of entry ports to use: `1`, `2`, ..., or `"all"` |
| `exit_port_span` | int or string | `1` | Number of exit ports to use: `1`, `2`, ..., or `"all"` |

**Legacy single-port format (still supported):**

| Field | Type | Default | Description |
|---|---|---|---|
| `ent_rel_x` | float | center | Entry port X (BL-relative) |
| `ent_rel_y` | float | center | Entry port Y (BL-relative) |
| `exi_rel_x` | float | center | Exit port X (BL-relative) |
| `exi_rel_y` | float | center | Exit port Y (BL-relative) |

**Example:**
```json
"B": {
  "width": 100, "height": 80,
  "entries_rel": [[0, 20], [0, 40], [0, 60]],
  "exits_rel": [[100, 20], [100, 60]],
  "entry_port_span": "all",
  "exit_port_span": 1
}
```

> See [Port Span Selection](#port-span-selection-entry_port_span--exit_port_span) for a detailed explanation with examples.

#### Clearance

Clearance defines the required buffer zone around a facility that other facilities cannot overlap. Clearance zones may overlap each other; only facility bodies must not overlap bodies or clearances of other facilities.

**Option 1 - Uniform (all sides equal):**
```json
"clearance": 10
```

**Option 2 - Directional [Left, Right, Bottom, Top]:**
```json
"clearance_lrtb": [2, 3, 4, 5]
```

**Option 3 - Polygon (irregular facilities only):**
```json
"clearance_polygon": [[x0,y0], [x1,y1], ...]
```

**Example - uniform clearance of 10 cells:**

```
+------ clearance halo (dashed) ------+
|                                      |
|    +---- facility body ----+         |
|    |                       |  10     |
| 10 |       Facility A      |  cells  |
|    |                       |         |
|    +-----------------------+         |
|              10 cells                |
+--------------------------------------+
```

**Clearance rotates with the facility.** For rotation R:
- 0 deg: (L, R, B, T)
- 90 deg: (B, T, R, L)
- 180 deg: (R, L, T, B)
- 270 deg: (T, B, L, R)

#### Zone Constraint Values

Per-facility attribute values for zone-based constraint checking.

| Field | Type | Description |
|---|---|---|
| `zone_values` | dict | Maps constraint name to numeric facility value |

```json
"A": {
  "width": 160, "height": 80,
  "zone_values": { "weight": 3.0, "height": 2.0, "dry": 2.0 }
}
```

> See [Zone Constraints](#zone-constraints) for the validation logic and examples.

### Flow Section

Defines material flow between facilities as weighted directed edges. This is the **primary optimization objective** -- the agent minimizes total `flow_weight * manhattan_distance(exit_port, entry_port)`.

**Format 1 - Edge list (recommended):**
```json
"flow": [
  ["A", "B", 1.0],
  ["B", "C", 0.8],
  ["A", "D", 0.5]
]
```

**Format 2 - Adjacency dict:**
```json
"flow": {
  "A": { "B": 1.0, "D": 0.5 },
  "B": { "C": 0.8 }
}
```

Both formats are equivalent. The edge `["A", "B", 1.0]` means material flows from A's **exit** ports to B's **entry** ports with weight 1.0.

### Zones Section

Spatial constraints that restrict where facilities can be placed.

#### Forbidden Areas

Permanently blocked regions where no facility body or clearance may overlap.

```json
"zones": {
  "forbidden": [
    { "shape_type": "rect", "rect": [0, 0, 150, 200] },
    { "shape_type": "irregular", "polygon": [[320, 320], [420, 320], [420, 420], [360, 470], [320, 420]] }
  ]
}
```

Area shape rules:

- `shape_type` is required.
- `shape_type: "rect"` uses `rect: [x0, y0, x1, y1]` (half-open: covers `[x0, x1) x [y0, y1)`).
- `shape_type: "irregular"` uses `polygon: [[x, y], ...]` in world/grid coordinates.

#### Constraint Zones

Attribute-based spatial zones. Each constraint has a grid-wide default value and optional area overrides.

```json
"zones": {
  "constraints": {
    "weight": {
      "dtype": "float",
      "op": "<=",
      "default": 10.0,
      "areas": [
        { "shape_type": "rect", "rect": [300, 0, 500, 500], "value": 25.0 }
      ]
    },
    "dry": {
      "dtype": "float",
      "op": ">=",
      "exclusive": "body",
      "default": { "value": 0.0, "id": "outside_dry_room" },
      "areas": [
        { "shape_type": "rect", "rect": [120, 80, 260, 240], "value": 1.0, "id": "dry_room_A" },
        { "shape_type": "irregular", "polygon": [[280, 90], [410, 90], [430, 220], [300, 250]], "value": 1.0, "id": "dry_room_B" }
      ]
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `dtype` | string | `"float"`, `"int"`, or `"bool"` |
| `op` | string | Comparison operator: `<`, `<=`, `>`, `>=`, `==`, `!=` |
| `exclusive` | bool or `"body"` | Optional. If enabled, facility body must not cross this constraint's zone-id boundary |
| `default` | value or object | Grid-wide default value. Exclusive mode also accepts `{ "value": ..., "id": ... }` |
| `areas[].shape_type` | string | Required. `"rect"` or `"irregular"` |
| `areas[].rect` | `[x0,y0,x1,y1]` | Required when `shape_type="rect"` |
| `areas[].polygon` | `[[x,y], ...]` | Required when `shape_type="irregular"` |
| `areas[].value` | value | Override zone value in this region |
| `areas[].id` | scalar | Required in exclusive mode. Physical partition id for boundary checks |

> See [Zone Constraints](#zone-constraints) for the validation logic and examples.

### Reset Section

Controls initial state when the environment resets.

| Field | Type | Description |
|---|---|---|
| `initial_placements` | dict | Pre-place facilities at fixed poses before the agent starts |
| `remaining_order` | list | Order in which the agent places remaining facilities |

```json
"reset": {
  "initial_placements": {
    "A": [130.0, 90.0, 0],
    "B": [260.0, 160.0, 1]
  },
  "remaining_order": ["C", "D", "E"]
}
```

Each value in `initial_placements` is `[x_c, y_c, orientation_index]`:
- `x_c, y_c`: center coordinates (float, same as `EnvAction`)
- `orientation_index`: index into `GroupSpec.orientations` list (0-based). If omitted (2-element list `[x_c, y_c]`), the engine tries all orientations and picks the best placeable one.

---

## Agents and Adapters

The system uses an **Agent + Adapter** architecture. Adapters translate raw engine state into model-specific observations and action spaces. Agents select actions from those action spaces.

### Agent Types

| Agent | Description | Compatible Adapters |
|---|---|---|
| **GreedyAgent** | Argmin of delta cost with softmax priors | All adapters |
| **MaskPlaceAgent** | CNN-based policy network (ResNet) | MaskPlaceAdapter |
| **AlphaChipAgent** | GNN-based policy (PyTorch Geometric) | AlphaChipAdapter |

### Adapter Types

| Adapter | Action Space | Observation | Best For |
|---|---|---|---|
| **GreedyAdapter** | Top-K sampled candidates | `{"action_costs": [K]}` | Fast heuristic search |
| **GreedyV2Adapter** | Top-K with different sampling | Same as above | Alternative sampling |
| **GreedyV3Adapter** | Edge-based boundary sampling | Same as above | Large grids, MCTS |
| **MaskPlaceAdapter** | Dense GxG grid (default 224x224) | CNN state maps | RL training |
| **AlphaChipAdapter** | Coarse GxG grid (default 128x128) | GNN graph tensors | RL training |

### Registry

Create valid (agent, adapter) pairs via the registry:

```python
from agents.registry import create

agent, adapter = create("greedyv3")                    # GreedyAgent + GreedyV3Adapter
agent, adapter = create("greedyv3", agent="greedy")    # Same, explicit
agent, adapter = create("maskplace")                   # MaskPlaceAgent + MaskPlaceAdapter
agent, adapter = create("maskplace", agent="greedy")   # GreedyAgent + MaskPlaceAdapter (hybrid)
```

**Compatibility matrix:**

| Method (Adapter) | `agent="greedy"` | `agent="maskplace"` | `agent="alphachip"` |
|---|---|---|---|
| `greedy` | Yes | - | - |
| `greedyv2` | Yes | - | - |
| `greedyv3` | Yes | - | - |
| `maskplace` | Yes | Yes (default) | - |
| `alphachip` | Yes | - | Yes (default) |

### Ordering Agent

**DifficultyOrderingAgent** reorders the remaining facility queue by placement difficulty (hardest first). Difficulty = `facility_area / free_space_after_zone_filtering`.

```python
# In inference.py:
ORDERING_MODE = "difficulty"  # or "none"
```

---

## Search Algorithms

Search operates at the adapter level, exploring multiple placement decisions via tree/beam expansion.

### MCTS (Monte Carlo Tree Search)

| Parameter | Default | Description |
|---|---|---|
| `num_simulations` | 50 | Number of tree simulations |
| `c_puct` | 2.0 | PUCT exploration constant |
| `rollout_enabled` | true | Use greedy rollout at leaf nodes |
| `rollout_depth` | 5 | Maximum rollout length |
| `dirichlet_epsilon` | 0.2 | Root noise mixing factor |
| `dirichlet_concentration` | 0.5 | Dirichlet alpha |
| `temperature` | 0.0 | Action selection temperature (0 = deterministic) |
| `pw_enabled` | false | Progressive widening for large action spaces |
| `pw_c` | 1.5 | Widening coefficient: `k(s) = ceil(pw_c * N(s)^pw_alpha)` |
| `pw_alpha` | 0.5 | Widening exponent |
| `track_top_k` | 0 | Track best K complete episodes (0 = disabled) |

### Beam Search

| Parameter | Default | Description |
|---|---|---|
| `beam_width` | 8 | Number of parallel beams |
| `depth` | 5 | Search depth |
| `expansion_topk` | 16 | Top-K children expanded per beam |
| `track_top_k` | 0 | Track best K complete episodes |

---

## Training

### Tianshou PPO (`train.py`)

```bash
python train.py --mode maskplace --env-json envs/env_configs/basic_01.json --device cuda
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | required | `maskplace` or `alphachip` |
| `--env-json` | `basic_01.json` | Path to env config |
| `--maskplace-grid` | 224 | MaskPlace grid resolution |
| `--coarse-grid` | 128 | AlphaChip coarse grid size |
| `--epoch` | 100 | Number of training epochs |
| `--step-per-epoch` | 2000 | Steps per epoch |
| `--step-per-collect` | 1000 | Steps per data collection |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-ratio` | 0.2 | PPO clipping ratio |
| `--vf-coef` | 0.5 | Value function loss weight |
| `--ent-coef` | 0.0 | Entropy bonus coefficient |
| `--device` | auto | `cuda` or `cpu` |

### TorchRL PPO (`train_torchrl.py`)

```bash
python train_torchrl.py --env-json envs/env_configs/basic_01.json --device cuda
```

| Argument | Default | Description |
|---|---|---|
| `--env-json` | `basic_01.json` | Path to env config |
| `--grid` | 224 | MaskPlace grid resolution |
| `--total-frames` | 200000 | Total training frames |
| `--frames-per-batch` | 1024 | Frames per rollout batch |
| `--mini-batch-size` | 32 | PPO mini-batch size |
| `--ppo-epochs` | 8 | PPO update epochs per batch |
| `--lr` | 3e-4 | Learning rate |
| `--ent-coef` | 0.01 | Entropy bonus coefficient |
| `--load-ckpt` | None | Resume from checkpoint |

---

## Inference

Configure `inference.py` by editing module-level constants (no CLI args):

```python
ENV_JSON       = "envs/env_configs/basic_01.json"
WRAPPER_MODE   = "greedyv3"    # adapter type
AGENT_MODE     = "greedy"      # agent type
SEARCH_MODE    = "mcts"        # "none", "mcts", "beam"
ORDERING_MODE  = "none"        # "none", "difficulty"

# MCTS tuning
MCTS_SIMS           = 1000
ROLLOUT_DEPTH        = 10
MCTS_TEMPERATURE     = 0.0
MCTS_PW_ENABLED      = False

# Adapter tuning
TOPK_K          = 50
TOPK_SCAN_STEP  = 5.0
TOPK_QUANT_STEP = 10.0

# Visualization
SHOW_FLOW   = True
SHOW_SCORE  = True
SHOW_MASKS  = True
```

Run:
```bash
python inference.py
```

Output is saved to `results/` with placement JSON, layout images, and top-K variant summaries.

---

## Web UI

Interactive facility placement interface with undo/redo and live search progress.

```bash
python run_webui.py
# Open http://localhost:8000
```

**Features:**
- Create sessions with configurable adapter, agent, and search settings
- Step-by-step manual or automatic placement
- Undo/redo with full state restoration
- Real-time search progress via WebSocket
- Visualization of candidates, zones, flows, and constraints

**REST API endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/session` | Create session with config |
| `GET` | `/api/session/{sid}` | Get current state |
| `POST` | `/api/session/{sid}/step` | Place one facility |
| `POST` | `/api/session/{sid}/undo` | Undo last placement |
| `POST` | `/api/session/{sid}/redo` | Redo placement |
| `POST` | `/api/session/{sid}/search` | Run MCTS/beam search |
| `DELETE` | `/api/session/{sid}` | Delete session |
| `WebSocket` | `/ws/{sid}` | Streaming search progress |

---

## Postprocessing

### Route Planning (`postprocess/pathfinder.py`)

Computes material flow paths between placed facilities using grid-based pathfinding.

```python
from postprocess.pathfinder import RoutePlanner

planner = RoutePlanner(env, algorithm="astar", allow_diagonal=False)
routes = planner.plan_all()  # routes for all flow edges
summary = planner.get_summary()
```

### Dynamic Group Generation (`postprocess/dynamic_group.py`)

Generates variable-footprint storage racks via BFS frontier expansion.

```python
from postprocess.dynamic_group import DynamicGroupGenerator

gen = DynamicGroupGenerator(env)
group = gen.generate(unit_w=10, unit_h=5, clearance_w=2, clearance_h=2, target_area=500)
```

### Dynamic Planner (`envs/placement/dynamic.py`)

Conv2D-based validity expansion for dynamic facilities with capacity optimization. Selects unit placements by minimizing a priority function of (height capacity, bbox expansion, flow penalty).

---

## Key Concepts Explained

### Coordinate System

- **Origin**: bottom-left `(0, 0)`
- **Facility position**: `(x_bl, y_bl)` = bottom-left corner
- **Rotation**: counter-clockwise in degrees: `{0, 90, 180, 270}`
- **Tensor indexing**: `tensor[y, x]` (row = y, column = x)
- **Port coordinates**: always relative to BL corner at rotation 0

### Port Span Selection (`entry_port_span` / `exit_port_span`)

When a facility has multiple entry/exit ports, `*_port_span` controls how many closest ports are averaged.

- `1`: use the single closest port (same behavior as old `min`)
- `k` (`k>=2`): average over the `k` closest valid ports
- `"all"`: average over all valid ports (same behavior as old `mean`)

Exit and entry spans are independent. For flow `A -> B`:

1. `A.exit_port_span` reduces A's exit ports
2. `B.entry_port_span` reduces B's entry ports

If requested span is larger than available ports, the engine clamps to available count and prints a warning.

#### Compact Summary Table

| Facility Ports | `span=1` | `span=2` | `span="all"` |
|---|---|---|---|
| 1 port | closest(only) | average of 1 | average of 1 |
| 2 ports | closest 1 | average of closest 2 | average of 2 |
| 3 ports | closest 1 | average of closest 2 | average of 3 |
| N ports | `min(d_1, ..., d_N)` | average of top-2 smallest | `(d_1 + ... + d_N) / N` |

### Zone Constraints

Zone constraints validate whether a facility's attribute satisfies a spatial zone requirement.

**Validation rule:** `facility_value op zone_value`

The `op` is read as "the facility's value must be `op` the zone's value".

**Example with weight constraint:**

```json
"zones": {
  "constraints": {
    "weight": {
      "dtype": "float",
      "op": "<=",
      "default": 10.0,
      "areas": [
        { "shape_type": "rect", "rect": [300, 0, 500, 500], "value": 25.0 }
      ]
    }
  }
}
```

Grid layout:

```
+-------------------+------------------+
|                   |                  |
|  weight limit     |  weight limit    |
|  = 10.0 (default) |  = 25.0          |
|                   |  (reinforced)    |
|  x=0..299         |  x=300..499      |
+-------------------+------------------+
```

- **Facility A** (`weight: 3.0`, op `<=`): 3.0 <= 10.0 = valid everywhere
- **Facility E** (`weight: 15.0`, op `<=`): 15.0 <= 10.0 = **invalid** in left zone; 15.0 <= 25.0 = valid in right zone

**Example with height constraint (non-exclusive):**

```json
"height": {
  "dtype": "float",
  "op": "<=",
  "default": 20.0,
  "areas": [
    { "shape_type": "rect", "rect": [0, 350, 500, 500], "value": 30.0 }
  ]
}
```

- **Facility with `height: 18.0`** (op `<=`): 18.0 <= 20.0 and 18.0 <= 30.0 -> valid everywhere
- **Facility with `height: 24.0`** (op `<=`): 24.0 <= 20.0 is **invalid** in default area; 24.0 <= 30.0 is valid in override area

#### Exclusive Zone Boundaries

`exclusive` controls whether crossing this constraint's partition boundary is allowed:

- `exclusive=false` (default): only value rule is checked (`facility_value op zone_value`).
- `exclusive=true` / `"body"`: value rule is still checked, and the facility **body** must stay inside one partition id.

Use `exclusive=false` for scalar constraints without physical walls (for example, ceiling height or floor load capacity).
Use `exclusive=true` for physically partitioned spaces (for example, industrial **dry rooms** / clean rooms with walls).

Exclusive example (`dry`):

```json
"dry": {
  "dtype": "float",
  "op": ">=",
  "exclusive": "body",
  "default": { "value": 0.0, "id": "outside" },
  "areas": [
    { "shape_type": "rect", "rect": [120, 80, 260, 240], "value": 1.0, "id": "dry_room_A" },
    { "shape_type": "rect", "rect": [280, 80, 420, 240], "value": 1.0, "id": "dry_room_B" }
  ]
}
```

With this config, a facility can satisfy dry values but still be invalid if its body crosses from `outside` to `dry_room_A`, or from `dry_room_A` to `dry_room_B`.

### Clearance Rotation

Directional clearance `[L, R, B, T]` rotates with the facility:

```
Rotation    Clearance mapping
--------    -----------------
  0 deg     [L, R, B, T]
 90 deg     [B, T, R, L]     (90 CCW: left becomes bottom, bottom becomes right, ...)
180 deg     [R, L, T, B]
270 deg     [T, B, L, R]

Example: clearance_lrtb = [2, 3, 4, 5]

  0 deg:            90 deg:
  +---5---+         +---3---+
  |       |         |       |
  2       3         4       5
  |       |         |       |
  +---4---+         +---2---+
```

### Delta Cost Pattern

All candidate evaluation uses **incremental** cost computation via `engine.delta_cost()`, not full cost recalculation. For each candidate position, only the cost change from the new facility's flow connections is computed. This is vectorized over all M candidates simultaneously:

```python
# Returns [M] tensor of cost deltas for M candidate positions
delta = engine.delta_cost(gid, x_batch, y_batch, rot_batch)
```

### State Copy and Static Sharing

When copying environment state (e.g., for MCTS tree nodes), static tensors (forbidden areas, constraint maps) are **shared by reference** while runtime tensors (occupancy, clearance) are cloned. This makes MCTS snapshots memory-efficient:

```python
state_copy = engine.get_state().copy()   # cheap: shares static maps
engine.set_state(state_copy)             # restore snapshot
```

---

## Example Configs

| Config | Description | Facilities |
|---|---|---|
| `basic_01.json` | Minimal rectangles, no constraints | 9 |
| `clearance_01.json` | Uniform 10-cell clearance halos | 9 |
| `zones_01.json` | Weight/height/dry zone constraints | 10 |
| `multiport_01.json` | Multiple ports with span selection (`1`, `k`, `"all"`) | 5 |
| `mixed_01.json` | Irregular polygons + rectangles | 8 |
| `placed_01.json` | 2 pre-placed, rest by agent | 9 |
