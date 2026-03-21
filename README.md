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
| `entry_port_mode` | string | `"min"` | Port aggregation: `"min"` or `"mean"` |
| `exit_port_mode` | string | `"min"` | Port aggregation: `"min"` or `"mean"` |

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
  "entry_port_mode": "mean",
  "exit_port_mode": "min"
}
```

> See [Port Aggregation Modes](#port-aggregation-modes-min-vs-mean) for a detailed explanation with examples.

#### Clearance

Clearance defines the required buffer zone around a facility that other facilities cannot overlap.

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
  "forbidden_areas": [
    { "rect": [0, 0, 150, 200] },
    { "rect": [400, 300, 500, 500] }
  ]
}
```

Each `rect` is `[x0, y0, x1, y1]` (half-open: covers `[x0, x1) x [y0, y1)`).

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
        { "rect": [300, 0, 500, 500], "value": 25.0 }
      ]
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `dtype` | string | `"float"`, `"int"`, or `"bool"` |
| `op` | string | Comparison operator: `<`, `<=`, `>`, `>=`, `==`, `!=` |
| `default` | value | Grid-wide default zone value |
| `areas[].rect` | `[x0,y0,x1,y1]` | Region with override value |
| `areas[].value` | value | Override zone value in this region |

> See [Zone Constraints](#zone-constraints) for the validation logic and examples.

### Reset Section

Controls initial state when the environment resets.

| Field | Type | Description |
|---|---|---|
| `initial_positions` | dict | Pre-place facilities at fixed poses before the agent starts |
| `remaining_order` | list | Order in which the agent places remaining facilities |

```json
"reset": {
  "initial_positions": {
    "A": [100, 50, 0],
    "B": [200, 100, 90]
  },
  "remaining_order": ["C", "D", "E"]
}
```

Each value in `initial_positions` is `[x_bl, y_bl, rotation]`:
- `x_bl, y_bl`: bottom-left corner coordinates (integer cells)
- `rotation`: degrees counter-clockwise, one of `{0, 90, 180, 270}`

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
| **GreedyAdapter** | Top-K sampled candidates | `{"action_delta": [K]}` | Fast heuristic search |
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

### Port Aggregation Modes (`min` vs `mean`)

When a facility has **multiple** entry or exit ports, the port aggregation mode controls how distances from those ports are combined into a single cost value.

**Setup:** Consider facility A (source) with 2 exit ports, and facility B (target) with 3 entry ports. The flow cost from A to B requires reducing the 2x3 port-pair distance matrix to a single number.

#### `"min"` mode (default)

Use the **closest** port pair. Only the single best-matching port contributes to cost.

**Example:** A 3x3 facility with 3 entry ports at `[[0,0], [0,1], [0,2]]` and port mode `"min"`:

```
Incoming material from facility X's exit at (5, 1):

  Port (0,0): distance = |5-0| + |1-0| = 6
  Port (0,1): distance = |5-0| + |1-1| = 5  <-- minimum
  Port (0,2): distance = |5-0| + |1-2| = 6

  min mode result = 5
```

Best interpretation: material is routed to whichever port is closest. Use this when ports are alternatives (e.g., multiple loading docks -- material goes to the nearest one).

#### `"mean"` mode

Use the **average** distance across all valid ports.

**Example:** Same setup as above:

```
  Port (0,0): distance = 6
  Port (0,1): distance = 5
  Port (0,2): distance = 6

  mean mode result = (6 + 5 + 6) / 3 = 5.67
```

Best interpretation: material must reach all ports. Use this when ports represent distributed access points that all need to be served (e.g., multiple workstations within a facility that all receive input).

#### Combining Modes

Exit and entry ports have **independent** modes. With flow A -> B:

1. A's `exit_port_mode` controls how A's exit ports are aggregated
2. B's `entry_port_mode` controls how B's entry ports are aggregated

**Full example with a 6x4 facility:**

```json
"A": {
  "width": 6, "height": 4,
  "exits_rel": [[6, 1], [6, 3]],
  "exit_port_mode": "min"
}
```

```
+------+
|      * (6,3) exit port 1
|  A   |
|      * (6,1) exit port 0
+------+
```

If B has `entry_port_mode: "mean"` with entries at `[[0, 2], [0, 6], [0, 10]]`:

```
Step 1: For each of A's exits, compute distance to EACH of B's entries (all pairs)
Step 2: Reduce B's entries using "mean" -> average across B's 3 entry ports
Step 3: Reduce A's exits using "min" -> pick the closer of A's 2 exit ports
```

#### Compact Summary Table

| Facility Ports | `"min"` | `"mean"` |
|---|---|---|
| 1 port | no effect | no effect |
| 2 ports | closest port wins | average of 2 |
| 3 ports | closest port wins | average of 3 |
| N ports | `min(d_1, ..., d_N)` | `(d_1 + ... + d_N) / N` |

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
        { "rect": [300, 0, 500, 500], "value": 25.0 }
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

**Example with dry constraint:**

```json
"dry": {
  "dtype": "float",
  "op": ">=",
  "default": 1.0,
  "areas": [
    { "rect": [0, 250, 250, 500], "value": 0.0 }
  ]
}
```

- **Facility with `dry: 2.0`** (op `>=`): 2.0 >= 1.0 = valid in default area; 2.0 >= 0.0 = valid in override area --> valid everywhere
- **Facility with `dry: 0.5`** (op `>=`): 0.5 >= 1.0 = **invalid** in default area; 0.5 >= 0.0 = valid in override area

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
| `multiport_01.json` | Multiple ports with min/mean modes | 5 |
| `mixed_01.json` | Irregular polygons + rectangles | 8 |
| `placed_01.json` | 2 pre-placed, rest by agent | 9 |
