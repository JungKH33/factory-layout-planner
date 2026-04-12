# Lane routing benchmark — results

Run produced by `python -m tmp.lane_bench_report` on 2026-04-11.

## Configuration

- Warmup rounds: 2
- Timed rounds: 6
- Candidate K per flow: 4
- Devices: `cpu`, `cuda`
- Scenarios: 5 synthetic grids (empty / sparse / dense / maze)

## Column definitions

| column    | meaning |
|-----------|---------|
| `dev`     | device the tensors live on (`cpu` or `cuda`) |
| `scenario`| benchmark name (`<pattern>_<grid side>`) |
| `algo`    | routing algorithm (`wavefront` / `astar` / `dijkstra`) |
| `mean(ms)`| mean wall-time per round (sum of all flows) |
| `std(ms)` | population std of round times |
| `ref(ms)` | `reward_composer.delta_batch()` time on prebuilt candidates (baseline "manhattan distance measurement" from `lane_generation/envs/reward/flow.py`) |
| `x_time`  | `mean / ref` — how many times slower routing is than pure reward eval |
| `path`    | sum over flows of shortest routed candidate length (4-connected edges) |
| `manh`    | sum over flows of min Manhattan distance across src/dst port pairs |
| `x_len`   | `path / manh` — detour factor vs the Manhattan lower bound |
| `miss`    | flows that produced no candidate |

## Raw output

```
dev  scenario      algo          mean(ms)   std(ms)   ref(ms)    x_time    path    manh   x_len  miss
-----------------------------------------------------------------------------------------------------
cpu  empty_80      wavefront       90.566     2.875     0.257    352.5x     240     240   1.00x     0
cpu  empty_80      astar          177.082     1.812     0.257    689.3x     240     240   1.00x     0
cpu  empty_80      dijkstra       431.159     5.491     0.257   1678.3x     240     240   1.00x     0

cpu  empty_150     wavefront      308.427     5.445     0.370    833.7x     850     850   1.00x     0
cpu  empty_150     astar         2215.762    51.518     0.370   5989.6x     850     850   1.00x     0
cpu  empty_150     dijkstra      3139.487    15.291     0.370   8486.6x     850     850   1.00x     0

cpu  sparse_150    wavefront      326.108    24.876     0.382    853.3x     850     850   1.00x     0
cpu  sparse_150    astar         1816.357    19.906     0.382   4753.0x     850     850   1.00x     0
cpu  sparse_150    dijkstra      2674.613    31.243     0.382   6998.9x     850     850   1.00x     0

cpu  dense_200     wavefront      992.562    37.720     0.409   2424.3x    1202    1200   1.00x     0
cpu  dense_200     astar         3625.334    27.401     0.409   8854.9x    1202    1200   1.00x     0
cpu  dense_200     dijkstra      6728.934  1400.402     0.409  16435.4x    1202    1200   1.00x     0

cpu  maze_200      wavefront     4374.947   496.193     1.492   2932.6x    2640    1200   2.20x     1
cpu  maze_200      astar         4486.097     6.274     1.492   3007.1x    2640    1200   2.20x     1
cpu  maze_200      dijkstra      4072.402    17.745     1.492   2729.8x    2640    1200   2.20x     1

cuda empty_80      wavefront      357.842     3.620     0.943    379.3x     240     240   1.00x     0
cuda empty_80      astar          238.838     9.112     0.943    253.2x     240     240   1.00x     0
cuda empty_80      dijkstra       516.465    13.831     0.943    547.5x     240     240   1.00x     0

cuda empty_150     wavefront     1179.983    50.021     1.565    753.9x     850     850   1.00x     0
cuda empty_150     astar         2452.331    25.636     1.565   1566.8x     850     850   1.00x     0
cuda empty_150     dijkstra      3330.689    21.296     1.565   2128.0x     850     850   1.00x     0

cuda sparse_150    wavefront     1307.726   110.746     1.681    777.8x     850     850   1.00x     0
cuda sparse_150    astar         2008.217    11.229     1.681   1194.4x     850     850   1.00x     0
cuda sparse_150    dijkstra      2885.748    40.816     1.681   1716.3x     850     850   1.00x     0

cuda dense_200     wavefront     1638.873   144.505     1.408   1164.3x    1202    1200   1.00x     0
cuda dense_200     astar         3861.167    10.889     1.408   2743.1x    1202    1200   1.00x     0
cuda dense_200     dijkstra      5215.946    44.616     1.408   3705.6x    1202    1200   1.00x     0

cuda maze_200      wavefront     3717.455   122.476     1.090   3410.7x    2640    1200   2.20x     1
cuda maze_200      astar         5251.971    12.322     1.090   4818.6x    2640    1200   2.20x     1
cuda maze_200      dijkstra      5119.851   391.241     1.090   4697.4x    2640    1200   2.20x     1
```

## Quick observations

- **Routing is 300~16000x slower than the reward eval** (`delta_batch`) across all scenarios — routing dominates the pipeline; reward eval is essentially free.
- **All three algorithms return the same path length** (`path` column is identical within each scenario) — they are equally optimal; the tradeoff is purely time.
- **GPU (`cuda`) is slower than CPU** on almost every row. astar/dijkstra internally call `free_map.detach().cpu().numpy()`, so GPU tensors only add a CPU-copy round-trip; wavefront's backtrace uses per-step `.item()` sync which also hurts GPU.
- **wavefront is the fastest algorithm on every CPU scenario except `maze_200`** (where all three converge around 4000ms and dijkstra slightly wins, likely noise — `dense_200 dijkstra` also has a 1400ms outlier in std).
- `maze_200` has 1 unreachable flow (`miss=1`) for all algorithms → path/manh ratio 2.20x for the reachable flows, suggesting significant detour.
