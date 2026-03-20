from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv, GroupId

from ...base import BaseAdapter


class AlphaChipAdapter(BaseAdapter):
    """AlphaChip-style coarse action wrapper: Discrete(G*G) actions.

    Provides:
    - action_mask: bool [G*G]  (valid-action mask for Discrete)
    - graph tensors: x, edge_index, edge_attr, current_node, netlist_metadata
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        coarse_grid: int,
    ):
        super().__init__()
        self.grid_width = 1
        self.grid_height = 1
        self.coarse_grid = int(coarse_grid)

        # AlphaChip graph tensors are adapter-specific caches, rebuilt when bound engine changes.
        self._graph_engine_key: Optional[Tuple[int, int, int]] = None
        self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        self.edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        self.action_space = gym.spaces.Discrete(self.coarse_grid * self.coarse_grid)
        self.observation_space = gym.spaces.Dict({})
        self.action_poses: Optional[torch.Tensor] = None  # float32 [G*G,2]

    def bind(self, engine: FactoryLayoutEnv) -> None:
        super().bind(engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)
        self._ensure_graph_static()

    def _ensure_graph_static(self) -> None:
        eng = self.engine
        key = (id(eng), int(len(eng.node_ids)), int(sum(len(v) for v in eng.group_flow.values())))
        if self._graph_engine_key == key:
            return
        self.edge_index, self.edge_attr = self._build_graph_static()
        self._graph_engine_key = key

    def _build_graph_static(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sparse graph tensors from engine.group_flow.

        Returns:
          edge_index: int64 [2,E]
          edge_attr: float32 [E,1]  (#nets/weight)
        """
        edges: List[List[int]] = []
        attrs: List[List[float]] = []
        for src, dsts in self.engine.group_flow.items():
            if src not in self.engine.gid_to_idx:
                continue
            for dst, w in dsts.items():
                if dst not in self.engine.gid_to_idx:
                    continue
                edges.append([self.engine.gid_to_idx[src], self.engine.gid_to_idx[dst]])
                attrs.append([float(w)])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
            edge_attr = torch.tensor(attrs, dtype=torch.float32, device=self.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32, device=self.device)
        return edge_index, edge_attr

    def cell_wh(self) -> Tuple[int, int]:
        g = int(self.coarse_grid)
        cell_w = int(math.ceil(self.grid_width / float(g)))
        cell_h = int(math.ceil(self.grid_height / float(g)))
        return cell_w, cell_h

    def _next_gid(self) -> Optional[GroupId]:
        return self.current_gid()

    def create_mask(self) -> torch.Tensor:
        gid = self._next_gid()
        if gid is None:
            self.action_poses = torch.zeros(
                (self.coarse_grid * self.coarse_grid, 2), dtype=torch.float32, device=self.device
            )
            return torch.zeros((self.coarse_grid * self.coarse_grid,), dtype=torch.bool, device=self.device)

        spec = self.engine.group_specs[gid]
        state = self.engine.get_state()

        g = int(self.coarse_grid)
        cell_w, cell_h = self.cell_wh()

        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        self.action_poses = torch.stack([cx, cy], dim=-1).view(g * g, 2).to(dtype=torch.float32)

        ok = spec.placeable_batch(
            state=state, gid=gid,
            x_c=cx.reshape(-1), y_c=cy.reshape(-1),
        )
        return ok

    def build_observation(self) -> Dict[str, Any]:
        self.mask = None
        obs: Dict[str, Any] = {}
        gid = self._next_gid()
        cur_idx = int(self.engine.gid_to_idx.get(gid, 0)) if gid is not None else 0
        obs["x"] = self._build_x()
        obs["edge_index"] = self.edge_index
        obs["edge_attr"] = self.edge_attr
        obs["current_node"] = torch.tensor([cur_idx], dtype=torch.long, device=self.device)
        obs["netlist_metadata"] = self._build_netlist_metadata()
        return obs

    def get_state_copy(self) -> Dict[str, object]:
        snap = dict(super().get_state_copy())
        if isinstance(self.action_poses, torch.Tensor):
            snap["action_poses"] = self.action_poses.clone()
        else:
            snap["action_poses"] = None
        return snap

    def set_state(self, state: Dict[str, object]) -> None:
        super().set_state(state)
        ax = state.get("action_poses", None)
        if isinstance(ax, torch.Tensor):
            self.action_poses = ax.to(device=self.device, dtype=torch.float32).clone()
        else:
            self.action_poses = None

    def _build_x(self) -> torch.Tensor:
        """Build node features x: float32 [N,8] from engine state."""
        N = int(len(self.engine.node_ids))
        x = torch.zeros((N, 8), dtype=torch.float32, device=self.device)
        # static features (w/h)
        for gid, i in self.engine.gid_to_idx.items():
            gg = self.engine.group_specs[gid]
            x[int(i), 0] = float(gg.width) / float(self.engine.grid_width)
            x[int(i), 1] = float(gg.height) / float(self.engine.grid_height)
        # dynamic features (placed + center)
        for gid in self.engine.get_state().placed:
            idx = self.engine.gid_to_idx.get(gid, None)
            if idx is None:
                continue
            p = self.engine.get_state().placements[gid]
            x_c = float(getattr(p, "x_c", (float(getattr(p, "min_x", 0.0)) + float(getattr(p, "max_x", 0.0))) / 2.0))
            y_c = float(getattr(p, "y_c", (float(getattr(p, "min_y", 0.0)) + float(getattr(p, "max_y", 0.0))) / 2.0))
            x[int(idx), 2] = 1.0
            x[int(idx), 3] = x_c / float(self.engine.grid_width)
            x[int(idx), 4] = y_c / float(self.engine.grid_height)
        return x

    def _build_netlist_metadata(self) -> torch.Tensor:
        """Build AlphaChip netlist_metadata: float32 [12]. Includes cost via env.cost()."""
        meta = torch.zeros((12,), dtype=torch.float32, device=self.device)
        placed_ratio = float(len(self.engine.get_state().placed)) / float(max(1, len(self.engine.node_ids)))
        remaining_ratio = float(len(self.engine.get_state().remaining)) / float(max(1, len(self.engine.node_ids)))
        cost = float(self.engine.cost())
        meta[0] = float(placed_ratio)
        meta[1] = float(remaining_ratio)
        # meta[2] = float(cost)
        meta[2] = 0.0
        #meta[3] = float(self.engine.grid_width)
        meta[3] = 0.0
        #meta[4] = float(self.engine.grid_height)
        meta[4] = 0.0
        return meta

if __name__ == "__main__":
    import time

    import torch
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    from envs.action_space import ActionSpace as CandidateSet
    from envs.action import EnvAction
    from envs.env_loader import load_env
    from envs.env_visualizer import plot_layout

    def _as_numpy(x: object) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _print_arr(name: str, v: object, *, max_rows: int = 3, max_cols: int = 8) -> None:
        arr = _as_numpy(v)
        if arr.size == 0:
            print(f"{name}: shape={arr.shape} empty")
            return
        with np.printoptions(precision=4, suppress=True):
            if arr.ndim == 0:
                print(f"{name}: shape={arr.shape} value={arr.item()}")
            elif arr.ndim == 1:
                head = arr[: min(arr.shape[0], max_cols)]
                print(
                    f"{name}: shape={arr.shape} dtype={arr.dtype} "
                    f"min={float(arr.min()):.4g} max={float(arr.max()):.4g} head={head}"
                )
            else:
                r = min(arr.shape[0], max_rows)
                c = min(arr.shape[1], max_cols)
                head = arr[:r, :c]
                print(
                    f"{name}: shape={arr.shape} dtype={arr.dtype} "
                    f"min={float(arr.min()):.4g} max={float(arr.max()):.4g} head=\n{head}"
                )

    def _print_alphachip_obs(obs: dict[str, object]) -> None:
        print("\n--- AlphaChip obs (key stats) ---")
        for k in ["x", "edge_index", "edge_attr", "netlist_metadata", "current_node", "action_mask"]:
            if k not in obs:
                print(f"{k}: <missing>")
                continue
            _print_arr(k, obs[k])

        if "action_mask" in obs:
            am = _as_numpy(obs["action_mask"]).astype(bool).reshape(-1)
            idxs = np.where(am)[0]
            print(f"valid_actions={int(am.sum())} first_valid_idxs={idxs[:10].tolist()}")
        if "current_node" in obs:
            cur = int(_as_numpy(obs["current_node"]).reshape(-1)[0].item())
            print(f"current_node={cur}")

    def _plot_alphachip_graph(obs: dict[str, object], *, title: str, max_edges: int = 2000) -> None:
        if "x" not in obs or "edge_index" not in obs or "current_node" not in obs:
            print("plot_graph: missing one of: x, edge_index, current_node")
            return

        x = _as_numpy(obs["x"])
        ei = _as_numpy(obs["edge_index"]).astype(np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            print(f"plot_graph: edge_index must be [2,E], got {ei.shape}")
            return
        n = int(x.shape[0]) if x.ndim >= 1 else 0
        e = int(ei.shape[1])
        cur = int(_as_numpy(obs["current_node"]).reshape(-1)[0].item())

        if max_edges > 0 and e > int(max_edges):
            ei = ei[:, : int(max_edges)]
            e = int(max_edges)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            G.add_edge(int(u), int(v))

        print(f"graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()} (cap={max_edges})")
        if 0 <= cur < n:
            print(f"graph: cur out_deg={G.out_degree(cur)} in_deg={G.in_degree(cur)}")
        else:
            print(f"graph: WARNING: current_node={cur} out of range [0,{n})")

        pos = nx.spring_layout(G, seed=42)
        node_colors = ["#d62728" if int(i) == int(cur) else "#1f77b4" for i in G.nodes()]
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_size=110, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=0.8, alpha=0.25)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    ENV_JSON = "envs/env_configs/constraints_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    adapter = AlphaChipAdapter(coarse_grid=32)

    t0 = time.perf_counter()
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)
    adapter.bind(engine)
    obs = adapter.build_observation()
    candidates = adapter.build_action_space()
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(candidates.mask.sum().item())
    a = int(torch.where(candidates.mask)[0][0].item()) if valid > 0 else 0

    # Print obs tensors + visualize graph structure.
    _print_alphachip_obs({k: obs[k] for k in obs.keys()})
    _plot_alphachip_graph({k: obs[k] for k in obs.keys()}, title="AlphaChip obs graph (reset)")

    # Build BL candidates for visualization (avoid plotting all invalid points).
    g = int(adapter.coarse_grid)
    idxs = torch.where(candidates.mask)[0]
    xy = candidates.poses[idxs]
    cand0 = CandidateSet(
        poses=xy,
        mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device),
        gid=candidates.gid,
        meta={"g": g},
    )
    plot_layout(engine, action_space=cand0)

    t1 = time.perf_counter()
    placement = adapter.decode_action(a, candidates)
    _obs_env2, _r, _term, _trunc, _info2 = engine.step_action(placement)
    obs2 = adapter.build_observation()
    candidates2 = adapter.build_action_space()
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot after one placement (new candidates)
    if int(candidates2.mask.shape[0]) > 0:
        _print_alphachip_obs({k: obs2[k] for k in obs2.keys()})
        _plot_alphachip_graph({k: obs2[k] for k in obs2.keys()}, title="AlphaChip obs graph (after 1 step)")
        idxs2 = torch.where(candidates2.mask)[0]
        xy2 = candidates2.poses[idxs2]
        cand1 = CandidateSet(
            poses=xy2,
            mask=torch.ones((xy2.shape[0],), dtype=torch.bool, device=device),
            gid=candidates2.gid,
            meta={"g": g},
        )
        plot_layout(engine, action_space=cand1)
    else:
        plot_layout(engine, action_space=None)

    print("AlphaChipAdapter demo")
    print(" env=", ENV_JSON, "device=", device, "G=", adapter.coarse_grid)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")
