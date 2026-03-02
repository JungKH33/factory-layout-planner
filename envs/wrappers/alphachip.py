from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId

from .base import BaseWrapper


class AlphaChipWrapperEnv(BaseWrapper):
    """AlphaChip-style coarse action wrapper: Discrete(G*G) actions.

    Provides:
    - action_mask: bool [G*G]  (valid-action mask for Discrete)
    - graph tensors: x, edge_index, edge_attr, current_node, netlist_metadata
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        coarse_grid: int,
        rot: int = 0,
    ):
        super().__init__(engine=engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)
        self.coarse_grid = int(coarse_grid)
        self.rot = int(rot)

        # AlphaChip graph tensors are AlphaChip-specific; build them here (not in engine).
        self.edge_index, self.edge_attr = self._build_graph_static()

        self.action_space = gym.spaces.Discrete(self.coarse_grid * self.coarse_grid)
        self.observation_space = gym.spaces.Dict({})

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
        return self.engine.remaining[0] if self.engine.remaining else None

    def decode_action(self, mask_index: int) -> Tuple[float, float, int, int, int]:
        """Decode coarse cell index to bottom-left integer coordinates (x_bl,y_bl,rot)."""
        gid = self._next_gid()
        if gid is None:
            return 0.0, 0.0, 0, 0, 0

        g = int(self.coarse_grid)
        a = int(mask_index)
        i = a // g
        j = a % g

        group = self.engine.groups[gid]
        rot = int(self.rot if group.rotatable else 0)
        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))

        cell_w, cell_h = self.cell_wh()
        cx = float(j * cell_w) + (cell_w / 2.0)
        cy = float(i * cell_h) + (cell_h / 2.0)
        x_bl = int(round(cx - (w_i / 2.0)))
        y_bl = int(round(cy - (h_i / 2.0)))
        return float(x_bl), float(y_bl), int(rot), int(i), int(j)

    def _valid_top_left_body(self, *, gid: GroupId, rot: int) -> torch.Tensor:
        """Return bool[H2,W2] where True means body window is clear of invalid and clear_invalid."""
        group = self.engine.groups[gid]
        w, h = self.engine.rotated_size(group, rot)
        kw = max(1, int(round(float(w))))
        kh = max(1, int(round(float(h))))

        inv = self.engine._invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        clr = self.engine._clear_invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv.dtype)

        ov_inv = F.conv2d(inv, kernel, padding=0).squeeze(0).squeeze(0)
        ov_clr = F.conv2d(clr, kernel, padding=0).squeeze(0).squeeze(0)
        return (ov_inv == 0) & (ov_clr == 0)

    def _valid_top_left_pad(self, *, gid: GroupId, rot: int) -> Tuple[torch.Tensor, int, int]:
        """Return (bool[H3,W3], cL, cB) for pad window invalid check.

        Indexing rule: pad top-left is (x_bl - cL, y_bl - cB).
        """
        group = self.engine.groups[gid]
        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))

        cL, cR, cB, cT = self.engine._clearance_lrtb(group, rot)
        cL_i, cR_i, cB_i, cT_i = int(cL), int(cR), int(cB), int(cT)
        kw = max(1, w_i + cL_i + cR_i)
        kh = max(1, h_i + cB_i + cT_i)

        inv = self.engine._invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv.dtype)
        ov = F.conv2d(inv, kernel, padding=0).squeeze(0).squeeze(0)
        return (ov == 0), int(cL_i), int(cB_i)

    def create_mask(self) -> torch.Tensor:
        gid = self._next_gid()
        if gid is None:
            return torch.zeros((self.coarse_grid * self.coarse_grid,), dtype=torch.bool, device=self.device)

        group = self.engine.groups[gid]
        rot = int(self.rot if group.rotatable else 0)
        body_ok = self._valid_top_left_body(gid=gid, rot=rot)  # bool[H2,W2]
        pad_ok, cL, cB = self._valid_top_left_pad(gid=gid, rot=rot)  # bool[H3,W3]

        H2, W2 = int(body_ok.shape[0]), int(body_ok.shape[1])
        H3, W3 = int(pad_ok.shape[0]), int(pad_ok.shape[1])

        g = int(self.coarse_grid)
        cell_w, cell_h = self.cell_wh()

        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))
        x_bl = torch.round(cx - (w_i / 2.0)).to(torch.long)
        y_bl = torch.round(cy - (h_i / 2.0)).to(torch.long)

        # body index
        inside_body = (x_bl >= 0) & (y_bl >= 0) & (x_bl < W2) & (y_bl < H2)
        idx_body = (y_bl * W2 + x_bl).to(torch.long)
        flat_body = body_ok.reshape(-1)

        # pad index (shifted by clearance)
        px = (x_bl - int(cL)).to(torch.long)
        py = (y_bl - int(cB)).to(torch.long)
        inside_pad = (px >= 0) & (py >= 0) & (px < W3) & (py < H3)
        idx_pad = (py * W3 + px).to(torch.long)
        flat_pad = pad_ok.reshape(-1)

        ok = torch.zeros((g, g), device=self.device, dtype=torch.bool)
        ok[inside_body] = flat_body[idx_body[inside_body]]
        # Pad must be inside bounds; otherwise invalid.
        ok = ok & inside_pad
        ok[inside_pad] = ok[inside_pad] & flat_pad[idx_pad[inside_pad]]
        return ok.reshape(-1)

    def _build_obs(self) -> Dict[str, Any]:
        assert self.mask is not None
        obs = dict(self.engine._build_obs())
        # Attach AlphaChip graph tensors (AlphaChip-specific; do not come from engine caches).
        gid = self._next_gid()
        cur_idx = int(self.engine.gid_to_idx.get(gid, 0)) if gid is not None else 0
        obs["x"] = self._build_x()
        obs["edge_index"] = self.edge_index
        obs["edge_attr"] = self.edge_attr
        obs["current_node"] = torch.tensor([cur_idx], dtype=torch.long, device=self.device)
        obs["netlist_metadata"] = self._build_netlist_metadata()
        obs["action_mask"] = self.mask
        return obs

    def _build_x(self) -> torch.Tensor:
        """Build node features x: float32 [N,8] from engine state."""
        N = int(len(self.engine.node_ids))
        x = torch.zeros((N, 8), dtype=torch.float32, device=self.device)
        # static features (w/h)
        for gid, i in self.engine.gid_to_idx.items():
            gg = self.engine.groups[gid]
            x[int(i), 0] = float(gg.width) / float(self.engine.grid_width)
            x[int(i), 1] = float(gg.height) / float(self.engine.grid_height)
        # dynamic features (placed + center + rot)
        for gid in self.engine.placed:
            idx = self.engine.gid_to_idx.get(gid, None)
            if idx is None:
                continue
            x_bl, y_bl, rot = self.engine.positions[gid]
            cx, cy = self.engine.center_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))
            x[int(idx), 2] = 1.0
            x[int(idx), 3] = float(cx) / float(self.engine.grid_width)
            x[int(idx), 4] = float(cy) / float(self.engine.grid_height)
            x[int(idx), 5] = float(int(rot) % 360) / 360.0
        return x

    def _build_netlist_metadata(self) -> torch.Tensor:
        """Build AlphaChip netlist_metadata: float32 [12]. Includes cost via env.cost()."""
        meta = torch.zeros((12,), dtype=torch.float32, device=self.device)
        placed_ratio = float(len(self.engine.placed)) / float(max(1, len(self.engine.node_ids)))
        remaining_ratio = float(len(self.engine.remaining)) / float(max(1, len(self.engine.node_ids)))
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

    def step(self, action: int):
        assert self.mask is not None
        x_bl, y_bl, rot, i, j = self.decode_action(int(action))
        obs_core, reward, terminated, truncated, info = self.engine.step_masked(
            action=int(action),
            x=float(x_bl),
            y=float(y_bl),
            rot=int(rot),
            mask=self.mask,
            action_space_n=int(self.action_space.n),
            extra_info={"cell_i": int(i), "cell_j": int(j)},
        )
        if terminated or truncated:
            # IMPORTANT: keep a stable observation schema for training/search code paths.
            # When done, there is no next placement; return wrapper obs with an all-false mask.
            self.mask = torch.zeros((int(self.action_space.n),), dtype=torch.bool, device=self.device)
            return self._build_obs(), reward, terminated, truncated, info

        self.mask = self.create_mask()
        return self._build_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    import time

    import torch
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    from envs.wrappers.candidate_set import CandidateSet
    from envs.json_loader import load_env
    from envs.visualizer import plot_layout

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
            print("[plot_graph] missing one of: x, edge_index, current_node")
            return

        x = _as_numpy(obs["x"])
        ei = _as_numpy(obs["edge_index"]).astype(np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            print(f"[plot_graph] edge_index must be [2,E], got {ei.shape}")
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

        print(f"[graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()} (cap={max_edges})")
        if 0 <= cur < n:
            print(f"[graph] cur out_deg={G.out_degree(cur)} in_deg={G.in_degree(cur)}")
        else:
            print(f"[graph] WARNING: current_node={cur} out of range [0,{n})")

        pos = nx.spring_layout(G, seed=42)
        node_colors = ["#d62728" if int(i) == int(cur) else "#1f77b4" for i in G.nodes()]
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_size=110, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=0.8, alpha=0.25)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    ENV_JSON = "env_configs/constraints_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    env = AlphaChipWrapperEnv(engine=engine, coarse_grid=32, rot=0)

    t0 = time.perf_counter()
    obs, _info = env.reset(options=loaded.reset_kwargs)
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(obs["action_mask"].sum().item())
    a = int(torch.where(obs["action_mask"])[0][0].item()) if valid > 0 else 0

    # Print obs tensors + visualize graph structure.
    _print_alphachip_obs({k: obs[k] for k in obs.keys()})
    _plot_alphachip_graph({k: obs[k] for k in obs.keys()}, title="AlphaChip obs graph (reset)")

    # Build BL candidates for visualization (avoid plotting all invalid points).
    gid = env.engine.remaining[0] if env.engine.remaining else None
    g = int(env.coarse_grid)
    idxs = torch.where(obs["action_mask"])[0]
    xy = torch.zeros((int(idxs.numel()), 3), dtype=torch.long, device=device)
    for t, ai in enumerate(idxs.tolist()):
        x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
        xy[t, 0] = int(x_bl)
        xy[t, 1] = int(y_bl)
        xy[t, 2] = int(rot)

    cand0 = CandidateSet(xyrot=xy, mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device), gid=gid, meta={"g": g})
    plot_layout(env, candidate_set=cand0)

    t1 = time.perf_counter()
    obs2, _r, _term, _trunc, _info2 = env.step(a)
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot after one placement (new candidates)
    if isinstance(obs2, dict) and ("action_mask" in obs2):
        _print_alphachip_obs({k: obs2[k] for k in obs2.keys()})
        _plot_alphachip_graph({k: obs2[k] for k in obs2.keys()}, title="AlphaChip obs graph (after 1 step)")
        gid2 = env.engine.remaining[0] if env.engine.remaining else None
        idxs2 = torch.where(obs2["action_mask"])[0]
        xy2 = torch.zeros((int(idxs2.numel()), 3), dtype=torch.long, device=device)
        for t, ai in enumerate(idxs2.tolist()):
            x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
            xy2[t, 0] = int(x_bl)
            xy2[t, 1] = int(y_bl)
            xy2[t, 2] = int(rot)
        cand1 = CandidateSet(xyrot=xy2, mask=torch.ones((xy2.shape[0],), dtype=torch.bool, device=device), gid=gid2, meta={"g": g})
        plot_layout(env, candidate_set=cand1)
    else:
        plot_layout(env, candidate_set=None)

    print("[AlphaChipWrapperEnv demo]")
    print(" env=", ENV_JSON, "device=", device, "G=", env.coarse_grid)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")

