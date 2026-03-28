from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data

from envs.action_space import ActionSpace
from ...base import Agent

from .model import AlphaChip

logger = logging.getLogger(__name__)


def _obs_to_pyg_data(obs: dict) -> Data:
    # env provides torch tensors already.
    return Data(
        x=obs["x"],
        edge_index=obs["edge_index"],
        edge_attr=obs["edge_attr"],
        netlist_metadata=obs["netlist_metadata"].view(1, -1),
        current_node=obs["current_node"].view(-1),
    )


@dataclass
class AlphaChipAgent:
    """AlphaChip policy agent (coarse actionspace only).

    - Expects action_space action count == coarse_grid*coarse_grid.
    - Uses env observation tensors to build PyG Data.
    - Loads checkpoint by explicit `checkpoint_path` only (no run_id shortcuts).
    """

    coarse_grid: int
    checkpoint_path: Optional[str] = None
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.coarse_grid = int(self.coarse_grid)
        self.device = self.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = AlphaChip(
            metadata_dim=12,
            node_feature_dim=8,
            max_grid_size=int(self.coarse_grid),
            device=self.device,
        )
        self.model.eval()

        if self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

        # One-time info (requested): AlphaChip expects coarse actionspace.
        logger.info(
            "AlphaChipAgent expects coarse actionspace: N=%d (G=%d)",
            self.coarse_grid * self.coarse_grid,
            self.coarse_grid,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        obj = torch.load(str(path), map_location=self.device)
        state = obj.get("model") if isinstance(obj, dict) else obj
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format at {path} (expected state_dict or dict with key 'model').")
        self.model.load_state_dict(state)
        self.model.eval()
        meta = obj.get("meta") if isinstance(obj, dict) else None
        logger.info("alphachip_agent loaded_checkpoint=%s", path)
        if isinstance(meta, dict):
            logger.info("alphachip_agent checkpoint_meta=%s", meta)

    @torch.no_grad()
    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        expected_n = int(self.coarse_grid * self.coarse_grid)
        n = int(action_space.valid_mask.shape[0])
        if n != expected_n:
            raise ValueError(
                f"AlphaChipAgent(policy) requires coarse actionspace: expected N={expected_n}, got N={n}. "
                f"(If you want TopK+AlphaChip, you need a mapping from TopK action_space to coarse logits.)"
            )

        data = _obs_to_pyg_data(obs)
        # AlphaChip expects int32 mask with batch dim [1, A].
        mask_flat = action_space.valid_mask.view(1, -1).to(dtype=torch.int32, device=self.device)
        logits_flat, _value = self.model(data, mask_flat=mask_flat, is_eval=True)  # [1, A]
        logits = logits_flat[0].to(dtype=torch.float32)
        # Convert to non-negative scores for downstream (MCTS expects non-negative).
        scores = torch.softmax(logits, dim=-1)
        return scores

    def select_action(self, *, obs: dict, action_space: ActionSpace) -> int:
        pol = self.policy(obs=obs, action_space=action_space)
        pol = pol.masked_fill(~action_space.valid_mask, float("-inf"))
        return int(torch.argmax(pol).item()) if pol.numel() > 0 else 0

    @torch.no_grad()
    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        expected_n = int(self.coarse_grid * self.coarse_grid)
        n = int(action_space.valid_mask.shape[0])
        if n != expected_n:
            raise ValueError(
                f"AlphaChipAgent(value) requires coarse actionspace: expected N={expected_n}, got N={n}. "
                f"(If you want TopK+AlphaChip, you need a mapping from TopK action_space to coarse logits.)"
            )

        data = _obs_to_pyg_data(obs)
        mask_flat = action_space.valid_mask.view(1, -1).to(dtype=torch.int32, device=self.device)
        _logits_flat, value_t = self.model(data, mask_flat=mask_flat, is_eval=True)
        return float(value_t.view(-1)[0].item()) if isinstance(value_t, torch.Tensor) and value_t.numel() > 0 else 0.0


if __name__ == "__main__":
    from envs.env_loader import load_env
    from actionspace.coarse import CoarseSelector

    ENV_JSON = "envs/env_configs/basic_01.json"
    COARSE_GRID = 32
    CHECKPOINT_PATH = None  # set to a .ckpt if available

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=dev)
    env = loaded.env
    env.log = False
    obs, _info = env.reset(options=loaded.reset_kwargs)

    selector = CoarseSelector(coarse_grid=COARSE_GRID, rot=0)
    agent: Agent = AlphaChipAgent(coarse_grid=COARSE_GRID, checkpoint_path=CHECKPOINT_PATH, device=dev)

    t0 = time.perf_counter()
    action_space = selector.build(env)
    a = agent.select_action(obs=obs, action_space=action_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("alphachip_agent demo")
    print(" env=", ENV_JSON, "device=", dev, "next_gid=", (env.get_state().remaining[0] if env.get_state().remaining else None))
    print(" action=", a, "valid=", int(action_space.valid_mask.sum().item()))
    print(f" elapsed_ms={dt_ms:.2f}")
