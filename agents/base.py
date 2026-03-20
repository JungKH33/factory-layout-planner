from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Any, Dict, Optional, Protocol

import torch

from envs.action import EnvAction
from envs.action import GroupId
from envs.action_space import ActionSpace
from envs.env import FactoryLayoutEnv


# ---------------------------------------------------------------------------
# Agent protocol (placement policy)
# ---------------------------------------------------------------------------

class Agent(Protocol):
    """Evaluate action_space for the given state."""

    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        """Return float32 [N] non-negative policy scores/probabilities (not necessarily normalized)."""

    def select_action(self, *, obs: dict, action_space: ActionSpace) -> int:
        """Return an action index in [0, N)."""

    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        """Return a scalar leaf value estimate for MCTS (higher should be better)."""


# ---------------------------------------------------------------------------
# Ordering agent protocol
# ---------------------------------------------------------------------------

class OrderingAgent(Protocol):
    """Reorder env.get_state().remaining before adapter candidate generation."""

    def reorder(self, *, env: FactoryLayoutEnv, obs: dict[str, Any]) -> None:
        """Mutate env.get_state().remaining through env APIs (e.g., env.reorder_remaining)."""


# ---------------------------------------------------------------------------
# Base decision adapter (ABC)
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Base decision-adapter for a FactoryLayoutEnv.

    Adapter responsibilities:
    - Build policy observation fields from engine state
    - Generate action-space mask/table from current engine state
    - Decode discrete `action` index to `EnvAction` using a given `ActionSpace`
    """

    metadata = {"render_modes": []}

    def __init__(self):
        self._engine: Optional[FactoryLayoutEnv] = None
        self.device = torch.device("cpu")
        self.mask: Optional[torch.Tensor] = None

    @property
    def engine(self) -> FactoryLayoutEnv:
        if self._engine is None:
            raise RuntimeError("adapter is not bound to engine; call adapter.bind(engine) first")
        return self._engine

    def bind(self, engine: FactoryLayoutEnv) -> None:
        """Bind runtime engine context (pipeline/search call this before adapter ops)."""
        self._engine = engine
        self.device = engine.device

    @abstractmethod
    def create_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def decode_action(self, action: int, action_space: ActionSpace) -> EnvAction:
        """Decode action index to EnvAction using action-space poses table."""
        a = self.validate_action_index(action, action_space)
        pose = action_space.poses[a]
        gid = action_space.gid
        if gid is None:
            raise ValueError("action_space.gid is required to decode EnvAction")
        return EnvAction(gid=gid, x_c=float(pose[0].item()), y_c=float(pose[1].item()))

    def num_valid_actions(self, action_space: ActionSpace) -> int:
        """Return number of valid actions in given action-space."""
        mask = action_space.mask.to(dtype=torch.bool, device=self.device).view(-1)
        return int(mask.to(torch.int64).sum().item())

    def validate_action_index(self, action: int, action_space: ActionSpace) -> int:
        """Validate action index against action-space range and mask."""
        mask = action_space.mask.to(dtype=torch.bool, device=self.device).view(-1)
        n_actions = int(mask.shape[0])
        valid_n = int(mask.to(torch.int64).sum().item())
        if n_actions <= 0 or valid_n <= 0:
            raise ValueError("no_valid_actions")
        a = int(action)
        if a < 0 or a >= n_actions:
            raise IndexError(f"action index out of range: {a} not in [0,{n_actions})")
        if not bool(mask[a].item()):
            raise ValueError(f"selected action is masked: index={a}")
        return a

    def current_gid(self) -> Optional[GroupId]:
        eng = self.engine
        if not eng.get_state().remaining:
            return None
        return eng.get_state().remaining[0]

    def action_space_seed(self) -> int:
        """Deterministic seed from current engine state."""
        eng = self.engine
        parts: list[str] = [
            str(int(eng.get_state().step_count)),
            ",".join(str(gid) for gid in eng.get_state().remaining),
        ]
        for gid in sorted(eng.get_state().placed, key=lambda x: str(x)):
            p = eng.get_state().placements.get(gid, None)
            if p is None:
                continue
            x_c = float(getattr(p, "x_c", (float(getattr(p, "min_x", 0.0)) + float(getattr(p, "max_x", 0.0))) / 2.0))
            y_c = float(getattr(p, "y_c", (float(getattr(p, "min_y", 0.0)) + float(getattr(p, "max_y", 0.0))) / 2.0))
            parts.append(f"{gid}:{x_c:.4f}:{y_c:.4f}")
        raw = "|".join(parts).encode("utf-8", errors="ignore")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], byteorder="big", signed=False) & 0x7FFFFFFF

    def _score_poses(self, gid: GroupId, poses: torch.Tensor) -> torch.Tensor:
        """Score candidate center poses — min delta cost across PLACEABLE variants.

        poses: [N, 2] float tensor of (x_c, y_c).
        Returns: [N] float32 — per-position minimum delta cost across all
        placeable rotation/mirror variants.  Positions with no placeable
        variant get inf.
        """
        spec = self.engine.group_specs[gid]
        return spec.cost_batch(
            gid=gid,
            poses=poses,
            state=self.engine.get_state(),
            reward=self.engine.reward_composer,
        )

    def build_action_space(self) -> ActionSpace:
        """Generate action_space from current engine state.

        If ``build_observation()`` already called ``create_mask()`` and
        stored ``self.mask``, the existing mask is reused to avoid
        redundant computation.
        """
        if self.mask is None:
            self.mask = self.create_mask()
        if not isinstance(self.mask, torch.Tensor):
            raise TypeError("create_mask() must return torch.Tensor")

        mask = self.mask.to(dtype=torch.bool, device=self.device).view(-1)
        n_actions = int(mask.shape[0])
        gid = self.current_gid()

        poses_raw = getattr(self, "action_poses", None)
        if not isinstance(poses_raw, torch.Tensor):
            raise ValueError("adapter must provide torch.Tensor action_poses from create_mask()")
        poses = poses_raw.to(dtype=torch.float32, device=self.device)
        if poses.ndim != 2 or int(poses.shape[0]) != n_actions or int(poses.shape[1]) != 2:
            raise ValueError(
                f"action_poses must have shape [N,2], got {tuple(poses.shape)} for N={n_actions}"
            )

        return ActionSpace(poses=poses, mask=mask, gid=gid)

    @abstractmethod
    def build_observation(self) -> Dict[str, Any]:
        """Build policy observation from current engine state.

        Each adapter defines its own observation format tailored to the
        model/agent it serves.  Greedy adapters return
        ``{"action_delta": Tensor[N]}`` for the greedy agent.
        """
        raise NotImplementedError

    # ---- state api (for wrapped search/MCTS) ----
    def get_state_copy(self) -> Dict[str, object]:
        """Return adapter-only state copy for deterministic restore in search algorithms."""
        state: Dict[str, object] = {}
        if isinstance(self.mask, torch.Tensor):
            state["mask"] = self.mask.clone()
        else:
            state["mask"] = None
        return state

    def set_state(self, state: Dict[str, object]) -> None:
        """Restore adapter-only state produced by `get_state_copy`."""
        m = state.get("mask", None)
        if isinstance(m, torch.Tensor):
            self.mask = m.to(device=self.device, dtype=torch.bool).clone()
        else:
            self.mask = None
