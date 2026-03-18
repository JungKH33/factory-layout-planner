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
        return EnvAction(gid=gid, x=int(pose[0].item()), y=int(pose[1].item()), orient=int(pose[2].item()))

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
            x_bl, y_bl, orient = p.pose()
            parts.append(f"{gid}:{int(x_bl)}:{int(y_bl)}:{int(orient)}")
        raw = "|".join(parts).encode("utf-8", errors="ignore")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], byteorder="big", signed=False) & 0x7FFFFFFF

    def _score_poses(self, gid: GroupId, poses: torch.Tensor) -> torch.Tensor:
        """Score candidate poses by evaluating all (rotation, mirror) variants.

        poses: [N, 3] tensor of (x_bl, y_bl, orient).
        Returns: [N] float32 — per-position minimum delta cost across all
        variants in each orient family.
        """
        spec = self.engine.group_specs[gid]
        needed = self.engine.reward_required
        N = int(poses.shape[0])
        if N == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)

        best = torch.full((N,), float('inf'), dtype=torch.float32, device=self.device)

        for o in poses[:, 2].unique().tolist():
            o_int = int(o)
            o_mask = (poses[:, 2] == o_int)
            sub = poses[o_mask]
            M = int(sub.shape[0])
            if M == 0:
                continue

            rotations = ((0, 180) if o_int == 0 else (90, 270)) if spec.rotatable else ((0,) if o_int == 0 else ())
            mirrors = (False, True) if spec.mirrorable else (False,)

            for rot in rotations:
                for m in mirrors:
                    features = spec.build_candidate_features(
                        x_bl=sub[:, 0], y_bl=sub[:, 1],
                        rotation=rot, mirror=m, needed=needed,
                    )
                    scores = self._features_to_delta(gid, sub, features)
                    best[o_mask] = torch.min(best[o_mask], scores)

        return best

    def _features_to_delta(
        self,
        gid: GroupId,
        poses: torch.Tensor,
        features: dict,
    ) -> torch.Tensor:
        """Build ActionSpace from feature dict and return delta cost."""
        entries = features.get("entries", None)
        exits = features.get("exits", None)
        entries_mask = None
        exits_mask = None
        if entries is not None:
            entries_mask = torch.ones(entries.shape[:2], dtype=torch.bool, device=self.device)
        if exits is not None:
            exits_mask = torch.ones(exits.shape[:2], dtype=torch.bool, device=self.device)
        aspace = ActionSpace(
            poses=poses.to(dtype=torch.long, device=self.device),
            mask=torch.ones(poses.shape[0], dtype=torch.bool, device=self.device),
            gid=gid,
            entries=entries, exits=exits,
            entries_mask=entries_mask, exits_mask=exits_mask,
            min_x=features.get("min_x"), max_x=features.get("max_x"),
            min_y=features.get("min_y"), max_y=features.get("max_y"),
        )
        return self.engine.delta_cost(gid, aspace)

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
        poses = poses_raw.to(dtype=torch.long, device=self.device)
        if poses.ndim != 2 or int(poses.shape[0]) != n_actions or int(poses.shape[1]) != 3:
            raise ValueError(
                f"action_poses must have shape [N,3], got {tuple(poses.shape)} for N={n_actions}"
            )

        meta: Dict[str, Any] = {}
        delta_raw = getattr(self, "action_delta", None)
        if isinstance(delta_raw, torch.Tensor):
            delta = delta_raw.to(dtype=torch.float32, device=self.device).view(-1)
            if int(delta.shape[0]) == n_actions:
                meta["action_delta"] = delta

        return ActionSpace(poses=poses, mask=mask, gid=gid, meta=(meta if meta else None))

    @abstractmethod
    def build_observation(self) -> Dict[str, Any]:
        """Build policy observation from current engine state.

        Each adapter defines its own observation format tailored to the
        model/agent it serves.  Greedy adapters may return ``{}`` since
        the greedy agent only uses ``action_space.meta["action_delta"]``.
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
