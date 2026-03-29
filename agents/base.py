from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Any, Dict, Optional, Protocol, Tuple

import torch

from envs.action import EnvAction
from envs.action import GroupId
from envs.action_space import ActionSpace
from envs.env import FactoryLayoutEnv
from envs.placement.base import GroupPlacement


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

    def __init__(
        self,
        *,
        expand_variants: bool = False,
        max_variants: int = 4,
    ):
        self._engine: Optional[FactoryLayoutEnv] = None
        self.device = torch.device("cpu")
        self.mask: Optional[torch.Tensor] = None
        self.expand_variants = bool(expand_variants)
        self.max_variants = int(max_variants)
        self.action_variant_indices: Optional[torch.Tensor] = None

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
        """Decode action index to EnvAction (center coords — env does center→BL)."""
        a = self.validate_action_index(action, action_space)
        pose = action_space.centers[a]  # center coords
        gid = action_space.group_id
        if gid is None:
            raise ValueError("action_space.group_id is required to decode EnvAction")
        variant_index = None
        if action_space.variant_indices is not None:
            variant_index = int(action_space.variant_indices[a].item())
        return EnvAction(
            group_id=gid,
            x_center=float(pose[0].item()),
            y_center=float(pose[1].item()),
            variant_index=variant_index,
        )

    def num_valid_actions(self, action_space: ActionSpace) -> int:
        """Return number of valid actions in given action-space."""
        mask = action_space.valid_mask.to(dtype=torch.bool, device=self.device).view(-1)
        return int(mask.to(torch.int64).sum().item())

    def validate_action_index(self, action: int, action_space: ActionSpace) -> int:
        """Validate action index against action-space range and mask."""
        mask = action_space.valid_mask.to(dtype=torch.bool, device=self.device).view(-1)
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
            x_center = float(getattr(p, "x_center", (float(getattr(p, "min_x", 0.0)) + float(getattr(p, "max_x", 0.0))) / 2.0))
            y_center = float(getattr(p, "y_center", (float(getattr(p, "min_y", 0.0)) + float(getattr(p, "max_y", 0.0))) / 2.0))
            parts.append(f"{gid}:{x_center:.4f}:{y_center:.4f}")
        raw = "|".join(parts).encode("utf-8", errors="ignore")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], byteorder="big", signed=False) & 0x7FFFFFFF

    def _centers_to_bl(
        self,
        gid: GroupId,
        centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[N, 2] centers → ([N, V] x_bl, [N, V] y_bl).  Fully vectorised."""
        spec = self.engine.group_specs[gid]
        bw = spec.body_widths   # [V]
        bh = spec.body_heights  # [V]
        x_bl = torch.round(centers[:, 0:1] - bw.unsqueeze(0) / 2.0).to(torch.long)  # [N, V]
        y_bl = torch.round(centers[:, 1:2] - bh.unsqueeze(0) / 2.0).to(torch.long)  # [N, V]
        return x_bl, y_bl

    def _build_center_map(self, gid: GroupId) -> torch.Tensor:
        """BL placeable maps shifted to center coords (union over shapes).

        Moved from ``StaticSpec.placeable_center_map``.
        """
        spec = self.engine.group_specs[gid]
        state = self.engine.get_state()
        H, W = state.maps.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        for shape_key in spec._variants_by_shape:
            body_mask, clearance_mask, clearance_origin, is_rect = spec.shape_tensors(shape_key)
            bl_map = state.placeable_map(
                gid=gid,
                body_mask=body_mask,
                clearance_mask=clearance_mask,
                clearance_origin=clearance_origin,
                is_rectangular=is_rect,
            )
            bh_s, bw_s = int(body_mask.shape[0]), int(body_mask.shape[1])
            dx, dy = bw_s // 2, bh_s // 2
            src_h = min(H - dy, int(bl_map.shape[0]))
            src_w = min(W - dx, int(bl_map.shape[1]))
            if src_h > 0 and src_w > 0:
                result[dy:dy + src_h, dx:dx + src_w] |= bl_map[:src_h, :src_w]
        return result

    def _score_poses(
        self,
        gid: GroupId,
        poses: torch.Tensor,
        *,
        per_variant: bool = False,
    ) -> torch.Tensor:
        """Score candidate center poses via BL-only spec API.

        per_variant=False (default): [N] float32 — min delta cost across
        all placeable variants.  Positions with no placeable variant get inf.

        per_variant=True: [N, V] float32 — per-variant delta cost.
        Non-placeable variant slots are inf.
        """
        spec = self.engine.group_specs[gid]
        x_bl, y_bl = self._centers_to_bl(gid, poses)
        return spec.score_batch(
            gid=gid,
            x_bl=x_bl,
            y_bl=y_bl,
            state=self.engine.get_state(),
            reward=self.engine.reward_composer,
            per_variant=per_variant,
        )

    def _apply_variant_expansion(
        self,
        gid: GroupId,
        center_poses: torch.Tensor,
        center_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Expand center candidates into (center, variant) pairs.

        Takes center-only candidates from any subclass's ``create_mask()`` and
        expands them into ``(center, variant_index)`` pairs, keeping the
        top *k* by cost.  Sets ``self.action_poses``, ``self.action_costs``,
        and ``self.action_variant_indices``.

        ``score_batch(per_variant=True)`` already returns inf for
        non-placeable variant slots, so no separate ``placeable_batch``
        call is needed.

        Returns:
            ``[k]`` bool mask.
        """
        spec = self.engine.group_specs[gid]
        V = len(spec.variants)
        max_vi = min(V, self.max_variants)

        vmask = center_mask.to(dtype=torch.bool, device=self.device).view(-1)
        vidx = torch.where(vmask)[0]

        empty = self._empty_variant_output(k)
        if int(vidx.numel()) == 0:
            return empty

        valid_poses = center_poses[vidx]  # [M, 2]

        # Single batch call — inf where not placeable
        cost_nv = self._score_poses(gid, valid_poses, per_variant=True)  # [M, V]

        # Per-center: keep only top max_variants variants
        if V > max_vi:
            cost_nv = cost_nv.clone()
            _, top_vi = torch.topk(cost_nv, k=max_vi, dim=1, largest=False)
            keep = torch.zeros_like(cost_nv, dtype=torch.bool)
            keep.scatter_(1, top_vi, True)
            cost_nv[~keep] = float("inf")

        # Flatten and pick top-k finite entries
        flat_cost = cost_nv.reshape(-1)  # [M*V]
        n_finite = int(torch.isfinite(flat_cost).sum().item())
        if n_finite == 0:
            return empty

        pick_k = min(k, n_finite)
        _, topk_flat = torch.topk(flat_cost, k=pick_k, largest=False)
        ci = topk_flat // V
        vi = topk_flat % V

        out_poses = torch.zeros((k, 2), dtype=torch.float32, device=self.device)
        out_delta = torch.full((k,), float("inf"), dtype=torch.float32, device=self.device)
        out_vi = torch.zeros((k,), dtype=torch.int64, device=self.device)
        out_mask = torch.zeros((k,), dtype=torch.bool, device=self.device)

        out_poses[:pick_k] = valid_poses[ci]
        out_delta[:pick_k] = flat_cost[topk_flat]
        out_vi[:pick_k] = vi
        out_mask[:pick_k] = True

        self.action_poses = out_poses
        self.action_costs = out_delta
        self.action_variant_indices = out_vi
        return out_mask

    def _empty_variant_output(self, k: int) -> torch.Tensor:
        self.action_poses = torch.zeros((k, 2), dtype=torch.float32, device=self.device)
        self.action_costs = torch.full((k,), float("inf"), dtype=torch.float32, device=self.device)
        self.action_variant_indices = None
        return torch.zeros((k,), dtype=torch.bool, device=self.device)

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

        vi_raw = getattr(self, "action_variant_indices", None)
        vi_indices = None
        if isinstance(vi_raw, torch.Tensor):
            vi_t = vi_raw.to(dtype=torch.int64, device=self.device)
            if vi_t.ndim == 1 and int(vi_t.shape[0]) == n_actions:
                vi_indices = vi_t

        return ActionSpace(centers=poses, valid_mask=mask, group_id=gid, variant_indices=vi_indices)

    @abstractmethod
    def build_observation(self) -> Dict[str, Any]:
        """Build policy observation from current engine state.

        Each adapter defines its own observation format tailored to the
        model/agent it serves.  Greedy adapters return
        ``{"action_costs": Tensor[N]}`` for the greedy agent.
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
        vit = getattr(self, "action_variant_indices", None)
        state["action_variant_indices"] = vit.clone() if isinstance(vit, torch.Tensor) else None
        return state

    def set_state(self, state: Dict[str, object]) -> None:
        """Restore adapter-only state produced by `get_state_copy`."""
        m = state.get("mask", None)
        if isinstance(m, torch.Tensor):
            self.mask = m.to(device=self.device, dtype=torch.bool).clone()
        else:
            self.mask = None
        vit = state.get("action_variant_indices", None)
        if isinstance(vit, torch.Tensor):
            self.action_variant_indices = vit.to(device=self.device, dtype=torch.int64).clone()
        else:
            self.action_variant_indices = None


# ---------------------------------------------------------------------------
# Base hierarchical adapter (for region/cell-based adapters)
# ---------------------------------------------------------------------------

class BaseHierarchicalAdapter(BaseAdapter):
    """Adapter that resolves actions to (gid, GroupPlacement, delta_cost).

    Used by region/cell-based adapters where the adapter owns variant
    resolution. Works with ``env.step_placement()`` instead of
    ``env.step_action()``.
    """

    @abstractmethod
    def resolve_action(self, action_idx: int, action_space: ActionSpace) -> Tuple[GroupId, GroupPlacement, float]:
        """Action index → (gid, GroupPlacement, delta_cost)."""
        raise NotImplementedError

    def cell_action_space(self, cell_idx: int) -> ActionSpace:
        """Return within-cell candidates as ActionSpace (for H-MCTS worker)."""
        raise NotImplementedError(f"{type(self).__name__}.cell_action_space() is not implemented")
