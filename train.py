# coding=utf-8
# Copyright 2026.
#
# MaskPlace PPO training (Tianshou) for factory-layout.
#
# Goals:
# - MaskPlace-only (no AlphaChip/PyG here).
# - Object-oriented structure: config → builder → trainer.
# - Compatibility guards for tianshou/gym API differences.
#
# Notes:
# - Env wrappers return torch.Tensors; we convert to numpy for Tianshou collectors.
# - MaskPlace consumes obs["state"] and returns (action_probs, value).

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import uuid
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tianshou.algorithm.modelfree.ppo import PPO  # type: ignore
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy  # type: ignore
from tianshou.algorithm.optim import AdamOptimizerFactory  # type: ignore
from tianshou.data.collector import Collector  # type: ignore
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams  # type: ignore
from tianshou.utils.net.common import AbstractDiscreteActor, ModuleWithVectorOutput  # type: ignore

from agents.placement.maskplace import MaskPlaceModel, MaskPlaceAdapter
from envs.env_loader import load_env

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class TrainConfig:
    # mode
    mode: str  # "maskplace" | "alphachip"

    # env / model
    env_json: str
    collision_check: str
    grid: int
    rot: int
    soft_coefficient: float

    # alphachip (coarse actionspace)
    coarse_grid: int
    alphachip_rot: int

    # tianshou / ppo
    train_env_num: int
    test_env_num: int
    epoch: int
    step_per_epoch: int
    step_per_collect: int
    repeat_per_collect: int
    batch_size: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float

    # runtime
    device: str
    # store obs["state"] tail as uint8 in numpy (buffer-friendly) and dequantize to float32 on model input
    state_uint8: bool


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()

    # mode
    p.add_argument("--mode", type=str, default="alphachip", choices=["maskplace", "alphachip"])

    # env/model
    p.add_argument("--env-json", type=str, default="envs/env_configs/basic_01.json")
    p.add_argument("--collision-check", type=str, default="auto", choices=["auto", "conv", "prefixsum"])
    p.add_argument("--maskplace-grid", type=int, default=224)
    p.add_argument("--maskplace-rot", type=int, default=0)
    p.add_argument("--soft-coefficient", type=float, default=1.0)

    # alphachip
    p.add_argument("--coarse-grid", type=int, default= 128)
    p.add_argument("--alphachip-rot", type=int, default=0)

    # rollout/training
    p.add_argument("--train-env-num", type=int, default=1)
    p.add_argument("--test-env-num", type=int, default=1)
    p.add_argument("--epoch", type=int, default= 100)
    p.add_argument("--step-per-epoch", type=int, default=2000)
    p.add_argument("--step-per-collect", type=int, default= 1000)
    p.add_argument("--repeat-per-collect", type=int, default=8)
    p.add_argument("--batch-size", type=int, default= 128)

    # ppo hyperparams
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--state-uint8", action="store_true", default=True)
    p.add_argument("--no-state-uint8", action="store_false", dest="state_uint8")

    a = p.parse_args()
    return TrainConfig(
        mode=str(a.mode),
        env_json=str(a.env_json),
        collision_check=str(a.collision_check),
        grid=int(a.maskplace_grid),
        rot=int(a.maskplace_rot),
        soft_coefficient=float(a.soft_coefficient),
        coarse_grid=int(a.coarse_grid),
        alphachip_rot=int(a.alphachip_rot),
        train_env_num=int(a.train_env_num),
        test_env_num=int(a.test_env_num),
        epoch=int(a.epoch),
        step_per_epoch=int(a.step_per_epoch),
        step_per_collect=int(a.step_per_collect),
        repeat_per_collect=int(a.repeat_per_collect),
        batch_size=int(a.batch_size),
        lr=float(a.lr),
        gamma=float(a.gamma),
        gae_lambda=float(a.gae_lambda),
        clip_ratio=float(a.clip_ratio),
        vf_coef=float(a.vf_coef),
        ent_coef=float(a.ent_coef),
        max_grad_norm=float(a.max_grad_norm),
        device=str(a.device),
        state_uint8=bool(a.state_uint8),
    )


# -----------------------------
# Small compatibility utilities
# -----------------------------


def unwrap_space(space_any: Any) -> gym.Space:
    """Tianshou VectorEnv may expose action_space/observation_space as list/tuple per-env."""
    return space_any[0] if isinstance(space_any, (list, tuple)) else space_any


def to_numpy_obs(obs: Any) -> Any:
    """Recursively convert torch.Tensor → numpy for Tianshou Collector."""
    if torch.is_tensor(obs):
        return obs.detach().cpu().numpy()
    if isinstance(obs, dict):
        return {k: to_numpy_obs(v) for k, v in obs.items()}
    return obs


def pack_state_uint8(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Pack obs['state'] into (pos_idx:int32, state_u8:uint8) to reduce replay-buffer memory.

    - pos_idx is stored separately as int32 (no scaling).
    - the remaining state tail is assumed to be mostly within [0,1], so it is quantized to uint8 via round(x*255).
    """
    if "state" not in obs:
        return obs
    st = obs["state"]
    if not isinstance(st, np.ndarray) or st.size == 0:
        return obs

    st2 = st
    is_batched = (st2.ndim == 2)
    if st2.ndim == 1:
        st2 = st2.reshape(1, -1)

    pos = np.round(st2[:, 0]).astype(np.int32)
    tail = st2[:, 1:].astype(np.float32, copy=False)
    tail = np.clip(tail, 0.0, 1.0)
    tail_u8 = np.round(tail * 255.0).astype(np.uint8)

    obs2 = dict(obs)
    obs2.pop("state", None)
    obs2["pos_idx"] = pos if is_batched else pos.reshape(-1)
    obs2["state_u8"] = tail_u8 if is_batched else tail_u8.reshape(-1)
    return obs2


def obs_state_to_tensor(obs: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    """Return float32 state tensor [B, S] from obs which may contain:
    - 'state' (float32), or
    - 'pos_idx' (int32) + 'state_u8' (uint8) where tail is dequantized via /255.
    """
    if "state" in obs:
        st = obs["state"]
        st_t = st.to(device=device, dtype=torch.float32) if torch.is_tensor(st) else torch.as_tensor(st, device=device, dtype=torch.float32)
        if st_t.dim() == 1:
            st_t = st_t.view(1, -1)
        return st_t

    if ("pos_idx" not in obs) or ("state_u8" not in obs):
        raise KeyError("obs must contain either 'state' or ('pos_idx' and 'state_u8').")

    pos = obs["pos_idx"]
    tail = obs["state_u8"]
    pos_t = pos.to(device=device, dtype=torch.float32) if torch.is_tensor(pos) else torch.as_tensor(pos, device=device, dtype=torch.float32)
    tail_t = tail.to(device=device, dtype=torch.float32) if torch.is_tensor(tail) else torch.as_tensor(tail, device=device, dtype=torch.float32)

    if pos_t.dim() == 0:
        pos_t = pos_t.view(1, 1)
    elif pos_t.dim() == 1:
        pos_t = pos_t.view(-1, 1)

    if tail_t.dim() == 1:
        tail_t = tail_t.view(1, -1)

    tail_t = tail_t / 255.0
    return torch.cat([pos_t, tail_t], dim=1)


# -----------------------------
# Gym wrappers for training stability
# -----------------------------


class NumpyObsWrapper(gym.Wrapper):
    """Convert torch-based obs to numpy-based obs for Tianshou."""

    def __init__(self, env: gym.Env, *, state_uint8: bool):
        super().__init__(env)
        self._state_uint8 = bool(state_uint8)

    def reset(self, **kwargs):
        # Inject JSON reset kwargs if present (DummyVectorEnv calls reset() without args).
        if "options" not in kwargs:
            # Backward compat: allow both names.
            if hasattr(self.env, "_reset_kwargs"):
                kwargs["options"] = getattr(self.env, "_reset_kwargs")
            elif hasattr(self.env, "_maskplace_reset_kwargs"):
                kwargs["options"] = getattr(self.env, "_maskplace_reset_kwargs")
        obs, info = self.env.reset(**kwargs)
        obs_np = to_numpy_obs(obs)
        if self._state_uint8 and isinstance(obs_np, dict):
            obs_np = pack_state_uint8(obs_np)
        return obs_np, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs_np = to_numpy_obs(obs)
        if self._state_uint8 and isinstance(obs_np, dict):
            obs_np = pack_state_uint8(obs_np)
        return obs_np, float(reward), bool(terminated), bool(truncated), info


# -----------------------------
# Tianshou actor/critic adaptors
# -----------------------------


class IdentityPreprocess(ModuleWithVectorOutput):
    """Required by AbstractDiscreteActor; we bypass preprocessing and use dict obs directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


class MaskPlaceActor(AbstractDiscreteActor):
    """Tianshou discrete actor: obs(dict)->action_probs[B,A]."""

    def __init__(self, model: MaskPlaceModel, *, action_dim: int):
        super().__init__(output_dim=int(action_dim))
        self.model = model
        self._preprocess = IdentityPreprocess(output_dim=int(action_dim))

    def get_preprocess_net(self) -> ModuleWithVectorOutput:  # type: ignore[override]
        return self._preprocess

    @staticmethod
    def _sanitize_probs(probs: torch.Tensor) -> torch.Tensor:
        # Keep policy outputs on a valid probability simplex (prevents buffer NaN failures).
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        s = probs.sum(dim=-1, keepdim=True)
        return torch.where(s > 0.0, probs / s, torch.full_like(probs, 1.0 / float(probs.shape[-1])))

    def forward(self, obs, state=None, info=None):  # type: ignore[override]
        st_t = obs_state_to_tensor(obs, device=self.model.device)
        probs, _value = self.model(st_t)
        probs = self._sanitize_probs(probs)
        return probs, state


class MaskPlaceCritic(nn.Module):
    def __init__(self, model: MaskPlaceModel):
        super().__init__()
        self.model = model

    def forward(self, obs, state=None, info=None):
        st_t = obs_state_to_tensor(obs, device=self.model.device)
        _probs, v = self.model(st_t)
        return v


def _to_torch_any(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _obs_to_pyg_batch(obs: Dict[str, Any], *, device: torch.device):
    """Convert (possibly batched) dict obs to a PyG Batch + masks.

    Supports obs from NumpyObsWrapper (numpy arrays) or direct torch tensors.
    Expected keys: x, edge_index, edge_attr, netlist_metadata, current_node, action_mask
    """
    # Lazy import so maskplace mode doesn't require torch_geometric.
    from torch_geometric.data import Batch, Data  # type: ignore

    x = obs["x"]
    edge_index = obs["edge_index"]
    edge_attr = obs["edge_attr"]
    netlist_metadata = obs["netlist_metadata"]
    current_node = obs["current_node"]
    action_mask = obs["action_mask"]

    # Determine batch size by action_mask (most reliably batched by vector env).
    am_t = _to_torch_any(action_mask, device=device, dtype=torch.bool)
    if am_t.dim() == 1:
        B = 1
        am_t = am_t.view(1, -1)
    else:
        B = int(am_t.shape[0])
        am_t = am_t.view(B, -1)

    # Helper to slice per-batch item if inputs are batched.
    def _slice_b(v: Any, b: int) -> Any:
        if torch.is_tensor(v):
            return v[b] if v.dim() >= 1 and int(v.shape[0]) == B else v
        # numpy arrays
        try:
            import numpy as np  # type: ignore

            if isinstance(v, np.ndarray) and v.ndim >= 1 and int(v.shape[0]) == B:
                return v[b]
        except Exception:
            pass
        return v

    data_list = []
    for b in range(B):
        xb = _to_torch_any(_slice_b(x, b), device=device, dtype=torch.float32)
        eib = _to_torch_any(_slice_b(edge_index, b), device=device, dtype=torch.long)
        eab = _to_torch_any(_slice_b(edge_attr, b), device=device, dtype=torch.float32)
        mb = _to_torch_any(_slice_b(netlist_metadata, b), device=device, dtype=torch.float32).view(1, -1)
        cb = _to_torch_any(_slice_b(current_node, b), device=device, dtype=torch.long).view(-1)

        data_list.append(Data(x=xb, edge_index=eib, edge_attr=eab, netlist_metadata=mb, current_node=cb))

    batch = Batch.from_data_list(data_list).to(device)
    mask_i32 = am_t.to(dtype=torch.int32)
    return batch, am_t, mask_i32


class AlphaChipActor(AbstractDiscreteActor):
    """Tianshou discrete actor for AlphaChip: obs(dict)->action_probs[B,A]."""

    def __init__(self, model: nn.Module, *, action_dim: int):
        super().__init__(output_dim=int(action_dim))
        self.model = model
        self._preprocess = IdentityPreprocess(output_dim=int(action_dim))

    def get_preprocess_net(self) -> ModuleWithVectorOutput:  # type: ignore[override]
        return self._preprocess

    def forward(self, obs, state=None, info=None):  # type: ignore[override]
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        batch, mask_bool, mask_i32 = _obs_to_pyg_batch(obs, device=device)
        logits_flat, _value = self.model(batch, mask_flat=mask_i32, is_eval=False)  # [B,A]
        logits = logits_flat.to(dtype=torch.float32)
        # If there are no valid actions, model masking can make all logits huge-negative.
        # For Categorical(logits=...), this is still finite, but we set to 0 for a clean uniform.
        no_valid = (mask_bool.sum(dim=-1) == 0)
        if bool(no_valid.any().item()):
            logits = logits.clone()
            logits[no_valid] = 0.0
        return logits, state


class AlphaChipCritic(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs, state=None, info=None):
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        batch, _mask_bool, mask_i32 = _obs_to_pyg_batch(obs, device=device)
        _logits_flat, v = self.model(batch, mask_flat=mask_i32, is_eval=False)
        return v


# -----------------------------
# Training (Tianshou OnPolicyTrainer)
# -----------------------------


def build_env_factory(*, cfg: TrainConfig) -> Tuple[Any, Dict[str, Any]]:
    """Return (make_env_fn, meta) where meta contains run_id and ckpt_dir."""
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4().hex[:6]}"
    ckpt_dir = (Path("results") / "checkpoints" / run_id).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load once to capture reset_kwargs from JSON; each env will create its own engine instance.
    loaded0 = load_env(
        cfg.env_json,
        device=torch.device("cpu"),
        collision_check=cfg.collision_check,
    )
    reset_kwargs = loaded0.reset_kwargs

    def _make_single_env() -> gym.Env:
        loaded = load_env(
            cfg.env_json,
            device=torch.device("cpu"),
            collision_check=cfg.collision_check,
        )
        engine = loaded.env
        engine.log = False
        if cfg.mode == "maskplace":
            env = MaskPlaceAdapter(
                engine=engine,
                grid=int(cfg.grid),
                soft_coefficient=float(cfg.soft_coefficient),
            )
            # pass reset kwargs via env.reset(options=...)
            env._reset_kwargs = reset_kwargs  # type: ignore[attr-defined]
            env = NumpyObsWrapper(env, state_uint8=bool(cfg.state_uint8))
        elif cfg.mode == "alphachip":
            from agents.placement.alphachip import AlphaChipAdapter

            env = AlphaChipAdapter(engine=engine, coarse_grid=int(cfg.coarse_grid))
            env._reset_kwargs = reset_kwargs  # type: ignore[attr-defined]
            env = NumpyObsWrapper(env, state_uint8=False)
        else:
            raise ValueError(f"Unknown mode={cfg.mode!r} (expected 'maskplace'|'alphachip')")
        return env

    return _make_single_env, {"run_id": run_id, "ckpt_dir": ckpt_dir}


def build_algo_and_collectors(*, cfg: TrainConfig, make_env_fn) -> Tuple[PPO, Collector, Collector, nn.Module, gym.Space]:
    train_envs = DummyVectorEnv([lambda: make_env_fn() for _ in range(int(cfg.train_env_num))])
    test_envs = DummyVectorEnv([lambda: make_env_fn() for _ in range(int(cfg.test_env_num))])

    action_space = unwrap_space(train_envs.action_space)
    action_dim = int(getattr(action_space, "n"))

    device = torch.device(str(cfg.device))
    if cfg.mode == "maskplace":
        model = MaskPlaceModel(grid=int(cfg.grid), device=device, soft_coefficient=float(cfg.soft_coefficient)).to(device)
        actor = MaskPlaceActor(model, action_dim=action_dim)
        critic = MaskPlaceCritic(model)
    elif cfg.mode == "alphachip":
        from agents.placement.alphachip import AlphaChip

        model = AlphaChip(
            metadata_dim=12,
            node_feature_dim=8,
            max_grid_size=int(cfg.coarse_grid),
            device=device,
        ).to(device)
        actor = AlphaChipActor(model, action_dim=action_dim)
        critic = AlphaChipCritic(model)
    else:
        raise ValueError(f"Unknown mode={cfg.mode!r} (expected 'maskplace'|'alphachip')")

    dist_fn = (lambda probs: Categorical(probs=probs)) if cfg.mode == "maskplace" else (lambda logits: Categorical(logits=logits))
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        deterministic_eval=False,
        action_space=action_space,
        observation_space=None,
        action_scaling=False,
    )

    algo = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=float(cfg.lr)),
        eps_clip=float(cfg.clip_ratio),
        vf_coef=float(cfg.vf_coef),
        ent_coef=float(cfg.ent_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        gae_lambda=float(cfg.gae_lambda),
        gamma=float(cfg.gamma),
        max_batchsize=int(cfg.batch_size),
    )

    train_collector = Collector(algo, train_envs)
    test_collector = Collector(algo, test_envs)
    return algo, train_collector, test_collector, model, action_space


def save_ckpt(*, path: Path, model: nn.Module, meta: Dict[str, Any]) -> None:
    payload = {"model": model.state_dict(), "meta": dict(meta)}
    torch.save(payload, str(path))


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    cfg = parse_args()
    make_env_fn, meta = build_env_factory(cfg=cfg)
    algo, train_collector, test_collector, model, _action_space = build_algo_and_collectors(cfg=cfg, make_env_fn=make_env_fn)

    run_id = str(meta["run_id"])
    ckpt_dir: Path = meta["ckpt_dir"]

    def _compute_score_fn(collect_stats) -> float:
        # tianshou 2.0.0: CollectStats.returns_stat.mean after refresh
        if getattr(collect_stats, "returns_stat", None) is None:
            try:
                collect_stats.refresh_return_stats()
            except Exception:
                return float("-inf")
        rs = getattr(collect_stats, "returns_stat", None)
        return float(getattr(rs, "mean", float("-inf")))

    def _save_best_fn(_algorithm) -> None:
        save_ckpt(
            path=ckpt_dir / "best.ckpt",
            model=model,
            meta={
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": str(cfg.mode),
                "env_json": cfg.env_json,
                "grid": int(cfg.grid),
                "rot": int(cfg.rot),
                "soft_coefficient": float(cfg.soft_coefficient),
                "coarse_grid": int(cfg.coarse_grid),
                "alphachip_rot": int(cfg.alphachip_rot),
                "state_uint8": bool(cfg.state_uint8),
            },
        )

    def _save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        save_ckpt(
            path=ckpt_dir / "latest.ckpt",
            model=model,
            meta={
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": int(epoch),
                "env_step": int(env_step),
                "gradient_step": int(gradient_step),
                "mode": str(cfg.mode),
                "env_json": cfg.env_json,
                "grid": int(cfg.grid),
                "rot": int(cfg.rot),
                "soft_coefficient": float(cfg.soft_coefficient),
                "coarse_grid": int(cfg.coarse_grid),
                "alphachip_rot": int(cfg.alphachip_rot),
                "state_uint8": bool(cfg.state_uint8),
            },
        )
        return str((ckpt_dir / "latest.ckpt").resolve())

    params = OnPolicyTrainerParams(
        training_collector=train_collector,
        test_collector=test_collector,
        max_epochs=int(cfg.epoch),
        epoch_num_steps=int(cfg.step_per_epoch),
        collection_step_num_env_steps=int(cfg.step_per_collect),
        update_step_num_repetitions=int(cfg.repeat_per_collect),
        batch_size=int(cfg.batch_size),
        test_step_num_episodes=1,
        compute_score_fn=_compute_score_fn,
        save_best_fn=_save_best_fn,
        save_checkpoint_fn=_save_checkpoint_fn,
        verbose=True,
        show_progress=True,
    )
    trainer = OnPolicyTrainer(algorithm=algo, params=params)
    stats = trainer.run()
    logger.info("Training finished: %s", stats)
    logger.info("checkpoint_dir=%s", ckpt_dir)


if __name__ == "__main__":
    main()
