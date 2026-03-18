# coding=utf-8
# Copyright 2026.
#
# TorchRL PPO training (MaskPlace) for factory-layout.
#
# Why this exists:
# - Tianshou's default Collector/Buffer path often forces numpy conversions.
# - TorchRL can keep env rollout + buffer + PPO update in torch tensors (GPU-friendly).
#
# Assumptions:
# - You will install `torchrl` and `tensordict` later.
# - `envs/wrappers/maskplace.py:MaskPlaceAdapter` already returns torch.Tensor obs (incl. 'state')
#   even on terminal/truncated (stable obs schema).
#
# Usage (example):
#   conda run -n factory python train_new.py --env-json envs/env_configs/placed_01.json --device cuda
#

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import uuid
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import time

from agents.placement.maskplace import MaskPlaceModel, MaskPlaceAdapter
from envs.env_loader import load_env

logger = logging.getLogger(__name__)


def _require_torchrl() -> None:
    """Fail fast with a clear message if torchrl/tensordict aren't installed yet."""
    try:
        import tensordict  # noqa: F401
        import torchrl  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TorchRL training requires `torchrl` and `tensordict`.\n"
            "Install them in your env, e.g.:\n"
            "  pip install torchrl tensordict\n"
            "Then re-run this script."
        ) from e


@dataclass(frozen=True)
class Cfg:
    env_json: str
    collision_check: str
    grid: int
    rot: int
    soft_coefficient: float
    device: str
    load_ckpt: Optional[str]
    load_strict: bool

    # PPO / rollout
    total_frames: int
    frames_per_batch: int
    mini_batch_size: int
    ppo_epochs: int

    gamma: float
    gae_lambda: float
    clip_epsilon: float
    vf_coef: float
    ent_coef: float
    lr: float
    max_grad_norm: float


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--env-json", type=str, default="envs/env_configs/basic_01.json")
    p.add_argument("--collision-check", type=str, default="auto", choices=["auto", "conv", "prefixsum"])
    p.add_argument("--grid", type=int, default=224)
    p.add_argument("--rot", type=int, default=0)
    p.add_argument("--soft-coefficient", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--load-ckpt", type=str, default=None, help="Optional path to a pretrained ckpt (expects {'model': state_dict} or a raw state_dict).")
    p.add_argument("--load-strict", action="store_true", default=False, help="Use strict=True when loading state_dict (default: False).")

    # rollout/training
    p.add_argument("--total-frames", type=int, default=200000)
    p.add_argument("--frames-per-batch", type=int, default= 1024)
    p.add_argument("--mini-batch-size", type=int, default= 32)
    p.add_argument("--ppo-epochs", type=int, default= 8)

    # PPO hyperparams
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    a = p.parse_args()
    return Cfg(
        env_json=str(a.env_json),
        collision_check=str(a.collision_check),
        grid=int(a.grid),
        rot=int(a.rot),
        soft_coefficient=float(a.soft_coefficient),
        device=str(a.device),
        load_ckpt=str(a.load_ckpt) if a.load_ckpt else None,
        load_strict=bool(a.load_strict),
        total_frames=int(a.total_frames),
        frames_per_batch=int(a.frames_per_batch),
        mini_batch_size=int(a.mini_batch_size),
        ppo_epochs=int(a.ppo_epochs),
        gamma=float(a.gamma),
        gae_lambda=float(a.gae_lambda),
        clip_epsilon=float(a.clip_epsilon),
        vf_coef=float(a.vf_coef),
        ent_coef=float(a.ent_coef),
        lr=float(a.lr),
        max_grad_norm=float(a.max_grad_norm),
    )


def _make_ckpt_dir() -> Tuple[str, Path]:
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4().hex[:6]}"
    ckpt_dir = (Path("results") / "checkpoints" / run_id).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_id, ckpt_dir


class _MaskPlaceTorchRLEnv:  # EnvBase subclass defined lazily after torchrl import
    pass


def build_torchrl_env(*, cfg: Cfg, device: torch.device):
    """Build a TorchRL EnvBase that wraps MaskPlaceAdapter and emits TensorDicts on `device`."""
    _require_torchrl()
    from tensordict import TensorDict
    from torchrl.envs import EnvBase
    from torchrl.data import (
        CompositeSpec,
        DiscreteTensorSpec,
        UnboundedContinuousTensorSpec,
        UnboundedDiscreteTensorSpec,
    )

    # We load the engine on the requested device (GPU-friendly).
    loaded0 = load_env(cfg.env_json, device=device, collision_check=cfg.collision_check)
    reset_kwargs = loaded0.reset_kwargs

    # Infer state dim deterministically from grid.
    g = int(cfg.grid)
    state_dim = 1 + 5 * (g * g) + 2
    action_n = g * g

    class MaskPlaceTorchRLEnv(EnvBase):
        def __init__(self):
            super().__init__(device=device)
            loaded = load_env(cfg.env_json, device=device, collision_check=cfg.collision_check)
            engine = loaded.env
            engine.log = False
            self._gym = MaskPlaceAdapter(
                engine=engine,
                grid=int(cfg.grid),
                soft_coefficient=float(cfg.soft_coefficient),
            )
            self._reset_kwargs = reset_kwargs

            self.action_spec = DiscreteTensorSpec(action_n, shape=(), dtype=torch.long, device=device)
            self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=device)
            # TorchRL expects "done" (bool) and usually "terminated". We'll provide both.
            self.done_spec = CompositeSpec(
                done=UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool, device=device),
                terminated=UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool, device=device),
            )
            self.observation_spec = CompositeSpec(
                state=UnboundedContinuousTensorSpec(shape=(state_dim,), dtype=torch.float32, device=device),
                # keep action_mask available if you want to add masking later
                action_mask=UnboundedDiscreteTensorSpec(shape=(action_n,), dtype=torch.bool, device=device),
            )

        def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
            obs, _info = self._gym.reset(options=self._reset_kwargs)
            st = obs["state"].to(device=device, dtype=torch.float32).view(-1)
            am = obs["action_mask"].to(device=device, dtype=torch.bool).view(-1)
            td = TensorDict(
                {
                    "state": st,
                    "action_mask": am,
                    "done": torch.zeros((1,), device=device, dtype=torch.bool),
                    "terminated": torch.zeros((1,), device=device, dtype=torch.bool),
                },
                batch_size=[],
                device=device,
            )
            return td

        def _step(self, tensordict: TensorDict) -> TensorDict:
            a = int(tensordict.get("action").item())
            obs, reward, terminated, truncated, _info = self._gym.step(a)
            done = bool(terminated) or bool(truncated)
            st = obs["state"].to(device=device, dtype=torch.float32).view(-1)
            am = obs["action_mask"].to(device=device, dtype=torch.bool).view(-1)
            next_td = TensorDict(
                {
                    "state": st,
                    "action_mask": am,
                    "reward": torch.tensor([float(reward)], device=device, dtype=torch.float32),
                    "done": torch.tensor([done], device=device, dtype=torch.bool),
                    "terminated": torch.tensor([bool(terminated)], device=device, dtype=torch.bool),
                },
                batch_size=[],
                device=device,
            )
            return next_td
        def _set_seed(self, seed: int | None):
            # 시드 설정 로직 (보통 None이 아니면 시드를 고정)
            if seed is not None:
                torch.manual_seed(seed)

    return MaskPlaceTorchRLEnv()


class MaskPlaceActorNet(nn.Module):
    """TorchRL actor net wrapper: state -> logits."""

    def __init__(self, model: MaskPlaceModel):
        super().__init__()
        self.model = model

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TorchRL may provide `state` without a batch dim (shape [S]).
        # MaskPlaceModel expects [B,S], so we add a batch dim temporarily and squeeze outputs back.
        squeeze0 = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze0 = True

        # model.actor returns probs (already softmaxed). Convert to logits for TorchRL Categorical.
        probs, _value = self.model(state)
        eps = 1.0e-8
        logits = torch.log(torch.clamp(probs, min=eps))
        return logits.squeeze(0) if squeeze0 else logits


class MaskPlaceValueNet(nn.Module):
    """TorchRL critic wrapper: state -> value."""

    def __init__(self, model: MaskPlaceModel):
        super().__init__()
        self.model = model

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TorchRL may provide `state` without a batch dim (shape [S]).
        squeeze0 = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze0 = True

        # MaskPlaceModel.forward returns (probs, value). Call once, discard probs.
        _probs, value = self.model(state)
        return value.squeeze(0) if squeeze0 else value


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    _require_torchrl()
    cfg = parse_args()
    device = torch.device(cfg.device)

    from tensordict.nn import TensorDictModule
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import StepCounter
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torch.distributions import Categorical
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    run_id, ckpt_dir = _make_ckpt_dir()

    # --- env (single env) ---
    base_env = build_torchrl_env(cfg=cfg, device=device)
    env = TransformedEnv(base_env, StepCounter())

    # --- model ---
    model = MaskPlaceModel(grid=int(cfg.grid), device=device, soft_coefficient=float(cfg.soft_coefficient)).to(device)
    if cfg.load_ckpt:
        payload = torch.load(str(cfg.load_ckpt), map_location=device)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        msg = model.load_state_dict(state_dict, strict=bool(cfg.load_strict))
        # When strict=False, report missing/unexpected keys to help debugging.
        try:
            missing = list(getattr(msg, "missing_keys", []))
            unexpected = list(getattr(msg, "unexpected_keys", []))
            if (not cfg.load_strict) and (missing or unexpected):
                logger.warning(
                    "load_ckpt strict=False missing_keys=%s unexpected_keys=%s",
                    missing,
                    unexpected,
                )
        except Exception:
            pass
        logger.info("load_ckpt loaded: %s", cfg.load_ckpt)
    # IMPORTANT: during rollout/collection, keep the model in eval mode to avoid BatchNorm
    # issues with batched execution in some TorchRL collector paths.
    model.eval()

    actor_module = TensorDictModule(
        module=MaskPlaceActorNet(model),
        in_keys=["state"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_module = TensorDictModule(
        module=MaskPlaceValueNet(model),
        in_keys=["state"],
        out_keys=["state_value"],
    )
    critic = ValueOperator(value_module, in_keys=["state"], out_keys=["state_value"])

    # --- collector ---
    # Create the collector under no_grad + eval, since some TorchRL versions run
    # a policy forward pass during collector init.
    with torch.no_grad():
        collector = SyncDataCollector(
            env,
            policy=actor,
            frames_per_batch=int(cfg.frames_per_batch),
            total_frames=int(cfg.total_frames),
            device=device,
            storing_device=device,
        )

    # --- buffer (on-policy: we keep last batch; still use RB API for minibatch sampling) ---
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(int(cfg.frames_per_batch), device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=int(cfg.mini_batch_size),
    )

    advantage = GAE(
        gamma=float(cfg.gamma),
        lmbda=float(cfg.gae_lambda),
        value_network=critic,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=float(cfg.clip_epsilon),
        entropy_coef=float(cfg.ent_coef),
        critic_coef=float(cfg.vf_coef),
        normalize_advantage=True,
    )

    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    best_score = float("-inf")
    total_collected = 0

    it = iter(collector)
    i = 0

    # Progress bar (tqdm is assumed to be installed per user request).
    from tqdm.auto import tqdm  # type: ignore

    pbar = tqdm(total=int(cfg.total_frames), desc="TorchRL PPO", unit="frames")

    t_start = time.perf_counter()
    while True:
        # --- rollout / collection (no grads, eval mode) ---
        model.eval()
        actor.eval()
        critic.eval()
        with torch.no_grad():
            batch = next(it)
            # batch is a tensordict with "next" already populated by EnvBase.step convention.
            batch = advantage(batch)

        rb.extend(batch.reshape(-1))

        # --- PPO update (enable grads, train mode) ---
        model.train()
        actor.train()
        critic.train()
        for _epoch in range(int(cfg.ppo_epochs)):
            for sub in rb:
                loss_td = loss_module(sub)
                loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if float(cfg.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))
                optim.step()

        # IMPORTANT: frame count is the rollout time dimension, NOT TensorDict.numel().
        # Using numel() would count scalar elements across all keys and break stopping/progress/fps.
        try:
            batch_frames = int(batch.batch_size[0])
        except Exception:
            batch_frames = int(batch.get(("next", "reward")).numel())
        total_collected += batch_frames

        # quick score proxy: mean episode return in this batch (if any completed)
        # TorchRL's StepCounter adds "step_count"; for simplicity we log mean reward.
        mean_r = float(batch.get(("next", "reward")).mean().item())
        elapsed = max(1.0e-9, time.perf_counter() - t_start)
        fps = float(total_collected) / elapsed
        # Avoid going over total.
        inc = min(batch_frames, int(cfg.total_frames) - int(pbar.n))
        if inc > 0:
            pbar.update(inc)
        pbar.set_postfix({"iter": i, "mean_r": f"{mean_r:.4f}", "fps": f"{fps:.1f}"})

        # save latest
        latest = ckpt_dir / "latest.ckpt"
        torch.save(
            {
                "model": model.state_dict(),
                "meta": {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "env_json": cfg.env_json,
                    "grid": int(cfg.grid),
                    "rot": int(cfg.rot),
                    "soft_coefficient": float(cfg.soft_coefficient),
                    "device": str(cfg.device),
                    "frames": int(total_collected),
                    "mean_reward": float(mean_r),
                },
            },
            str(latest),
        )

        if mean_r > best_score:
            best_score = mean_r
            best = ckpt_dir / "best.ckpt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "env_json": cfg.env_json,
                        "grid": int(cfg.grid),
                        "rot": int(cfg.rot),
                        "soft_coefficient": float(cfg.soft_coefficient),
                        "device": str(cfg.device),
                        "frames": int(total_collected),
                        "mean_reward": float(mean_r),
                    },
                },
                str(best),
            )
            logger.info("best updated best_mean_reward=%.6f", best_score)

        rb.empty()

        if total_collected >= int(cfg.total_frames):
            break

        i += 1

    logger.info("Training finished.")
    logger.info("checkpoint_dir=%s", ckpt_dir)
    pbar.close()


if __name__ == "__main__":
    main()
