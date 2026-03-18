"""Dynamic Storage Wrapper for inference pipeline.

DynamicStorageEnv를 GreedyV3Adapter와 유사한 인터페이스로 감싸서
기존 inference 파이프라인 (Agent, MCTS 등)과 호환되도록 합니다.

사용법:
    from postprocess.dynamic_env import DynamicStorageEnv, DynamicGroupConfig
    from postprocess.dynamic_wrapper import DynamicStorageWrapper
    
    dynamic_env = DynamicStorageEnv(base_env, configs, group_flow)
    wrapper = DynamicStorageWrapper(dynamic_env, k=50)
    
    obs, info = wrapper.reset()
    action, = ...  # 0 ~ k-1
    obs, reward, term, trunc, info = wrapper.step(action)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch

from .dynamic_env import DynamicStorageEnv, PlacementResult
from envs.action_space import ActionSpace as CandidateSet


class DynamicStorageWrapper(gym.Env):
    """DynamicStorageEnv용 Top-K Wrapper.
    
    GreedyV3Adapter와 유사한 인터페이스:
    - Action space: Discrete(k)
    - Observation: dynamic env base observation (no action-space fields)
    - cost 기반으로 top-k 후보 선택
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        dynamic_env: DynamicStorageEnv,
        k: int = 50,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            dynamic_env: DynamicStorageEnv 인스턴스
            k: top-k 후보 개수
            random_seed: 랜덤 시드
        """
        super().__init__()
        self.dynamic_env = dynamic_env
        self.k = k
        self._rng = random.Random(random_seed)
        
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Dict({})
        
        # 현재 후보들
        self.action_poses: Optional[torch.Tensor] = None  # [k, 3] (x, y, rot)
        self.mask: Optional[torch.Tensor] = None  # [k] bool
        
        # inference.py 호환용
        self.engine = dynamic_env.base_env
    
    @property
    def device(self) -> torch.device:
        return self.dynamic_env.device

    def current_gid(self) -> Optional[str]:
        """현재 배치 대상 그룹 gid."""
        if self.dynamic_env.config is None:
            return None
        return str(self.dynamic_env.config.gid)
    
    # ========== Action-Space Generation ==========
    
    def create_mask(self) -> torch.Tensor:
        """유효한 action에서 top-k 후보 생성 (cost 기반 정렬)."""
        # 모든 유효 action 수집
        full_mask = self.dynamic_env.get_action_mask()
        valid_indices = torch.where(full_mask)[0]
        
        if valid_indices.numel() == 0:
            # 유효한 action 없음
            self.action_poses = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
            return torch.zeros(self.k, dtype=torch.bool, device=self.device)
        
        # 각 유효 action의 world 좌표 수집
        action_space: List[Tuple[int, int, int, int]] = []  # (action_idx, world_x, world_y, rot)
        
        for idx in valid_indices.tolist():
            gx, gy, rot = self.dynamic_env.decode_action(int(idx))
            wx, wy = self.dynamic_env.grid_to_world(gx, gy)
            action_space.append((int(idx), wx, wy, rot))
        
        # cost 계산 (rot별로 분리)
        candidates_by_rot: Dict[int, List[Tuple[int, int, int]]] = {}
        for action_idx, wx, wy, rot in action_space:
            if rot not in candidates_by_rot:
                candidates_by_rot[rot] = []
            candidates_by_rot[rot].append((action_idx, wx, wy))
        
        # rot별 cost 계산 후 합치기
        scored: List[Tuple[float, int, int, int, int]] = []  # (cost, action_idx, x, y, rot)
        
        for rot, cands in candidates_by_rot.items():
            if not cands:
                continue
            positions = [(wx, wy) for _, wx, wy in cands]
            costs = self.dynamic_env._get_cost_batch(positions, rot)
            
            for i, (action_idx, wx, wy) in enumerate(cands):
                cost = float(costs[i].item())
                scored.append((cost, action_idx, wx, wy, rot))
        
        # cost 낮은 순 정렬
        scored.sort(key=lambda x: x[0])
        
        # top-k 선택
        selected = scored[:self.k]
        
        # 결과 텐서 생성
        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
        mask = torch.zeros(self.k, dtype=torch.bool, device=self.device)
        
        for i, (cost, action_idx, wx, wy, rot) in enumerate(selected):
            xyrot[i, 0] = wx
            xyrot[i, 1] = wy
            xyrot[i, 2] = rot
            mask[i] = True
        
        self.action_poses = xyrot
        return mask
    
    # ========== Observation ==========
    
    def build_action_space(self) -> CandidateSet:
        """Build action-space from current dynamic env state."""
        self.mask = self.create_mask()
        if not isinstance(self.mask, torch.Tensor):
            raise ValueError("create_mask() must return torch.Tensor")
        if not isinstance(self.action_poses, torch.Tensor):
            raise ValueError("action_poses must be torch.Tensor after create_mask()")
        mask_t = self.mask.to(dtype=torch.bool, device=self.device).view(-1)
        xyrot_t = self.action_poses.to(dtype=torch.long, device=self.device)
        if xyrot_t.ndim != 2 or int(xyrot_t.shape[1]) != 3:
            raise ValueError(f"action_poses must have shape [N,3], got {tuple(xyrot_t.shape)}")
        if int(xyrot_t.shape[0]) != int(mask_t.shape[0]):
            raise ValueError(
                f"action-space size mismatch: poses={int(xyrot_t.shape[0])}, mask={int(mask_t.shape[0])}"
            )
        return CandidateSet(poses=xyrot_t, mask=mask_t, gid=self.current_gid())
    
    # ========== Action Decode ==========
    
    def decode_action(
        self,
        action: int,
        action_space: Optional[CandidateSet] = None,
    ) -> Tuple[float, float, int, int, int]:
        """action (0~k-1) → (x_bl, y_bl, rot, 0, action).
        
        Returns:
            (x_bl, y_bl, rot, 0, action_idx)
        """
        del action_space
        a = int(action)
        if self.action_poses is None or a < 0 or a >= self.k:
            return 0.0, 0.0, 0, 0, 0
        
        xyz = self.action_poses[a]
        return float(xyz[0].item()), float(xyz[1].item()), int(xyz[2].item()), 0, a
    
    # ========== Gym API ==========
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """환경 리셋."""
        _obs, info = self.dynamic_env.reset(seed=seed, options=options)
        self.action_poses = None
        self.mask = None
        return self.build_observation(), info

    def build_observation(self) -> Dict[str, Any]:
        """Build policy observation only (no action-space fields)."""
        base_obs = dict(self.dynamic_env._build_obs())
        base_obs.pop("action_mask", None)
        base_obs.pop("action_poses", None)
        return base_obs
    
    def step(self, action: int):
        """Step 실행.
        
        action은 0 ~ k-1 인덱스.
        """
        x, y, rot, _, action_idx = self.decode_action(int(action))
        
        # 유효하지 않은 action 체크
        if self.mask is None or action_idx >= self.k or not bool(self.mask[action_idx].item()):
            return self.build_observation(), -1.0, False, False, {"reason": "invalid_action"}
        
        # dynamic_env의 action으로 변환
        gx = int(x) // self.dynamic_env.stride_x
        gy = int(y) // self.dynamic_env.stride_y
        dynamic_action = self.dynamic_env.encode_action(gx, gy, rot)
        
        # step 실행
        _obs, reward, terminated, truncated, info = self.dynamic_env.step(dynamic_action)

        return self.build_observation(), reward, terminated, truncated, info
    
    # ========== State Copy (MCTS 호환) ==========
    
    def get_state_copy(self) -> Dict[str, Any]:
        """현재 상태 저장."""
        snap = self.dynamic_env.get_state_copy()
        snap["wrapper_rng_state"] = self._rng.getstate()
        if self.action_poses is not None:
            snap["action_poses"] = self.action_poses.clone()
        else:
            snap["action_poses"] = None
        if self.mask is not None:
            snap["mask"] = self.mask.clone()
        else:
            snap["mask"] = None
        return snap
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """상태 복원."""
        self.dynamic_env.set_state(state)
        
        rs = state.get("wrapper_rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        
        ax = state.get("action_poses", None)
        if isinstance(ax, torch.Tensor):
            self.action_poses = ax.to(device=self.device, dtype=torch.long).clone()
        else:
            self.action_poses = None
        
        m = state.get("mask", None)
        if isinstance(m, torch.Tensor):
            self.mask = m.to(device=self.device, dtype=torch.bool).clone()
        else:
            self.mask = None
    
    # ========== Visualization 호환 ==========
    
    @property
    def current_result(self) -> Optional[PlacementResult]:
        """현재 배치 결과."""
        return self.dynamic_env.current_result
    
    def render(self, **kwargs):
        """시각화."""
        return self.dynamic_env.render(**kwargs)
