"""Dynamic Storage Wrapper for inference pipeline.

DynamicStorageEnv를 GreedyWrapperV3Env와 유사한 인터페이스로 감싸서
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


class DynamicStorageWrapper(gym.Env):
    """DynamicStorageEnv용 Top-K Wrapper.
    
    GreedyWrapperV3Env와 유사한 인터페이스:
    - Action space: Discrete(k)
    - Observation: {"action_mask": [k], "action_xyrot": [k, 3]}
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
        self.action_xyrot: Optional[torch.Tensor] = None  # [k, 3] (x, y, rot)
        self.mask: Optional[torch.Tensor] = None  # [k] bool
        
        # inference.py 호환용
        self.engine = dynamic_env.base_env
    
    @property
    def device(self) -> torch.device:
        return self.dynamic_env.device
    
    # ========== Candidate Generation ==========
    
    def create_mask(self) -> torch.Tensor:
        """유효한 action에서 top-k 후보 생성 (cost 기반 정렬)."""
        # 모든 유효 action 수집
        full_mask = self.dynamic_env.get_action_mask()
        valid_indices = torch.where(full_mask)[0]
        
        if valid_indices.numel() == 0:
            # 유효한 action 없음
            self.action_xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
            return torch.zeros(self.k, dtype=torch.bool, device=self.device)
        
        # 각 유효 action의 world 좌표 수집
        candidates: List[Tuple[int, int, int, int]] = []  # (action_idx, world_x, world_y, rot)
        
        for idx in valid_indices.tolist():
            gx, gy, rot = self.dynamic_env.decode_action(int(idx))
            wx, wy = self.dynamic_env.grid_to_world(gx, gy)
            candidates.append((int(idx), wx, wy, rot))
        
        # cost 계산 (rot별로 분리)
        candidates_by_rot: Dict[int, List[Tuple[int, int, int]]] = {}
        for action_idx, wx, wy, rot in candidates:
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
        
        self.action_xyrot = xyrot
        return mask
    
    # ========== Observation ==========
    
    def _build_obs(self) -> Dict[str, Any]:
        """Observation 빌드."""
        assert self.mask is not None
        assert self.action_xyrot is not None
        
        # base obs (action_mask 제외)
        base_obs = self.dynamic_env._build_obs()
        base_obs.pop("action_mask", None)  # wrapper의 mask 사용
        
        obs = {
            "action_mask": self.mask,
            "action_xyrot": self.action_xyrot,
        }
        obs.update(base_obs)
        return obs
    
    # ========== Action Decode ==========
    
    def decode_action(self, action: int) -> Tuple[float, float, int, int, int]:
        """action (0~k-1) → (x_bl, y_bl, rot, 0, action).
        
        Returns:
            (x_bl, y_bl, rot, 0, cand_idx)
        """
        a = int(action)
        if self.action_xyrot is None or a < 0 or a >= self.k:
            return 0.0, 0.0, 0, 0, 0
        
        xyz = self.action_xyrot[a]
        return float(xyz[0].item()), float(xyz[1].item()), int(xyz[2].item()), 0, a
    
    # ========== Gym API ==========
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """환경 리셋."""
        obs, info = self.dynamic_env.reset(seed=seed, options=options)
        self.mask = self.create_mask()
        return self._build_obs(), info
    
    def step(self, action: int):
        """Step 실행.
        
        action은 0 ~ k-1 인덱스.
        """
        assert self.mask is not None
        
        x, y, rot, _, cand_idx = self.decode_action(int(action))
        
        # 유효하지 않은 action 체크
        if cand_idx >= self.k or not self.mask[cand_idx]:
            return self._build_obs(), -1.0, False, False, {"reason": "invalid_action"}
        
        # dynamic_env의 action으로 변환
        gx = int(x) // self.dynamic_env.stride_x
        gy = int(y) // self.dynamic_env.stride_y
        dynamic_action = self.dynamic_env.encode_action(gx, gy, rot)
        
        # step 실행
        obs, reward, terminated, truncated, info = self.dynamic_env.step(dynamic_action)
        
        # 새 mask 생성 (terminated 아닌 경우)
        if not (terminated or truncated):
            self.mask = self.create_mask()
            return self._build_obs(), reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info
    
    # ========== Snapshot (MCTS 호환) ==========
    
    def get_snapshot(self) -> Dict[str, Any]:
        """현재 상태 저장."""
        snap = self.dynamic_env.get_snapshot()
        snap["wrapper_rng_state"] = self._rng.getstate()
        if self.action_xyrot is not None:
            snap["action_xyrot"] = self.action_xyrot.clone()
        else:
            snap["action_xyrot"] = None
        if self.mask is not None:
            snap["mask"] = self.mask.clone()
        else:
            snap["mask"] = None
        return snap
    
    def set_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """상태 복원."""
        self.dynamic_env.set_snapshot(snapshot)
        
        rs = snapshot.get("wrapper_rng_state", None)
        if rs is not None:
            try:
                self._rng.setstate(rs)
            except Exception:
                pass
        
        ax = snapshot.get("action_xyrot", None)
        if isinstance(ax, torch.Tensor):
            self.action_xyrot = ax.to(device=self.device, dtype=torch.long).clone()
        else:
            self.action_xyrot = None
        
        m = snapshot.get("mask", None)
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
