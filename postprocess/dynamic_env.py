"""Dynamic Storage Environment.

시작점에서 stride로 확장하며 셀을 배치하는 환경.
conv2d로 placeable mask를 계산하고, cost 기반 greedy로 확장합니다.

Action Space: Discrete(grid_h * grid_w * n_rots)
- action → (grid_x, grid_y, rot)
- grid_x, grid_y: stride grid 좌표
- rot: 0 또는 90

사용법:
    env = DynamicStorageEnv(base_env, configs, group_flow)
    obs = env.reset()
    action = ...  # encode_action(grid_x, grid_y, rot)
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId


@dataclass
class DynamicGroupConfig:
    """동적 그룹 설정 (하나의 그룹)."""
    gid: str                   # 그룹 ID
    total_cells: int           # 목표 총 셀 수 (3D 전체, 스택 포함)
    cell_width: float          # 셀 너비 (grid 단위)
    cell_depth: float          # 셀 깊이 (grid 단위, 2D에서의 height)
    clearance_w: float = 0     # block width 방향 여백
    clearance_h: float = 0     # block height(depth) 방향 여백
    gap_w: float = 0           # cell width 방향 이격
    gap_h: float = 0           # cell height(depth) 방향 이격
    rotatable: bool = False    # 90도 회전 허용
    # z축 높이 (m 단위)
    cell_z_height: float = 1.0   # 셀 1개 z축 높이
    z_overhead: float = 0.0      # 높이 여유 (다리 + 상단)
    # step당 최대 unit 수 (0이면 제한 없음)
    max_units_per_step: int = 0


@dataclass
class PlacementResult:
    """배치 결과."""
    cells: Set[Tuple[int, int]] = field(default_factory=set)
    unit_cells: Set[Tuple[int, int]] = field(default_factory=set)
    unit_positions: List[Tuple[int, int, int]] = field(default_factory=list)
    unit_stacks: List[int] = field(default_factory=list)
    num_units: int = 0
    total_cells: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    def to_mask(self, H: int, W: int) -> torch.Tensor:
        """전체 영역 mask."""
        mask = torch.zeros((H, W), dtype=torch.bool)
        for x, y in self.cells:
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = True
        return mask


class DynamicStorageEnv(gym.Env):
    """시작점 기반 동적 Storage 배치 환경.
    
    Action Space: Discrete(grid_h * grid_w * n_rots)
    - action을 decode하면 (grid_x, grid_y, rot)
    - grid 좌표 → world 좌표: x = grid_x * stride_x, y = grid_y * stride_y
    
    conv2d로 placeable mask를 계산하고,
    시작점에서 stride로 확장하며 cost 낮은 곳부터 greedy하게 셀을 채웁니다.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        base_env: FactoryLayoutEnv,
        configs: List[DynamicGroupConfig],
        group_flow: Dict[str, Dict[str, float]],
        *,
        reward_cell_size: int = 50,
        empty_threshold: float = 0.1,
    ):
        """
        Args:
            base_env: 기존 FactoryLayoutEnv (constraint 검사용)
            configs: 동적 그룹 설정 리스트
            group_flow: 그룹 간 flow 정보 {src_gid: {dst_gid: weight}}
            reward_cell_size: reward 계산용 셀 크기 (grid 단위)
            empty_threshold: 이 비율 이하면 "비어있음"으로 판정
        """
        super().__init__()
        self.base_env = base_env
        self.configs = configs
        self.group_flow = group_flow
        
        # 현재 배치 중인 그룹 (첫 번째로 시작)
        self._current_config_idx = 0
        self.config = configs[0] if configs else None
        self.gid = self.config.gid if self.config else None
        self.allow_rotation = self.config.rotatable if self.config else False
        
        # Reward 관련 (env 레벨)
        self.reward_cell_size = reward_cell_size
        self.empty_threshold = empty_threshold
        
        # config에서 값 추출 (rot=0 기준)
        self.cell_w = int(self.config.cell_width) if self.config else 10
        self.cell_h = int(self.config.cell_depth) if self.config else 10
        self.clearance_w = int(self.config.clearance_w) if self.config else 0
        self.clearance_h = int(self.config.clearance_h) if self.config else 0
        self.gap_w = int(self.config.gap_w) if self.config else 0
        self.gap_h = int(self.config.gap_h) if self.config else 0
        
        # block = cell + clearance (rot=0 기준)
        self.block_w = self.cell_w + 2 * self.clearance_w
        self.block_h = self.cell_h + 2 * self.clearance_h
        
        # stride = cell + gap (rot=0 기준, rot별 stride는 _get_stride 사용)
        self.stride_x = self.cell_w + self.gap_w
        self.stride_y = self.cell_h + self.gap_h
        
        # z축 높이 맵 (천장 높이)
        self._height_map = base_env._height_map  # [H, W] 천장 높이
        
        self.H = base_env.grid_height
        self.W = base_env.grid_width
        self.device = base_env.device
        
        # 유효 영역 (배치 가능)
        self._valid = ~(base_env._occ_invalid | base_env._static_invalid)
        self._initial_valid = self._valid.clone()
        
        # 배치된 unit 영역 추적
        self._unit_used = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        
        # Rotation 수
        self._n_rots = 2 if self.allow_rotation and self.cell_w != self.cell_h else 1
        
        # Placeable masks (stride=1로 계산) - {rot: [H-bh+1, W-bw+1] bool tensor}
        self._placeable_masks: Dict[int, torch.Tensor] = {}
        self._kernel_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._placeable_version: int = 0
        self._action_mask_cache: Optional[torch.Tensor] = None
        self._action_mask_cache_version: int = -1
        self._stack_grids: Dict[int, torch.Tensor] = {}
        # 배치 히스토리 (시각화/리포트용): gid -> [(ux, uy, rot, stack), ...]
        self.placed_history: Dict[str, List[Tuple[int, int, int, int]]] = {}
        # 현재 그룹에서 배치된 cell 수 (남은 cell 추적용)
        self._placed_cells = 0
        self._update_placeable_masks()
        
        # Action space용 grid 크기 (stride 기준)
        self._grid_h = max(1, (self.H - self.block_h) // self.stride_y + 1)
        self._grid_w = max(1, (self.W - self.block_w) // self.stride_x + 1)
        
        # Action space
        self.action_space = gym.spaces.Discrete(self._grid_h * self._grid_w * self._n_rots)
        self.observation_space = gym.spaces.Dict({})
        
        # 현재 배치 결과
        self.current_result: Optional[PlacementResult] = None
        
        # 남은 그룹 (wrapper 호환용)
        self._remaining_configs = list(configs)

        # stack grid (height 기반)는 config/rot에 대해 캐시
        self._build_stack_grids()
    
    # ========== Properties (Wrapper 호환) ==========
    
    @property
    def remaining(self) -> List[str]:
        """남은 그룹 ID 리스트 (wrapper 호환)."""
        return [c.gid for c in self._remaining_configs]
    
    @property
    def placed(self) -> Set[str]:
        """배치 완료된 그룹 ID set."""
        all_gids = {c.gid for c in self.configs}
        remaining_gids = {c.gid for c in self._remaining_configs}
        return all_gids - remaining_gids
    
    # ========== Size/Clearance by rotation ==========
    
    def _get_block_size(self, rot: int) -> Tuple[int, int]:
        """rotation에 따른 block 크기 (width, height)."""
        if rot == 0:
            return self.block_w, self.block_h
        else:
            return self.block_h, self.block_w
    
    def _get_cell_size(self, rot: int) -> Tuple[int, int]:
        """rotation에 따른 cell 크기 (width, height)."""
        if rot == 0:
            return self.cell_w, self.cell_h
        else:
            return self.cell_h, self.cell_w
    
    def _get_clearance(self, rot: int) -> Tuple[int, int]:
        """rotation에 따른 clearance (w, h)."""
        if rot == 0:
            return self.clearance_w, self.clearance_h
        else:
            return self.clearance_h, self.clearance_w
    
    def _get_gap(self, rot: int) -> Tuple[int, int]:
        """rotation에 따른 gap (w, h)."""
        if rot == 0:
            return self.gap_w, self.gap_h
        else:
            return self.gap_h, self.gap_w
    
    def _get_stride(self, rot: int) -> Tuple[int, int]:
        """rotation에 따른 stride (w, h)."""
        cw, ch = self._get_cell_size(rot)
        gw, gh = self._get_gap(rot)
        return cw + gw, ch + gh
    
    # ========== conv2d로 Placeable Mask 계산 ==========
    
    def _update_placeable_masks(self):
        """conv2d stride=1로 각 rotation의 placeable mask 계산."""
        valid_float = self._valid.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        
        rotations = [0]
        if self._n_rots == 2:
            rotations.append(90)
        
        for rot in rotations:
            bw, bh = self._get_block_size(rot)
            
            if bh > self.H or bw > self.W:
                self._placeable_masks[rot] = torch.zeros(
                    (1, 1), dtype=torch.bool, device=self.device
                )
                continue
            
            key = (bh, bw)
            kernel = self._kernel_cache.get(key)
            if kernel is None or kernel.device != self.device:
                kernel = torch.ones((1, 1, bh, bw), device=self.device)
                self._kernel_cache[key] = kernel
            conv_result = F.conv2d(
                valid_float, kernel, padding=0, stride=1
            ).squeeze()
            
            if conv_result.dim() == 0:
                conv_result = conv_result.unsqueeze(0).unsqueeze(0)
            elif conv_result.dim() == 1:
                conv_result = conv_result.unsqueeze(0)
            
            placeable = (conv_result == bw * bh)
            self._placeable_masks[rot] = placeable

        # version bump + caches invalidation
        self._placeable_version += 1
        self._action_mask_cache = None
        self._action_mask_cache_version = -1
    
    # ========== 스택 계산 ==========
    
    def get_stack_count(self, x: int, y: int, rot: int) -> int:
        """해당 위치에서 쌓을 수 있는 스택 개수 계산."""
        if self.config is None:
            return 0
            
        cw, ch = self._get_cell_size(rot)
        cl_w, cl_h = self._get_clearance(rot)
        
        ux, uy = x + cl_w, y + cl_h
        
        if ux + cw > self.W or uy + ch > self.H:
            return 0
        
        min_ceiling = self._height_map[uy:uy+ch, ux:ux+cw].min().item()
        
        if math.isnan(min_ceiling):
            return self.config.total_cells
        
        if self.config.cell_z_height <= 0:
            return 1
        
        available = min_ceiling - self.config.z_overhead
        
        if math.isnan(available):
            return self.config.total_cells
        
        if available <= 0:
            return 0
        
        result = available / self.config.cell_z_height
        
        if math.isnan(result) or math.isinf(result):
            return self.config.total_cells
        
        return int(result)
    
    # ========== Placeable 조회 ==========
    
    def is_placeable(self, x: int, y: int, rot: int) -> bool:
        """(x, y) 위치에서 rot 방향으로 배치 가능한지 확인."""
        pm = self._placeable_masks.get(rot, None)
        if pm is None:
            return False
        
        max_y, max_x = pm.shape
        if x < 0 or y < 0 or x >= max_x or y >= max_y:
            return False
        
        return bool(pm[y, x].item())
    
    # ========== Action Encoding/Decoding ==========
    
    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """action index → (grid_x, grid_y, rot)."""
        grid_size = self._grid_h * self._grid_w
        rot_idx = action // grid_size
        remaining = action % grid_size
        grid_y = remaining // self._grid_w
        grid_x = remaining % self._grid_w
        rot = 90 if rot_idx == 1 and self._n_rots == 2 else 0
        return grid_x, grid_y, rot
    
    def encode_action(self, grid_x: int, grid_y: int, rot: int) -> int:
        """(grid_x, grid_y, rot) → action index."""
        rot_idx = 1 if rot == 90 and self._n_rots == 2 else 0
        grid_size = self._grid_h * self._grid_w
        return rot_idx * grid_size + grid_y * self._grid_w + grid_x
    
    def grid_to_world(self, grid_x: int, grid_y: int, rot: int = 0) -> Tuple[int, int]:
        """grid 좌표 → world 좌표 (block bottom-left)."""
        stride_w, stride_h = self._get_stride(rot)
        return grid_x * stride_w, grid_y * stride_h
    
    # ========== Action Mask ==========
    
    def get_action_mask(self) -> torch.Tensor:
        """유효한 action mask [action_space.n] 반환."""
        if self._action_mask_cache is not None and self._action_mask_cache_version == self._placeable_version:
            return self._action_mask_cache

        grid_size = self._grid_h * self._grid_w
        total_actions = grid_size * self._n_rots
        mask = torch.zeros(total_actions, dtype=torch.bool, device=self.device)

        # grid 좌표 인덱스
        gx = torch.arange(self._grid_w, device=self.device, dtype=torch.long)
        gy = torch.arange(self._grid_h, device=self.device, dtype=torch.long)

        for rot_idx in range(self._n_rots):
            rot = 90 if rot_idx == 1 else 0
            pm = self._placeable_masks.get(rot, None)
            if pm is None:
                continue

            # rot별 stride로 world 좌표 계산
            stride_w, stride_h = self._get_stride(rot)
            xs = gx * stride_w  # [grid_w]
            ys = gy * stride_h  # [grid_h]

            max_y, max_x = pm.shape
            valid_x = xs < max_x
            valid_y = ys < max_y
            if not bool(valid_x.any().item()) or not bool(valid_y.any().item()):
                continue

            xs_v = xs[valid_x]
            ys_v = ys[valid_y]
            # grid[gy,gx] = pm[ys, xs]
            grid_mask = torch.zeros((self._grid_h, self._grid_w), dtype=torch.bool, device=self.device)
            sub = pm[ys_v.view(-1, 1), xs_v.view(1, -1)]
            grid_mask[valid_y.nonzero(as_tuple=False).view(-1).long()[:, None],
                      valid_x.nonzero(as_tuple=False).view(-1).long()[None, :]] = sub

            flat = grid_mask.view(-1)
            start = rot_idx * grid_size
            mask[start:start + grid_size] = flat

        self._action_mask_cache = mask
        self._action_mask_cache_version = self._placeable_version
        return mask
    
    # ========== Reward 계산 ==========
    
    def _compute_occupancy_grid(self) -> torch.Tensor:
        """공장을 reward_cell_size로 나눠 점유율 계산."""
        cell_size = self.reward_cell_size
        grid_h = (self.H + cell_size - 1) // cell_size
        grid_w = (self.W + cell_size - 1) // cell_size
        
        occupancy = torch.zeros((grid_h, grid_w), dtype=torch.float, device=self.device)
        occupied = ~self._valid
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                y0 = gy * cell_size
                y1 = min((gy + 1) * cell_size, self.H)
                x0 = gx * cell_size
                x1 = min((gx + 1) * cell_size, self.W)
                
                cell_area = (y1 - y0) * (x1 - x0)
                if cell_area <= 0:
                    continue
                
                occupied_count = occupied[y0:y1, x0:x1].sum().item()
                occupancy[gy, gx] = occupied_count / cell_area
        
        return occupancy
    
    def _compute_empty_ratio(self) -> float:
        """비어있는 셀 비율 계산."""
        occupancy = self._compute_occupancy_grid()
        empty_cells = (occupancy < self.empty_threshold).sum().item()
        total_cells = occupancy.numel()
        return empty_cells / total_cells if total_cells > 0 else 0.0
    
    # ========== Cost Map ==========
    
    def _get_cost_at(self, x: int, y: int, rot: int) -> float:
        """해당 위치의 cost 계산 (단일)."""
        costs = self._get_cost_batch([(x, y)], rot)
        return float(costs[0].item()) if len(costs) > 0 else 0.0
    
    def _get_cost_batch(self, positions: List[Tuple[int, int]], rot: int) -> torch.Tensor:
        """여러 위치의 flow cost를 병렬 계산."""
        if not positions:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        N = len(positions)
        cw, ch = self._get_cell_size(rot)
        
        xs = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        ys = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        cand_cx = xs + cw / 2.0
        cand_cy = ys + ch / 2.0
        
        # base_env.placed는 set, positions는 dict
        placed_nodes = list(self.base_env.placed)
        if not placed_nodes:
            return torch.zeros(N, dtype=torch.float32, device=self.device)
        
        centers_p: List[Tuple[float, float]] = []
        w_out: List[float] = []
        w_in: List[float] = []
        
        for p in placed_nodes:
            # positions dict에서 좌표 가져오기
            placement = self.base_env.positions.get(p)
            if placement is None:
                continue
            px, py = placement[0], placement[1]
            g = self.base_env.groups[p]
            p_rot = placement[2] if len(placement) > 2 else 0
            if p_rot in (90, 270):
                pw, ph = g.height, g.width
            else:
                pw, ph = g.width, g.height
            pcx = px + pw / 2.0
            pcy = py + ph / 2.0
            centers_p.append((pcx, pcy))
            
            w_out.append(float(self.group_flow.get(self.gid, {}).get(p, 0.0)))
            w_in.append(float(self.group_flow.get(p, {}).get(self.gid, 0.0)))
        
        if not centers_p:
            return torch.zeros(N, dtype=torch.float32, device=self.device)
        
        centers_t = torch.tensor(centers_p, dtype=torch.float32, device=self.device)
        w_out_t = torch.tensor(w_out, dtype=torch.float32, device=self.device).view(1, -1)
        w_in_t = torch.tensor(w_in, dtype=torch.float32, device=self.device).view(1, -1)
        
        dist = (cand_cx.view(-1, 1) - centers_t[:, 0].view(1, -1)).abs() + \
               (cand_cy.view(-1, 1) - centers_t[:, 1].view(1, -1)).abs()
        
        cost = (dist * w_out_t).sum(dim=1) + (dist * w_in_t).sum(dim=1)
        return cost
    
    # ========== 확장 로직 (Fast BFS) ==========

    def _build_stack_grids(self) -> None:
        """height map 기반 stack count grid를 rot별로 캐시.

        - unit_used/valid와 무관 (천장 높이만 사용) → config가 바뀔 때만 재계산
        - NaN 높이는 inf로 간주 → stack은 total_cells로 캡
        - rot별로 다른 grid 크기/stride 사용
        """
        self._stack_grids = {}
        if self.config is None:
            return

        rotations = [0]
        if self._n_rots == 2:
            rotations.append(90)

        for rot in rotations:
            cw, ch = self._get_cell_size(rot)
            cl_w, cl_h = self._get_clearance(rot)
            stride_w, stride_h = self._get_stride(rot)
            bw, bh = self._get_block_size(rot)
            
            # rot별 grid 크기
            grid_w = max(1, (self.W - bw) // stride_w + 1)
            grid_h = max(1, (self.H - bh) // stride_h + 1)

            stack_grid = torch.zeros((grid_h, grid_w), dtype=torch.int32, device=self.device)

            if self._height_map is None:
                stack_grid.fill_(int(self.config.total_cells))
                self._stack_grids[rot] = stack_grid
                continue

            # unit footprint가 맵 안에 들어오는 grid 위치만 계산
            xs = torch.arange(grid_w, device=self.device, dtype=torch.long) * stride_w + cl_w
            ys = torch.arange(grid_h, device=self.device, dtype=torch.long) * stride_h + cl_h
            max_ux = self.W - cw
            max_uy = self.H - ch
            valid_x = xs <= max_ux
            valid_y = ys <= max_uy
            if not bool(valid_x.any().item()) or not bool(valid_y.any().item()):
                self._stack_grids[rot] = stack_grid
                continue

            xs_v = xs[valid_x]
            ys_v = ys[valid_y]

            # min ceiling for each unit origin (uy,ux): [H-ch+1, W-cw+1]
            hmap = torch.nan_to_num(self._height_map, nan=float("inf"))
            neg = (-hmap).unsqueeze(0).unsqueeze(0)
            pooled = F.max_pool2d(neg, kernel_size=(ch, cw), stride=1)
            min_ceiling = (-pooled).squeeze(0).squeeze(0)

            ceiling_sub = min_ceiling[ys_v.view(-1, 1), xs_v.view(1, -1)]

            cell_h = float(self.config.cell_z_height)
            overhead = float(self.config.z_overhead)

            if cell_h <= 0:
                stacks_sub = torch.ones_like(ceiling_sub, dtype=torch.int32)
            else:
                available = ceiling_sub - overhead
                stacks_f = torch.floor(available / cell_h)

                # inf는 total_cells, 음수는 0
                stacks_f = torch.where(
                    torch.isinf(ceiling_sub) | torch.isnan(stacks_f),
                    torch.tensor(float(self.config.total_cells), device=self.device),
                    stacks_f,
                )
                stacks_f = torch.clamp(stacks_f, 0, float(self.config.total_cells))
                stacks_sub = stacks_f.to(torch.int32)

            # stacks_sub을 stack_grid에 반영
            gy_idx = valid_y.nonzero(as_tuple=False).view(-1).long()
            gx_idx = valid_x.nonzero(as_tuple=False).view(-1).long()
            stack_grid[gy_idx[:, None], gx_idx[None, :]] = stacks_sub
            self._stack_grids[rot] = stack_grid

    def expand_from_point(self, start_x: int, start_y: int, rot: int) -> PlacementResult:
        """시작점에서 연결요소(BFS)로 확장 배치.

        성능 목표:
        - CPU↔GPU 왕복 제거
        - flood fill/conv 반복 제거
        - stack count는 rot별 grid 캐시 사용

        NOTE:
        - 결과의 cells/unit_cells set은 생성하지 않음(성능).
          step()에서 unit_positions로 _valid를 슬라이스 업데이트하도록 변경.
        """
        if self.config is None:
            return PlacementResult()

        # 남은 cell 수 계산 (이미 배치된 것 제외)
        remaining = int(self.config.total_cells) - self._placed_cells
        if remaining <= 0:
            return PlacementResult()
        target = remaining
        bw, bh = self._get_block_size(rot)
        cw, ch = self._get_cell_size(rot)
        cl_w, cl_h = self._get_clearance(rot)
        stride_w, stride_h = self._get_stride(rot)

        pm = self._placeable_masks.get(rot, None)
        if pm is None:
            return PlacementResult()
        max_y, max_x = pm.shape

        stack_grid = self._stack_grids.get(rot)
        if stack_grid is None:
            self._build_stack_grids()
            stack_grid = self._stack_grids.get(rot)
        if stack_grid is None:
            return PlacementResult()

        # rot별 stride 기준 grid 크기
        grid_w = max(1, (self.W - bw) // stride_w + 1)
        grid_h = max(1, (self.H - bh) // stride_h + 1)

        start_gx = start_x // stride_w
        start_gy = start_y // stride_h
        if start_gx < 0 or start_gy < 0 or start_gx >= grid_w or start_gy >= grid_h:
            return PlacementResult()

        # 시작점이 placeable 아니면 실패
        wx0 = start_gx * stride_w
        wy0 = start_gy * stride_h
        if wx0 >= max_x or wy0 >= max_y or not bool(pm[wy0, wx0].item()):
            return PlacementResult()

        # visited: Python bytearray로 빠르게
        visited = bytearray(grid_h * grid_w)
        def _idx(gx: int, gy: int) -> int:
            return gy * grid_w + gx

        q: deque[Tuple[int, int]] = deque()
        q.append((start_gx, start_gy))
        visited[_idx(start_gx, start_gy)] = 1

        unit_positions: List[Tuple[int, int, int]] = []
        unit_stacks: List[int] = []
        total_stacks = 0

        # bbox는 block 기준으로 계산
        bbox_x0 = 10**9
        bbox_y0 = 10**9
        bbox_x1 = -1
        bbox_y1 = -1

        # 동일 expand 내에서의 unit 겹침은 사각형 리스트로 체크(전체 텐서 clone 회피)
        placed_units: List[Tuple[int, int]] = []  # (ux,uy)만 저장 (크기는 cw,ch로 고정)

        def _overlap_local(ux: int, uy: int) -> bool:
            for px, py in placed_units:
                if not (ux + cw <= px or px + cw <= ux or uy + ch <= py or py + ch <= uy):
                    return True
            return False

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # unit 개수 제한 (0이면 제한 없음)
        max_units = self.config.max_units_per_step if self.config.max_units_per_step > 0 else float('inf')

        while q and total_stacks < target and len(unit_positions) < max_units:
            gx, gy = q.popleft()

            wx = gx * stride_w
            wy = gy * stride_h

            # placeable 체크
            if wx >= max_x or wy >= max_y or not bool(pm[wy, wx].item()):
                continue

            # placement 시도
            stack = int(stack_grid[gy, gx].item())
            if stack <= 0:
                continue

            ux, uy = wx + cl_w, wy + cl_h
            # bounds (unit)
            if ux < 0 or uy < 0 or ux + cw > self.W or uy + ch > self.H:
                continue
            if _overlap_local(ux, uy):
                continue
            if bool(self._unit_used[uy:uy+ch, ux:ux+cw].any().item()):
                continue

            # 배치 성공! unit 추가
            unit_positions.append((ux, uy, rot))
            unit_stacks.append(stack)
            placed_units.append((ux, uy))
            total_stacks += stack

            # bbox update (block area)
            bx, by = wx, wy
            bbox_x0 = min(bbox_x0, bx)
            bbox_y0 = min(bbox_y0, by)
            bbox_x1 = max(bbox_x1, bx + bw)
            bbox_y1 = max(bbox_y1, by + bh)

            # 배치 성공한 경우에만 인접 노드를 큐에 추가 (연결 보장)
            for dx, dy in directions:
                ngx, ngy = gx + dx, gy + dy
                if ngx < 0 or ngy < 0 or ngx >= grid_w or ngy >= grid_h:
                    continue
                ii = _idx(ngx, ngy)
                if visited[ii]:
                    continue
                visited[ii] = 1
                q.append((ngx, ngy))

        if not unit_positions:
            return PlacementResult()

        bbox = (bbox_x0, bbox_y0, bbox_x1, bbox_y1)
        return PlacementResult(
            cells=set(),
            unit_cells=set(),
            unit_positions=unit_positions,
            unit_stacks=unit_stacks,
            num_units=len(unit_positions),
            total_cells=total_stacks,
            bbox=bbox,
        )
    
    # ========== Snapshot (MCTS 호환) ==========
    
    def get_snapshot(self) -> Dict[str, Any]:
        """현재 상태 저장."""
        return {
            "base_snap": self.base_env.get_snapshot(),
            "_valid": self._valid.clone(),
            "_unit_used": self._unit_used.clone(),
            "_current_config_idx": self._current_config_idx,
            "_remaining_configs": list(self._remaining_configs),
            "current_result": self.current_result,
            "placeable_masks": {k: v.clone() for k, v in self._placeable_masks.items()},
            "placed_history": {k: list(v) for k, v in self.placed_history.items()},
            "_placed_cells": self._placed_cells,
        }
    
    def set_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """상태 복원."""
        if "base_snap" in snapshot:
            self.base_env.set_snapshot(snapshot["base_snap"])
        self._valid = snapshot["_valid"].clone()
        self._unit_used = snapshot["_unit_used"].clone()
        self._current_config_idx = snapshot["_current_config_idx"]
        self._remaining_configs = list(snapshot["_remaining_configs"])
        self.current_result = snapshot.get("current_result")
        if "placeable_masks" in snapshot:
            self._placeable_masks = {k: v.clone() for k, v in snapshot["placeable_masks"].items()}
        self.placed_history = {k: list(v) for k, v in snapshot.get("placed_history", {}).items()}
        self._placed_cells = snapshot.get("_placed_cells", 0)
        
        # config 업데이트 + 파생값 재계산
        if self._remaining_configs:
            self.config = self._remaining_configs[0]
            self.gid = self.config.gid
            self.allow_rotation = self.config.rotatable
            self._setup_for_current_config()
        else:
            self.config = None
            self.gid = None
    
    # ========== Gym API ==========
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """시작점 선택 후 확장 배치."""
        grid_x, grid_y, rot = self.decode_action(int(action))
        world_x, world_y = self.grid_to_world(grid_x, grid_y, rot)
        gid_before = self.gid
        
        if not self.is_placeable(world_x, world_y, rot):
            return self._build_obs(), -1.0, False, False, {
                "reason": "not_placeable",
                "grid_x": grid_x, "grid_y": grid_y, "rot": rot,
                "world_x": world_x, "world_y": world_y,
            }
        
        result = self.expand_from_point(world_x, world_y, rot)
        
        if result.num_units == 0:
            return self._build_obs(), -1.0, False, False, {"reason": "no_placement"}
        
        cw, ch = self._get_cell_size(rot)
        for ux, uy, _ in result.unit_positions:
            self._unit_used[uy:uy+ch, ux:ux+cw] = True

        # block 영역을 슬라이스로 invalid 처리 (cells set 순회 제거)
        bw, bh = self._get_block_size(rot)
        cl_w, cl_h = self._get_clearance(rot)
        for ux, uy, _ in result.unit_positions:
            bx = ux - cl_w
            by = uy - cl_h
            if bx < 0 or by < 0 or bx + bw > self.W or by + bh > self.H:
                continue
            self._valid[by:by+bh, bx:bx+bw] = False
        
        self.current_result = result

        # 히스토리 누적 (gid 기준)
        if gid_before is not None and result.unit_positions:
            buf = self.placed_history.setdefault(gid_before, [])
            stacks = result.unit_stacks
            for i, (ux, uy, r) in enumerate(result.unit_positions):
                s = int(stacks[i]) if i < len(stacks) else 0
                buf.append((int(ux), int(uy), int(r), s))

        self._update_placeable_masks()
        
        # 배치된 cell 수 누적
        self._placed_cells += result.total_cells
        
        empty_ratio = self._compute_empty_ratio()
        reward = empty_ratio
        
        # 현재 그룹 배치 완료 체크 (누적 기준)
        terminated = False
        if self.config and self._placed_cells >= self.config.total_cells:
            # 현재 그룹 완료, 다음 그룹으로
            if self._remaining_configs:
                self._remaining_configs.pop(0)
            
            if not self._remaining_configs:
                terminated = True
            else:
                # 다음 그룹 설정
                self.config = self._remaining_configs[0]
                self.gid = self.config.gid
                self.allow_rotation = self.config.rotatable
                # 새 그룹에 맞게 재설정 + placed_cells 리셋
                self._placed_cells = 0
                self._setup_for_current_config()
        
        info = {
            "reason": "placed",
            "grid_x": grid_x, "grid_y": grid_y, "rot": rot,
            "world_x": world_x, "world_y": world_y,
            "num_units": result.num_units,
            "total_cells": result.total_cells,  # 이번 step에서 배치된 cell 수
            "placed_cells": self._placed_cells,  # 누적 배치된 cell 수
            "target_cells": self.config.total_cells if self.config else 0,  # 목표 cell 수
            "unit_stacks": result.unit_stacks,
            "bbox": result.bbox,
            "empty_ratio": empty_ratio,
            # 이번 step에서 배치한 gid(배치 직전 gid)
            "current_gid": gid_before,
        }
        
        return self._build_obs(), reward, terminated, False, info
    
    def _setup_for_current_config(self):
        """현재 config에 맞게 파라미터 재설정."""
        if self.config is None:
            return
        
        self.cell_w = int(self.config.cell_width)
        self.cell_h = int(self.config.cell_depth)
        self.clearance_w = int(self.config.clearance_w)
        self.clearance_h = int(self.config.clearance_h)
        self.gap_w = int(self.config.gap_w)
        self.gap_h = int(self.config.gap_h)
        self.block_w = self.cell_w + 2 * self.clearance_w
        self.block_h = self.cell_h + 2 * self.clearance_h
        self.stride_x = self.cell_w + self.gap_w
        self.stride_y = self.cell_h + self.gap_h
        self._n_rots = 2 if self.allow_rotation and self.cell_w != self.cell_h else 1
        
        self._grid_h = max(1, (self.H - self.block_h) // self.stride_y + 1)
        self._grid_w = max(1, (self.W - self.block_w) // self.stride_x + 1)
        self.action_space = gym.spaces.Discrete(self._grid_h * self._grid_w * self._n_rots)
        
        self._update_placeable_masks()
        self._build_stack_grids()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """환경 리셋."""
        super().reset(seed=seed)
        
        self._valid = self._initial_valid.clone()
        self._unit_used = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        self.current_result = None
        self.placed_history = {}
        self._remaining_configs = list(self.configs)
        self._placed_cells = 0  # 현재 그룹 배치된 cell 수 초기화
        
        if self._remaining_configs:
            self.config = self._remaining_configs[0]
            self.gid = self.config.gid
            self.allow_rotation = self.config.rotatable
            self._setup_for_current_config()
        
        self._update_placeable_masks()
        
        return self._build_obs(), {}
    
    def _build_obs(self) -> Dict[str, torch.Tensor]:
        """Observation 빌드."""
        return {
            "valid": self._valid.clone(),
            "unit_used": self._unit_used.clone(),
            "action_mask": self.get_action_mask(),
        }
    
    # ========== 시각화 ==========
    
    def render(self, show_grid: bool = False):
        """시각화 (matplotlib)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        invalid_np = (~self._valid).cpu().numpy()
        ax.imshow(invalid_np, cmap='Greys', alpha=0.5, origin='lower', 
                  extent=[0, self.W, 0, self.H])
        
        if self.current_result:
            for ux, uy, rot in self.current_result.unit_positions:
                cw, ch = self._get_cell_size(rot)
                cl_w, cl_h = self._get_clearance(rot)
                bw, bh = self._get_block_size(rot)
                
                rect = patches.Rectangle(
                    (ux, uy), cw, ch,
                    linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.7
                )
                ax.add_patch(rect)
                
                bx, by = ux - cl_w, uy - cl_h
                rect_c = patches.Rectangle(
                    (bx, by), bw, bh,
                    linewidth=1, edgecolor='gray', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect_c)
        
        if show_grid:
            pm = self._placeable_masks.get(0, None)
            if pm is not None:
                indices = pm.nonzero(as_tuple=False)
                for idx in indices:
                    gy, gx = idx[0].item(), idx[1].item()
                    x, y = self.grid_to_world(gx, gy)
                    ax.plot(x, y, 'b.', markersize=2, alpha=0.3)
        
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect('equal')
        ax.set_title(f"Dynamic Storage (units: {self.current_result.num_units if self.current_result else 0})")
        
        plt.tight_layout()
        return fig, ax
