"""Dynamic group generation using block-based placement.

블록 단위로 배치 가능 위치를 검사하고, cost가 낮은 곳부터 채워서
동적 그룹을 생성합니다.

사용법:
    generator = DynamicGroupGenerator(env)
    result = generator.generate(
        unit_w=8,
        unit_h=20,
        clearance_w=2,
        clearance_h=3,
        target_area=500,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from collections import deque


@dataclass
class DynamicGroup:
    """동적 그룹 생성 결과."""
    cells: Set[Tuple[int, int]]  # 전체 영역 셀 집합 (unit + clearance)
    unit_cells: Set[Tuple[int, int]]  # unit 본체 셀 집합
    unit_positions: List[Tuple[int, int, int]]  # 각 unit의 (x, y, rotation)
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: int  # 전체 면적 (셀 개수)
    unit_w: int
    unit_h: int
    clearance_w: int
    clearance_h: int
    num_units: int  # 배치된 unit 수
    
    @property
    def block_w(self) -> int:
        return self.unit_w + 2 * self.clearance_w
    
    @property
    def block_h(self) -> int:
        return self.unit_h + 2 * self.clearance_h
    
    def to_mask(self, grid_width: int, grid_height: int) -> torch.Tensor:
        """전체 영역을 bool mask로 변환."""
        mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
        for x, y in self.cells:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                mask[y, x] = True
        return mask
    
    def to_unit_mask(self, grid_width: int, grid_height: int) -> torch.Tensor:
        """unit 본체만 bool mask로 변환."""
        mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
        for x, y in self.unit_cells:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                mask[y, x] = True
        return mask


class DynamicGroupGenerator:
    """블록 기반 동적 그룹 생성기.
    
    사용법:
        generator = DynamicGroupGenerator(env)
        result = generator.generate(block_w=10, block_h=10, target_area=500)
    """
    
    def __init__(self, env):
        """
        Args:
            env: FactoryLayoutEnv 인스턴스
        """
        self.env = env
        self.grid_width = env.grid_width
        self.grid_height = env.grid_height
        self.device = env.device
        
        # 유효 영역 (배치 가능한 셀)
        self._valid = ~(env._occ_invalid | env._static_invalid)
    
    def generate(
        self,
        unit_w: int,
        unit_h: int,
        clearance_w: int,
        clearance_h: int,
        target_area: int,
        rot: int = 0,
        allow_rotation: bool = False,
        rotation_mode: str = 'mixed',
        gid: Optional[str] = None,
    ) -> Optional[DynamicGroup]:
        """연결요소 기반(시작점→이웃 확장) 동적 그룹 생성.
        
        stride_x = unit_w (가로로 unit만큼 이동, clearance 겹침)
        stride_y = unit_h + clearance_h (세로로 unit + clearance만큼 이동)
        
        Args:
            unit_w: unit 본체 가로 크기
            unit_h: unit 본체 세로 크기
            clearance_w: 좌우 여백 (한쪽 기준)
            clearance_h: 상하 여백 (한쪽 기준)
            target_area: 목표 면적 (셀 개수)
            rot: 시작점에서 선택된 rotation (0 또는 90). 이 rotation으로만 이어붙임.
            allow_rotation: (호환용) 사용하지 않음. rot로 고정됨.
            rotation_mode: (호환용) 사용하지 않음. rot로 고정됨.
            gid: (선택) cost 계산용 그룹 ID
        
        Returns:
            DynamicGroup 또는 None (실패 시)
        """
        H, W = self.grid_height, self.grid_width
        
        # ===== 단일 rotation 설정 (처음 action에서 정해진 rot로 고정) =====
        if rot not in (0, 90):
            raise ValueError(f"rot must be 0 or 90, got {rot}")

        if rot == 0:
            uw, uh = unit_w, unit_h
            cw, ch = clearance_w, clearance_h
        else:
            # 회전 시 unit/clearance 모두 swap
            uw, uh = unit_h, unit_w
            cw, ch = clearance_h, clearance_w

        bw = uw + 2 * cw
        bh = uh + 2 * ch

        # stride: 동일 rot에서 이어붙이기 기준으로 설정
        # (기존 코드가 stride_x = unit_w + clearance_w, stride_y = unit_h 였으므로 동일 규칙 적용)
        stride_x = uw + cw
        stride_y = uh

        if bh > H or bw > W:
            return None

        # ===== placeable grid 계산 (conv2d, stride grid 좌표) =====
        valid_float = self._valid.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        kernel = torch.ones((1, 1, bh, bw), device=self.device)
        valid_conv = F.conv2d(valid_float, kernel, padding=0, stride=(stride_y, stride_x)).squeeze()

        if valid_conv.dim() == 0:
            valid_conv = valid_conv.unsqueeze(0).unsqueeze(0)
        elif valid_conv.dim() == 1:
            valid_conv = valid_conv.unsqueeze(0)

        placeable = (valid_conv == bw * bh)  # [grid_h, grid_w] on device
        if not placeable.any():
            return None

        # CPU로 내려서 BFS 수행 (파이썬 루프 + 디바이스 동기화 방지)
        placeable_cpu = placeable.detach().cpu()
        grid_h, grid_w = placeable_cpu.shape

        # ===== 시작점: BL (x 최소, y 최소) =====
        idxs = placeable_cpu.nonzero(as_tuple=False)  # [N,2] (gy,gx)
        gy = idxs[:, 0].long()
        gx = idxs[:, 1].long()
        bx = gx * stride_x
        by = gy * stride_y
        key = bx * (H + 1) + by  # x 우선, tie는 y
        start_i = int(torch.argmin(key).item())
        start_gx = int(gx[start_i].item())
        start_gy = int(gy[start_i].item())

        # ===== BFS 연결요소 확장 =====
        cells_mask = torch.zeros((H, W), dtype=torch.bool)  # 전체(block)
        unit_mask = torch.zeros((H, W), dtype=torch.bool)   # unit
        unit_used = torch.zeros((H, W), dtype=torch.bool)   # unit 충돌 방지

        unit_positions: List[Tuple[int, int, int]] = []
        num_units = 0
        area = 0

        visited = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        q: deque[Tuple[int, int]] = deque()
        q.append((start_gx, start_gy))
        visited[start_gy, start_gx] = True

        def try_place(gx_i: int, gy_i: int) -> bool:
            nonlocal area, num_units
            if not bool(placeable_cpu[gy_i, gx_i].item()):
                return False

            bx_i = gx_i * stride_x
            by_i = gy_i * stride_y
            ux_i = bx_i + cw
            uy_i = by_i + ch

            if bx_i < 0 or by_i < 0 or bx_i + bw > W or by_i + bh > H:
                return False
            if ux_i < 0 or uy_i < 0 or ux_i + uw > W or uy_i + uh > H:
                return False

            if bool(unit_used[uy_i:uy_i+uh, ux_i:ux_i+uw].any().item()):
                return False

            # block 영역 누적
            block_region = cells_mask[by_i:by_i+bh, bx_i:bx_i+bw]
            added = int((~block_region).sum().item())
            cells_mask[by_i:by_i+bh, bx_i:bx_i+bw] = True
            area += added

            # unit 영역 누적
            unit_mask[uy_i:uy_i+uh, ux_i:ux_i+uw] = True
            unit_used[uy_i:uy_i+uh, ux_i:ux_i+uw] = True

            unit_positions.append((int(ux_i), int(uy_i), int(rot)))
            num_units += 1
            return True

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # stride grid 4방향

        while q and area < target_area:
            gx_i, gy_i = q.popleft()

            placed = try_place(gx_i, gy_i)
            if placed:
                # 배치 성공한 노드에서만 이웃 확장
                for dx, dy in directions:
                    ngx = gx_i + dx
                    ngy = gy_i + dy
                    if ngx < 0 or ngy < 0 or ngx >= grid_w or ngy >= grid_h:
                        continue
                    if bool(visited[ngy, ngx].item()):
                        continue
                    visited[ngy, ngx] = True
                    if bool(placeable_cpu[ngy, ngx].item()):
                        q.append((ngx, ngy))

        if area <= 0 or not unit_positions:
            return None

        # mask -> set 변환 (마지막 1회)
        cells_idx = cells_mask.nonzero(as_tuple=False)  # [N,2] (y,x)
        unit_idx = unit_mask.nonzero(as_tuple=False)
        cells: Set[Tuple[int, int]] = set((int(x.item()), int(y.item())) for y, x in cells_idx)
        unit_cells: Set[Tuple[int, int]] = set((int(x.item()), int(y.item())) for y, x in unit_idx)

        # bbox
        ys = cells_idx[:, 0]
        xs = cells_idx[:, 1]
        x_min = int(xs.min().item())
        x_max = int(xs.max().item())
        y_min = int(ys.min().item())
        y_max = int(ys.max().item())

        # 출력의 unit_w/unit_h는 "입력 기준"을 유지(기존 시그니처 호환)
        # 실제 unit_positions는 rot에 맞는 uw/uh로 배치되어 있음.
        return DynamicGroup(
            cells=cells,
            unit_cells=unit_cells,
            unit_positions=unit_positions,
            bbox=(x_min, y_min, x_max + 1, y_max + 1),
            area=int(area),
            unit_w=unit_w,
            unit_h=unit_h,
            clearance_w=clearance_w,
            clearance_h=clearance_h,
            num_units=num_units,
        )
    
    def generate_multiple(
        self,
        count: int,
        unit_w: int,
        unit_h: int,
        clearance_w: int,
        clearance_h: int,
        target_area: int,
        allow_rotation: bool = False,
        rotation_mode: str = 'mixed',
        **kwargs,
    ) -> List[DynamicGroup]:
        """여러 개의 동적 그룹 생성 (서로 겹치지 않게).
        
        Args:
            count: 생성할 그룹 수
            unit_w: unit 본체 가로 크기
            unit_h: unit 본체 세로 크기
            clearance_w: 좌우 여백
            clearance_h: 상하 여백
            target_area: 각 그룹의 목표 면적
            allow_rotation: 90도 회전 허용 여부
            rotation_mode: 'mixed' 또는 'sequential'
            **kwargs: generate()에 전달할 추가 인자
        
        Returns:
            DynamicGroup 리스트
        """
        groups = []
        original_valid = self._valid.clone()
        
        for _ in range(count):
            group = self.generate(
                unit_w=unit_w,
                unit_h=unit_h,
                clearance_w=clearance_w,
                clearance_h=clearance_h,
                target_area=target_area,
                allow_rotation=allow_rotation,
                rotation_mode=rotation_mode,
                **kwargs,
            )
            if group is None:
                break
            
            groups.append(group)
            
            # 생성된 영역을 유효 영역에서 제외
            for x, y in group.cells:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    self._valid[y, x] = False
        
        # 원래 상태 복원
        self._valid = original_valid
        
        return groups


if __name__ == "__main__":
    """사용 예시 (시각화 포함).
    
    1. env 로드 후 일부 그룹 배치
    2. conv2d로 배치 가능 위치 계산 시각화
    3. 동적 그룹 생성 및 시각화
    """
    import sys
    sys.path.insert(0, str(__file__).rsplit("postprocess", 1)[0])
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from envs.json_loader import load_env
    
    # 1. env 로드 및 일부 그룹 배치
    config_path = "env_configs/clearance_03.json"
    print(f"[1] Loading: {config_path}")
    loaded = load_env(config_path)
    env = loaded.env
    env.reset()
    
    H, W = env.grid_height, env.grid_width
    print(f"    Grid: {W}x{H}")
    print(f"    Groups: {list(env.groups.keys())}")
    print(f"    Remaining: {env.remaining}")
    
    # 일부 그룹 배치 (A, B, C)
    print("\n[2] Placing some groups...")
    placements = [
        ("H", 200, 300, 0),   # A 그룹: (200, 300), 회전 없음
        ("B", 350, 100, 0),   # B 그룹: (350, 100), 회전 없음
        ("E", 180, 100, 0),   # C 그룹: (180, 100), 회전 없음
    ]
    
    for gid, x, y, rot in placements:
        if gid in env.remaining:
            obs, reward, terminated, truncated, info = env.step_place(x=x, y=y, rot=rot)
            print(f"    Placed {gid} at ({x}, {y}, rot={rot}) - reason: {info.get('reason')}")
    
    print(f"    Placed: {list(env.placed)}")
    print(f"    Remaining: {env.remaining}")
    
    # 3. 동적 그룹 생성
    print("\n[3] Generating dynamic groups...")
    generator = DynamicGroupGenerator(env)
    
    # 파라미터 설정
    unit_w, unit_h = 40, 16
    clearance_w, clearance_h = 6, 4
    block_w = unit_w + 2 * clearance_w
    block_h = unit_h + 2 * clearance_h
    stride_x = unit_w                  # generate() 함수와 동일
    stride_y = unit_h + clearance_h    # generate() 함수와 동일
    allow_rotation = True
    rotation_mode = 'sequential'
    
    print(f"    unit_size={unit_w}x{unit_h}, clearance=({clearance_w}, {clearance_h})")
    print(f"    block_size={block_w}x{block_h}, stride=({stride_x}, {stride_y})")
    
    groups = generator.generate_multiple(
        count=1,
        unit_w=unit_w,
        unit_h=unit_h,
        clearance_w=clearance_w,
        clearance_h=clearance_h,
        target_area=80000,
        allow_rotation=allow_rotation,
        rotation_mode=rotation_mode,
    )
    
    print(f"    Generated {len(groups)} groups")
    print(f"    allow_rotation={allow_rotation}, rotation_mode={rotation_mode}")
    for i, g in enumerate(groups):
        rot_counts = {}
        for _, _, rot in g.unit_positions:
            rot_counts[rot] = rot_counts.get(rot, 0) + 1
        print(f"    [Group {i+1}] area={g.area}, units={g.num_units}, rotations={rot_counts}")
    
    # 4. conv2d로 배치 가능 위치 계산 (시각화용)
    print("\n[4] Computing conv2d placeable map...")
    valid_float = generator._valid.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kernel = torch.ones((1, 1, block_h, block_w), device=generator.device)
    
    # conv2d 결과: 각 위치에서 block 영역 내 valid 셀 수
    conv_result = F.conv2d(valid_float, kernel, padding=0, stride=(stride_y, stride_x)).squeeze()
    
    # 배치 가능 = block 전체가 valid (sum == block_w * block_h)
    placeable_map = (conv_result == block_w * block_h)
    
    conv_h, conv_w = conv_result.shape
    print(f"    conv2d output shape: {conv_w}x{conv_h}")
    print(f"    Placeable positions: {placeable_map.sum().item()}")
    
    # 5. 시각화 (2x2 subplot)
    print("\n[5] Visualizing...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    current_valid = generator._valid.cpu().numpy()
    static_invalid = env._static_invalid.cpu().numpy()
    occ_invalid = env._occ_invalid.cpu().numpy()
    
    # --- (1) Env State ---
    ax1 = axes[0, 0]
    env_img = np.zeros((H, W, 3))
    env_img[current_valid] = [0.9, 0.9, 0.9]
    env_img[static_invalid] = [0.3, 0.3, 0.3]
    env_img[occ_invalid] = [0.8, 0.3, 0.3]
    
    ax1.imshow(env_img, origin='lower', aspect='equal')
    ax1.set_title(f'(1) Env State (placed: {list(env.placed)})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    for gid in env.placed:
        if gid in env.positions:
            x_bl, y_bl, rot = env.positions[gid]
            grp = env.groups[gid]
            if rot == 0:
                gw, gh = grp.width, grp.height
            else:
                gw, gh = grp.height, grp.width
            rect = plt.Rectangle(
                (x_bl - 0.5, y_bl - 0.5), gw, gh,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax1.add_patch(rect)
            ax1.text(x_bl + gw/2, y_bl + gh/2, gid, ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='yellow')
    
    legend_patches_env = [
        mpatches.Patch(color=[0.9, 0.9, 0.9], label='Valid'),
        mpatches.Patch(color=[0.3, 0.3, 0.3], label='Forbidden'),
        mpatches.Patch(color=[0.8, 0.3, 0.3], label='Occupied'),
    ]
    ax1.legend(handles=legend_patches_env, loc='upper right', fontsize=7)
    
    # --- (2) Conv2D Placeable Map ---
    ax2 = axes[0, 1]
    conv_vis = np.zeros((H, W, 3))
    conv_vis[current_valid] = [0.85, 0.85, 0.85]
    conv_vis[~current_valid] = [0.3, 0.3, 0.3]
    
    placeable_indices = placeable_map.nonzero(as_tuple=False)
    for idx in placeable_indices:
        gy, gx = idx[0].item(), idx[1].item()
        ox = gx * stride_x
        oy = gy * stride_y
        for dy in range(block_h):
            for dx in range(block_w):
                px, py = ox + dx, oy + dy
                if 0 <= px < W and 0 <= py < H:
                    conv_vis[py, px] = [0.4, 0.8, 0.4]
    
    ax2.imshow(conv_vis, origin='lower', aspect='equal')
    ax2.set_title(f'(2) Conv2D Placeable Map\nblock={block_w}x{block_h}, stride={stride_x}x{stride_y}, positions={placeable_map.sum().item()}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    for gy in range(conv_h):
        for gx in range(conv_w):
            ox = gx * stride_x
            oy = gy * stride_y
            color = 'green' if placeable_map[gy, gx] else 'red'
            ax2.plot(ox, oy, 'o', markersize=2, color=color, alpha=0.5)
    
    # --- (3) Cost Map ---
    ax3 = axes[1, 0]
    
    # 배경 (valid/invalid)
    cost_vis = np.zeros((H, W, 3))
    cost_vis[current_valid] = [0.85, 0.85, 0.85]
    cost_vis[~current_valid] = [0.3, 0.3, 0.3]
    
    ax3.imshow(cost_vis, origin='lower', aspect='equal')
    
    # cost 계산 및 점으로 표시
    if placeable_indices.numel() > 0:
        cx, cy = W / 2.0, H / 2.0
        costs = []
        positions = []
        for idx in placeable_indices:
            gy, gx = idx[0].item(), idx[1].item()
            ox = gx * stride_x
            oy = gy * stride_y
            center_x = ox + block_w / 2.0
            center_y = oy + block_h / 2.0
            cost = abs(center_x - cx) + abs(center_y - cy)
            costs.append(cost)
            positions.append((ox, oy))
        
        min_cost, max_cost = min(costs), max(costs)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        
        # 점으로만 표시 (green=low cost, red=high cost)
        for (ox, oy), cost in zip(positions, costs):
            norm_cost = (cost - min_cost) / cost_range
            dot_color = (norm_cost, 1 - norm_cost, 0.2)
            ax3.plot(ox, oy, 'o', markersize=4, color=dot_color)
    
    ax3.set_title('(3) Cost Map (green=low, red=high)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # --- (4) Generated Dynamic Groups ---
    ax4 = axes[1, 1]
    
    # 배경
    bg_img = np.zeros((H, W, 3))
    bg_img[current_valid] = [0.9, 0.9, 0.9]
    bg_img[~current_valid] = [0.3, 0.3, 0.3]
    
    # 배치된 그룹 표시 (연한 빨강)
    bg_img[occ_invalid] = [0.7, 0.5, 0.5]
    
    # 동적 그룹 색상 (clearance=연한색, unit=진한색)
    group_colors = [
        ([0.6, 0.9, 0.6], [0.2, 0.6, 0.2]),  # 연두 / 진녹
        ([0.6, 0.7, 0.9], [0.2, 0.3, 0.7]),  # 연파랑 / 진파랑
        ([0.9, 0.8, 0.6], [0.7, 0.5, 0.2]),  # 연주황 / 주황
    ]
    
    for i, group in enumerate(groups):
        clearance_color, unit_color = group_colors[i % len(group_colors)]
        
        # clearance 영역 (전체 - unit)
        clearance_cells = group.cells - group.unit_cells
        for x, y in clearance_cells:
            if 0 <= x < W and 0 <= y < H:
                bg_img[y, x] = clearance_color
        
        # unit 본체
        for x, y in group.unit_cells:
            if 0 <= x < W and 0 <= y < H:
                bg_img[y, x] = unit_color
    
    ax4.imshow(bg_img, origin='lower', aspect='equal')
    ax4.set_title(f'(4) Generated Dynamic Groups ({len(groups)}) - Unit(dark) / Clearance(light)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    
    # unit 경계 그리기
    for i, group in enumerate(groups):
        for ux, uy, rot in group.unit_positions:
            if rot == 0:
                uw, uh = group.unit_w, group.unit_h
            else:
                uw, uh = group.unit_h, group.unit_w
            
            rect = plt.Rectangle(
                (ux - 0.5, uy - 0.5), uw, uh,
                linewidth=1, edgecolor='black', facecolor='none'
            )
            ax4.add_patch(rect)
    
    # 배치된 그룹 레이블
    for gid in env.placed:
        if gid in env.positions:
            x_bl, y_bl, rot = env.positions[gid]
            grp = env.groups[gid]
            if rot == 0:
                gw, gh = grp.width, grp.height
            else:
                gw, gh = grp.height, grp.width
            ax4.text(x_bl + gw/2, y_bl + gh/2, gid, ha='center', va='center', 
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # 범례
    legend_patches = [
        mpatches.Patch(color=[0.7, 0.5, 0.5], label='Placed groups (env)'),
    ]
    for i, g in enumerate(groups):
        clearance_color, unit_color = group_colors[i % len(group_colors)]
        legend_patches.append(
            mpatches.Patch(color=unit_color, label=f'DynGroup{i+1} unit ({g.num_units} units)')
        )
        legend_patches.append(
            mpatches.Patch(color=clearance_color, label=f'DynGroup{i+1} clearance')
        )
    ax4.legend(handles=legend_patches, loc='upper right', fontsize=6)
    
    plt.tight_layout()
    plt.savefig('dynamic_group_result.png', dpi=150)
    print("    Saved: dynamic_group_result.png")
    plt.show()
    
    # ========================================================================
    # 6. Rotation Offset Comparison (before vs after offset)
    # ========================================================================
    print("\n" + "="*70)
    print("[6] Rotation Offset Example")
    print("="*70)
    
    # Settings - asymmetric clearance to show offset effect
    r_unit_w, r_unit_h = 10, 14
    r_clear_w, r_clear_h = 2, 4
    
    # Original (rot=0)
    orig_block_w = r_unit_w + 2 * r_clear_w  # 14
    orig_block_h = r_unit_h + 2 * r_clear_h  # 22
    
    # Rotated (rot=90): unit and clearance swap
    rot_unit_w, rot_unit_h = r_unit_h, r_unit_w  # 14, 10
    rot_clear_w, rot_clear_h = r_clear_h, r_clear_w  # 4, 2
    rot_block_w = rot_unit_w + 2 * rot_clear_w  # 22
    rot_block_h = rot_unit_h + 2 * rot_clear_h  # 14
    
    # Stride (based on original)
    r_stride_x = r_unit_w  # 10
    r_stride_y = r_unit_h + r_clear_h  # 18
    
    # Offset for rotated block (y-axis only)
    # stride_y = unit_h + clearance_h 로 세로 간격 결정
    # 회전하면 clearance_h가 바뀌므로 y축 정렬 필요
    offset_y = r_clear_h - rot_clear_h  # 4 - 2 = 2
    
    print(f"    Original: unit={r_unit_w}x{r_unit_h}, clearance=({r_clear_w},{r_clear_h})")
    print(f"    Rotated:  unit={rot_unit_w}x{rot_unit_h}, clearance=({rot_clear_w},{rot_clear_h})")
    print(f"    Stride: ({r_stride_x}, {r_stride_y})")
    print(f"    Offset_y: {offset_y}")
    
    # Colors
    color_unit_orig = [0.4, 0.4, 0.4]      # gray
    color_clear_orig = [0.6, 0.9, 0.6]     # light green
    color_unit_rot = [0.3, 0.3, 0.6]       # blue
    color_clear_rot = [0.6, 0.7, 0.9]      # light blue
    color_overlap = [0.9, 0.3, 0.3]        # red for overlap
    color_bg = [0.95, 0.95, 0.95]
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    
    # Bottom row: original blocks (3 in a row)
    num_orig_x = 3
    img_w = r_stride_x * (num_orig_x - 1) + orig_block_w + 6
    img_h = r_stride_y + rot_block_h + 6  # bottom row + top rotated block
    
    # Rotated block position: on top of middle original block
    rot_grid_x = 1  # above the middle original block
    
    def draw_scene(ax, img, apply_offset, title):
        img[:] = color_bg
        
        # Track original clearance top boundary for overlap detection
        orig_clear_top = orig_block_h  # top of original block's clearance
        
        # === Draw bottom row: original blocks (clearance first, then unit) ===
        # Clearance
        for gx in range(num_orig_x):
            bx = gx * r_stride_x
            by = 0
            for dy in range(orig_block_h):
                for dx in range(orig_block_w):
                    px, py = bx + dx, by + dy
                    if 0 <= px < img_w and 0 <= py < img_h:
                        img[py, px] = color_clear_orig
        # Unit
        for gx in range(num_orig_x):
            bx = gx * r_stride_x
            by = 0
            ux, uy = bx + r_clear_w, by + r_clear_h
            for dy in range(r_unit_h):
                for dx in range(r_unit_w):
                    px, py = ux + dx, uy + dy
                    if 0 <= px < img_w and 0 <= py < img_h:
                        img[py, px] = color_unit_orig
        
        # === Draw rotated block on top ===
        rot_bx = rot_grid_x * r_stride_x  # x position same for both
        if apply_offset:
            rot_by = r_stride_y + offset_y  # with y offset
        else:
            rot_by = r_stride_y  # no offset
        
        # Check if rotated block's bottom clearance overlaps original's top clearance
        # Overlap occurs when rot_by < orig_clear_top
        overlap_y_start = rot_by
        overlap_y_end = min(rot_by + rot_clear_h, orig_clear_top)  # rotated bottom clearance
        
        # Rotated clearance
        for dy in range(rot_block_h):
            for dx in range(rot_block_w):
                px, py = rot_bx + dx, rot_by + dy
                if 0 <= px < img_w and 0 <= py < img_h:
                    # Check if this is in the overlap region (below original clearance top)
                    if py < orig_clear_top:
                        img[py, px] = color_overlap  # OVERLAP with original clearance!
                    else:
                        img[py, px] = color_clear_rot
        
        # Rotated unit
        rot_ux, rot_uy = rot_bx + rot_clear_w, rot_by + rot_clear_h
        for dy in range(rot_unit_h):
            for dx in range(rot_unit_w):
                px, py = rot_ux + dx, rot_uy + dy
                if 0 <= px < img_w and 0 <= py < img_h:
                    img[py, px] = color_unit_rot
        
        ax.imshow(img, origin='lower', aspect='equal')
        
        # Draw borders for original blocks
        for gx in range(num_orig_x):
            bx = gx * r_stride_x
            by = 0
            ax.add_patch(plt.Rectangle((bx - 0.5, by - 0.5), orig_block_w, orig_block_h,
                         lw=1, edgecolor='gray', facecolor='none', linestyle='--'))
            ux, uy = bx + r_clear_w, by + r_clear_h
            ax.add_patch(plt.Rectangle((ux - 0.5, uy - 0.5), r_unit_w, r_unit_h,
                         lw=2, edgecolor='black', facecolor='none'))
        
        # Rotated block border
        ax.add_patch(plt.Rectangle((rot_bx - 0.5, rot_by - 0.5), rot_block_w, rot_block_h,
                     lw=1.5, edgecolor='darkblue', facecolor='none', linestyle='--'))
        ax.add_patch(plt.Rectangle((rot_ux - 0.5, rot_uy - 0.5), rot_unit_w, rot_unit_h,
                     lw=2, edgecolor='darkblue', facecolor='none'))
        
        # Draw horizontal line at original clearance top boundary
        ax.axhline(y=orig_clear_top - 0.5, color='red', linestyle=':', lw=1.5, alpha=0.7)
        
        ax.set_xlim(-2, img_w)
        ax.set_ylim(-2, img_h)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # ===== Left: Without offset (overlap occurs) =====
    img_left = np.ones((img_h, img_w, 3)) * color_bg
    draw_scene(axes2[0], img_left, apply_offset=False, 
               title='Without Offset\n(rotated block overlaps original clearance = RED)')
    
    # ===== Right: With offset (no overlap) =====
    img_right = np.ones((img_h, img_w, 3)) * color_bg
    draw_scene(axes2[1], img_right, apply_offset=True,
               title=f'With Offset_y = {offset_y}\n(no overlap)')
    
    # Offset arrow on right plot (y-axis only)
    rot_bx = rot_grid_x * r_stride_x
    arrow_x = rot_bx + rot_block_w + 2  # right side of rotated block
    no_offset_by = r_stride_y
    with_offset_by = r_stride_y + offset_y
    axes2[1].annotate('', 
        xy=(arrow_x, with_offset_by),
        xytext=(arrow_x, no_offset_by),
        arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
    axes2[1].text(arrow_x + 1, (no_offset_by + with_offset_by) / 2,
        f'offset_y = {offset_y}', ha='left', va='center',
        fontsize=10, color='red', fontweight='bold')
    
    # Legend
    legend_items = [
        mpatches.Patch(facecolor=color_unit_orig, edgecolor='black', label='Original Unit'),
        mpatches.Patch(facecolor=color_clear_orig, edgecolor='black', label='Original Clearance'),
        mpatches.Patch(facecolor=color_unit_rot, edgecolor='darkblue', label='Rotated Unit'),
        mpatches.Patch(facecolor=color_clear_rot, edgecolor='darkblue', label='Rotated Clearance'),
        mpatches.Patch(facecolor=color_overlap, edgecolor='red', label='OVERLAP'),
    ]
    axes2[0].legend(handles=legend_items, loc='upper right', fontsize=8)
    axes2[1].legend(handles=legend_items, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('rotation_offset.png', dpi=150)
    print(f"    Saved: rotation_offset.png")
    plt.show()
