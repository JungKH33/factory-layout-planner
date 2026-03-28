"""CPU/GPU 및 병렬처리 속도 테스트 스크립트."""
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from envs.action_space import ActionSpace
from envs.env_loader import load_env


def _score_poses(env, gid, x, y, rotation=None):
    """Test helper: score center poses via spec.score_batch (BL API)."""
    spec = env.group_specs[gid]
    x_t = x if torch.is_tensor(x) else torch.tensor([float(x)], dtype=torch.float32, device=env.device)
    y_t = y if torch.is_tensor(y) else torch.tensor([float(y)], dtype=torch.float32, device=env.device)
    x_t = x_t.to(dtype=torch.float32, device=env.device).view(-1)
    y_t = y_t.to(dtype=torch.float32, device=env.device).view(-1)
    poses = torch.stack([x_t, y_t], dim=-1)
    bw = spec.body_widths   # [V]
    bh = spec.body_heights  # [V]
    x_bl = torch.round(poses[:, 0:1] - bw.unsqueeze(0) / 2.0).to(torch.long)
    y_bl = torch.round(poses[:, 1:2] - bh.unsqueeze(0) / 2.0).to(torch.long)
    return spec.score_batch(
        gid=gid, x_bl=x_bl, y_bl=y_bl,
        state=env.get_state(),
        reward=env.reward_composer,
    )


def visualize_placeable_map_small():
    """작은 그리드로 conv2d 과정 시각화 (숫자 표시)."""
    # 작은 예시 그리드 (10x12)
    H, W = 10, 12
    kh, kw = 3, 4  # 설비 크기
    
    # 임의의 invalid 맵 생성
    invalid = torch.zeros((H, W), dtype=torch.float32)
    invalid[2:5, 3:6] = 1  # 금지 영역 1
    invalid[6:8, 8:11] = 1  # 금지 영역 2
    
    # conv2d 연산
    inv_f = invalid.view(1, 1, H, W)
    kernel = torch.ones((1, 1, kh, kw), dtype=torch.float32)
    overlap = F.conv2d(inv_f, kernel, padding=0).squeeze()
    placeable = (overlap == 0).int()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # 1. Invalid Map
    ax1 = axes[0]
    im1 = ax1.imshow(invalid.numpy(), cmap='Reds', origin='lower', vmin=0, vmax=1)
    for i in range(H):
        for j in range(W):
            ax1.text(j, i, int(invalid[i, j].item()), ha='center', va='center', fontsize=9)
    ax1.set_title(f'Invalid Map (input)\n{H}x{W}', fontsize=12)
    ax1.set_xticks(range(W))
    ax1.set_yticks(range(H))
    ax1.grid(True, linewidth=0.5)
    
    # 2. Kernel
    ax2 = axes[1]
    kernel_np = kernel.squeeze().numpy()
    im2 = ax2.imshow(kernel_np, cmap='Blues', origin='lower', vmin=0, vmax=1)
    for i in range(kh):
        for j in range(kw):
            ax2.text(j, i, int(kernel_np[i, j]), ha='center', va='center', fontsize=12)
    ax2.set_title(f'Kernel (facility)\n{kh}x{kw}', fontsize=12)
    ax2.set_xticks(range(kw))
    ax2.set_yticks(range(kh))
    ax2.grid(True, linewidth=0.5)
    
    # 3. Placeable Map
    ax3 = axes[2]
    placeable_np = placeable.numpy()
    pH, pW = placeable_np.shape
    im3 = ax3.imshow(placeable_np, cmap='Greens', origin='lower', vmin=0, vmax=1)
    for i in range(pH):
        for j in range(pW):
            ax3.text(j, i, int(placeable_np[i, j]), ha='center', va='center', fontsize=9)
    count = int(placeable.sum().item())
    total = pH * pW
    ax3.set_title(f'Placeable Map (output)\n{pH}x{pW}, valid={count}/{total}', fontsize=12)
    ax3.set_xticks(range(pW))
    ax3.set_yticks(range(pH))
    ax3.grid(True, linewidth=0.5)
    
    plt.suptitle(f'conv2d: Invalid[{H},{W}] ⊛ Kernel[{kh},{kw}] → Placeable[{pH},{pW}]', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('placeable_map_visualization.png', dpi=150)
    plt.show()


def test_placeable_conv2d(env, gid, n_iter=10):
    """conv2d로 placeable 검사 (병렬)."""
    g = env.group_specs[gid]
    kw, kh = int(g.width), int(g.height)
    invalid = env.get_maps().invalid.clone()
    
    # 웜업
    for _ in range(3):
        env._placeable_top_left_count(invalid=invalid, kw=kw, kh=kh)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 측정
    start = time.perf_counter()
    for _ in range(n_iter):
        env._placeable_top_left_count(invalid=invalid, kw=kw, kh=kh)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / n_iter
    
    return elapsed


def test_placeable_loop(env, gid, n_samples=1000):
    """loop로 placeable 검사 (순차) - n_samples개 위치만 검사."""
    g = env.group_specs[gid]
    kw, kh = int(g.width), int(g.height)
    invalid = env.get_maps().invalid
    H, W = invalid.shape
    
    # 검사할 위치들 (랜덤 샘플)
    max_y, max_x = H - kh + 1, W - kw + 1
    positions = [(int(torch.randint(0, max_x, (1,)).item()), 
                  int(torch.randint(0, max_y, (1,)).item())) 
                 for _ in range(n_samples)]
    
    # 측정: 각 위치마다 footprint 영역 검사
    start = time.perf_counter()
    for x, y in positions:
        # footprint 영역이 invalid와 겹치는지 검사
        region = invalid[y:y+kh, x:x+kw]
        is_valid = not region.any().item()
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed, n_samples


def test_placeable_loop_all(env, gid):
    """loop로 전체 placeable 검사 (순차)."""
    g = env.group_specs[gid]
    kw, kh = int(g.width), int(g.height)
    invalid = env.get_maps().invalid
    H, W = invalid.shape
    max_y, max_x = H - kh + 1, W - kw + 1
    total = max_y * max_x
    
    # 전체 위치 검사
    start = time.perf_counter()
    count = 0
    for y in range(max_y):
        for x in range(max_x):
            region = invalid[y:y+kh, x:x+kw]
            if not region.any().item():
                count += 1
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed, total, count


def test_estimate_batch(env, gid, n_calls=100):
    """delta_cost 배치 실행 (병렬)."""
    x = torch.randint(0, env.grid_width, (n_calls,), device=env.device).float()
    y = torch.randint(0, env.grid_height, (n_calls,), device=env.device).float()
    rot = torch.zeros(n_calls, dtype=torch.long, device=env.device)
    
    # 웜업
    for _ in range(5):
        _score_poses(env, gid, x, y, rot)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    _score_poses(env, gid, x, y, rot)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def test_estimate_loop(env, gid, n_calls=100):
    """delta_cost 순차 실행 (loop)."""
    coords = [
        (float(torch.randint(0, env.grid_width, (1,)).item()),
         float(torch.randint(0, env.grid_height, (1,)).item()))
        for _ in range(n_calls)
    ]
    
    # 웜업
    for _ in range(5):
        _score_poses(env, gid, coords[0][0], coords[0][1], 0)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for x, y in coords:
        _score_poses(env, gid, x, y, 0)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def test_placeable_loop_device(env, gid):
    """loop로 전체 placeable 검사 (순차) - device에서 실행."""
    g = env.group_specs[gid]
    kw, kh = int(g.width), int(g.height)
    invalid = env.get_maps().invalid
    H, W = invalid.shape
    max_y, max_x = H - kh + 1, W - kw + 1
    total = max_y * max_x
    
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    count = 0
    for y in range(max_y):
        for x in range(max_x):
            region = invalid[y:y+kh, x:x+kw]
            if not region.any().item():
                count += 1
    
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed, total, count


def main():
    ENV_JSON = "envs/env_configs/basic_01.json"
    N = 100
    
    print("=" * 60)
    print("Placeable 검사: conv2d vs loop / CPU vs GPU 속도 비교")
    print("=" * 60)
    
    # CPU 환경
    loaded_cpu = load_env(ENV_JSON, device=torch.device("cpu"))
    env_cpu = loaded_cpu.env
    env_cpu.reset()
    gid = list(env_cpu.group_specs.keys())[0]
    g = env_cpu.group_specs[gid]
    
    print(f"\n설비: {gid}, 크기: {int(g.width)}x{int(g.height)}")
    print(f"그리드: {env_cpu.grid_width}x{env_cpu.grid_height}")
    
    # CPU 측정
    cpu_conv = test_placeable_conv2d(env_cpu, gid)
    cpu_loop, total, count = test_placeable_loop_device(env_cpu, gid)
    
    # GPU 환경
    gpu_conv, gpu_loop = 0, 0
    if torch.cuda.is_available():
        loaded_gpu = load_env(ENV_JSON, device=torch.device("cuda"))
        env_gpu = loaded_gpu.env
        env_gpu.reset()
        
        # GPU 웜업
        for _ in range(10):
            test_placeable_conv2d(env_gpu, gid)
        torch.cuda.synchronize()
        
        gpu_conv = test_placeable_conv2d(env_gpu, gid)
        gpu_loop, _, _ = test_placeable_loop_device(env_gpu, gid)
    
    # 결과 출력
    print(f"\n[Placeable 검사 속도 비교] (검사 위치: {total}개)")
    print(f"{'':20} {'CPU':>12} {'GPU':>12} {'GPU 속도향상':>12}")
    print("-" * 60)
    print(f"{'conv2d (병렬)':20} {cpu_conv:>10.3f}ms {gpu_conv:>10.3f}ms {cpu_conv/gpu_conv if gpu_conv else 0:>10.1f}x")
    print(f"{'loop (순차)':20} {cpu_loop:>10.3f}ms {gpu_loop:>10.3f}ms {cpu_loop/gpu_loop if gpu_loop else 0:>10.1f}x")
    print("-" * 60)
    print(f"{'conv2d 속도향상 (CPU)':20} {cpu_loop/cpu_conv:>10.1f}x")
    if gpu_conv:
        print(f"{'conv2d 속도향상 (GPU)':20} {gpu_loop/gpu_conv:>10.1f}x")
    print(f"\n배치 가능 위치: {count}개 / {total}개")
    
    # delta_cost 배치 처리
    print(f"\n" + "=" * 60)
    print(f"delta_cost: 배치 vs 순차 / CPU vs GPU 속도 비교")
    print("=" * 60)
    
    cpu_batch = test_estimate_batch(env_cpu, gid, N)
    cpu_loop = test_estimate_loop(env_cpu, gid, N)
    gpu_batch, gpu_loop = 0, 0
    if torch.cuda.is_available():
        gpu_batch = test_estimate_batch(env_gpu, gid, N)
        gpu_loop = test_estimate_loop(env_gpu, gid, N)
    
    print(f"\n[delta_cost 속도 비교] (검사 위치: {N}개)")
    print(f"{'':20} {'CPU':>12} {'GPU':>12} {'GPU 속도향상':>12}")
    print("-" * 60)
    print(f"{'배치 (병렬)':20} {cpu_batch:>10.3f}ms {gpu_batch:>10.3f}ms {cpu_batch/gpu_batch if gpu_batch else 0:>10.1f}x")
    print(f"{'loop (순차)':20} {cpu_loop:>10.3f}ms {gpu_loop:>10.3f}ms {cpu_loop/gpu_loop if gpu_loop else 0:>10.1f}x")
    print("-" * 60)
    print(f"{'배치 속도향상 (CPU)':20} {cpu_loop/cpu_batch:>10.1f}x")
    if gpu_batch:
        print(f"{'배치 속도향상 (GPU)':20} {gpu_loop/gpu_batch:>10.1f}x")
    
    print("\n" + "=" * 60)
    
    # Placeable map 시각화
    print("\n[Placeable Map 시각화 (작은 예시)]")
    visualize_placeable_map_small()
    print("저장됨: placeable_map_visualization.png")


if __name__ == "__main__":
    main()
