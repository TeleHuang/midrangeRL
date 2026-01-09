# -*- coding: utf-8 -*-
"""
torch.compile 性能对比测试

对比启用/不启用 torch.compile 的性能差异
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_gym.tensor_env import TensorEnv
from config import CONFIG


def run_benchmark(env, num_steps=500, warmup_steps=50, label=""):
    """运行性能基准测试"""
    device = env._device
    num_envs = env.num_envs
    dt = 1.0 / 60.0
    
    # 预分配动作
    actions = {
        'p1_rudder': torch.rand(num_envs, device=device) * 2 - 1,
        'p1_throttle': torch.ones(num_envs, device=device),
        'p1_fire': torch.zeros(num_envs, dtype=torch.bool, device=device),
        'p2_rudder': torch.rand(num_envs, device=device) * 2 - 1,
        'p2_throttle': torch.ones(num_envs, device=device),
        'p2_fire': torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    
    env.reset()
    
    # 预热阶段
    print(f"  [{label}] Warmup {warmup_steps} steps...")
    for _ in range(warmup_steps):
        env.step(actions, dt=dt)
    torch.cuda.synchronize()
    
    # 重置后正式测试
    env.reset()
    
    # 正式测试
    print(f"  [{label}] Running {num_steps} steps...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_steps):
        env.step(actions, dt=dt)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    total_env_steps = num_steps * num_envs
    throughput = total_env_steps / elapsed
    
    return {
        'elapsed': elapsed,
        'throughput': throughput,
        'steps': num_steps,
        'env_steps': total_env_steps,
    }


def main():
    print("=" * 60)
    print("torch.compile 性能对比测试")
    print("=" * 60)
    
    # 检查 PyTorch 版本
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 测试参数
    num_envs = 1024
    num_steps = 500
    warmup_steps = 100  # compile版本需要更多预热
    
    print(f"\n测试配置: {num_envs} 并行环境, {num_steps} 步")
    
    # ========== 测试1: 不启用 torch.compile ==========
    print("\n" + "-" * 40)
    print("测试1: 不启用 torch.compile")
    print("-" * 40)
    
    config_no_compile = CONFIG.copy()
    config_no_compile['use_torch_compile'] = False
    
    env_no_compile = TensorEnv(config_no_compile, num_envs=num_envs, device='cuda')
    result_no_compile = run_benchmark(
        env_no_compile, 
        num_steps=num_steps, 
        warmup_steps=50,
        label="No Compile"
    )
    
    print(f"  耗时: {result_no_compile['elapsed']:.3f}s")
    print(f"  吞吐: {result_no_compile['throughput']:,.0f} env-steps/s")
    
    del env_no_compile
    torch.cuda.empty_cache()
    
    # ========== 测试2: 启用 torch.compile ==========
    print("\n" + "-" * 40)
    print("测试2: 启用 torch.compile")
    print("-" * 40)
    
    config_compile = CONFIG.copy()
    config_compile['use_torch_compile'] = True
    
    env_compile = TensorEnv(config_compile, num_envs=num_envs, device='cuda')
    result_compile = run_benchmark(
        env_compile, 
        num_steps=num_steps, 
        warmup_steps=warmup_steps,  # 更多预热用于编译
        label="Compile"
    )
    
    print(f"  耗时: {result_compile['elapsed']:.3f}s")
    print(f"  吞吐: {result_compile['throughput']:,.0f} env-steps/s")
    
    # ========== 结果对比 ==========
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    
    speedup = result_compile['throughput'] / result_no_compile['throughput']
    improvement = (speedup - 1) * 100
    
    print(f"\n{'配置':<20} {'吞吐 (env-steps/s)':<25} {'相对性能':<15}")
    print("-" * 60)
    print(f"{'无 compile':<20} {result_no_compile['throughput']:>20,.0f} {'1.00x (基准)':>15}")
    print(f"{'启用 compile':<20} {result_compile['throughput']:>20,.0f} {f'{speedup:.2f}x':>15}")
    
    print(f"\n性能变化: {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    if improvement > 5:
        print("结论: torch.compile 带来显著性能提升，建议在长时间训练中启用")
    elif improvement > 0:
        print("结论: torch.compile 略有提升，可选择性启用")
    else:
        print("结论: torch.compile 未带来提升，建议保持默认（不启用）")
    
    # 实时性分析
    game_fps = 60
    realtime_threshold = game_fps * num_envs
    
    print(f"\n实时性分析:")
    print(f"  实时阈值: {realtime_threshold:,} env-steps/s (60fps × {num_envs} envs)")
    print(f"  无compile: {result_no_compile['throughput']/realtime_threshold:.2f}x 实时")
    print(f"  启用compile: {result_compile['throughput']/realtime_threshold:.2f}x 实时")


if __name__ == "__main__":
    main()
