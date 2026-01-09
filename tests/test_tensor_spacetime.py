# -*- coding: utf-8 -*-
"""
GPU版时空图性能测试脚本

测试内容:
    1. 基准性能测试（不同并行度）
    2. 与CPU版性能对比
    3. 内存占用测试
    4. 功能正确性验证

使用方法:
    python tests/test_tensor_spacetime.py

测试并行度: 32, 64, 128, 256, 512, 1024
"""

import sys
import os
import time
import torch

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from spacetime.tensor_spacetime import (
    TensorMissileLookup,
    TensorThreatCone,
    TensorTrajectoryPredictor,
    TensorSpacetimeComputer,
)


def test_missile_lookup_performance():
    """测试导弹查找表性能"""
    print("\n" + "=" * 60)
    print("测试1: 导弹查找表批量插值性能")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    lookup = TensorMissileLookup(device=device)
    lookup.load()
    
    # 测试不同批量大小
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    num_iterations = 100
    
    print(f"\n{'批量大小':>10} | {'平均耗时(ms)':>12} | {'吞吐量(查询/秒)':>15}")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # 准备测试数据
        carrier_speeds = torch.rand(batch_size, device=device) * 300 + 200  # 200-500 m/s
        times = torch.rand(batch_size, 4, device=device) * 20  # 0-20秒
        
        # 预热
        for _ in range(10):
            _ = lookup.interpolate_distance_batch(carrier_speeds, times)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 计时测试
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = lookup.interpolate_distance_batch(carrier_speeds, times)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        avg_time = elapsed / num_iterations
        throughput = batch_size * num_iterations / (elapsed / 1000)
        
        print(f"{batch_size:>10} | {avg_time:>12.3f} | {throughput:>15,.0f}")


def test_threat_cone_performance():
    """测试威胁锥计算性能"""
    print("\n" + "=" * 60)
    print("测试2: 威胁锥批量计算性能")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lookup = TensorMissileLookup(device=device)
    lookup.load()
    threat_cone = TensorThreatCone(lookup, device=device)
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    num_iterations = 100
    
    print(f"\n{'批量大小':>10} | {'平均耗时(ms)':>12} | {'吞吐量(锥体/秒)':>15}")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        carrier_speeds = torch.rand(batch_size, device=device) * 300 + 200
        carrier_angles = torch.rand(batch_size, device=device) * 2 * 3.14159
        
        # 预热
        for _ in range(10):
            _ = threat_cone.compute_batch(carrier_speeds, carrier_angles)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = threat_cone.compute_batch(carrier_speeds, carrier_angles)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        avg_time = elapsed / num_iterations
        throughput = batch_size * num_iterations / (elapsed / 1000)
        
        print(f"{batch_size:>10} | {avg_time:>12.3f} | {throughput:>15,.0f}")


def test_trajectory_prediction_performance():
    """测试轨迹预测性能"""
    print("\n" + "=" * 60)
    print("测试3: 轨迹预测批量计算性能")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictor = TensorTrajectoryPredictor(CONFIG, device=device)
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    num_iterations = 50  # 轨迹预测较慢，减少迭代次数
    
    print(f"\n飞机轨迹预测 (20秒, 41步):")
    print(f"{'批量大小':>10} | {'平均耗时(ms)':>12} | {'吞吐量(轨迹/秒)':>15}")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # 初始状态
        x = torch.rand(batch_size, device=device) * 50000
        y = torch.rand(batch_size, device=device) * 50000
        vx = torch.rand(batch_size, device=device) * 200 + 200
        vy = torch.rand(batch_size, device=device) * 200 - 100
        rudder = torch.rand(batch_size, device=device) * 2 - 1
        throttle = torch.ones(batch_size, device=device)
        
        # 预热
        for _ in range(5):
            _ = predictor.predict_aircraft_batch(x, y, vx, vy, rudder, throttle)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = predictor.predict_aircraft_batch(x, y, vx, vy, rudder, throttle)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        avg_time = elapsed / num_iterations
        throughput = batch_size * num_iterations / (elapsed / 1000)
        
        print(f"{batch_size:>10} | {avg_time:>12.3f} | {throughput:>15,.0f}")


def test_full_spacetime_computer_performance():
    """测试完整时空图计算器性能"""
    print("\n" + "=" * 60)
    print("测试4: 完整时空图计算器性能 (update + get_features)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    num_iterations = 50
    
    print(f"\n{'并行环境数':>10} | {'update(ms)':>12} | {'features(ms)':>12} | {'总吞吐(env/s)':>15}")
    print("-" * 60)
    
    for num_envs in batch_sizes:
        computer = TensorSpacetimeComputer(CONFIG, num_envs=num_envs, device=device)
        
        # 模拟TensorEnv的states字典
        states = {
            'x': torch.rand(num_envs, 20, device=device) * 50000,
            'y': torch.rand(num_envs, 20, device=device) * 50000,
            'vx': torch.rand(num_envs, 20, device=device) * 400,
            'vy': torch.rand(num_envs, 20, device=device) * 400 - 200,
            'speed': torch.rand(num_envs, 20, device=device) * 300 + 200,
            'angle': torch.rand(num_envs, 20, device=device) * 360,
            'rudder': torch.rand(num_envs, 20, device=device) * 2 - 1,
            'throttle': torch.ones(num_envs, 20, device=device),
            'alive': torch.ones(num_envs, 20, dtype=torch.bool, device=device),
            'is_active': torch.ones(num_envs, 20, dtype=torch.bool, device=device),
            'is_missile': torch.zeros(num_envs, 20, dtype=torch.bool, device=device),
        }
        
        # 预热
        for i in range(5):
            computer.update(states, step=i * 100, force_update=True)
            _ = computer.get_features(states, player_id=1)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 测试update性能
        start_time = time.perf_counter()
        for i in range(num_iterations):
            computer.update(states, step=i * 100, force_update=True)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        update_elapsed = (time.perf_counter() - start_time) * 1000
        update_avg = update_elapsed / num_iterations
        
        # 测试get_features性能
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = computer.get_features(states, player_id=1)
            _ = computer.get_features(states, player_id=2)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        features_elapsed = (time.perf_counter() - start_time) * 1000
        features_avg = features_elapsed / num_iterations
        
        # 总吞吐量（假设每步需要update + 2次get_features）
        total_time_per_step = update_avg + features_avg
        throughput = num_envs * 1000 / total_time_per_step
        
        print(f"{num_envs:>10} | {update_avg:>12.3f} | {features_avg:>12.3f} | {throughput:>15,.0f}")


def test_memory_usage():
    """测试GPU内存占用"""
    print("\n" + "=" * 60)
    print("测试5: GPU内存占用")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存测试")
        return
    
    device = 'cuda'
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    
    print(f"\n{'并行环境数':>10} | {'内存占用(MB)':>12} | {'每环境(KB)':>12}")
    print("-" * 40)
    
    for num_envs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        computer = TensorSpacetimeComputer(CONFIG, num_envs=num_envs, device=device)
        
        states = {
            'x': torch.rand(num_envs, 20, device=device) * 50000,
            'y': torch.rand(num_envs, 20, device=device) * 50000,
            'vx': torch.rand(num_envs, 20, device=device) * 400,
            'vy': torch.rand(num_envs, 20, device=device) * 400,
            'speed': torch.rand(num_envs, 20, device=device) * 300 + 200,
            'angle': torch.rand(num_envs, 20, device=device) * 360,
            'rudder': torch.rand(num_envs, 20, device=device) * 2 - 1,
            'throttle': torch.ones(num_envs, 20, device=device),
            'alive': torch.ones(num_envs, 20, dtype=torch.bool, device=device),
            'is_active': torch.ones(num_envs, 20, dtype=torch.bool, device=device),
            'is_missile': torch.zeros(num_envs, 20, dtype=torch.bool, device=device),
        }
        
        computer.update(states, step=100, force_update=True)
        _ = computer.get_features(states, player_id=1)
        
        torch.cuda.synchronize()
        
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024
        used_memory = final_memory - initial_memory
        per_env_kb = used_memory * 1024 / num_envs
        
        print(f"{num_envs:>10} | {used_memory:>12.2f} | {per_env_kb:>12.2f}")


def test_correctness():
    """验证功能正确性"""
    print("\n" + "=" * 60)
    print("测试6: 功能正确性验证")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 4
    
    computer = TensorSpacetimeComputer(CONFIG, num_envs=num_envs, device=device)
    
    # 构造已知状态
    states = {
        'x': torch.tensor([[10000, 40000] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'y': torch.tensor([[25000, 25000] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'vx': torch.tensor([[300, -300] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'vy': torch.tensor([[0, 0] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'speed': torch.tensor([[300, 300] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'angle': torch.tensor([[0, 180] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'rudder': torch.tensor([[0, 0] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'throttle': torch.tensor([[1, 1] + [0]*18]*num_envs, dtype=torch.float32, device=device),
        'alive': torch.tensor([[True, True] + [False]*18]*num_envs, dtype=torch.bool, device=device),
        'is_active': torch.tensor([[True, True] + [False]*18]*num_envs, dtype=torch.bool, device=device),
        'is_missile': torch.tensor([[False, False] + [False]*18]*num_envs, dtype=torch.bool, device=device),
    }
    
    # 更新
    computer.update(states, step=100, force_update=True)
    
    # 获取特征
    p1_features = computer.get_features(states, player_id=1)
    p2_features = computer.get_features(states, player_id=2)
    
    print(f"\nP1特征形状: {p1_features.shape}")
    print(f"P2特征形状: {p2_features.shape}")
    print(f"P1特征范围: [{p1_features.min().item():.4f}, {p1_features.max().item():.4f}]")
    print(f"P2特征范围: [{p2_features.min().item():.4f}, {p2_features.max().item():.4f}]")
    
    # 验证威胁度量
    threat_metrics = computer.get_threat_metrics(states)
    print(f"\nP1在威胁锥内: {threat_metrics['p1_in_threat']}")
    print(f"P2在威胁锥内: {threat_metrics['p2_in_threat']}")
    print(f"P1威胁比例: {threat_metrics['p1_threat_ratio']}")
    print(f"P2威胁比例: {threat_metrics['p2_threat_ratio']}")
    
    print("\n功能验证通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("GPU版时空图性能测试")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: CUDA不可用，将使用CPU测试")
    
    # 运行测试
    test_missile_lookup_performance()
    test_threat_cone_performance()
    test_trajectory_prediction_performance()
    test_full_spacetime_computer_performance()
    test_memory_usage()
    test_correctness()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
