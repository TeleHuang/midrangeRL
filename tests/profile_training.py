# -*- coding: utf-8 -*-
"""
GPU训练流程性能分析脚本

使用PyTorch Profiler对完整训练流程进行性能分析，定位瓶颈。

分析内容:
    1. TensorEnv.step() - 环境步进（物理+碰撞检测）
    2. Agent.act() - 智能体决策
    3. 时空图计算（如果启用）
    4. CPU-GPU同步开销

输出:
    - 控制台性能摘要
    - Chrome trace文件（可用chrome://tracing查看）
    - 表格化分析报告

使用方法:
    python tests/profile_training.py --num-envs 256 --steps 100
    
查看trace:
    打开 chrome://tracing 或 edge://tracing
    加载生成的 .json 文件
"""

import argparse
import sys
import os
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from env_gym import TensorEnv
from agents.rule_based.rule_agent import CrankAgent


def parse_args():
    parser = argparse.ArgumentParser(description='GPU训练流程性能分析')
    parser.add_argument('--num-envs', '-n', type=int, default=256, help='并行环境数')
    parser.add_argument('--steps', '-s', type=int, default=200, help='分析步数')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='设备')
    parser.add_argument('--time-scale', '-t', type=float, default=1.0, help='时间加速')
    parser.add_argument('--output', '-o', type=str, default='profile_trace', help='输出文件前缀')
    parser.add_argument('--with-spacetime', action='store_true', help='是否启用时空图')
    parser.add_argument('--warmup', type=int, default=10, help='预热步数')
    parser.add_argument('--detailed', action='store_true', help='详细输出每个操作')
    return parser.parse_args()


def run_profiled_training(args):
    """运行带Profiler的训练流程"""
    
    device = args.device
    num_envs = args.num_envs
    dt = (1.0 / 60.0) * args.time_scale
    
    print("=" * 70)
    print("GPU训练流程性能分析")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"并行环境数: {num_envs}")
    print(f"分析步数: {args.steps}")
    print(f"时间加速: {args.time_scale}x")
    print(f"启用时空图: {args.with_spacetime}")
    print()
    
    # 创建环境
    config = CONFIG.copy()
    env = TensorEnv(config, num_envs=num_envs, device=device)
    
    # 创建Agent（双方都用CrankAgent）
    agent_p1 = CrankAgent(device=device)
    agent_p2 = CrankAgent(device=device)
    
    # 可选：创建时空图计算器
    spacetime_computer = None
    if args.with_spacetime:
        from spacetime.tensor_spacetime import TensorSpacetimeComputer
        spacetime_computer = TensorSpacetimeComputer(config, num_envs=num_envs, device=device)
        print("已创建TensorSpacetimeComputer")
    
    # 初始化
    obs = env.reset()
    obs_p1 = obs['p1']
    obs_p2 = obs['p2']
    agent_p1.reset()
    agent_p2.reset()
    
    # ========== 预热阶段 ==========
    print(f"\n预热 {args.warmup} 步...")
    for _ in range(args.warmup):
        action_p1 = agent_p1.act(obs_p1, dt=dt)
        action_p2 = agent_p2.act(obs_p2, dt=dt)
        
        full_action = {
            'p1_rudder': action_p1['rudder'],
            'p1_throttle': action_p1['throttle'],
            'p1_fire': action_p1['fire'],
            'p2_rudder': action_p2['rudder'],
            'p2_throttle': action_p2['throttle'],
            'p2_fire': action_p2['fire'],
        }
        
        next_obs, rewards, dones, infos = env.step(full_action, dt=dt)
        
        if spacetime_computer:
            spacetime_computer.update(env.states, step=_, force_update=True)
            _ = spacetime_computer.get_features(env.states, player_id=1)
        
        if dones.any():
            env.reset(env_mask=dones)
            agent_p1.reset(env_mask=dones)
            agent_p2.reset(env_mask=dones)
        
        obs_p1 = next_obs['p1']
        obs_p2 = next_obs['p2']
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("预热完成")
    
    # ========== 性能计时（无Profiler开销）==========
    print(f"\n性能计时 {args.steps} 步...")
    
    timing = {
        'agent_p1': [],
        'agent_p2': [],
        'env_step': [],
        'spacetime_update': [],
        'spacetime_features': [],
        'reset': [],
        'total': [],
    }
    
    for step in range(args.steps):
        step_start = time.perf_counter()
        
        # Agent P1 决策
        t0 = time.perf_counter()
        action_p1 = agent_p1.act(obs_p1, dt=dt)
        if device == 'cuda':
            torch.cuda.synchronize()
        timing['agent_p1'].append(time.perf_counter() - t0)
        
        # Agent P2 决策
        t0 = time.perf_counter()
        action_p2 = agent_p2.act(obs_p2, dt=dt)
        if device == 'cuda':
            torch.cuda.synchronize()
        timing['agent_p2'].append(time.perf_counter() - t0)
        
        # 构造动作
        full_action = {
            'p1_rudder': action_p1['rudder'],
            'p1_throttle': action_p1['throttle'],
            'p1_fire': action_p1['fire'],
            'p2_rudder': action_p2['rudder'],
            'p2_throttle': action_p2['throttle'],
            'p2_fire': action_p2['fire'],
        }
        
        # 环境步进
        t0 = time.perf_counter()
        next_obs, rewards, dones, infos = env.step(full_action, dt=dt)
        if device == 'cuda':
            torch.cuda.synchronize()
        timing['env_step'].append(time.perf_counter() - t0)
        
        # 时空图计算
        if spacetime_computer:
            t0 = time.perf_counter()
            spacetime_computer.update(env.states, step=step, force_update=(step % 30 == 0))
            if device == 'cuda':
                torch.cuda.synchronize()
            timing['spacetime_update'].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            _ = spacetime_computer.get_features(env.states, player_id=1)
            _ = spacetime_computer.get_features(env.states, player_id=2)
            if device == 'cuda':
                torch.cuda.synchronize()
            timing['spacetime_features'].append(time.perf_counter() - t0)
        
        # 处理结束
        if dones.any():
            t0 = time.perf_counter()
            env.reset(env_mask=dones)
            agent_p1.reset(env_mask=dones)
            agent_p2.reset(env_mask=dones)
            if device == 'cuda':
                torch.cuda.synchronize()
            timing['reset'].append(time.perf_counter() - t0)
        
        obs_p1 = next_obs['p1']
        obs_p2 = next_obs['p2']
        
        timing['total'].append(time.perf_counter() - step_start)
    
    # ========== 输出计时结果 ==========
    print("\n" + "=" * 70)
    print("性能计时结果（毫秒）")
    print("=" * 70)
    print(f"{'操作':<25} {'平均':>10} {'最小':>10} {'最大':>10} {'占比':>10}")
    print("-" * 70)
    
    total_avg = sum(timing['total']) / len(timing['total']) * 1000
    
    for name, times in timing.items():
        if times:
            avg = sum(times) / len(times) * 1000
            min_t = min(times) * 1000
            max_t = max(times) * 1000
            pct = avg / total_avg * 100 if total_avg > 0 else 0
            print(f"{name:<25} {avg:>10.3f} {min_t:>10.3f} {max_t:>10.3f} {pct:>9.1f}%")
    
    # 计算吞吐量
    total_time = sum(timing['total'])
    throughput = num_envs * args.steps / total_time
    steps_per_sec = args.steps / total_time
    
    print("-" * 70)
    print(f"总耗时: {total_time:.3f} 秒")
    print(f"步进速率: {steps_per_sec:.1f} steps/s")
    print(f"环境吞吐: {throughput:,.0f} env-steps/s")
    print(f"单环境每秒: {throughput / num_envs:.1f} steps/s (约{throughput / num_envs / 60:.1f}倍实时)")
    
    # ========== PyTorch Profiler详细分析 ==========
    print(f"\n{'='*70}")
    print("PyTorch Profiler 详细分析")
    print("=" * 70)
    
    # 重置环境
    obs = env.reset()
    obs_p1 = obs['p1']
    obs_p2 = obs['p2']
    agent_p1.reset()
    agent_p2.reset()
    if spacetime_computer:
        spacetime_computer.reset()
    
    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    profile_steps = min(50, args.steps)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.detailed,
    ) as prof:
        for step in range(profile_steps):
            with record_function("agent_p1_act"):
                action_p1 = agent_p1.act(obs_p1, dt=dt)
            
            with record_function("agent_p2_act"):
                action_p2 = agent_p2.act(obs_p2, dt=dt)
            
            full_action = {
                'p1_rudder': action_p1['rudder'],
                'p1_throttle': action_p1['throttle'],
                'p1_fire': action_p1['fire'],
                'p2_rudder': action_p2['rudder'],
                'p2_throttle': action_p2['throttle'],
                'p2_fire': action_p2['fire'],
            }
            
            with record_function("env_step"):
                next_obs, rewards, dones, infos = env.step(full_action, dt=dt)
            
            if spacetime_computer:
                with record_function("spacetime_update"):
                    spacetime_computer.update(env.states, step=step, force_update=(step % 30 == 0))
                
                with record_function("spacetime_features"):
                    _ = spacetime_computer.get_features(env.states, player_id=1)
                    _ = spacetime_computer.get_features(env.states, player_id=2)
            
            if dones.any():
                with record_function("reset"):
                    env.reset(env_mask=dones)
                    agent_p1.reset(env_mask=dones)
                    agent_p2.reset(env_mask=dones)
            
            obs_p1 = next_obs['p1']
            obs_p2 = next_obs['p2']
    
    # 输出Profiler结果
    print("\nCPU时间 Top 20 操作:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device == 'cuda':
        print("\nCUDA时间 Top 20 操作:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 导出trace文件
    trace_path = f"{args.output}_{num_envs}envs.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace已导出: {trace_path}")
    print("可用 chrome://tracing 或 edge://tracing 查看")
    
    # 内存使用
    if device == 'cuda':
        print(f"\n{'='*70}")
        print("GPU内存使用")
        print("=" * 70)
        print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print(f"峰值: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    env.close()
    print("\n分析完成!")


def main():
    args = parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("警告: CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    run_profiled_training(args)


if __name__ == '__main__':
    main()
