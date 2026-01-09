# -*- coding: utf-8 -*-
"""
TensorEnv 测试脚本
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_gym.tensor_env import TensorEnv
from env_gym.config_manager import ConfigManager


def test_single_env():
    """测试单环境"""
    print("=" * 50)
    print("测试单环境")
    print("=" * 50)
    
    config = ConfigManager.DEFAULT_CONFIG.copy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    env = TensorEnv(config, num_envs=1, device=device)
    
    # 测试重置
    obs = env.reset()
    print(f"重置成功，观察空间键: {obs.keys()}")
    print(f"P1观察形状: {obs['p1']['x'].shape}")
    
    # 测试步骤
    actions = {
        'p1_rudder': torch.tensor([0.5], device=device),
        'p1_throttle': torch.tensor([1.0], device=device),
        'p1_fire': torch.tensor([False], device=device),
        'p2_rudder': torch.tensor([-0.5], device=device),
        'p2_throttle': torch.tensor([1.0], device=device),
        'p2_fire': torch.tensor([False], device=device),
    }
    
    obs, rewards, dones, infos = env.step(actions)
    print(f"步骤执行成功")
    print(f"奖励: P1={rewards['p1'].item():.2f}, P2={rewards['p2'].item():.2f}")
    print(f"完成: {dones.item()}")
    
    print("单环境测试通过！\n")


def test_multi_env():
    """测试多环境并行"""
    print("=" * 50)
    print("测试多环境并行")
    print("=" * 50)
    
    config = ConfigManager.DEFAULT_CONFIG.copy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 4
    
    env = TensorEnv(config, num_envs=num_envs, device=device)
    
    # 测试重置
    obs = env.reset()
    print(f"重置成功，环境数量: {num_envs}")
    print(f"P1观察形状: {obs['p1']['x'].shape}")
    
    # 测试步骤
    actions = {
        'p1_rudder': torch.randn(num_envs, device=device) * 0.5,
        'p1_throttle': torch.ones(num_envs, device=device),
        'p1_fire': torch.zeros(num_envs, dtype=torch.bool, device=device),
        'p2_rudder': torch.randn(num_envs, device=device) * 0.5,
        'p2_throttle': torch.ones(num_envs, device=device),
        'p2_fire': torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    
    for step in range(10):
        obs, rewards, dones, infos = env.step(actions)
        if dones.any():
            print(f"步骤 {step}: {dones.sum().item()} 个环境完成")
            # 重置完成的环境
            env.reset(dones)
    
    print("多环境测试通过！\n")


def test_firing():
    """测试导弹发射"""
    print("=" * 50)
    print("测试导弹发射")
    print("=" * 50)
    
    config = ConfigManager.DEFAULT_CONFIG.copy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = TensorEnv(config, num_envs=1, device=device)
    obs = env.reset()
    
    # 发射导弹
    actions = {
        'p1_rudder': torch.tensor([0.0], device=device),
        'p1_throttle': torch.tensor([1.0], device=device),
        'p1_fire': torch.tensor([True], device=device),
        'p2_rudder': torch.tensor([0.0], device=device),
        'p2_throttle': torch.tensor([1.0], device=device),
        'p2_fire': torch.tensor([False], device=device),
    }
    
    obs, rewards, dones, infos = env.step(actions)
    
    # 检查导弹是否创建
    active_missiles = (env.states['is_missile'] & env.states['is_active']).sum()
    print(f"激活的导弹数量: {active_missiles.item()}")
    print(f"P1剩余导弹: {env.states['missile_count'][0, env.P1_IDX].item()}")
    
    assert active_missiles.item() > 0, "导弹应该被创建"
    assert env.states['missile_count'][0, env.P1_IDX].item() == 5, "导弹数量应该减少"
    
    print("导弹发射测试通过！\n")


if __name__ == '__main__':
    print("开始测试 TensorEnv\n")
    
    try:
        test_single_env()
        test_multi_env()
        test_firing()
        print("=" * 50)
        print("所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

