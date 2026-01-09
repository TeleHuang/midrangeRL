# -*- coding: utf-8 -*-
"""
Gymnasium 兼容的包装器
将 TensorEnv 包装成标准的 Gymnasium 环境接口
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional
from env_gym.tensor_env import TensorEnv


class MidrangeRLEnv(gym.Env):
    """Gymnasium 环境包装器"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, config: Optional[Dict] = None, num_envs: int = 1, device: str = 'cuda'):
        """初始化环境
        
        Args:
            config: 配置字典
            num_envs: 并行环境数量（注意：Gymnasium通常期望单环境，这里保留多环境支持用于内部优化）
            device: 计算设备
        """
        super().__init__()
        
        if config is None:
            from config import CONFIG
            config = CONFIG.copy()
        
        self.config = config
        self.num_envs = num_envs
        self.device = device
        
        # 创建底层环境
        self.env = TensorEnv(config, num_envs=num_envs, device=device)
        
        # 定义观察空间和动作空间
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
        # 用于单环境模式的当前环境索引
        self.current_env_idx = 0
        
    def _define_observation_space(self):
        """定义观察空间"""
        # 基于 TensorEnv 的观察空间结构
        return spaces.Dict({
            'x': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'y': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'angle': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'speed': spaces.Box(low=0.0, high=2.0, shape=(self.num_envs,), dtype=np.float32),
            'missiles': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'alive': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'enemy_distance': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'enemy_relative_angle': spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'enemy_speed': spaces.Box(low=0.0, high=2.0, shape=(self.num_envs,), dtype=np.float32),
            'enemy_alive': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
        })
    
    def _define_action_space(self):
        """定义动作空间"""
        # 每个玩家的动作：rudder, throttle, fire
        return spaces.Dict({
            'p1_rudder': spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'p1_throttle': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'p1_fire': spaces.MultiBinary(self.num_envs),
            'p2_rudder': spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'p2_throttle': spaces.Box(low=0.0, high=1.0, shape=(self.num_envs,), dtype=np.float32),
            'p2_fire': spaces.MultiBinary(self.num_envs),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # 重置底层环境
        observations = self.env.reset()
        
        # 转换为numpy数组（Gymnasium标准）
        obs_np = self._tensor_to_numpy(observations)
        
        info = {
            'p1_alive': self.env.states['alive'][:, self.env.P1_IDX].cpu().numpy(),
            'p2_alive': self.env.states['alive'][:, self.env.P2_IDX].cpu().numpy(),
        }
        
        return obs_np, info
    
    def step(self, action: Dict[str, np.ndarray]):
        """执行一步"""
        # 将numpy动作转换为tensor
        action_tensor = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action_tensor[key] = torch.from_numpy(value).to(self.device)
            else:
                action_tensor[key] = torch.tensor(value, device=self.device)
        
        # 执行步骤
        observations, rewards, dones, infos = self.env.step(action_tensor)
        
        # 转换为numpy
        obs_np = self._tensor_to_numpy(observations)
        rewards_np = {k: v.cpu().numpy() for k, v in rewards.items()}
        dones_np = dones.cpu().numpy()
        
        # 构造info
        info = {
            'winner': self.env.winner.cpu().numpy(),
            'p1_alive': self.env.states['alive'][:, self.env.P1_IDX].cpu().numpy(),
            'p2_alive': self.env.states['alive'][:, self.env.P2_IDX].cpu().numpy(),
        }
        
        return obs_np, rewards_np, dones_np, False, info
    
    def _tensor_to_numpy(self, observations: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """将tensor观察转换为numpy数组"""
        # 合并p1和p2的观察（或者返回其中一个，取决于使用场景）
        # 这里返回p1的观察作为主要观察
        obs_p1 = observations['p1']
        obs_np = {}
        for key, value in obs_p1.items():
            obs_np[key] = value.cpu().numpy()
        return obs_np
    
    def render(self):
        """渲染环境（可选实现）"""
        # TODO: 实现可视化
        pass
    
    def close(self):
        """关闭环境"""
        self.env.close()


class VectorizedMidrangeRLEnv:
    """向量化环境包装器（用于多环境并行训练）"""
    
    def __init__(self, config: Optional[Dict] = None, num_envs: int = 32, device: str = 'cuda'):
        """初始化向量化环境
        
        Args:
            config: 配置字典
            num_envs: 并行环境数量
            device: 计算设备
        """
        if config is None:
            from config import CONFIG
            config = CONFIG.copy()
        
        self.config = config
        self.num_envs = num_envs
        self.device = device
        
        # 创建底层环境
        self.env = TensorEnv(config, num_envs=num_envs, device=device)
        
    def reset(self, env_mask: Optional[torch.Tensor] = None):
        """重置环境"""
        return self.env.reset(env_mask)
    
    def step(self, actions: Dict[str, torch.Tensor]):
        """执行步骤"""
        return self.env.step(actions)
    
    def get_observations(self):
        """获取观察"""
        return self.env.get_observations()
    
    def close(self):
        """关闭环境"""
        self.env.close()

