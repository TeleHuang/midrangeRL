# -*- coding: utf-8 -*-
"""
Actor-Critic 神经网络定义
用于 PPO 智能体的策略网络和价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from typing import Tuple, Dict
import math


class ActorNetwork(nn.Module):
    """Actor 网络（策略网络）
    
    输入：10维观察特征
    输出：
        - rudder: 均值和对数标准差（连续动作，使用高斯分布+tanh）
        - fire: 开火概率（离散动作，使用伯努利分布）
    """
    
    def __init__(self, obs_dim: int = 10, hidden_dim1: int = 128, hidden_dim2: int = 64):
        """初始化 Actor 网络
        
        Args:
            obs_dim: 观察空间维度
            hidden_dim1: 第一隐藏层维度
            hidden_dim2: 第二隐藏层维度
        """
        super().__init__()
        
        # 共享特征提取层
        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 方向舵输出（均值和对数标准差）
        self.rudder_mean = nn.Linear(hidden_dim2, 1)
        self.rudder_log_std = nn.Linear(hidden_dim2, 1)
        
        # 开火输出（概率）
        self.fire_prob = nn.Linear(hidden_dim2, 1)
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        # 使用正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 策略输出层使用较小的初始化
        nn.init.orthogonal_(self.rudder_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.rudder_log_std.weight, gain=0.01)
        nn.init.orthogonal_(self.fire_prob.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: 观察张量 [batch_size, obs_dim]
        
        Returns:
            rudder_mean: 方向舵均值 [batch_size, 1]
            rudder_log_std: 方向舵对数标准差 [batch_size, 1]
            fire_prob: 开火概率 [batch_size, 1]
        """
        # 特征提取
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        # 方向舵输出
        rudder_mean = self.rudder_mean(x)
        rudder_log_std = self.rudder_log_std(x)
        # 限制log_std范围，防止数值不稳定
        rudder_log_std = torch.clamp(rudder_log_std, min=-20, max=2)
        
        # 开火概率输出
        fire_prob = torch.sigmoid(self.fire_prob(x))
        
        return rudder_mean, rudder_log_std, fire_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """获取动作
        
        Args:
            obs: 观察张量 [batch_size, obs_dim]
            deterministic: 是否使用确定性策略（评估模式）
        
        Returns:
            action_dict: 包含 rudder, fire, log_prob 的字典
        """
        rudder_mean, rudder_log_std, fire_prob = self.forward(obs)
        
        if deterministic:
            # 确定性策略：使用均值和阈值
            rudder_action = torch.tanh(rudder_mean.squeeze(-1))
            fire_action = (fire_prob.squeeze(-1) > 0.5)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            # 随机策略：采样
            # 方向舵：使用高斯分布 + tanh 压缩
            rudder_std = torch.exp(rudder_log_std)
            rudder_dist = Normal(rudder_mean, rudder_std)
            rudder_sample = rudder_dist.rsample()  # 使用重参数化技巧
            rudder_action = torch.tanh(rudder_sample).squeeze(-1)
            
            # 计算 log_prob（考虑 tanh 变换）
            rudder_log_prob = rudder_dist.log_prob(rudder_sample).squeeze(-1)
            # tanh 变换的雅可比修正
            rudder_log_prob -= torch.log(1 - rudder_action.pow(2) + 1e-6)
            
            # 开火：使用伯努利分布
            fire_dist = Bernoulli(fire_prob.squeeze(-1))
            fire_action = fire_dist.sample().bool()
            fire_log_prob = fire_dist.log_prob(fire_action.float())
            
            # 总log_prob（假设动作独立）
            log_prob = rudder_log_prob + fire_log_prob
        
        return {
            'rudder': rudder_action,
            'fire': fire_action,
            'log_prob': log_prob
        }
    
    def evaluate_actions(self, obs: torch.Tensor, rudder_actions: torch.Tensor, 
                        fire_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率和熵
        
        Args:
            obs: 观察张量 [batch_size, obs_dim]
            rudder_actions: 方向舵动作 [batch_size]
            fire_actions: 开火动作 [batch_size]
        
        Returns:
            log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
        """
        rudder_mean, rudder_log_std, fire_prob = self.forward(obs)
        
        # 方向舵：反向tanh变换以获取原始采样值
        rudder_actions_clamped = torch.clamp(rudder_actions, -0.999, 0.999)
        rudder_sample = torch.atanh(rudder_actions_clamped).unsqueeze(-1)
        
        rudder_std = torch.exp(rudder_log_std)
        rudder_dist = Normal(rudder_mean, rudder_std)
        rudder_log_prob = rudder_dist.log_prob(rudder_sample).squeeze(-1)
        # tanh 变换的雅可比修正
        rudder_log_prob -= torch.log(1 - rudder_actions_clamped.pow(2) + 1e-6)
        
        # 方向舵熵（高斯分布）
        rudder_entropy = rudder_dist.entropy().squeeze(-1)
        
        # 开火：伯努利分布
        fire_dist = Bernoulli(fire_prob.squeeze(-1))
        fire_log_prob = fire_dist.log_prob(fire_actions.float())
        fire_entropy = fire_dist.entropy()
        
        # 总log_prob和熵
        log_probs = rudder_log_prob + fire_log_prob
        entropy = rudder_entropy + fire_entropy
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic 网络（价值网络）
    
    输入：10维观察特征
    输出：状态价值 V(s)
    """
    
    def __init__(self, obs_dim: int = 10, hidden_dim1: int = 128, hidden_dim2: int = 64):
        """初始化 Critic 网络
        
        Args:
            obs_dim: 观察空间维度
            hidden_dim1: 第一隐藏层维度
            hidden_dim2: 第二隐藏层维度
        """
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.value_head = nn.Linear(hidden_dim2, 1)
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 价值输出层使用较小的初始化
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            obs: 观察张量 [batch_size, obs_dim]
        
        Returns:
            value: 状态价值 [batch_size]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.value_head(x).squeeze(-1)
        return value
