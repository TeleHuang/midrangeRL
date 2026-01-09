# -*- coding: utf-8 -*-
"""
PPO 智能体实现
包含 RolloutBuffer 和 PPOAgent 类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np

from agents.base_agent import BaseAgent
from agents.learned.networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """经验缓冲区，用于存储轨迹数据"""
    
    def __init__(self, buffer_size: int, num_envs: int, obs_dim: int, device: torch.device):
        """初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小（时间步数）
            num_envs: 并行环境数量
            obs_dim: 观察空间维度
            device: 计算设备
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        
        # 初始化存储
        self.observations = torch.zeros((buffer_size, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions_rudder = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.actions_fire = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=device)
        
        # GAE 计算结果
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        
        self.pos = 0
        self.full = False
    
    def add(self, obs: torch.Tensor, action_rudder: torch.Tensor, action_fire: torch.Tensor,
            log_prob: torch.Tensor, reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        """添加一步经验
        
        Args:
            obs: 观察 [num_envs, obs_dim]
            action_rudder: 方向舵动作 [num_envs]
            action_fire: 开火动作 [num_envs]
            log_prob: 动作对数概率 [num_envs]
            reward: 奖励 [num_envs]
            value: 状态价值 [num_envs]
            done: 终止标志 [num_envs]
        """
        self.observations[self.pos] = obs
        self.actions_rudder[self.pos] = action_rudder
        self.actions_fire[self.pos] = action_fire
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float = 0.99, 
                                      gae_lambda: float = 0.95):
        """计算 GAE 和目标回报
        
        Args:
            last_values: 最后一步的状态价值 [num_envs]
            gamma: 折扣因子
            gae_lambda: GAE 参数
        """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = ~self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = ~self.dones[step]
                next_values = self.values[step + 1]
            
            # TD 误差
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            
            # GAE
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # 目标回报
        self.returns = self.advantages + self.values
    
    def get_samples(self, batch_size: int):
        """获取打乱的训练批次
        
        Args:
            batch_size: 批次大小
        
        Yields:
            batch_dict: 包含批次数据的字典
        """
        # 展平数据
        total_samples = self.buffer_size * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)
        
        # 分批次
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # 计算原始索引
            time_indices = batch_indices // self.num_envs
            env_indices = batch_indices % self.num_envs
            
            yield {
                'observations': self.observations[time_indices, env_indices],
                'actions_rudder': self.actions_rudder[time_indices, env_indices],
                'actions_fire': self.actions_fire[time_indices, env_indices],
                'old_log_probs': self.log_probs[time_indices, env_indices],
                'advantages': self.advantages[time_indices, env_indices],
                'returns': self.returns[time_indices, env_indices],
            }
    
    def clear(self):
        """清空缓冲区"""
        self.pos = 0
        self.full = False


class PPOAgent(BaseAgent):
    """PPO 智能体
    
    实现基于 PPO-Clip 算法的强化学习智能体
    """
    
    def __init__(self, device: str = 'cuda', learning_rate: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 n_steps: int = 2048, batch_size: int = 64, n_epochs: int = 10):
        """初始化 PPO 智能体
        
        Args:
            device: 计算设备
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            clip_epsilon: PPO 裁剪系数
            value_loss_coef: 价值损失权重
            entropy_coef: 熵正则化权重
            max_grad_norm: 梯度裁剪阈值
            n_steps: 每次更新收集的步数
            batch_size: mini-batch 大小
            n_epochs: 每次更新的 epoch 数
        """
        super().__init__(device)
        
        # 超参数
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # 网络
        self.obs_dim = 10  # 根据设计文档
        self.actor = ActorNetwork(obs_dim=self.obs_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim=self.obs_dim).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # 经验缓冲区（延迟初始化）
        self.rollout_buffer = None
        
        # 训练统计
        self.training_mode = True
    
    def _ensure_buffer_initialized(self):
        """确保缓冲区已初始化"""
        if self.rollout_buffer is None:
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.n_steps,
                num_envs=self.num_envs,
                obs_dim=self.obs_dim,
                device=self.device
            )
    
    def _obs_dict_to_tensor(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将观察字典转换为张量
        
        Args:
            observation: 观察字典
        
        Returns:
            obs_tensor: 观察张量 [num_envs, obs_dim]
        """
        # 按顺序提取特征
        keys = ['x', 'y', 'angle', 'speed', 'missiles', 'alive', 
                'enemy_distance', 'enemy_relative_angle', 'enemy_speed', 'enemy_alive']
        obs_list = [observation[key] for key in keys]
        obs_tensor = torch.stack(obs_list, dim=-1)
        return obs_tensor
    
    def act(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """根据观察返回动作
        
        Args:
            observation: 观察字典
        
        Returns:
            动作字典：{rudder, throttle, fire}
        """
        # 转换观察
        obs_tensor = self._obs_dict_to_tensor(observation)
        
        # 获取动作
        with torch.no_grad():
            action_dict = self.actor.get_action(obs_tensor, deterministic=not self.training_mode)
        
        # 构造返回字典（添加固定油门）
        return {
            'rudder': action_dict['rudder'],
            'throttle': torch.ones(self.num_envs, device=self.device),  # 固定满油门
            'fire': action_dict['fire'],
        }
    
    def act_with_value(self, observation: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """返回动作、价值和对数概率（用于训练）
        
        Args:
            observation: 观察字典
        
        Returns:
            action_dict: 动作字典
            value: 状态价值 [num_envs]
            log_prob: 动作对数概率 [num_envs]
        """
        obs_tensor = self._obs_dict_to_tensor(observation)
        
        with torch.no_grad():
            action_dict = self.actor.get_action(obs_tensor, deterministic=False)
            value = self.critic(obs_tensor)
        
        return {
            'rudder': action_dict['rudder'],
            'throttle': torch.ones(self.num_envs, device=self.device),
            'fire': action_dict['fire'],
        }, value, action_dict['log_prob']
    
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """重置 agent 内部状态
        
        Args:
            env_mask: 可选，指定哪些环境需要重置的掩码
        """
        # 当前实现无需重置（无 RNN 状态）
        pass
    
    def update(self) -> Dict[str, float]:
        """执行 PPO 更新
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        self._ensure_buffer_initialized()
        
        # 累计统计
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        num_updates = 0
        
        # 归一化优势
        advantages = self.rollout_buffer.advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages = advantages.view(self.n_steps, self.num_envs)
        
        # 多个 epoch 训练
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get_samples(self.batch_size):
                obs = batch['observations']
                actions_rudder = batch['actions_rudder']
                actions_fire = batch['actions_fire']
                old_log_probs = batch['old_log_probs']
                advantages_batch = batch['advantages']
                returns = batch['returns']
                
                # 评估当前策略
                log_probs, entropy = self.actor.evaluate_actions(obs, actions_rudder, actions_fire)
                values = self.critic(obs)
                
                # 策略损失（PPO-Clip）
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, returns)
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # 近似 KL 散度和裁剪比例
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean().item()
                    total_approx_kl += approx_kl
                    total_clip_fraction += clip_fraction
                
                num_updates += 1
        
        # 清空缓冲区
        self.rollout_buffer.clear()
        
        # 返回平均统计
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'approx_kl': total_approx_kl / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
        }
    
    def train(self) -> 'PPOAgent':
        """设置为训练模式"""
        self.training_mode = True
        self.actor.train()
        self.critic.train()
        return self
    
    def eval(self) -> 'PPOAgent':
        """设置为评估模式"""
        self.training_mode = False
        self.actor.eval()
        self.critic.eval()
        return self
    
    def to(self, device) -> 'PPOAgent':
        """迁移到指定设备"""
        super().to(device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.rollout_buffer is not None:
            # 需要重新创建缓冲区
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.n_steps,
                num_envs=self.num_envs,
                obs_dim=self.obs_dim,
                device=self.device
            )
        return self
    
    def save(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_name(self) -> str:
        """返回 agent 名称"""
        return "PPOAgent"
