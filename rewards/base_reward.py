# -*- coding: utf-8 -*-
"""
Reward 抽象基类
定义所有 reward 函数必须实现的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class BaseReward(ABC):
    """Reward 抽象基类
    
    统一接口设计原则：
    1. 接收 step 前后状态或 env 提供的信息
    2. 返回标量 reward（支持批量操作）
    """
    
    def __init__(self, device: str = 'cuda'):
        """初始化 reward 函数
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device) if isinstance(device, str) else device
    
    @abstractmethod
    def compute(
        self,
        obs_before: Dict[str, torch.Tensor],
        obs_after: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        done: torch.Tensor,
        info: Dict[str, Any]
    ) -> torch.Tensor:
        """计算奖励
        
        Args:
            obs_before: step 前的观察 [num_envs, ...]
            obs_after: step 后的观察 [num_envs, ...]
            action: 执行的动作 [num_envs, ...]
            done: 是否结束 [num_envs]
            info: 环境返回的额外信息，通常包含：
                - winner: 胜利者 [num_envs] (0=draw, 1=p1, 2=p2)
                - p1_alive: P1 存活状态 [num_envs]
                - p2_alive: P2 存活状态 [num_envs]
                
        Returns:
            reward: 标量奖励 tensor [num_envs]
        """
        pass
    
    def reset(self) -> None:
        """重置 reward 函数内部状态（如果有）
        
        某些 reward 函数可能有内部状态（如累计计数器），
        在环境重置时需要清除。
        """
        pass
    
    def to(self, device: str) -> 'BaseReward':
        """将 reward 函数移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self，支持链式调用
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        return self
    
    def get_name(self) -> str:
        """返回 reward 函数名称（用于日志和调试）"""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.get_name()}(device={self.device})"


class ZeroReward(BaseReward):
    """占位 Reward（始终返回 0）
    
    用于测试和作为基线。
    """
    
    def compute(
        self,
        obs_before: Dict[str, torch.Tensor],
        obs_after: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        done: torch.Tensor,
        info: Dict[str, Any]
    ) -> torch.Tensor:
        """返回零奖励
        
        Returns:
            zeros tensor [num_envs]
        """
        num_envs = done.shape[0]
        return torch.zeros(num_envs, device=self.device)
