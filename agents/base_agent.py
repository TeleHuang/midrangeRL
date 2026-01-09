# -*- coding: utf-8 -*-
"""
Agent 抽象基类
定义所有 agent 必须实现的统一接口，适用于规则 agent 和 RL agent
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import numpy as np


class BaseAgent(ABC):
    """Agent 抽象基类
    
    统一接口设计原则：
    1. 输入：环境 observation（支持 tensor 和 numpy）
    2. 输出：合法 action（与 env_gym 接口完全一致）
    3. 支持批量操作（多环境并行）
    
    动作空间格式：
    - rudder: float [-1.0, 1.0] 方向舵
    - throttle: float [0.0, 1.0] 油门
    - fire: bool 开火
    """
    
    def __init__(self, device: str = 'cuda'):
        """初始化 agent
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self._num_envs = 1
    
    @property
    def num_envs(self) -> int:
        """返回 agent 支持的环境数量"""
        return self._num_envs
    
    @num_envs.setter
    def num_envs(self, value: int):
        """设置 agent 支持的环境数量"""
        self._num_envs = value
    
    @abstractmethod
    def act(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """根据观察返回动作
        
        Args:
            observation: 观察字典，包含：
                - x: 归一化位置 x [num_envs]
                - y: 归一化位置 y [num_envs]
                - angle: 归一化角度 [num_envs]
                - speed: 归一化速度 [num_envs]
                - missiles: 归一化剩余导弹数 [num_envs]
                - alive: 存活状态 [num_envs]
                - enemy_distance: 归一化敌方距离 [num_envs]
                - enemy_relative_angle: 归一化敌方相对角度 [num_envs]
                - enemy_speed: 归一化敌方速度 [num_envs]
                - enemy_alive: 敌方存活状态 [num_envs]
                
        Returns:
            动作字典：
                - rudder: 方向舵 [-1.0, 1.0] tensor [num_envs]
                - throttle: 油门 [0.0, 1.0] tensor [num_envs]
                - fire: 开火指令 bool tensor [num_envs]
        """
        pass
    
    @abstractmethod
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """重置 agent 内部状态
        
        当环境重置时调用此方法，用于清除 agent 的历史记忆、
        隐藏状态等（例如 RNN 隐藏状态）。
        
        Args:
            env_mask: 可选，指定哪些环境需要重置的掩码 [num_envs]
                     如果为 None，则重置所有环境的状态
        """
        pass
    
    def to(self, device: Union[str, torch.device]) -> 'BaseAgent':
        """将 agent 移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self，支持链式调用
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        return self
    
    def train(self) -> 'BaseAgent':
        """设置为训练模式（用于 RL agent）
        
        Returns:
            self，支持链式调用
        """
        return self
    
    def eval(self) -> 'BaseAgent':
        """设置为评估模式（用于 RL agent）
        
        Returns:
            self，支持链式调用
        """
        return self
    
    def get_name(self) -> str:
        """返回 agent 名称（用于日志和调试）"""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.get_name()}(device={self.device}, num_envs={self.num_envs})"
