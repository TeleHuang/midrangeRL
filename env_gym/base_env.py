# -*- coding: utf-8 -*-
"""
统一的环境接口基类
定义所有环境实现必须遵循的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch


class BaseEnv(ABC):
    """环境基类，定义标准接口"""
    
    @abstractmethod
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """重置环境
        
        Args:
            env_mask: 可选，指定哪些环境需要重置的掩码
            
        Returns:
            dict: 初始观察空间
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """执行环境步骤
        
        Args:
            actions: 动作字典，格式为 {'p1_rudder': tensor, 'p1_throttle': tensor, 'p1_fire': tensor,
                                     'p2_rudder': tensor, 'p2_throttle': tensor, 'p2_fire': tensor}
            
        Returns:
            tuple: (observations, rewards, dones, infos)
                - observations: 观察空间字典
                - rewards: 奖励字典 {'p1': tensor, 'p2': tensor}
                - dones: 完成标志 tensor [num_envs]
                - infos: 额外信息字典
        """
        pass
    
    @abstractmethod
    def get_observations(self) -> Dict[str, torch.Tensor]:
        """获取当前观察空间
        
        Returns:
            dict: 观察空间字典
        """
        pass
    
    @abstractmethod
    def compute_rewards(self, done_mask: torch.Tensor, winner_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算奖励
        
        Args:
            done_mask: 完成的环境掩码
            winner_mask: 胜利者掩码（可选）
            
        Returns:
            dict: 奖励字典 {'p1': tensor, 'p2': tensor}
        """
        pass
    
    def get_render_state(self, env_idx: int = 0) -> Dict[str, Any]:
        """获取用于渲染的状态（单个环境）
        
        Args:
            env_idx: 环境索引（默认为0）
            
        Returns:
            dict: 包含 aircraft1, aircraft2, missiles, game_over, winner 的渲染状态
                - aircraft1: 红方飞机状态
                - aircraft2: 蓝方飞机状态
                - missiles: 所有激活导弹列表
                - game_over: 游戏是否结束
                - winner: 胜利者 ('red', 'blue', 'draw', None)
        """
        raise NotImplementedError("Subclasses should implement get_render_state()")
    
    @property
    def num_envs(self) -> int:
        """返回并行环境数量"""
        return getattr(self, '_num_envs', 1)
    
    @property
    def device(self) -> torch.device:
        """返回计算设备"""
        return getattr(self, '_device', torch.device('cpu'))
    
    def close(self):
        """关闭环境，释放资源"""
        pass

