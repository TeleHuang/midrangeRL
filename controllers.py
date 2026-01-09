# -*- coding: utf-8 -*-
"""
控制器抽象层
定义统一的控制接口，隔离人类玩家和AI Agent的输入来源
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pygame
import torch
import math


class PlayerController(ABC):
    """玩家控制器抽象基类
    
    定义统一的控制接口，屏蔽输入来源差异（键盘 vs AI）
    """
    
    def __init__(self, env_backend: str = 'numpy'):
        """初始化控制器
        
        Args:
            env_backend: 环境后端类型 ('numpy' 或 'tensor')
        """
        self.env_backend = env_backend
    
    @abstractmethod
    def get_action(self, observation: dict, keys: pygame.key.ScancodeWrapper, dt: float) -> dict:
        """根据观察和输入返回动作字典
        
        Args:
            observation: 包含当前玩家和敌方状态信息的字典
            keys: Pygame 按键状态对象
            dt: 时间步长（秒）
        
        Returns:
            动作字典，包含键：'rudder', 'throttle_delta', 'fire'
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置控制器内部状态"""
        pass


class HumanController(PlayerController):
    """人类玩家控制器
    
    处理键盘输入，支持舵量渐变和开火边缘检测
    """
    
    def __init__(self, env_backend: str = 'numpy', key_mapping: Optional[dict] = None):
        """初始化人类控制器
        
        Args:
            env_backend: 环境后端类型
            key_mapping: 自定义键位映射（可选）
        """
        super().__init__(env_backend)
        
        # 默认键位映射
        if key_mapping is None:
            key_mapping = {
                'left': pygame.K_LEFT,
                'right': pygame.K_RIGHT,
                'up': pygame.K_UP,
                'down': pygame.K_DOWN,
                'fire': pygame.K_EQUALS
            }
        self.key_mapping = key_mapping
        
        # 边缘检测状态
        self.prev_fire_key = False
        
        # tensor 后端的舵量渐变状态
        self.current_rudder = 0.0
    
    def get_action(self, observation: dict, keys: pygame.key.ScancodeWrapper, dt: float) -> dict:
        """处理键盘输入并生成动作
        
        Args:
            observation: 观察字典（本控制器不使用）
            keys: Pygame 按键状态
            dt: 时间步长
        
        Returns:
            动作字典
        """
        # 读取方向键状态
        rudder_input = 0
        if keys[self.key_mapping['left']]:
            rudder_input = -1
        elif keys[self.key_mapping['right']]:
            rudder_input = 1
        
        # 读取油门键状态
        throttle_input = 0
        if keys[self.key_mapping['up']]:
            throttle_input = 1
        elif keys[self.key_mapping['down']]:
            throttle_input = -1
        
        # 开火边缘检测
        current_fire_key = keys[self.key_mapping['fire']]
        
        if self.env_backend == 'tensor':
            # tensor 后端：边缘检测 + 舵量渐变
            fire = current_fire_key and not self.prev_fire_key
            self.prev_fire_key = current_fire_key
            
            # 舵量渐变逻辑
            rudder_change_rate = 1.0  # 1秒从0到1
            rudder_return_rate = 1.0  # 1秒回正
            
            if rudder_input != 0:
                self.current_rudder += rudder_change_rate * dt * rudder_input
                self.current_rudder = max(-1.0, min(1.0, self.current_rudder))
            else:
                # 无输入时回正
                if abs(self.current_rudder) > 0.01:
                    if self.current_rudder > 0:
                        self.current_rudder = max(0, self.current_rudder - rudder_return_rate * dt)
                    else:
                        self.current_rudder = min(0, self.current_rudder + rudder_return_rate * dt)
                else:
                    self.current_rudder = 0
            
            return {
                'rudder': self.current_rudder,
                'throttle_delta': float(throttle_input),
                'fire': fire
            }
        else:
            # numpy 后端：已有内置冷却机制，直接返回
            return {
                'rudder': float(rudder_input),
                'throttle_delta': float(throttle_input),
                'fire': current_fire_key
            }
    
    def reset(self) -> None:
        """重置控制器内部状态"""
        self.prev_fire_key = False
        self.current_rudder = 0.0


class ModelController(PlayerController):
    """AI Agent 控制器
    
    调用 Agent 的 act 方法生成动作，并处理观察和动作的格式转换
    """
    
    def __init__(self, agent, env_backend: str = 'numpy'):
        """初始化模型控制器
        
        Args:
            agent: BaseAgent 实例
            env_backend: 环境后端类型
        """
        super().__init__(env_backend)
        self.agent = agent
        self.agent.num_envs = 1  # 游戏模式下单环境
        self.device = agent.device
    
    def get_action(self, observation: dict, keys: pygame.key.ScancodeWrapper, dt: float) -> dict:
        """调用 Agent 生成动作
        
        Args:
            observation: 观察字典
            keys: Pygame 按键状态（本控制器不使用）
            dt: 时间步长
        
        Returns:
            动作字典
        """
        # 将观察转换为张量格式
        obs_tensor = self._convert_obs_to_tensor(observation)
        
        # 调用 Agent 决策（传递 dt 参数）
        action_tensor = self.agent.act(obs_tensor, dt=dt)
        
        # 将张量动作转换为标量字典
        action_dict = self._convert_action_to_dict(action_tensor)
        
        return action_dict
    
    def reset(self) -> None:
        """重置 Agent 内部状态"""
        self.agent.reset()
    
    def _convert_obs_to_tensor(self, observation: dict) -> dict:
        """将观察字典转换为张量格式
        
        Args:
            observation: 包含标量或 numpy 数组的字典
        
        Returns:
            所有值为形状 [1] 的 torch.Tensor 字典
        """
        obs_tensor = {}
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                # 已经是张量，确保形状为 [1]
                if value.dim() == 0:
                    obs_tensor[key] = value.unsqueeze(0).to(self.device)
                else:
                    obs_tensor[key] = value.to(self.device)
            elif isinstance(value, bool):
                obs_tensor[key] = torch.tensor([value], dtype=torch.bool, device=self.device)
            elif isinstance(value, (int, float)):
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
            else:
                # numpy 数组或其他可转换类型
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        
        return obs_tensor
    
    def _convert_action_to_dict(self, action: dict) -> dict:
        """将张量动作转换为标量字典
        
        Args:
            action: Agent 输出的张量动作字典
        
        Returns:
            包含 Python 标量的动作字典
        """
        action_dict = {}
        for key, value in action.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    # 标量张量
                    action_dict[key] = value.item()
                else:
                    # [1] 形状的张量
                    action_dict[key] = value[0].item()
            else:
                action_dict[key] = value
        
        # 确保返回正确的键名（Agent 使用 'throttle'，game_play 使用 'throttle_delta'）
        if 'throttle' in action_dict and 'throttle_delta' not in action_dict:
            action_dict['throttle_delta'] = action_dict.pop('throttle')
        
        return action_dict
