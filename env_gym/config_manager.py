# -*- coding: utf-8 -*-
"""
统一的配置管理系统
"""

from typing import Dict, Any
import json
import os


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        # 游戏基本配置
        'battlefield_size': 50000,
        'hit_radius': 100,
        'self_destruct_speed': 200,
        'initial_missiles': 6,
        'missile_launch_offset': 20.0,
        
        # 战斗机参数
        'FIGHTER_TERMINAL_VELOCITY': 400,
        'FIGHTER_MIN_TURN_RADIUS': 1000,
        'FIGHTER_MAX_THRUST': 1.5 * 9.8,
        'FIGHTER_LIFT_DRAG_RATIO': 5,
        
        # 导弹参数
        'MISSILE_TERMINAL_VELOCITY': 400,
        'MISSILE_MIN_TURN_RADIUS': 1000,
        'MISSILE_THRUST': 15 * 9.8,
        'MISSILE_ENGINE_DURATION': 10.0,
        'MISSILE_LIFT_DRAG_RATIO': 3,
        
        # 制导参数
        'guidance_gain': 5.0,
        
        # 奖励权重
        'reward_hit_enemy': 100.0,
        'reward_get_hit': -200.0,
        'reward_survive': 0.1,
        'reward_missile_used': -5.0,
        'reward_win': 500.0,
        'reward_lose': -500.0,
        'reward_draw': -50.0,
    }
    
    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        """加载配置
        
        Args:
            config_path: 配置文件路径（可选）
            
        Returns:
            dict: 配置字典
        """
        config = cls.DEFAULT_CONFIG.copy()
        
        # 如果提供了配置文件路径，从文件加载
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        # 从环境变量加载（可选）
        # 例如：MIDRANGE_RL_BATTLEFIELD_SIZE=60000
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], config_path: str):
        """保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 保存路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def merge_configs(cls, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置字典
        
        Args:
            *configs: 多个配置字典
            
        Returns:
            dict: 合并后的配置
        """
        merged = cls.DEFAULT_CONFIG.copy()
        for config in configs:
            merged.update(config)
        return merged


def get_default_config() -> Dict[str, Any]:
    """获取默认配置（兼容旧代码）"""
    return ConfigManager.DEFAULT_CONFIG.copy()

