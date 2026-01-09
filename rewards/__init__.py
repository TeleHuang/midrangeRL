# -*- coding: utf-8 -*-
"""
Reward 模块
提供统一的 reward 接口和各种 reward 组合
"""

from rewards.base_reward import BaseReward, ZeroReward

__all__ = ['BaseReward', 'ZeroReward']
