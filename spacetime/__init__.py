# -*- coding: utf-8 -*-
"""
时空图模块
提供时空图计算和渲染功能

主要组件：

CPU版（单环境可视化使用）:
- SpacetimeComputer: 时空图计算器（核心计算模块）
- SpacetimeRenderer: 时空图渲染器（3D可视化）
- MissileTables: 导弹查找表管理器
- TrajectoryPredictor: 轨迹预测器
"""

from spacetime.spacetime_core import (
    SpacetimePoint,
    SpacetimeTrail,
    MissileTables,
    ThreatConeGenerator,
    TrajectoryPredictor,
    SpacetimeComputer,
)

from spacetime.spacetime_renderer import (
    Camera3D,
    SpacetimeRenderer,
)

__all__ = [
    # CPU版
    'SpacetimePoint',
    'SpacetimeTrail',
    'MissileTables',
    'ThreatConeGenerator',
    'TrajectoryPredictor',
    'SpacetimeComputer',
    'Camera3D',
    'SpacetimeRenderer',
]
