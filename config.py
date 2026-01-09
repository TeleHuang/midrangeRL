# -*- coding: utf-8 -*-

"""
配置文件 - 定义游戏中各种飞行器的物理参数
"""

# 游戏基本配置
GAME_CONFIG = {
    'BATTLEFIELD_SIZE': 50000,  # 初始位置仍使用的参考尺寸 (50km)
    'INITIAL_DISTANCE_RATIO': 0.3,  # 初始位置比例（0-0.5）
    'WINDOW_WIDTH': 1600,       # 总窗口宽度
    'WINDOW_HEIGHT': 1000,      # 顶部100 + 中间800 + 底部100
    # 视图/相机相关配置
    'VIEW_WORLD_WIDTH': 50000,   # 单个视口水平覆盖的世界宽度（米）
    'VIEW_WORLD_HEIGHT': 50000,  # 单个视口垂直覆盖的世界高度（米）
    'GRID_SPACING': 1000,        # 地面参考网格间隔（米）
    'PLAYER_ANCHOR_X_RATIO': 0.5,# 玩家在视图中的水平锚点（0-1）
}

# 战斗机参数
FIGHTER_CONFIG = {
    'FIGHTER_TERMINAL_VELOCITY': 400,  # 自由落体终端速度 (m/s)
    'FIGHTER_MIN_TURN_RADIUS': 1000,   # 最小转弯半径 (m)
    'FIGHTER_MAX_THRUST': 1.5 * 9.8,  # 最大推力 (m/s²)
    'FIGHTER_LIFT_DRAG_RATIO': 5,     # 最大升阻比
    'FIGHTER_MISSILES': 6,            # 导弹数量
}

# 导弹参数
MISSILE_CONFIG = {
    'MISSILE_TERMINAL_VELOCITY': 400, # 自由落体终端速度 (m/s)
    'MISSILE_MIN_TURN_RADIUS': 1000,  # 最小转弯半径 (m)
    'MISSILE_THRUST': 15 * 9.8,       # 发动机推力 (m/s²)
    'MISSILE_ENGINE_DURATION': 10.0,  # 发动机工作时间 (s)
    'MISSILE_LIFT_DRAG_RATIO': 2,     # 最大升阻比
    'MISSILE_GUIDANCE_GAIN': 200,     # 比例导引增益系数
}

# 物理常数
PHYSICS_CONSTANTS = {
    'G': 9.8,  # 重力加速度 (m/s²)
}

# 合并所有配置
CONFIG = {
    **GAME_CONFIG,
    **FIGHTER_CONFIG,
    **MISSILE_CONFIG,
    **PHYSICS_CONSTANTS,
} 