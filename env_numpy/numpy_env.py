# -*- coding: utf-8 -*-
"""
NumPy 环境封装类
将 env_numpy 的各个模块封装成统一的环境接口
与 TensorEnv 保持接口兼容，用于 game_play 的可视化验证
"""

import math
from collections import deque
from typing import Dict, Tuple, Any, Optional

from env_numpy.aerodynamic import Aerodynamic
from env_numpy.missile_guidance import MissileGuidance


class RenderableEntity:
    """可渲染实体类，用于统一 visualization 接口"""
    
    def __init__(self, x, y, angle, speed, color, is_player1=True, is_missile=False):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        
        # 速度向量
        rad = math.radians(angle)
        self.vx = math.cos(rad) * speed
        self.vy = math.sin(rad) * speed
        
        self.throttle = 1.0
        self.rudder = 0.0
        self.missiles = 6
        self.color = color
        self.is_player1 = is_player1
        self.is_missile = is_missile
        self.trail = deque(maxlen=100)
        self.trail_update_count = 0
        self.turn_rate = 0.0
        self.n_load = 0.0
        self.g_load = 0.0
        self.mach = speed / 340.0
        self.alive = True
        
        if is_missile:
            self.target = None
            self.engine_time = 10.0


class NumpyEnv:
    """NumPy 环境封装类，提供与 TensorEnv 兼容的接口"""
    
    # 颜色定义
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    LIGHT_RED = (255, 150, 150)
    LIGHT_BLUE = (150, 150, 255)
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化环境
        
        Args:
            config: 配置字典
        """
        if config is None:
            from config import CONFIG
            config = CONFIG.copy()
        
        self.config = config
        self.battlefield_size = config.get('BATTLEFIELD_SIZE', 50000)
        self.initial_distance_ratio = config.get('INITIAL_DISTANCE_RATIO', 0.4)
        
        # 初始化物理和制导模块
        self.aero = Aerodynamic(config)
        self.guidance = MissileGuidance()
        
        # 游戏状态
        self.aircraft1 = None
        self.aircraft2 = None
        self.missiles = []
        self.game_over = False
        self.winner = None
        
        # 发射冷却
        self._last_fire_time_p1 = 0.0
        self._last_fire_time_p2 = 0.0
        self._fire_cooldown = config.get('fire_cooldown', 0.5)
        self._game_time = 0.0
        
        # 输入状态
        self._rudder_input = {1: 0, 2: 0}
        self._throttle_input = {1: 0, 2: 0}
        
        # 初始化环境
        self.reset()
    
    def reset(self) -> Dict:
        """重置环境到初始状态
        
        Returns:
            dict: 渲染状态
        """
        # 计算初始位置
        red_x = self.battlefield_size * self.initial_distance_ratio
        red_y = self.battlefield_size * self.initial_distance_ratio
        blue_x = self.battlefield_size * (1 - self.initial_distance_ratio)
        blue_y = self.battlefield_size * (1 - self.initial_distance_ratio)
        
        # 计算初始朝向（朝向战场中心）
        center = self.battlefield_size / 2
        red_angle = math.degrees(math.atan2(center - red_y, center - red_x))
        blue_angle = math.degrees(math.atan2(center - blue_y, center - blue_x))
        
        # 创建飞机
        initial_speed = self.config.get('FIGHTER_INITIAL_SPEED', 300)
        self.aircraft1 = RenderableEntity(red_x, red_y, red_angle, initial_speed, self.RED, True)
        self.aircraft1.missiles = self.config.get('FIGHTER_MISSILES', 6)
        
        self.aircraft2 = RenderableEntity(blue_x, blue_y, blue_angle, initial_speed, self.BLUE, False)
        self.aircraft2.missiles = self.config.get('FIGHTER_MISSILES', 6)
        
        # 清空导弹
        self.missiles = []
        
        # 重置游戏状态
        self.game_over = False
        self.winner = None
        self._game_time = 0.0
        self._last_fire_time_p1 = 0.0
        self._last_fire_time_p2 = 0.0
        
        return self.get_render_state()
    
    def step(self, actions: Dict, dt: float = 1/60.0) -> Tuple[bool, Optional[str]]:
        """执行一步环境更新
        
        Args:
            actions: 动作字典 {
                'p1_rudder': float [-1, 1],
                'p1_throttle': float [0, 1],
                'p1_fire': bool,
                'p2_rudder': float [-1, 1],
                'p2_throttle': float [0, 1],
                'p2_fire': bool,
            }
            dt: 时间步长
            
        Returns:
            tuple: (game_over, winner)
        """
        if self.game_over:
            return self.game_over, self.winner
        
        self._game_time += dt
        
        # 应用动作
        self._apply_actions(actions, dt)
        
        # 处理开火
        self._handle_firing(actions)
        
        # 更新物理
        self._update_physics(dt)
        
        # 更新导弹制导
        self._update_missile_guidance(dt)
        
        # 检查碰撞
        self._check_collisions()
        
        # 更新轨迹
        self._update_trails()
        
        # 检查游戏结束
        self._check_game_over()
        
        return self.game_over, self.winner
    
    def _apply_actions(self, actions: Dict, dt: float):
        """应用玩家动作"""
        # 处理 P1 舵量（渐变式输入）
        p1_rudder_input = actions.get('p1_rudder', 0.0)
        if abs(p1_rudder_input) > 0.01:
            # 有输入时逐渐改变舵量
            rudder_change_rate = 1.0
            self.aircraft1.rudder += rudder_change_rate * dt * p1_rudder_input
            self.aircraft1.rudder = max(-1.0, min(1.0, self.aircraft1.rudder))
        else:
            # 无输入时舵量回正
            if abs(self.aircraft1.rudder) > 0.01:
                rudder_return_rate = 1.0
                if self.aircraft1.rudder > 0:
                    self.aircraft1.rudder = max(0, self.aircraft1.rudder - rudder_return_rate * dt)
                else:
                    self.aircraft1.rudder = min(0, self.aircraft1.rudder + rudder_return_rate * dt)
            else:
                self.aircraft1.rudder = 0
        
        # 处理 P1 油门
        p1_throttle_input = actions.get('p1_throttle_delta', 0.0)
        if abs(p1_throttle_input) > 0.01:
            throttle_change_rate = 0.5
            self.aircraft1.throttle += throttle_change_rate * dt * p1_throttle_input
            self.aircraft1.throttle = max(0.0, min(1.0, self.aircraft1.throttle))
        
        # 直接设置油门（如果提供）
        if 'p1_throttle' in actions:
            self.aircraft1.throttle = max(0.0, min(1.0, actions['p1_throttle']))
        
        # 处理 P2 舵量
        p2_rudder_input = actions.get('p2_rudder', 0.0)
        if abs(p2_rudder_input) > 0.01:
            rudder_change_rate = 1.0
            self.aircraft2.rudder += rudder_change_rate * dt * p2_rudder_input
            self.aircraft2.rudder = max(-1.0, min(1.0, self.aircraft2.rudder))
        else:
            if abs(self.aircraft2.rudder) > 0.01:
                rudder_return_rate = 1.0
                if self.aircraft2.rudder > 0:
                    self.aircraft2.rudder = max(0, self.aircraft2.rudder - rudder_return_rate * dt)
                else:
                    self.aircraft2.rudder = min(0, self.aircraft2.rudder + rudder_return_rate * dt)
            else:
                self.aircraft2.rudder = 0
        
        # 处理 P2 油门
        p2_throttle_input = actions.get('p2_throttle_delta', 0.0)
        if abs(p2_throttle_input) > 0.01:
            throttle_change_rate = 0.5
            self.aircraft2.throttle += throttle_change_rate * dt * p2_throttle_input
            self.aircraft2.throttle = max(0.0, min(1.0, self.aircraft2.throttle))
        
        if 'p2_throttle' in actions:
            self.aircraft2.throttle = max(0.0, min(1.0, actions['p2_throttle']))
    
    def _handle_firing(self, actions: Dict):
        """处理导弹发射"""
        # P1 开火
        p1_fire = actions.get('p1_fire', False)
        if p1_fire and self.aircraft1.alive and self.aircraft1.missiles > 0:
            if self._game_time - self._last_fire_time_p1 >= self._fire_cooldown:
                missile = self._create_missile(self.aircraft1, self.aircraft2)
                if missile:
                    self.missiles.append(missile)
                    self.aircraft1.missiles -= 1
                    self._last_fire_time_p1 = self._game_time
        
        # P2 开火
        p2_fire = actions.get('p2_fire', False)
        if p2_fire and self.aircraft2.alive and self.aircraft2.missiles > 0:
            if self._game_time - self._last_fire_time_p2 >= self._fire_cooldown:
                missile = self._create_missile(self.aircraft2, self.aircraft1)
                if missile:
                    self.missiles.append(missile)
                    self.aircraft2.missiles -= 1
                    self._last_fire_time_p2 = self._game_time
    
    def _create_missile(self, source, target) -> Optional[RenderableEntity]:
        """创建导弹"""
        offset = 20
        missile_x = source.x + math.cos(math.radians(source.angle)) * offset
        missile_y = source.y + math.sin(math.radians(source.angle)) * offset
        
        color = self.LIGHT_RED if source.is_player1 else self.LIGHT_BLUE
        missile = RenderableEntity(missile_x, missile_y, source.angle, source.speed, color, source.is_player1, True)
        missile.vx = source.vx
        missile.vy = source.vy
        missile.target = target
        missile.engine_time = self.config.get('MISSILE_ENGINE_DURATION', 10.0)
        
        return missile
    
    def _update_physics(self, dt: float):
        """更新物理状态"""
        if self.aircraft1.alive:
            self.aero.calculate_aircraft_physics(self.aircraft1, dt)
        if self.aircraft2.alive:
            self.aero.calculate_aircraft_physics(self.aircraft2, dt)
        
        for missile in self.missiles[:]:
            self.aero.calculate_aircraft_physics(missile, dt)
    
    def _update_missile_guidance(self, dt: float):
        """更新导弹制导"""
        for missile in self.missiles[:]:
            if missile.target and missile.target.alive:
                missile.rudder = self.guidance.calculate_guidance(missile, missile.target)
            else:
                missile.rudder = 0.0
    
    def _check_collisions(self):
        """检查碰撞"""
        hit_radius = self.config.get('hit_radius', 100)
        self_destruct_speed = self.config.get('self_destruct_speed', 200)
        
        for missile in self.missiles[:]:
            # 检查命中
            if missile.target and missile.target.alive:
                dx = missile.x - missile.target.x
                dy = missile.y - missile.target.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < hit_radius:
                    missile.target.alive = False
                    missile.alive = False
                    self.missiles.remove(missile)
                    continue
            
            # 检查自毁（速度过低）
            if missile.speed < self_destruct_speed:
                missile.alive = False
                self.missiles.remove(missile)
                continue
    
    def _update_trails(self):
        """更新轨迹"""
        trail_interval = 5
        
        for entity in [self.aircraft1, self.aircraft2] + self.missiles:
            if not hasattr(entity, 'alive') or not entity.alive:
                continue
            entity.trail_update_count += 1
            if entity.trail_update_count >= trail_interval:
                entity.trail.append((entity.x, entity.y))
                entity.trail_update_count = 0
    
    def _check_game_over(self):
        """检查游戏是否结束"""
        p1_alive = self.aircraft1.alive
        p2_alive = self.aircraft2.alive
        
        # 有一方被击毁
        if not p1_alive and not p2_alive:
            self.game_over = True
            self.winner = 'draw'
        elif not p1_alive:
            self.game_over = True
            self.winner = 'blue'
        elif not p2_alive:
            self.game_over = True
            self.winner = 'red'
        
        # 双方弹药耗尽且无在途导弹
        if not self.game_over:
            if self.aircraft1.missiles == 0 and self.aircraft2.missiles == 0 and not self.missiles:
                self.game_over = True
                self.winner = 'draw'
    
    def get_render_state(self) -> Dict:
        """获取用于渲染的状态
        
        Returns:
            dict: 包含 aircraft1, aircraft2, missiles, game_over, winner
        """
        return {
            'aircraft1': self.aircraft1,
            'aircraft2': self.aircraft2,
            'missiles': self.missiles,
            'game_over': self.game_over,
            'winner': self.winner,
        }
    
    def close(self):
        """关闭环境"""
        pass
