# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import deque

class GameEvents:
    def __init__(self, battlefield_size=50000):
        self.battlefield_size = battlefield_size
        self.missiles = []  # 存储所有导弹
        self.game_over = False
        self.winner = None
    
    def fire_missile(self, aircraft, target):
        """从飞机发射导弹
        
        Args:
            aircraft: 发射导弹的飞机
            target: 导弹的目标
            
        Returns:
            dict: 新创建的导弹对象，如果无法发射则返回None
        """
        # 检查是否还有导弹
        if aircraft.missiles <= 0:
            return None
        
        # 减少导弹数量
        aircraft.missiles -= 1
        
        # 创建导弹实体（位置略微前移,以避免碰撞）
        offset = 20  # 前移距离（米）
        missile_x = aircraft.x + math.cos(math.radians(aircraft.angle)) * offset
        missile_y = aircraft.y + math.sin(math.radians(aircraft.angle)) * offset
        
        # 创建导弹对象
        from env_numpy.aerodynamic import Aerodynamic
        aero = Aerodynamic()
        missile = aero.create_aircraft(missile_x, missile_y, aircraft.angle, is_missile=True)
        
        # 设置导弹的其他属性
        missile['speed'] = aircraft.speed  # 导弹继承飞机的速度
        missile['target'] = target  # 设置目标
        missile['source'] = aircraft  # 记录发射者
        missile['is_player1'] = getattr(aircraft, 'is_player1', False)  # 记录发射方
        
        # 设置轨迹
        missile['trail'] = deque(maxlen=100)
        missile['trail_update_count'] = 0
        
        # 返回导弹对象
        return missile
    
    def check_missile_hit(self, missile):
        """检查导弹是否击中目标
        
        Args:
            missile: 导弹对象
            
        Returns:
            bool: 是否击中目标
        """
        if not missile['target']:
            return False
            
        target = missile['target']
        dx = missile['x'] - target.x
        dy = missile['y'] - target.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        return distance < 100  # 导弹命中半径为100米
    
    def check_missile_self_destruct(self, missile):
        """检查导弹是否应该自毁（速度过低）
        
        Args:
            missile: 导弹对象
            
        Returns:
            bool: 是否自毁
        """
        return missile['speed'] < 200  # 速度低于200m/s自毁
    
    def update_battlefield_boundary(self, entity):
        """确保实体不会飞出战场边界
        
        Args:
            entity: 飞机或导弹对象
        """
        entity.x = max(0, min(self.battlefield_size, entity.x))
        entity.y = max(0, min(self.battlefield_size, entity.y))
    
    def check_game_over(self, aircraft1, aircraft2):
        """检查游戏是否结束
        
        Args:
            aircraft1: 红方飞机
            aircraft2: 蓝方飞机
            
        Returns:
            tuple: (是否结束, 胜利者)
        """
        # 如果有一方被击毁，游戏结束
        if not hasattr(aircraft1, 'alive') or not hasattr(aircraft2, 'alive'):
            # 初始化存活状态
            aircraft1.alive = True
            aircraft2.alive = True
        
        # 检查双方存活状态
        if not aircraft1.alive and not aircraft2.alive:
            return True, 'draw'  # 同归于尽
        elif not aircraft1.alive:
            return True, 'blue'  # 蓝方胜利
        elif not aircraft2.alive:
            return True, 'red'   # 红方胜利
        
        # 检查是否弹药耗尽且无在途导弹
        if aircraft1.missiles == 0 and aircraft2.missiles == 0 and not self.missiles:
            if aircraft1.alive and aircraft2.alive:
                return True, 'draw'
            elif aircraft1.alive and not aircraft2.alive:
                return True, 'red'
            elif not aircraft1.alive and aircraft2.alive:
                return True, 'blue'
            else:
                return True, 'draw'
        
        return False, None  # 游戏继续
