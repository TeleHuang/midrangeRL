# -*- coding: utf-8 -*-

import math
import numpy as np
import time

class MissileGuidance:
    def __init__(self, guidance_gain=200):
        """初始化导弹制导系统
        
        Args:
            guidance_gain: 制导增益系数
        """
        self.guidance_gain = guidance_gain
        self.guidance_history = {}  # 存储每个导弹的制导历史
    
    def calculate_guidance(self, missile, target):
        """计算导弹制导指令（比例引导法）
        
        Args:
            missile: 导弹对象
            target: 目标对象
            
        Returns:
            float: 舵量指令 (-1 到 1)
        """
        if not target:
            return 0.0
        
        # 获取当前时间
        current_time = time.time()
        
        # 创建或获取导弹的制导历史
        missile_id = id(missile)
        if missile_id not in self.guidance_history:
            self.guidance_history[missile_id] = {
                'last_los_angle': None,
                'last_time': None
            }
        
        history = self.guidance_history[missile_id]
        
        # 计算当前弹目连线角度
        dx = target.x - missile['x'] if isinstance(missile, dict) else target.x - missile.x
        dy = target.y - missile['y'] if isinstance(missile, dict) else target.y - missile.y
        current_los_angle = math.degrees(math.atan2(dy, dx))
        
        # 计算视线角速度
        if history['last_los_angle'] is not None and history['last_time'] is not None:
            dt = current_time - history['last_time']
            if dt > 0:
                # 处理角度跨越360度的情况
                angle_diff = (current_los_angle - history['last_los_angle'] + 180) % 360 - 180
                los_rate = angle_diff / dt  # 角速度（度/秒）
            else:
                los_rate = 0
        else:
            los_rate = 0
        
        # 更新历史数据
        history['last_los_angle'] = current_los_angle
        history['last_time'] = current_time
        
        # 计算制导指令（比例导引）
        rudder = self.guidance_gain * los_rate / 180.0  # 归一化到[-1,1]范围
        rudder = max(-1.0, min(1.0, rudder))  # 限制舵量范围
        
        return rudder
    
    def clean_history(self, missile_id):
        """清除指定导弹的制导历史
        
        Args:
            missile_id: 导弹的id
        """
        if missile_id in self.guidance_history:
            del self.guidance_history[missile_id]