# -*- coding: utf-8 -*-
"""
时空图核心计算模块 (Spacetime Diagram Core)

功能概述:
    时空图是一种将空战信息在三维空间中展示的可视化方式：
    - X/Y轴: 二维空间位置
    - T轴(竖轴): 时间（未来向下，过去向上）
    
    通过时空图，飞行员可以直观地看到：
    - 自身轨迹预测（当前舵量预测 + 全力回转逃脱曲线）
    - 全力回转逃脱线是指旋转至逃脱方向即停止旋转进入直线加速的预测线
    - 敌方威胁锥（导弹可达范围）
    - 来袭导弹的预测轨迹

开发注意事项：
    - 本模块为CPU端程序,是为了在较少地引发bug的前提下验证时空图的设想
    - 本模块相对不注重性能优化，主要关注功能实现
    - RL训练时使用的时空图是由tensor_spacetime计算的，性能更高
    - 本模块帧率更高，曲线更光滑，用更大的性能消耗为game_play的CPU模式提供精良的游戏体验

模块组成:
    - SpacetimePoint: 时空点数据结构 (x, y, t, speed, angle)
    - SpacetimeTrail: 时空轨迹管理（预测）
    - MissileTables: 导弹飞行轨迹查找表（预计算）
    - ThreatConeGenerator: 威胁锥几何生成器
    - TrajectoryPredictor: 飞机/导弹轨迹预测器
    - SpacetimeComputer: 时空图计算综合管理器

依赖关系:
    MissileTables <- ThreatConeGenerator
    config <- TrajectoryPredictor  
    (MissileTables, ThreatConeGenerator, TrajectoryPredictor) <- SpacetimeComputer
"""

import math
import json
import os
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# 基础数据结构
# =============================================================================

class SpacetimePoint:
    """
    时空点 - 时空图中的基本元素
    
    坐标约定:
        - x, y: 世界空间坐标（米）
        - t: 相对时间（秒），过去为正，未来为负
        - speed: 该时刻的速度（m/s）
        - angle: 该时刻的航向角（度）
    
    使用__slots__优化内存占用，因为会创建大量实例。
    """
    __slots__ = ['x', 'y', 't', 'speed', 'angle']
    
    def __init__(self, x: float, y: float, t: float, speed: float = 0, angle: float = 0):
        self.x = x
        self.y = y
        self.t = t  # 相对于当前时刻的时间偏移（过去为正，未来为负）
        self.speed = speed
        self.angle = angle
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.t)
    
    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 't': self.t, 'speed': self.speed, 'angle': self.angle}


class SpacetimeTrail:
    """
    时空轨迹管理器
    
    包含一个实体（飞机/导弹）的：
    - 预测轨迹: 未来的多条预测线（如current/left_turn/right_turn）
    - 预测时间戳: 用于时间同步（预测线随时间流逝上升）
    """
    
    def __init__(self):
        """
        初始化时空轨迹管理器
        注意：历史轨迹功能已弃用，仅保留预测功能
        """
        # 预测轨迹列表
        self.predictions: Dict[str, List[SpacetimePoint]] = {}
        
        # 预测创建时间（用于时间同步）
        self.prediction_times: Dict[str, float] = {}
    
    def set_prediction(self, name: str, points: List[SpacetimePoint], create_time: float = 0.0):
        """设置预测轨迹"""
        self.predictions[name] = points
        self.prediction_times[name] = create_time
    
    def get_prediction(self, name: str) -> List[SpacetimePoint]:
        """获取预测轨迹"""
        return self.predictions.get(name, [])
    
    def get_prediction_time(self, name: str) -> float:
        """获取预测创建时间"""
        return self.prediction_times.get(name, 0.0)
    
    def clear_predictions(self):
        """清除所有预测"""
        self.predictions.clear()
        self.prediction_times.clear()


# =============================================================================
# 导弹查找表与威胁锥
# =============================================================================

class MissileTables:
    """
    导弹飞行轨迹查找表
    
    通过预计算不同载机速度下导弹的飞行轨迹（时间-距离-速度），
    避免运行时重复计算。
    
    查找表由 generate_missile_tables.py 脚本预生成，
    存储在 missile_tables.json 文件中。
    
    主要用途:
        - 威胁锥生成：查询某时刻导弹能飞多远
        - 导弹轨迹预测：根据当前速度查询衰减曲线
    """
    
    def __init__(self, table_path: str = None):
        """
        Args:
            table_path: 查找表JSON文件路径
        """
        if table_path is None:
            table_path = os.path.join(os.path.dirname(__file__), 'missile_tables.json')
        
        self.table_path = table_path
        self.trajectories = {}
        self.metadata = {}
        self._loaded = False
        
    def load(self):
        """加载查找表"""
        if self._loaded:
            return
            
        if not os.path.exists(self.table_path):
            print(f"警告: 导弹查找表不存在: {self.table_path}")
            print("请先运行 generate_missile_tables.py 生成查找表")
            self._loaded = True
            return
        
        with open(self.table_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.trajectories = data.get('trajectories', {})
        self._loaded = True
        
        print(f"已加载导弹查找表: {len(self.trajectories)} 条轨迹")
    
    def get_trajectory(self, carrier_speed: float) -> List[Dict]:
        """
        获取指定载机速度的导弹飞行轨迹
        
        Args:
            carrier_speed: 载机速度 (m/s)
            
        Returns:
            轨迹点列表 [{time, distance, speed, mach}, ...]
        """
        if not self._loaded:
            self.load()
        
        if not self.trajectories:
            return []
        
        # 找到最接近的速度
        speed_min = self.metadata.get('speed_min', 200)
        speed_max = self.metadata.get('speed_max', 500)
        speed_step = self.metadata.get('speed_step', 20)
        
        # 限制在有效范围内
        speed = max(speed_min, min(speed_max, carrier_speed))
        
        # 四舍五入到最近的采样点
        rounded_speed = round(speed / speed_step) * speed_step
        rounded_speed = max(speed_min, min(speed_max, rounded_speed))
        
        return self.trajectories.get(str(int(rounded_speed)), [])
    
    def interpolate_distance(self, carrier_speed: float, time: float) -> float:
        """
        插值计算指定时间点的导弹飞行距离
        
        Args:
            carrier_speed: 载机速度
            time: 飞行时间
            
        Returns:
            飞行距离 (m)
        """
        trajectory = self.get_trajectory(carrier_speed)
        if not trajectory:
            return time * carrier_speed  # 简单估算
        
        # 线性插值
        prev_point = trajectory[0]
        for point in trajectory[1:]:
            if point['time'] >= time:
                # 在 prev_point 和 point 之间插值
                t0, d0 = prev_point['time'], prev_point['distance']
                t1, d1 = point['time'], point['distance']
                if t1 > t0:
                    ratio = (time - t0) / (t1 - t0)
                    return d0 + (d1 - d0) * ratio
                return d0
            prev_point = point
        
        # 超出表范围
        return trajectory[-1]['distance'] if trajectory else 0


class ThreatConeGenerator:
    """
    威胁锥几何生成器
    
    威胁锥表示敌方导弹在未来20秒内的可达范围。
    几何结构：
        - 母线: 从载机位置向前半球延伸的3条线（正前方、左侧90°、右侧90°）
        - 时间截面: 5/10/15/20秒时刻的椭圆截面（表示该时刻导弹可达边界）
    
    侧向衰减:
        导弹向侧方发射时，需要先转向，消耗能量，
        因此侧向射程只有正前方的50%（side_factor=0.5）。
    """
    
    def __init__(self, missile_tables: MissileTables):
        self.missile_tables = missile_tables
        
        # 威胁锥参数
        self.front_angle = 0  # 正前方角度
        self.side_factor = 0.5  # 侧向距离衰减因子
        self.num_meridians = 3  # 母线数量（正前方和两侧，共3条）
        self.num_time_slices = 4  # 时间截面数量（5s, 10s, 15s, 20s）
        
    def generate_cone_geometry(self, carrier_x: float, carrier_y: float, 
                               carrier_angle: float, carrier_speed: float,
                               future_duration: float = 20.0) -> Dict:
        """
        生成威胁锥几何数据
        
        Args:
            carrier_x, carrier_y: 载机位置
            carrier_angle: 载机朝向角度（度）
            carrier_speed: 载机速度
            future_duration: 未来时间窗口
            
        Returns:
            威胁锥几何数据
        """
        trajectory = self.missile_tables.get_trajectory(carrier_speed)
        if not trajectory:
            return {'meridians': [], 'ellipses': []}
        
        angle_rad = math.radians(carrier_angle)
        
        # 生成母线
        meridians = []
        
        # 定义母线角度偏移（相对于载机朝向）
        # 只画前半球：-90° 到 +90°
        meridian_angles = np.linspace(-90, 90, self.num_meridians)
        
        for offset_angle in meridian_angles:
            # 计算母线方向
            meridian_angle = carrier_angle + offset_angle
            meridian_rad = math.radians(meridian_angle)
            
            # 侧向衰减因子
            side_ratio = abs(math.cos(math.radians(offset_angle)))
            # 侧向使用 side_factor
            distance_factor = side_ratio + (1 - side_ratio) * self.side_factor
            
            # 生成母线上的点
            points = []
            for point in trajectory:
                if point['time'] > future_duration:
                    break
                    
                distance = point['distance'] * distance_factor
                
                # 计算空间位置
                px = carrier_x + math.cos(meridian_rad) * distance
                py = carrier_y + math.sin(meridian_rad) * distance
                
                points.append(SpacetimePoint(px, py, -point['time'], point['speed']))
            
            if points:
                meridians.append({
                    'angle_offset': offset_angle,
                    'points': points
                })
        
        # 生成时间截面椭圆（0秒不绘制，因为是一个点）
        ellipses = []
        time_slices = [5, 10, 15, 20]
        
        for t in time_slices:
            if t > future_duration:
                break
                
            # 获取该时间点的导弹飞行距离
            front_distance = self.missile_tables.interpolate_distance(carrier_speed, t)
            side_distance = front_distance * self.side_factor
            
            # 生成半椭圆点（前半球）
            ellipse_points = []
            angles = np.linspace(-90, 90, 19)  # 每10度一个点
            
            for offset_angle in angles:
                offset_rad = math.radians(offset_angle)
                
                # 椭圆插值
                side_ratio = abs(math.sin(offset_rad))
                front_ratio = abs(math.cos(offset_rad))
                
                # 椭圆半径
                r = 1.0 / math.sqrt(
                    (front_ratio / (front_distance + 1e-6)) ** 2 +
                    (side_ratio / (side_distance + 1e-6)) ** 2 + 1e-12
                ) if (front_distance > 0 or side_distance > 0) else 0
                
                # 方向
                point_angle = carrier_angle + offset_angle
                point_rad = math.radians(point_angle)
                
                px = carrier_x + math.cos(point_rad) * r
                py = carrier_y + math.sin(point_rad) * r
                
                ellipse_points.append(SpacetimePoint(px, py, -t))
            
            ellipses.append({
                'time': t,
                'points': ellipse_points
            })
        
        return {
            'meridians': meridians,
            'ellipses': ellipses,
            'carrier_pos': (carrier_x, carrier_y),
            'carrier_angle': carrier_angle,
            'carrier_speed': carrier_speed
        }


# =============================================================================
# 轨迹预测器
# =============================================================================

class TrajectoryPredictor:
    """
    轨迹预测器 - 预测飞机和导弹的未杨20秒轨迹
    
    预测类型:
        1. 当前舵量预测: 保持当前操纵的轨迹
        2. 全力回转逃离: 全舵量左/右转，背离敌人后直线飞行
        3. 导弹轨迹预测: 追踪目标预测轨迹，考虑制导能量损耗
    
    物理模型:
        采用与游戏一致的简化气动模型：
        - 阻力 = Cd * v²，其中Cd = Cd0 + k*Cl²（诱导阻力）
        - 向心加速度 = v² * Cl_max * rudder
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            from config import CONFIG
            config = CONFIG
        
        self.config = config
        
        # 飞机物理参数
        self.fighter_terminal_velocity = config.get('FIGHTER_TERMINAL_VELOCITY', 400)
        self.fighter_min_turn_radius = config.get('FIGHTER_MIN_TURN_RADIUS', 1000)
        self.fighter_cl_max = 1 / self.fighter_min_turn_radius
        self.fighter_max_thrust = config.get('FIGHTER_MAX_THRUST', 1.5 * 9.8)
        self.fighter_lift_drag_ratio = config.get('FIGHTER_LIFT_DRAG_RATIO', 5)
        
        # 导弹物理参数
        self.missile_terminal_velocity = config.get('MISSILE_TERMINAL_VELOCITY', 400)
        self.missile_min_turn_radius = config.get('MISSILE_MIN_TURN_RADIUS', 1000)
        self.missile_cl_max = 1 / self.missile_min_turn_radius
        self.missile_thrust = config.get('MISSILE_THRUST', 15 * 9.8)
        self.missile_engine_duration = config.get('MISSILE_ENGINE_DURATION', 10.0)
        self.missile_lift_drag_ratio = config.get('MISSILE_LIFT_DRAG_RATIO', 2)
        
        self.G = config.get('G', 9.8)
        self.no_threat_speed = 340.0  # 1马赫
        
        # 预测参数
        self.prediction_dt = 0.2  # 预测步长（较大以提高性能）
        self.prediction_duration = 20.0  # 预测时长（20秒）
        
    def _simulate_step(self, x: float, y: float, vx: float, vy: float,
                       rudder: float, throttle: float, is_missile: bool,
                       engine_time: float, dt: float) -> Tuple:
        """
        单步物理模拟
        
        物理计算流程:
            1. 计算当前速度方向(nx,ny)和垂直方向(pnx,pny)
            2. 计算阻力系数 Cd = Cd0 + k*Cl²
            3. 计算并行加速度(推力-阻力)和向心加速度(转向)
            4. 更新速度和位置
        
        Args:
            x, y: 当前位置
            vx, vy: 当前速度分量
            rudder: 舵量 (-1 到 1)
            throttle: 油门 (0 到 1)
            is_missile: 是否为导弹
            engine_time: 导弹剩余发动机时间
            dt: 时间步长
            
        Returns:
            (x, y, vx, vy, speed, angle, engine_time)
        """
        
        if is_missile:
            terminal_velocity = self.missile_terminal_velocity
            cl_max = self.missile_cl_max
            max_thrust = self.missile_thrust
            lift_drag_ratio = self.missile_lift_drag_ratio
            thrust_accel = max_thrust if engine_time > 0 else 0.0
            engine_time = max(0, engine_time - dt)
        else:
            terminal_velocity = self.fighter_terminal_velocity
            cl_max = self.fighter_cl_max
            max_thrust = self.fighter_max_thrust
            lift_drag_ratio = self.fighter_lift_drag_ratio
            thrust_accel = throttle * max_thrust
        
        epsilon = 1e-7
        vSquare = vx * vx + vy * vy
        inv_v = 1.0 / math.sqrt(vSquare + epsilon)
        
        nx = vx * inv_v
        ny = vy * inv_v
        pnx = -ny
        pny = nx
        
        Cd0 = self.G / (terminal_velocity ** 2)
        Cl_intermediate = abs(rudder) * cl_max
        k = 1.0 / (4.0 * Cd0 * (lift_drag_ratio ** 2) + epsilon)
        Cd = Cd0 + k * (Cl_intermediate ** 2)
        
        drag_accel = Cd * vSquare
        parallel_accel_mag = thrust_accel - drag_accel
        centripetal_accel_mag = vSquare * cl_max * rudder
        
        ax = nx * parallel_accel_mag + pnx * centripetal_accel_mag
        ay = ny * parallel_accel_mag + pny * centripetal_accel_mag
        
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        
        speed = math.sqrt(vx_new ** 2 + vy_new ** 2)
        angle = math.degrees(math.atan2(vy_new, vx_new))
        
        return x_new, y_new, vx_new, vy_new, speed, angle, engine_time
    
    def _get_attr(self, obj, attr, default=0):
        """安全获取属性，支持对象和字典"""
        if hasattr(obj, attr):
            return getattr(obj, attr, default)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    
    def predict_aircraft_trajectory(self, aircraft, rudder: float,
                                    duration: float = None) -> List[SpacetimePoint]:
        """
        预测飞机轨迹
        
        Args:
            aircraft: 飞机对象（需要有 x, y, vx, vy, throttle 属性）
            rudder: 舵量 (-1 to 1)
            duration: 预测时长
            
        Returns:
            预测轨迹点列表
        """
        if duration is None:
            duration = self.prediction_duration
        
        # 获取初始状态
        x = self._get_attr(aircraft, 'x', 0)
        y = self._get_attr(aircraft, 'y', 0)
        vx = self._get_attr(aircraft, 'vx', 0)
        vy = self._get_attr(aircraft, 'vy', 0)
        throttle = self._get_attr(aircraft, 'throttle', 1.0)
        
        points = []
        time = 0.0
        
        while time <= duration:
            speed = math.sqrt(vx * vx + vy * vy)
            angle = math.degrees(math.atan2(vy, vx))
            points.append(SpacetimePoint(x, y, -time, speed, angle))
            
            x, y, vx, vy, speed, angle, _ = self._simulate_step(
                x, y, vx, vy, rudder, throttle, False, 0, self.prediction_dt
            )
            time += self.prediction_dt
        
        return points
    
    def predict_escape_trajectory(self, aircraft, rudder: float, 
                                   enemy_x: float, enemy_y: float,
                                   duration: float = None) -> List[SpacetimePoint]:
        """
        预测逃离敌人的飞机轨迹
        
        转向至完全背离敌人后改为直线飞行
        使用开始时刻的敌我连线方向作为回转目标
        
        Args:
            aircraft: 飞机对象
            rudder: 初始舵量 (-1 或 1)
            enemy_x, enemy_y: 敌人位置
            duration: 预测时长
            
        Returns:
            预测轨迹点列表
        """
        if duration is None:
            duration = self.prediction_duration
        
        # 获取初始状态
        x = self._get_attr(aircraft, 'x', 0)
        y = self._get_attr(aircraft, 'y', 0)
        vx = self._get_attr(aircraft, 'vx', 0)
        vy = self._get_attr(aircraft, 'vy', 0)
        throttle = self._get_attr(aircraft, 'throttle', 1.0)
        
        # 计算开始时刻的逃离方向（与敌我连线相反）
        to_enemy_x = enemy_x - x
        to_enemy_y = enemy_y - y
        enemy_dist = math.sqrt(to_enemy_x * to_enemy_x + to_enemy_y * to_enemy_y)
        
        if enemy_dist > 1:
            # 逃离方向 = 敌我连线的反方向（单位向量）
            escape_dir_x = -to_enemy_x / enemy_dist
            escape_dir_y = -to_enemy_y / enemy_dist
        else:
            # 敌人太近，使用当前速度方向
            speed = math.sqrt(vx * vx + vy * vy)
            if speed > 1:
                escape_dir_x = vx / speed
                escape_dir_y = vy / speed
            else:
                escape_dir_x, escape_dir_y = 1, 0
        
        points = []
        time = 0.0
        current_rudder = rudder
        has_turned_away = False  # 是否已经转到逃离方向
        
        # 背离判定阈值：速度方向与逃离方向的点积 > 0.95 (约≤ 18度)
        ESCAPE_THRESHOLD = 0.95
        
        while time <= duration:
            speed = math.sqrt(vx * vx + vy * vy)
            angle = math.degrees(math.atan2(vy, vx))
            points.append(SpacetimePoint(x, y, -time, speed, angle))
            
            # 检查是否已转到逃离方向
            if not has_turned_away and speed > 1:
                # 当前速度方向（单位向量）
                vel_dir_x = vx / speed
                vel_dir_y = vy / speed
                
                # 计算速度方向与逃离方向的点积
                dot_product = vel_dir_x * escape_dir_x + vel_dir_y * escape_dir_y
                
                if dot_product >= ESCAPE_THRESHOLD:
                    # 已转到逃离方向，改为直线飞行
                    has_turned_away = True
                    current_rudder = 0
            
            x, y, vx, vy, speed, angle, _ = self._simulate_step(
                x, y, vx, vy, current_rudder, throttle, False, 0, self.prediction_dt
            )
            time += self.prediction_dt
        
        return points
    
    def predict_missile_trajectory(self, missile, target_trajectory: List[SpacetimePoint],
                                   duration: float = None) -> List[SpacetimePoint]:
        """
        预测导弹轨迹（追踪目标的预测轨迹）
        
        无威胁判定：导弹曾超过音速后减速到音速以下
        考虑制导转向消耗能量的情况
        
        Args:
            missile: 导弹对象
            target_trajectory: 目标的预测轨迹
            duration: 预测时长
            
        Returns:
            预测轨迹点列表
        """
        if duration is None:
            duration = self.prediction_duration
        
        # 获取初始状态
        x = self._get_attr(missile, 'x', 0)
        y = self._get_attr(missile, 'y', 0)
        vx = self._get_attr(missile, 'vx', 0)
        vy = self._get_attr(missile, 'vy', 0)
        engine_time = self._get_attr(missile, 'engine_time', 10.0)
        
        points = []
        time = 0.0
        step_idx = 0
        
        # 跟踪导弹是否曾超过音速
        initial_speed = math.sqrt(vx * vx + vy * vy)
        ever_supersonic = initial_speed >= self.no_threat_speed
        
        while time <= duration:
            speed = math.sqrt(vx * vx + vy * vy)
            angle = math.degrees(math.atan2(vy, vx))
            points.append(SpacetimePoint(x, y, -time, speed, angle))
            
            # 更新是否曾超音速
            if speed >= self.no_threat_speed:
                ever_supersonic = True
            
            # 检查是否已无威胁（必须曾超过音速，再减速到音速以下才算无威胁）
            if ever_supersonic and speed < self.no_threat_speed:
                break
            
            # 获取目标当前预测位置（线性插值）
            if target_trajectory and step_idx < len(target_trajectory):
                target_point = target_trajectory[step_idx]
                target_x, target_y = target_point.x, target_point.y
            elif target_trajectory:
                target_point = target_trajectory[-1]
                target_x, target_y = target_point.x, target_point.y
            else:
                # 没有目标轨迹，导弹保持当前方向直飞（rudder=0）
                rudder = 0.0
                x, y, vx, vy, speed, angle, engine_time = self._simulate_step(
                    x, y, vx, vy, rudder, 1.0, True, engine_time, self.prediction_dt
                )
                time += self.prediction_dt
                step_idx += 1
                continue
            
            # 计算制导舵量（比例导引简化版）
            dx = target_x - x
            dy = target_y - y
            target_angle = math.degrees(math.atan2(dy, dx))
            
            angle_diff = (target_angle - angle + 180) % 360 - 180
            rudder = max(-1.0, min(1.0, angle_diff / 45.0))
            
            x, y, vx, vy, speed, angle, engine_time = self._simulate_step(
                x, y, vx, vy, rudder, 1.0, True, engine_time, self.prediction_dt
            )
            time += self.prediction_dt
            step_idx += 1
        
        return points


# =============================================================================
# 时空图计算综合管理器
# =============================================================================

class SpacetimeComputer:
    """
    时空图计算综合管理器
    
    负责统一管理时空图的所有计算：
        - 历史轨迹记录（5秒，用于内部处理，不显示）
        - 未来轨迹预测Ｈ20秒）
        - 敌方威胁锥生成
        - 预测时间戳管理（用于时间同步）
    
    更新策略:
        - 历史记录: 每帧更新
        - 预测计算: 每0.5秒更新一次（性能优化）
    
    使用流程:
        1. initialize() - 初始化（加载导弹查找表）
        2. update() - 每帧调用，更新时空数据
        3. get_*() - 获取数据供渲染器使用
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            from config import CONFIG
            config = CONFIG
        
        self.config = config
        
        # 时间窗口
        self.future_duration = 20.0  # 未杨20秒
        
        # 子模块
        self.missile_tables = MissileTables()
        self.threat_cone_gen = ThreatConeGenerator(self.missile_tables)
        self.trajectory_predictor = TrajectoryPredictor(config)
        
        # 玩家时空数据
        self.player_trails: Dict[int, SpacetimeTrail] = {}
        
        # 导弹时空数据
        self.missile_trails: Dict[int, SpacetimeTrail] = {}
        
        # 更新间隔
        self.update_interval = 0.5  # 每0.5秒更新预测
        self._last_update_time = 0.0
        
        # 当前游戏时间
        self._current_time = 0.0
    
    def _get_attr(self, obj, attr, default=0):
        """安全获取属性，支持对象和字典"""
        if hasattr(obj, attr):
            return getattr(obj, attr, default)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    
    def _generate_simple_trajectory(self, target) -> List[SpacetimePoint]:
        """根据目标当前状态生成简单直线轨迹（用于备用）"""
        x = self._get_attr(target, 'x', 0)
        y = self._get_attr(target, 'y', 0)
        vx = self._get_attr(target, 'vx', 0)
        vy = self._get_attr(target, 'vy', 0)
        speed = math.sqrt(vx * vx + vy * vy)
        angle = math.degrees(math.atan2(vy, vx))
        
        points = []
        dt = 0.2  # 时间步长
        for i in range(int(self.future_duration / dt) + 1):
            t = i * dt
            px = x + vx * t
            py = y + vy * t
            points.append(SpacetimePoint(px, py, -t, speed, angle))
        
        return points
        
    def initialize(self):
        """初始化（加载查找表等）"""
        self.missile_tables.load()
        
        # 创建两个玩家的轨迹记录
        self.player_trails[1] = SpacetimeTrail()
        self.player_trails[2] = SpacetimeTrail()
        
    def update(self, aircraft1, aircraft2, missiles: List, 
               game_time: float, force_update: bool = False):
        """
        更新时空图数据
        
        Args:
            aircraft1: 玩家1飞机
            aircraft2: 玩家2飞机
            missiles: 导弹列表
            game_time: 当前游戏时间
            force_update: 是否强制更新预测
        """
        self._current_time = game_time
        
        # 管理导弹预测对象生命周期
        active_missile_ids = set()
        for missile in missiles:
            if not getattr(missile, 'alive', True):
                continue
            # 优先使用 slot_idx (Tensor后端) 以保持 ID 稳定
            missile_id = getattr(missile, 'slot_idx', id(missile))
            active_missile_ids.add(missile_id)
            
            if missile_id not in self.missile_trails:
                self.missile_trails[missile_id] = SpacetimeTrail()
        
        # 清理已消失的导弹
        dead_ids = set(self.missile_trails.keys()) - active_missile_ids
        for dead_id in dead_ids:
            del self.missile_trails[dead_id]
        
        # 检查是否需要更新预测
        should_update = force_update or (game_time - self._last_update_time >= self.update_interval)
        
        if should_update:
            self._update_predictions(aircraft1, aircraft2, missiles)
            self._last_update_time = game_time
    
    def _update_predictions(self, aircraft1, aircraft2, missiles):
        """更新所有预测轨迹"""
        current_time = self._current_time
        
        # 玩家1的预测
        if hasattr(aircraft1, 'alive') and aircraft1.alive:
            rudder = getattr(aircraft1, 'rudder', 0)
            
            # 当前舵量预测
            current_pred = self.trajectory_predictor.predict_aircraft_trajectory(
                aircraft1, rudder
            )
            self.player_trails[1].set_prediction('current', current_pred, current_time)
            
            # 全力回转逃离敌人（背离敌人后直线飞行）
            enemy_x = self._get_attr(aircraft2, 'x', 0)
            enemy_y = self._get_attr(aircraft2, 'y', 0)
            
            left_pred = self.trajectory_predictor.predict_escape_trajectory(
                aircraft1, -1.0, enemy_x, enemy_y
            )
            self.player_trails[1].set_prediction('left_turn', left_pred, current_time)
            
            right_pred = self.trajectory_predictor.predict_escape_trajectory(
                aircraft1, 1.0, enemy_x, enemy_y
            )
            self.player_trails[1].set_prediction('right_turn', right_pred, current_time)
        
        # 玩家2的预测
        if hasattr(aircraft2, 'alive') and aircraft2.alive:
            rudder = getattr(aircraft2, 'rudder', 0)
            
            current_pred = self.trajectory_predictor.predict_aircraft_trajectory(
                aircraft2, rudder
            )
            self.player_trails[2].set_prediction('current', current_pred, current_time)
            
            # 全力回转逃离敌人
            enemy_x = self._get_attr(aircraft1, 'x', 0)
            enemy_y = self._get_attr(aircraft1, 'y', 0)
            
            left_pred = self.trajectory_predictor.predict_escape_trajectory(
                aircraft2, -1.0, enemy_x, enemy_y
            )
            self.player_trails[2].set_prediction('left_turn', left_pred, current_time)
            
            right_pred = self.trajectory_predictor.predict_escape_trajectory(
                aircraft2, 1.0, enemy_x, enemy_y
            )
            self.player_trails[2].set_prediction('right_turn', right_pred, current_time)
        
        # 导弹预测
        for missile in missiles:
            if not getattr(missile, 'alive', True):
                continue
            missile_id = getattr(missile, 'slot_idx', id(missile))
            
            if missile_id not in self.missile_trails:
                continue
            
            # 获取目标的当前舵量预测轨迹
            target = getattr(missile, 'target', None)
            if target and hasattr(target, 'is_player1'):
                # 目标是玩家1则获取玩家1的预测轨迹
                target_player = 1 if target.is_player1 else 2
                target_pred = self.player_trails.get(target_player, SpacetimeTrail())
                target_trajectory = target_pred.get_prediction('current')
                # 如果目标预测轨迹为空，使用目标当前位置生成简单直线轨迹
                if not target_trajectory:
                    target_trajectory = self._generate_simple_trajectory(target)
            else:
                target_trajectory = []
            
            missile_pred = self.trajectory_predictor.predict_missile_trajectory(
                missile, target_trajectory
            )
            self.missile_trails[missile_id].set_prediction('trajectory', missile_pred, current_time)
    
    def get_threat_cone(self, enemy, for_player: int) -> Dict:
        """
        获取敌方的威胁锥几何数据
        
        Args:
            enemy: 敌方飞机
            for_player: 观察者玩家ID
            
        Returns:
            威胁锥几何数据
        """
        if not hasattr(enemy, 'alive') or not enemy.alive:
            return {'meridians': [], 'ellipses': []}
        
        return self.threat_cone_gen.generate_cone_geometry(
            enemy.x, enemy.y, enemy.angle, enemy.speed,
            self.future_duration
        )
    
    def get_player_spacetime_data(self, player_id: int) -> Dict:
        """
        获取玩家的时空图数据
        
        Args:
            player_id: 玩家ID (1 或 2)
            
        Returns:
            时空图数据字典
        """
        trail = self.player_trails.get(player_id)
        if trail is None:
            return {}
        
        return {
            'current_prediction': trail.get_prediction('current'),
            'current_prediction_time': trail.get_prediction_time('current'),
            'left_turn_prediction': trail.get_prediction('left_turn'),
            'left_turn_prediction_time': trail.get_prediction_time('left_turn'),
            'right_turn_prediction': trail.get_prediction('right_turn'),
            'right_turn_prediction_time': trail.get_prediction_time('right_turn'),
        }
    
    def get_missile_spacetime_data(self, missile_id: int) -> Dict:
        """获取导弹的时空图数据"""
        trail = self.missile_trails.get(missile_id)
        if trail is None:
            return {}
        
        return {
            'trajectory_prediction': trail.get_prediction('trajectory'),
            'trajectory_prediction_time': trail.get_prediction_time('trajectory'),
        }
    
    def get_current_time(self) -> float:
        """获取当前游戏时间"""
        return self._current_time
    
    def get_sparse_features(self, player_id: int, num_points: int = 10) -> np.ndarray:
        """
        获取稀疏采样的时空特征（用于RL输入）
        
        Args:
            player_id: 玩家ID
            num_points: 每条轨迹的采样点数
            
        Returns:
            特征数组 shape: (num_features,)
        """
        features = []
        
        trail = self.player_trails.get(player_id)
        if trail is None:
            return np.zeros(num_points * 4 * 3)  # 3条预测线，每点4个值
        
        # 采样当前预测
        for pred_name in ['current', 'left_turn', 'right_turn']:
            pred = trail.get_prediction(pred_name)
            if pred:
                # 均匀采样
                indices = np.linspace(0, len(pred) - 1, num_points, dtype=int)
                for idx in indices:
                    pt = pred[idx]
                    features.extend([pt.x, pt.y, pt.t, pt.speed])
            else:
                features.extend([0] * (num_points * 4))
        
        return np.array(features, dtype=np.float32)
