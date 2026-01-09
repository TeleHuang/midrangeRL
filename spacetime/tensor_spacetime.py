# -*- coding: utf-8 -*-
"""
GPU并行时空图计算模块 (Tensor Spacetime)

功能概述:
    在GPU上批量计算时空图数据，支持数百到上千环境并行：
    - 威胁锥批量生成：基于预计算查找表的向量化插值
    - 轨迹预测批量生成：飞机/导弹的未来轨迹预测
    - 时空特征提取：稀疏采样供agent观测和奖励计算使用

设计原则:
    - 与TensorEnv架构对齐：批处理维度[num_envs, ...]
    - 预计算查找表直接加载到GPU，避免运行时重复计算
    - 纯向量化操作，无Python循环（除初始化）
    - 最小化CPU-GPU数据传输

性能目标:
    - 256-1024环境并行时保持高性能
    - 对纯CPU版本有绝对性能优势

模块组成:
    - TensorMissileLookup: GPU版导弹飞行查找表
    - TensorThreatCone: GPU版威胁锥批量计算
    - TensorTrajectoryPredictor: GPU版轨迹预测器
    - TensorSpacetimeComputer: 综合管理器

依赖:
    - torch: GPU张量计算
    - config: 物理参数配置
"""

import torch
import math
import json
import os
from typing import Dict, Tuple, Optional, List
from spacetime.spacetime_core import SpacetimePoint


# =============================================================================
# GPU版导弹飞行查找表
# =============================================================================

class TensorMissileLookup:
    """
    GPU版导弹飞行查找表
    
    将预计算的导弹飞行轨迹数据加载到GPU，
    支持批量向量化插值查询。
    
    查找表结构:
        - time_table: [num_speeds, num_time_points] 时间点
        - distance_table: [num_speeds, num_time_points] 对应距离
        - speed_table: [num_speeds, num_time_points] 对应速度
        
    速度映射:
        carrier_speed → 表索引 (200-500 m/s → 0-15)
    """
    
    def __init__(self, table_path: str = None, device: str = 'cuda'):
        self._device = torch.device(device) if isinstance(device, str) else device
        
        if table_path is None:
            table_path = os.path.join(os.path.dirname(__file__), 'missile_tables.json')
        
        self.table_path = table_path
        self._loaded = False
        
        # 查找表张量 (初始化为空)
        self.time_table: Optional[torch.Tensor] = None      # [num_speeds, max_points]
        self.distance_table: Optional[torch.Tensor] = None  # [num_speeds, max_points]
        self.speed_table: Optional[torch.Tensor] = None     # [num_speeds, max_points]
        self.num_points: Optional[torch.Tensor] = None      # [num_speeds] 每个速度的有效点数
        
        # 速度范围元数据
        self.speed_min = 200.0
        self.speed_max = 500.0
        self.speed_step = 20.0
        self.num_speeds = 16  # (500-200)/20 + 1
        
    def load(self):
        """加载查找表到GPU"""
        if self._loaded:
            return
        
        if not os.path.exists(self.table_path):
            print(f"警告: 导弹查找表不存在: {self.table_path}")
            self._create_default_table()
            self._loaded = True
            return
        
        with open(self.table_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        self.speed_min = float(metadata.get('speed_min', 200))
        self.speed_max = float(metadata.get('speed_max', 500))
        self.speed_step = float(metadata.get('speed_step', 20))
        self.num_speeds = int((self.speed_max - self.speed_min) / self.speed_step) + 1
        
        trajectories = data.get('trajectories', {})
        
        # 找出最大点数
        max_points = 0
        for speed_str, traj in trajectories.items():
            max_points = max(max_points, len(traj))
        
        # 创建张量
        time_list = []
        distance_list = []
        speed_list = []
        num_points_list = []
        
        for i in range(self.num_speeds):
            speed = int(self.speed_min + i * self.speed_step)
            traj = trajectories.get(str(speed), [])
            
            times = [p['time'] for p in traj]
            distances = [p['distance'] for p in traj]
            speeds = [p['speed'] for p in traj]
            
            # 填充到max_points
            n = len(times)
            times.extend([times[-1] if times else 0] * (max_points - n))
            distances.extend([distances[-1] if distances else 0] * (max_points - n))
            speeds.extend([speeds[-1] if speeds else 0] * (max_points - n))
            
            time_list.append(times)
            distance_list.append(distances)
            speed_list.append(speeds)
            num_points_list.append(n)
        
        self.time_table = torch.tensor(time_list, dtype=torch.float32, device=self._device)
        self.distance_table = torch.tensor(distance_list, dtype=torch.float32, device=self._device)
        self.speed_table = torch.tensor(speed_list, dtype=torch.float32, device=self._device)
        self.num_points = torch.tensor(num_points_list, dtype=torch.long, device=self._device)
        self.max_points = max_points
        
        self._loaded = True
        print(f"[TensorMissileLookup] 已加载查找表到GPU: {self.num_speeds}个速度档位, 最大{max_points}个时间点")
    
    def _create_default_table(self):
        """创建默认查找表（简化模型）"""
        max_points = 121  # 0-60秒，每0.5秒一个点
        
        time_list = []
        distance_list = []
        speed_list = []
        
        for i in range(self.num_speeds):
            carrier_speed = self.speed_min + i * self.speed_step
            times = []
            distances = []
            speeds = []
            
            # 简化物理模型
            speed = carrier_speed
            distance = 0.0
            dt = 0.5
            
            for _ in range(max_points):
                t = len(times) * dt
                times.append(t)
                distances.append(distance)
                speeds.append(speed)
                
                # 简化减速模型
                if t < 10.0:  # 发动机阶段
                    speed = min(speed + 50 * dt, 1500)  # 加速
                else:
                    speed = max(speed - 10 * dt, 50)  # 减速
                distance += speed * dt
            
            time_list.append(times)
            distance_list.append(distances)
            speed_list.append(speeds)
        
        self.time_table = torch.tensor(time_list, dtype=torch.float32, device=self._device)
        self.distance_table = torch.tensor(distance_list, dtype=torch.float32, device=self._device)
        self.speed_table = torch.tensor(speed_list, dtype=torch.float32, device=self._device)
        self.num_points = torch.full((self.num_speeds,), max_points, dtype=torch.long, device=self._device)
        self.max_points = max_points
    
    def get_speed_index(self, carrier_speed: torch.Tensor) -> torch.Tensor:
        """
        将载机速度映射到查找表索引
        
        Args:
            carrier_speed: [num_envs] 或 [num_envs, num_entities] 载机速度
            
        Returns:
            [same shape] 索引 (0 到 num_speeds-1)
        """
        # 限制在有效范围内
        clamped = torch.clamp(carrier_speed, self.speed_min, self.speed_max)
        # 四舍五入到最近的采样点
        index = torch.round((clamped - self.speed_min) / self.speed_step).long()
        return torch.clamp(index, 0, self.num_speeds - 1)
    
    def interpolate_distance_batch(self, carrier_speed: torch.Tensor, 
                                   time: torch.Tensor) -> torch.Tensor:
        """
        批量插值计算指定时间点的导弹飞行距离
        
        Args:
            carrier_speed: [batch_size] 载机速度
            time: [batch_size] 或 [batch_size, num_times] 飞行时间
            
        Returns:
            [same shape as time] 飞行距离
        """
        if not self._loaded:
            self.load()
        
        speed_idx = self.get_speed_index(carrier_speed)  # [batch_size]
        
        # 处理time维度
        if time.dim() == 1:
            time = time.unsqueeze(-1)  # [batch_size, 1]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = carrier_speed.shape[0]
        num_times = time.shape[1]
        
        # 获取对应速度的时间和距离表
        # time_table[speed_idx]: [batch_size, max_points]
        time_vals = self.time_table[speed_idx]  # [batch_size, max_points]
        dist_vals = self.distance_table[speed_idx]  # [batch_size, max_points]
        
        # 扩展time到 [batch_size, num_times, 1] 用于比较
        time_expanded = time.unsqueeze(-1)  # [batch_size, num_times, 1]
        time_vals_expanded = time_vals.unsqueeze(1)  # [batch_size, 1, max_points]
        
        # 找到time在表中的位置 (二分查找的向量化版本)
        # 使用searchsorted会更快，但这里用比较实现兼容性更好
        # 找到第一个 >= time 的点
        greater_mask = time_vals_expanded >= time_expanded  # [batch_size, num_times, max_points]
        
        # 获取上界索引
        upper_idx = greater_mask.long().argmax(dim=-1)  # [batch_size, num_times]
        upper_idx = torch.clamp(upper_idx, 1, self.max_points - 1)
        lower_idx = upper_idx - 1
        
        # 批量索引取值
        batch_indices = torch.arange(batch_size, device=self._device).unsqueeze(1).expand(-1, num_times)
        
        t0 = time_vals[batch_indices, lower_idx]  # [batch_size, num_times]
        t1 = time_vals[batch_indices, upper_idx]
        d0 = dist_vals[batch_indices, lower_idx]
        d1 = dist_vals[batch_indices, upper_idx]
        
        # 线性插值
        dt = t1 - t0 + 1e-7
        ratio = (time - t0) / dt
        ratio = torch.clamp(ratio, 0, 1)
        
        distance = d0 + (d1 - d0) * ratio
        
        if squeeze_output:
            distance = distance.squeeze(-1)
        
        return distance


# =============================================================================
# GPU版威胁锥批量计算
# =============================================================================

class TensorThreatCone:
    """
    GPU版威胁锥批量计算
    
    批量生成所有环境中敌方飞机的威胁锥几何数据。
    
    输出格式（用于特征提取，非渲染）:
        - front_distances: [num_envs, num_time_slices] 正前方距离
        - side_distances: [num_envs, num_time_slices] 侧向距离
        
    时间截面: [5, 10, 15, 20] 秒
    """
    
    def __init__(self, missile_lookup: TensorMissileLookup, device: str = 'cuda'):
        self._device = torch.device(device) if isinstance(device, str) else device
        self.missile_lookup = missile_lookup
        
        # 威胁锥参数
        self.side_factor = 0.5  # 侧向距离衰减因子
        self.time_slices = torch.tensor([5.0, 10.0, 15.0, 20.0], device=self._device, dtype=torch.float32)
        self.num_time_slices = 4
        
        # 用于特征输出的角度采样 (前半球: -90° 到 90°)
        self.num_angles = 9  # 每22.5度一个点
        angles_deg = torch.linspace(-90, 90, self.num_angles, device=self._device, dtype=torch.float32)
        self.angles_rad = angles_deg * (math.pi / 180.0)
        
    def compute_batch(self, carrier_speed: torch.Tensor,
                      carrier_angle: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        批量计算威胁锥数据
        
        Args:
            carrier_speed: [num_envs] 载机速度
            carrier_angle: [num_envs] 载机朝向角度（弧度）
            
        Returns:
            威胁锥特征字典:
                - front_distances: [num_envs, num_time_slices] 正前方距离
                - side_distances: [num_envs, num_time_slices] 侧向距离
                - cone_radii: [num_envs, num_time_slices, num_angles] 各方向距离
        """
        num_envs = carrier_speed.shape[0]
        
        # 获取各时间截面的正前方距离
        # time_slices: [num_time_slices]
        # carrier_speed: [num_envs]
        # 需要: [num_envs, num_time_slices]
        
        time_expanded = self.time_slices.unsqueeze(0).expand(num_envs, -1)  # [num_envs, num_time_slices]
        speed_expanded = carrier_speed.unsqueeze(1).expand(-1, self.num_time_slices)  # [num_envs, num_time_slices]
        
        # 批量插值
        front_distances = self.missile_lookup.interpolate_distance_batch(
            carrier_speed, self.time_slices.unsqueeze(0).expand(num_envs, -1)
        )  # [num_envs, num_time_slices]
        
        # 侧向距离
        side_distances = front_distances * self.side_factor  # [num_envs, num_time_slices]
        
        # 计算各方向的威胁锥半径 (椭圆插值)
        # angles_rad: [num_angles]
        # 扩展到 [num_envs, num_time_slices, num_angles]
        
        cos_angles = torch.cos(self.angles_rad)  # [num_angles]
        sin_angles = torch.sin(self.angles_rad).abs()  # [num_angles]
        
        # front_distances: [num_envs, num_time_slices] -> [num_envs, num_time_slices, 1]
        # side_distances: [num_envs, num_time_slices] -> [num_envs, num_time_slices, 1]
        front_exp = front_distances.unsqueeze(-1)  # [num_envs, num_time_slices, 1]
        side_exp = side_distances.unsqueeze(-1)    # [num_envs, num_time_slices, 1]
        
        # 椭圆半径公式: r = 1 / sqrt((cos/a)² + (sin/b)²)
        # a = front_distance, b = side_distance
        eps = 1e-6
        inv_r_sq = (cos_angles / (front_exp + eps)) ** 2 + (sin_angles / (side_exp + eps)) ** 2
        cone_radii = 1.0 / (torch.sqrt(inv_r_sq) + eps)  # [num_envs, num_time_slices, num_angles]
        
        return {
            'front_distances': front_distances,
            'side_distances': side_distances,
            'cone_radii': cone_radii,
            'time_slices': self.time_slices,
            'angles_rad': self.angles_rad,
        }
    
    def get_threat_features(self, carrier_speed: torch.Tensor,
                           carrier_x: torch.Tensor,
                           carrier_y: torch.Tensor,
                           carrier_angle: torch.Tensor,
                           observer_x: torch.Tensor,
                           observer_y: torch.Tensor) -> torch.Tensor:
        """
        提取威胁锥特征（相对于观察者）
        
        Args:
            carrier_*: [num_envs] 敌方（威胁源）状态
            observer_*: [num_envs] 己方（观察者）状态
            
        Returns:
            [num_envs, feature_dim] 威胁特征向量
        """
        cone_data = self.compute_batch(carrier_speed, carrier_angle)
        front_dist = cone_data['front_distances']  # [num_envs, 4]
        
        # 计算观察者到威胁源的相对位置
        dx = observer_x - carrier_x
        dy = observer_y - carrier_y
        distance = torch.sqrt(dx * dx + dy * dy + 1e-7)
        
        # 计算观察者相对于威胁源朝向的角度
        observer_angle = torch.atan2(dy, dx)  # 从威胁源看观察者的角度
        relative_angle = observer_angle - carrier_angle  # 相对于载机朝向
        
        # 归一化相对角度到 [-pi, pi]
        relative_angle = torch.remainder(relative_angle + math.pi, 2 * math.pi) - math.pi
        
        # 计算该角度方向的威胁锥半径
        cos_rel = torch.cos(relative_angle).abs()
        sin_rel = torch.sin(relative_angle).abs()
        
        side_dist = front_dist * self.side_factor
        eps = 1e-6
        inv_r_sq = (cos_rel.unsqueeze(-1) / (front_dist + eps)) ** 2 + \
                   (sin_rel.unsqueeze(-1) / (side_dist + eps)) ** 2
        threat_radius = 1.0 / (torch.sqrt(inv_r_sq) + eps)  # [num_envs, 4]
        
        # 特征: 观察者距离 / 威胁半径 (< 1 表示在威胁锥内)
        threat_ratio = distance.unsqueeze(-1) / (threat_radius + eps)  # [num_envs, 4]
        
        # 组合特征
        features = torch.cat([
            front_dist / 50000.0,     # 归一化前向距离
            threat_ratio,              # 威胁比例
            (relative_angle / math.pi).unsqueeze(-1),  # 相对角度
        ], dim=-1)  # [num_envs, 9]
        
        return features


# =============================================================================
# GPU版轨迹预测器
# =============================================================================

class TensorTrajectoryPredictor:
    """
    GPU版轨迹预测器
    
    批量预测飞机和导弹的未来轨迹。
    使用纯张量操作，无Python循环。
    
    预测输出:
        - 轨迹点序列: [num_envs, num_steps, 4] (x, y, speed, angle)
        - 稀疏采样特征: [num_envs, feature_dim]
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self._device = torch.device(device) if isinstance(device, str) else device
        
        # 飞机物理参数
        self.fighter_terminal_velocity = torch.tensor(
            config.get('FIGHTER_TERMINAL_VELOCITY', 400), device=self._device, dtype=torch.float32)
        fighter_min_turn_radius = config.get('FIGHTER_MIN_TURN_RADIUS', 1000)
        self.fighter_cl_max = torch.tensor(1.0 / fighter_min_turn_radius, device=self._device, dtype=torch.float32)
        self.fighter_max_thrust = torch.tensor(
            config.get('FIGHTER_MAX_THRUST', 1.5 * 9.8), device=self._device, dtype=torch.float32)
        self.fighter_lift_drag_ratio = torch.tensor(
            config.get('FIGHTER_LIFT_DRAG_RATIO', 5), device=self._device, dtype=torch.float32)
        
        # 导弹物理参数
        self.missile_terminal_velocity = torch.tensor(
            config.get('MISSILE_TERMINAL_VELOCITY', 400), device=self._device, dtype=torch.float32)
        missile_min_turn_radius = config.get('MISSILE_MIN_TURN_RADIUS', 1000)
        self.missile_cl_max = torch.tensor(1.0 / missile_min_turn_radius, device=self._device, dtype=torch.float32)
        self.missile_thrust = torch.tensor(
            config.get('MISSILE_THRUST', 15 * 9.8), device=self._device, dtype=torch.float32)
        self.missile_engine_duration = torch.tensor(
            config.get('MISSILE_ENGINE_DURATION', 10.0), device=self._device, dtype=torch.float32)
        self.missile_lift_drag_ratio = torch.tensor(
            config.get('MISSILE_LIFT_DRAG_RATIO', 2), device=self._device, dtype=torch.float32)
        
        self.G = torch.tensor(9.8, device=self._device, dtype=torch.float32)
        self.no_threat_speed = torch.tensor(340.0, device=self._device, dtype=torch.float32)
        
        # 预测参数 - 优化后减少步数
        self.prediction_dt = 1.0  # 更大步长提高性能
        self.prediction_duration = 20.0
        self.num_steps = int(self.prediction_duration / self.prediction_dt) + 1  # 21步
        
        self.epsilon = 1e-7
    
    def _simulate_step_batch(self, x: torch.Tensor, y: torch.Tensor,
                             vx: torch.Tensor, vy: torch.Tensor,
                             rudder: torch.Tensor, throttle: torch.Tensor,
                             is_missile: torch.Tensor, engine_time: torch.Tensor,
                             dt: float) -> Tuple[torch.Tensor, ...]:
        """
        批量单步物理模拟
        
        Args:
            x, y, vx, vy: [batch_size] 位置和速度
            rudder, throttle: [batch_size] 控制量
            is_missile: [batch_size] bool 是否为导弹
            engine_time: [batch_size] 导弹剩余发动机时间
            dt: 时间步长
            
        Returns:
            更新后的 (x, y, vx, vy, speed, angle, engine_time)
        """
        # 根据实体类型选择参数
        terminal_velocity = torch.where(is_missile, self.missile_terminal_velocity, self.fighter_terminal_velocity)
        cl_max = torch.where(is_missile, self.missile_cl_max, self.fighter_cl_max)
        max_thrust = torch.where(is_missile, self.missile_thrust, self.fighter_max_thrust)
        lift_drag_ratio = torch.where(is_missile, self.missile_lift_drag_ratio, self.fighter_lift_drag_ratio)
        
        # 推力计算
        thrust_active = torch.where(is_missile, (engine_time > self.epsilon).float(), throttle)
        thrust_accel = thrust_active * max_thrust
        engine_time = torch.where(is_missile, torch.clamp(engine_time - dt, min=0.0), engine_time)
        
        # 核心物理计算
        vSquare = vx * vx + vy * vy
        inv_v = torch.rsqrt(vSquare + self.epsilon)
        
        nx = vx * inv_v
        ny = vy * inv_v
        pnx = -ny
        pny = nx
        
        # 空气动力学
        Cd0 = self.G / (terminal_velocity ** 2 + self.epsilon)
        Cl_intermediate_sq = (torch.abs(rudder) * cl_max) ** 2
        k = 1.0 / (4.0 * Cd0 * (lift_drag_ratio ** 2) + self.epsilon)
        Cd = Cd0 + k * Cl_intermediate_sq
        
        drag_accel = Cd * vSquare
        parallel_accel_mag = thrust_accel - drag_accel
        centripetal_accel_mag = vSquare * cl_max * rudder
        
        ax = nx * parallel_accel_mag + pnx * centripetal_accel_mag
        ay = ny * parallel_accel_mag + pny * centripetal_accel_mag
        
        # 更新状态 - 使用中点积分法减小误差
        # 先用欧拉法预测半步
        vx_half = vx + ax * (dt * 0.5)
        vy_half = vy + ay * (dt * 0.5)
        
        # 重新计算半步位置的受力（简化处理，只更新方向相关项）
        # 这里为了性能，只用半步速度修正位移，不完全重算受力
        x_new = x + vx_half * dt
        y_new = y + vy_half * dt
        
        # 速度更新仍用欧拉法（或者也可以改进，但位移误差通常更关键）
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        
        # 额外的数值稳定性修正：对于向心力导致的加速
        # 理论上向心力不改变速率，但离散积分会导致速率增加
        # 修正方法：强制向心力分量只改变方向不改变速率
        # 或者更简单的：分离切向和法向积分
        
        # === 改进的积分方法 ===
        # 切向加速度 (改变速率)
        # parallel_accel_mag
        
        # 法向加速度 (改变方向)
        # centripetal_accel_mag
        
        # 1. 更新速率
        v_current = torch.sqrt(vSquare + self.epsilon)
        v_new_mag = v_current + parallel_accel_mag * dt
        v_new_mag = torch.clamp(v_new_mag, min=0.0)
        
        # 2. 更新方向
        # 角速度 omega = a_n / v
        omega = centripetal_accel_mag / (v_current + self.epsilon)
        angle_current = torch.atan2(vy, vx)
        angle_new = angle_current + omega * dt  # 逆时针为正？需确认坐标系
        # 注意：这里 centripetal_accel_mag 包含了 rudder 符号，rudder>0 向左转(逆时针?)
        # 检查坐标系：x右, y前(上)。
        # nx, ny 是速度方向。pnx=-ny, pny=nx 是左侧法向?
        # 如果 vx=0, vy=1 (向上), nx=0, ny=1. pnx=-1, pny=0 (向左).
        # 所以 pnx, pny 是左手法向。rudder>0 (左转) => a_n > 0 => 向左加速 => 角度增加。
        # atan2(y, x): 逆时针为正。
        # 所以 angle_new = angle + omega * dt 是对的。
        
        # 3. 重构速度矢量
        vx_new = v_new_mag * torch.cos(angle_new)
        vy_new = v_new_mag * torch.sin(angle_new)
        
        # 4. 更新位置 (使用平均速度)
        x_new = x + (vx + vx_new) * 0.5 * dt
        y_new = y + (vy + vy_new) * 0.5 * dt
        
        speed = v_new_mag
        angle = angle_new
        
        return x_new, y_new, vx_new, vy_new, speed, angle, engine_time
    
    def predict_batch_combined(self, 
                                x: torch.Tensor, y: torch.Tensor,
                                vx: torch.Tensor, vy: torch.Tensor,
                                rudder: torch.Tensor, throttle: torch.Tensor,
                                is_missile: torch.Tensor = None,
                                engine_time: torch.Tensor = None) -> torch.Tensor:
        """
        批量预测轨迹（合并版，支持多条轨迹一次计算）
        
        Args:
            x, y, vx, vy: [batch_size] 初始状态
            rudder, throttle: [batch_size] 控制量
            is_missile: [batch_size] bool 是否为导弹
            engine_time: [batch_size] 导弹发动机时间
            
        Returns:
            [batch_size, num_steps, 4] 轨迹点 (x, y, speed, angle)
        """
        batch_size = x.shape[0]
        if is_missile is None:
            is_missile = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        if engine_time is None:
            engine_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        
        # 预分配轨迹缓冲区
        trajectory = torch.zeros(batch_size, self.num_steps, 4, device=self._device)
        
        # 当前状态
        cur_x, cur_y, cur_vx, cur_vy = x.clone(), y.clone(), vx.clone(), vy.clone()
        cur_engine = engine_time.clone()
        
        for step in range(self.num_steps):
            speed = torch.sqrt(cur_vx ** 2 + cur_vy ** 2 + self.epsilon)
            angle = torch.atan2(cur_vy, cur_vx)
            
            trajectory[:, step, 0] = cur_x
            trajectory[:, step, 1] = cur_y
            trajectory[:, step, 2] = speed
            trajectory[:, step, 3] = angle
            
            if step < self.num_steps - 1:
                cur_x, cur_y, cur_vx, cur_vy, _, _, cur_engine = self._simulate_step_batch(
                    cur_x, cur_y, cur_vx, cur_vy, rudder, throttle,
                    is_missile, cur_engine, self.prediction_dt
                )
        
        return trajectory
    
    def predict_aircraft_batch(self, x: torch.Tensor, y: torch.Tensor,
                               vx: torch.Tensor, vy: torch.Tensor,
                               rudder: torch.Tensor, throttle: torch.Tensor) -> torch.Tensor:
        """批量预测飞机轨迹"""
        return self.predict_batch_combined(x, y, vx, vy, rudder, throttle)
    
    def predict_escape_batch(self, x: torch.Tensor, y: torch.Tensor,
                            vx: torch.Tensor, vy: torch.Tensor,
                            throttle: torch.Tensor,
                            enemy_x: torch.Tensor, enemy_y: torch.Tensor,
                            turn_direction: float) -> torch.Tensor:
        """
        批量预测逃离轨迹（全力回转后直飞）
        
        Args:
            x, y, vx, vy: [num_envs] 初始状态
            throttle: [num_envs] 油门
            enemy_x, enemy_y: [num_envs] 敌人位置
            turn_direction: -1.0 (左转) 或 1.0 (右转)
            
        Returns:
            [num_envs, num_steps, 4] 轨迹点
        """
        num_envs = x.shape[0]
        is_missile = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        engine_time = torch.zeros(num_envs, dtype=torch.float32, device=self._device)
        
        # 计算逃离方向（与敌我连线相反）
        to_enemy_x = enemy_x - x
        to_enemy_y = enemy_y - y
        enemy_dist = torch.sqrt(to_enemy_x ** 2 + to_enemy_y ** 2 + self.epsilon)
        escape_dir_x = -to_enemy_x / enemy_dist
        escape_dir_y = -to_enemy_y / enemy_dist
        
        # 存储轨迹
        trajectory = torch.zeros(num_envs, self.num_steps, 4, device=self._device)
        
        # 当前状态
        cur_x, cur_y, cur_vx, cur_vy = x.clone(), y.clone(), vx.clone(), vy.clone()
        rudder = torch.full((num_envs,), turn_direction, device=self._device)
        has_turned = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        
        ESCAPE_THRESHOLD = 0.95
        
        for step in range(self.num_steps):
            speed = torch.sqrt(cur_vx ** 2 + cur_vy ** 2 + self.epsilon)
            angle = torch.atan2(cur_vy, cur_vx)
            
            trajectory[:, step, 0] = cur_x
            trajectory[:, step, 1] = cur_y
            trajectory[:, step, 2] = speed
            trajectory[:, step, 3] = angle
            
            if step < self.num_steps - 1:
                # 检查是否已转到逃离方向
                vel_dir_x = cur_vx / speed
                vel_dir_y = cur_vy / speed
                dot_product = vel_dir_x * escape_dir_x + vel_dir_y * escape_dir_y
                
                newly_turned = (~has_turned) & (dot_product >= ESCAPE_THRESHOLD)
                has_turned = has_turned | newly_turned
                rudder = torch.where(has_turned, torch.zeros_like(rudder), rudder)
                
                cur_x, cur_y, cur_vx, cur_vy, _, _, _ = self._simulate_step_batch(
                    cur_x, cur_y, cur_vx, cur_vy, rudder, throttle,
                    is_missile, engine_time, self.prediction_dt
                )
        
        return trajectory
    
    def predict_missile_batch(self, x: torch.Tensor, y: torch.Tensor,
                             vx: torch.Tensor, vy: torch.Tensor,
                             engine_time: torch.Tensor,
                             target_trajectory: torch.Tensor,
                             active_mask: torch.Tensor) -> torch.Tensor:
        """
        批量预测导弹轨迹（追踪目标）
        
        Args:
            x, y, vx, vy: [num_missiles] 导弹初始状态
            engine_time: [num_missiles] 剩余发动机时间
            target_trajectory: [num_missiles, num_steps, 4] 目标预测轨迹
            active_mask: [num_missiles] 是否激活
            
        Returns:
            [num_missiles, num_steps, 4] 导弹轨迹
        """
        num_missiles = x.shape[0]
        if num_missiles == 0:
            return torch.zeros(0, self.num_steps, 4, device=self._device)
        
        is_missile = torch.ones(num_missiles, dtype=torch.bool, device=self._device)
        throttle = torch.ones(num_missiles, device=self._device)
        
        trajectory = torch.zeros(num_missiles, self.num_steps, 4, device=self._device)
        
        cur_x, cur_y, cur_vx, cur_vy = x.clone(), y.clone(), vx.clone(), vy.clone()
        cur_engine_time = engine_time.clone()
        
        # 跟踪是否曾超音速（用于无威胁判定）
        initial_speed = torch.sqrt(vx ** 2 + vy ** 2)
        ever_supersonic = initial_speed >= self.no_threat_speed
        still_threat = active_mask.clone()
        
        for step in range(self.num_steps):
            speed = torch.sqrt(cur_vx ** 2 + cur_vy ** 2 + self.epsilon)
            angle = torch.atan2(cur_vy, cur_vx)
            
            trajectory[:, step, 0] = cur_x
            trajectory[:, step, 1] = cur_y
            trajectory[:, step, 2] = speed
            trajectory[:, step, 3] = angle
            
            if step < self.num_steps - 1:
                # 更新是否曾超音速
                ever_supersonic = ever_supersonic | (speed >= self.no_threat_speed)
                
                # 检查无威胁条件
                no_threat = ever_supersonic & (speed < self.no_threat_speed)
                still_threat = still_threat & (~no_threat)
                
                # 获取目标当前位置
                step_idx = min(step, target_trajectory.shape[1] - 1)
                target_x = target_trajectory[:, step_idx, 0]
                target_y = target_trajectory[:, step_idx, 1]
                
                # 计算制导舵量
                dx = target_x - cur_x
                dy = target_y - cur_y
                target_angle = torch.atan2(dy, dx)
                angle_diff = target_angle - angle
                angle_diff = torch.remainder(angle_diff + math.pi, 2 * math.pi) - math.pi
                rudder = torch.clamp(angle_diff / (math.pi / 4), -1.0, 1.0)
                
                # 无威胁的导弹停止更新
                rudder = torch.where(still_threat, rudder, torch.zeros_like(rudder))
                
                # 更新状态
                cur_x, cur_y, cur_vx, cur_vy, _, _, cur_engine_time = self._simulate_step_batch(
                    cur_x, cur_y, cur_vx, cur_vy, rudder, throttle,
                    is_missile, cur_engine_time, self.prediction_dt
                )
                
                # 无威胁的导弹保持最后位置
                cur_x = torch.where(still_threat, cur_x, trajectory[:, step, 0])
                cur_y = torch.where(still_threat, cur_y, trajectory[:, step, 1])
        
        return trajectory
    
    def get_trajectory_features(self, trajectory: torch.Tensor,
                               sample_times: List[float] = None) -> torch.Tensor:
        """
        从轨迹中提取稀疏特征（优化版：批量索引）
        
        Args:
            trajectory: [batch_size, num_steps, 4] 轨迹
            sample_times: 采样时间点（秒），默认 [5, 10, 15, 20]
            
        Returns:
            [batch_size, num_samples * 4] 特征向量
        """
        if sample_times is None:
            sample_times = [5.0, 10.0, 15.0, 20.0]
        
        # 计算采样索引
        sample_indices = torch.tensor(
            [min(int(t / self.prediction_dt), self.num_steps - 1) for t in sample_times],
            device=self._device, dtype=torch.long
        )
        
        # 一次性批量索引 [batch_size, num_samples, 4]
        sampled = trajectory[:, sample_indices, :]
        
        # 展平为 [batch_size, num_samples * 4]
        return sampled.reshape(trajectory.shape[0], -1)


# =============================================================================
# 综合管理器
# =============================================================================

class TensorSpacetimeComputer:
    """
    GPU版时空图综合管理器
    
    统一管理批量环境的时空图计算：
    - 威胁锥批量生成
    - 轨迹预测批量生成
    - 特征提取（供agent观测和奖励计算）
    
    与TensorEnv集成:
        - 接收TensorEnv.states字典作为输入
        - 输出标准化特征张量
        
    使用流程:
        1. __init__(): 初始化，加载查找表
        2. update(): 每step调用，更新时空数据
        3. get_features(): 获取agent观测特征
        4. get_reward_signals(): 获取奖励计算信号
    """
    
    def __init__(self, config: Dict, num_envs: int, max_entities: int = 20, device: str = 'cuda'):
        """
        Args:
            config: 配置字典
            num_envs: 并行环境数量
            max_entities: 每个环境最大实体数
            device: 计算设备
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        self.config = config
        self.num_envs = num_envs
        self.max_entities = max_entities
        
        # 实体索引常量
        self.P1_IDX = 0
        self.P2_IDX = 1
        self.FIRST_MISSILE_IDX = 2
        
        # 子模块
        self.missile_lookup = TensorMissileLookup(device=device)
        self.threat_cone = TensorThreatCone(self.missile_lookup, device=device)
        self.trajectory_predictor = TensorTrajectoryPredictor(config, device=device)
        
        # 预测轨迹缓存 (减少重复计算)
        self.cached_p1_trajectory: Optional[torch.Tensor] = None  # [num_envs, num_steps, 4]
        self.cached_p2_trajectory: Optional[torch.Tensor] = None
        self.cached_p1_escape_left: Optional[torch.Tensor] = None
        self.cached_p1_escape_right: Optional[torch.Tensor] = None
        self.cached_p2_escape_left: Optional[torch.Tensor] = None
        self.cached_p2_escape_right: Optional[torch.Tensor] = None
        
        # 更新控制
        self.update_interval = 0.5  # 每0.5秒更新预测
        self._last_update_step = -1000
        self._current_step = 0
        
        # 初始化
        self.missile_lookup.load()
    
    def update(self, states: Dict[str, torch.Tensor], step: int, force_update: bool = False):
        """
        更新时空图数据（优化版：合并批量计算）
        
        Args:
            states: TensorEnv的states字典
            step: 当前step数
            force_update: 是否强制更新
        """
        self._current_step = step
        
        # 缓存环境0的状态用于渲染
        if self.num_envs > 0:
            self.last_states_env0 = {
                k: v[0].clone() for k, v in states.items() 
                if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == self.num_envs
            }
            self.last_states_env0_time = step * (1.0/60.0)

        # 检查是否需要更新（约每30步更新一次，对应0.5秒@60fps）
        steps_since_update = step - self._last_update_step
        should_update = force_update or (steps_since_update >= 30)
        
        if not should_update:
            return
        
        self._last_update_step = step
        
        # 提取状态
        p1_x = states['x'][:, self.P1_IDX]
        p1_y = states['y'][:, self.P1_IDX]
        p1_vx = states['vx'][:, self.P1_IDX]
        p1_vy = states['vy'][:, self.P1_IDX]
        p1_rudder = states['rudder'][:, self.P1_IDX]
        p1_throttle = states['throttle'][:, self.P1_IDX]
        
        p2_x = states['x'][:, self.P2_IDX]
        p2_y = states['y'][:, self.P2_IDX]
        p2_vx = states['vx'][:, self.P2_IDX]
        p2_vy = states['vy'][:, self.P2_IDX]
        p2_rudder = states['rudder'][:, self.P2_IDX]
        p2_throttle = states['throttle'][:, self.P2_IDX]
        
        # ==== 合并批量计算: 将6条轨迹合并为1次计算 ====
        # 轨迹顺序: [P1当前, P1左转, P1右转, P2当前, P2左转, P2右转]
        num_envs = self.num_envs
        
        # P1逃离轨迹的舵量（全力转向）
        p1_escape_rudder_left = torch.full_like(p1_rudder, -1.0)
        p1_escape_rudder_right = torch.full_like(p1_rudder, 1.0)
        p2_escape_rudder_left = torch.full_like(p2_rudder, -1.0)
        p2_escape_rudder_right = torch.full_like(p2_rudder, 1.0)
        
        # 堆叠所有初始状态 [6*num_envs]
        all_x = torch.cat([p1_x, p1_x, p1_x, p2_x, p2_x, p2_x], dim=0)
        all_y = torch.cat([p1_y, p1_y, p1_y, p2_y, p2_y, p2_y], dim=0)
        all_vx = torch.cat([p1_vx, p1_vx, p1_vx, p2_vx, p2_vx, p2_vx], dim=0)
        all_vy = torch.cat([p1_vy, p1_vy, p1_vy, p2_vy, p2_vy, p2_vy], dim=0)
        all_rudder = torch.cat([p1_rudder, p1_escape_rudder_left, p1_escape_rudder_right,
                                p2_rudder, p2_escape_rudder_left, p2_escape_rudder_right], dim=0)
        all_throttle = torch.cat([p1_throttle, p1_throttle, p1_throttle,
                                  p2_throttle, p2_throttle, p2_throttle], dim=0)
        
        # 一次性计算6条轨迹
        all_trajectories = self.trajectory_predictor.predict_batch_combined(
            all_x, all_y, all_vx, all_vy, all_rudder, all_throttle
        )  # [6*num_envs, num_steps, 4]
        
        # 拆分结果
        self.cached_p1_trajectory = all_trajectories[0:num_envs]
        self.cached_p1_escape_left = all_trajectories[num_envs:2*num_envs]
        self.cached_p1_escape_right = all_trajectories[2*num_envs:3*num_envs]
        self.cached_p2_trajectory = all_trajectories[3*num_envs:4*num_envs]
        self.cached_p2_escape_left = all_trajectories[4*num_envs:5*num_envs]
        self.cached_p2_escape_right = all_trajectories[5*num_envs:6*num_envs]
    
    def initialize(self):
        """兼容性接口"""
        self.missile_lookup.load()

    def get_current_time(self) -> float:
        """获取当前游戏时间（基于step）"""
        return self._current_step * (1.0/60.0)

    def _tensor_to_points(self, trajectory_tensor: torch.Tensor, time_offset: float = 0.0) -> List[SpacetimePoint]:
        """Convert [num_steps, 4] tensor to List[SpacetimePoint]"""
        if trajectory_tensor is None:
            return []
        
        points = []
        dt = self.trajectory_predictor.prediction_dt
        cpu_tensor = trajectory_tensor.cpu()
        
        for i in range(cpu_tensor.shape[0]):
            t = i * dt
            # x, y, speed, angle
            x, y, speed, angle = cpu_tensor[i].tolist()
            angle_deg = math.degrees(angle)
            points.append(SpacetimePoint(x, y, -t - time_offset, speed, angle_deg))
        return points

    def get_player_spacetime_data(self, player_id: int) -> Dict:
        """渲染接口：获取玩家时空数据"""
        if player_id == 1:
            traj = self.cached_p1_trajectory
            left = self.cached_p1_escape_left
            right = self.cached_p1_escape_right
        else:
            traj = self.cached_p2_trajectory
            left = self.cached_p2_escape_left
            right = self.cached_p2_escape_right
        
        if traj is None:
            return {}
            
        # Use env 0
        update_time = self._last_update_step * (1.0/60.0)
        
        return {
            'current_prediction': self._tensor_to_points(traj[0]),
            'current_prediction_time': update_time,
            'left_turn_prediction': self._tensor_to_points(left[0]),
            'left_turn_prediction_time': update_time,
            'right_turn_prediction': self._tensor_to_points(right[0]),
            'right_turn_prediction_time': update_time,
        }

    def get_missile_spacetime_data(self, missile_id: int) -> Dict:
        """渲染接口：获取导弹时空数据"""
        # missile_id is slot_idx
        if not hasattr(self, 'last_states_env0'):
            return {}
            
        states = self.last_states_env0
        
        # Check if missile is active and valid index
        if missile_id >= len(states['is_active']) or not states['is_active'][missile_id] or not states['is_missile'][missile_id]:
            return {}
            
        # Prepare inputs for single prediction
        x = states['x'][missile_id].view(1)
        y = states['y'][missile_id].view(1)
        vx = states['vx'][missile_id].view(1)
        vy = states['vy'][missile_id].view(1)
        engine_time = states['engine_time'][missile_id].view(1)
        
        # Need target trajectory
        target_idx = states['target_idx'][missile_id].long().item()
        
        target_traj = None
        if target_idx == self.P1_IDX:
            target_traj = self.cached_p1_trajectory[0:1] if self.cached_p1_trajectory is not None else None
        elif target_idx == self.P2_IDX:
            target_traj = self.cached_p2_trajectory[0:1] if self.cached_p2_trajectory is not None else None
            
        if target_traj is None:
            target_traj = torch.zeros(1, 21, 4, device=self._device)
            if target_idx < self.max_entities:
                tx = states['x'][target_idx]
                ty = states['y'][target_idx]
                target_traj[:, :, 0] = tx
                target_traj[:, :, 1] = ty
        
        active_mask = torch.tensor([True], device=self._device)
        
        # Predict
        traj = self.trajectory_predictor.predict_missile_batch(
            x, y, vx, vy, engine_time, target_traj, active_mask
        )
        
        update_time = self.last_states_env0_time
        
        return {
            'trajectory_prediction': self._tensor_to_points(traj[0]),
            'trajectory_prediction_time': update_time
        }

    def get_threat_cone(self, enemy, for_player: int) -> Dict:
        """渲染接口：获取威胁锥几何"""
        if not hasattr(self, 'last_states_env0'):
            return {'meridians': [], 'ellipses': []}
            
        states = self.last_states_env0
        enemy_idx = self.P2_IDX if for_player == 1 else self.P1_IDX
        
        if not states['alive'][enemy_idx]:
            return {'meridians': [], 'ellipses': []}
            
        speed = states['speed'][enemy_idx].view(1)
        angle = states['angle'][enemy_idx].view(1) * (math.pi / 180.0) 
        
        cone_data = self.threat_cone.compute_batch(speed, angle)
        
        radii = cone_data['cone_radii'][0] 
        angles_rad = cone_data['angles_rad']
        time_slices = cone_data['time_slices']
        
        cx = states['x'][enemy_idx].item()
        cy = states['y'][enemy_idx].item()
        ca = angle.item()
        cos_a = math.cos(ca)
        sin_a = math.sin(ca)
        
        meridians = []
        indices_to_draw = [0, 4, 8]
        
        cpu_radii = radii.cpu().numpy()
        cpu_angles = angles_rad.cpu().numpy()
        cpu_times = time_slices.cpu().numpy()
        
        for idx in indices_to_draw:
            if idx >= len(cpu_angles): continue
            
            theta = cpu_angles[idx]
            points = []
            points.append(SpacetimePoint(cx, cy, 0.0, speed.item(), math.degrees(ca)))
            
            for t_idx, t_val in enumerate(cpu_times):
                r = cpu_radii[t_idx, idx]
                lx = r * math.cos(theta)
                ly = r * math.sin(theta)
                wx = cx + lx * cos_a - ly * sin_a
                wy = cy + lx * sin_a + ly * cos_a
                points.append(SpacetimePoint(wx, wy, -t_val, 0, 0))
            
            meridians.append({'points': points})
            
        ellipses = []
        for t_idx, t_val in enumerate(cpu_times):
            points = []
            for a_idx, theta in enumerate(cpu_angles):
                r = cpu_radii[t_idx, a_idx]
                lx = r * math.cos(theta)
                ly = r * math.sin(theta)
                wx = cx + lx * cos_a - ly * sin_a
                wy = cy + lx * sin_a + ly * cos_a
                points.append(SpacetimePoint(wx, wy, -t_val, 0, 0))
            ellipses.append({'points': points})
            
        return {'meridians': meridians, 'ellipses': ellipses}

    def get_features(self, states: Dict[str, torch.Tensor], 
                    player_id: int) -> torch.Tensor:
        """
        获取指定玩家的时空图特征
        
        Args:
            states: TensorEnv的states字典
            player_id: 玩家ID (1 或 2)
            
        Returns:
            [num_envs, feature_dim] 特征向量
            
        特征组成 (共约60维):
            - 己方当前轨迹采样 [16] (4点 * 4维)
            - 己方左转轨迹采样 [16]
            - 己方右转轨迹采样 [16]
            - 敌方威胁锥特征 [9]
        """
        if player_id == 1:
            own_traj = self.cached_p1_trajectory
            own_escape_left = self.cached_p1_escape_left
            own_escape_right = self.cached_p1_escape_right
            enemy_idx = self.P2_IDX
            own_idx = self.P1_IDX
        else:
            own_traj = self.cached_p2_trajectory
            own_escape_left = self.cached_p2_escape_left
            own_escape_right = self.cached_p2_escape_right
            enemy_idx = self.P1_IDX
            own_idx = self.P2_IDX
        
        # 如果缓存为空，返回零向量
        if own_traj is None:
            return torch.zeros(self.num_envs, 57, device=self._device)
        
        # 提取轨迹特征
        own_features = self.trajectory_predictor.get_trajectory_features(own_traj)  # [num_envs, 16]
        left_features = self.trajectory_predictor.get_trajectory_features(own_escape_left)  # [num_envs, 16]
        right_features = self.trajectory_predictor.get_trajectory_features(own_escape_right)  # [num_envs, 16]
        
        # 提取威胁锥特征
        enemy_speed = states['speed'][:, enemy_idx]
        enemy_x = states['x'][:, enemy_idx]
        enemy_y = states['y'][:, enemy_idx]
        enemy_angle = states['angle'][:, enemy_idx] * (math.pi / 180.0)  # 转弧度
        
        own_x = states['x'][:, own_idx]
        own_y = states['y'][:, own_idx]
        
        threat_features = self.threat_cone.get_threat_features(
            enemy_speed, enemy_x, enemy_y, enemy_angle, own_x, own_y
        )  # [num_envs, 9]
        
        # 归一化轨迹特征
        # x, y: / 50000, speed: / 500, angle: / pi
        own_features = self._normalize_trajectory_features(own_features)
        left_features = self._normalize_trajectory_features(left_features)
        right_features = self._normalize_trajectory_features(right_features)
        
        # 组合所有特征
        features = torch.cat([
            own_features,
            left_features,
            right_features,
            threat_features,
        ], dim=-1)
        
        return features
    
    def _normalize_trajectory_features(self, features: torch.Tensor) -> torch.Tensor:
        """归一化轨迹特征（优化版：向量化操作）"""
        # features: [batch_size, num_samples * 4]
        # 每4个值: x, y, speed, angle
        batch_size = features.shape[0]
        num_samples = features.shape[1] // 4
        
        # 预计算归一化因子
        norm_factors = torch.tensor(
            [50000.0, 50000.0, 500.0, math.pi] * num_samples,
            device=self._device, dtype=torch.float32
        )
        
        # 一次性广播除法
        return features / norm_factors
    
    def get_threat_metrics(self, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取威胁度量指标（用于奖励计算）
        
        Returns:
            字典包含:
                - p1_in_threat: [num_envs] P1是否在P2威胁锥内
                - p2_in_threat: [num_envs] P2是否在P1威胁锥内
                - p1_threat_ratio: [num_envs] P1距离/威胁半径
                - p2_threat_ratio: [num_envs] P2距离/威胁半径
        """
        # P1视角：敌人是P2
        p1_x = states['x'][:, self.P1_IDX]
        p1_y = states['y'][:, self.P1_IDX]
        p2_x = states['x'][:, self.P2_IDX]
        p2_y = states['y'][:, self.P2_IDX]
        p2_speed = states['speed'][:, self.P2_IDX]
        p2_angle = states['angle'][:, self.P2_IDX] * (math.pi / 180.0)
        
        # P1受到P2威胁
        p1_threat_features = self.threat_cone.get_threat_features(
            p2_speed, p2_x, p2_y, p2_angle, p1_x, p1_y
        )
        p1_threat_ratio = p1_threat_features[:, 4:8].min(dim=-1)[0]  # 取最小威胁比
        p1_in_threat = p1_threat_ratio < 1.0
        
        # P2视角
        p1_speed = states['speed'][:, self.P1_IDX]
        p1_angle = states['angle'][:, self.P1_IDX] * (math.pi / 180.0)
        
        p2_threat_features = self.threat_cone.get_threat_features(
            p1_speed, p1_x, p1_y, p1_angle, p2_x, p2_y
        )
        p2_threat_ratio = p2_threat_features[:, 4:8].min(dim=-1)[0]
        p2_in_threat = p2_threat_ratio < 1.0
        
        return {
            'p1_in_threat': p1_in_threat,
            'p2_in_threat': p2_in_threat,
            'p1_threat_ratio': p1_threat_ratio,
            'p2_threat_ratio': p2_threat_ratio,
        }
    
    def reset(self, env_mask: torch.Tensor = None):
        """
        重置指定环境的时空数据
        
        Args:
            env_mask: [num_envs] bool 需要重置的环境
        """
        if env_mask is None:
            # 清空所有缓存
            self.cached_p1_trajectory = None
            self.cached_p2_trajectory = None
            self.cached_p1_escape_left = None
            self.cached_p1_escape_right = None
            self.cached_p2_escape_left = None
            self.cached_p2_escape_right = None
            self._last_update_step = -1000
        else:
            # 部分重置：标记需要重新计算
            # 这里简单实现为强制下次更新
            self._last_update_step = -1000
