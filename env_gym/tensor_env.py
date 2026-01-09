# -*- coding: utf-8 -*-
"""
改进的 Tensor 环境实现
- 支持多环境并行 [num_envs, max_entities]
- 统一的实体槽位系统
- 完整的RL接口（奖励函数、观察空间）
"""

import torch
import math
from typing import Dict, Tuple, Any, Optional
from env_gym.base_env import BaseEnv


class TensorAerodynamic:
    """气动力学计算模块（支持多环境）- 全float32精度"""
    
    def __init__(self, config, device='cuda'):
        self._device = torch.device(device) if isinstance(device, str) else device
        _dtype = torch.float32  # 统一单精度
        
        # 飞机参数 - 显式float32
        self.FIGHTER_TERMINAL_VELOCITY = torch.tensor(float(config.get('FIGHTER_TERMINAL_VELOCITY', 400)), device=self._device, dtype=_dtype)
        self.FIGHTER_MIN_TURN_RADIUS = torch.tensor(float(config.get('FIGHTER_MIN_TURN_RADIUS', 1000)), device=self._device, dtype=_dtype)
        self.FIGHTER_CL_MAX = 1.0 / self.FIGHTER_MIN_TURN_RADIUS
        self.FIGHTER_MAX_THRUST = torch.tensor(float(config.get('FIGHTER_MAX_THRUST', 1.5 * 9.8)), device=self._device, dtype=_dtype)
        self.FIGHTER_LIFT_DRAG_RATIO = torch.tensor(float(config.get('FIGHTER_LIFT_DRAG_RATIO', 5)), device=self._device, dtype=_dtype)

        # 导弹参数 - 显式float32
        self.MISSILE_TERMINAL_VELOCITY = torch.tensor(float(config.get('MISSILE_TERMINAL_VELOCITY', 400)), device=self._device, dtype=_dtype)
        self.MISSILE_MIN_TURN_RADIUS = torch.tensor(float(config.get('MISSILE_MIN_TURN_RADIUS', 1000)), device=self._device, dtype=_dtype)
        self.MISSILE_CL_MAX = 1.0 / self.MISSILE_MIN_TURN_RADIUS
        self.MISSILE_THRUST = torch.tensor(float(config.get('MISSILE_THRUST', 15 * 9.8)), device=self._device, dtype=_dtype)
        self.MISSILE_ENGINE_DURATION = torch.tensor(float(config.get('MISSILE_ENGINE_DURATION', 10.0)), device=self._device, dtype=_dtype)
        self.MISSILE_LIFT_DRAG_RATIO = torch.tensor(float(config.get('MISSILE_LIFT_DRAG_RATIO', 3)), device=self._device, dtype=_dtype)

        self.G = torch.tensor(9.8, device=self._device, dtype=_dtype)
        self.epsilon = 1e-7

    def calculate_physics(self, states, dt):
        """计算物理更新（支持 [num_envs, max_entities] 形状）"""
        vx = states['vx']
        vy = states['vy']
        rudder = states['rudder']
        throttle = states['throttle']
        is_missile_mask = states['is_missile']
        is_active_mask = states['is_active']
        engine_time = states['engine_time']

        # 根据实体类型选择参数
        terminal_velocity = torch.where(is_missile_mask, self.MISSILE_TERMINAL_VELOCITY, self.FIGHTER_TERMINAL_VELOCITY)
        cl_max = torch.where(is_missile_mask, self.MISSILE_CL_MAX, self.FIGHTER_CL_MAX)
        max_thrust = torch.where(is_missile_mask, self.MISSILE_THRUST, self.FIGHTER_MAX_THRUST)
        lift_drag_ratio = torch.where(is_missile_mask, self.MISSILE_LIFT_DRAG_RATIO, self.FIGHTER_LIFT_DRAG_RATIO)

        # 更新导弹引擎时间
        engine_time = torch.where(is_missile_mask & is_active_mask, 
                                 torch.clamp(engine_time - dt, min=0.0), engine_time)
        states['engine_time'] = engine_time

        # 计算推力加速度
        thrust_active = torch.where(is_missile_mask, (engine_time > self.epsilon).float(), throttle)
        thrust_accel = thrust_active * max_thrust

        # 核心矢量物理计算
        vSquare = vx*vx + vy*vy
        inv_v = torch.rsqrt(vSquare + self.epsilon)

        nx = vx * inv_v
        ny = vy * inv_v
        pnx = -ny
        pny = nx

        # 计算空气动力系数
        Cd0 = self.G / (terminal_velocity**2 + self.epsilon)
        k = 1.0 / (4.0 * Cd0 * (lift_drag_ratio**2) + self.epsilon)
        Cl_intermediate_sq = (torch.abs(rudder) * cl_max)**2
        Cd = Cd0 + k * Cl_intermediate_sq

        # 计算阻力加速度大小
        drag_accel = Cd * vSquare

        # 计算平行和垂直加速度大小
        parallel_accel_mag = thrust_accel - drag_accel
        centripetal_accel_mag = vSquare * cl_max * rudder

        # 合成总加速度矢量
        ax = nx * parallel_accel_mag + pnx * centripetal_accel_mag
        ay = ny * parallel_accel_mag + pny * centripetal_accel_mag

        # 更新状态（只更新 active 的实体）
        active_float_mask = is_active_mask.float()

        vx_new = vx + ax * dt * active_float_mask
        vy_new = vy + ay * dt * active_float_mask

        x_new = states['x'] + vx_new * dt * active_float_mask
        y_new = states['y'] + vy_new * dt * active_float_mask

        # 更新核心状态
        states['vx'] = vx_new
        states['vy'] = vy_new
        states['x'] = x_new
        states['y'] = y_new

        # 计算辅助标量
        speed_new = torch.sqrt(vx_new**2 + vy_new**2 + self.epsilon)
        angle_new_rad = torch.atan2(vy_new, vx_new)
        angle_new_deg = torch.rad2deg(angle_new_rad)

        # 过载和角速度
        actual_n_load = torch.abs(vSquare * cl_max * rudder) / (self.G + self.epsilon)
        turn_rate_rad = speed_new * cl_max * rudder

        states['speed'] = speed_new * active_float_mask
        states['angle'] = angle_new_deg % 360
        states['n_load'] = actual_n_load * active_float_mask
        states['turn_rate'] = torch.rad2deg(turn_rate_rad) * active_float_mask

        return states


class TensorMissileGuidance:
    """导弹制导模块（支持多环境）- 全float32精度"""
    
    def __init__(self, config, device='cuda'):
        self._device = torch.device(device) if isinstance(device, str) else device
        _dtype = torch.float32  # 统一单精度
        
        # 从MISSILE_GUIDANCE_GAIN读取配置，并将角度制增益转换为弧度制
        # CPU端使用角度制: rudder = gain * los_rate_deg / 180
        # GPU端使用弧度制: rudder = gain_rad * los_rate_rad
        # 换算: gain_rad = gain_deg / π
        guidance_gain_deg = config.get('MISSILE_GUIDANCE_GAIN', 200)
        self.guidance_gain = torch.tensor(float(guidance_gain_deg) / math.pi, device=self._device, dtype=_dtype)
        self.epsilon = 1e-7

    def calculate_guidance(self, states, dt):
        """计算所有激活导弹的制导指令（支持多环境）"""
        # states 形状: [num_envs, max_entities]
        missile_mask = states['is_missile'] & states['is_active']
        if not torch.any(missile_mask):
            return states

        target_idx = states['target_idx']  # [num_envs, max_entities]
        
        # 获取所有导弹的索引（环境索引和实体索引）
        env_indices, entity_indices = torch.nonzero(missile_mask, as_tuple=True)
        
        if len(env_indices) == 0:
            return states

        # 获取导弹位置
        missile_x = states['x'][env_indices, entity_indices]
        missile_y = states['y'][env_indices, entity_indices]
        
        # 获取目标索引
        valid_target_indices = target_idx[env_indices, entity_indices]
        
        # 检查目标有效性
        target_valid = (valid_target_indices >= 0) & (valid_target_indices < states['x'].shape[1])
        
        # 检查目标是否激活
        target_env_indices = env_indices[target_valid]
        target_entity_indices = valid_target_indices[target_valid]
        target_active = states['is_active'][target_env_indices, target_entity_indices]
        
        # 创建最终制导掩码
        final_guidance_mask = torch.zeros_like(target_valid)
        final_guidance_mask[target_valid] = target_active
        
        if not torch.any(final_guidance_mask):
            # 将无效导弹舵量置零
            invalid_mask = missile_mask & ~final_guidance_mask.unsqueeze(0).expand_as(missile_mask)
            states['rudder'][invalid_mask] = 0.0
            return states

        # 获取需要制导的导弹和目标
        guidance_env_indices = env_indices[final_guidance_mask]
        guidance_entity_indices = entity_indices[final_guidance_mask]
        guidance_target_indices = valid_target_indices[final_guidance_mask]
        
        guidance_missile_x = states['x'][guidance_env_indices, guidance_entity_indices]
        guidance_missile_y = states['y'][guidance_env_indices, guidance_entity_indices]
        guidance_target_x = states['x'][guidance_env_indices, guidance_target_indices]
        guidance_target_y = states['y'][guidance_env_indices, guidance_target_indices]
        guidance_prev_los_angle = states['prev_los_angle'][guidance_env_indices, guidance_entity_indices]

        # 计算当前弹目连线角度
        dx = guidance_target_x - guidance_missile_x
        dy = guidance_target_y - guidance_missile_y
        current_los_angle_rad = torch.atan2(dy, dx)

        # 计算视线角速度
        angle_diff = current_los_angle_rad - guidance_prev_los_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        los_rate_rad = angle_diff / (dt + self.epsilon)

        # 计算制导指令
        rudder_command = self.guidance_gain * los_rate_rad
        rudder_command = torch.clamp(rudder_command, -1.0, 1.0)

        # 更新状态
        states['prev_los_angle'][guidance_env_indices, guidance_entity_indices] = current_los_angle_rad
        states['rudder'][guidance_env_indices, guidance_entity_indices] = rudder_command

        return states


class TensorEnv(BaseEnv):
    """Tensor 环境（支持多环境并行）"""
    
    def __init__(self, config, num_envs=1, max_entities_per_env=20, device='cuda'):
        super().__init__()
        self._num_envs = num_envs
        self.max_entities = max_entities_per_env
        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device
        self.battlefield_size = torch.tensor(config.get('battlefield_size', 50000), device=device)
        self.hit_radius_sq = torch.tensor(config.get('hit_radius', 100)**2, device=device)
        self.self_destruct_speed_sq = torch.tensor(config.get('self_destruct_speed', 200)**2, device=device)
        self.initial_missiles = config.get('initial_missiles', 6)
        self.missile_launch_offset = config.get('missile_launch_offset', 20.0)
        
        # 奖励权重配置
        self.reward_config = {
            'hit_enemy': config.get('reward_hit_enemy', 100.0),
            'get_hit': config.get('reward_get_hit', -200.0),
            'survive': config.get('reward_survive', 0.1),
            'missile_used': config.get('reward_missile_used', -5.0),
            'win': config.get('reward_win', 500.0),
            'lose': config.get('reward_lose', -500.0),
            'draw': config.get('reward_draw', -50.0),
        }
        
        # 预分配奖励常量 tensor（避免重复创建）
        _dtype = torch.float32
        self._reward_win = torch.tensor(self.reward_config['win'], device=device, dtype=_dtype)
        self._reward_lose = torch.tensor(self.reward_config['lose'], device=device, dtype=_dtype)
        self._reward_draw = torch.tensor(self.reward_config['draw'], device=device, dtype=_dtype)
        self._reward_survive = torch.tensor(self.reward_config['survive'], device=device, dtype=_dtype)
        self._reward_zero = torch.tensor(0.0, device=device, dtype=_dtype)
        self._winner_p1 = torch.tensor(1, device=device, dtype=torch.long)
        self._winner_p2 = torch.tensor(2, device=device, dtype=torch.long)

        # 实体索引常量（必须在初始化状态之前设置）
        self.P1_IDX = 0
        self.P2_IDX = 1
        self.FIRST_MISSILE_IDX = 2
        
        # 初始化物理和制导模块
        self.physics = TensorAerodynamic(config, device)
        self.guidance = TensorMissileGuidance(config, device)

        # 初始化状态字典
        self.states = self._initialize_states()
        
        # 游戏状态
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.winner = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0: draw, 1: p1, 2: p2
        
        # ==== Kernel优化: 使用torch.compile减少kernel启动开销 ====
        # 仅在PyTorch 2.0+ 与 CUDA Capability >= 7.0 的设备上启用
        self._use_compile = config.get('use_torch_compile', False)
        if self._use_compile and torch.cuda.is_available():
            # 检查 CUDA Capability，Triton 需要 >= 7.0
            cuda_cap = torch.cuda.get_device_capability(self._device)
            if cuda_cap[0] < 7:
                print(f"[TensorEnv] torch.compile disabled: CUDA Capability {cuda_cap[0]}.{cuda_cap[1]} < 7.0 (Triton requirement)")
                self._use_compile = False
            else:
                try:
                    # 编译物理计算核心函数
                    self.physics.calculate_physics = torch.compile(
                        self.physics.calculate_physics,
                        mode='reduce-overhead',
                        fullgraph=False
                    )
                    print(f"[TensorEnv] torch.compile enabled for physics calculations")
                except Exception as e:
                    print(f"[TensorEnv] torch.compile failed: {e}")
                    self._use_compile = False
        elif self._use_compile:
            print(f"[TensorEnv] torch.compile disabled: CUDA not available")
            self._use_compile = False

    def _initialize_states(self):
        """初始化状态字典 [num_envs, max_entities]"""
        states = {
            'x': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'y': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'vx': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'vy': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'rudder': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'throttle': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'fire_command': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.bool),
            'target_idx': torch.full((self.num_envs, self.max_entities), -1, device=self._device, dtype=torch.long),
            'source_idx': torch.full((self.num_envs, self.max_entities), -1, device=self._device, dtype=torch.long),
            'engine_time': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'prev_los_angle': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'missile_count': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.long),
            'is_missile': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.bool),
            'is_active': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.bool),
            'is_player1': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.bool),
            'alive': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.bool),
            'speed': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'angle': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'n_load': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
            'turn_rate': torch.zeros(self.num_envs, self.max_entities, device=self._device, dtype=torch.float32),
        }
        self._reset_env(states)
        return states

    def _reset_env(self, states, env_mask=None):
        """重置环境状态（完全向量化，无Python循环）"""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self._device)
        
        # 使用布尔掩码直接索引，避免nonzero
        # 重置所有状态为非激活
        states['is_active'][env_mask] = False
        states['alive'][env_mask] = False
        
        # 预计算常量
        p1_speed = 300.0
        p2_speed = 300.0
        p1_angle_deg = 0.0
        p2_angle_deg = 180.0
        p1_rad = math.radians(p1_angle_deg)
        p2_rad = math.radians(p2_angle_deg)
        p1_vx = math.cos(p1_rad) * p1_speed
        p1_vy = math.sin(p1_rad) * p1_speed
        p2_vx = math.cos(p2_rad) * p2_speed
        p2_vy = math.sin(p2_rad) * p2_speed
        
        # P1 (红方) - 批量赋值
        states['x'][env_mask, self.P1_IDX] = 10000.0
        states['y'][env_mask, self.P1_IDX] = float(self.battlefield_size) / 2.0
        states['vx'][env_mask, self.P1_IDX] = p1_vx
        states['vy'][env_mask, self.P1_IDX] = p1_vy
        states['throttle'][env_mask, self.P1_IDX] = 1.0
        states['rudder'][env_mask, self.P1_IDX] = 0.0
        states['is_missile'][env_mask, self.P1_IDX] = False
        states['is_active'][env_mask, self.P1_IDX] = True
        states['is_player1'][env_mask, self.P1_IDX] = True
        states['alive'][env_mask, self.P1_IDX] = True
        states['missile_count'][env_mask, self.P1_IDX] = self.initial_missiles
        states['angle'][env_mask, self.P1_IDX] = p1_angle_deg
        states['speed'][env_mask, self.P1_IDX] = p1_speed
        
        # P2 (蓝方) - 批量赋值
        states['x'][env_mask, self.P2_IDX] = float(self.battlefield_size) - 10000.0
        states['y'][env_mask, self.P2_IDX] = float(self.battlefield_size) / 2.0
        states['vx'][env_mask, self.P2_IDX] = p2_vx
        states['vy'][env_mask, self.P2_IDX] = p2_vy
        states['throttle'][env_mask, self.P2_IDX] = 1.0
        states['rudder'][env_mask, self.P2_IDX] = 0.0
        states['is_missile'][env_mask, self.P2_IDX] = False
        states['is_active'][env_mask, self.P2_IDX] = True
        states['is_player1'][env_mask, self.P2_IDX] = False
        states['alive'][env_mask, self.P2_IDX] = True
        states['missile_count'][env_mask, self.P2_IDX] = self.initial_missiles
        states['angle'][env_mask, self.P2_IDX] = p2_angle_deg
        states['speed'][env_mask, self.P2_IDX] = p2_speed
        
        # 重置其他槽位（导弹槽位）- 批量操作
        missile_slots = slice(self.FIRST_MISSILE_IDX, self.max_entities)
        # 展开env_mask为[num_envs, 1]以广播到所有导弹槽位
        expanded_mask = env_mask.unsqueeze(1)  # [num_envs, 1]
        
        # 数值类型重置为0
        for key in ['x', 'y', 'vx', 'vy', 'throttle', 'rudder', 'speed', 'angle', 
                    'n_load', 'turn_rate', 'engine_time', 'prev_los_angle']:
            if key in states:
                states[key][env_mask, self.FIRST_MISSILE_IDX:] = 0.0
        
        # 布尔类型重置为False
        for key in ['is_missile', 'is_active', 'is_player1', 'alive', 'fire_command']:
            if key in states:
                states[key][env_mask, self.FIRST_MISSILE_IDX:] = False
        
        # 整数类型重置
        for key in ['target_idx', 'source_idx']:
            if key in states:
                states[key][env_mask, self.FIRST_MISSILE_IDX:] = -1
        
        if 'missile_count' in states:
            states['missile_count'][env_mask, self.FIRST_MISSILE_IDX:] = 0

    def reset(self, env_mask=None):
        """重置环境"""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self._device)
        
        self._reset_env(self.states, env_mask)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.winner = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        
        return self.get_observations()

    def step(self, actions, dt=1/60.0):
        """执行环境步骤"""
        # 辅助函数：将输入转换为 tensor
        def to_tensor(val, default, dtype=torch.float32):
            if val is None:
                return default
            if isinstance(val, torch.Tensor):
                return val.to(self._device)
            # 标量或 list 转换为 tensor
            if isinstance(val, (int, float, bool)):
                return torch.full((self.num_envs,), val, dtype=dtype, device=self._device)
            return torch.tensor(val, dtype=dtype, device=self._device)
        
        # 应用动作（支持批处理和单环境模式）
        # actions 格式: {'p1_rudder': [num_envs] 或 float, 'p1_throttle': [num_envs] 或 float, ...}
        p1_rudder = to_tensor(actions.get('p1_rudder'), torch.zeros(self.num_envs, device=self._device))
        p1_throttle = to_tensor(actions.get('p1_throttle'), torch.ones(self.num_envs, device=self._device))
        p1_fire = to_tensor(actions.get('p1_fire'), torch.zeros(self.num_envs, device=self._device), dtype=torch.bool)
        
        p2_rudder = to_tensor(actions.get('p2_rudder'), torch.zeros(self.num_envs, device=self._device))
        p2_throttle = to_tensor(actions.get('p2_throttle'), torch.ones(self.num_envs, device=self._device))
        p2_fire = to_tensor(actions.get('p2_fire'), torch.zeros(self.num_envs, device=self._device), dtype=torch.bool)
        
        # 更新飞机控制
        self.states['rudder'][:, self.P1_IDX] = torch.clamp(p1_rudder, -1.0, 1.0)
        self.states['throttle'][:, self.P1_IDX] = torch.clamp(p1_throttle, 0.0, 1.0)
        self.states['fire_command'][:, self.P1_IDX] = p1_fire
        
        self.states['rudder'][:, self.P2_IDX] = torch.clamp(p2_rudder, -1.0, 1.0)
        self.states['throttle'][:, self.P2_IDX] = torch.clamp(p2_throttle, 0.0, 1.0)
        self.states['fire_command'][:, self.P2_IDX] = p2_fire

        # 处理开火
        self._handle_firing()

        # 导弹制导
        self.states = self.guidance.calculate_guidance(self.states, dt)

        # 物理更新
        self.states = self.physics.calculate_physics(self.states, dt)

        # 事件检测
        self._check_events()

        # 检查游戏结束
        done_mask, winner_mask = self._check_game_over()
        
        # 更新完成状态（向量化，无条件分支）
        new_done_mask = done_mask & ~self.done
        self.done = done_mask
        # 使用torch.where而不是条件索引
        self.winner = torch.where(new_done_mask, winner_mask, self.winner)

        # 计算奖励
        rewards = self.compute_rewards(new_done_mask, winner_mask)

        # 获取观察
        observations = self.get_observations()

        # 构造info
        infos = {
            'winner': self.winner.clone(),
            'p1_alive': self.states['alive'][:, self.P1_IDX].clone(),
            'p2_alive': self.states['alive'][:, self.P2_IDX].clone(),
        }

        return observations, rewards, self.done.clone(), infos

    def _handle_firing(self):
        """处理开火指令（完全向量化，无Python循环）
        
        策略：P1导弹使用偶数槽位（从2开始），P2导弹使用奇数槽位（从3开始）
        这样每个玩家最多拥有(max_entities-2)/2枚导弹
        """
        # 检查P1和P2是否需要开火
        p1_fire = (self.states['fire_command'][:, self.P1_IDX] & 
                   self.states['is_active'][:, self.P1_IDX] &
                   (self.states['missile_count'][:, self.P1_IDX] > 0))
        p2_fire = (self.states['fire_command'][:, self.P2_IDX] & 
                   self.states['is_active'][:, self.P2_IDX] &
                   (self.states['missile_count'][:, self.P2_IDX] > 0))
        
        # 处理P1开火
        if p1_fire.any():
            self._fire_missile_batch(p1_fire, is_player1=True)
        
        # 处理P2开火
        if p2_fire.any():
            self._fire_missile_batch(p2_fire, is_player1=False)
    
    def _fire_missile_batch(self, fire_mask: torch.Tensor, is_player1: bool):
        """批量发射导弹（向量化）
        
        Args:
            fire_mask: 需要开火的环境掩码 [num_envs]
            is_player1: 是否为玩家1
        """
        source_idx = self.P1_IDX if is_player1 else self.P2_IDX
        target_idx = self.P2_IDX if is_player1 else self.P1_IDX
        
        # 计算每个玩家的导弹槽位范围
        # P1使用偶数槽位: 2,4,6,...  P2使用奇数槽位: 3,5,7,...
        slot_start = self.FIRST_MISSILE_IDX if is_player1 else (self.FIRST_MISSILE_IDX + 1)
        slot_step = 2
        max_missiles_per_player = (self.max_entities - self.FIRST_MISSILE_IDX) // 2
        
        # 统计当前已激活的导弹数量（按玩家）
        player_missile_slots = torch.arange(slot_start, self.max_entities, slot_step, device=self._device)
        active_count = self.states['is_active'][:, player_missile_slots].sum(dim=1)  # [num_envs]
        
        # 查找第一个可用槽位
        slots_inactive = ~self.states['is_active'][:, player_missile_slots]  # [num_envs, num_slots]
        # 用argmax找第一个True（当全为False时返回0）
        first_inactive_local = slots_inactive.float().argmax(dim=1)  # [num_envs]
        has_slot = slots_inactive.any(dim=1)  # [num_envs]
        
        # 只有同时需要开火且有槽位的环境才能发射
        actual_fire = fire_mask & has_slot
        
        # 计算实际槽位索引
        actual_slot_idx = player_missile_slots[first_inactive_local]  # [num_envs]
        
        # 消耗弹药并重置开火指令（对所有请求开火的）
        self.states['missile_count'][:, source_idx] = torch.where(
            fire_mask,
            self.states['missile_count'][:, source_idx] - 1,
            self.states['missile_count'][:, source_idx]
        )
        self.states['fire_command'][:, source_idx] = torch.where(
            fire_mask,
            torch.zeros_like(self.states['fire_command'][:, source_idx]),
            self.states['fire_command'][:, source_idx]
        )
        
        # 批量计算导弹初始状态（对所有环境）
        source_x = self.states['x'][:, source_idx]
        source_y = self.states['y'][:, source_idx]
        source_vx = self.states['vx'][:, source_idx]
        source_vy = self.states['vy'][:, source_idx]
        source_angle_rad = torch.atan2(source_vy, source_vx)
        
        # 计算发射位置
        offset_x = torch.cos(source_angle_rad) * self.missile_launch_offset
        offset_y = torch.sin(source_angle_rad) * self.missile_launch_offset
        missile_x = source_x + offset_x
        missile_y = source_y + offset_y
        
        # 计算初始制导角度
        target_x = self.states['x'][:, target_idx]
        target_y = self.states['y'][:, target_idx]
        init_dx = target_x - missile_x
        init_dy = target_y - missile_y
        init_los_angle = torch.atan2(init_dy, init_dx)
        
        # 使用scatter或直接索引写入（只写入actual_fire为True的环境）
        # 由于每个环境只写一个槽位，使用循环是不可避免的，但可以用高级索引
        # 构建索引: [env_idx, slot_idx] 对
        env_indices = torch.arange(self.num_envs, device=self._device)[actual_fire]
        slot_indices = actual_slot_idx[actual_fire]
        
        if len(env_indices) == 0:
            return
        
        # 批量写入导弹状态
        self.states['x'][env_indices, slot_indices] = missile_x[actual_fire]
        self.states['y'][env_indices, slot_indices] = missile_y[actual_fire]
        self.states['vx'][env_indices, slot_indices] = source_vx[actual_fire]
        self.states['vy'][env_indices, slot_indices] = source_vy[actual_fire]
        self.states['is_missile'][env_indices, slot_indices] = True
        self.states['is_active'][env_indices, slot_indices] = True
        self.states['is_player1'][env_indices, slot_indices] = is_player1
        self.states['alive'][env_indices, slot_indices] = False
        self.states['throttle'][env_indices, slot_indices] = 0.0
        self.states['rudder'][env_indices, slot_indices] = 0.0
        self.states['missile_count'][env_indices, slot_indices] = 0
        self.states['engine_time'][env_indices, slot_indices] = self.physics.MISSILE_ENGINE_DURATION
        self.states['target_idx'][env_indices, slot_indices] = target_idx
        self.states['source_idx'][env_indices, slot_indices] = source_idx
        self.states['prev_los_angle'][env_indices, slot_indices] = init_los_angle[actual_fire]

    def _check_events(self):
        """检查碰撞、自毁等事件（向量化，减少同步）"""
        active_missile_mask = self.states['is_missile'] & self.states['is_active']
        
        # 获取所有激活导弹的索引
        env_indices, entity_indices = torch.nonzero(active_missile_mask, as_tuple=True)
        
        if len(env_indices) == 0:
            return
        
        missile_x = self.states['x'][env_indices, entity_indices]
        missile_y = self.states['y'][env_indices, entity_indices]
        target_idx = self.states['target_idx'][env_indices, entity_indices]
        
        # 检查目标有效性
        target_valid = (target_idx >= 0) & (target_idx < self.max_entities)
        
        # 获取目标位置
        target_env_indices = env_indices[target_valid]
        target_entity_indices = target_idx[target_valid]
        
        if len(target_env_indices) > 0:
            target_x = self.states['x'][target_env_indices, target_entity_indices]
            target_y = self.states['y'][target_env_indices, target_entity_indices]
            target_alive = self.states['alive'][target_env_indices, target_entity_indices]
            
            # 计算距离
            dx = missile_x[target_valid] - target_x
            dy = missile_y[target_valid] - target_y
            dist_sq = dx*dx + dy*dy
            
            # 判断命中
            hit_mask_local = (dist_sq < self.hit_radius_sq) & target_alive
            
            # 批量处理命中（无条件分支）
            hit_env_indices = target_env_indices[hit_mask_local]
            hit_entity_indices = entity_indices[target_valid][hit_mask_local]
            hit_target_indices = target_entity_indices[hit_mask_local]
            
            # 使命中的导弹失效
            self.states['is_active'][hit_env_indices, hit_entity_indices] = False
            
            # 使被击中的飞机失效
            self.states['alive'][hit_env_indices, hit_target_indices] = False
            self.states['is_active'][hit_env_indices, hit_target_indices] = False
        
        # 检查自毁（向量化，无条件分支）
        current_active_missile_mask = self.states['is_missile'] & self.states['is_active']
        missile_v_sq = self.states['vx']**2 + self.states['vy']**2
        self_destruct_mask = (missile_v_sq < self.self_destruct_speed_sq) & current_active_missile_mask
        # 直接应用掩码，无需检查.any()
        self.states['is_active'] = torch.where(
            self_destruct_mask,
            torch.zeros_like(self.states['is_active']),
            self.states['is_active']
        )
        
        # 注意：无边界检测，采用无限大地图设计
        # 避免非物理边界，防止 RL 学到依赖地图边缘的奇怪策略

    def _check_game_over(self):
        """检查游戏是否结束"""
        p1_alive = self.states['alive'][:, self.P1_IDX]
        p2_alive = self.states['alive'][:, self.P2_IDX]
        p1_missiles = self.states['missile_count'][:, self.P1_IDX]
        p2_missiles = self.states['missile_count'][:, self.P2_IDX]
        
        # 检查是否有激活的导弹
        active_missiles = self.states['is_missile'] & self.states['is_active']
        any_active_missiles = torch.any(active_missiles, dim=1)
        
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        winner = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        
        # 双方都被击毁
        both_dead = ~p1_alive & ~p2_alive
        done |= both_dead
        # winner 保持为 0 (draw)
        
        # P1被击毁
        p1_dead = ~p1_alive & p2_alive
        done |= p1_dead
        winner = torch.where(p1_dead, self._winner_p2, winner)
        
        # P2被击毁
        p2_dead = p1_alive & ~p2_alive
        done |= p2_dead
        winner = torch.where(p2_dead, self._winner_p1, winner)
        
        # 双方弹药耗尽且无在途导弹
        ammo_empty = (p1_missiles == 0) & (p2_missiles == 0) & ~any_active_missiles
        done |= ammo_empty
        # winner 保持为 0 (draw)
        
        return done, winner

    def compute_rewards(self, done_mask, winner_mask=None):
        """计算奖励（完全向量化，使用预分配常量）"""
        if winner_mask is None:
            winner_mask = self.winner
        
        p1_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        p2_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        
        # 游戏结束奖励（使用预分配常量，无重复创建）
        # P1胜利
        p1_wins = done_mask & (winner_mask == 1)
        p1_reward = torch.where(p1_wins, self._reward_win, p1_reward)
        p2_reward = torch.where(p1_wins, self._reward_lose, p2_reward)
        
        # P2胜利
        p2_wins = done_mask & (winner_mask == 2)
        p1_reward = torch.where(p2_wins, self._reward_lose, p1_reward)
        p2_reward = torch.where(p2_wins, self._reward_win, p2_reward)
        
        # 平局
        draws = done_mask & (winner_mask == 0)
        p1_reward = torch.where(draws, self._reward_draw, p1_reward)
        p2_reward = torch.where(draws, self._reward_draw, p2_reward)
        
        # 生存奖励
        p1_alive = self.states['alive'][:, self.P1_IDX] & ~self.done
        p2_alive = self.states['alive'][:, self.P2_IDX] & ~self.done
        p1_reward += torch.where(p1_alive, self._reward_survive, self._reward_zero)
        p2_reward += torch.where(p2_alive, self._reward_survive, self._reward_zero)
        
        return {'p1': p1_reward, 'p2': p2_reward}

    def get_observations(self):
        """获取观察空间（归一化、相对位置）"""
        # 获取飞机状态
        p1_x = self.states['x'][:, self.P1_IDX]
        p1_y = self.states['y'][:, self.P1_IDX]
        p1_angle = self.states['angle'][:, self.P1_IDX]
        p1_speed = self.states['speed'][:, self.P1_IDX]
        p1_missiles = self.states['missile_count'][:, self.P1_IDX]
        p1_alive = self.states['alive'][:, self.P1_IDX]
        
        p2_x = self.states['x'][:, self.P2_IDX]
        p2_y = self.states['y'][:, self.P2_IDX]
        p2_angle = self.states['angle'][:, self.P2_IDX]
        p2_speed = self.states['speed'][:, self.P2_IDX]
        p2_missiles = self.states['missile_count'][:, self.P2_IDX]
        p2_alive = self.states['alive'][:, self.P2_IDX]
        
        # 计算相对位置和距离
        dx = p2_x - p1_x
        dy = p2_y - p1_y
        distance = torch.sqrt(dx*dx + dy*dy + self.physics.epsilon)
        bearing = torch.atan2(dy, dx) * (180.0 / math.pi)
        relative_angle_p1 = (bearing - p1_angle + 180) % 360 - 180
        relative_angle_p2 = (bearing - p1_angle + 180) % 360 - 180
        
        # 构造观察空间
        obs_p1 = {
            # 己方状态（归一化）
            'x': p1_x / self.battlefield_size,
            'y': p1_y / self.battlefield_size,
            'angle': p1_angle / 360.0,
            'speed': p1_speed / 400.0,
            'missiles': p1_missiles.float() / self.initial_missiles,
            'alive': p1_alive.float(),
            
            # 敌方相对状态
            'enemy_distance': distance / self.battlefield_size,
            'enemy_relative_angle': relative_angle_p1 / 180.0,
            'enemy_speed': p2_speed / 400.0,
            'enemy_alive': p2_alive.float(),
        }
        
        obs_p2 = {
            'x': p2_x / self.battlefield_size,
            'y': p2_y / self.battlefield_size,
            'angle': p2_angle / 360.0,
            'speed': p2_speed / 400.0,
            'missiles': p2_missiles.float() / self.initial_missiles,
            'alive': p2_alive.float(),
            
            'enemy_distance': distance / self.battlefield_size,
            'enemy_relative_angle': relative_angle_p2 / 180.0,
            'enemy_speed': p1_speed / 400.0,
            'enemy_alive': p1_alive.float(),
        }
        
        # TODO: 添加导弹信息（最多跟踪5枚最近的导弹）
        
        return {'p1': obs_p1, 'p2': obs_p2}

    def get_render_state(self, env_idx: int = 0) -> dict:
        """获取用于渲染的状态（单个环境）
        
        Args:
            env_idx: 环境索引（默认为0）
            
        Returns:
            dict: 包含 aircraft1, aircraft2, missiles 的渲染状态
        """
        from collections import deque
        
        # 可渲染实体类（支持属性访问）
        class RenderableEntity:
            def __init__(self, data_dict):
                for key, value in data_dict.items():
                    setattr(self, key, value)
        
        # 创建可渲染实体的辅助函数
        def create_renderable(idx, color, is_missile=False):
            is_active = bool(self.states['is_active'][env_idx, idx].cpu())
            # 导弹的 alive 应该等于 is_active（用于渲染判断）
            alive_val = is_active if is_missile else bool(self.states['alive'][env_idx, idx].cpu())
            data = {
                'x': float(self.states['x'][env_idx, idx].cpu()),
                'y': float(self.states['y'][env_idx, idx].cpu()),
                'angle': float(self.states['angle'][env_idx, idx].cpu()),
                'speed': float(self.states['speed'][env_idx, idx].cpu()),
                'alive': alive_val,
                'is_active': is_active,
                'color': color,
                'trail': deque(maxlen=100),  # 空轨迹，需要外部维护
                'is_missile': is_missile,
                'rudder': float(self.states['rudder'][env_idx, idx].cpu()),
                'throttle': float(self.states['throttle'][env_idx, idx].cpu()),
                'turn_rate': float(self.states['turn_rate'][env_idx, idx].cpu()),
                'missiles': int(self.states['missile_count'][env_idx, idx].cpu()),
                'mach': float(self.states['speed'][env_idx, idx].cpu()) / 340.0,
                'g_load': float(self.states['n_load'][env_idx, idx].cpu()),
                'n_load': float(self.states['n_load'][env_idx, idx].cpu()),
                'vx': float(self.states['vx'][env_idx, idx].cpu()),
                'vy': float(self.states['vy'][env_idx, idx].cpu()),
                'engine_time': float(self.states['engine_time'][env_idx, idx].cpu()) if is_missile else 0.0,
            }
            return RenderableEntity(data)
        
        # 颜色定义
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        LIGHT_RED = (255, 150, 150)
        LIGHT_BLUE = (150, 150, 255)
        
        # 获取飞机状态
        aircraft1 = create_renderable(self.P1_IDX, RED, is_missile=False)
        aircraft2 = create_renderable(self.P2_IDX, BLUE, is_missile=False)
        
        # 获取所有激活的导弹
        missiles = []
        for slot_idx in range(self.FIRST_MISSILE_IDX, self.max_entities):
            if self.states['is_active'][env_idx, slot_idx] and self.states['is_missile'][env_idx, slot_idx]:
                is_p1_missile = bool(self.states['is_player1'][env_idx, slot_idx].cpu())
                missile_color = LIGHT_RED if is_p1_missile else LIGHT_BLUE
                missile = create_renderable(slot_idx, missile_color, is_missile=True)
                missile.is_player1 = is_p1_missile
                missile.slot_idx = slot_idx  # 用于轨迹标识
                missiles.append(missile)
        
        return {
            'aircraft1': aircraft1,
            'aircraft2': aircraft2,
            'missiles': missiles,
            'game_over': bool(self.done[env_idx].cpu()),
            'winner': self._get_winner_str(env_idx),
        }
    
    def _get_winner_str(self, env_idx: int) -> str:
        """获取胜利者字符串"""
        if not self.done[env_idx]:
            return None
        winner_val = int(self.winner[env_idx].cpu())
        if winner_val == 1:
            return 'red'
        elif winner_val == 2:
            return 'blue'
        else:
            return 'draw'

