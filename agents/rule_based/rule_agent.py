# -*- coding: utf-8 -*-
"""
规则型智能体实现
基于状态机的中距空战战术 Agent，支持完全向量化的批量决策
"""

import torch
import math
from typing import Dict, Optional
from agents.base_agent import BaseAgent


class CrankAgent(BaseAgent):
    """基于状态机的中距空战规则 Agent
    
    实现三状态战术决策：
    - Approach: 接敌，追踪敌机
    - Fire: 满足条件时发射导弹
    - Defend: 检测威胁并执行防御机动
    
    完全向量化实现，支持高性能批量运行
    """
    
    # 状态常量
    STATE_APPROACH = 0
    STATE_FIRE = 1
    STATE_DEFEND = 2
    
    # 距离阈值（米）
    MAX_MISSILE_RANGE = 25000.0
    FIRE_RANGE_RATIO = 0.8
    THREAT_DISTANCE = 18000.0
    SAFE_DISTANCE = 25000.0
    MIN_SAFE_DISTANCE = 8000.0
    
    # 角度阈值（度）
    FIRE_ANGLE_THRESHOLD = 30.0
    HEADING_THRESHOLD = 5.0
    MAX_HEADING_ERROR = 45.0
    
    # 速度阈值（米/秒）
    MISSILE_EFFECTIVE_SPEED = 150.0
    
    # 开火冷却时间（秒）
    FIRE_COOLDOWN = 10  # 开火后 10 秒内不再开火，确保导弹一发一发打
    
    def __init__(self, device: str = 'cpu'):
        """初始化规则 Agent
        
        Args:
            device: 计算设备（规则 Agent 建议使用 'cpu'）
        """
        super().__init__(device)
        
        # 初始化状态张量（会在第一次 act 时根据 num_envs 创建）
        self.states = None
        self.threat_detected = None
        self.last_fire_time = None
        self.current_time = None  # 当前游戏时间
        self._initialized = False
    
    def _ensure_initialized(self, num_envs: int):
        """确保状态张量已初始化
        
        Args:
            num_envs: 环境数量
        """
        if not self._initialized or (self.states is not None and self.states.shape[0] != num_envs):
            self.states = torch.full((num_envs,), self.STATE_APPROACH, dtype=torch.long, device=self.device)
            self.threat_detected = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self.last_fire_time = torch.full((num_envs,), -999.0, dtype=torch.float32, device=self.device)  # 初始设为很久以前
            self.current_time = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
            self._num_envs = num_envs
            self._initialized = True
    
    def act(self, observation: Dict[str, torch.Tensor], dt: float = 1/60.0) -> Dict[str, torch.Tensor]:
        """根据观察生成动作（完全向量化）
        
        Args:
            observation: 观察字典，所有值为 [num_envs] 张量
            dt: 时间步长（秒），默认 1/60秒
        
        Returns:
            动作字典，所有值为 [num_envs] 张量
        """
        # 获取批量大小
        num_envs = observation['x'].shape[0]
        self._ensure_initialized(num_envs)
        
        # 更新当前时间
        self.current_time += dt
        
        # 1. 检测威胁
        threat_detected = self._detect_threats(observation)
        
        # 2. 更新状态机
        self._update_state(observation, threat_detected)
        
        # 3. 根据当前状态批量生成动作
        actions = self._generate_actions(observation, threat_detected)
        
        return actions
    
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """重置 Agent 状态
        
        Args:
            env_mask: 可选，形状 [num_envs] 的布尔张量，指定要重置的环境
        """
        if not self._initialized:
            return
        
        if env_mask is None:
            # 重置所有环境
            self.states.fill_(self.STATE_APPROACH)
            self.threat_detected.fill_(False)
            self.last_fire_time.fill_(-999.0)
            self.current_time.fill_(0.0)
        else:
            # 仅重置指定环境
            self.states[env_mask] = self.STATE_APPROACH
            self.threat_detected[env_mask] = False
            self.last_fire_time[env_mask] = -999.0
            self.current_time[env_mask] = 0.0
    
    def _detect_threats(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """检测威胁（向量化实现）
        
        Args:
            observation: 观察字典
        
        Returns:
            形状 [num_envs] 的布尔张量，表示是否检测到威胁
        """
        # 简化版威胁检测：基于敌方距离
        # 实际应用中可以加入导弹信息（如果观察中包含 missiles_info）
        enemy_distance = observation.get('enemy_distance', torch.full_like(observation['x'], float('inf')))
        
        # 威胁条件：敌方距离过近
        threat = enemy_distance < self.THREAT_DISTANCE
        
        # 注意：这里简化了威胁检测，实际中应检测来袭导弹
        # 如果观察中包含导弹信息，可以在此处添加更复杂的逻辑
        
        return threat
    
    def _update_state(self, observation: Dict[str, torch.Tensor], threat_detected: torch.Tensor) -> None:
        """更新状态机（批量处理）
        
        Args:
            observation: 观察字典
            threat_detected: 威胁检测结果 [num_envs]
        """
        enemy_distance = observation.get('enemy_distance', torch.full_like(observation['x'], float('inf')))
        heading_error = self._calculate_heading_error_batch(observation)
        missiles = observation.get('missiles', torch.zeros_like(observation['x']))
        
        # 计算开火条件
        fire_range = self.MAX_MISSILE_RANGE * self.FIRE_RANGE_RATIO
        
        # 检查冷却时间：距离上次开火是否已经超过冷却时间
        time_since_last_fire = self.current_time - self.last_fire_time
        cooldown_ready = time_since_last_fire >= self.FIRE_COOLDOWN
        
        can_fire = (
            (enemy_distance < fire_range) &
            (torch.abs(heading_error) < self.FIRE_ANGLE_THRESHOLD) &
            (missiles > 0) &
            cooldown_ready  # 添加冷却判断，确保不连射
        )
        
        # 强制防御条件
        should_defend = threat_detected | (enemy_distance < self.MIN_SAFE_DISTANCE)
        
        # 状态转换（按优先级）
        # 优先级1：威胁 → Defend
        self.states[should_defend] = self.STATE_DEFEND
        
        # 优先级2：可以开火 → Fire（仅对 Approach 状态）
        can_transition_to_fire = (self.states == self.STATE_APPROACH) & can_fire & ~should_defend
        self.states[can_transition_to_fire] = self.STATE_FIRE
        
        # 优先级3：Fire 完成 → Approach（开火后立即转回，避免连续开火）
        fire_complete = (self.states == self.STATE_FIRE) & ~can_fire
        self.states[fire_complete] = self.STATE_APPROACH
        
        # 优先级4：威胁解除 → Approach
        threat_clear = (self.states == self.STATE_DEFEND) & ~should_defend
        self.states[threat_clear] = self.STATE_APPROACH
    
    def _generate_actions(self, observation: Dict[str, torch.Tensor], threat_detected: torch.Tensor) -> Dict[str, torch.Tensor]:
        """批量生成动作（完全向量化，无条件分支）
        
        Args:
            observation: 观察字典
            threat_detected: 威胁检测结果
        
        Returns:
            动作字典
        """
        # 计算所有可能的动作（完全向量化）
        approach_rudder = self._compute_approach_rudder(observation)
        defend_result = self._compute_defend_actions(observation)
        
        # 状态掩码
        approach_mask = (self.states == self.STATE_APPROACH)
        fire_mask = (self.states == self.STATE_FIRE)
        defend_mask = (self.states == self.STATE_DEFEND)
        
        # 使用 torch.where 混合不同状态的动作（无条件分支）
        # 优先级：defend > fire > approach（通过嵌套where实现）
        rudder = torch.where(defend_mask, defend_result['rudder'],
                    torch.where(fire_mask, approach_rudder,
                        torch.where(approach_mask, approach_rudder,
                            torch.zeros_like(approach_rudder))))
        
        throttle = torch.where(defend_mask, defend_result['throttle'],
                      torch.ones_like(defend_result['throttle']))
        
        # Fire状态开火
        fire = fire_mask
        
        # 更新开火时间（向量化：使用where而不是条件索引）
        self.last_fire_time = torch.where(fire_mask, self.current_time, self.last_fire_time)
        
        return {
            'rudder': rudder,
            'throttle': throttle,
            'fire': fire
        }
    
    def _compute_approach_rudder(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算追踪舵量（完全向量化，返回所有环境）
        
        使用torch.where代替条件分支，无CPU-GPU同步
        """
        heading_error = self._calculate_heading_error_batch(observation)
        
        # 比例控制基础值
        proportional_rudder = torch.clamp(heading_error / self.MAX_HEADING_ERROR, -1.0, 1.0)
        
        # 使用where实现：大误差满舵，小误差比例控制
        rudder = torch.where(
            heading_error > self.HEADING_THRESHOLD, 
            torch.ones_like(heading_error),
            torch.where(
                heading_error < -self.HEADING_THRESHOLD,
                -torch.ones_like(heading_error),
                proportional_rudder
            )
        )
        
        return rudder
    
    def _compute_defend_actions(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算防御动作（完全向量化，返回所有环境）
        
        使用torch.where代替条件分支，无CPU-GPU同步
        """
        enemy_distance = observation.get('enemy_distance', 
            torch.full_like(observation['x'], float('inf')))
        
        # 计算逃逸方向舵量
        enemy_angle = self._calculate_target_angle_batch(observation)
        escape_angle = self._normalize_angle(enemy_angle + 180.0)
        self_angle = observation['angle']
        heading_error = self._normalize_angle(escape_angle - self_angle)
        
        # 比例控制基础值
        proportional_rudder = torch.clamp(heading_error / self.MAX_HEADING_ERROR, -1.0, 1.0)
        
        # 向量化舵量计算
        rudder = torch.where(
            heading_error > self.HEADING_THRESHOLD,
            torch.ones_like(heading_error),
            torch.where(
                heading_error < -self.HEADING_THRESHOLD,
                -torch.ones_like(heading_error),
                proportional_rudder
            )
        )
        
        # 向量化油门计算
        throttle = torch.where(
            enemy_distance < 5000, 
            torch.zeros_like(enemy_distance),
            torch.where(
                enemy_distance < 15000,
                torch.full_like(enemy_distance, 0.5),
                torch.ones_like(enemy_distance)
            )
        )
        
        return {'rudder': rudder, 'throttle': throttle}
    
    def _calculate_heading_error_batch(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """批量计算航向误差
        
        Args:
            observation: 观察字典
        
        Returns:
            航向误差 [num_envs]，范围 [-180, 180] 度
        """
        target_angle = self._calculate_target_angle_batch(observation)
        self_angle = observation['angle']
        heading_error = self._normalize_angle(target_angle - self_angle)
        return heading_error
    
    def _calculate_target_angle_batch(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """批量计算目标角度
        
        Args:
            observation: 观察字典
        
        Returns:
            目标角度 [num_envs]，单位：度
        """
        # 从观察中提取位置信息
        # 注意：观察中可能直接包含 enemy_relative_angle，或需要从位置计算
        if 'enemy_relative_angle' in observation:
            # 如果直接提供了相对角度，使用它
            return observation['enemy_relative_angle']
        
        # 否则从位置计算（假设观察中有敌方位置信息）
        # 这里简化处理：使用 enemy_distance 和假设的角度
        # 实际应用中需要根据具体观察格式调整
        
        # 如果观察中包含敌方绝对位置
        if 'enemy_x' in observation and 'enemy_y' in observation:
            dx = observation['enemy_x'] - observation['x']
            dy = observation['enemy_y'] - observation['y']
            target_angle = torch.atan2(dy, dx) * 180.0 / math.pi
            return target_angle
        
        # 如果只有相对角度，直接返回
        if 'enemy_relative_angle' in observation:
            return observation['enemy_relative_angle']
        
        # 备用：返回当前朝向（表示不改变方向）
        return observation['angle']
    
    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """归一化角度到 [-180, 180] 度
        
        Args:
            angle: 角度张量 [num_envs]
        
        Returns:
            归一化后的角度
        """
        normalized = (angle + 180.0) % 360.0 - 180.0
        return normalized
    
    def get_name(self) -> str:
        """返回 Agent 名称"""
        return "CrankAgent"
