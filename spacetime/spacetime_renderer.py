# -*- coding: utf-8 -*-
"""
时空图3D渲染模块 (Spacetime Diagram Renderer)

功能概述:
    使用Pygame实现时空图的三维透视投影渲染。
    将时空数据（由spacetime_core.py或tensor_spacetime.py计算）转换为2D屏幕像素。
    可在两个后端之间切换

渲染内容:
    - 时空立方体: 50km x 50km x 20秒 的三维边框
    - 时间层级网格: 0/5/10/15秒的水平网格
    - 轨迹预测线: 当前舵量预测 + 全力回转逃脱曲线
    - 敌方威胁锥: 导弹可达范围的母线和时间截面
    - 导弹预测轨迹: 来袭导弹的未来路径
    - HUD信息: 飞机/导弹的马赫数和G值

坐标系:
    - 世界空间: X向右, Y向前, T向上(过去)向下(未来)
    - 相机空间: X向右, Y向上, Z向观察者
    - 屏幕空间: X向右, Y向下 (左上角原点)

模块组成:
    - Camera3D: 3D相机，处理透视投影和坐标变换
    - SpacetimeRenderer: 时空图渲染器，绘制所有可视元素
"""

import math
import pygame
from typing import List, Tuple, Dict, Optional, Any

from spacetime.spacetime_core import SpacetimePoint, SpacetimeComputer


# =============================================================================
# 3D相机
# =============================================================================

class Camera3D:
    """
    3D相机 - 处理世界到屏幕的透视投影
    
    坐标变换流程:
        世界坐标 -> 以玩家为中心 -> Yaw旋转 -> Pitch旋转 -> 透视除法 -> 屏幕坐标
    
    可调参数:
        - pitch: 俯仰角（30°俯视）
        - yaw: 偏航角（绕时间轴旋转，玩家可Q/E控制）
        - fov: 视场角（60°）
        - world_scale: 空间缩放（50km映射到[-1,1]）
        - time_scale: 时间缩放（20秒压缩到[-0.5,0.5]，避免竖轴太高）
    """
    
    def __init__(self, view_width: int, view_height: int):
        self.view_width = view_width
        self.view_height = view_height
        
        # 相机参数
        self.pitch = 30.0  # 俯角（正值为俯视）
        self.yaw = 0.0  # 偏航角（绕竖直轴旋转）
        
        # 投影参数
        self.fov = 60.0  # 视场角（度）
        self.near = 0.1  # 近平面
        self.far = 100000.0  # 远平面
        
        # 相机距离场景中心的距离（用于透视效果）
        self.distance = 30000.0  # 米
        
        # 缩放因子：将世界坐标转换为归一化坐标
        self.world_scale = 1.0 / 25000.0  # 50km范围映射到[-1, 1]
        self.time_scale = 1.0 / 20.0  # 20秒时间范围映射到[-0.5, 0.5]（压缩竖轴）
        
        # 预计算三角函数
        self._update_transform()
    
    def _update_transform(self):
        """更新变换矩阵参数"""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        
        self.cos_pitch = math.cos(pitch_rad)
        self.sin_pitch = math.sin(pitch_rad)
        self.cos_yaw = math.cos(yaw_rad)
        self.sin_yaw = math.sin(yaw_rad)
        
        # 透视投影参数
        self.focal_length = 1.0 / math.tan(math.radians(self.fov / 2))
    
    def set_yaw(self, yaw: float):
        """设置偏航角"""
        self.yaw = yaw % 360
        self._update_transform()
    
    def rotate_yaw(self, delta: float):
        """旋转偏航角"""
        self.yaw = (self.yaw + delta) % 360
        self._update_transform()
    
    def world_to_camera(self, x: float, y: float, t: float, 
                        center_x: float, center_y: float) -> Tuple[float, float, float]:
        """
        将世界坐标转换为相机坐标
        
        坐标系约定：
        - 世界空间: X向右, Y向前, T向上(过去)向下(未来)
        - 相机空间: X向右, Y向上, Z向外(向观察者)
        
        Args:
            x, y: 世界空间坐标
            t: 时间坐标（过去为正，未来为负）
            center_x, center_y: 场景中心（玩家位置）
            
        Returns:
            (cam_x, cam_y, cam_z): 相机空间坐标
        """
        # 1. 平移到以玩家为中心
        dx = (x - center_x) * self.world_scale
        dy = (y - center_y) * self.world_scale
        dz = t * self.time_scale  # 时间轴：过去在上(正), 未来在下(负)
        
        # 2. 绕竖直轴(Z/时间轴)旋转 (yaw) - 在水平面旋转
        rx = dx * self.cos_yaw - dy * self.sin_yaw
        ry = dx * self.sin_yaw + dy * self.cos_yaw
        rz = dz
        
        # 3. 绕X轴旋转 (pitch) - 俯仰
        # 俯视时，远处在上，近处在下
        # pitch > 0 表示俯视
        # 反转X轴以保持与常规2D视图一致（而不是镜像）
        cam_x = -rx
        cam_y = rz * self.cos_pitch + ry * self.sin_pitch  # 时间轴投影到Y
        cam_z = -rz * self.sin_pitch + ry * self.cos_pitch  # 深度
        
        # 4. 添加相机距离
        cam_z += self.distance * self.world_scale
        
        return cam_x, cam_y, cam_z
    
    def camera_to_screen(self, cam_x: float, cam_y: float, cam_z: float,
                        offset_x: float = 0, offset_y: float = 0) -> Tuple[float, float, float]:
        """
        透视投影：将相机坐标转换为屏幕坐标
        
        Args:
            cam_x, cam_y, cam_z: 相机空间坐标
            offset_x, offset_y: 视口偏移
            
        Returns:
            (screen_x, screen_y, depth): 屏幕坐标和深度
        """
        if cam_z <= self.near * self.world_scale:
            return None, None, None  # 在近平面后面
        
        # 透视除法
        inv_z = 1.0 / cam_z
        proj_x = cam_x * self.focal_length * inv_z
        proj_y = cam_y * self.focal_length * inv_z
        
        # 转换到屏幕坐标
        screen_x = offset_x + self.view_width / 2 + proj_x * self.view_width / 2
        screen_y = offset_y + self.view_height / 2 - proj_y * self.view_height / 2  # Y轴向上
        
        return screen_x, screen_y, cam_z
    
    def project(self, x: float, y: float, t: float,
               center_x: float, center_y: float,
               offset_x: float = 0, offset_y: float = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        完整的3D到2D投影
        
        Args:
            x, y: 世界空间坐标
            t: 时间坐标
            center_x, center_y: 场景中心
            offset_x, offset_y: 视口偏移
            
        Returns:
            (screen_x, screen_y, depth) 或 (None, None, None)
        """
        cam_x, cam_y, cam_z = self.world_to_camera(x, y, t, center_x, center_y)
        return self.camera_to_screen(cam_x, cam_y, cam_z, offset_x, offset_y)
    
    def project_direction(self, x: float, y: float, angle: float,
                         center_x: float, center_y: float,
                         offset_x: float = 0, offset_y: float = 0) -> Optional[float]:
        """
        将世界空间中的方向投影到屏幕空间
        
        Args:
            x, y: 世界空间位置
            angle: 世界空间中的角度（度）
            center_x, center_y: 场景中心
            offset_x, offset_y: 视口偏移
            
        Returns:
            屏幕空间中的角度（度），如果无法计算则返回None
        """
        # 计算方向向量的终点
        angle_rad = math.radians(angle)
        dx = math.cos(angle_rad) * 100  # 100米的小位移
        dy = math.sin(angle_rad) * 100
        
        # 投影起点和终点
        sx1, sy1, _ = self.project(x, y, 0, center_x, center_y, offset_x, offset_y)
        sx2, sy2, _ = self.project(x + dx, y + dy, 0, center_x, center_y, offset_x, offset_y)
        
        if sx1 is None or sx2 is None:
            return None
        
        # 计算屏幕空间中的角度
        screen_angle = math.degrees(math.atan2(sy2 - sy1, sx2 - sx1))
        return screen_angle


# =============================================================================
# 时空图渲染器
# =============================================================================

class SpacetimeRenderer:
    """
    时空图渲染器 - 将时空数据绘制到Pygame Surface
    
    渲染分层(从后到前):
        1. 背景填充
        2. 时空立方体边框
        3. 时间层级网格 (0/5/10/15秒)
        4. 敌方威胁锥
        5. 玩家轨迹预测线
        6. 导弹预测轨迹
        7. 当前位置标记 + HUD
        8. 时间轴标签
    
    颜色约定:
        - 红方: RED/LIGHT_RED
        - 蓝方: BLUE/LIGHT_BLUE  
        - 全力回转线: YELLOW
        - 无威胁导弹段: WHITE
        - 威胁锥: THREAT_RED/THREAT_BLUE
    
    时间同步:
        预测线根据 (current_time - prediction_time) 计算时间偏移，
        使预测线随时间流逝自然上升，避免视觉滞后。
    """
    
    def __init__(self, view_width: int, view_height: int, config: Dict = None):
        """
        Args:
            view_width: 单个视口宽度
            view_height: 单个视口高度
            config: 配置字典
        """
        self.view_width = view_width
        self.view_height = view_height
        self.config = config or {}
        
        # 创建两个玩家的相机
        self.cameras = {
            1: Camera3D(view_width, view_height),
            2: Camera3D(view_width, view_height)
        }
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_RED = (255, 150, 150)
        self.LIGHT_BLUE = (150, 150, 255)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        self.GRID_COLOR = (60, 60, 60)
        self.THREAT_RED = (255, 100, 100)    # 红方威胁锥颜色
        self.THREAT_BLUE = (100, 100, 255)   # 蓝方威胁锥颜色
        
        # 时间层级颜色（未来时间层）
        self.LAYER_COLORS = {
            0: (150, 150, 150),   # 现在
            -5: (100, 100, 100),  # 5秒后
            -10: (70, 70, 70),    # 10秒后
            -15: (50, 50, 50),    # 15秒后
            -20: (35, 35, 35),    # 20秒后
        }
        
        # 显示选项
        self.show_turn_predictions = {1: True, 2: True}  # 是否显示全力掉头曲线
        
        # 立方体参数
        self.cube_half_size = 25000  # 米，立方体半边长
        self.future_time = 20.0  # 未来时间窗口（20秒）
        
    def set_show_turn_predictions(self, player_id: int, show: bool):
        """设置是否显示全力掉头预测线"""
        self.show_turn_predictions[player_id] = show
    
    def toggle_turn_predictions(self, player_id: int):
        """切换全力掉头预测线显示"""
        self.show_turn_predictions[player_id] = not self.show_turn_predictions[player_id]
        return self.show_turn_predictions[player_id]
    
    def rotate_camera(self, player_id: int, delta: float):
        """旋转玩家相机"""
        if player_id in self.cameras:
            self.cameras[player_id].rotate_yaw(delta)
    
    def _draw_line_3d(self, surface: pygame.Surface, camera: Camera3D,
                     p1: Tuple[float, float, float], p2: Tuple[float, float, float],
                     center: Tuple[float, float], offset: Tuple[float, float],
                     color: Tuple[int, int, int], width: int = 1) -> bool:
        """
        绘制3D线段
        
        Args:
            surface: Pygame surface
            camera: 3D相机
            p1, p2: 3D点 (x, y, t)
            center: 场景中心 (x, y)
            offset: 视口偏移 (x, y)
            color: 线条颜色
            width: 线条宽度
            
        Returns:
            是否成功绘制
        """
        sx1, sy1, d1 = camera.project(p1[0], p1[1], p1[2], center[0], center[1], 
                                       offset[0], offset[1])
        sx2, sy2, d2 = camera.project(p2[0], p2[1], p2[2], center[0], center[1],
                                       offset[0], offset[1])
        
        if sx1 is None or sx2 is None:
            return False
        
        # 检查是否在视口内
        if not self._in_viewport(sx1, sy1, offset) and not self._in_viewport(sx2, sy2, offset):
            return False
        
        # 根据深度调整颜色（远处变暗）
        avg_depth = (d1 + d2) / 2
        depth_factor = max(0.3, min(1.0, 1.5 - avg_depth * 0.5))
        adjusted_color = tuple(int(c * depth_factor) for c in color)
        
        pygame.draw.line(surface, adjusted_color, (sx1, sy1), (sx2, sy2), width)
        return True
    
    def _in_viewport(self, sx: float, sy: float, offset: Tuple[float, float]) -> bool:
        """检查点是否在视口内"""
        margin = 50
        return (offset[0] - margin <= sx <= offset[0] + self.view_width + margin and
                offset[1] - margin <= sy <= offset[1] + self.view_height + margin)
    
    def _draw_spacetime_cube(self, surface: pygame.Surface, camera: Camera3D,
                            center: Tuple[float, float], offset: Tuple[float, float]):
        """绘制时空长方体框架（仅未来部分）0秒到-20秒）"""
        hs = self.cube_half_size  # 半边长
        
        # 时间范围：0（现在）到 -20（未来）
        t_top = 0                     # 现在（顶面）
        t_bottom = -self.future_time  # 20秒后（底面）
        
        corners = [
            (-hs, -hs, t_top), (hs, -hs, t_top), (hs, hs, t_top), (-hs, hs, t_top),  # 顶面
            (-hs, -hs, t_bottom), (hs, -hs, t_bottom), (hs, hs, t_bottom), (-hs, hs, t_bottom),  # 底面
        ]
        
        # 将相对坐标转换为世界坐标
        world_corners = [(center[0] + c[0], center[1] + c[1], c[2]) for c in corners]
        
        # 长方体边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 顶面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 底面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直边
        ]
        
        for i, j in edges:
            self._draw_line_3d(surface, camera, world_corners[i], world_corners[j],
                              center, offset, self.GRID_COLOR, 1)
    
    def _draw_time_grids(self, surface: pygame.Surface, camera: Camera3D,
                        center: Tuple[float, float], offset: Tuple[float, float]):
        """绘制时间层级网格"""
        hs = self.cube_half_size
        grid_spacing = 5000  # 网格间距（米）
        
        # 四个时间层：0（现在）, -5, -10, -15秒后
        time_levels = [0, -5, -10, -15]
        
        for t_level in time_levels:
            color = self.LAYER_COLORS.get(t_level, self.GRID_COLOR)
            
            # 绘制网格线
            for offset_val in range(-int(hs), int(hs) + 1, grid_spacing):
                # 平行于X轴的线
                p1 = (center[0] + offset_val, center[1] - hs, t_level)
                p2 = (center[0] + offset_val, center[1] + hs, t_level)
                self._draw_line_3d(surface, camera, p1, p2, center, offset, color, 1)
                
                # 平行于Y轴的线
                p1 = (center[0] - hs, center[1] + offset_val, t_level)
                p2 = (center[0] + hs, center[1] + offset_val, t_level)
                self._draw_line_3d(surface, camera, p1, p2, center, offset, color, 1)
    
    def _draw_trajectory(self, surface: pygame.Surface, camera: Camera3D,
                        points: List[SpacetimePoint], center: Tuple[float, float],
                        offset: Tuple[float, float], color: Tuple[int, int, int],
                        width: int = 2, time_offset: float = 0.0):
        """绘制轨迹线
        
        Args:
            time_offset: 时间偏移（将轨迹点的t值偏移，正值表示向上/过去方向移动）
        """
        if len(points) < 2:
            return
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            # 应用时间偏移：预测点随时间流逝向上移动
            t1 = p1.t + time_offset
            t2 = p2.t + time_offset
            self._draw_line_3d(surface, camera, 
                              (p1.x, p1.y, t1), (p2.x, p2.y, t2),
                              center, offset, color, width)
    
    def _draw_threat_cone(self, surface: pygame.Surface, camera: Camera3D,
                         cone_data: Dict, center: Tuple[float, float],
                         offset: Tuple[float, float], enemy_id: int):
        """绘制威胁锥
        
        Args:
            enemy_id: 敌方玩家ID，用于确定颜色
        """
        if not cone_data:
            return
        
        # 根据敌方ID选择颜色
        threat_color = self.THREAT_RED if enemy_id == 1 else self.THREAT_BLUE
        
        meridians = cone_data.get('meridians', [])
        ellipses = cone_data.get('ellipses', [])
        
        # 绘制母线
        for meridian in meridians:
            points = meridian.get('points', [])
            if len(points) < 2:
                continue
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # 根据速度调整颜色（接近音速时变白）
                if p2.speed < 340:
                    color = self.WHITE
                else:
                    color = threat_color
                
                self._draw_line_3d(surface, camera,
                                  (p1.x, p1.y, p1.t), (p2.x, p2.y, p2.t),
                                  center, offset, color, 1)
        
        # 绘制时间截面椭圆
        for ellipse in ellipses:
            points = ellipse.get('points', [])
            if len(points) < 2:
                continue
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                self._draw_line_3d(surface, camera,
                                  (p1.x, p1.y, p1.t), (p2.x, p2.y, p2.t),
                                  center, offset, threat_color, 1)
    
    def _draw_current_position_marker(self, surface: pygame.Surface, camera: Camera3D,
                                      x: float, y: float, angle: float, speed: float,
                                      center: Tuple[float, float], offset: Tuple[float, float],
                                      color: Tuple[int, int, int], is_missile: bool = False,
                                      show_hud: bool = False, g_force: float = 0.0):
        """绘制当前位置标记（在t=0平面上）"""
        sx, sy, _ = camera.project(x, y, 0, center[0], center[1], offset[0], offset[1])
        
        if sx is None:
            return
        
        # 计算屏幕空间中的方向角度
        screen_angle = camera.project_direction(x, y, angle, center[0], center[1], 
                                                offset[0], offset[1])
        if screen_angle is None:
            screen_angle = angle  # 回退到原始角度
        
        angle_rad = math.radians(screen_angle)
        
        if is_missile:
            # 导弹：三角形
            size = 6
            points = [
                (sx + math.cos(angle_rad) * size * 2, sy + math.sin(angle_rad) * size * 2),
                (sx + math.cos(angle_rad + 2.1) * size, sy + math.sin(angle_rad + 2.1) * size),
                (sx + math.cos(angle_rad - 2.1) * size, sy + math.sin(angle_rad - 2.1) * size),
            ]
            pygame.draw.polygon(surface, color, points)
            
            # 导弹速度方向线
            end_x = sx + math.cos(angle_rad) * 15
            end_y = sy + math.sin(angle_rad) * 15
            pygame.draw.line(surface, color, (sx, sy), (end_x, end_y), 1)
            
            # 导弹HUD：马赫数和G值
            if show_hud:
                mach = speed / 340.0
                font = pygame.font.Font(None, 16)
                hud_text = f"{mach:.1f}Ma {g_force:.0f}G"
                text_surface = font.render(hud_text, True, color)
                surface.blit(text_surface, (sx + 8, sy - 12))
        else:
            # 飞机：圆形
            pygame.draw.circle(surface, color, (int(sx), int(sy)), 6)
            
            # 速度方向线
            end_x = sx + math.cos(angle_rad) * 20
            end_y = sy + math.sin(angle_rad) * 20
            pygame.draw.line(surface, color, (sx, sy), (end_x, end_y), 2)
            
            # 显示HUD：马赫数和G值
            if show_hud:
                mach = speed / 340.0
                font = pygame.font.Font(None, 18)
                hud_text = f"{mach:.1f}Ma {g_force:.1f}G"
                text_surface = font.render(hud_text, True, color)
                surface.blit(text_surface, (sx + 10, sy - 15))
    
    def draw_spacetime_view(self, surface: pygame.Surface, player_id: int,
                           player, enemy, missiles: List,
                           spacetime_computer: SpacetimeComputer,
                           offset_x: int = 0, offset_y: int = 0):
        """
        绘制单个玩家的时空图视图
        
        Args:
            surface: Pygame surface
            player_id: 玩家ID (1 或 2)
            player: 玩家飞机
            enemy: 敌方飞机
            missiles: 导弹列表
            spacetime_computer: 时空图计算器
            offset_x, offset_y: 视口偏移
        """
        camera = self.cameras.get(player_id)
        if camera is None:
            return
        
        center = (player.x, player.y)
        offset = (offset_x, offset_y)
        
        # 设置裁剪区域，防止绘制内容越过视口边界
        clip_rect = pygame.Rect(offset_x, offset_y, self.view_width, self.view_height)
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        # 填充背景
        pygame.draw.rect(surface, self.BLACK, 
                        (offset_x, offset_y, self.view_width, self.view_height))
        
        # 绘制时空立方体
        self._draw_spacetime_cube(surface, camera, center, offset)
        
        # 绘制时间层级网格
        self._draw_time_grids(surface, camera, center, offset)
        
        # 获取时空数据
        player_data = spacetime_computer.get_player_spacetime_data(player_id)
        enemy_id = 2 if player_id == 1 else 1
        enemy_data = spacetime_computer.get_player_spacetime_data(enemy_id)
        
        # 获取当前时间（用于时间同步）
        current_time = spacetime_computer.get_current_time()
        
        # 玩家颜色
        player_color = self.RED if player_id == 1 else self.BLUE
        
        # 绘制玩家当前舵量预测
        current_pred = player_data.get('current_prediction', [])
        current_pred_time = player_data.get('current_prediction_time', 0.0)
        # 时间偏移：当前时间 - 预测创建时间，预测线随时间流逝向上移动
        time_offset = current_time - current_pred_time
        # 使用低饱和度颜色
        pred_color = self.LIGHT_RED if player_id == 1 else self.LIGHT_BLUE
        self._draw_trajectory(surface, camera, current_pred, center, offset,
                             pred_color, 2, time_offset)
        
        # 绘制全力掉头预测（如果启用）
        if self.show_turn_predictions.get(player_id, True):
            left_pred = player_data.get('left_turn_prediction', [])
            left_pred_time = player_data.get('left_turn_prediction_time', 0.0)
            left_time_offset = current_time - left_pred_time
            
            right_pred = player_data.get('right_turn_prediction', [])
            right_pred_time = player_data.get('right_turn_prediction_time', 0.0)
            right_time_offset = current_time - right_pred_time
            
            self._draw_trajectory(surface, camera, left_pred, center, offset,
                                 self.YELLOW, 1, left_time_offset)
            self._draw_trajectory(surface, camera, right_pred, center, offset,
                                 self.YELLOW, 1, right_time_offset)
        
        # 绘制敌方威胁锥
        if hasattr(enemy, 'alive') and enemy.alive:
            threat_cone = spacetime_computer.get_threat_cone(enemy, player_id)
            self._draw_threat_cone(surface, camera, threat_cone, center, offset, enemy_id)
        
        # 绘制导弹
        for missile in missiles:
            if not getattr(missile, 'alive', True):
                continue
            
            missile_id = getattr(missile, 'slot_idx', id(missile))
            missile_data = spacetime_computer.get_missile_spacetime_data(missile_id)
            
            # 导弹颜色
            is_player1_missile = getattr(missile, 'is_player1', True)
            missile_color = self.LIGHT_RED if is_player1_missile else self.LIGHT_BLUE
            
            # 预测轨迹（玩家能看到敌方发射的、追击自己的导弹）
            # 红方(player_id=1)看到蓝方发射的导弹(is_player1=False)
            # 蓝方(player_id=2)看到红方发射的导弹(is_player1=True)
            is_enemy_missile = (player_id == 1 and not is_player1_missile) or \
                               (player_id == 2 and is_player1_missile)
            if is_enemy_missile:
                missile_pred = missile_data.get('trajectory_prediction', [])
                missile_pred_time = missile_data.get('trajectory_prediction_time', 0.0)
                missile_time_offset = current_time - missile_pred_time
                
                # 移除“无威胁”白线逻辑，始终使用导弹颜色绘制完整轨迹
                self._draw_trajectory(surface, camera, missile_pred, center, offset,
                                     missile_color, 2, missile_time_offset)
            
            # 导弹当前位置（显示HUD）
            missile_speed = math.sqrt(getattr(missile, 'vx', 0)**2 + getattr(missile, 'vy', 0)**2)
            missile_g = getattr(missile, 'g_load', getattr(missile, 'n_load', 0.0))
            self._draw_current_position_marker(surface, camera, missile.x, missile.y,
                                              missile.angle, missile_speed, center, offset,
                                              missile_color, True, True, missile_g)
        
        # 绘制玩家当前位置（显示HUD）
        player_speed = math.sqrt(getattr(player, 'vx', 0)**2 + getattr(player, 'vy', 0)**2)
        # G值优先从g_load获取，回退到n_load，最后用0
        player_g = getattr(player, 'g_load', getattr(player, 'n_load', 0.0))
        self._draw_current_position_marker(surface, camera, player.x, player.y,
                                          player.angle, player_speed, center, offset, 
                                          player_color, False, True, player_g)
        
        # 绘制敌方当前位置（显示HUD）
        if hasattr(enemy, 'alive') and enemy.alive:
            enemy_color = self.BLUE if player_id == 1 else self.RED
            enemy_speed = math.sqrt(getattr(enemy, 'vx', 0)**2 + getattr(enemy, 'vy', 0)**2)
            enemy_g = getattr(enemy, 'g_load', getattr(enemy, 'n_load', 0.0))
            self._draw_current_position_marker(surface, camera, enemy.x, enemy.y,
                                              enemy.angle, enemy_speed, center, offset, 
                                              enemy_color, False, True, enemy_g)
        
        # 恢复裁剪区域
        surface.set_clip(old_clip)
        
        # 绘制边框（在裁剪区域外）
        pygame.draw.rect(surface, self.WHITE, 
                        (offset_x, offset_y, self.view_width, self.view_height), 1)
        
        # 绘制时间轴标签
        self._draw_time_labels(surface, camera, center, offset)
    
    def _draw_time_labels(self, surface: pygame.Surface, camera: Camera3D,
                         center: Tuple[float, float], offset: Tuple[float, float]):
        """绘制时间轴标签"""
        # 简单文本标签
        font = pygame.font.Font(None, 20)
        
        labels = [
            (0, "现在"),
            (-5, "5s"),
            (-10, "10s"),
            (-15, "15s"),
            (-20, "20s"),
        ]
        
        hs = self.cube_half_size
        for t, text in labels:
            # 在长方体边缘显示标签
            sx, sy, d = camera.project(center[0] - hs, center[1] - hs, t,
                                       center[0], center[1], offset[0], offset[1])
            if sx is not None:
                text_surface = font.render(text, True, self.WHITE)
                surface.blit(text_surface, (sx + 5, sy - 8))
