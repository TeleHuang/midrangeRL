# -*- coding: utf-8 -*-

import math
import os

import pygame

# 时空图模块（延迟导入以避免循环依赖）
_spacetime_module = None

def get_system_font():
    """获取系统中可用的中文字体"""
    # Windows系统字体路径
    windows_fonts = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
    ]
    
    # 尝试加载Windows系统字体
    for font_path in windows_fonts:
        if os.path.exists(font_path):
            try:
                return pygame.font.Font(font_path, 24)
            except:
                continue
    
    # 如果Windows字体都失败，尝试使用系统字体
    try:
        return pygame.font.SysFont('microsoft yahei', 24)
    except:
        try:
            return pygame.font.SysFont('simhei', 24)
        except:
            # 如果都失败，使用默认字体
            return pygame.font.Font(None, 36)

def _get_spacetime_module():
    """延迟加载时空图模块"""
    global _spacetime_module
    if _spacetime_module is None:
        try:
            from spacetime import SpacetimeComputer, SpacetimeRenderer
            _spacetime_module = {
                'SpacetimeComputer': SpacetimeComputer,
                'SpacetimeRenderer': SpacetimeRenderer,
                'available': True
            }
        except ImportError as e:
            print(f"警告: 时空图模块加载失败: {e}")
            _spacetime_module = {'available': False}
    return _spacetime_module


class Visualizer:
    def __init__(self, window_width, window_height, config, font, spacetime_computer=None):
        # 窗口与布局尺寸
        self.window_width = window_width
        self.window_height = window_height
        self.top_panel_h = 100
        self.bottom_panel_h = 100
        self.battle_height = self.window_height - self.top_panel_h - self.bottom_panel_h  # 期望800

        # 视口尺寸（左右各一个正方形）
        self.view_width = self.window_width // 2          # 预期800
        self.view_height = self.view_width                # 强制正方形
        self.view_offset_y = self.top_panel_h             # 战斗区域起始Y

        # 世界覆盖范围
        self.view_world_width = config.get('VIEW_WORLD_WIDTH', window_width)
        self.view_world_height = config.get('VIEW_WORLD_HEIGHT', self.view_world_width)
        self.grid_spacing = config.get('GRID_SPACING', 1000)
        self.anchor_ratio = config.get('PLAYER_ANCHOR_X_RATIO', 0.5)
        self.font = font

        # 比例换算（世界坐标 -> 视图坐标）
        self.pixels_per_meter_x = self.view_width / float(self.view_world_width)
        self.pixels_per_meter_y = self.view_height / float(self.view_world_height)
        self.player_anchor_x = self.view_width * self.anchor_ratio
        self.player_anchor_y = self.view_offset_y + self.view_height * 0.5

        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_RED = (255, 150, 150)
        self.LIGHT_BLUE = (150, 150, 255)
        self.GREEN = (0, 255, 0)
        self.GRID_COLOR = (60, 60, 60)

        # 创建游戏窗口
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('中距空战游戏')

        # 视图模式
        self.view_mode = "normal"  # 可以是 "normal" 或 "spacetime"（时空图）
        
        # 时空图组件
        self._spacetime_computer = spacetime_computer
        self._spacetime_renderer = None
        self._spacetime_initialized = (spacetime_computer is not None)
        self._game_time = 0.0
        self._config = config
        
        # 如果传入了computer，预先加载renderer
        if self._spacetime_initialized:
             self._init_renderer_only()

    # ---- 坐标变换与可视范围 ----
    def _world_to_screen(self, wx, wy, cam_pos, offset_x):
        dx = wx - cam_pos[0]
        dy = wy - cam_pos[1]
        sx = offset_x + self.player_anchor_x + dx * self.pixels_per_meter_x
        sy = self.player_anchor_y + dy * self.pixels_per_meter_y
        return sx, sy

    def _in_view(self, wx, wy, cam_pos):
        half_w = self.view_world_width * 0.5
        half_h = self.view_world_height * 0.5
        dx = wx - cam_pos[0]
        dy = wy - cam_pos[1]
        return abs(dx) <= half_w and abs(dy) <= half_h

    # ---- 基础绘制 ----
    def _draw_grid(self, cam_pos, offset_x):
        half_w = self.view_world_width * 0.5
        half_h = self.view_world_height * 0.5
        x_start = math.floor((cam_pos[0] - half_w) / self.grid_spacing) * self.grid_spacing
        x_end = math.ceil((cam_pos[0] + half_w) / self.grid_spacing) * self.grid_spacing
        y_start = math.floor((cam_pos[1] - half_h) / self.grid_spacing) * self.grid_spacing
        y_end = math.ceil((cam_pos[1] + half_h) / self.grid_spacing) * self.grid_spacing

        for gx in range(int(x_start), int(x_end + self.grid_spacing), self.grid_spacing):
            sx, _ = self._world_to_screen(gx, cam_pos[1], cam_pos, offset_x)
            pygame.draw.line(self.window, self.GRID_COLOR, (sx, self.view_offset_y), (sx, self.view_offset_y + self.view_height))

        for gy in range(int(y_start), int(y_end + self.grid_spacing), self.grid_spacing):
            _, sy = self._world_to_screen(cam_pos[0], gy, cam_pos, offset_x)
            pygame.draw.line(self.window, self.GRID_COLOR, (offset_x, sy), (offset_x + self.view_width, sy))

        # 分屏边框
        pygame.draw.rect(self.window, self.WHITE, (offset_x, self.view_offset_y, self.view_width, self.view_height), 1)

    def _draw_trail(self, trail, color, cam_pos, offset_x):
        if len(trail) < 2:
            return
        points = []
        for i, (tx, ty) in enumerate(trail):
            if not self._in_view(tx, ty, cam_pos):
                continue
            px, py = self._world_to_screen(tx, ty, cam_pos, offset_x)
            points.append((px, py))
            if len(points) > 1:
                pygame.draw.line(self.window, color, points[-2], points[-1], 1)

    def _draw_aircraft_symbol(self, screen_pos, angle_deg, color, is_missile=False):
        sx, sy = screen_pos
        if is_missile:
            triangle_size = 4
            angle_rad = math.radians(angle_deg)
            point1 = (sx + math.cos(angle_rad) * triangle_size * 2,
                      sy + math.sin(angle_rad) * triangle_size * 2)
            point2 = (sx + math.cos(angle_rad + math.pi * 2 / 3) * triangle_size,
                      sy + math.sin(angle_rad + math.pi * 2 / 3) * triangle_size)
            point3 = (sx + math.cos(angle_rad + math.pi * 4 / 3) * triangle_size,
                      sy + math.sin(angle_rad + math.pi * 4 / 3) * triangle_size)
            pygame.draw.polygon(self.window, color, [point1, point2, point3])
        else:
            pygame.draw.circle(self.window, color, (int(sx), int(sy)), 5)

    def _draw_velocity_vector(self, screen_pos, angle_deg, speed, color):
        sx, sy = screen_pos
        vector_length = speed / 10
        end_x = sx + math.cos(math.radians(angle_deg)) * vector_length
        end_y = sy + math.sin(math.radians(angle_deg)) * vector_length
        pygame.draw.line(self.window, color, (sx, sy), (end_x, end_y), 2)

    def _draw_arrow_indicator(self, cam_pos, target_pos, offset_x, color):
        dx = target_pos[0] - cam_pos[0]
        dy = target_pos[1] - cam_pos[1]
        half_w = self.view_world_width * 0.5
        half_h = self.view_world_height * 0.5

        if dx == 0 and dy == 0:
            return

        # 找到射线与视口矩形的交点
        t_candidates = []
        if dx != 0:
            t_candidates.append(abs(half_w / dx))
        if dy != 0:
            t_candidates.append(abs(half_h / dy))
        t = min(t_candidates) if t_candidates else 1.0
        edge_dx = dx * t
        edge_dy = dy * t

        screen_x, screen_y = self._world_to_screen(cam_pos[0] + edge_dx, cam_pos[1] + edge_dy, cam_pos, offset_x)
        angle_deg = math.degrees(math.atan2(dy, dx))
        arrow_size = 12
        angle_rad = math.radians(angle_deg)
        back_angle1 = angle_rad + math.pi - math.pi / 6
        back_angle2 = angle_rad + math.pi + math.pi / 6

        tip = (screen_x, screen_y)
        tail1 = (screen_x + math.cos(back_angle1) * arrow_size,
                 screen_y + math.sin(back_angle1) * arrow_size)
        tail2 = (screen_x + math.cos(back_angle2) * arrow_size,
                 screen_y + math.sin(back_angle2) * arrow_size)
        pygame.draw.polygon(self.window, color, [tip, tail1, tail2], width=2)

    def _draw_entity_label(self, screen_pos, entity, color):
        label = f"{getattr(entity, 'mach', 0):.1f}Ma G{getattr(entity, 'g_load', 0):.0f}"
        text_surface = self.font.render(label, True, color)
        w, h = text_surface.get_width(), text_surface.get_height()
        scaled_surface = pygame.transform.smoothscale(text_surface, (max(1, w // 1.5), max(1, h // 1.5)))
        offset = (10, -10)
        self.window.blit(scaled_surface, (screen_pos[0] + offset[0], screen_pos[1] + offset[1]))

    # ---- 视图绘制 ----
    def draw_split_view(self, aircraft1, aircraft2, missiles):
        self.window.fill(self.BLACK)
        self._draw_single_view(aircraft1, aircraft2, missiles, offset_x=0)
        self._draw_single_view(aircraft2, aircraft1, missiles, offset_x=self.view_width)

    def _draw_single_view(self, player, enemy, missiles, offset_x):
        cam_pos = (player.x, player.y)
        self._draw_grid(cam_pos, offset_x)

        # 玩家及敌机
        player_screen = (offset_x + self.player_anchor_x, self.player_anchor_y)
        self._draw_trail(getattr(player, 'trail', []), player.color, cam_pos, offset_x)
        self._draw_aircraft_symbol(player_screen, player.angle, player.color, False)
        self._draw_velocity_vector(player_screen, player.angle, player.speed, player.color)
        self._draw_entity_label(player_screen, player, player.color)

        if enemy.alive:
            if self._in_view(enemy.x, enemy.y, cam_pos):
                enemy_screen = self._world_to_screen(enemy.x, enemy.y, cam_pos, offset_x)
                self._draw_trail(getattr(enemy, 'trail', []), enemy.color, cam_pos, offset_x)
                self._draw_aircraft_symbol(enemy_screen, enemy.angle, enemy.color, False)
                self._draw_velocity_vector(enemy_screen, enemy.angle, enemy.speed, enemy.color)
            else:
                self._draw_arrow_indicator(cam_pos, (enemy.x, enemy.y), offset_x, enemy.color)

        # 导弹
        for missile in missiles:
            if not missile.alive:
                continue
            if not self._in_view(missile.x, missile.y, cam_pos):
                continue
            missile_screen = self._world_to_screen(missile.x, missile.y, cam_pos, offset_x)
            self._draw_trail(getattr(missile, 'trail', []), missile.color, cam_pos, offset_x)
            self._draw_aircraft_symbol(missile_screen, missile.angle, missile.color, True)
            self._draw_velocity_vector(missile_screen, missile.angle, missile.speed, missile.color)
            self._draw_entity_label(missile_screen, missile, missile.color)

    def _render_columns(self, lines, start_x, start_y, color, cols=2, col_width=180, row_h=22):
        rows = (len(lines) + cols - 1) // cols
        for idx, text in enumerate(lines):
            col = idx // rows
            row = idx % rows
            x = start_x + col * col_width
            y = start_y + row * row_h
            text_surface = self.font.render(text, True, color)
            self.window.blit(text_surface, (x, y))

    def draw_ui(self, aircraft1, aircraft2, game_over=False, winner=None):
        # 获取飞机属性
        def extract(a, is_red=True):
            if isinstance(a, dict):
                speed = int(a['speed'])
                rudder = a.get('rudder', 0) * 100
                throttle = a.get('throttle', 0) * 100
                turn_rate = a.get('turn_rate', 0)
                missiles = a.get('missiles', 0)
            else:
                speed = int(a.speed)
                rudder = a.rudder * 100
                throttle = a.throttle * 100
                turn_rate = getattr(a, 'turn_rate', 0)
                missiles = a.missiles
            status = [
                ("红方状态:" if is_red else "蓝方状态:"),
                f"速度: {speed}m/s",
                f"舵量: {rudder:.1f}%",
                f"油门: {throttle:.1f}%",
                f"角速度: {turn_rate:.1f}°/s",
                f"导弹: {missiles}",
            ]
            return status

        red_status = extract(aircraft1, True)
        blue_status = extract(aircraft2, False)

        # 顶部状态栏（高度100）
        top_y = 0
        self._render_columns(red_status, start_x=10, start_y=top_y, color=self.RED, cols=2, col_width=180, row_h=22)
        self._render_columns(blue_status, start_x=self.window_width // 2 + 10, start_y=top_y, color=self.BLUE, cols=2, col_width=180, row_h=22)

        # 底部操作提示与预留区域（高度100）
        bottom_y = self.window_height - self.bottom_panel_h + 10
        
        if self.view_mode == "spacetime":
            controls_text = self.font.render(
                '红方: WASD+T, Q/E旋转, R掉头线 | 蓝方: 方向键+=, 9/0旋转, -掉头线 | V切换模式', 
                True, self.WHITE
            )
        else:
            controls_text = self.font.render(
                '红方: WASD + T | 蓝方: 方向键 + = | V切换模式', 
                True, self.WHITE
            )
        
        # 时空图状态提示
        if self.view_mode == "spacetime" and self._spacetime_renderer:
            red_turn = "开" if self._spacetime_renderer.show_turn_predictions.get(1, True) else "关"
            blue_turn = "开" if self._spacetime_renderer.show_turn_predictions.get(2, True) else "关"
            placeholder_text = self.font.render(
                f'红方掉头线: {red_turn} | 蓝方掉头线: {blue_turn}', 
                True, self.GREEN
            )
        else:
            placeholder_text = self.font.render('按钮预留区', True, self.GREEN)
        
        self.window.blit(controls_text, (self.window_width // 2 - controls_text.get_width() // 2, bottom_y))
        self.window.blit(placeholder_text, (self.window_width // 2 - placeholder_text.get_width() // 2, bottom_y + 30))

        # 视图模式指示
        view_mode_text = self.font.render(f'视图模式: {"常规" if self.view_mode == "normal" else "时空图"}',
                                         True, self.GREEN)
        self.window.blit(view_mode_text, (10, bottom_y + 60))

        # 如果游戏结束，显示获胜者
        if game_over:
            if winner == 'draw':
                winner_text = "游戏结束！平局！"
            else:
                winner_text = f"游戏结束！{'红方' if winner == 'red' else '蓝方'}获胜！"
            text = self.font.render(winner_text, True, self.WHITE)
            text_rect = text.get_rect(center=(self.window_width / 2, self.window_height / 2))
            self.window.blit(text, text_rect)

    def toggle_view_mode(self):
        """切换视图模式（常规/时空图）"""
        self.view_mode = "spacetime" if self.view_mode == "normal" else "normal"
        
        # 首次切换到时空图时初始化
        if self.view_mode == "spacetime":
            self._init_spacetime()
        
        return self.view_mode
    
    def _init_renderer_only(self):
        """仅初始化渲染器（当Computer已存在时）"""
        module = _get_spacetime_module()
        if not module.get('available', False):
            return False
            
        try:
            self._spacetime_renderer = module['SpacetimeRenderer'](
                self.view_width, 
                self.view_height,
                self._config
            )
            return True
        except Exception as e:
            print(f"时空图渲染器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_spacetime(self):
        """初始化时空图组件"""
        if self._spacetime_initialized:
            return True
        
        module = _get_spacetime_module()
        if not module.get('available', False):
            print("时空图模块不可用，请先生成导弹查找表")
            return False
        
        try:
            if self._spacetime_computer is None:
                self._spacetime_computer = module['SpacetimeComputer'](self._config)
                self._spacetime_computer.initialize()
            
            self._spacetime_renderer = module['SpacetimeRenderer'](
                self.view_width, 
                self.view_height,
                self._config
            )
            
            self._spacetime_initialized = True
            print("时空图模块初始化成功")
            return True
        except Exception as e:
            print(f"时空图初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def rotate_spacetime_camera(self, player_id: int, delta: float):
        """旋转时空图相机
        
        Args:
            player_id: 玩家ID (1 或 2)
            delta: 旋转角度（度）
        """
        if self._spacetime_renderer:
            self._spacetime_renderer.rotate_camera(player_id, delta)
    
    def toggle_turn_predictions(self, player_id: int):
        """切换玩家的全力掉头预测线显示
        
        Args:
            player_id: 玩家ID (1 或 2)
            
        Returns:
            bool: 当前显示状态
        """
        if self._spacetime_renderer:
            return self._spacetime_renderer.toggle_turn_predictions(player_id)
        return False
    
    def update_spacetime(self, aircraft1, aircraft2, missiles, dt: float):
        """更新时空图数据
        
        Args:
            aircraft1: 玩家1飞机
            aircraft2: 玩家2飞机
            missiles: 导弹列表
            dt: 时间步长
        """
        if not self._spacetime_initialized:
            return
        
        self._game_time += dt
        self._spacetime_computer.update(aircraft1, aircraft2, missiles, self._game_time)
    
    def draw_spacetime_split_view(self, aircraft1, aircraft2, missiles):
        """绘制分屏时空图视图
        
        Args:
            aircraft1: 玩家1飞机
            aircraft2: 玩家2飞机  
            missiles: 导弹列表
        """
        if not self._spacetime_initialized:
            # 如果时空图未初始化，回退到普通视图
            self.draw_split_view(aircraft1, aircraft2, missiles)
            return
        
        self.window.fill(self.BLACK)
        
        # 绘制左侧（红方）时空图
        self._spacetime_renderer.draw_spacetime_view(
            self.window, 1, aircraft1, aircraft2, missiles,
            self._spacetime_computer,
            offset_x=0, offset_y=self.view_offset_y
        )
        
        # 绘制右侧（蓝方）时空图
        self._spacetime_renderer.draw_spacetime_view(
            self.window, 2, aircraft2, aircraft1, missiles,
            self._spacetime_computer,
            offset_x=self.view_width, offset_y=self.view_offset_y
        )
    
    def get_spacetime_features(self, player_id: int, num_points: int = 10):
        """获取时空图稀疏特征（用于RL输入）
        
        Args:
            player_id: 玩家ID
            num_points: 采样点数
            
        Returns:
            np.ndarray: 特征数组
        """
        if not self._spacetime_initialized or self._spacetime_computer is None:
            import numpy as np
            return np.zeros(num_points * 4 * 3, dtype=np.float32)
        
        return self._spacetime_computer.get_sparse_features(player_id, num_points)

    def clear(self):
        """清空屏幕"""
        self.window.fill(self.BLACK)

    def update(self):
        """更新显示"""
        pygame.display.flip()
