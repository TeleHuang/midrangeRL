# -*- coding: utf-8 -*-

'''
游戏背景：
游戏场景是现实中三代机超视距空战的俯视图2D简化版本,
战场是无穷大二维平面,交战的红蓝双方各为1架战斗机,
初始位置在边长50km的正方形对角线的10%与90%处,初始速度300,朝向正方形战场中心,油门为最大,舵量为0。
每架战斗机携带6发中距弹(挂在战斗机上时只是一个数字,不作为实体计算,也不考虑其质量),
胜利条件是在保存自己的实力情况下消灭全部敌机,双方导弹均耗尽时未分出胜负算平局。
'''

import pygame
import math
import os

# 导入环境组件
from visualization import Visualizer, get_system_font
from env_numpy.numpy_env import NumpyEnv
from env_warp.warp_env import WarpEnv
from config import CONFIG
from controllers import PlayerController, HumanController, ModelController

# 初始化Pygame
pygame.init()

# 从配置文件读取常量
WINDOW_WIDTH = CONFIG.get('WINDOW_WIDTH', 800)
WINDOW_HEIGHT = CONFIG.get('WINDOW_HEIGHT', 800)

class Game:
    """游戏主类，支持多种环境后端和控制器"""
    
    # 时空图相机旋转速度（度/秒）
    CAMERA_ROTATE_SPEED = 90.0
    
    def __init__(self, config=None, env_backend='numpy', 
                 red_controller=None, blue_controller=None):
        """初始化游戏
        
        Args:
            config: 配置字典
            env_backend: 环境后端 ('numpy' 或 'tensor')
            red_controller: 红方控制器，如为 None 则使用默认 HumanController
            blue_controller: 蓝方控制器，如为 None 则使用默认 HumanController
        """
        # 使用传入的配置或默认配置
        self.config = config if config is not None else CONFIG
        self.env_backend = env_backend
        
        # 初始化字体
        self.font = get_system_font()
        
        # 初始化时空图组件 (Warp后端)
        self.spacetime_computer = None
        # TensorSpacetimeComputer 已被移除，Warp版时空图尚未实现
        # if self.env_backend == 'warp':
        #     ...

        # 初始化可视化系统
        self.visualizer = Visualizer(WINDOW_WIDTH, WINDOW_HEIGHT, self.config, self.font, 
                                   spacetime_computer=self.spacetime_computer)
        
        # 初始化环境
        self.env = self._create_env()
        
        # 初始化控制器
        if red_controller is None:
            # 默认红方使用 WASD 键位
            red_keys = {
                'left': pygame.K_a,
                'right': pygame.K_d,
                'up': pygame.K_w,
                'down': pygame.K_s,
                'fire': pygame.K_t
            }
            red_controller = HumanController(env_backend, red_keys)
        
        if blue_controller is None:
            # 默认蓝方使用箭头键
            blue_keys = {
                'left': pygame.K_LEFT,
                'right': pygame.K_RIGHT,
                'up': pygame.K_UP,
                'down': pygame.K_DOWN,
                'fire': pygame.K_EQUALS
            }
            blue_controller = HumanController(env_backend, blue_keys)
        
        self.red_controller = red_controller
        self.blue_controller = blue_controller
        
        # 获取初始渲染状态
        render_state = self.env.get_render_state()
        self.aircraft1 = render_state['aircraft1']
        self.aircraft2 = render_state['aircraft2']
        self.missiles = render_state['missiles']
        self.game_over = render_state['game_over']
        self.winner = render_state['winner']
        
        # tensor后端的轨迹维护（为每个实体保存轨迹）
        from collections import deque
        self._trails = {
            'aircraft1': deque(maxlen=100),
            'aircraft2': deque(maxlen=100),
            'missiles': {}  # missile_slot_idx -> deque
        }
        self._trail_update_counter = 0
    
    def _create_env(self):
        """创建环境实例"""
        if self.env_backend == 'numpy':
            return NumpyEnv(self.config)
        elif self.env_backend == 'warp':
            return WarpEnv(self.config, num_envs=1, device='cuda')
        else:
            raise ValueError(f"Unknown env_backend: {self.env_backend}")
    
    def _build_observation(self, aircraft, enemy, missiles):
        """从渲染状态构建观察字典
        
        Args:
            aircraft: 自己的飞机对象
            enemy: 敌方飞机对象
            missiles: 导弹列表
        
        Returns:
            观察字典
        """
        import math
        
        # 提取基础信息
        obs = {
            'x': aircraft.x,
            'y': aircraft.y,
            'angle': aircraft.angle,
            'speed': aircraft.speed,
            'missiles': aircraft.missiles,
            'alive': aircraft.alive,
        }
        
        # 计算相对量
        dx = enemy.x - aircraft.x
        dy = enemy.y - aircraft.y
        obs['enemy_distance'] = math.sqrt(dx**2 + dy**2)
        obs['enemy_relative_angle'] = math.degrees(math.atan2(dy, dx))
        obs['enemy_speed'] = enemy.speed
        obs['enemy_alive'] = enemy.alive
        
        # 添加敌方位置（用于 CrankAgent 计算目标角度）
        obs['enemy_x'] = enemy.x
        obs['enemy_y'] = enemy.y
        
        return obs
    
    
    def update_render_state(self):
        """从环境获取渲染状态"""
        render_state = self.env.get_render_state()
        self.aircraft1 = render_state['aircraft1']
        self.aircraft2 = render_state['aircraft2']
        self.missiles = render_state['missiles']
        self.game_over = render_state['game_over']
        self.winner = render_state['winner']
        
        # 为 warp 后端维护轨迹
        if self.env_backend == 'warp':
            self._trail_update_counter += 1
            if self._trail_update_counter >= 5:  # 每5帧更新一次轨迹
                self._trail_update_counter = 0
                
                # 更新飞机轨迹
                if hasattr(self.aircraft1, 'alive') and self.aircraft1.alive:
                    self._trails['aircraft1'].append((self.aircraft1.x, self.aircraft1.y))
                if hasattr(self.aircraft2, 'alive') and self.aircraft2.alive:
                    self._trails['aircraft2'].append((self.aircraft2.x, self.aircraft2.y))
                
                # 更新导弹轨迹
                active_missile_slots = set()
                for missile in self.missiles:
                    slot_idx = getattr(missile, 'slot_idx', id(missile))
                    active_missile_slots.add(slot_idx)
                    if slot_idx not in self._trails['missiles']:
                        from collections import deque
                        self._trails['missiles'][slot_idx] = deque(maxlen=100)
                    self._trails['missiles'][slot_idx].append((missile.x, missile.y))
                
                # 清理已不存在的导弹轨迹
                dead_slots = set(self._trails['missiles'].keys()) - active_missile_slots
                for dead_slot in dead_slots:
                    del self._trails['missiles'][dead_slot]
            
            # 将维护的轨迹赋值给实体
            self.aircraft1.trail = self._trails['aircraft1']
            self.aircraft2.trail = self._trails['aircraft2']
            for missile in self.missiles:
                slot_idx = getattr(missile, 'slot_idx', id(missile))
                if slot_idx in self._trails['missiles']:
                    missile.trail = self._trails['missiles'][slot_idx]
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # 计算时间步长
            dt = clock.tick(60) / 1000.0
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_v:
                        self.visualizer.toggle_view_mode()
                    # 时空图掉头线开关
                    elif event.key == pygame.K_r:
                        # 红方切换掉头预测线
                        self.visualizer.toggle_turn_predictions(1)
                    elif event.key == pygame.K_MINUS:
                        # 蓝方切换掉头预测线
                        self.visualizer.toggle_turn_predictions(2)
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 时空图相机控制（在时空图模式下持续处理）
            if self.visualizer.view_mode == "spacetime":
                # 红方: Q/E 旋转相机
                if keys[pygame.K_q]:
                    self.visualizer.rotate_spacetime_camera(1, -self.CAMERA_ROTATE_SPEED * dt)
                if keys[pygame.K_e]:
                    self.visualizer.rotate_spacetime_camera(1, self.CAMERA_ROTATE_SPEED * dt)
                
                # 蓝方: 9/0 旋转相机
                if keys[pygame.K_9]:
                    self.visualizer.rotate_spacetime_camera(2, -self.CAMERA_ROTATE_SPEED * dt)
                if keys[pygame.K_0]:
                    self.visualizer.rotate_spacetime_camera(2, self.CAMERA_ROTATE_SPEED * dt)
            
            # 构建观察
            red_obs = self._build_observation(self.aircraft1, self.aircraft2, self.missiles)
            blue_obs = self._build_observation(self.aircraft2, self.aircraft1, self.missiles)
            
            # 从控制器获取动作
            red_action = self.red_controller.get_action(red_obs, keys, dt)
            blue_action = self.blue_controller.get_action(blue_obs, keys, dt)
            
            # 合并为环境所需的动作字典
            actions = {
                'p1_rudder': red_action['rudder'],
                'p1_throttle_delta': red_action['throttle_delta'],
                'p1_fire': red_action['fire'],
                'p2_rudder': blue_action['rudder'],
                'p2_throttle_delta': blue_action['throttle_delta'],
                'p2_fire': blue_action['fire'],
            }
            
            # 执行环境步骤
            if not self.game_over:
                result = self.env.step(actions, dt)
                # 处理不同后端的返回格式
                if self.env_backend == 'numpy':
                    # NumpyEnv 返回 (game_over, winner)
                    self.game_over, self.winner = result
                else:
                    # TensorEnv 返回 (observations, rewards, dones, infos)
                    _, _, dones, infos = result
                    self.game_over = bool(dones[0].item()) if hasattr(dones[0], 'item') else bool(dones[0])
                    if self.game_over:
                        winner_val = int(infos['winner'][0].item()) if hasattr(infos['winner'][0], 'item') else int(infos['winner'][0])
                        if winner_val == 1:
                            self.winner = 'red'
                        elif winner_val == 2:
                            self.winner = 'blue'
                        else:
                            self.winner = 'draw'
            
            # 更新渲染状态
            self.update_render_state()
            
            # 更新时空图数据（无论什么模式都更新，以保持数据连续性）
            if self.spacetime_computer:
                # Tensor backend
                self.visualizer._game_time += dt
                step = int(self.visualizer._game_time * 60) # Approx step
                self.spacetime_computer.update(self.env.states, step)
            elif self.visualizer._spacetime_initialized:
                self.visualizer.update_spacetime(self.aircraft1, self.aircraft2, self.missiles, dt)
            
            # 绘制游戏画面
            if self.visualizer.view_mode == "spacetime":
                self.visualizer.draw_spacetime_split_view(self.aircraft1, self.aircraft2, self.missiles)
            else:
                self.visualizer.draw_split_view(self.aircraft1, self.aircraft2, self.missiles)
            self.visualizer.draw_ui(self.aircraft1, self.aircraft2, self.game_over, self.winner)
            self.visualizer.update()
        
        pygame.quit()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='中距空战游戏')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'warp'],
                        help='环境后端: numpy(默认) 或 warp')
    parser.add_argument('--red_type', type=str, default='human', choices=['human', 'ai'],
                        help='红方控制类型: human(默认) 或 ai')
    parser.add_argument('--blue_type', type=str, default='human', choices=['human', 'ai'],
                        help='蓝方控制类型: human(默认) 或 ai')
    parser.add_argument('--red_agent', type=str, default='phase2', choices=['phase2', 'phase3'],
                        help='红方 Agent 类型 (当 red_type=ai 时生效)')
    parser.add_argument('--blue_agent', type=str, default='phase2', choices=['phase2', 'phase3'],
                        help='蓝方 Agent 类型 (当 blue_type=ai 时生效)')
    args = parser.parse_args()
    
    # Helper to load PPO agent
    def load_ppo_agent(device='cpu'):
        import os
        import glob
        from agents.learned.ppo_agent_discrete import DiscretePPOAgent
        
        agent = DiscretePPOAgent(device=device, num_envs=1)
        
        # Find latest checkpoint
        checkpoints = glob.glob('checkpoints/ppo_step_*.pt')
        if not checkpoints:
            print("Warning: No PPO checkpoints found. Using random init.")
            return agent
            
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Loading PPO agent from {latest_ckpt}")
        agent.load(latest_ckpt)
        return agent

    # Agent 类型映射
    AGENT_REGISTRY = {
        # 'crank': lambda: __import__('agents.rule_based.rule_agent', fromlist=['CrankAgent']).CrankAgent(device='cpu')
        # CrankAgent 已被删除，暂无默认 rule_based agent
        # 如果需要，可以添加 curriculum_agents 中的 agent
        'phase2': lambda: __import__('agents.rule_based.curriculum_agents', fromlist=['Phase2Opponent']).Phase2Opponent(device='cpu', num_envs=1),
        'phase3': lambda: __import__('agents.rule_based.curriculum_agents', fromlist=['Phase3Opponent']).Phase3Opponent(device='cpu', num_envs=1),
        'ppo': lambda: load_ppo_agent(device='cpu')
    }
    
    # 创建控制器
    def create_controller(player_type, agent_type, env_backend, key_mapping):
        """创建控制器工厂函数
        
        Args:
            player_type: 'human' 或 'ai'
            agent_type: Agent 类型字符串 ('crank', ...)
            env_backend: 环境后端
            key_mapping: 键位映射
        
        Returns:
            PlayerController 实例
        """
        if player_type == 'human':
            return HumanController(env_backend, key_mapping)
        elif player_type == 'ai':
            if agent_type not in AGENT_REGISTRY:
                raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
            agent = AGENT_REGISTRY[agent_type]()
            return ModelController(agent, env_backend)
        else:
            raise ValueError(f"Unknown player type: {player_type}")
    
    # 红方键位 (WASD + T)
    red_keys = {
        'left': pygame.K_a,
        'right': pygame.K_d,
        'up': pygame.K_w,
        'down': pygame.K_s,
        'fire': pygame.K_t
    }
    
    # 蓝方键位 (箭头键 + =)
    blue_keys = {
        'left': pygame.K_LEFT,
        'right': pygame.K_RIGHT,
        'up': pygame.K_UP,
        'down': pygame.K_DOWN,
        'fire': pygame.K_EQUALS
    }
    
    # 创建控制器
    red_controller = create_controller(args.red_type, args.red_agent, args.backend, red_keys)
    blue_controller = create_controller(args.blue_type, args.blue_agent, args.backend, blue_keys)
    
    # 创建并运行游戏
    print(f"游戏配置: 后端={args.backend}, 红方={args.red_type}, 蓝方={args.blue_type}")
    if args.red_type == 'ai':
        print(f"  红方 Agent: {args.red_agent}")
    if args.blue_type == 'ai':
        print(f"  蓝方 Agent: {args.blue_agent}")
    
    game = Game(CONFIG, env_backend=args.backend, 
                red_controller=red_controller, blue_controller=blue_controller)
    game.run()
