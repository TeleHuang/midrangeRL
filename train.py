# -*- coding: utf-8 -*-
"""
RL 训练入口脚本

功能：
- 从命令行参数选择 reward 类、agent、对手、并行环境数量、时间加速倍率
- 初始化 env_gym
- 在循环中调用 agent.act()、env.step()、reward_fn()

"""

import argparse
import sys
import torch
import pygame  # Added for visualization
from typing import Dict, Any, Optional, Type

# 环境
from env_gym import TensorEnv
from config import CONFIG
from visualization import Visualizer, get_system_font  # Added for visualization

# 时空图
try:
    from spacetime.tensor_spacetime import TensorSpacetimeComputer
except ImportError:
    TensorSpacetimeComputer = None

# Agent 和 Reward
from agents.base_agent import BaseAgent
from agents.learned import PPOAgent
from agents.rule_based.rule_agent import CrankAgent
from rewards.base_reward import BaseReward
from rewards import ZeroReward


# ============================================================================
# Agent 注册表
# ============================================================================
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    'ppo': PPOAgent,              # PPO 智能体（被训练）
    'rule_based': CrankAgent,     # 规则对手
    'placeholder': None,          # 占位（稍后注册）
}


# ============================================================================
# Reward 注册表
# ============================================================================
REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {
    'zero': ZeroReward,
    # 'default': DefaultReward,  # TODO: 稍后实现
}


# ============================================================================
# 占位 Agent（用于骨架测试）
# ============================================================================
class PlaceholderAgent(BaseAgent):
    """占位 Agent，返回随机动作（用于骨架测试）"""
    
    def act(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """返回随机动作"""
        num_envs = observation['x'].shape[0]
        return {
            'rudder': torch.zeros(num_envs, device=self.device),
            'throttle': torch.ones(num_envs, device=self.device),
            'fire': torch.zeros(num_envs, dtype=torch.bool, device=self.device),
        }
    
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """占位 agent 无需重置"""
        pass


# 注册占位 agent
AGENT_REGISTRY['placeholder'] = PlaceholderAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='中距空战 RL 训练脚本（骨架）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Agent 配置
    parser.add_argument(
        '--agent', '-a',
        type=str,
        default='placeholder',
        choices=list(AGENT_REGISTRY.keys()),
        help='被训练的 agent 类型'
    )
    parser.add_argument(
        '--opponent', '-o',
        type=str,
        default='placeholder',
        choices=list(AGENT_REGISTRY.keys()),
        help='对手 agent 类型'
    )
    
    # Reward 配置
    parser.add_argument(
        '--reward', '-r',
        type=str,
        default='zero',
        choices=list(REWARD_REGISTRY.keys()),
        help='奖励函数类型'
    )
    
    # 环境配置
    parser.add_argument(
        '--num-envs', '-n',
        type=int,
        default=32,
        help='并行环境数量'
    )
    parser.add_argument(
        '--time-scale', '-t',
        type=float,
        default=1.0,
        help='时间加速倍率（dt 乘数）'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备'
    )
    
    # 训练配置
    parser.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='最大训练步数'
    )
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=3e-4,
        help='学习率'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='每次更新收集的步数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='mini-batch 大小'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=10,
        help='每次更新的 epoch 数'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='折扣因子'
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='GAE 参数'
    )
    parser.add_argument(
        '--clip-epsilon',
        type=float,
        default=0.2,
        help='PPO 裁剪系数'
    )
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='价值损失权重'
    )
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='熵正则化权重'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10000,
        help='保存模型的步数间隔'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='日志输出的步数间隔'
    )
    parser.add_argument(
        '--episode-max-steps',
        type=int,
        default=3600,  # 60 秒 @ 60 FPS
        help='单局最大步数'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='开启实时可视化（显示第0号环境）'
    )

    return parser.parse_args()


def create_agent(agent_type: str, device: str, num_envs: int, args: Optional[argparse.Namespace] = None) -> BaseAgent:
    """创建 agent 实例"""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"未知的 agent 类型: {agent_type}，可用: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    
    # PPO 智能体需要额外参数
    if agent_type == 'ppo' and args is not None:
        agent = agent_class(
            device=device,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
        )
    else:
        agent = agent_class(device=device)
    
    agent.num_envs = num_envs
    return agent


def create_reward(reward_type: str, device: str) -> BaseReward:
    """创建 reward 实例"""
    if reward_type not in REWARD_REGISTRY:
        raise ValueError(f"未知的 reward 类型: {reward_type}，可用: {list(REWARD_REGISTRY.keys())}")
    
    reward_class = REWARD_REGISTRY[reward_type]
    return reward_class(device=device)


def training_loop(
    env: TensorEnv,
    agent: BaseAgent,
    opponent: BaseAgent,
    reward_fn: BaseReward,
    args: argparse.Namespace
) -> None:
    """主训练循环
    
    Args:
        env: 环境实例
        agent: 被训练的 agent（控制 P1）
        opponent: 对手 agent（控制 P2）
        reward_fn: 自定义奖励函数
        args: 命令行参数
    """
    device = args.device
    dt = (1.0 / 60.0) * args.time_scale  # 基础 dt 为 1/60 秒
    
    # 统计变量
    total_steps = 0
    episode_count = 0
    
    # 用于记录 episode 统计
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # 判断是否是 PPO 智能体
    is_ppo_agent = hasattr(agent, 'act_with_value') and hasattr(agent, 'update')
    
    # PPO 特定：经验收集计数器
    if is_ppo_agent:
        agent._ensure_buffer_initialized()
        steps_since_update = 0
    
    print(f"\n{'='*60}")
    print("开始训练循环")
    print(f"  设备: {device}")
    print(f"  并行环境数: {env.num_envs}")
    print(f"  时间加速: {args.time_scale}x")
    print(f"  Agent: {agent.get_name()}")
    print(f"  Opponent: {opponent.get_name()}")
    print(f"  Reward: {reward_fn.get_name()}")
    if is_ppo_agent:
        print(f"  PPO n_steps: {args.n_steps}")
        print(f"  PPO batch_size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    # 可视化初始化
    visualizer = None
    spacetime_computer = None
    if args.render:
        pygame.init()
        # 使用系统字体
        font = get_system_font()
        # 获取窗口大小配置
        window_width = CONFIG.get('WINDOW_WIDTH', 800)
        window_height = CONFIG.get('WINDOW_HEIGHT', 800)
        
        # 初始化 TensorSpacetimeComputer
        if TensorSpacetimeComputer:
            spacetime_computer = TensorSpacetimeComputer(CONFIG, num_envs=env.num_envs, device=args.device)
            spacetime_computer.initialize()
            
        visualizer = Visualizer(window_width, window_height, CONFIG, font, spacetime_computer=spacetime_computer)
        print("可视化已开启")

    # 初始化环境
    obs = env.reset()
    obs_p1 = obs['p1']
    obs_p2 = obs['p2']
    
    # 重置 agent 状态
    agent.reset()
    opponent.reset()
    reward_fn.reset()
    
    # 训练指标统计
    episode_returns = []
    episode_lengths = []
    recent_policy_loss = 0.0
    recent_value_loss = 0.0
    recent_entropy = 0.0
    
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    while total_steps < args.max_steps:
        # ================================================================
        # 1. Agent 决策
        # ================================================================
        if is_ppo_agent:
            # PPO: 需要收集 value 和 log_prob
            action_p1_dict, value_p1, log_prob_p1 = agent.act_with_value(obs_p1)
        else:
            # 非 PPO: 只需要动作
            action_p1_dict = agent.act(obs_p1)
        
        # 对手 agent 控制 P2
        action_p2 = opponent.act(obs_p2)
        
        # ================================================================
        # 2. 构造完整动作并执行 step
        # ================================================================
        full_action = {
            'p1_rudder': action_p1_dict['rudder'],
            'p1_throttle': action_p1_dict['throttle'],
            'p1_fire': action_p1_dict['fire'],
            'p2_rudder': action_p2['rudder'],
            'p2_throttle': action_p2['throttle'],
            'p2_fire': action_p2['fire'],
        }
        
        # 保存 step 前的观察
        obs_before_p1 = {k: v.clone() for k, v in obs_p1.items()}
        
        # 执行 step
        next_obs, env_rewards, dones, infos = env.step(full_action, dt=dt)
        next_obs_p1 = next_obs['p1']
        next_obs_p2 = next_obs['p2']
        
        # ================================================================
        # 可视化处理 (显示环境0)
        # ================================================================
        if visualizer:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if args.render:
                        pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        visualizer.toggle_view_mode()
                    elif event.key == pygame.K_r:
                        visualizer.toggle_turn_predictions(1)
                    elif event.key == pygame.K_MINUS:
                        visualizer.toggle_turn_predictions(2)
            
            # 处理持续按键（相机控制）
            keys = pygame.key.get_pressed()
            if visualizer.view_mode == "spacetime":
                # 红方: Q/E 旋转相机
                if keys[pygame.K_q]:
                    visualizer.rotate_spacetime_camera(1, -90.0 * dt)
                if keys[pygame.K_e]:
                    visualizer.rotate_spacetime_camera(1, 90.0 * dt)
                
                # 蓝方: 9/0 旋转相机
                if keys[pygame.K_9]:
                    visualizer.rotate_spacetime_camera(2, -90.0 * dt)
                if keys[pygame.K_0]:
                    visualizer.rotate_spacetime_camera(2, 90.0 * dt)

            # 获取渲染状态（环境0）
            render_state = env.get_render_state(0)
            
            # 更新时空图
            if spacetime_computer:
                 spacetime_computer.update(env.states, total_steps)
            elif visualizer._spacetime_initialized:
                 visualizer.update_spacetime(
                     render_state['aircraft1'], 
                     render_state['aircraft2'], 
                     render_state['missiles'], 
                     dt
                 )
            
            # 绘制
            if visualizer.view_mode == "spacetime":
                visualizer.draw_spacetime_split_view(
                    render_state['aircraft1'], 
                    render_state['aircraft2'], 
                    render_state['missiles']
                )
            else:
                visualizer.draw_split_view(
                    render_state['aircraft1'], 
                    render_state['aircraft2'], 
                    render_state['missiles']
                )
            
            visualizer.draw_ui(
                render_state['aircraft1'], 
                render_state['aircraft2'], 
                render_state['game_over'], 
                render_state['winner']
            )
            visualizer.update()

        # ================================================================
        # 3. 计算自定义奖励
        # ================================================================
        custom_reward = reward_fn.compute(
            obs_before=obs_before_p1,
            obs_after=next_obs_p1,
            action=action_p1_dict,
            done=dones,
            info=infos
        )
        
        # 合并环境奖励和自定义奖励
        total_reward_p1 = env_rewards['p1'] + custom_reward
        
        # ================================================================
        # 4. PPO 经验收集
        # ================================================================
        if is_ppo_agent:
            # 将观察转换为张量
            obs_tensor = agent._obs_dict_to_tensor(obs_p1)
            agent.rollout_buffer.add(
                obs=obs_tensor,
                action_rudder=action_p1_dict['rudder'],
                action_fire=action_p1_dict['fire'],
                log_prob=log_prob_p1,
                reward=total_reward_p1,
                value=value_p1,
                done=dones
            )
            steps_since_update += 1
            
            # 达到更新步数
            if steps_since_update >= args.n_steps:
                # 计算最后一步的 value
                with torch.no_grad():
                    next_obs_tensor = agent._obs_dict_to_tensor(next_obs_p1)
                    last_values = agent.critic(next_obs_tensor)
                
                # 计算 GAE 和 returns
                agent.rollout_buffer.compute_returns_and_advantages(
                    last_values=last_values,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda
                )
                
                # 执行 PPO 更新
                loss_dict = agent.update()
                
                # 记录损失
                recent_policy_loss = loss_dict['policy_loss']
                recent_value_loss = loss_dict['value_loss']
                recent_entropy = loss_dict['entropy']
                
                steps_since_update = 0
        
        # ================================================================
        # 5. 处理 episode 结束
        # ================================================================
        if dones.any():
            done_indices = torch.nonzero(dones, as_tuple=True)[0]
            
            # 统计胜负
            winner = infos['winner']
            p1_wins += (winner[done_indices] == 1).sum().item()
            p2_wins += (winner[done_indices] == 2).sum().item()
            draws += (winner[done_indices] == 0).sum().item()
            episode_count += len(done_indices)
            
            if args.verbose:
                for idx in done_indices:
                    w = winner[idx].item()
                    result = 'P1胜' if w == 1 else ('P2胜' if w == 2 else '平局')
                    print(f"  [Episode {episode_count}] 环境 {idx.item()}: {result}")
            
            # 重置完成的环境
            env.reset(env_mask=dones)
            
            # 重置 agent 状态
            agent.reset(env_mask=dones)
            opponent.reset(env_mask=dones)
        
        # ================================================================
        # 6. 更新状态
        # ================================================================
        obs_p1 = next_obs_p1
        obs_p2 = next_obs_p2
        total_steps += 1
        
        # ================================================================
        # 7. 日志输出
        # ================================================================
        if total_steps % args.log_interval == 0:
            win_rate = p1_wins / max(episode_count, 1) * 100
            log_msg = f"[Step {total_steps:>6}] Episodes: {episode_count}, P1胜率: {win_rate:.1f}%"
            
            if is_ppo_agent:
                log_msg += f" | Loss: P={recent_policy_loss:.3f} V={recent_value_loss:.3f} E={recent_entropy:.3f}"
            
            print(log_msg)
        
        # ================================================================
        # 8. 模型保存
        # ================================================================
        if is_ppo_agent and total_steps % args.save_interval == 0 and total_steps > 0:
            save_path = f'checkpoints/ppo_agent_step_{total_steps}.pt'
            agent.save(save_path)
            print(f"  → 模型已保存: {save_path}")
    
    # 训练结束统计
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"  总步数: {total_steps}")
    print(f"  总局数: {episode_count}")
    if episode_count > 0:
        print(f"  P1 胜率: {p1_wins/episode_count*100:.1f}%")
        print(f"  P2 胜率: {p2_wins/episode_count*100:.1f}%")
        print(f"  平局率: {draws/episode_count*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n{'='*60}")
    print("中距空战 RL 训练脚本（骨架）")
    print(f"{'='*60}")
    print(f"配置:")
    print(f"  Agent: {args.agent}")
    print(f"  Opponent: {args.opponent}")
    print(f"  Reward: {args.reward}")
    print(f"  并行环境数: {args.num_envs}")
    print(f"  时间加速: {args.time_scale}x")
    print(f"  设备: {args.device}")
    print(f"  最大步数: {args.max_steps}")
    
    # 创建环境
    config = CONFIG.copy()
    env = TensorEnv(config, num_envs=args.num_envs, device=args.device)
    
    # 创建 agent
    agent = create_agent(args.agent, args.device, args.num_envs, args)
    opponent = create_agent(args.opponent, args.device, args.num_envs, args)
    
    # 创建 reward 函数
    reward_fn = create_reward(args.reward, args.device)
    
    try:
        # 运行训练循环
        training_loop(env, agent, opponent, reward_fn, args)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    finally:
        env.close()
        if args.render:
            pygame.quit()
        print("环境已关闭")


if __name__ == '__main__':
    main()
