# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
import torch
import numpy as np
import msvcrt
import pygame
from typing import Dict, Any, Optional, Type

from env_warp.warp_env import WarpEnv
from config import CONFIG
from visualization import Visualizer, get_system_font

from agents.base_agent import BaseAgent
from agents.learned.ppo_agent_discrete import DiscretePPOAgent
from agents.rule_based.curriculum_agents import Phase2Opponent, Phase3Opponent
from rewards.base_reward import BaseReward
from rewards import ZeroReward
from rewards.curriculum_rewards import Phase1Reward, Phase2Reward, Phase3Reward, StandardReward

# ============================================================================
# Adapters
# ============================================================================
def adapter_warp_to_gym_obs(warp_obs_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert WarpEnv tensor observation to Gym-style dict observation"""
    # warp_obs indices:
    # 0:x, 1:y, 2:angle, 3:speed, 4:missiles, 5:alive, 6:e_dist, 7:e_rel_ang, 8:e_speed, 9:e_alive
    
    return {
        'x': warp_obs_tensor[:, 0],
        'y': warp_obs_tensor[:, 1],
        'angle': warp_obs_tensor[:, 2],
        'speed': warp_obs_tensor[:, 3],
        'missiles': warp_obs_tensor[:, 4],
        'alive': warp_obs_tensor[:, 5],
        'enemy_distance': warp_obs_tensor[:, 6],
        'enemy_relative_angle': warp_obs_tensor[:, 7],
        'enemy_speed': warp_obs_tensor[:, 8],
        'enemy_alive': warp_obs_tensor[:, 9],
        # 'stg': ... # Not supported yet in WarpEnv
    }

# ============================================================================
# Registries
# ============================================================================
AGENT_REGISTRY = {
    'ppo': DiscretePPOAgent,
    'phase2_opponent': Phase2Opponent,
    'phase3_opponent': Phase3Opponent,
    'placeholder': None,
}

REWARD_REGISTRY = {
    'zero': ZeroReward,
    'phase1': Phase1Reward,
    'phase2': Phase2Reward,
    'phase3': Phase3Reward,
    'standard': StandardReward,
}

class PlaceholderAgent(BaseAgent):#这个Agent名字得改一下，现在这个名字不准确
    def act(self, obs):
        num_envs = self.num_envs
        return {
            'rudder': torch.zeros(num_envs, device=self.device),
            'fire': torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            'throttle': torch.ones(num_envs, device=self.device)
        }

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        pass

AGENT_REGISTRY['placeholder'] = PlaceholderAgent

# ============================================================================
# Helpers
# ============================================================================
def create_agent(agent_type: str, device: str, num_envs: int, args: Optional[argparse.Namespace] = None):
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_type}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    
    if agent_type == 'ppo':
        # DiscretePPOAgent specific params
        agent = DiscretePPOAgent(
            device=device,
            num_envs=num_envs,
            num_steps=args.n_steps if args else 128,
            learning_rate=args.learning_rate if args else 2.5e-4,
            minibatch_size=args.batch_size if args else 32768,
            update_epochs=args.n_epochs if args else 4
        )
    else:
        agent = agent_class(device=device, num_envs=num_envs)
        
    return agent

def create_reward(reward_type: str, device: str) -> BaseReward:
    if reward_type not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward: {reward_type}")
    return REWARD_REGISTRY[reward_type](device=device)

def run_visualization_episode(env, agent, opponent=None, game_seconds=60, speed_x=6):
    """Run a visualization playback from recorded frames"""
    print(f"Retrieving recorded frames from GPU buffer...")
    
    if hasattr(env, 'get_recorded_frames'):
        frames = env.get_recorded_frames()
    else:
        print("Error: Environment does not support frame recording.")
        return

    if not frames:
        print("Warning: No frames recorded yet.")
        return

    # Playback
    # FPS = 1 / ( (viz_interval * dt) / speed_x )
    # viz_interval = 36, dt = 1/60 => 0.6s game time per frame
    # At 6x speed => 0.1s real time per frame => 10 FPS
    
    playback_fps = 10
    total_frames = len(frames)
    
    if total_frames < 100:
        print(f"Warning: Buffer partial ({total_frames}/100 frames). Visualization will be shorter.")
        print("Note: Buffer clears on restart. Run for at least 3600 steps to fill.")
    
    real_time_duration = total_frames / playback_fps
    print(f"Playing back {total_frames} frames at {speed_x}x speed (~{real_time_duration:.1f}s real time)...")
    
    pygame.init()
    font = get_system_font()
    vis = Visualizer(800, 800, CONFIG, font)
    clock = pygame.time.Clock()
    
    running = True
    idx = 0
    while running and idx < len(frames):
        dt = clock.tick(playback_fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        state = frames[idx]
        vis.draw_split_view(state['aircraft1'], state['aircraft2'], state['missiles'])
        vis.draw_ui(state['aircraft1'], state['aircraft2'], state['game_over'], state['winner'])
        vis.update()
        idx += 1
        
    pygame.quit()
    print("Visualization finished.")

def handle_pause(env, agent, opponent, global_step):
    print("\n[PAUSED] Training paused.")
    print("Options: [r]esume, [v]isualize, [s]ave, [q]uit")
    
    while True:
        if msvcrt.kbhit():
            try:
                cmd = msvcrt.getch().lower()
                if cmd == b'r':
                    print("Resuming...")
                    break
                elif cmd == b'v':
                    run_visualization_episode(env, agent, opponent)
                    print("\n[PAUSED] Options: [r]esume, [v]isualize, [s]ave, [q]uit")
                elif cmd == b's':
                    os.makedirs("checkpoints", exist_ok=True)
                    path = f"checkpoints/ppo_step_{global_step}_manual.pt"
                    agent.save(path)
                    print(f"Saved: {path}")
                    print("[PAUSED] Options: [r]esume, [v]isualize, [s]ave, [q]uit")
                elif cmd == b'q':
                    sys.exit(0)
            except Exception:
                pass
        time.sleep(0.1)

# ============================================================================
# Training Loop
# ============================================================================
def training_loop(
    env: WarpEnv,
    agent: BaseAgent,
    opponent: BaseAgent,
    reward_fn: BaseReward,
    args: argparse.Namespace,
    steps_limit: Optional[int] = None
):
    device = args.device
    max_steps = steps_limit if steps_limit else args.max_steps
    
    print(f"\nStart Training Loop: {max_steps} steps")
    print(f"  Agent: {type(agent).__name__}")
    print(f"  Opponent: {type(opponent).__name__}")
    print(f"  Reward: {type(reward_fn).__name__}")
    
    obs_dict = env.reset()
    obs_p1 = obs_dict['p1']
    obs_p2 = obs_dict['p2']
    
    # Opponent reset
    if hasattr(opponent, 'reset'):
        opponent.reset()
    reward_fn.reset()
    
    total_steps = 0
    start_time = time.time()
    
    is_ppo = isinstance(agent, DiscretePPOAgent)
    
    try:
        while total_steps < max_steps:
            # Check pause
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'p':
                    handle_pause(env, agent, opponent, total_steps)
                    obs_dict = env.get_observations()
                    obs_p1 = obs_dict['p1']
                    obs_p2 = obs_dict['p2']
            
            # 1. P1 Action
            if is_ppo:
                action_p1, logprob_p1, value_p1 = agent.get_action(obs_p1)
            else:
                # Placeholder or other
                obs_p1_dict = adapter_warp_to_gym_obs(obs_p1)
                act_dict = agent.act(obs_p1_dict)
                # Convert to discrete
                r = act_dict['rudder']
                r_idx = torch.ones_like(r, dtype=torch.long)
                r_idx = torch.where(r < -0.3, torch.tensor(0, device=device), r_idx)
                r_idx = torch.where(r > 0.3, torch.tensor(2, device=device), r_idx)
                f_idx = act_dict['fire'].long()
                action_p1 = torch.stack([r_idx, f_idx], dim=-1)
            
            # 2. P2 Action
            obs_p2_dict = adapter_warp_to_gym_obs(obs_p2)
            if hasattr(opponent, 'get_action'): # If opponent is PPO (Self-Play)
                action_p2, _, _ = opponent.get_action(obs_p2)
            else:
                act_p2_dict = opponent.act(obs_p2_dict)
                r2 = act_p2_dict['rudder']
                r2_idx = torch.ones_like(r2, dtype=torch.long)
                r2_idx = torch.where(r2 < -0.3, torch.tensor(0, device=device), r2_idx)
                r2_idx = torch.where(r2 > 0.3, torch.tensor(2, device=device), r2_idx)
                f2_idx = act_p2_dict['fire'].long()
                action_p2 = torch.stack([r2_idx, f2_idx], dim=-1)
                
            # 3. Step
            next_obs_dict, env_rewards, dones, infos = env.step({
                'p1_action': action_p1,
                'p2_action': action_p2
            })
            
            next_obs_p1 = next_obs_dict['p1']
            next_obs_p2 = next_obs_dict['p2']
            
            # 4. Custom Reward
            # We construct obs dicts for reward function
            obs_before_dict = {'p1': adapter_warp_to_gym_obs(obs_p1), 'p2': adapter_warp_to_gym_obs(obs_p2)}
            obs_after_dict = {'p1': adapter_warp_to_gym_obs(next_obs_p1), 'p2': adapter_warp_to_gym_obs(next_obs_p2)}
            
            # Action dict for reward
            # Extract rudder/fire from discrete action
            rudder_val = torch.zeros(env.num_envs, device=device)
            # 0:-1, 1:0, 2:1
            rudder_val = torch.where(action_p1[:, 0] == 0, torch.tensor(-1.0, device=device), rudder_val)
            rudder_val = torch.where(action_p1[:, 0] == 2, torch.tensor(1.0, device=device), rudder_val)
            
            action_p1_dict = {
                'rudder': rudder_val,
                'fire': action_p1[:, 1],
                'throttle': torch.ones(env.num_envs, device=device)
            }
            
            custom_reward = reward_fn.compute(
                obs_before=obs_before_dict,
                obs_after=obs_after_dict,
                action=action_p1_dict,
                done=dones,
                info=infos
            )
            
            # Total reward
            # Note: Env rewards are win/loss (+500/-500).
            # RewardFn also computes win/loss.
            # We should avoid double counting.
            # If we use custom reward, we probably want to use IT exclusively.
            total_reward = custom_reward
            
            # 5. Store & Update (PPO)
            if is_ppo:
                agent.store(obs_p1, action_p1, logprob_p1, value_p1, total_reward, dones)
                
                if agent.step >= agent.num_steps:
                    elapsed = time.time() - start_time
                    sps = total_steps / (elapsed + 1e-6)
                    print(f"Step {total_steps}: SPS={sps:.0f} | Updating...")
                    
                    v_loss, pg_loss, ent_loss, approx_kl = agent.update(obs_p1, dones)
                    print(f"  Loss: v={v_loss:.4f} p={pg_loss:.4f} ent={ent_loss:.4f} kl={approx_kl:.4f}")
            
            # 6. Handle Done
            if dones.any():
                # Reset opponent state
                if hasattr(opponent, 'reset'):
                    opponent.reset(env_mask=dones)
                
                # Env already reset internally for done envs, returning new obs
                reset_obs = env.reset(env_mask=dones) # Re-fetch to be safe/consistent? 
                # Actually env.step returns new obs for done envs (reset happened inside? No, WarpEnv logic check)
                # WarpEnv.step -> _compute_rewards_torch returns dones.
                # It does NOT auto-reset inside step unless we added it?
                # Check WarpEnv.step:
                # return obs, rewards, dones, infos
                # It does NOT reset.
                
                # So we MUST reset here.
                # Unlike train_warp.py original which reset inside main loop.
                reset_obs_dict = env.reset(env_mask=dones)
                obs_p1 = reset_obs_dict['p1']
                obs_p2 = reset_obs_dict['p2']
            else:
                obs_p1 = next_obs_p1
                obs_p2 = next_obs_p2
            
            total_steps += env.num_envs
            
            # Periodic Save
            if total_steps % 1000000 == 0: # Save every 1M steps approx
                 if is_ppo:
                     agent.save(f"checkpoints/ppo_step_{total_steps}.pt")
    
    except KeyboardInterrupt:
        print("Interrupted")

def run_curriculum(env: WarpEnv, agent: BaseAgent, args: argparse.Namespace):
    phases = [
        {
            'name': 'Phase 1: Intercept & Save Ammo',
            'scenario': 'phase1',
            'opponent_type': 'placeholder',
            'reward_type': 'phase1',
            'steps': 50000000, # More steps for Warp (faster)
            'desc': 'Target spawns randomly 40km away'
        },
        {
            'name': 'Phase 2: Rmax Turn Back',
            'scenario': 'standard',
            'opponent_type': 'phase2_opponent',
            'reward_type': 'phase2',
            'steps': 100000000,
            'desc': 'Simple script opponent'
        },
        {
            'name': 'Phase 3: Snake & Counter-Attack',
            'scenario': 'standard',
            'opponent_type': 'phase3_opponent',
            'reward_type': 'phase3',
            'steps': 200000000,
            'desc': 'Advanced script opponent'
        },
        {
            'name': 'Phase 4: Self-Play',
            'scenario': 'standard',
            'opponent_type': 'self',
            'reward_type': 'standard',
            'steps': 500000000,
            'desc': 'Self-play'
        }
    ]
    
    for i, phase in enumerate(phases):
        print(f"\n\n{'#'*80}")
        print(f"Starting {phase['name']}")
        print(f"Description: {phase['desc']}")
        print(f"{'#'*80}\n")
        
        # Setup Scenario
        env.scenario = phase['scenario']
        
        # Setup Opponent
        if phase['opponent_type'] == 'self':
            prev_ckpt = f"checkpoints/phase_{i}_final.pt"
            opponent = create_agent(args.agent, args.device, args.num_envs, args)
            if os.path.exists(prev_ckpt):
                opponent.load(prev_ckpt)
                print(f"Loaded self-opponent from {prev_ckpt}")
            else:
                print("Warning: No previous checkpoint for self-play")
        else:
            opponent = create_agent(phase['opponent_type'], args.device, args.num_envs, args)
            
        # Setup Reward
        reward_fn = create_reward(phase['reward_type'], args.device)
        
        # Run
        training_loop(env, agent, opponent, reward_fn, args, steps_limit=phase['steps'])
        
        # Save
        if hasattr(agent, 'save'):
            path = f"checkpoints/phase_{i+1}_final.pt"
            agent.save(path)
            print(f"Saved phase checkpoint: {path}")

# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'cold_start'])
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--opponent', type=str, default='placeholder')
    parser.add_argument('--reward', type=str, default='standard')
    parser.add_argument('--num-envs', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-steps', type=int, default=100000000)
    
    # PPO params
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Config
    # Update global config with args if needed
    
    print(f"Initializing WarpEnv with {args.num_envs} envs on {args.device}...")
    env = WarpEnv(CONFIG, num_envs=args.num_envs, device=args.device)
    
    print(f"Initializing Agent {args.agent}...")
    agent = create_agent(args.agent, args.device, args.num_envs, args)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)
        
    os.makedirs("checkpoints", exist_ok=True)
    
    if args.mode == 'cold_start':
        run_curriculum(env, agent, args)
    else:
        opponent = create_agent(args.opponent, args.device, args.num_envs, args)
        reward_fn = create_reward(args.reward, args.device)
        training_loop(env, agent, opponent, reward_fn, args)

if __name__ == '__main__':
    main()
