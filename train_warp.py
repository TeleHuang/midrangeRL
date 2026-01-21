import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys
import pygame

# Import new modules
from RL.models import ActorCriticAgent
from RL.ppo_v1 import PPO, RolloutStorage
from RL.utils import KeyboardListener, ConsolePrinter
from RL.rewards import StandardReward
from RL.opponent import RuleBasedAgent
from env_warp.aero_warp import MidrangeEnvGPU, NUM_ENVS, SUB_STEPS
from visualization import Visualizer, get_system_font
import config

# --- Hyperparameters ---
LR = 2.5e-4 # Match old config
NUM_STEPS = 128  # Match old config
TOTAL_UPDATES = 100000
MINI_BATCH_SIZE = 32768
PPO_EPOCHS = 4

# Adapter for Visualization
class WarpEntityAdapter:
    def __init__(self, pos, vel, heading, active, team):
        self.x = pos[0]
        self.y = pos[1]
        self.speed = np.linalg.norm(vel)
        self.angle = np.degrees(heading)
        self.color = (0, 0, 255) if team == 0 else (255, 0, 0)
        self.alive = active > 0.5 if active is not None else True
        self.trail = [] 
        self.rudder = 0
        self.throttle = 0
        self.missiles = 1
        self.mach = self.speed / 340.0
        self.g_load = 1.0

def visualize_episode(env, agent, opponent, steps=600):
    print("\nStarting Visualization (Playback)...")
    pygame.init()
    font = get_system_font()
    vis = Visualizer(1600, 1000, config.CONFIG, font)
    
    # Get recorded frames from the last episode
    frames_data = env.get_recorded_frames() # [steps, 2, 12]
    
    clock = pygame.time.Clock()
    running = True
    
    # Playback loop
    for i in range(len(frames_data)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: # Toggle back
                    running = False
                    
        if not running:
            break
            
        # Parse frame i
        # Agent 0 data
        f0_data = frames_data[i, 0]
        # Agent 1 data
        f1_data = frames_data[i, 1]
        
        # Data layout:
        # 0: x, 1: y, 2: vx, 3: vy, 4: heading, 5: health, 6: team
        # 7: m_x, 8: m_y, 9: m_vx, 10: m_vy, 11: m_active
        
        def make_fighter_adapter(data):
            pos = data[0:2]
            vel = data[2:4]
            heading = data[4]
            active = data[5] > 0
            team = int(data[6])
            return WarpEntityAdapter(pos, vel, heading, active, team)
            
        def make_missile_adapter(data, team):
            pos = data[7:9]
            vel = data[9:11]
            active = data[11] > 0.5
            # Recalculate heading for visualizer
            heading = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.1 else 0.0
            return WarpEntityAdapter(pos, vel, heading, active, team)

        f0 = make_fighter_adapter(f0_data)
        f1 = make_fighter_adapter(f1_data)
        m0 = make_missile_adapter(f0_data, 0)
        m1 = make_missile_adapter(f1_data, 1)
        
        vis.draw_split_view(f0, f1, [m0, m1])
        vis.draw_ui(f0, f1)
        vis.update()
        
        clock.tick(60)
        
    pygame.quit()
    print("Visualization Ended.")

def main():
    print("Initializing Environment...")
    # Ensure directory exists
    os.makedirs("checkpoints", exist_ok=True)
    
    env = MidrangeEnvGPU(NUM_ENVS)
    device = torch.device("cuda")
    
    print("Initializing Components...")
    # Agent only for P1 (so input dim is same, but we only train on P1 data)
    agent = ActorCriticAgent(env.obs_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # Use larger batch size for stability with simple PPO
    ppo = PPO(agent, optimizer, mini_batch_size=MINI_BATCH_SIZE, ppo_epochs=PPO_EPOCHS)
    
    # Storage for P1 only!
    storage = RolloutStorage(NUM_STEPS, NUM_ENVS, env.obs_dim, device)
    
    reward_fn = StandardReward(device=device)
    opponent = RuleBasedAgent(env, mode='pursuit') # Simple pursuit opponent
    
    kb_listener = KeyboardListener()
    
    global_step = 0
    obs = env.reset() # [num_agents, obs_dim]
    
    print(f"Start Training: {NUM_ENVS} Envs. Agent vs RuleBased.")
    print("Controls: [P] Pause | [R] Resume | [V] Visualize | [S] Save | [Q] Quit")
    
    paused = False
    
    for update in range(1, TOTAL_UPDATES + 1):
        # Anneal LR
        frac = 1.0 - (update - 1.0) / TOTAL_UPDATES
        lrnow = frac * LR
        optimizer.param_groups[0]["lr"] = lrnow
        
        physics_start = time.perf_counter()
        storage.reset()
        
        for step in range(NUM_STEPS):
            # Input Handling
            key = kb_listener.get_key()
            if key:
                if key == b'q':
                    print("\nQuitting...")
                    kb_listener.stop()
                    return
                elif key == b'p':
                    print("\nPaused. Press R to resume.")
                    paused = True
                    while paused:
                        k = kb_listener.get_key()
                        if k == b'r':
                            paused = False
                            print("Resumed.")
                        elif k == b'v':
                             visualize_episode(env, agent, opponent)
                        elif k == b's':
                            path = f"checkpoints/ppo_manual_{global_step}.pt"
                            torch.save(agent.state_dict(), path)
                            print(f"\nSaved to {path}")
                        elif k == b'q':
                            kb_listener.stop()
                            return
                        time.sleep(0.1)
                elif key == b'v':
                    visualize_episode(env, agent, opponent)
                    obs = env.reset()
                elif key == b's':
                    path = f"checkpoints/ppo_manual_{global_step}.pt"
                    torch.save(agent.state_dict(), path)
                    print(f"\nSaved to {path}")

            # 1. Split Observation
            obs_p1 = obs[0::2]
            obs_p2 = obs[1::2]
            
            # 2. Agent Action
            with torch.no_grad():
                action_p1, logprob_p1, _, value_p1 = agent.get_action_and_value(obs_p1)
                
            # 3. Opponent Action
            action_p2 = opponent.get_action(obs_p2)
            
            # 4. Combine Actions for Env
            actions_all = torch.zeros((env.num_agents, 2), device=device)
            
            # Map P1 (0,1,2 -> -1,0,1)
            actions_all[0::2, 0] = action_p1[:, 0].float() - 1.0
            actions_all[0::2, 1] = action_p1[:, 1].float()
            
            # Map P2
            actions_all[1::2, 0] = action_p2[:, 0].float() - 1.0
            actions_all[1::2, 1] = action_p2[:, 1].float()
            
            # 5. Step
            next_obs, env_rewards, dones = env.step(actions_all)
            
            # 6. Custom Reward Calculation
            # We need to construct infos for win/loss from raw env state if possible
            # But here we trust env_rewards to contain sparse win/loss or we infer it
            # The StandardReward checks infos['winner']. 
            # Our WarpEnv currently returns dones but no complex infos dict with winner.
            # We need to hack or fix WarpEnv to return winner info, OR infer it from health.
            # Let's assume StandardReward can handle missing winner info (it checks if 'winner' in infos).
            # To get winner info, we'd need access to 'health'.
            # For now, let's use the env_rewards (which has +0.01 survival) and ADD our shaped reward.
            
            # obs_p1 is Pre-Transition P1 state
            # obs_p2 is Pre-Transition P2 state
            shaped_reward = reward_fn.compute(obs_p1, obs_p2, action_p1, dones[0::2], {})
            
            # Total Reward
            # env_rewards is [num_agents]. P1 rewards are at 0::2
            total_reward = shaped_reward # + env_rewards[0::2] # Avoid double counting if shaped includes survival
            
            # 7. Store
            storage.insert(obs_p1, action_p1, logprob_p1, total_reward, dones[0::2], value_p1.flatten())
            
            obs = next_obs
            global_step += NUM_ENVS
            
            if step % 10 == 0:
                percent = (step / NUM_STEPS) * 100
                ConsolePrinter.print_status(f"[Physics] Step {step}/{NUM_STEPS} ({percent:.1f}%) | Total Steps: {global_step}")

        # End of Rollout
        physics_end = time.perf_counter()
        sps = (NUM_STEPS * SUB_STEPS) / (physics_end - physics_start)
        
        ConsolePrinter.clear_line()
        print(f"[Physics] Done. Per-Env SPS: {sps:.1f}")
        
        # Update
        ConsolePrinter.print_status("[Train] Updating...")
        
        obs_p1_next = next_obs[0::2]
        dones_p1 = dones[0::2]
        
        with torch.no_grad():
            next_value = agent.get_value(obs_p1_next).reshape(1, -1)
            
        loss = ppo.update(storage, next_value, dones_p1)
        
        ConsolePrinter.clear_line()
        print(f"[Train] Update {update} | Reward: {storage.rewards.mean().item():.4f} | Loss: {loss:.4f} | LR: {lrnow:.2e}")
        
        # Periodic Save
        if update % 50 == 0:
            torch.save(agent.state_dict(), f"checkpoints/ppo_step_{global_step}.pt")

    kb_listener.stop()

if __name__ == "__main__":
    main()
