import torch
import numpy as np

class RewardConfig:
    def __init__(self):
        self.win_reward = 500.0
        self.loss_penalty = -500.0
        self.tick_survival_reward = 0.005
        self.orientation_weight = 0.05
        self.distance_weight = 0.001
        self.action_penalty = 0.0

class StandardReward:
    def __init__(self, config=None, device='cuda'):
        self.config = config if config else RewardConfig()
        self.device = device
        
    def compute(self, obs_p1, obs_p2, actions_p1, dones, infos):
        """
        Compute rewards for P1 based on current state and transition.
        obs shape: [num_envs, obs_dim]
        
        Indices based on aero_warp.py get_obs_kernel:
        0: x (norm)
        1: y (norm)
        2: vx (norm)
        3: vy (norm)
        4: cos(heading)
        5: sin(heading)
        6: missile_active
        7: enemy_x
        8: enemy_y
        9: enemy_vx
        10: enemy_vy
        11: enemy_cos
        12: enemy_sin
        ...
        """
        num_envs = obs_p1.shape[0]
        rewards = torch.zeros(num_envs, device=self.device)
        
        # 1. Survival Reward (Tick)
        rewards += self.config.tick_survival_reward
        
        # 2. Orientation Reward (Align velocity vector with vector to enemy)
        # P1 State
        p1_x = obs_p1[:, 0]
        p1_y = obs_p1[:, 1]
        p1_vx = obs_p1[:, 2]
        p1_vy = obs_p1[:, 3]
        
        # Enemy State
        e_x = obs_p1[:, 7]
        e_y = obs_p1[:, 8]
        
        # Vector to Enemy
        dx = e_x - p1_x
        dy = e_y - p1_y
        dist_sq = dx*dx + dy*dy
        dist = torch.sqrt(dist_sq + 1e-6)
        
        # Normalize vectors
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        p1_speed = torch.sqrt(p1_vx*p1_vx + p1_vy*p1_vy + 1e-6)
        p1_vx_norm = p1_vx / p1_speed
        p1_vy_norm = p1_vy / p1_speed
        
        # Dot product (Cosine Similarity)
        # alignment = (v . d)
        alignment = p1_vx_norm * dx_norm + p1_vy_norm * dy_norm
        
        rewards += alignment * self.config.orientation_weight
        
        # 3. Distance Reward (Encourage closing in, but not too close? For now just close in)
        # Normalized distance (0 to 2 range approx)
        # Reward = (1.0 - dist) * weight
        rewards += (1.0 - dist) * self.config.distance_weight
        
        # 4. Win/Loss (Sparse)
        # infos['winner'] should contain 1 (P1 win) or 2 (P2 win)
        # Note: infos is from WarpEnv step
        if 'winner' in infos:
            winner = infos['winner'] # Tensor [num_envs]
            rewards = torch.where(winner == 1, torch.tensor(self.config.win_reward, device=self.device), rewards)
            rewards = torch.where(winner == 2, torch.tensor(self.config.loss_penalty, device=self.device), rewards)
            
        # 5. Draw/Timeout
        # Handled by done without winner? 
        # Usually standard PPO handles done by bootstrapping or zero value.
        
        return rewards
