import torch

class RuleBasedAgent:
    """
    A simple rule-based opponent for Phase 1/2 curriculum.
    """
    def __init__(self, env, mode='pursuit'):
        self.env = env
        self.mode = mode
        self.device = "cuda"
        
    def get_action(self, obs):
        """
        obs: [num_envs, obs_dim]
        Returns: [num_envs, 2] (rudder, fire)
        """
        num_envs = obs.shape[0]
        
        # Parse Obs (Assuming same structure as P1)
        # 0:x, 1:y, 2:vx, 3:vy ... 7:e_x, 8:e_y
        
        p_x = obs[:, 0]
        p_y = obs[:, 1]
        p_vx = obs[:, 2]
        p_vy = obs[:, 3]
        
        e_x = obs[:, 7]
        e_y = obs[:, 8]
        
        # Vector to Target
        dx = e_x - p_x
        dy = e_y - p_y
        
        # Current Heading
        speed = torch.sqrt(p_vx*p_vx + p_vy*p_vy + 1e-6)
        heading_x = p_vx / speed
        heading_y = p_vy / speed
        
        # Desired Heading (Pursuit)
        dist = torch.sqrt(dx*dx + dy*dy + 1e-6)
        desired_x = dx / dist
        desired_y = dy / dist
        
        # Cross Product (2D) to determine turn direction
        # cp = hx * dy - hy * dx
        cp = heading_x * desired_y - heading_y * desired_x
        
        # Rudder Logic
        # if cp > threshold -> Turn Left (-1)
        # if cp < -threshold -> Turn Right (1)
        # Note: Check coordinate system. 
        # Usually positive cross product means target is to the left.
        # Warp kernel: rudder=-1 (Left), rudder=1 (Right)
        
        rudder = torch.zeros(num_envs, device=self.device)
        threshold = 0.05
        
        # If target is left (cp > 0), we want rudder = -1? 
        # Let's check aero_warp.py:
        # pn = (-ny, nx) -> Left normal
        # centripetal = v^2 * cl * rudder
        # accel = pn * centripetal
        # If rudder > 0, accel is along pn (Left). 
        # So rudder=1 is LEFT turn in typical math, but comment said:
        # "54-> pn = wp.vec2(-n[1], n[0]) # 垂直方向 (左转为正)"
        # "57-> rudder = actions[tid] - 1.0 # 0,1,2 -> -1,0,1"
        # "68-> accel = ... + pn * centripetal"
        # So positive rudder -> positive pn -> Left Turn.
        
        # So: cp > 0 (Target Left) -> Rudder = 1.0
        # cp < 0 (Target Right) -> Rudder = -1.0
        
        rudder = torch.where(cp > threshold, torch.tensor(1.0, device=self.device), rudder)
        rudder = torch.where(cp < -threshold, torch.tensor(-1.0, device=self.device), rudder)
        
        # Discrete Action Mapping
        # -1 -> 0
        # 0 -> 1
        # 1 -> 2
        rudder_idx = torch.ones(num_envs, dtype=torch.float32, device=self.device) # Default 1 (Straight)
        rudder_idx = torch.where(rudder < -0.5, torch.tensor(0.0, device=self.device), rudder_idx)
        rudder_idx = torch.where(rudder > 0.5, torch.tensor(2.0, device=self.device), rudder_idx)
        
        # Fire Logic
        # Fire if aligned and close enough?
        # For now, passive (no fire) to be easy target in Phase 1
        fire_idx = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        
        if self.mode == 'aggressive':
            # Fire if aligned
            dot = heading_x * desired_x + heading_y * desired_y
            fire_idx = torch.where((dot > 0.95) & (dist < 0.2), torch.tensor(1.0, device=self.device), fire_idx)
            
        return torch.stack([rudder_idx, fire_idx], dim=1)
