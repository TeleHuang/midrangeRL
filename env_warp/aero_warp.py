import warp as wp
import torch
import numpy as np
import time

wp.init()

# --- Constants ---
# Game Config
NUM_ENVS = 10000
DT = 0.1  # Physics time step
SUB_STEPS = 10  # Physics steps per decision step
MAX_STEPS = 600 # Decision steps per episode (6000 physics steps)

# Physics Constants (from config.py)
G = 9.8
FIGHTER_RMIN = 1000.0
FIGHTER_VT = 400.0
FIGHTER_THRUST_ACC = 1.5 * G
FIGHTER_LD = 5.0
# Derived Fighter Constants
FIGHTER_CL_MAX = 1.0 / FIGHTER_RMIN
FIGHTER_CD0 = G / (FIGHTER_VT ** 2)
FIGHTER_K = 1.0 / (4.0 * FIGHTER_CD0 * (FIGHTER_LD ** 2) + 1e-6)

# Missile Constants
MISSILE_RMIN = 1000.0
MISSILE_VT = 400.0
MISSILE_THRUST_ACC = 15.0 * G
MISSILE_TIME = 10.0
MISSILE_LD = 2.0
MISSILE_GAIN = 200.0
MISSILE_RANGE = 100.0 # Hit radius
# Derived Missile Constants
MISSILE_CL_MAX = 1.0 / MISSILE_RMIN
MISSILE_CD0 = G / (MISSILE_VT ** 2)
MISSILE_K = 1.0 / (4.0 * MISSILE_CD0 * (MISSILE_LD ** 2) + 1e-6)

# Map Limits
MAP_SIZE = 50000.0

@wp.struct
class Fighter:
    pos: wp.vec2
    vel: wp.vec2
    heading: float # radians
    health: float # 1.0 alive, <= 0 dead
    team: int # 0 or 1

@wp.struct
class Missile:
    pos: wp.vec2
    vel: wp.vec2
    active: int # 0: ready, 1: flying
    timer: float
    prev_los: float # For PN guidance derivative

@wp.kernel
def init_kernel(
    fighters: wp.array(dtype=Fighter),
    missiles: wp.array(dtype=Missile),
    seed: int
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    
    # Initialize Fighter
    # Random position in [-5000, 5000]
    fighters[tid].pos = wp.vec2(
        wp.randf(state) * 10000.0 - 5000.0,
        wp.randf(state) * 10000.0 - 5000.0
    )
    # Random velocity
    speed = 200.0 + wp.randf(state) * 100.0
    angle = wp.randf(state) * 6.283185
    fighters[tid].vel = wp.vec2(wp.cos(angle), wp.sin(angle)) * speed
    fighters[tid].heading = angle
    fighters[tid].health = 1.0
    fighters[tid].team = tid % 2
    
    # Initialize Missile (Inactive)
    missiles[tid].active = 0
    missiles[tid].timer = 0.0
    missiles[tid].pos = fighters[tid].pos
    missiles[tid].vel = fighters[tid].vel

@wp.func
def compute_aero_forces(
    vel: wp.vec2,
    rudder: float,
    thrust: float,
    cl_max: float,
    cd0: float,
    k: float,
    dt: float
):
    # Returns (accel, new_vel)
    v_sq = wp.dot(vel, vel)
    v_mag = wp.length(vel)
    
    if v_mag < 0.1:
        return wp.vec2(0.0, 0.0)
        
    n = vel / v_mag
    pn = wp.vec2(-n[1], n[0]) # Perpendicular
    
    # Aerodynamics
    cl = rudder * cl_max
    cd = cd0 + k * cl * cl
    
    drag = cd * v_sq
    lift_acc = v_sq * cl # Centripetal acceleration
    
    acc_tangent = thrust - drag
    acc_normal = lift_acc
    
    accel = n * acc_tangent + pn * acc_normal
    return accel

@wp.kernel
def physics_step_kernel(
    fighters: wp.array(dtype=Fighter),
    missiles: wp.array(dtype=Missile),
    actions: wp.array(dtype=wp.vec2), # [rudder, fire]
    rewards: wp.array(dtype=float),
    dones: wp.array(dtype=int),
    dt: float
):
    tid = wp.tid()
    
    # Identify self and enemy
    # tid is absolute index. env_id = tid // 2.
    # enemy_idx is adjacent.
    env_id = tid // 2
    is_p2 = tid % 2
    enemy_idx = env_id * 2 + (1 - is_p2)
    
    f = fighters[tid]
    m = missiles[tid]
    f_enemy = fighters[enemy_idx]
    
    # --- 0. Check Done (Reset Logic) ---
    # If self or enemy is dead, we might need to reset. 
    # But for parallel envs, we usually let the episode finish or auto-reset.
    # Here we just check health.
    if f.health <= 0.0:
        # Respawn logic (Simple hard reset for now)
        # Random respawn
        state = wp.rand_init(tid, tid)
        f.pos = wp.vec2(wp.randf(state)*10000.0-5000.0, wp.randf(state)*10000.0-5000.0)
        angle = wp.randf(state) * 6.28
        f.vel = wp.vec2(wp.cos(angle), wp.sin(angle)) * 300.0
        f.health = 1.0
        # Reset missile
        m.active = 0
        fighters[tid] = f
        missiles[tid] = m
        dones[tid] = 1 # Mark done for PPO
        return # Skip this step
    else:
        dones[tid] = 0

    # --- 1. Fighter Control & Physics ---
    action = actions[tid]
    rudder_cmd = action[0] # -1 to 1
    fire_cmd = action[1]   # > 0.5 to fire
    
    # Apply Physics to Fighter
    accel = compute_aero_forces(
        f.vel, rudder_cmd, FIGHTER_THRUST_ACC, 
        FIGHTER_CL_MAX, FIGHTER_CD0, FIGHTER_K, dt
    )
    f.vel = f.vel + accel * dt
    f.pos = f.pos + f.vel * dt
    f.heading = wp.atan2(f.vel[1], f.vel[0])
    
    # --- 2. Missile Logic ---
    # Fire Logic
    if m.active == 0:
        # Carry the missile
        m.pos = f.pos
        m.vel = f.vel
        
        if fire_cmd > 0.5:
            # Launch!
            m.active = 1
            m.timer = MISSILE_TIME
            m.prev_los = 0.0 # Reset
            # Initial boost? Or just inherit velocity?
            # Inherit velocity
    else:
        # Missile is flying
        m.timer = m.timer - dt
        
        # Check target (Enemy)
        target_pos = f_enemy.pos
        rel_pos = target_pos - m.pos
        dist = wp.length(rel_pos)
        
        # Guidance (Proportional Navigation)
        # Calculate LOS angle
        los_angle = wp.degrees(wp.atan2(rel_pos[1], rel_pos[0]))
        
        # LOS Rate
        # Handle wrap around -180/180
        # We need a stable way. 
        # For first step, prev_los might be 0. 
        # We can store the actual angle.
        
        # Simple finite difference
        # Need to handle the wrap around logic:
        diff = los_angle - m.prev_los
        if diff > 180.0:
            diff = diff - 360.0
        if diff < -180.0:
            diff = diff + 360.0
            
        los_rate = diff / dt
        
        # Update prev_los
        m.prev_los = los_angle
        
        # Calculate Rudder
        # Gain * Rate. 
        # Note: rate is in deg/s. 
        # If rate is 1 deg/s, rudder = 200/180 = 1.11 -> clamped to 1.
        missile_rudder = (MISSILE_GAIN * los_rate) / 180.0
        missile_rudder = wp.clamp(missile_rudder, -1.0, 1.0)
        
        # Thrust logic
        thrust = 0.0
        if m.timer > 0.0:
            thrust = MISSILE_THRUST_ACC
            
        # Apply Physics to Missile
        m_accel = compute_aero_forces(
            m.vel, missile_rudder, thrust,
            MISSILE_CL_MAX, MISSILE_CD0, MISSILE_K, dt
        )
        m.vel = m.vel + m_accel * dt
        m.pos = m.pos + m.vel * dt
        
        # Collision Check (Missile vs Enemy)
        # Only if missile is active and enemy is alive
        if f_enemy.health > 0.0:
            if dist < MISSILE_RANGE:
                # Hit!
                f_enemy.health = 0.0
                m.active = 0 # Missile destroyed
                # Write back enemy dead immediately? 
                # Yes, fighters[enemy_idx] needs update.
                # But we can't write to other threads safely in Warp unless atomic.
                # Here, tid and enemy_idx are distinct. 
                # Actually, race condition possible if both shoot each other same frame?
                # It's fine, worst case double kill.
                fighters[enemy_idx].health = 0.0
                
        # Timeout / Miss
        # If timer < -10.0 (coast for 10s after burnout), kill missile
        if m.timer < -10.0:
             m.active = 0

    # --- 3. Compute Rewards ---
    reward = 0.0
    
    # Reward for survival
    reward = reward + 0.01
    
    # Reward for hitting enemy (Check if enemy died this step? Hard to track without prev state)
    # We can check if we just killed them.
    # But better: sparse reward + dense shaping.
    
    # Distance shaping (Encourage getting closer)
    # r_pos = f_enemy.pos - f.pos
    # dist = wp.length(r_pos)
    # reward += 0.0001 * (5000.0 - dist) # Small reward for being close?
    
    # Angle shaping (Encourage pointing at enemy)
    # ...
    
    # Store data
    fighters[tid] = f
    missiles[tid] = m
    rewards[tid] = reward

@wp.kernel
def get_obs_kernel(
    fighters: wp.array(dtype=Fighter),
    missiles: wp.array(dtype=Missile),
    obs: wp.array(dtype=float, ndim=2) # [num_agents, obs_dim]
):
    tid = wp.tid()
    env_id = tid // 2
    is_p2 = tid % 2
    enemy_idx = env_id * 2 + (1 - is_p2)
    
    f = fighters[tid]
    f_enemy = fighters[enemy_idx]
    m = missiles[tid]
    m_enemy = missiles[enemy_idx]
    
    # Normalize inputs
    # Pos: / 5000.0
    # Vel: / 400.0
    
    # Self State (7)
    obs[tid, 0] = f.pos[0] / 5000.0
    obs[tid, 1] = f.pos[1] / 5000.0
    obs[tid, 2] = f.vel[0] / 400.0
    obs[tid, 3] = f.vel[1] / 400.0
    obs[tid, 4] = wp.cos(f.heading)
    obs[tid, 5] = wp.sin(f.heading)
    obs[tid, 6] = float(m.active)
    
    # Enemy State (6)
    obs[tid, 7] = f_enemy.pos[0] / 5000.0
    obs[tid, 8] = f_enemy.pos[1] / 5000.0
    obs[tid, 9] = f_enemy.vel[0] / 400.0
    obs[tid, 10] = f_enemy.vel[1] / 400.0
    obs[tid, 11] = wp.cos(f_enemy.heading)
    obs[tid, 12] = wp.sin(f_enemy.heading)
    
    # Missile State (Self) (4)
    obs[tid, 13] = m.pos[0] / 5000.0
    obs[tid, 14] = m.pos[1] / 5000.0
    obs[tid, 15] = m.vel[0] / 400.0
    obs[tid, 16] = m.vel[1] / 400.0
    
    # Missile State (Enemy) (4)
    obs[tid, 17] = m_enemy.pos[0] / 5000.0
    obs[tid, 18] = m_enemy.pos[1] / 5000.0
    obs[tid, 19] = m_enemy.vel[0] / 400.0
    obs[tid, 20] = m_enemy.vel[1] / 400.0

@wp.kernel
def record_viz_kernel(
    fighters: wp.array(dtype=Fighter),
    missiles: wp.array(dtype=Missile),
    viz_buffer: wp.array(dtype=float, ndim=3), # [frames, agents, features]
    frame_idx: int
):
    # Record env 0 (agents 0 and 1)
    tid = wp.tid() # 0 or 1
    
    f = fighters[tid]
    m = missiles[tid]
    
    # Features: 
    # 0: x, 1: y, 2: vx, 3: vy, 4: heading, 5: health, 6: team
    # 7: m_x, 8: m_y, 9: m_vx, 10: m_vy, 11: m_active
    
    viz_buffer[frame_idx, tid, 0] = f.pos[0]
    viz_buffer[frame_idx, tid, 1] = f.pos[1]
    viz_buffer[frame_idx, tid, 2] = f.vel[0]
    viz_buffer[frame_idx, tid, 3] = f.vel[1]
    viz_buffer[frame_idx, tid, 4] = f.heading
    viz_buffer[frame_idx, tid, 5] = f.health
    viz_buffer[frame_idx, tid, 6] = float(f.team)
    
    viz_buffer[frame_idx, tid, 7] = m.pos[0]
    viz_buffer[frame_idx, tid, 8] = m.pos[1]
    viz_buffer[frame_idx, tid, 9] = m.vel[0]
    viz_buffer[frame_idx, tid, 10] = m.vel[1]
    viz_buffer[frame_idx, tid, 11] = float(m.active)

class MidrangeEnvGPU:
    def __init__(self, num_envs=NUM_ENVS):
        self.num_envs = num_envs
        self.num_agents = num_envs * 2
        self.obs_dim = 21
        
        # Allocation
        self.fighters = wp.empty(self.num_agents, dtype=Fighter)
        self.missiles = wp.empty(self.num_agents, dtype=Missile)
        
        self.actions = wp.zeros(self.num_agents, dtype=wp.vec2)
        self.rewards = wp.zeros(self.num_agents, dtype=float)
        self.dones = wp.zeros(self.num_agents, dtype=int)
        self.obs = wp.zeros((self.num_agents, self.obs_dim), dtype=float)
        
        # Viz Buffer
        self.max_steps = MAX_STEPS
        self.viz_buffer = wp.zeros((self.max_steps, 2, 12), dtype=float)
        self.step_count = 0
        
        # Torch Views
        self.obs_torch = wp.to_torch(self.obs)
        self.rewards_torch = wp.to_torch(self.rewards)
        self.dones_torch = wp.to_torch(self.dones)
        
        # Initialize
        wp.launch(init_kernel, dim=self.num_agents, inputs=[self.fighters, self.missiles, int(time.time())])
        
        # Capture Graph
        self.use_graph = True
        if self.use_graph:
            self.graph = self._create_graph()
            
    def _create_graph(self):
        wp.capture_begin()
        for _ in range(SUB_STEPS):
            wp.launch(
                physics_step_kernel,
                dim=self.num_agents,
                inputs=[self.fighters, self.missiles, self.actions, self.rewards, self.dones, DT]
            )
        return wp.capture_end()
        
    def step(self, actions_torch):
        # actions_torch: [num_agents, 2] (rudder, fire)
        # We assume actions are already processed to be compatible (e.g. one-hot or raw values)
        # But here we expect raw float values: rudder in [-1, 1], fire in [0, 1]
        
        # Copy actions
        wp.copy(self.actions, wp.from_torch(actions_torch))
        
        # Run Physics
        if self.use_graph:
            wp.capture_launch(self.graph)
        else:
            for _ in range(SUB_STEPS):
                wp.launch(
                    physics_step_kernel,
                    dim=self.num_agents,
                    inputs=[self.fighters, self.missiles, self.actions, self.rewards, self.dones, DT]
                )
        
        # Record Viz
        wp.launch(record_viz_kernel, dim=2, inputs=[self.fighters, self.missiles, self.viz_buffer, self.step_count])
        self.step_count = (self.step_count + 1) % self.max_steps
        
        # Get Obs
        wp.launch(get_obs_kernel, dim=self.num_agents, inputs=[self.fighters, self.missiles, self.obs])
        
        return self.obs_torch, self.rewards_torch, self.dones_torch

    def reset(self):
        wp.launch(init_kernel, dim=self.num_agents, inputs=[self.fighters, self.missiles, int(time.time())])
        wp.launch(get_obs_kernel, dim=self.num_agents, inputs=[self.fighters, self.missiles, self.obs])
        self.step_count = 0
        return self.obs_torch
        
    def get_recorded_frames(self):
        # Return numpy array of recorded frames
        # Need to handle circular buffer if needed, but here we just return the whole buffer
        # Assume we visualize from start
        return self.viz_buffer.numpy()
