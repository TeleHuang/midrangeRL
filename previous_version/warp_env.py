# -*- coding: utf-8 -*-
import warp as wp
import torch
import math
import numpy as np
import time
from typing import Dict, Optional, Tuple, Any

from .warp_utils import init_warp, get_warp_device, from_torch, to_torch
from .base_env import BaseEnv

# ============================================================================
# Warp Kernels
# ============================================================================

@wp.struct
class GameConfig:
    dt: float
    battlefield_size: float
    
    # Fighter
    fighter_v_term: float
    fighter_cl_max: float
    fighter_max_thrust: float
    fighter_ld_ratio: float
    
    # Missile
    missile_v_term: float
    missile_cl_max: float
    missile_thrust: float
    missile_ld_ratio: float
    missile_engine_duration: float
    missile_launch_offset: float
    
    # Game
    hit_radius_sq: float
    self_destruct_speed_sq: float
    guidance_gain: float
    
    # Indices
    p1_idx: int
    p2_idx: int
    first_missile_idx: int

@wp.kernel
def physics_step_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    vx: wp.array(dtype=float, ndim=2),
    vy: wp.array(dtype=float, ndim=2),
    speed: wp.array(dtype=float, ndim=2),
    angle: wp.array(dtype=float, ndim=2),
    rudder: wp.array(dtype=float, ndim=2),
    throttle: wp.array(dtype=float, ndim=2),
    engine_time: wp.array(dtype=float, ndim=2),
    n_load: wp.array(dtype=float, ndim=2),
    turn_rate: wp.array(dtype=float, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2), # 0 or 1
    is_active: wp.array(dtype=int, ndim=2),  # 0 or 1
    config: GameConfig
):
    env_id, ent_id = wp.tid()
    
    if is_active[env_id, ent_id] == 0:
        return

    # Constants
    g = 9.8
    epsilon = 1e-7
    
    # Load state
    _vx = vx[env_id, ent_id]
    _vy = vy[env_id, ent_id]
    _rudder = rudder[env_id, ent_id]
    _throttle = throttle[env_id, ent_id]
    _is_missile = is_missile[env_id, ent_id]
    _engine_time = engine_time[env_id, ent_id]
    
    # Params selection
    terminal_velocity = config.fighter_v_term
    cl_max = config.fighter_cl_max
    max_thrust = config.fighter_max_thrust
    ld_ratio = config.fighter_ld_ratio
    
    if _is_missile == 1:
        terminal_velocity = config.missile_v_term
        cl_max = config.missile_cl_max
        max_thrust = config.missile_thrust
        ld_ratio = config.missile_ld_ratio
        
        # Update engine time
        _engine_time = wp.max(0.0, _engine_time - config.dt)
        engine_time[env_id, ent_id] = _engine_time
    
    # Thrust
    thrust_accel = 0.0
    if _is_missile == 1:
        if _engine_time > epsilon:
            thrust_accel = max_thrust
    else:
        thrust_accel = _throttle * max_thrust
        
    # Vector calcs
    v_sq = _vx*_vx + _vy*_vy
    speed_val = wp.sqrt(v_sq + epsilon)
    
    nx = _vx / speed_val
    ny = _vy / speed_val
    pnx = -ny
    pny = nx
    
    # Aero
    cd0 = g / (terminal_velocity*terminal_velocity + epsilon)
    k = 1.0 / (4.0 * cd0 * ld_ratio*ld_ratio + epsilon)
    cl_sq = (_rudder * cl_max) * (_rudder * cl_max) # wp.abs not needed for square
    cd = cd0 + k * cl_sq
    
    drag_accel = cd * v_sq
    
    parallel_accel = thrust_accel - drag_accel
    centripetal_accel = v_sq * cl_max * _rudder
    
    ax = nx * parallel_accel + pnx * centripetal_accel
    ay = ny * parallel_accel + pny * centripetal_accel
    
    # Integration
    vx_new = _vx + ax * config.dt
    vy_new = _vy + ay * config.dt
    x_new = x[env_id, ent_id] + vx_new * config.dt
    y_new = y[env_id, ent_id] + vy_new * config.dt
    
    # Store back
    vx[env_id, ent_id] = vx_new
    vy[env_id, ent_id] = vy_new
    x[env_id, ent_id] = x_new
    y[env_id, ent_id] = y_new
    
    # Aux
    speed_new = wp.sqrt(vx_new*vx_new + vy_new*vy_new + epsilon)
    angle_rad = wp.atan2(vy_new, vx_new)
    angle_deg = wp.degrees(angle_rad)
    # Normalize angle 0-360
    angle_deg = angle_deg % 360.0
    if angle_deg < 0.0:
        angle_deg += 360.0
        
    speed[env_id, ent_id] = speed_new
    angle[env_id, ent_id] = angle_deg
    
    # N load & turn rate
    actual_n = wp.abs(v_sq * cl_max * _rudder) / g
    turn_rate_rad = speed_new * cl_max * _rudder
    
    n_load[env_id, ent_id] = actual_n
    turn_rate[env_id, ent_id] = wp.degrees(turn_rate_rad)

@wp.kernel
def guidance_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    rudder: wp.array(dtype=float, ndim=2),
    prev_los_angle: wp.array(dtype=float, ndim=2),
    target_idx: wp.array(dtype=int, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2), # Need to check if target is alive
    config: GameConfig
):
    env_id, ent_id = wp.tid()
    
    if is_active[env_id, ent_id] == 0 or is_missile[env_id, ent_id] == 0:
        return
        
    tgt_idx = target_idx[env_id, ent_id]
    if tgt_idx < 0:
        return
        
    # Check target alive
    # Assuming target is in the same env
    if is_active[env_id, tgt_idx] == 0:
        return
        
    # Guidance logic
    mx = x[env_id, ent_id]
    my = y[env_id, ent_id]
    tx = x[env_id, tgt_idx]
    ty = y[env_id, tgt_idx]
    
    dx = tx - mx
    dy = ty - my
    
    cur_los = wp.atan2(dy, dx)
    
    # Angle diff
    prev_los = prev_los_angle[env_id, ent_id]
    diff = cur_los - prev_los
    
    # Wrap pi
    pi = 3.14159265359
    diff = (diff + pi) % (2.0 * pi) - pi
    
    los_rate = diff / (config.dt + 1e-7)
    
    cmd = config.guidance_gain * los_rate
    cmd = wp.clamp(cmd, -1.0, 1.0)
    
    rudder[env_id, ent_id] = cmd
    prev_los_angle[env_id, ent_id] = cur_los

@wp.kernel
def event_check_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    vx: wp.array(dtype=float, ndim=2),
    vy: wp.array(dtype=float, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2),
    target_idx: wp.array(dtype=int, ndim=2),
    # Outputs
    hit_events: wp.array(dtype=int, ndim=2), # [num_envs, max_entities] - 1 if entity got hit
    config: GameConfig
):
    env_id, ent_id = wp.tid()
    
    if is_active[env_id, ent_id] == 0:
        return
        
    # 1. Self destruct (missile too slow)
    if is_missile[env_id, ent_id] == 1:
        _vx = vx[env_id, ent_id]
        _vy = vy[env_id, ent_id]
        v_sq = _vx*_vx + _vy*_vy
        if v_sq < config.self_destruct_speed_sq:
            is_active[env_id, ent_id] = 0
            alive[env_id, ent_id] = 0
            return # Dead
            
    # 2. Hit check (only missiles check hits)
    if is_missile[env_id, ent_id] == 1:
        tgt_idx = target_idx[env_id, ent_id]
        if tgt_idx >= 0 and is_active[env_id, tgt_idx] == 1:
            dx = x[env_id, ent_id] - x[env_id, tgt_idx]
            dy = y[env_id, ent_id] - y[env_id, tgt_idx]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < config.hit_radius_sq:
                # HIT!
                # Mark missile as dead
                is_active[env_id, ent_id] = 0
                alive[env_id, ent_id] = 0
                
                # Mark target as hit (using atomic to be safe, though scatter might be ok if 1 missile hits 1 target)
                # But here we write to hit_events
                wp.atomic_add(hit_events, env_id, tgt_idx, 1)

@wp.kernel
def apply_hits_kernel(
    hit_events: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2)
):
    env_id, ent_id = wp.tid()
    if hit_events[env_id, ent_id] > 0:
        is_active[env_id, ent_id] = 0
        alive[env_id, ent_id] = 0
        hit_events[env_id, ent_id] = 0 # Reset event

@wp.kernel
def fire_kernel(
    # Inputs
    fire_cmd: wp.array(dtype=int, ndim=2),
    is_player1: wp.array(dtype=int, ndim=2), # To check ownership
    missile_count: wp.array(dtype=int, ndim=2),
    # State to read from source
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    vx: wp.array(dtype=float, ndim=2),
    vy: wp.array(dtype=float, ndim=2),
    # State to write to missile
    out_x: wp.array(dtype=float, ndim=2),
    out_y: wp.array(dtype=float, ndim=2),
    out_vx: wp.array(dtype=float, ndim=2),
    out_vy: wp.array(dtype=float, ndim=2),
    out_is_missile: wp.array(dtype=int, ndim=2),
    out_is_active: wp.array(dtype=int, ndim=2),
    out_is_player1: wp.array(dtype=int, ndim=2),
    out_alive: wp.array(dtype=int, ndim=2),
    out_throttle: wp.array(dtype=float, ndim=2),
    out_rudder: wp.array(dtype=float, ndim=2),
    out_engine_time: wp.array(dtype=float, ndim=2),
    out_target_idx: wp.array(dtype=int, ndim=2),
    out_source_idx: wp.array(dtype=int, ndim=2),
    out_prev_los_angle: wp.array(dtype=float, ndim=2),
    # Config
    config: GameConfig,
    max_entities: int
):
    # One thread per environment
    env_id = wp.tid()
    
    # Check P1 Fire
    p1_idx = config.p1_idx
    if fire_cmd[env_id, p1_idx] == 1 and missile_count[env_id, p1_idx] > 0 and out_is_active[env_id, p1_idx] == 1:
        # Find slot (Even slots: 2, 4, ...)
        found_slot = int(-1)
        for s in range(config.first_missile_idx, max_entities):
            if s % 2 == 0: # Even slot for P1
                if out_is_active[env_id, s] == 0:
                    found_slot = s
                    break
        
        if found_slot != -1:
            # Fire!
            missile_count[env_id, p1_idx] -= 1
            fire_cmd[env_id, p1_idx] = 0 # Consume command
            
            # Init missile
            mx = x[env_id, p1_idx]
            my = y[env_id, p1_idx]
            mvx = vx[env_id, p1_idx]
            mvy = vy[env_id, p1_idx]
            angle = wp.atan2(mvy, mvx)
            
            # Offset
            ox = wp.cos(angle) * config.missile_launch_offset
            oy = wp.sin(angle) * config.missile_launch_offset
            
            out_x[env_id, found_slot] = mx + ox
            out_y[env_id, found_slot] = my + oy
            out_vx[env_id, found_slot] = mvx
            out_vy[env_id, found_slot] = mvy
            
            out_is_missile[env_id, found_slot] = 1
            out_is_active[env_id, found_slot] = 1
            out_is_player1[env_id, found_slot] = 1
            out_alive[env_id, found_slot] = 1 # Missiles are "alive" until hit
            
            out_throttle[env_id, found_slot] = 0.0
            out_rudder[env_id, found_slot] = 0.0
            out_engine_time[env_id, found_slot] = config.missile_engine_duration
            
            out_target_idx[env_id, found_slot] = config.p2_idx
            out_source_idx[env_id, found_slot] = p1_idx
            
            # Init LOS
            tx = x[env_id, config.p2_idx]
            ty = y[env_id, config.p2_idx]
            idx = tx - (mx + ox)
            idy = ty - (my + oy)
            out_prev_los_angle[env_id, found_slot] = wp.atan2(idy, idx)

    # Check P2 Fire
    p2_idx = config.p2_idx
    if fire_cmd[env_id, p2_idx] == 1 and missile_count[env_id, p2_idx] > 0 and out_is_active[env_id, p2_idx] == 1:
        # Find slot (Odd slots: 3, 5, ...)
        found_slot = int(-1)
        for s in range(config.first_missile_idx, max_entities):
            if s % 2 != 0: # Odd slot for P2
                if out_is_active[env_id, s] == 0:
                    found_slot = s
                    break
        
        if found_slot != -1:
            # Fire!
            missile_count[env_id, p2_idx] -= 1
            fire_cmd[env_id, p2_idx] = 0
            
            mx = x[env_id, p2_idx]
            my = y[env_id, p2_idx]
            mvx = vx[env_id, p2_idx]
            mvy = vy[env_id, p2_idx]
            angle = wp.atan2(mvy, mvx)
            
            ox = wp.cos(angle) * config.missile_launch_offset
            oy = wp.sin(angle) * config.missile_launch_offset
            
            out_x[env_id, found_slot] = mx + ox
            out_y[env_id, found_slot] = my + oy
            out_vx[env_id, found_slot] = mvx
            out_vy[env_id, found_slot] = mvy
            
            out_is_missile[env_id, found_slot] = 1
            out_is_active[env_id, found_slot] = 1
            out_is_player1[env_id, found_slot] = 0
            out_alive[env_id, found_slot] = 1
            
            out_throttle[env_id, found_slot] = 0.0
            out_rudder[env_id, found_slot] = 0.0
            out_engine_time[env_id, found_slot] = config.missile_engine_duration
            
            out_target_idx[env_id, found_slot] = config.p1_idx
            out_source_idx[env_id, found_slot] = p2_idx
            
            tx = x[env_id, config.p1_idx]
            ty = y[env_id, config.p1_idx]
            idx = tx - (mx + ox)
            idy = ty - (my + oy)
            out_prev_los_angle[env_id, found_slot] = wp.atan2(idy, idx)

@wp.kernel
def reset_phase1_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    vx: wp.array(dtype=float, ndim=2),
    vy: wp.array(dtype=float, ndim=2),
    throttle: wp.array(dtype=float, ndim=2),
    rudder: wp.array(dtype=float, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    is_player1: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2),
    missile_count: wp.array(dtype=int, ndim=2),
    angle: wp.array(dtype=float, ndim=2),
    speed: wp.array(dtype=float, ndim=2),
    target_idx: wp.array(dtype=int, ndim=2),
    source_idx: wp.array(dtype=int, ndim=2),
    fire_cmd: wp.array(dtype=int, ndim=2),
    env_mask: wp.array(dtype=int, ndim=1),
    config: GameConfig,
    initial_missiles: int,
    seed: int
):
    env_id = wp.tid()
    if env_mask[env_id] == 0:
        return
    
    # Reset missiles (loop)
    for i in range(config.first_missile_idx, 20):
        x[env_id, i] = 0.0
        y[env_id, i] = 0.0
        vx[env_id, i] = 0.0
        vy[env_id, i] = 0.0
        is_missile[env_id, i] = 0
        is_active[env_id, i] = 0
        is_player1[env_id, i] = 0
        alive[env_id, i] = 0
        missile_count[env_id, i] = 0
        target_idx[env_id, i] = -1
        source_idx[env_id, i] = -1
        fire_cmd[env_id, i] = 0
        
    # P1 (Center)
    p1 = config.p1_idx
    x[env_id, p1] = config.battlefield_size / 2.0
    y[env_id, p1] = config.battlefield_size / 2.0
    
    # Random heading for P1
    state = wp.rand_init(seed, env_id)
    p1_angle = wp.randf(state) * 360.0
    angle[env_id, p1] = p1_angle
    
    p1_angle_rad = wp.radians(p1_angle)
    p1_speed = 300.0
    vx[env_id, p1] = wp.cos(p1_angle_rad) * p1_speed
    vy[env_id, p1] = wp.sin(p1_angle_rad) * p1_speed
    
    throttle[env_id, p1] = 1.0
    rudder[env_id, p1] = 0.0
    is_missile[env_id, p1] = 0
    is_active[env_id, p1] = 1
    is_player1[env_id, p1] = 1
    alive[env_id, p1] = 1
    missile_count[env_id, p1] = initial_missiles
    speed[env_id, p1] = p1_speed
    target_idx[env_id, p1] = -1
    source_idx[env_id, p1] = -1
    fire_cmd[env_id, p1] = 0
    
    # P2 (Random direction 40km away)
    p2 = config.p2_idx
    dist = 40000.0
    p2_rel_angle = wp.randf(state) * 2.0 * 3.14159
    
    x[env_id, p2] = x[env_id, p1] + wp.cos(p2_rel_angle) * dist
    y[env_id, p2] = y[env_id, p1] + wp.sin(p2_rel_angle) * dist
    
    # P2 heading towards P1 (roughly)
    # At 40km, P2 should fly towards P1 to engage
    p2_angle_rad = p2_rel_angle + 3.14159 # Opposite direction
    # Add some noise
    p2_angle_rad += (wp.randf(state) - 0.5) * 1.0 # +/- 0.5 rad
    
    p2_speed = 300.0
    vx[env_id, p2] = wp.cos(p2_angle_rad) * p2_speed
    vy[env_id, p2] = wp.sin(p2_angle_rad) * p2_speed
    
    angle[env_id, p2] = wp.degrees(p2_angle_rad) % 360.0
    throttle[env_id, p2] = 1.0
    rudder[env_id, p2] = 0.0
    is_missile[env_id, p2] = 0
    is_active[env_id, p2] = 1
    is_player1[env_id, p2] = 0
    alive[env_id, p2] = 1
    missile_count[env_id, p2] = initial_missiles
    speed[env_id, p2] = p2_speed
    target_idx[env_id, p2] = -1
    source_idx[env_id, p2] = -1
    fire_cmd[env_id, p2] = 0

@wp.kernel
def reset_standard_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    vx: wp.array(dtype=float, ndim=2),
    vy: wp.array(dtype=float, ndim=2),
    throttle: wp.array(dtype=float, ndim=2),
    rudder: wp.array(dtype=float, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    is_player1: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2),
    missile_count: wp.array(dtype=int, ndim=2),
    angle: wp.array(dtype=float, ndim=2),
    speed: wp.array(dtype=float, ndim=2),
    target_idx: wp.array(dtype=int, ndim=2),
    source_idx: wp.array(dtype=int, ndim=2),
    fire_cmd: wp.array(dtype=int, ndim=2),
    env_mask: wp.array(dtype=int, ndim=1),
    config: GameConfig,
    initial_missiles: int
):
    env_id = wp.tid()
    if env_mask[env_id] == 0:
        return
    
    # Reset missiles (loop)
    for i in range(config.first_missile_idx, 20): # Hardcoded max entities for loop
        x[env_id, i] = 0.0
        y[env_id, i] = 0.0
        vx[env_id, i] = 0.0
        vy[env_id, i] = 0.0
        is_missile[env_id, i] = 0
        is_active[env_id, i] = 0
        is_player1[env_id, i] = 0
        alive[env_id, i] = 0
        missile_count[env_id, i] = 0
        target_idx[env_id, i] = -1
        source_idx[env_id, i] = -1
        fire_cmd[env_id, i] = 0
        
    # P1
    p1 = config.p1_idx
    x[env_id, p1] = 10000.0
    y[env_id, p1] = config.battlefield_size / 2.0
    vx[env_id, p1] = 300.0
    vy[env_id, p1] = 0.0
    throttle[env_id, p1] = 1.0
    rudder[env_id, p1] = 0.0
    is_missile[env_id, p1] = 0
    is_active[env_id, p1] = 1
    is_player1[env_id, p1] = 1
    alive[env_id, p1] = 1
    missile_count[env_id, p1] = initial_missiles
    angle[env_id, p1] = 0.0
    speed[env_id, p1] = 300.0
    target_idx[env_id, p1] = -1
    source_idx[env_id, p1] = -1
    fire_cmd[env_id, p1] = 0
    
    # P2
    p2 = config.p2_idx
    x[env_id, p2] = config.battlefield_size - 10000.0
    y[env_id, p2] = config.battlefield_size / 2.0
    vx[env_id, p2] = -300.0
    vy[env_id, p2] = 0.0
    throttle[env_id, p2] = 1.0
    rudder[env_id, p2] = 0.0
    is_missile[env_id, p2] = 0
    is_active[env_id, p2] = 1
    is_player1[env_id, p2] = 0
    alive[env_id, p2] = 1
    missile_count[env_id, p2] = initial_missiles
    angle[env_id, p2] = 180.0
    speed[env_id, p2] = 300.0
    target_idx[env_id, p2] = -1
    source_idx[env_id, p2] = -1
    fire_cmd[env_id, p2] = 0

@wp.kernel
def map_actions_kernel(
    p1_action_discrete: wp.array(dtype=int, ndim=2), # [num_envs, 2]
    p2_action_discrete: wp.array(dtype=int, ndim=2),
    # Output to states
    rudder: wp.array(dtype=float, ndim=2),
    throttle: wp.array(dtype=float, ndim=2),
    fire_cmd: wp.array(dtype=int, ndim=2),
    config: GameConfig
):
    env_id = wp.tid()
    
    # P1
    # Action 0: Rudder (0: Left, 1: Straight, 2: Right)
    rudder_idx = p1_action_discrete[env_id, 0]
    # Action 1: Fire (0: Hold, 1: Fire)
    fire_idx = p1_action_discrete[env_id, 1]
    
    r_val = 0.0
    if rudder_idx == 0: r_val = -1.0
    elif rudder_idx == 2: r_val = 1.0
    
    rudder[env_id, config.p1_idx] = r_val
    throttle[env_id, config.p1_idx] = 1.0
    fire_cmd[env_id, config.p1_idx] = fire_idx
    
    # P2
    rudder_idx_2 = p2_action_discrete[env_id, 0]
    fire_idx_2 = p2_action_discrete[env_id, 1]
    
    r_val_2 = 0.0
    if rudder_idx_2 == 0: r_val_2 = -1.0
    elif rudder_idx_2 == 2: r_val_2 = 1.0
    
    rudder[env_id, config.p2_idx] = r_val_2
    throttle[env_id, config.p2_idx] = 1.0
    fire_cmd[env_id, config.p2_idx] = fire_idx_2

@wp.kernel
def get_obs_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    angle: wp.array(dtype=float, ndim=2),
    speed: wp.array(dtype=float, ndim=2),
    missile_count: wp.array(dtype=int, ndim=2),
    alive: wp.array(dtype=int, ndim=2),
    config: GameConfig,
    initial_missiles: int,
    obs_p1: wp.array(dtype=float, ndim=2),
    obs_p2: wp.array(dtype=float, ndim=2)
):
    env_id = wp.tid()
    
    p1 = config.p1_idx
    p2 = config.p2_idx
    
    # Helper for P1
    dx = x[env_id, p2] - x[env_id, p1]
    dy = y[env_id, p2] - y[env_id, p1]
    dist = wp.sqrt(dx*dx + dy*dy + 1e-7)
    bearing = wp.degrees(wp.atan2(dy, dx))
    
    # P1 Obs (0:x, 1:y, 2:angle, 3:speed, 4:missiles, 5:alive, 6:e_dist, 7:e_rel_ang, 8:e_speed, 9:e_alive)
    obs_p1[env_id, 0] = x[env_id, p1] / config.battlefield_size
    obs_p1[env_id, 1] = y[env_id, p1] / config.battlefield_size
    obs_p1[env_id, 2] = angle[env_id, p1] / 360.0
    obs_p1[env_id, 3] = speed[env_id, p1] / 400.0
    obs_p1[env_id, 4] = float(missile_count[env_id, p1]) / float(initial_missiles)
    obs_p1[env_id, 5] = float(alive[env_id, p1])
    obs_p1[env_id, 6] = dist / config.battlefield_size
    rel_ang_p1 = (bearing - angle[env_id, p1] + 180.0) % 360.0 - 180.0
    obs_p1[env_id, 7] = rel_ang_p1 / 180.0
    obs_p1[env_id, 8] = speed[env_id, p2] / 400.0
    obs_p1[env_id, 9] = float(alive[env_id, p2])
    
    # Helper for P2
    dx2 = -dx
    dy2 = -dy
    bearing2 = wp.degrees(wp.atan2(dy2, dx2))
    
    # P2 Obs
    obs_p2[env_id, 0] = x[env_id, p2] / config.battlefield_size
    obs_p2[env_id, 1] = y[env_id, p2] / config.battlefield_size
    obs_p2[env_id, 2] = angle[env_id, p2] / 360.0
    obs_p2[env_id, 3] = speed[env_id, p2] / 400.0
    obs_p2[env_id, 4] = float(missile_count[env_id, p2]) / float(initial_missiles)
    obs_p2[env_id, 5] = float(alive[env_id, p2])
    obs_p2[env_id, 6] = dist / config.battlefield_size
    rel_ang_p2 = (bearing2 - angle[env_id, p2] + 180.0) % 360.0 - 180.0
    obs_p2[env_id, 7] = rel_ang_p2 / 180.0
    obs_p2[env_id, 8] = speed[env_id, p1] / 400.0
    obs_p2[env_id, 9] = float(alive[env_id, p1])


@wp.kernel
def record_viz_state_kernel(
    x: wp.array(dtype=float, ndim=2),
    y: wp.array(dtype=float, ndim=2),
    angle: wp.array(dtype=float, ndim=2),
    speed: wp.array(dtype=float, ndim=2),
    alive: wp.array(dtype=int, ndim=2),
    missile_count: wp.array(dtype=int, ndim=2),
    rudder: wp.array(dtype=float, ndim=2),
    throttle: wp.array(dtype=float, ndim=2),
    turn_rate: wp.array(dtype=float, ndim=2),
    is_missile: wp.array(dtype=int, ndim=2),
    is_active: wp.array(dtype=int, ndim=2),
    is_player1: wp.array(dtype=int, ndim=2),
    frame_idx: int,
    viz_buffer: wp.array(dtype=float, ndim=3)
):
    # Only process env 0
    # tid corresponds to entity index in env 0
    ent_id = wp.tid()
    env_id = 0
    
    # viz_buffer: [frames, max_entities, features]
    viz_buffer[frame_idx, ent_id, 0] = x[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 1] = y[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 2] = angle[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 3] = speed[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 4] = float(alive[env_id, ent_id])
    viz_buffer[frame_idx, ent_id, 5] = float(missile_count[env_id, ent_id])
    viz_buffer[frame_idx, ent_id, 6] = rudder[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 7] = throttle[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 8] = turn_rate[env_id, ent_id]
    viz_buffer[frame_idx, ent_id, 9] = float(is_missile[env_id, ent_id])
    viz_buffer[frame_idx, ent_id, 10] = float(is_active[env_id, ent_id])
    viz_buffer[frame_idx, ent_id, 11] = float(is_player1[env_id, ent_id])


class WarpEnv(BaseEnv):
    def __init__(self, config, num_envs=1, max_entities=20, device='cuda'):
        super().__init__()
        init_warp()
        self._warp_device = get_warp_device(device)
        self._device = torch.device(device)
        self._num_envs = num_envs
        self.max_entities = max_entities
        
        # Config
        self.cfg_struct = GameConfig()
        self.cfg_struct.dt = 1.0/60.0
        self.cfg_struct.battlefield_size = float(config.get('battlefield_size', 50000))
        self.cfg_struct.fighter_v_term = float(config.get('FIGHTER_TERMINAL_VELOCITY', 400))
        self.cfg_struct.fighter_cl_max = 1.0 / float(config.get('FIGHTER_MIN_TURN_RADIUS', 1000))
        self.cfg_struct.fighter_max_thrust = float(config.get('FIGHTER_MAX_THRUST', 1.5 * 9.8))
        self.cfg_struct.fighter_ld_ratio = float(config.get('FIGHTER_LIFT_DRAG_RATIO', 5))
        self.cfg_struct.missile_v_term = float(config.get('MISSILE_TERMINAL_VELOCITY', 400))
        self.cfg_struct.missile_cl_max = 1.0 / float(config.get('MISSILE_MIN_TURN_RADIUS', 1000))
        self.cfg_struct.missile_thrust = float(config.get('MISSILE_THRUST', 15 * 9.8))
        self.cfg_struct.missile_ld_ratio = float(config.get('MISSILE_LIFT_DRAG_RATIO', 3))
        self.cfg_struct.missile_engine_duration = float(config.get('MISSILE_ENGINE_DURATION', 10.0))
        self.cfg_struct.missile_launch_offset = float(config.get('missile_launch_offset', 20.0))
        self.cfg_struct.hit_radius_sq = float(config.get('hit_radius', 100)**2)
        self.cfg_struct.self_destruct_speed_sq = float(config.get('self_destruct_speed', 200)**2)
        self.cfg_struct.guidance_gain = float(config.get('MISSILE_GUIDANCE_GAIN', 200)) / math.pi
        self.cfg_struct.p1_idx = 0
        self.cfg_struct.p2_idx = 1
        self.cfg_struct.first_missile_idx = 2
        
        # Visualization Buffer (on device)
        # Stores [frames, max_entities, features] for env 0
        self.viz_frames = 100
        self.viz_interval = 36 # Record every 36 steps (assuming 60s * 60fps = 3600 steps)
        self.viz_step_count = 0
        self.viz_write_idx = 0
        self.viz_filled = False
        self.viz_buffer = wp.zeros((self.viz_frames, self.max_entities, 12), dtype=float, device=self._warp_device)
        
        # Arrays
        shape = (num_envs, max_entities)
        self.x = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.y = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.vx = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.vy = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.rudder = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.throttle = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.fire_cmd = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.target_idx = wp.full(shape, -1, dtype=int, device=self._warp_device)
        self.source_idx = wp.full(shape, -1, dtype=int, device=self._warp_device)
        self.engine_time = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.prev_los_angle = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.missile_count = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.is_missile = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.is_active = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.is_player1 = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.alive = wp.zeros(shape, dtype=int, device=self._warp_device)
        self.speed = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.angle = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.n_load = wp.zeros(shape, dtype=float, device=self._warp_device)
        self.turn_rate = wp.zeros(shape, dtype=float, device=self._warp_device)
        
        self.hit_events = wp.zeros(shape, dtype=int, device=self._warp_device)
        
        # Obs buffers
        self.initial_missiles = int(config.get('initial_missiles', 6))
        self.obs_dim = 10
        self.obs_p1_wp = wp.zeros((num_envs, self.obs_dim), dtype=float, device=self._warp_device)
        self.obs_p2_wp = wp.zeros((num_envs, self.obs_dim), dtype=float, device=self._warp_device)
        
        # Graph
        self.graph = None
        
        # Action buffers for Graph Capture (Fix for CUDA Graph Input Trap)
        self.p1_action_buffer = wp.zeros((num_envs, 2), dtype=int, device=self._warp_device)
        self.p2_action_buffer = wp.zeros((num_envs, 2), dtype=int, device=self._warp_device)
        
        # Rewards (using torch for now)
        self.reward_config = config
        
        self.scenario = 'standard'
        
    def reset(self, env_mask=None):
        if env_mask is None:
            # Full reset
            mask_wp = wp.ones(self.num_envs, dtype=int, device=self._warp_device)
        else:
            # env_mask is tensor
            mask_wp = from_torch(env_mask.int())
            
        import random
        seed = random.randint(0, 1000000)
        
        if self.scenario == 'phase1':
            wp.launch(
                kernel=reset_phase1_kernel,
                dim=self.num_envs,
                inputs=[
                    self.x, self.y, self.vx, self.vy, self.throttle, self.rudder,
                    self.is_missile, self.is_active, self.is_player1, self.alive,
                    self.missile_count, self.angle, self.speed, self.target_idx,
                    self.source_idx, self.fire_cmd, mask_wp, self.cfg_struct, self.initial_missiles,
                    seed
                ],
                device=self._warp_device
            )
        else:
            wp.launch(
                kernel=reset_standard_kernel,
                dim=self.num_envs,
                inputs=[
                    self.x, self.y, self.vx, self.vy, self.throttle, self.rudder,
                    self.is_missile, self.is_active, self.is_player1, self.alive,
                    self.missile_count, self.angle, self.speed, self.target_idx,
                    self.source_idx, self.fire_cmd, mask_wp, self.cfg_struct, self.initial_missiles
                ],
                device=self._warp_device
            )
        
        return self.get_observations()

    def step(self, actions: Dict[str, torch.Tensor], dt=1/60.0):
        # actions: {'p1_action': [N, 2], 'p2_action': [N, 2]}
        
        # Copy actions to fixed buffers to ensure Graph Capture works correctly
        p1_act_torch = actions['p1_action'].int()
        p2_act_torch = actions['p2_action'].int()
        
        # Ensure we are using the fixed buffers
        # wp.from_torch creates a view, we need to copy contents to our pre-allocated buffers
        p1_view = from_torch(p1_act_torch)
        p2_view = from_torch(p2_act_torch)
        
        wp.copy(self.p1_action_buffer, p1_view)
        wp.copy(self.p2_action_buffer, p2_view)
        
        if self.graph is None:
            # Capture graph
            wp.capture_begin(device=self._warp_device)
            # Use fixed buffers as inputs
            self._run_step_kernels(self.p1_action_buffer, self.p2_action_buffer)
            self.graph = wp.capture_end(device=self._warp_device)
        
        # Replay
        wp.capture_launch(self.graph)
        
        # Record visualization frame
        if self.viz_step_count % self.viz_interval == 0:
            wp.launch(
                record_viz_state_kernel,
                dim=self.max_entities, # Only for env 0
                inputs=[
                    self.x, self.y, self.angle, self.speed, self.alive, self.missile_count,
                    self.rudder, self.throttle, self.turn_rate, self.is_missile, self.is_active, self.is_player1,
                    self.viz_write_idx, self.viz_buffer
                ],
                device=self._warp_device
            )
            self.viz_write_idx = (self.viz_write_idx + 1) % self.viz_frames
            if self.viz_write_idx == 0:
                self.viz_filled = True
        
        self.viz_step_count += 1
        
        # Obs
        obs = self.get_observations()
        
        # Rewards & Done (Compute in Torch for now to save time implementation)
        # We need to expose some state to torch
        rewards, dones, infos = self._compute_rewards_torch()
        
        return obs, rewards, dones, infos

    def _run_step_kernels(self, p1_act, p2_act):
        # Map actions
        wp.launch(map_actions_kernel, self.num_envs, inputs=[p1_act, p2_act, self.rudder, self.throttle, self.fire_cmd, self.cfg_struct], device=self._warp_device)
        
        # Fire
        wp.launch(fire_kernel, self.num_envs, inputs=[self.fire_cmd, self.is_player1, self.missile_count, self.x, self.y, self.vx, self.vy, self.x, self.y, self.vx, self.vy, self.is_missile, self.is_active, self.is_player1, self.alive, self.throttle, self.rudder, self.engine_time, self.target_idx, self.source_idx, self.prev_los_angle, self.cfg_struct, self.max_entities], device=self._warp_device)
        
        # Guidance
        wp.launch(guidance_kernel, (self.num_envs, self.max_entities), inputs=[self.x, self.y, self.rudder, self.prev_los_angle, self.target_idx, self.is_missile, self.is_active, self.alive, self.cfg_struct], device=self._warp_device)
        
        # Physics
        wp.launch(physics_step_kernel, (self.num_envs, self.max_entities), inputs=[self.x, self.y, self.vx, self.vy, self.speed, self.angle, self.rudder, self.throttle, self.engine_time, self.n_load, self.turn_rate, self.is_missile, self.is_active, self.cfg_struct], device=self._warp_device)
        
        # Events
        wp.launch(event_check_kernel, (self.num_envs, self.max_entities), inputs=[self.x, self.y, self.vx, self.vy, self.is_missile, self.is_active, self.alive, self.target_idx, self.hit_events, self.cfg_struct], device=self._warp_device)
        
        # Apply Hits
        wp.launch(apply_hits_kernel, (self.num_envs, self.max_entities), inputs=[self.hit_events, self.is_active, self.alive], device=self._warp_device)

    def get_render_state(self, env_idx: int = 0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Helper class for render entities
        class RenderEntity:
            def __init__(self, x, y, angle, speed, alive, missiles=0, slot_idx=None, color=(255, 255, 255), rudder=0.0, throttle=0.0, turn_rate=0.0):
                self.x = x
                self.y = y
                self.angle = angle
                self.speed = speed
                self.alive = alive
                self.missiles = missiles
                self.slot_idx = slot_idx
                self.color = color
                self.rudder = rudder
                self.throttle = throttle
                self.turn_rate = turn_rate
        
        # Helper to get CPU numpy array for a specific env
        def get_cpu(arr, idx):
            t = to_torch(arr)
            return t[idx].float().cpu().numpy() # Ensure float for most, int for others handled below
            
        def get_cpu_int(arr, idx):
            t = to_torch(arr)
            return t[idx].int().cpu().numpy()

        t1 = time.perf_counter()
        
        # Synchronize before fetching to measure pure fetch time vs wait time
        # Note: to_torch().cpu() implicitly synchronizes, so the first call will absorb the wait time.
        
        x = get_cpu(self.x, env_idx)
        t2 = time.perf_counter() # Time after first fetch (includes GPU wait)
        
        y = get_cpu(self.y, env_idx)
        angle = get_cpu(self.angle, env_idx)
        speed = get_cpu(self.speed, env_idx)
        alive = get_cpu_int(self.alive, env_idx)
        missile_count = get_cpu_int(self.missile_count, env_idx)
        is_missile = get_cpu_int(self.is_missile, env_idx)
        is_active = get_cpu_int(self.is_active, env_idx)
        
        # Extra fields for viz
        rudder = get_cpu(self.rudder, env_idx)
        throttle = get_cpu(self.throttle, env_idx)
        turn_rate = get_cpu(self.turn_rate, env_idx)
        is_player1 = get_cpu_int(self.is_player1, env_idx)
        
        t3 = time.perf_counter() # Time after all fetches
        
        p1_idx = self.cfg_struct.p1_idx
        p2_idx = self.cfg_struct.p2_idx
        
        # Colors
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        LIGHT_RED = (255, 150, 150)
        LIGHT_BLUE = (150, 150, 255)
        
        # Aircraft 1
        ac1 = RenderEntity(
            x[p1_idx], y[p1_idx], angle[p1_idx], speed[p1_idx], 
            bool(alive[p1_idx]), int(missile_count[p1_idx]),
            color=RED,
            rudder=float(rudder[p1_idx]),
            throttle=float(throttle[p1_idx]),
            turn_rate=float(turn_rate[p1_idx])
        )
        
        # Aircraft 2
        ac2 = RenderEntity(
            x[p2_idx], y[p2_idx], angle[p2_idx], speed[p2_idx], 
            bool(alive[p2_idx]), int(missile_count[p2_idx]),
            color=BLUE,
            rudder=float(rudder[p2_idx]),
            throttle=float(throttle[p2_idx]),
            turn_rate=float(turn_rate[p2_idx])
        )
        
        # Missiles
        missiles = []
        for i in range(self.max_entities):
            if is_missile[i] and is_active[i]:
                # Determine color
                m_color = LIGHT_RED if is_player1[i] else LIGHT_BLUE
                
                m = RenderEntity(
                    x[i], y[i], angle[i], speed[i], bool(alive[i]), slot_idx=i,
                    color=m_color
                )
                missiles.append(m)
                
        # Game Over / Winner
        game_over = not (bool(alive[p1_idx]) and bool(alive[p2_idx]))
        winner = None
        if game_over:
            if bool(alive[p1_idx]) and not bool(alive[p2_idx]):
                winner = 'red'
            elif bool(alive[p2_idx]) and not bool(alive[p1_idx]):
                winner = 'blue'
            else:
                winner = 'draw'
        
        t4 = time.perf_counter()
        
        print(f"[Timing] Total: {(t4-t0)*1000:.2f}ms | "
              f"GPU Wait+First Fetch: {(t2-t1)*1000:.2f}ms | "
              f"Rest Fetches: {(t3-t2)*1000:.2f}ms | "
              f"Logic: {(t4-t3)*1000:.2f}ms")
              
        return {
            'aircraft1': ac1,
            'aircraft2': ac2,
            'missiles': missiles,
            'game_over': game_over,
            'winner': winner
        }

    def get_recorded_frames(self) -> list:
        """
        Retrieves the recorded frames from the visualization buffer.
        Returns a list of dictionaries compatible with the renderer.
        """
        # Synchronize and fetch buffer
        wp.synchronize_device(self._warp_device)
        cpu_buffer = self.viz_buffer.numpy()
        
        frames = []
        
        # Helper class for render entities
        class RenderEntity:
            def __init__(self, x, y, angle, speed, alive, missiles=0, slot_idx=None, color=(255, 255, 255), rudder=0.0, throttle=0.0, turn_rate=0.0):
                self.x = x
                self.y = y
                self.angle = angle
                self.speed = speed
                self.alive = alive
                self.missiles = missiles
                self.slot_idx = slot_idx
                self.color = color
                self.rudder = rudder
                self.throttle = throttle
                self.turn_rate = turn_rate
        
        # Colors
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        LIGHT_RED = (255, 150, 150)
        LIGHT_BLUE = (150, 150, 255)
        
        p1_idx = self.cfg_struct.p1_idx
        p2_idx = self.cfg_struct.p2_idx
        
        # Reorder buffer to be chronological
        if self.viz_filled:
            # Full buffer: The oldest frame is at viz_write_idx
            indices = np.arange(self.viz_frames)
            indices = np.roll(indices, -self.viz_write_idx)
            ordered_buffer = cpu_buffer[indices]
            num_frames = self.viz_frames
        else:
            # Partial buffer: frames [0, viz_write_idx) are valid
            if self.viz_write_idx == 0:
                return [] # No frames recorded
            ordered_buffer = cpu_buffer[:self.viz_write_idx]
            num_frames = self.viz_write_idx
        
        for i in range(num_frames):
            frame_data = ordered_buffer[i] # [max_entities, 12]
            
            # Extract data
            x = frame_data[:, 0]
            y = frame_data[:, 1]
            angle = frame_data[:, 2]
            speed = frame_data[:, 3]
            alive = frame_data[:, 4].astype(int)
            missile_count = frame_data[:, 5].astype(int)
            rudder = frame_data[:, 6]
            throttle = frame_data[:, 7]
            turn_rate = frame_data[:, 8]
            is_missile = frame_data[:, 9].astype(int)
            is_active = frame_data[:, 10].astype(int)
            is_player1 = frame_data[:, 11].astype(int)
            
            # Aircraft 1
            ac1 = RenderEntity(
                x[p1_idx], y[p1_idx], angle[p1_idx], speed[p1_idx], 
                bool(alive[p1_idx]), int(missile_count[p1_idx]),
                color=RED,
                rudder=float(rudder[p1_idx]),
                throttle=float(throttle[p1_idx]),
                turn_rate=float(turn_rate[p1_idx])
            )
            
            # Aircraft 2
            ac2 = RenderEntity(
                x[p2_idx], y[p2_idx], angle[p2_idx], speed[p2_idx], 
                bool(alive[p2_idx]), int(missile_count[p2_idx]),
                color=BLUE,
                rudder=float(rudder[p2_idx]),
                throttle=float(throttle[p2_idx]),
                turn_rate=float(turn_rate[p2_idx])
            )
            
            # Missiles
            missiles = []
            for j in range(self.max_entities):
                if is_missile[j] and is_active[j]:
                    m_color = LIGHT_RED if is_player1[j] else LIGHT_BLUE
                    m = RenderEntity(
                        x[j], y[j], angle[j], speed[j], bool(alive[j]), slot_idx=j,
                        color=m_color
                    )
                    missiles.append(m)
                    
            # Game Over / Winner
            game_over = not (bool(alive[p1_idx]) and bool(alive[p2_idx]))
            winner = None
            if game_over:
                if bool(alive[p1_idx]) and not bool(alive[p2_idx]):
                    winner = 'red'
                elif bool(alive[p2_idx]) and not bool(alive[p1_idx]):
                    winner = 'blue'
                else:
                    winner = 'draw'
                    
            frames.append({
                'aircraft1': ac1,
                'aircraft2': ac2,
                'missiles': missiles,
                'game_over': game_over,
                'winner': winner
            })
            
        return frames

    def get_observations(self):
        wp.launch(get_obs_kernel, self.num_envs, inputs=[self.x, self.y, self.angle, self.speed, self.missile_count, self.alive, self.cfg_struct, self.initial_missiles, self.obs_p1_wp, self.obs_p2_wp], device=self._warp_device)
        
        return {
            'p1': to_torch(self.obs_p1_wp),
            'p2': to_torch(self.obs_p2_wp)
        }

    def _compute_rewards_torch(self):
        # Zero-copy conversion
        alive_t = to_torch(self.alive)
        p1_alive = alive_t[:, self.cfg_struct.p1_idx].bool()
        p2_alive = alive_t[:, self.cfg_struct.p2_idx].bool()
        
        # Check Done
        # Simple rule: if either dies
        done = (~p1_alive) | (~p2_alive)
        
        # Rewards
        rew_p1 = torch.zeros(self.num_envs, device=alive_t.device)
        rew_p2 = torch.zeros(self.num_envs, device=alive_t.device)
        
        # Win/Lose
        win_reward = 500.0
        rew_p1 = torch.where(p1_alive & ~p2_alive, torch.tensor(win_reward, device=alive_t.device), rew_p1)
        rew_p1 = torch.where(~p1_alive & p2_alive, torch.tensor(-win_reward, device=alive_t.device), rew_p1)
        
        rew_p2 = torch.where(p2_alive & ~p1_alive, torch.tensor(win_reward, device=alive_t.device), rew_p2)
        rew_p2 = torch.where(~p2_alive & p1_alive, torch.tensor(-win_reward, device=alive_t.device), rew_p2)
        
        infos = {'winner': torch.zeros(self.num_envs, dtype=torch.long, device=alive_t.device)}
        infos['winner'] = torch.where(p1_alive & ~p2_alive, torch.tensor(1, device=alive_t.device), infos['winner'])
        infos['winner'] = torch.where(p2_alive & ~p1_alive, torch.tensor(2, device=alive_t.device), infos['winner'])
        
        return {'p1': rew_p1, 'p2': rew_p2}, done, infos
        
    def compute_rewards(self, done_mask, winner_mask=None):
        pass # Not used directly in new loop
        
    @property
    def states(self) -> Dict[str, torch.Tensor]:
        """
        Expose underlying state tensors for spacetime computer compatibility.
        Returns a dict of torch tensors referencing the warp arrays.
        """
        return {
            'x': to_torch(self.x),
            'y': to_torch(self.y),
            'vx': to_torch(self.vx),
            'vy': to_torch(self.vy),
            'rudder': to_torch(self.rudder),
            'throttle': to_torch(self.throttle),
            'angle': to_torch(self.angle),
            'speed': to_torch(self.speed),
            'is_missile': to_torch(self.is_missile).bool(),
            'is_active': to_torch(self.is_active).bool(),
            'alive': to_torch(self.alive).bool(),
            'target_idx': to_torch(self.target_idx),
            'source_idx': to_torch(self.source_idx),
            'engine_time': to_torch(self.engine_time),
            'missile_count': to_torch(self.missile_count)
        }
