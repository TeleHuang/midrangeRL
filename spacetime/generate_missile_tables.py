# -*- coding: utf-8 -*-
"""
导弹飞行曲线预计算脚本

功能：计算载机速度 200-500 m/s 范围内导弹直线飞行的时空曲线
输出：生成查找表供威胁锥渲染使用 + 可视化图表

物理模型说明：
    - 导弹推力: 15G (147 m/s²)，发动机工作10秒
    - 阻力模型: drag = Cd0 * v²，其中Cd0 = G / terminal_velocity²
    - 理论极速: sqrt(15) * 400 ≈ 1549 m/s ≈ 4.56马赫
    - 发动机关机后，导弹由于阻力逐渐减速
"""

import numpy as np
import json
import os
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass  # 如果字体不可用，使用默认字体

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import CONFIG


class MissileFlightSimulator:
    """导弹飞行模拟器 - 用于预计算导弹直线飞行轨迹"""
    
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        
        self.config = config
        
        # 导弹物理参数
        self.terminal_velocity = config.get('MISSILE_TERMINAL_VELOCITY', 400)
        self.min_turn_radius = config.get('MISSILE_MIN_TURN_RADIUS', 1000)
        self.thrust = config.get('MISSILE_THRUST', 15 * 9.8)  # m/s²
        self.engine_duration = config.get('MISSILE_ENGINE_DURATION', 10.0)  # s
        self.lift_drag_ratio = config.get('MISSILE_LIFT_DRAG_RATIO', 2)
        self.G = config.get('G', 9.8)
        
        # 预计算阻力相关系数
        self.Cd0 = self.G / (self.terminal_velocity ** 2)
        
        # 无威胁阈值 (低于此速度认为导弹无威胁)
        self.no_threat_speed = 340.0  # 1马赫
        
    def simulate_straight_flight(self, initial_speed, dt=0.1, max_time=60.0):
        """
        模拟导弹直线飞行（理想最大射程，不考虑制导转向消耗）
        
        威胁锥打表用途：模拟完整飞行过程直到速度接近0或时间上限
        "无威胁"判定仅在预测有制导导弹轨迹时进行
        
        Args:
            initial_speed: 导弹初始速度 (通常等于载机速度)
            dt: 时间步长 (秒)
            max_time: 最大模拟时间 (秒)
            
        Returns:
            list of dict: 每个时间点的 {time, distance, speed}
        """
        trajectory = []
        
        # 初始状态
        distance = 0.0  # 飞行距离
        speed = initial_speed
        engine_time = self.engine_duration
        time = 0.0
        
        # 最小有效速度（低于此速度停止模拟）
        min_speed = 50.0  # m/s
        
        while time <= max_time and speed > min_speed:
            # 记录当前状态
            trajectory.append({
                'time': time,
                'distance': distance,
                'speed': speed,
                'mach': speed / 340.0
            })
            
            # 计算推力加速度
            if engine_time > 0:
                thrust_accel = self.thrust
                engine_time -= dt
            else:
                thrust_accel = 0.0
            
            # 计算阻力加速度 (直线飞行无诱导阻力)
            drag_accel = self.Cd0 * speed * speed
            
            # 净加速度
            accel = thrust_accel - drag_accel
            
            # 更新速度和位置
            speed += accel * dt
            speed = max(0, speed)  # 速度不能为负
            distance += speed * dt
            time += dt
        
        # 记录最终状态
        if trajectory and (time > trajectory[-1]['time'] + dt/2):
            trajectory.append({
                'time': time,
                'distance': distance,
                'speed': speed,
                'mach': speed / 340.0
            })
        
        return trajectory
    
    def generate_lookup_table(self, speed_min=200, speed_max=500, speed_step=10):
        """
        生成导弹飞行曲线查找表
        
        Args:
            speed_min: 最小载机速度
            speed_max: 最大载机速度  
            speed_step: 速度采样步长
            
        Returns:
            dict: 查找表数据
        """
        lookup_table = {
            'metadata': {
                'speed_min': speed_min,
                'speed_max': speed_max,
                'speed_step': speed_step,
                'missile_params': {
                    'terminal_velocity': self.terminal_velocity,
                    'min_turn_radius': self.min_turn_radius,
                    'thrust': self.thrust,
                    'engine_duration': self.engine_duration,
                    'lift_drag_ratio': self.lift_drag_ratio,
                    'no_threat_speed': self.no_threat_speed,
                }
            },
            'trajectories': {}
        }
        
        speeds = list(range(speed_min, speed_max + 1, speed_step))
        
        for speed in speeds:
            trajectory = self.simulate_straight_flight(speed)
            # 只保留关键点以减小文件大小 (每0.5秒一个点)
            # 使用四舍五入避免浮点数精度问题
            sampled = []
            for t in trajectory:
                time_mod = round(t['time'] * 2) / 2  # 四舍五入到0.5的倍数
                if abs(t['time'] - time_mod) < 0.05 or t == trajectory[-1]:
                    sampled.append(t)
            lookup_table['trajectories'][str(speed)] = sampled
        
        return lookup_table
    
    def get_threat_cone_profile(self, initial_speed, side_factor=0.5):
        """
        获取威胁锥轮廓
        正前方使用完整导弹射程
        正侧方使用 side_factor 倍的距离
        
        Args:
            initial_speed: 载机速度
            side_factor: 侧向距离衰减因子
            
        Returns:
            dict: 威胁锥参数
        """
        trajectory = self.simulate_straight_flight(initial_speed)
        
        # 提取时间-距离曲线
        times = [t['time'] for t in trajectory]
        front_distances = [t['distance'] for t in trajectory]
        side_distances = [t['distance'] * side_factor for t in trajectory]
        speeds = [t['speed'] for t in trajectory]
        
        return {
            'initial_speed': initial_speed,
            'times': times,
            'front_distances': front_distances,
            'side_distances': side_distances,
            'speeds': speeds,
            'max_range': front_distances[-1] if front_distances else 0,
            'time_to_no_threat': times[-1] if times else 0,
        }


def plot_trajectories(simulator, save_path=None):
    """
    绘制导弹飞行曲线可视化
    
    使用完整模拟数据绘图（而非采样后的查找表），以确保曲线平滑准确
    
    Args:
        simulator: MissileFlightSimulator 实例
        save_path: 可选的图像保存路径
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 采样速度点（用于图例显示）
    sample_speeds = [200, 300, 400, 500]
    colors = ['blue', 'green', 'orange', 'red']
    
    # 发动机工作时间
    engine_duration = simulator.engine_duration
    
    # ====== 1. 时间-距离曲线 ======
    ax1 = axes[0, 0]
    for speed, color in zip(sample_speeds, colors):
        trajectory = simulator.simulate_straight_flight(speed)
        times = [t['time'] for t in trajectory]
        distances = [t['distance'] / 1000 for t in trajectory]
        ax1.plot(times, distances, color=color, label=f'{speed} m/s', linewidth=1.5)
    
    # 添加发动机关机时间标记
    ax1.axvline(x=engine_duration, color='gray', linestyle=':', alpha=0.7, label=f'发动机关机 ({engine_duration}s)')
    
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('飞行距离 (km)')
    ax1.set_title('导弹飞行距离-时间曲线')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)
    
    # ====== 2. 时间-速度曲线 ======
    ax2 = axes[0, 1]
    max_speed_reached = 0
    for speed, color in zip(sample_speeds, colors):
        trajectory = simulator.simulate_straight_flight(speed)
        times = [t['time'] for t in trajectory]
        speeds = [t['speed'] for t in trajectory]
        ax2.plot(times, speeds, color=color, label=f'{speed} m/s', linewidth=1.5)
        max_speed_reached = max(max_speed_reached, max(speeds) if speeds else 0)
    
    # 添加音速参考线
    ax2.axhline(y=340, color='r', linestyle='--', linewidth=1, label='音速 (340 m/s)')
    # 添加发动机关机时间标记
    ax2.axvline(x=engine_duration, color='gray', linestyle=':', alpha=0.7, label=f'发动机关机')
    
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('速度 (m/s)')
    ax2.set_title('导弹速度-时间曲线')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, max_speed_reached * 1.1)
    
    # ====== 3. 时间-马赫数曲线 ======
    ax3 = axes[1, 0]
    max_mach_reached = 0
    for speed, color in zip(sample_speeds, colors):
        trajectory = simulator.simulate_straight_flight(speed)
        times = [t['time'] for t in trajectory]
        machs = [t['mach'] for t in trajectory]
        ax3.plot(times, machs, color=color, label=f'{speed} m/s', linewidth=1.5)
        max_mach_reached = max(max_mach_reached, max(machs) if machs else 0)
    
    # 添加马赫1参考线
    ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='马赫 1')
    # 添加发动机关机时间标记
    ax3.axvline(x=engine_duration, color='gray', linestyle=':', alpha=0.7, label=f'发动机关机')
    
    ax3.set_xlabel('时间 (秒)')
    ax3.set_ylabel('马赫数')
    ax3.set_title('导弹马赫数-时间曲线')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 60)
    ax3.set_ylim(0, max_mach_reached * 1.1)
    
    # ====== 4. 载机速度 vs 导弹性能 ======
    ax4 = axes[1, 1]
    all_speeds = list(range(200, 501, 20))
    max_ranges = []
    peak_machs = []
    
    for speed in all_speeds:
        trajectory = simulator.simulate_straight_flight(speed)
        if trajectory:
            max_ranges.append(trajectory[-1]['distance'] / 1000)
            peak_machs.append(max(t['mach'] for t in trajectory))
        else:
            max_ranges.append(0)
            peak_machs.append(0)
    
    ax4_twin = ax4.twinx()
    line1, = ax4.plot(all_speeds, max_ranges, 'b-o', markersize=4, label='最大射程')
    line2, = ax4_twin.plot(all_speeds, peak_machs, 'g-s', markersize=4, label='峰值马赫数')
    
    ax4.set_xlabel('载机速度 (m/s)')
    ax4.set_ylabel('最大射程 (km)', color='b')
    ax4_twin.set_ylabel('峰值马赫数', color='g')
    ax4.set_title('载机速度 vs 导弹性能')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.legend(handles=[line1, line2], loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_spacetime_3d(lookup_table, save_path=None):
    """绘制3D时空图示意"""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 选择几个代表性速度绘制
    sample_speeds = ['200', '300', '400', '500']
    colors = ['blue', 'green', 'orange', 'red']
    
    for speed_str, color in zip(sample_speeds, colors):
        if speed_str not in lookup_table['trajectories']:
            continue
            
        trajectory = lookup_table['trajectories'][speed_str]
        times = [t['time'] for t in trajectory]
        distances = [t['distance'] / 1000 for t in trajectory]
        
        # 正前方曲线 (y = distance, x = 0, z = -time)
        # 时间向下为未来
        ax.plot([0]*len(times), distances, [-t for t in times], 
                color=color, linewidth=2, label=f'{speed_str} m/s 正前方')
        
        # 侧向曲线 (距离减半)
        side_distances = [d * 0.5 for d in distances]
        ax.plot(side_distances, [0]*len(times), [-t for t in times],
                color=color, linewidth=1, linestyle='--', alpha=0.6)
        ax.plot([-d for d in side_distances], [0]*len(times), [-t for t in times],
                color=color, linewidth=1, linestyle='--', alpha=0.6)
    
    # 绘制时间截面
    for t_level in [0, -5, -10, -15]:
        # 简单的圆形截面示意
        theta = np.linspace(0, np.pi, 50)  # 只画前半球
        radius = 5  # km
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = [t_level] * len(theta)
        ax.plot(x, y, z, 'k-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X (km) - 侧向')
    ax.set_ylabel('Y (km) - 前向')
    ax.set_zlabel('时间 (秒)')
    ax.set_title('威胁锥时空示意图\n(向下为未来)')
    ax.legend(loc='upper left')
    
    # 设置视角
    ax.view_init(elev=30, azim=-60)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"3D图表已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数 - 生成查找表并可视化"""
    
    print("=" * 60)
    print("导弹飞行曲线预计算脚本")
    print("=" * 60)
    
    # 创建模拟器
    simulator = MissileFlightSimulator()
    
    print(f"\n导弹物理参数:")
    print(f"  终端速度（阻力参数）: {simulator.terminal_velocity} m/s")
    print(f"  推力加速度: {simulator.thrust:.1f} m/s² ({simulator.thrust/9.8:.1f}G)")
    print(f"  发动机时间: {simulator.engine_duration} s")
    print(f"  升阻比: {simulator.lift_drag_ratio}")
    
    # 计算理论极速
    theoretical_max_speed = simulator.terminal_velocity * math.sqrt(simulator.thrust / simulator.G)
    theoretical_max_mach = theoretical_max_speed / 340.0
    print(f"  理论极速: {theoretical_max_speed:.0f} m/s ({theoretical_max_mach:.2f}马赫)")
    
    # 生成查找表
    print("\n生成查找表...")
    lookup_table = simulator.generate_lookup_table(
        speed_min=200,
        speed_max=500, 
        speed_step=20
    )
    
    # 保存查找表
    output_dir = os.path.dirname(os.path.abspath(__file__))
    table_path = os.path.join(output_dir, 'missile_tables.json')
    
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(lookup_table, f, indent=2, ensure_ascii=False)
    print(f"查找表已保存到: {table_path}")
    
    # 打印摘要
    print("\n查找表摘要:")
    for speed_str, trajectory in sorted(lookup_table['trajectories'].items(),
                                         key=lambda x: int(x[0])):
        if trajectory:
            max_dist = trajectory[-1]['distance'] / 1000
            max_time = trajectory[-1]['time']
            print(f"  载机 {speed_str:>3} m/s: 最大射程 {max_dist:>5.1f} km, "
                  f"失效时间 {max_time:>4.1f} s")
    
    # 绘制可视化（使用完整模拟数据，曲线更平滑）
    print("\n生成可视化图表...")
    plot_path = os.path.join(output_dir, 'missile_curves.png')
    plot_trajectories(simulator, save_path=plot_path)
    
    plot_3d_path = os.path.join(output_dir, 'threat_cone_3d.png')
    plot_spacetime_3d(lookup_table, save_path=plot_3d_path)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
