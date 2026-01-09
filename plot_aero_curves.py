import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG


def compute_curves(terminal_velocity, min_turn_radius, lift_drag_ratio, g=9.8, mach_speed=340.0):
    epsilon = 1e-7
    cl_max = 1.0 / float(min_turn_radius)
    cd0 = g / float(terminal_velocity) ** 2
    k = 1.0 / (4.0 * cd0 * (float(lift_drag_ratio) ** 2) + epsilon)
    rudder = np.linspace(-1.0, 1.0, 801)
    cl = np.abs(rudder) * cl_max
    cd = cd0 + k * (cl ** 2)
    drag_accel = cd * (mach_speed ** 2)
    ld = cl / cd
    return rudder, drag_accel, ld


def main():
    g = CONFIG.get("G", 9.8)
    mach_speed = 340.0
    r_m, da_m, ld_m = compute_curves(
        CONFIG.get("MISSILE_TERMINAL_VELOCITY", 400),
        CONFIG.get("MISSILE_MIN_TURN_RADIUS", 1000),
        CONFIG.get("MISSILE_LIFT_DRAG_RATIO", 2),
        g,
        mach_speed,
    )
    r_f, da_f, ld_f = compute_curves(
        CONFIG.get("FIGHTER_TERMINAL_VELOCITY", 400),
        CONFIG.get("FIGHTER_MIN_TURN_RADIUS", 1000),
        CONFIG.get("FIGHTER_LIFT_DRAG_RATIO", 5),
        g,
        mach_speed,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(r_m, da_m, label="Drag Accel", color="tab:blue")
    ax.set_title("Missile Drag Acceleration vs Rudder (Mach 1)")
    ax.set_xlabel("Rudder")
    ax.set_ylabel("Drag Accel (m/s^2)")
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(r_m, ld_m, label="L/D", color="tab:orange")
    ax.set_title("Missile Lift-to-Drag Ratio (L/D) vs Rudder")
    ax.set_xlabel("Rudder")
    ax.set_ylabel("L/D")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(r_f, da_f, label="Drag Accel", color="tab:blue")
    ax.set_title("Fighter Drag Acceleration vs Rudder (Mach 1)")
    ax.set_xlabel("Rudder")
    ax.set_ylabel("Drag Accel (m/s^2)")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(r_f, ld_f, label="L/D", color="tab:orange")
    ax.set_title("Fighter Lift-to-Drag Ratio (L/D) vs Rudder")
    ax.set_xlabel("Rudder")
    ax.set_ylabel("L/D")
    ax.grid(True)

    fig.suptitle("Aerodynamic Curves vs Rudder", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
