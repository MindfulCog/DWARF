import os
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(sim, mass_per_tracer=1.0):
    kinetic_energy = 0.5 * mass_per_tracer * np.sum(np.linalg.norm(sim.velocities, axis=0)**2)
    return kinetic_energy

def compute_angular_momentum(sim, mass_per_tracer=1.0):
    r = sim.positions.T
    v = sim.velocities.T
    L = np.cross(r, v)
    return mass_per_tracer * np.sum(L[:, 2])  # Z-component

def log_dynamics(sim, output_dir="output", filename="dynamics_log.csv", mass_per_tracer=1.0):
    os.makedirs(output_dir, exist_ok=True)
    num_steps = len(sim.displacements)
    kinetic = []
    angular = []
    tidal = []

    base_positions = sim.tracers.T  # shape (N, 2)

    for step in range(num_steps):
        disp = sim.displacements[step]  # shape (N,)
        direction = base_positions / (np.linalg.norm(base_positions, axis=1, keepdims=True) + 1e-9)
        delta_pos = direction * disp[:, np.newaxis]  # shape (N, 2)
        current_pos = base_positions + delta_pos
        vel = delta_pos / sim.time_step
        ke = 0.5 * mass_per_tracer * np.sum(np.linalg.norm(vel, axis=1)**2)
        height = np.mean(np.linalg.norm(delta_pos, axis=1))
        tidal.append(height)
        L = np.cross(current_pos, vel)
        Lz = mass_per_tracer * np.sum(L)

        kinetic.append(ke)
        angular.append(Lz)

    # Save to CSV
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w") as f:
        f.write("step,kinetic_energy,angular_momentum_z,tidal_height\n")
        for i in range(num_steps):
            f.write(f"{i},{kinetic[i]},{angular[i]},{tidal[i]}\n")
    print(f"Dynamics log saved to {csv_path}")

    # Plot graphs
    plt.figure(figsize=(10, 5))
    plt.plot(kinetic, label="Kinetic Energy")
    plt.plot(angular, label="Angular Momentum (Z)")
    plt.title("DWARF Tidal Energy and Angular Momentum Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dynamics_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Dynamics plot saved to {plot_path}")
