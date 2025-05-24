import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def plot_displacement(sim, output_dir="output", filename="avg_displacement.png"):
    os.makedirs(output_dir, exist_ok=True)
    avg_displacement = sim.displacements.mean(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_displacement)
    plt.title("Average Tidal Displacement of Ocean Tracers Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Average Displacement (meters)")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved displacement plot to {path}")

def plot_snapshot(sim, step=-1, output_dir="output", filename="snapshot.png"):
    os.makedirs(output_dir, exist_ok=True)
    if step == -1:
        step = len(sim.displacements) - 1
    disp = sim.displacements[step]
    angles = np.arctan2(sim.positions[1], sim.positions[0])

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(angles, disp, c=disp, cmap='plasma', edgecolor='k')
    plt.colorbar(sc, label='Displacement Magnitude (m)')
    plt.title(f"Tidal Bulge Snapshot (Step {step})")
    plt.xlabel("Angular Position (radians)")
    plt.ylabel("Radial Displacement (meters)")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved snapshot plot to {path}")
