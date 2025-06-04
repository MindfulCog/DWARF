
# dwarf_atom_2d_sim.py

import numpy as np
import csv

# === PARAMETERS ===

# Domain
DOMAIN_SIZE = 100.0  # Bohr radii
DT = 1e-18  # seconds
NUM_STEPS = 1_000_000

# Proton
PROTON_MASS = 1836.0
PROTON_POSITION = np.array([DOMAIN_SIZE / 2, DOMAIN_SIZE / 2])
PROTON_SPIN_RPM = 81918

# Electron
ELECTRON_MASS = 1.0
ELECTRON_POSITION = PROTON_POSITION + np.array([5.0, 0.0])  # start 5 Bohr away
ELECTRON_VELOCITY = np.array([0.0, 0.0])  # start at rest
PROTON_SPIN_RPM = 2100000000

# DWARF Force
K_DWARF = 5e6
FORCE_EXPONENT = 2.22

# Memory
TAU_M = 1e-15  # seconds
memory_value = 1.0

# Wake
A_WAKE = 1.0
L_WAKE = 2.0  # Bohr radii

# Drag
MU = 0.005

# Repulsion
K_REPEL = 1e7
REPEL_N = 8

# === LOGGING SETUP ===

positions = []
velocities = []
forces = []
energies = []
angular_momentum = []
memory_log = []

# === SIMULATION LOOP ===

electron_pos = ELECTRON_POSITION.copy()
electron_vel = ELECTRON_VELOCITY.copy()
time = 0.0

for step in range(NUM_STEPS):
    # Distance
    r_vec = electron_pos - PROTON_POSITION
    r = np.linalg.norm(r_vec)
    if r < 1e-5:
        r = 1e-5  # prevent div by zero
    
    r_hat = r_vec / r
    
    # DWARF force
    F_dwarf_mag = K_DWARF / r**FORCE_EXPONENT
    F_dwarf = -F_dwarf_mag * r_hat
    
    # Memory decay
    memory_value *= np.exp(-DT / TAU_M)
    F_memory = memory_value
    
    # Wake boost
    wake_boost = A_WAKE * np.exp(-r / L_WAKE)
    
    # Drag
    F_drag = -MU * electron_vel
    
    # Repulsion
    F_repel_mag = K_REPEL / r**REPEL_N
    F_repel = F_repel_mag * r_hat
    
    # Total force
    F_total = (F_dwarf * F_memory * wake_boost) + F_drag + F_repel
    
    # Update velocity
    electron_vel += (F_total / ELECTRON_MASS) * DT
    
    # Update position
    electron_pos += electron_vel * DT
    
    # Update time
    time += DT
    
    # === LOG DATA ===
    positions.append([time, electron_pos[0], electron_pos[1]])
    velocities.append([time, electron_vel[0], electron_vel[1]])
    forces.append([time, F_total[0], F_total[1]])
    
    kinetic_energy = 0.5 * ELECTRON_MASS * np.dot(electron_vel, electron_vel)
    potential_energy = -K_DWARF / (r ** (FORCE_EXPONENT - 1))  # rough potential
    total_energy = kinetic_energy + potential_energy
    
    energies.append([time, kinetic_energy, potential_energy, total_energy])
    
    # Angular momentum (scalar)
    L = ELECTRON_MASS * (electron_pos[0] * electron_vel[1] - electron_pos[1] * electron_vel[0])
    angular_momentum.append([time, L])
    
    # Memory field
    memory_log.append([time, memory_value])
    
# === SAVE LOGS ===

def save_csv(filename, data, headers):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

save_csv("positions.csv", positions, ["time", "x", "y"])
save_csv("velocities.csv", velocities, ["time", "v_x", "v_y"])
save_csv("forces.csv", forces, ["time", "F_x", "F_y"])
save_csv("energies.csv", energies, ["time", "KE", "PE", "Total_E"])
save_csv("angular_momentum.csv", angular_momentum, ["time", "L"])
save_csv("memory_field.csv", memory_log, ["time", "memory_value"])

print("Simulation complete. Logs saved.")
