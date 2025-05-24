# DWARF Particle Emergence Simulation (PyTorch Version)
# Author: Tyler Nagel & Caelum

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Simulation Config
GRID_SIZE = 64
DX = 1.0
DT = 0.01
STEPS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Physical Parameters
nu = 0.01    # Viscosity
beta = 0.001 # Damping
k = 1.0      # Potential coefficient
n = 2.0      # Potential exponent

# Create 3D grid
def create_grid(size):
    x = torch.linspace(-size//2, size//2, size)
    grid = torch.meshgrid(x, x, x, indexing='ij')
    return grid

X, Y, Z = create_grid(GRID_SIZE)
X, Y, Z = X.to(DEVICE), Y.to(DEVICE), Z.to(DEVICE)

# Initialize density rho and velocity field u = (ux, uy, uz)
rho = torch.zeros_like(X)

# Define multiple vortices
vortex_params = [
    (0.0, 0.0, 0.0, 1.0),    # (x0, y0, z0, strength)
    (10.0, 10.0, 0.0, -1.0), # Second vortex with opposite rotation
]

for x0, y0, z0, strength in vortex_params:
    sigma = 5.0
    blob = torch.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))
    rho += blob.to(DEVICE)

ux = torch.zeros_like(X)
uy = torch.zeros_like(Y)
uz = torch.zeros_like(Z)

for x0, y0, z0, strength in vortex_params:
    ux += -0.1 * strength * (Y - y0)
    uy +=  0.1 * strength * (X - x0)

# Define gradient and Laplacian functions
def gradient(f):
    return torch.gradient(f, spacing=(DX, DX, DX), dim=(0,1,2))

def laplacian(f):
    return sum(torch.gradient(torch.gradient(f, spacing=(DX,), dim=(d,))[0], spacing=(DX,), dim=(d,))[0] for d in range(3))

# Diagnostic logs
energy_log = []
enstrophy_log = []
radius_log = []
peak_density_log = []

# Tracer particles (x, y, z positions)
num_tracers = 100
tracers = torch.rand(num_tracers, 3).to(DEVICE) * GRID_SIZE

# Time evolution loop
for step in tqdm(range(STEPS)):
    with torch.no_grad():
        # Compute potential and its gradient
        Phi = k * rho**n
        dPhidx, dPhidy, dPhidz = gradient(Phi)

        # Compute convective derivatives
        conv_x = ux * gradient(ux)[0] + uy * gradient(ux)[1] + uz * gradient(ux)[2]
        conv_y = ux * gradient(uy)[0] + uy * gradient(uy)[1] + uz * gradient(uy)[2]
        conv_z = ux * gradient(uz)[0] + uy * gradient(uz)[1] + uz * gradient(uz)[2]

        # Compute viscous terms
        lap_ux = laplacian(ux)
        lap_uy = laplacian(uy)
        lap_uz = laplacian(uz)

        # Update velocity fields
        ux += DT * (-conv_x - dPhidx + nu * lap_ux - beta * ux)
        uy += DT * (-conv_y - dPhidy + nu * lap_uy - beta * uy)
        uz += DT * (-conv_z - dPhidz + nu * lap_uz - beta * uz)

        # Update density via continuity equation
        div_u = gradient(ux)[0] + gradient(uy)[1] + gradient(uz)[2]
        rho -= DT * (rho * div_u)
        rho = torch.clamp(rho, min=0.0)

        # Update tracers (nearest grid velocity)
        ti = tracers.long().clamp(0, GRID_SIZE-1)
        tracer_vel = torch.stack([ux[ti[:,0],ti[:,1],ti[:,2]],
                                  uy[ti[:,0],ti[:,1],ti[:,2]],
                                  uz[ti[:,0],ti[:,1],ti[:,2]]], dim=1)
        tracers += tracer_vel * DT
        tracers = torch.remainder(tracers, GRID_SIZE)  # wrap-around

        # Diagnostics
        energy = 0.5 * (ux**2 + uy**2 + uz**2).sum().item()
        vorticity = (gradient(uy)[0] - gradient(ux)[1])**2
        enstrophy = vorticity.sum().item()
        rms_radius = torch.sqrt(((X**2 + Y**2 + Z**2) * rho).sum() / rho.sum()).item()
        peak_density = rho.max().item()

        energy_log.append(energy)
        enstrophy_log.append(enstrophy)
        radius_log.append(rms_radius)
        peak_density_log.append(peak_density)

        # Visualize slice every 100 steps
        if step % 100 == 0:
            plt.imshow(rho[:, :, GRID_SIZE//2].cpu().numpy(), cmap='inferno')
            plt.title(f"Density Slice at Step {step}")
            plt.colorbar()
            plt.savefig(f"rho_slice_{step:04d}.png")
            plt.close()

            # Tracer scatter
            plt.scatter(tracers[:,0].cpu(), tracers[:,1].cpu(), c='cyan', s=2)
            plt.xlim(0, GRID_SIZE)
            plt.ylim(0, GRID_SIZE)
            plt.title(f"Tracer Positions at Step {step}")
            plt.savefig(f"tracers_{step:04d}.png")
            plt.close()

# Save diagnostic logs
np.save("energy_log.npy", energy_log)
np.save("enstrophy_log.npy", enstrophy_log)
np.save("radius_log.npy", radius_log)
np.save("peak_density_log.npy", peak_density_log)
