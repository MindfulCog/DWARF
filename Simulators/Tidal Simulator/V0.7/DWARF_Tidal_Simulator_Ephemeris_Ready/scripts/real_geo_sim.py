import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("output", exist_ok=True)
os.makedirs("output/3d/real_geo", exist_ok=True)
mask_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fundy_mask_50x50.npy')
mask_path = os.path.abspath(mask_path)
from sim.tidal_core import DWARFTidalSimulator
from visuals.plot_tools import plot_displacement, plot_snapshot
from visuals.plot_3d import export_all_3d
from sim.log_dynamics import log_dynamics
from sim.ephemeris_core import get_positions_utc

from sim.ephemeris_core import get_positions_utc
from sim.ephemeris_core import get_positions_utc
class RegionalGeoSim(DWARFTidalSimulator):
    def __init__(self, region_bounds, grid_res=100, time_step=60, total_steps=200, mask_path="/mnt/data/DWARF_Tidal_Simulator/data/fundy_mask_50x50.npy"):
        self.bounds = region_bounds
        self.grid_res = grid_res
        self.time_step = time_step
        self.total_steps = total_steps
        self.rotation_rate = 7.29e-5
        self.earth_radius = 6.37e6

        # Terrain mask
        self.mask = np.load(mask_path)
        self.positions, self.velocities, self.tracers = self._initialize_grid()
        self.num_tracers = self.positions.shape[1]
        self.displacements = []

        # Body params
        self.moon_mass = 7.35e22
        self.sun_mass = 1.989e30
        self.drag_coefficient = 1e-9
        self.tracer_depths = 100 + 300 * self.mask.flatten()  # mock: 100m base + 300m if water
        self.tracer_masses = 1025 * self.tracer_depths * (1.0 / self.grid_res**2)  # rho*depth*area
        self.wake_decay_alpha = 2.0  # default to inviscid

    def _initialize_grid(self):
        xmin, xmax, ymin, ymax = self.bounds
        x = np.linspace(xmin, xmax, self.grid_res)
        y = np.linspace(ymin, ymax, self.grid_res)
        xv, yv = np.meshgrid(x, y)
        ocean_x = xv[self.mask == 1]
        ocean_y = yv[self.mask == 1]
        tracers = np.array([ocean_x, ocean_y])
        velocities = np.zeros_like(tracers)
        return tracers.copy(), velocities, tracers

    def run_step(self, step):
        # Use real ephemeris positions
        utc = (2025, 5, 1, 0, 0, step)
        moon_pos, sun_pos = get_positions_utc(*utc)

        for i in range(self.num_tracers):
            if not np.all(np.isfinite(self.positions[:, i])):
                continue  # skip corrupted tracer
            tracer = self.positions[:, i]

            # Moon force
            r_moon = moon_pos - tracer
            r_moon_mag = np.linalg.norm(r_moon)
            alpha_moon = 1.0 + (r_moon_mag > 2.5e7) * 1.0  # transition alpha: near 1.0, far 2.0
            r_moon_mag_alpha = r_moon_mag**alpha_moon + 1e-6
            moon_force = self.moon_mass / r_moon_mag_alpha * (r_moon / (r_moon_mag + 1e-6))

            # Sun force
            r_sun = sun_pos - tracer
            r_sun_mag = np.linalg.norm(r_sun)
            alpha_sun = 1.0 + (r_sun_mag > 2.5e8) * 1.0  # sun's far field triggers higher alpha
            r_sun_mag_alpha = r_sun_mag**alpha_sun + 1e-6
            sun_force = self.sun_mass / r_sun_mag_alpha * (r_sun / (r_sun_mag + 1e-6))

            # Net force
            total_force = 1e-7 * (moon_force + sun_force)

            # Drag

            # Coriolis (2D)
            vx, vy = self.velocities[:, i]
            coriolis = 2 * self.rotation_rate * np.array([-vy, vx])

            speed = norm(self.velocities[:, i])
            nonlinear_drag = -self.drag_coefficient * speed * self.velocities[:, i]
        self.tracer_depths = 100 + 300 * self.mask.flatten()  # mock: 100m base + 300m if water
        self.tracer_masses = 1025 * self.tracer_depths * (1.0 / self.grid_res**2)  # rho*depth*area
        self.wake_decay_alpha = 2.0  # default to inviscid
        resistance_force = -1e-8 * (self.positions[:, i] - self.tracers[:, i])
        self.velocities[:, i] += total_force + nonlinear_drag + coriolis + resistance_force
        self.positions[:, i] += self.velocities[:, i]

        disp = np.linalg.norm(self.positions - self.tracers, axis=0)
        if len(self.displacements) <= step:
            self.displacements.append(disp)
        else:
            self.displacements[step] = disp

def run_real_geo_sim():
    print("🌍 Running DWARF Simulation (Real Ephemeris)...")
    region = (-1e6, 1e6, -5e5, 5e5)
    sim = RegionalGeoSim(region_bounds=region, grid_res=50, total_steps=9000, mask_path=mask_path)
    for step in range(sim.total_steps):
        sim.run_step(step)
    sim.displacements = np.array(sim.displacements)
    plot_displacement(sim, filename="real_geo_displacement.png")
    plot_snapshot(sim, filename="real_geo_snapshot.png")
    export_all_3d(sim, output_dir="output/3d/real_geo")
    log_dynamics(sim, filename="real_geo_dynamics_v07.csv")
    print("✅ Real-Time Ephemeris Simulation Complete.")
if __name__ == "__main__":
    run_real_geo_sim()

