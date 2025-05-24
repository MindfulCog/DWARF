import numpy as np
import matplotlib.pyplot as plt
from sim.tidal_core import DWARFTidalSimulator
from visuals.plot_tools import plot_displacement, plot_snapshot
from visuals.plot_3d import export_all_3d
from sim.log_dynamics import log_dynamics

class SolarEclipseSim(DWARFTidalSimulator):
    def __init__(self, num_tracers=1000, time_step=60, total_steps=200):
        super().__init__(num_tracers=num_tracers, time_step=time_step, total_steps=total_steps)
        self.sun_distance = 1.496e11  # meters
        self.sun_mass = 1.989e30  # kg

    def run_step(self, step):
        time = step * self.time_step
        moon_angle = self.rotation_rate * time
        sun_angle = moon_angle  # perfect eclipse alignment

        # Position Sun and Moon inline
        moon_x = self.moon_distance * np.cos(moon_angle)
        moon_y = self.moon_distance * np.sin(moon_angle)
        sun_x = self.sun_distance * np.cos(sun_angle)
        sun_y = self.sun_distance * np.sin(sun_angle)

        moon_pos = np.array([moon_x, moon_y])
        sun_pos = np.array([sun_x, sun_y])

        for i in range(self.num_tracers):
            tracer = self.positions[:, i]

            # Moon-induced flow
            r_moon = moon_pos - tracer
            r_moon_mag = np.linalg.norm(r_moon)
            moon_pressure = self.moon_mass / (r_moon_mag**2 + 1e6)
            moon_dir = r_moon / (r_moon_mag + 1e-6)

            # Sun-induced flow
            r_sun = sun_pos - tracer
            r_sun_mag = np.linalg.norm(r_sun)
            sun_pressure = self.sun_mass / (r_sun_mag**2 + 1e10)
            sun_dir = r_sun / (r_sun_mag + 1e-6)

            total_force = 1e-7 * (moon_pressure * moon_dir + sun_pressure * sun_dir)
            self.velocities[:, i] += total_force
            self.velocities[:, i] -= 1e-9 * self.positions[:, i]
            self.positions[:, i] += self.velocities[:, i]

        displacement = np.linalg.norm(self.positions - self.tracers, axis=0)
        if len(self.displacements) <= step:
            self.displacements.append(displacement)
        else:
            self.displacements[step] = displacement

def run_eclipse_sim():
    print("🌞 Running Solar Eclipse DWARF Simulation...")
    sim = SolarEclipseSim(num_tracers=500, total_steps=200)
    sim.run()
    plot_displacement(sim, filename="solar_eclipse_displacement.png")
    plot_snapshot(sim, filename="solar_eclipse_snapshot.png")
    export_all_3d(sim, output_dir="output/3d/solar_eclipse")
    log_dynamics(sim, filename="solar_eclipse_dynamics.csv")
    print("✅ Solar Eclipse Simulation Complete.")

if __name__ == "__main__":
    run_eclipse_sim()
