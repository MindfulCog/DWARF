import numpy as np
import matplotlib.pyplot as plt
from sim.tidal_core import DWARFTidalSimulator
from visuals.plot_tools import plot_displacement, plot_snapshot
from visuals.plot_3d import export_all_3d
from sim.log_dynamics import log_dynamics

class TriWakeSim(DWARFTidalSimulator):
    def __init__(self, num_tracers=1000, time_step=60, total_steps=200):
        super().__init__(num_tracers=num_tracers, time_step=time_step, total_steps=total_steps)
        self.sun_distance = 1.496e11  # meters
        self.sun_mass = 1.989e30  # kg
        self.rogue_mass = 1.0e23  # kg
        self.rogue_orbit_radius = 1.0e9  # meters
        self.rogue_orbit_speed = 2 * np.pi / (self.total_steps * self.time_step) * 3  # artificial fast flyby

    def run_step(self, step):
        time = step * self.time_step
        moon_angle = self.rotation_rate * time
        sun_angle = moon_angle
        rogue_angle = self.rogue_orbit_speed * time

        # Positions
        moon_pos = np.array([
            self.moon_distance * np.cos(moon_angle),
            self.moon_distance * np.sin(moon_angle)
        ])
        sun_pos = np.array([
            self.sun_distance * np.cos(sun_angle),
            self.sun_distance * np.sin(sun_angle)
        ])
        rogue_pos = np.array([
            self.rogue_orbit_radius * np.cos(rogue_angle),
            self.rogue_orbit_radius * np.sin(rogue_angle)
        ])

        for i in range(self.num_tracers):
            tracer = self.positions[:, i]

            # Moon
            r_moon = moon_pos - tracer
            r_moon_mag = np.linalg.norm(r_moon)
            moon_pressure = self.moon_mass / (r_moon_mag**2 + 1e6)
            moon_dir = r_moon / (r_moon_mag + 1e-6)

            # Sun
            r_sun = sun_pos - tracer
            r_sun_mag = np.linalg.norm(r_sun)
            sun_pressure = self.sun_mass / (r_sun_mag**2 + 1e10)
            sun_dir = r_sun / (r_sun_mag + 1e-6)

            # Rogue
            r_rogue = rogue_pos - tracer
            r_rogue_mag = np.linalg.norm(r_rogue)
            rogue_pressure = self.rogue_mass / (r_rogue_mag**2 + 1e6)
            rogue_dir = r_rogue / (r_rogue_mag + 1e-6)

            # Total field influence
            total_force = 1e-7 * (moon_pressure * moon_dir + sun_pressure * sun_dir + rogue_pressure * rogue_dir)
            self.velocities[:, i] += total_force
            self.velocities[:, i] -= 1e-9 * self.positions[:, i]
            self.positions[:, i] += self.velocities[:, i]

        displacement = np.linalg.norm(self.positions - self.tracers, axis=0)
        if len(self.displacements) <= step:
            self.displacements.append(displacement)
        else:
            self.displacements[step] = displacement

def run_tri_wake_sim():
    print("☄️ Running Tri-Wake DWARF Simulation...")
    sim = TriWakeSim(num_tracers=500, total_steps=200)
    sim.run()
    plot_displacement(sim, filename="tri_wake_displacement.png")
    plot_snapshot(sim, filename="tri_wake_snapshot.png")
    export_all_3d(sim, output_dir="output/3d/tri_wake")
    log_dynamics(sim, filename="tri_wake_dynamics.csv")
    print("✅ Tri-Wake Simulation Complete.")

if __name__ == "__main__":
    run_tri_wake_sim()
