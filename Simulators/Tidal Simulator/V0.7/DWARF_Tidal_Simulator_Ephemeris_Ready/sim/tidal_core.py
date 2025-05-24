import numpy as np

class DWARFTidalSimulator:
    def __init__(self, num_tracers=1000, time_step=60, total_steps=1000):
        self.earth_radius = 6.37e6  # meters
        self.moon_distance = 3.84e8  # meters
        self.moon_mass = 7.35e22  # kg
        self.earth_mass = 5.97e24  # kg
        self.rotation_rate = 7.29e-5  # rad/s
        self.num_tracers = num_tracers
        self.time_step = time_step
        self.total_steps = total_steps

        self.angles = np.linspace(0, 2 * np.pi, self.num_tracers)
        self.tracers = np.array([
            self.earth_radius * np.cos(self.angles),
            self.earth_radius * np.sin(self.angles)
        ])
        self.velocities = np.zeros_like(self.tracers)
        self.positions = self.tracers.copy()
        self.displacements = []

    def run_step(self, step):
        time = step * self.time_step
        moon_x = self.moon_distance * np.cos(self.rotation_rate * time)
        moon_y = self.moon_distance * np.sin(self.rotation_rate * time)
        moon_pos = np.array([moon_x, moon_y])

        for i in range(self.num_tracers):
            tracer = self.positions[:, i]
            r_vec = moon_pos - tracer
            r_mag = np.linalg.norm(r_vec)
            pressure = self.moon_mass / (r_mag**2 + 1e6)
            bulge_direction = r_vec / (r_mag + 1e-6)
            self.velocities[:, i] += 1e-7 * pressure * bulge_direction
            self.velocities[:, i] -= 1e-9 * self.positions[:, i]
            self.positions[:, i] += self.velocities[:, i]

        displacement = np.linalg.norm(self.positions - self.tracers, axis=0)
        if len(self.displacements) <= step:
            self.displacements.append(displacement)
        else:
            self.displacements[step] = displacement

    def run(self):
        for step in range(self.total_steps):
            self.run_step(step)
        self.displacements = np.array(self.displacements)
