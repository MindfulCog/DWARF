
import numpy as np

# Simulation Parameters
num_tracers = 500
num_steps = 300
tracers = np.random.uniform(-15, 15, (num_tracers, 3))
paths = np.zeros((num_steps, num_tracers, 3))

# Mass Sources Configuration
for t in range(num_steps):
    r = 15 - t * 0.07
    theta = t * 0.1
    mass_a = np.array([-r * np.cos(theta), -r * np.sin(theta), 0])
    mass_b = np.array([r * np.cos(theta), r * np.sin(theta), 0])

    for i in range(num_tracers):
        vec_to_a = mass_a - tracers[i]
        vec_to_b = mass_b - tracers[i]
        dist_a = np.linalg.norm(vec_to_a)
        dist_b = np.linalg.norm(vec_to_b)

        if dist_a > 1e-2:
            tracers[i] += 0.02 * vec_to_a / (dist_a**2)
        if dist_b > 1e-2:
            tracers[i] += 0.02 * vec_to_b / (dist_b**2)

        paths[t, i] = tracers[i]

# Save output to CSV for Blender import
for i in range(num_tracers):
    with open(f"tracer_{i}.csv", "w") as f:
        f.write("x,y,z\n")
        for t in range(num_steps):
            x, y, z = paths[t, i]
            f.write(f"{x},{y},{z}\n")
