
import numpy as np
import matplotlib.pyplot as plt

# Grid setup
grid_size = 100
rho = np.ones((grid_size, grid_size))
vx = np.zeros_like(rho)
vy = np.zeros_like(rho)
dt = 0.1
k = 1.0
gamma = 2.0
alpha = 0.5
v_max = 5.0
eps = 1e-6

blob_radius = 5
blob_density = 10.0
blob_path = [(i, grid_size // 2) for i in range(20, 80)]
blob_step = 0
tracer_spacing = 10
tracers = [[i, j] for i in range(0, grid_size, tracer_spacing) for j in range(0, grid_size, tracer_spacing)]

def inject_blob(rho, center, radius, density):
    x0, y0 = center
    for i in range(grid_size):
        for j in range(grid_size):
            if np.sqrt((i - x0)**2 + (j - y0)**2) <= radius:
                rho[i, j] += density

def laplacian(f):
    return (-4 * f + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1))

def compute_effective_pressure(rho, vx, vy):
    kinetic = 0.5 * (vx**2 + vy**2)
    pressure_term = k * rho**(gamma - 1)
    curvature_term = alpha * laplacian(rho)
    return kinetic - pressure_term + curvature_term

def compute_velocity_update(vx, vy, rho, L):
    dvx = np.zeros_like(vx)
    dvy = np.zeros_like(vy)
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            dvx[i, j] = -dt * (L[i+1, j] - L[i-1, j]) / (2 * rho[i, j])
            dvy[i, j] = -dt * (L[i, j+1] - L[i, j-1]) / (2 * rho[i, j])
    return vx + dvx, vy + dvy

def advect_density(rho, vx, vy):
    new_rho = np.copy(rho)
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            x_back = i - vx[i, j] * dt
            y_back = j - vy[i, j] * dt
            x0 = int(np.clip(x_back, 0, grid_size - 1))
            y0 = int(np.clip(y_back, 0, grid_size - 1))
            new_rho[i, j] = rho[x0, y0]
    return new_rho

def update_tracers(tracers, vx, vy):
    new_tracers = []
    for x, y in tracers:
        i, j = int(np.clip(x, 0, grid_size - 2)), int(np.clip(y, 0, grid_size - 2))
        dx = vx[i, j] * dt
        dy = vy[i, j] * dt
        new_tracers.append([x + dx, y + dy])
    return new_tracers

snapshots = []
tracer_history = []
for step in range(200):
    if blob_step < len(blob_path):
        inject_blob(rho, blob_path[blob_step], blob_radius, blob_density)
        blob_step += 1
    L = compute_effective_pressure(rho, vx, vy)
    vx, vy = compute_velocity_update(vx, vy, rho + eps, L)
    vx = np.clip(vx, -v_max, v_max)
    vy = np.clip(vy, -v_max, v_max)
    rho = advect_density(rho, vx, vy)
    tracers = update_tracers(tracers, vx, vy)
    if step % 25 == 0:
        snapshots.append(np.copy(rho))
        tracer_history.append(np.copy(tracers))

# Final snapshot visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("DWARF v0.5 Wake Snapshot (Final Frame)")
img = ax.imshow(snapshots[-1], cmap='inferno', origin='lower')
tr_x = [p[1] for p in tracer_history[-1]]
tr_y = [p[0] for p in tracer_history[-1]]
ax.plot(tr_x, tr_y, 'wo', markersize=2)
plt.colorbar(img, ax=ax, label='Density')
plt.tight_layout()
plt.show()
