import numpy as np
from config import GRID_SIZE, DT, DWARF_FORCE_EXPONENT, GLOBAL_DRAG
from logger import log_step

def extract_pos(p):
    if hasattr(p, 'position'):
        return p.position
    elif isinstance(p, dict) and 'position' in p:
        return p['position']
    return None

def simulate_step(step, particles, logs=None):
    update_memory(particles)
    apply_forces(particles)
    update_positions(particles)
    if logs is not None:
        log_step(step, particles, logs)

def update_memory(particles):
    # Placeholder for memory field update logic
    pass

def apply_forces(particles):
    for i, p in enumerate(particles):
        pos_p = extract_pos(p)
        if pos_p is None:
            continue

        total_force = np.array([0.0, 0.0])

        for j, other in enumerate(particles):
            if i == j:
                continue
            pos_o = extract_pos(other)
            if pos_o is None:
                continue

            dx = pos_o[0] - pos_p[0]
            dy = pos_o[1] - pos_p[1]
            r = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
            force_mag = 1.0 / r ** DWARF_FORCE_EXPONENT
            total_force[0] += force_mag * dx / r
            total_force[1] += force_mag * dy / r

        if hasattr(p, 'velocity'):
            p.velocity += total_force * DT
            p.velocity *= GLOBAL_DRAG
        elif isinstance(p, dict) and 'velocity' in p:
            p['velocity'][0] += total_force[0] * DT
            p['velocity'][1] += total_force[1] * DT
            p['velocity'][0] *= GLOBAL_DRAG
            p['velocity'][1] *= GLOBAL_DRAG

def update_positions(particles):
    for p in particles:
        if hasattr(p, 'position') and hasattr(p, 'velocity'):
            p.position += p.velocity * DT
        elif isinstance(p, dict) and 'position' in p and 'velocity' in p:
            p['position'][0] += p['velocity'][0] * DT
            p['position'][1] += p['velocity'][1] * DT
