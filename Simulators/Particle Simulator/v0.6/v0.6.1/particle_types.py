
import numpy as np
from config import PARTICLE_TYPES, GRID_SIZE, NUM_PROTONS, NUM_ELECTRONS, NUM_NEUTRONS

def create_particle(p_type, pos, vel):
    props = PARTICLE_TYPES[p_type]
    return {
        'type': p_type,
        'mass': props['mass'],
        'charge': props['charge'],
        'color': props['color'],
        'spin': props['spin'],
        'pos': np.array(pos, dtype=np.float32),
        'vel': np.array(vel, dtype=np.float32),
        'memory': np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
        'id': -1
    }

def generate_default_particles():
    particles = []
    center = GRID_SIZE // 2
    spacing = 50  # Keep particles well separated

    # Proton(s) - at center
    for i in range(NUM_PROTONS):
        p = create_particle('proton', [center, center], [0.0, 0.0])
        p['id'] = i  # Ensure ID is set
        particles.append(p)

    # Electron(s) - position further from center with initial velocity
    for i in range(NUM_ELECTRONS):
        # Position electron away from proton with significant initial velocity
        e = create_particle('electron', [center + spacing, center], [0.0, 20.0])
        e['id'] = i + NUM_PROTONS
        particles.append(e)

    # Neutron(s)
    for i in range(NUM_NEUTRONS):
        n = create_particle('neutron', [center, center + spacing], [0.0, 0.0])
        n['id'] = i + NUM_PROTONS + NUM_ELECTRONS
        particles.append(n)

    return particles