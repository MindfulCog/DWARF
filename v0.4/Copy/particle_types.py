
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
        'memory': np.zeros(2, dtype=np.float32),
        'id': -1
    }

def generate_default_particles():
    particles = []
    center = GRID_SIZE // 2
    spacing = 10

    # Proton(s)
    for i in range(NUM_PROTONS):
        p = create_particle('proton', [center + (i+1)*spacing, center], [0.0, 0.0])
        particles.append(p)

    # Electron(s)
    for i in range(NUM_ELECTRONS):
        e = create_particle('electron', [center - (i+1)*spacing, center], [0.0, 0.0])
        particles.append(e)

    # Neutron(s)
    for i in range(NUM_NEUTRONS):
        n = create_particle('neutron', [center, center + (i+1)*spacing], [0.0, 0.0])
        particles.append(n)

    return particles
