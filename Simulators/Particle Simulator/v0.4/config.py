
# config.py

GRID_SIZE = 256  # Simulation grid size
DT = 1e-1      # Time step for simulation

# Physics constants
DWARF_FORCE_EXPONENT = 2.22
MEMORY_DECAY = 0.001
SATURATION_LIMIT = 1000

# Particle type counts
NUM_PROTONS = 1
NUM_ELECTRONS = 1
NUM_NEUTRONS = 0

# Universal drag coefficient applied to all particles
DEFAULT_DRAG = 0.98
GLOBAL_DRAG = 0.98  # Adjust as needed for damping strength


# Particle definitions with color and mass
PARTICLE_TYPES = {
    'proton': {
        'mass': 1836,
        'charge': +1.0,
        'color': (1.0, 0.0, 0.0, 1.0),  # red
        'drag': 0.98,
        'spin': 81918  # Example: harmonic spin value from earlier tests
    },
    'electron': {
        'mass': 1,
        'charge': -1.0,
        'color': (0.0, 0.0, 1.0, 1.0),  # blue
        'drag': 0.98,
        'spin': 150000000000  # You can tweak this as needed
    },
    'neutron': {
        'mass': 1839,
        'charge': 0.0,
        'color': (0.5, 0.5, 0.5, 1.0),  # grey
        'drag': 0.98,
        'spin': 0  # neutrons can be spin-neutral for now
    }
}
