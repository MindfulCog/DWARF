# Make sure the config.py includes the saturation limit
import numpy as np

# Grid and simulation parameters
GRID_SIZE = 2048 # Size of the simulation grid (pixels)
DT = 3.6e-3  # Time step for simulation
GLOBAL_DRAG = 0.9985  # Global drag coefficient (energy loss per step)
DWARF_FORCE_EXPONENT = 2.22  # DWARF model uses r^2.22 instead of r^2
SATURATION_LIMIT = 10.0  # Maximum field memory magnitude before saturation

# Particle configuration
NUM_PROTONS = 1
NUM_ELECTRONS = 1
NUM_NEUTRONS = 0

# Define particle types and properties
PARTICLE_TYPES = {
    'proton': {
        'mass': 1836.0,
        'charge': 1.0,
        'spin': 420000,  # Spin value affects curl
        'color': (1.0, 0.0, 0.0, 1.0),  # RGBA
        'size': 12.0,
    },
    'electron': {
        'mass': 1.0,
        'charge': -1.0,
        'spin': -2220000000,  # Electron spin (opposite direction to proton)
        'color': (0.0, 0.0, 1.0, 1.0),  # RGBA
        'size': 8.0,
    },
    'neutron': {
        'mass': 1836.0,
        'charge': 0.0,
        'spin': 81918,  # Similar to proton
        'color': (0.5, 0.5, 0.5, 1.0),  # RGBA
        'size': 10.0,
    }
}