# Make sure the config.py includes the saturation limit
import numpy as np

# Grid and simulation parameters
GRID_SIZE = 2048 # Size of the simulation grid (pixels)
DT = 2e-3  # Time step for simulation
GLOBAL_DRAG = 0.9995  # Global drag coefficient (energy loss per step)
DWARF_FORCE_EXPONENT = 2.22  # DWARF model uses r^2.22 instead of r^2
SATURATION_LIMIT = 15.0  # Maximum field memory magnitude before saturation

# Particle configuration
NUM_PROTONS = 1
NUM_ELECTRONS = 1
NUM_NEUTRONS = 0

# New logging parameters
LOG_INTERVAL = 10  # Log quantum metrics every 10 steps
DETAILED_LOG_INTERVAL = 100  # Detailed logging every 100 steps
LOG_DIR = "logs/quantum_metrics"  # Directory for log files

class SimulationConfig:
    """Configuration parameters for the simulation."""
    
    def __init__(self):
        # Original DWARF simulation parameters
        self.time_step = DT
        self.max_steps = 10000
        self.viz_interval = 50
        
        # Vortex field parameters
        self.proton_mass = 1.6726219e-27  # kg
        self.electron_mass = 9.10938356e-31  # kg
        self.mass_ratio = self.proton_mass / self.electron_mass
        
        # Initial conditions
        self.initial_electron_position = [1.0, 0.0, 0.0]
        self.initial_electron_velocity = [0.0, 0.5, 0.0]
        
        # New quantum analysis parameters
        self.show_quantum_analysis = True
        self.quantum_analysis_interval = 500
        self.resonance_detection_threshold = 0.7
        self.probability_tracker_buffer_size = 10000
        
        # Logging parameters
        self.log_interval = LOG_INTERVAL
        self.detailed_log_interval = DETAILED_LOG_INTERVAL
        self.log_dir = LOG_DIR
        
# Define particle types and properties
PARTICLE_TYPES = {
    'proton': {
        'mass': 1836.0,
        'charge': 1.0,
        'spin': 133000,  # Spin value affects curl
        'color': (1.0, 0.0, 0.0, 1.0),  # RGBA
        'size': 12.0,
    },
    'electron': {
        'mass': 1.0,
        'charge': -1.0,
        'spin': -2100000000,  # Electron spin (opposite direction to proton)
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