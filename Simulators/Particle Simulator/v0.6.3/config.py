"""
Simulation configuration for DWARF Simulator.
Generated on 2025-05-30 22:15:00
"""
import numpy as np

# Grid size for the simulation
GRID_SIZE = 512.0

# Physics constants
DT = 0.001
DWARF_FORCE_EXPONENT = 2.22  # Fixed: actual float value instead of string reference
GLOBAL_DRAG = 0.99
DRAG_RATE = GLOBAL_DRAG
SATURATION_LIMIT = 5.0
MEMORY_DECAY = 0.995

# Boundary handling constants
BOUNDARY_MODE = 'wrap'
BOUNDARY_WRAP = 'wrap'
BOUNDARY_DAMP = 'damp'

# Feature flags
VORTEX_FIELD_ENABLED = True

# Visualization constants
STABLE_ORBIT_WIDTH = 0.15
SPIN_ORBIT_SCALE = 0.001

class SimulationConfig:
    """
    Configuration parameters for the DWARF simulation.
    """
    def __init__(self):
        # Grid and simulation parameters
        self.grid_size = float(GRID_SIZE)
        self.time_step = float(DT)
        self.max_steps = 10000
        self.viz_interval = 10
        
        # Physics parameters
        self.force_exponent = float(DWARF_FORCE_EXPONENT)
        self.global_drag = float(GLOBAL_DRAG) 
        self.drag_rate = float(DRAG_RATE)
        self.saturation_limit = float(SATURATION_LIMIT)
        self.memory_decay = float(MEMORY_DECAY)
        
        # Boundary handling
        self.boundary_mode = BOUNDARY_MODE
        
        # Feature flags
        self.vortex_field_enabled = VORTEX_FIELD_ENABLED
        
        # Visualization constants
        self.stable_orbit_width = float(STABLE_ORBIT_WIDTH)
        self.spin_orbit_scale = float(SPIN_ORBIT_SCALE)
        
        # Particle configuration
        self.num_protons = 1
        self.num_electrons = 1
        self.num_neutrons = 0
        
        self.proton_spin = 133000.0
        self.electron_spin = -2100000000.0
        self.neutron_spin = 0
        
        # Initial particle positions and velocities
        self.initial_proton_position = [256.0, 256.0]
        self.initial_electron_position = [280.0, 256.0]
        self.initial_electron_velocity = [-20.0, 0.0]