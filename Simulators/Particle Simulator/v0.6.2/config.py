"""
Simulation configuration for DWARF Simulator.
Generated on 2025-05-30 11:08:09
"""
import numpy as np

GRID_SIZE = 512
DT = 0.001
GLOBAL_DRAG = 0.99
SATURATION_LIMIT = 5.0
DWARF_FORCE_EXPONENT = 2.22
MEMORY_DECAY = 995.0

class SimulationConfig:
    """
    Configuration parameters for the DWARF simulation.
    """
    def __init__(self):
        self.grid_size = 512
        self.dt = 0.001
        self.global_drag = 0.99
        self.saturation_limit = 5.0
        self.dwarf_force_exponent = 2.22
        self.memory_decay = 995.0
        self.time_step = 0.001
        self.max_steps = 20000
        self.viz_interval = 30
        self.global_drag = 0.99
        self.saturation_limit = 5.0
        self.memory_decay = 995.0
        self.num_protons = 1
        self.num_electrons = 1
        self.num_neutrons = 0
        self.proton_spin = 133000.0
        self.electron_spin = -2100000000.0
        self.neutron_spin = 0
        self.initial_electron_position = [257.0, 257.0]
        self.initial_electron_velocity = [0.0, 0.0]
        self.initial_proton_position = [256.0, 256.0]