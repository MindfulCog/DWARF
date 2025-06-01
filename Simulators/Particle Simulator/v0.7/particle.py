import numpy as np

class Particle:
    """Particle class for DWARF simulation with spin properties"""
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.mass = 1.0
        self.charge = 0.0
        
        # Spin properties required for DWARF physics
        self.spin = np.zeros(3)
        self.torque = np.zeros(3)
        self.moment_of_inertia = 1.0  # Simplified as scalar