import numpy as np
from vector import Vector3

class Particle:
    """Optimized particle class"""
    
    def __init__(self, particle_id: int = 0):
        self.id = particle_id
        self.position = Vector3(0.0, 0.0, 0.0)
        self.velocity = Vector3(0.0, 0.0, 0.0)
        self.force = Vector3(0.0, 0.0, 0.0)
        self.spin = Vector3(0.0, 0.0, 0.0)
        self.mass = 1.0
        self.temperature = 300.0
        self.particle_type = "proton"
        self.material_type = 1
    
    def get_kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.velocity.magnitude() ** 2
    
    def __str__(self) -> str:
        return f"Particle(id={self.id}, pos={self.position}, vel={self.velocity})"