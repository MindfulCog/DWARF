from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import deque

class Particle:
    """Base class for all particles in the DWARF simulator."""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 mass: float = 1.0, spin: float = 1.0, 
                 color: str = '#FFFFFF', name: str = "Particle"):
        """
        Initialize a particle.
        
        Args:
            x: X position
            y: Y position
            vx: Initial X velocity
            vy: Initial Y velocity
            mass: Particle mass
            spin: Particle spin (-1 to 1)
            color: Display color
            name: Particle name
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = 0.0
        self.ay = 0.0
        self.mass = mass
        self.spin = spin
        self.color = color
        self.name = name
        self.initial_x = x
        self.initial_y = y
        self.initial_vx = vx
        self.initial_vy = vy
        
        # Trail for visualization
        self.trail_length = 50
        self.trail = deque(maxlen=self.trail_length)
        self.trail.append((x, y))
    
    def add_trail_point(self, x: float, y: float) -> None:
        """Add a point to the particle's trail."""
        self.trail.append((x, y))
    
    def reset_trail(self) -> None:
        """Clear the particle's trail and reset position."""
        self.trail.clear()
        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = self.initial_vx
        self.vy = self.initial_vy
        self.ax = 0.0
        self.ay = 0.0
        self.trail.append((self.x, self.y))
    
    def get_trail(self) -> Tuple[List[float], List[float]]:
        """Get lists of x and y coordinates for the trail."""
        x_trail = [point[0] for point in self.trail]
        y_trail = [point[1] for point in self.trail]
        return x_trail, y_trail


class Proton(Particle):
    """Proton particle with positive spin and charge."""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 mass: float = 1.0, spin: float = 0.8):
        """Initialize a proton with default positive spin."""
        super().__init__(x, y, vx, vy, mass, spin, '#FF5555', "Proton")
        self.charge = 1.0


class Electron(Particle):
    """Electron particle with negative spin and charge."""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 mass: float = 0.1, spin: float = -0.8):
        """Initialize an electron with default negative spin."""
        super().__init__(x, y, vx, vy, mass, spin, '#55AAFF', "Electron")
        self.charge = -1.0


class Neutron(Particle):
    """Neutron particle with zero charge but significant mass and spin."""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 mass: float = 1.0, spin: float = 0.1):
        """Initialize a neutron with small default spin."""
        super().__init__(x, y, vx, vy, mass, spin, '#AAAAAA', "Neutron")
        self.charge = 0.0