import numpy as np
from typing import Tuple, Dict, Any
import numpy.typing as npt

# Import Particle class - this is a forward reference that will be resolved at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from particle_types import Particle

def calculate_forces(particle1: 'Particle', particle2: 'Particle', 
                    force_exponent: float = 2.22,
                    saturation_limit: float = 5.0) -> Tuple[float, float]:
    """
    Calculate DWARF forces between two particles.
    The DWARF force differs from standard gravity by using a 2.22 exponent
    instead of the standard 2.0 inverse square law.
    
    Args:
        particle1: First particle
        particle2: Second particle
        force_exponent: DWARF theory force exponent (default: 2.22)
        saturation_limit: Maximum force magnitude to prevent numerical instability
        
    Returns:
        Tuple of (force_x, force_y) components
    """
    # Calculate distance vector
    dx = particle2.x - particle1.x
    dy = particle2.y - particle1.y
    
    # Softening to prevent division by zero
    softening = 1.0
    r_squared = dx**2 + dy**2 + softening**2
    r = np.sqrt(r_squared)
    
    # DWARF force with 2.22 exponent (standard gravity would use 2.0)
    # This creates the characteristic DWARF wake pattern
    force_magnitude = particle1.mass * particle2.mass / (r**force_exponent)
    
    # Spin coupling modifies the force
    spin_coupling = particle1.spin * particle2.spin
    
    # Negative spin coupling can create repulsion
    if spin_coupling < 0:
        force_magnitude *= -0.5  # Repulsive force is weaker
    
    # Apply saturation to prevent extreme forces
    force_magnitude = np.clip(force_magnitude, -saturation_limit, saturation_limit)
    
    # Calculate force components
    if r > 0:
        force_x = force_magnitude * dx / r
        force_y = force_magnitude * dy / r
    else:
        force_x, force_y = 0, 0
    
    # Spin-induced wake effect (perpendicular component)
    # This is unique to DWARF theory - creates curl in the force field
    wake_strength = 0.2 * abs(spin_coupling)
    wake_x = -force_magnitude * dy / r * wake_strength * particle1.spin
    wake_y = force_magnitude * dx / r * wake_strength * particle1.spin
    
    # Combine direct force with wake effect
    total_fx = force_x + wake_x
    total_fy = force_y + wake_y
    
    return total_fx, total_fy

def apply_memory_field(particle: 'Particle', 
                      field: npt.NDArray) -> Tuple[float, float]:
    """
    Calculate force on a particle from the memory field.
    The memory field is a core concept of DWARF theory, representing
    the "memory" of particle movements in space.
    
    Args:
        particle: Particle to calculate forces for
        field: Memory field array (x, y, density, curl_x, curl_y)
        
    Returns:
        Tuple of (force_x, force_y) from the field
    """
    # Get field dimensions
    field_h, field_w = field.shape[0:2]
    
    # Convert particle position to field coordinates
    field_x = int(particle.x * field_w / 800)  # Assuming width=800
    field_y = int(particle.y * field_h / 600)  # Assuming height=600
    
    # Ensure within field bounds
    if 0 <= field_x < field_w and 0 <= field_y < field_h:
        # Get field values at particle position
        density = field[field_y, field_x, 0]
        field_vx = field[field_y, field_x, 1]  # Vector x component
        field_vy = field[field_y, field_x, 2]  # Vector y component
        curl_x = field[field_y, field_x, 3]    # Curl x component
        curl_y = field[field_y, field_x, 4]    # Curl y component
        
        # Calculate force from field gradient
        # Particles are attracted to higher density regions
        force_scale = 0.05 * particle.mass
        
        # Check neighboring cells to calculate gradient
        gradient_x, gradient_y = 0, 0
        
        if field_x > 0:
            gradient_x -= field[field_y, field_x-1, 0]
        if field_x < field_w-1:
            gradient_x += field[field_y, field_x+1, 0]
            
        if field_y > 0:
            gradient_y -= field[field_y-1, field_x, 0]
        if field_y < field_h-1:
            gradient_y += field[field_y+1, field_x, 0]
        
        # Force from density gradient
        fx = gradient_x * force_scale
        fy = gradient_y * force_scale
        
        # Force from field curl (spin effect)
        curl_effect = 1.0 * particle.spin
        fx += curl_y * curl_effect
        fy += -curl_x * curl_effect
        
        # Interaction with field vectors
        vector_effect = 0.5
        fx += field_vx * vector_effect
        fy += field_vy * vector_effect
        
        return fx, fy
    
    # Return zero force if outside field
    return 0, 0

def calculate_field_at_point(x: float, y: float, 
                           particles: list['Particle'],
                           width: float = 800,
                           height: float = 600) -> Tuple[float, float, float]:
    """
    Calculate field properties at a specific point based on all particles.
    
    Args:
        x: X coordinate
        y: Y coordinate
        particles: List of all particles
        width: Simulation width
        height: Simulation height
        
    Returns:
        Tuple of (density, vector_x, vector_y)
    """
    density = 0
    vector_x = 0
    vector_y = 0
    
    # Normalize coordinates
    nx = x / width
    ny = y / height
    
    for particle in particles:
        # Normalize particle coordinates
        px = particle.x / width
        py = particle.y / height
        
        # Calculate distance from point to particle
        dx = nx - px
        dy = ny - py
        dist_squared = dx**2 + dy**2
        dist = np.sqrt(dist_squared) + 0.001  # Add small value to prevent division by zero
        
        # Particle contribution to field density
        contribution = particle.mass / (dist**1.5)  # Note: using 1.5 power for wider field
        density += contribution
        
        # Particle contribution to field vector
        # Vector points away from particle with positive spin, towards with negative
        if particle.spin != 0:
            spin_direction = np.sign(particle.spin)
            speed = np.sqrt(particle.vx**2 + particle.vy**2)
            
            # Scale factor based on particle properties
            scale = 0.1 * particle.mass * speed / dist
            
            # Base vectors point radially
            vx_radial = -dx / dist * scale
            vy_radial = -dy / dist * scale
            
            # Spin creates tangential component
            vx_tangential = -dy / dist * scale
            vy_tangential = dx / dist * scale
            
            # Combine based on spin
            vector_x += vx_radial + spin_direction * vx_tangential * 0.5
            vector_y += vy_radial + spin_direction * vy_tangential * 0.5
    
    return density, vector_x, vector_y