import numpy as np
import time

def add_boundary_conditions(physics_instance):
    """Add boundary checking to keep particles in view"""
    original_update = physics_instance.update_particles
    
    def update_with_boundaries(particles, dt):
        # Record start time for performance tracking
        start_time = time.time()
        
        # Call original update
        original_update(particles, dt)
        
        # Apply boundary conditions
        grid = physics_instance.grid
        boundary = grid.base_resolution * grid.cell_size / 2 * 0.95  # 95% of grid size
        
        for particle in particles:
            # Check position boundaries
            for i in range(3):
                if particle.position[i] > boundary:
                    particle.position[i] = boundary
                    particle.velocity[i] *= -0.5  # Bounce with energy loss
                elif particle.position[i] < -boundary:
                    particle.position[i] = -boundary
                    particle.velocity[i] *= -0.5  # Bounce with energy loss
            
            # Limit maximum velocity to prevent particles moving too fast
            max_vel = 2.0
            vel_magnitude = np.linalg.norm(particle.velocity)
            if vel_magnitude > max_vel:
                particle.velocity = particle.velocity * (max_vel / vel_magnitude)
        
        # Store calculation time for performance metrics
        physics_instance.last_calculation_time = time.time() - start_time
    
    # Replace update method with the bounded version
    physics_instance.update_particles = update_with_boundaries
    physics_instance.last_calculation_time = 0.0
    
    return physics_instance