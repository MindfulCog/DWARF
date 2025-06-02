class Particle:
    def __init__(self, mass, charge, position, velocity=None):
        """
        Initialize a particle with the given properties
        
        Parameters:
        - mass: float
        - charge: float
        - position: [x, y] coordinates
        - velocity: [vx, vy] velocity components, defaults to [0, 0]
        """
        self.mass = mass if mass is not None else 1.0
        self.charge = charge if charge is not None else 0.0
        self.position = position if position is not None else [0.0, 0.0]
        self.velocity = velocity if velocity is not None else [0.0, 0.0]
        self.acceleration = [0.0, 0.0]  # Initialize acceleration

    def update_position(self, dt):
        """
        Update position based on velocity and acceleration
        """
        # Update velocity based on acceleration
        self.velocity[0] += self.acceleration[0] * dt
        self.velocity[1] += self.acceleration[1] * dt
        
        # Update position based on velocity
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt

    def calculate_force(self, other_particle):
        """
        Calculate force between this particle and another particle
        """
        # Calculate distance vector between particles
        dx = other_particle.position[0] - self.position[0]
        dy = other_particle.position[1] - self.position[1]
        
        # Calculate magnitude of distance
        r_squared = dx**2 + dy**2
        if r_squared == 0:
            return [0, 0]  # Avoid division by zero
            
        r = r_squared**0.5
        
        # Calculate unit vector
        r_hat = [dx/r, dy/r]
        
        # Calculate force magnitude (simplified for now)
        force_magnitude = self.charge * other_particle.charge / r_squared
        
        # Calculate force vector
        force = [force_magnitude * r_hat[0], force_magnitude * r_hat[1]]
        return force