import numpy as np
import cupy as cp
from dwarf_math import DWARFMath

class DWARFPhysics:
    """Physics engine for DWARF particle simulator"""
    
    def __init__(self, grid, time_step=0.01):
        self.grid = grid
        self.time_step = time_step
        self.dwarf_math = DWARFMath()
        self.bond_detector = None
        
    def set_bond_detector(self, bond_detector):
        self.bond_detector = bond_detector
        
    def initialize(self):
        """Initialize the physics engine"""
        # Initialize DWARF math components
        # FIXED: use base_resolution instead of resolution
        self.dwarf_math.initialize_memory_field(self.grid.base_resolution)
        
    def update(self, particle_system, dt):
        """Update physics for all particles"""
        # First update memory field from particles
        self.dwarf_math.update_memory_field(particle_system.particles, self.grid, dt)
        
        # Calculate all forces
        self._calculate_forces(particle_system)
        
        # Update particles
        self._update_particles(particle_system, dt)
        
        # Detect and maintain bonds
        if self.bond_detector:
            self.bond_detector.update()
            
    def _calculate_forces(self, particle_system):
        """Calculate all forces on particles"""
        num_particles = len(particle_system.particles)
        
        # Reset all forces
        for i in range(num_particles):
            particle_system.particles[i].force = np.zeros(3)
            particle_system.particles[i].torque = np.zeros(3)
        
        # Calculate particle-particle interactions
        for i in range(num_particles):
            p1 = particle_system.particles[i]
            
            # Apply memory field force
            memory_force = self.dwarf_math.calculate_memory_field_force(p1, self.grid)
            p1.force += memory_force
            
            # Apply memory field torque - affects spin dynamics
            memory_torque = self.dwarf_math.calculate_memory_torque(p1, self.grid)
            p1.torque += memory_torque
            
            # Calculate forces between particles
            for j in range(i+1, num_particles):
                p2 = particle_system.particles[j]
                
                # Calculate core DWARF force between particles
                force = self.dwarf_math.calculate_force(p1, p2)
                
                # Apply force to both particles (Newton's third law)
                p1.force += force
                p2.force -= force
                
                # Calculate spin-coupling torque
                # This creates spin-orbit coupling effects
                r_vec = p2.position - p1.position
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:
                    r_hat = r_vec / r
                    
                    # Spin-coupling strength decreases with distance
                    coupling_strength = 0.01 / (r * r)
                    
                    # Torque tries to align spins with each other
                    # and with orbital angular momentum
                    torque1 = coupling_strength * np.cross(p1.spin, p2.spin)
                    torque2 = coupling_strength * np.cross(p1.spin, r_hat)
                    
                    p1.torque += torque1 + torque2
                    p2.torque -= torque1
                    p2.torque += coupling_strength * np.cross(p2.spin, r_hat)
        
        # Apply bond forces - bonds act like springs
        if hasattr(particle_system, 'bonds'):
            for bond in particle_system.bonds:
                p1, p2 = bond
                
                # Bond spring force
                r_vec = p2.position - p1.position
                r = np.linalg.norm(r_vec)
                
                # Determine equilibrium distance based on particle types
                if p1.particle_type == 'proton' and p2.particle_type == 'electron':
                    # Hydrogen-like bond - equilibrium at 0.5 Bohr radius
                    r0 = 0.5
                else:
                    # Other bonds - equilibrium at 0.8
                    r0 = 0.8
                    
                # Hooke's law spring force
                k = 0.5  # Spring constant
                spring_force = k * (r - r0) * r_vec / r
                
                # Apply to both particles
                p1.force += spring_force
                p2.force -= spring_force
    
    def _update_particles(self, particle_system, dt):
        """Update particle positions and velocities"""
        for particle in particle_system.particles:
            # Apply force to update velocity
            acceleration = particle.force / particle.mass
            particle.velocity += acceleration * dt
            
            # Apply drag force - simplifies the simulation
            particle.velocity *= (1.0 - 0.01 * dt)
            
            # Apply damping to spin
            particle.spin *= (1.0 - 0.005 * dt)
            
            # Apply torque to update spin
            # Precession and nutation of spin vector
            spin_change = np.cross(particle.spin, particle.torque) * dt
            particle.spin += spin_change
            
            # Renormalize spin vector (maintain unit length)
            spin_magnitude = np.linalg.norm(particle.spin)
            if spin_magnitude > 1e-10:
                particle.spin = particle.spin / spin_magnitude
            
            # Update position with velocity
            particle.position += particle.velocity * dt
            
            # Apply position constraints - keep in simulation bounds
            grid_half_size = self.grid.size / 2.0
            for i in range(3):
                # Bounce off boundaries with loss
                if particle.position[i] > grid_half_size - 0.1:
                    particle.position[i] = grid_half_size - 0.1
                    particle.velocity[i] *= -0.8  # Energy loss on bounce
                    
                elif particle.position[i] < -grid_half_size + 0.1:
                    particle.position[i] = -grid_half_size + 0.1
                    particle.velocity[i] *= -0.8  # Energy loss on bounce