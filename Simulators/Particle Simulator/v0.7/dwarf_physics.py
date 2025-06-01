import numpy as np
import cupy as cp
from dwarf_math import DWARFMath

class DWARFPhysics:
    """Physics engine implementing DWARF theory interactions"""
    
    def __init__(self, grid, time_step=0.01):
        self.grid = grid
        self.time_step = time_step
        self.dwarf_math = DWARFMath()
        self.bond_detector = None  # Will be initialized later
        
    def initialize(self):
        """Initialize physics engine"""
        self.dwarf_math.initialize_memory_field(self.grid.resolution)
        
    def set_bond_detector(self, bond_detector):
        """Set the bond detector instance"""
        self.bond_detector = bond_detector
        
    def calculate_particle_interactions(self, particles):
        """Calculate all particle-particle interactions"""
        n = len(particles)
        
        # Reset forces
        for particle in particles:
            particle.reset_forces()
        
        # Calculate pairwise interactions
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = particles[i], particles[j]
                
                # Skip interactions between bonded particles (handled differently)
                if p2 in p1.bonded_with:
                    continue
                
                # Calculate DWARF force
                force = self.dwarf_math.calculate_force(p1, p2)
                
                # Apply Newton's third law
                p1.apply_force(force)
                p2.apply_force(-force)
                
                # Add specific interaction effects based on particle types
                self.apply_specific_interaction(p1, p2)
    
    def apply_specific_interaction(self, p1, p2):
        """Apply specific interactions based on particle types"""
        # Determine interaction type
        if p1.particle_type == "proton" and p2.particle_type == "electron":
            self.proton_electron_interaction(p1, p2)
        elif p1.particle_type == "electron" and p2.particle_type == "proton":
            self.proton_electron_interaction(p2, p1)
        elif p1.particle_type == "proton" and p2.particle_type == "neutron":
            self.proton_neutron_interaction(p1, p2)
        elif p1.particle_type == "neutron" and p2.particle_type == "proton":
            self.proton_neutron_interaction(p2, p1)
        elif p1.particle_type == "proton" and p2.particle_type == "proton":
            self.proton_proton_interaction(p1, p2)
        elif p1.particle_type == "electron" and p2.particle_type == "electron":
            self.electron_electron_interaction(p1, p2)
        elif p1.particle_type == "neutron" and p2.particle_type == "neutron":
            self.neutron_neutron_interaction(p1, p2)
        elif (p1.particle_type == "electron" and p2.particle_type == "neutron") or \
             (p1.particle_type == "neutron" and p2.particle_type == "electron"):
            electron = p1 if p1.particle_type == "electron" else p2
            neutron = p2 if p2.particle_type == "neutron" else p1
            self.electron_neutron_interaction(electron, neutron)
    
    # All particle interaction functions remain the same (CPU-side computation)
    def proton_electron_interaction(self, proton, electron):
        # (Implementation remains unchanged - using NumPy)
        pass
        
    def proton_neutron_interaction(self, proton, neutron):
        # (Implementation remains unchanged - using NumPy)
        pass
        
    def proton_proton_interaction(self, p1, p2):
        # (Implementation remains unchanged - using NumPy)
        pass
        
    def electron_electron_interaction(self, e1, e2):
        # (Implementation remains unchanged - using NumPy)
        pass
        
    def neutron_neutron_interaction(self, n1, n2):
        # (Implementation remains unchanged - using NumPy)
        pass
        
    def electron_neutron_interaction(self, electron, neutron):
        # (Implementation remains unchanged - using NumPy)
        pass
    
    def apply_field_forces(self, particles):
        """Apply forces from the memory field to particles"""
        for particle in particles:
            # Calculate memory field force
            memory_force = self.dwarf_math.calculate_memory_field_force(particle, self.grid)
            particle.apply_force(memory_force)
            
            # Apply torque based on memory field curl
            memory_torque = self.dwarf_math.calculate_memory_torque(particle, self.grid)
            particle.apply_torque(memory_torque)
    
    def update_bonded_particles(self, particles):
        """Special physics for bonded particles (atoms)"""
        for p1 in particles:
            for p2 in p1.bonded_with:
                if p1.particle_type == "proton" and p2.particle_type == "electron":
                    proton, electron = p1, p2
                elif p1.particle_type == "electron" and p2.particle_type == "proton":
                    proton, electron = p2, p1
                else:
                    continue  # Not a proton-electron pair
                    
                # Calculate relative position and distance
                r_vec = electron.position - proton.position
                distance = np.linalg.norm(r_vec)
                
                if distance < 1e-10:
                    continue
                    
                # Unit direction vector
                r_hat = r_vec / distance
                
                # Keep electron at optimal orbital distance
                optimal_distance = 1.0  # Bohr radius
                
                # Force to maintain orbital distance
                binding_strength = 0.1
                distance_error = distance - optimal_distance
                binding_force = binding_strength * distance_error * r_hat
                
                # Apply binding force
                electron.apply_force(-binding_force)
                proton.apply_force(binding_force)
                
                # Maintain orbital velocity
                rel_vel = electron.velocity - proton.velocity
                radial_vel = np.dot(rel_vel, r_hat) * r_hat
                
                # Damping force to reduce radial velocity (keeps orbit circular)
                damping_strength = 0.05
                damping_force = -damping_strength * radial_vel
                
                electron.apply_force(damping_force)
                proton.apply_force(-damping_force)
    
    def integrate_motion(self, particles, dt):
        """Integrate particle motion using semi-implicit Euler"""
        for particle in particles:
            # First update velocity based on force
            particle.update_velocity(dt)
            
            # Then update position based on new velocity
            particle.update_position(dt)
            
            # Update spin based on torque
            particle.update_spin(dt)
    
    def update(self, particle_system, dt):
        """Main physics update function"""
        particles = particle_system.particles
        
        # Update memory field
        self.dwarf_math.update_memory_field(particles, self.grid, dt)
        
        # Calculate particle interactions
        self.calculate_particle_interactions(particles)
        
        # Apply field forces
        self.apply_field_forces(particles)
        
        # Special physics for bonded particles
        self.update_bonded_particles(particles)
        
        # Integrate motion
        self.integrate_motion(particles, dt)
        
        # Run bond detection if initialized
        if self.bond_detector:
            self.bond_detector.update()