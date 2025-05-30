"""
Enhanced wrapper class for DWARF physics core with comprehensive logging and visualization.
Includes additional diagnostics for proton dynamics.
"""
import numpy as np
from physics_core import simulate_step, calculate_displacement, apply_forces, update_memory
from particle_types import generate_default_particles
from logger import log_step, log_event
from config import SimulationConfig

class DwarfVortexSimulator:
    """
    Wrapper for DWARF physics core that provides an object-oriented interface.
    This version captures and exposes all core physics parameters.
    """
    def __init__(self, config):
        """
        Initialize the DWARF vortex simulator.
        
        Args:
            config: SimulationConfig instance containing simulation parameters
        """
        # Store configuration
        self.config = config
        
        # Initialize grid size
        self.GRID_SIZE = getattr(config, 'GRID_SIZE', 2048)
        
        # Initialize particles
        self.particles = generate_default_particles()
        
        # Center particles relative to origin (0,0) instead of grid coordinates
        for p in self.particles:
            if 'pos' in p:
                # Adjust position to be centered at origin
                p['pos'] = p['pos'] - np.array([self.GRID_SIZE/2, self.GRID_SIZE/2])
        
        # Add electron position and velocity tracking
        # Note: DWARF core uses 2D vectors, but we'll store 3D for quantum effects
        self.electron_position = np.array([config.initial_electron_position[0] - self.GRID_SIZE/2, 
                                          config.initial_electron_position[1] - self.GRID_SIZE/2, 
                                          0.0])  # Add z=0 for 3D
        self.electron_velocity = np.array([config.initial_electron_velocity[0],
                                          config.initial_electron_velocity[1],
                                          0.0])  # Add z=0 for 3D
        
        # Initialize electron energy
        self.electron_energy = 0.0
        
        # Set up electron particle (required for physics_core)
        for p in self.particles:
            if p['type'] == 'electron':
                # Use only X,Y components for DWARF physics
                p['pos'] = np.array([self.electron_position[0], self.electron_position[1]])
                p['vel'] = np.array([self.electron_velocity[0], self.electron_velocity[1]])
                break
                
        # Track simulation step and time
        self.step_count = 0
        self.current_time = 0.0
        
        # Initialize vortex field visualization data
        self.vortex_field_data = {}
        
        # Initialize particle movement tracking
        self.particle_prev_pos = {}
        for p in self.particles:
            p_id = p.get('id', -1)
            self.particle_prev_pos[p_id] = p['pos'].copy() if 'pos' in p else np.array([0, 0])
        
        log_event("INIT", "Initialized DwarfVortexSimulator wrapper")
        
    def step(self, dt):
        """
        Perform one simulation step.
        
        Args:
            dt: Time step
            
        Returns:
            dict: Vortex field data
        """
        # Run simulation step using core physics
        success = simulate_step(self.step_count, self.particles)
        
        if not success:
            print(f"Simulation step {self.step_count} failed")
            
        # Track particle dynamics, especially proton movement
        self._track_particle_dynamics()
            
        # Update tracked electron position and velocity
        electron_particle = None
        proton_particle = None
        
        # Find electron and proton
        for p in self.particles:
            if p['type'] == 'electron':
                electron_particle = p
            elif p['type'] == 'proton':
                proton_particle = p
        
        if electron_particle:
            # Get 2D position and velocity from physics core
            pos_2d = electron_particle['pos']
            vel_2d = electron_particle['vel']
            
            # Update 3D versions (keeping z=0)
            self.electron_position = np.array([pos_2d[0], pos_2d[1], 0.0])
            self.electron_velocity = np.array([vel_2d[0], vel_2d[1], 0.0])
            
            # Calculate energy (kinetic + potential)
            if proton_particle:
                r_vec = calculate_displacement(proton_particle['pos'], pos_2d)
                r = np.linalg.norm(r_vec)
                if r > 0:
                    # Coulomb potential (in eV)
                    self.electron_energy = -13.6 / r  # Simplified potential energy in eV
        
        # Update time
        self.current_time += dt
        
        # Collect comprehensive physics data for logging
        physics_data = self._collect_physics_data()
        
        # Log the step with comprehensive physics data
        log_step(self.step_count, self.current_time, 
                 self.electron_position, self.electron_velocity, 
                 self.electron_energy)
                
        # Log detailed physics data
        from logger import log_physics_data
        log_physics_data(self.step_count, self.current_time, physics_data)
                
        # Increment step counter
        self.step_count += 1
        
        # Return vortex field data
        return physics_data
    
    def _track_particle_dynamics(self):
        """Track and report significant particle movement, especially protons."""
        for p in self.particles:
            p_id = p.get('id', -1)
            p_type = p.get('type', 'unknown')
            
            # Skip if we don't have position data
            if 'pos' not in p:
                continue
                
            # Get previous position
            prev_pos = self.particle_prev_pos.get(p_id, np.array([0, 0]))
            
            # Calculate displacement
            displacement = np.linalg.norm(p['pos'] - prev_pos)
            
            # Set threshold based on particle type
            threshold = 0.00001 if p_type == 'proton' else 0.1
            
            # Log significant movement
            if displacement > threshold:
                # For protons, we want to know about any movement
                if p_type == 'proton':
                    log_event("DYNAMICS", f"Proton {p_id} moved: {displacement:.8f} units")
                    print(f"Proton {p_id} moved: {displacement:.8f} units")
                
                # Update previous position
                self.particle_prev_pos[p_id] = p['pos'].copy()
        
    def _collect_physics_data(self):
        """
        Collect comprehensive physics data from all particles.
        
        Returns:
            dict: Physics data for visualization and logging
        """
        physics_data = {
            'step': self.step_count,
            'time': self.current_time,
            'particles': []
        }
        
        # Collect detailed particle data
        for p in self.particles:
            particle_data = {
                'type': p['type'],
                'id': p.get('id', -1),
                'pos': p['pos'].copy(),
                'vel': p['vel'].copy() if 'vel' in p else np.array([0, 0]),
                'angular_momentum': p.get('angular_momentum', 0),
                'curl': p.get('curl', 0),
                'spin': p.get('spin', 0),
                'net_force': p.get('net_force', np.array([0, 0])).copy() if 'net_force' in p else np.array([0, 0]),
            }
            
            # Memory-related fields
            if 'field_memory' in p:
                particle_data['field_memory'] = p['field_memory'].copy()
            if 'memory_gradient' in p:
                particle_data['memory_gradient'] = p['memory_gradient'].copy()
            if 'memory_vector' in p:
                particle_data['memory_vector'] = p['memory_vector'].copy()
            
            # Shell data if available
            if 'shell_data' in p:
                particle_data['shell_data'] = p['shell_data'].copy()
                
            physics_data['particles'].append(particle_data)
            
        # Calculate distances between particles
        distances = {}
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i < j:  # Avoid duplicates
                    key = f"{p1.get('id', i)}-{p2.get('id', j)}"
                    r_vec = calculate_displacement(p1['pos'], p2['pos'])
                    distances[key] = np.linalg.norm(r_vec)
        
        physics_data['distances'] = distances
        
        # Store vortex field data for visualization
        self.vortex_field_data = physics_data
        
        return physics_data
    
    def get_electron_position(self):
        """
        Get the current position of the electron.
        
        Returns:
            ndarray: Position vector (3D)
        """
        return self.electron_position
        
    def get_electron_velocity(self):
        """
        Get the current velocity of the electron.
        
        Returns:
            ndarray: Velocity vector (3D)
        """
        return self.electron_velocity
        
    def get_electron_energy(self):
        """
        Get the current energy of the electron.
        
        Returns:
            float: Energy in eV
        """
        return self.electron_energy
    
    def get_vortex_field_data(self):
        """
        Get the current vortex field data.
        
        Returns:
            dict: Vortex field data
        """
        return self.vortex_field_data
    
    def get_vortex_field_at(self, position):
        """
        Get the vortex field strength at the specified position.
        
        Args:
            position: Position vector
            
        Returns:
            float: Field strength
        """
        # This is a simplification - a real implementation would use
        # the physics core to calculate the field strength
        
        # Convert to 2D position for DWARF physics if needed
        if len(position) > 2:
            position_2d = position[:2]
        else:
            position_2d = position
            
        field_strength = 0.0
        
        # Accumulate field contributions from all particles
        for p in self.particles:
            if 'pos' in p and 'spin' in p and p['spin'] != 0:
                r_vec = calculate_displacement(p['pos'], position_2d)
                r = np.linalg.norm(r_vec)
                if r > 0:
                    # Use DWARF field formula based on spin and memory
                    field_contribution = abs(p['spin']) / (10000 * r**2.0)
                    
                    # Add memory contribution if available
                    if 'field_memory' in p:
                        memory_mag = np.linalg.norm(p['field_memory'])
                        field_contribution *= (1.0 + memory_mag * 0.1)
                        
                    field_strength += field_contribution
                    
        return field_strength
    
    def modify_particle_spin(self, particle_id, new_spin):
        """
        Modify the spin of a particle.
        
        Args:
            particle_id: ID of the particle to modify
            new_spin: New spin value
            
        Returns:
            bool: True if particle found and modified
        """
        for p in self.particles:
            if p.get('id', -1) == particle_id:
                p['spin'] = new_spin
                log_event("SPIN_MODIFIED", f"Set particle {particle_id} spin to {new_spin}")
                return True
        
        return False