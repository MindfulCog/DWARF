"""
DWARF Vortex Simulator with advanced physics and memory field integration.
"""
import numpy as np
import time
import os

# Import physics core for simulation
from physics_core import simulate_step, calculate_displacement, apply_forces, update_memory, DWARF_FORCE_EXPONENT

# Import particle types
from particle_types import generate_default_particles

class DwarfVortexSimulator:
    """
    DWARF Vortex Simulator - Core simulator class that manages the DWARF physics model.
    """
    def __init__(self, config):
        """
        Initialize the simulator with configuration.
        
        Args:
            config: SimulationConfig instance
        """
        self.config = config
        self.GRID_SIZE = getattr(config, 'GRID_SIZE', 2048)
        
        # Initialize particles
        self.particles = generate_default_particles(config)
        
        # Convert grid coordinates to centered coordinates (0,0 at center)
        for p in self.particles:
            if 'pos' in p:
                p['pos'] = p['pos'] - self.GRID_SIZE/2
        
        # Initialize electron tracking
        self.electron_position = np.zeros(3)  # 3D for compatibility
        self.electron_velocity = np.zeros(3)
        self.electron_energy = 0.0
        
        # Initialize simulation state
        self.step_count = 0
        self.current_time = 0.0
        
        # Set up log writers using logger module
        try:
            from logger import init_loggers, get_logger
            self.log_writers, self.log_files = init_loggers()
            print("Logger initialized")
        except Exception as e:
            print(f"Warning: Could not initialize loggers: {e}")
            self.log_writers = None
            self.log_files = None
    
    def step(self, dt):
        """
        Perform one simulation step.
        
        Args:
            dt: Time step
            
        Returns:
            dict: Vortex field data
        """
        # Ensure dt is a float
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            print(f"Warning: Invalid dt value '{dt}', using default from config")
            dt = float(self.config.time_step)
        
        # Get logger for physics core
        try:
            from logger import get_logger
            logger = get_logger()
        except (ImportError, NameError):
            logger = None
        
        # Run simulation step using core physics, passing logger
        success = simulate_step(self.step_count, self.particles, logs=logger)
        
        if not success:
            print(f"Simulation step {self.step_count} failed")
            
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
        
        # Track proton movement for diagnostics
        for p in self.particles:
            if p['type'] == 'proton':
                # Store previous position if not already tracked
                if not hasattr(p, 'prev_pos') or not isinstance(p['prev_pos'], np.ndarray):
                    p['prev_pos'] = p['pos'].copy()
                
                # Calculate displacement from initial position
                displacement = np.linalg.norm(p['pos'] - p['prev_pos'])
                if displacement > 0.00001:  # Movement threshold
                    print(f"Proton moved: {displacement:.8f} units")
                    p['prev_pos'] = p['pos'].copy()
        
        # Update time using the provided dt parameter (now safely cast to float)
        self.current_time += dt
        
        # Collect physics data
        physics_data = self._collect_physics_data()
        
        # Debug info
        print(f"Step {self.step_count}: Time={self.current_time:.6f}, dt={dt:.6f}")
        
        # Explicit logging of physics data - split into separate try blocks
        try:
            # First log the electron trajectory data
            from logger import log_step
            log_step(self.step_count, self.current_time, 
                  self.electron_position, self.electron_velocity, 
                  self.electron_energy)
        except (ImportError, NameError) as e:
            print(f"Warning: Could not log step data: {e}")
        
        # Use the logger object directly if available
        try:
            from logger import get_logger
            logger = get_logger()
            if logger:
                # Add all physics data to the log
                logger.log_data(physics_data)
        except (ImportError, NameError) as e:
            print(f"Warning: Could not log detailed physics data: {e}")
                
        # Increment step counter
        self.step_count += 1
        
        # Return vortex field data
        return physics_data
    
    def get_electron_position(self):
        """
        Get the current electron position.
        
        Returns:
            array: 3D position vector
        """
        return self.electron_position
        
    def get_electron_velocity(self):
        """
        Get the current electron velocity.
        
        Returns:
            array: 3D velocity vector
        """
        return self.electron_velocity
        
    def get_electron_energy(self):
        """
        Get the current electron energy.
        
        Returns:
            float: Energy value
        """
        return self.electron_energy
        
    def get_vortex_field_at(self, pos):
        """
        Calculate the vortex field strength at a point.
        
        Args:
            pos: 2D position vector
            
        Returns:
            float: Field strength
        """
        # Initialize field strength
        field_strength = 0.0
        
        # Ensure DWARF_FORCE_EXPONENT is a float
        try:
            force_exponent = float(DWARF_FORCE_EXPONENT)
        except (TypeError, ValueError):
            print(f"Warning: Invalid DWARF_FORCE_EXPONENT '{DWARF_FORCE_EXPONENT}', using default 2.0")
            force_exponent = 2.0
        
        # Sum contributions from all particles
        for p in self.particles:
            if 'pos' not in p or 'spin' not in p:
                continue
                
            # Calculate distance to particle
            r_vec = calculate_displacement(pos, p['pos'])
            r = np.linalg.norm(r_vec)
            
            # Avoid division by zero
            if r < 0.1:
                continue
                
            # Field strength using consistent force exponent
            field_strength += p['spin'] / (r**force_exponent)
            
            # Add memory field contribution
            if 'field_memory' in p:
                memory_mag = np.linalg.norm(p['field_memory'])
                field_strength += memory_mag / (r**force_exponent) * 0.01
        
        return field_strength
        
    def modify_particle_spin(self, particle_id, spin_value):
        """
        Modify a particle's spin value.
        
        Args:
            particle_id: ID of the particle
            spin_value: New spin value
        """
        for p in self.particles:
            if p.get('id', None) == particle_id:
                p['spin'] = spin_value
                print(f"Modified spin for {p['type']}_{particle_id}: {spin_value}")
                return
        
        print(f"Warning: Particle with ID {particle_id} not found")
        
    def _collect_physics_data(self):
        """
        Collect physics data for visualization and analysis.
        
        Returns:
            dict: Physics data including particles, fields and other metrics
        """
        physics_data = {
            'step': self.step_count,
            'time': self.current_time,
            'particles': []
        }
        
        # Collect particle data
        for p in self.particles:
            particle_data = {
                'type': p.get('type', 'unknown'),
                'id': p.get('id', 0),
                'pos': p.get('pos', np.zeros(2)),
                'vel': p.get('vel', np.zeros(2))
            }
            
            # Include additional data if available
            for attr in ['spin', 'charge', 'mass', 'field_memory', 'curl', 'angular_momentum',
                      'memory_gradient', 'memory_vector', 'net_force']:
                if attr in p:
                    particle_data[attr] = p[attr]
            
            physics_data['particles'].append(particle_data)
            
        # Calculate distances between particles
        physics_data['distances'] = {}
        
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i >= j:  # Skip self and avoid duplicates
                    continue
                    
                key = f"{p1['type']}_{p1.get('id', i)}_to_{p2['type']}_{p2.get('id', j)}"
                r_vec = calculate_displacement(p1['pos'], p2['pos'])
                physics_data['distances'][key] = float(np.linalg.norm(r_vec))
        
        return physics_data
        
    def get_vortex_field_data(self):
        """
        Get current vortex field data for visualization.
        
        Returns:
            dict: Vortex field data
        """
        return self._collect_physics_data()

    def __del__(self):
        """Cleanup resources on destruction."""
        # Close log files if they exist
        if hasattr(self, 'log_files') and self.log_files:
            for f in self.log_files.values():
                try:
                    f.close()
                except:
                    pass