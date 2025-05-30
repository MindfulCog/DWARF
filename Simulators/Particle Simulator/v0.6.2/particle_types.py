"""
Particle type definitions for DWARF simulator.
"""
import numpy as np

def generate_default_particles(config=None):
    """
    Generate the default set of particles based on configuration.
    
    Args:
        config: SimulationConfig instance (optional)
    
    Returns:
        list: List of particle dictionaries
    """
    particles = []
    
    # Use defaults if no config provided
    if config is None:
        # Default values
        grid_size = 2048
        num_protons = 1
        num_electrons = 1
        num_neutrons = 0
        proton_spin = 1000000
        electron_spin = -1000000
        neutron_spin = 0
        initial_proton_position = [grid_size/2, grid_size/2]
        initial_electron_position = [grid_size/2, grid_size/2 + 10]
        initial_electron_velocity = [0.0, 5.0]
    else:
        # Use configuration values
        grid_size = getattr(config, 'GRID_SIZE', 2048)
        num_protons = getattr(config, 'num_protons', 1)
        num_electrons = getattr(config, 'num_electrons', 1)
        num_neutrons = getattr(config, 'num_neutrons', 0)
        proton_spin = getattr(config, 'proton_spin', 1000000)
        electron_spin = getattr(config, 'electron_spin', -1000000)
        neutron_spin = getattr(config, 'neutron_spin', 0)
        initial_proton_position = getattr(config, 'initial_proton_position', [grid_size/2, grid_size/2])
        initial_electron_position = getattr(config, 'initial_electron_position', [grid_size/2, grid_size/2 + 10])
        initial_electron_velocity = getattr(config, 'initial_electron_velocity', [0.0, 5.0])
    
    # Generate protons
    for i in range(num_protons):
        # For multiple protons, space them out
        offset = i * 20 if num_protons > 1 else 0
        pos = np.array([initial_proton_position[0] + offset, initial_proton_position[1]])
        
        particles.append({
            'type': 'proton',
            'id': i,
            'pos': pos,
            'vel': np.array([0.0, 0.0]),
            'mass': 1836.15, # Proton mass relative to electron
            'charge': 1.0,
            'spin': proton_spin,
            'field_memory': np.zeros(2),
        })
    
    # Generate electrons
    for i in range(num_electrons):
        # For multiple electrons, space them out
        offset = i * 10 if num_electrons > 1 else 0
        pos = np.array([initial_electron_position[0] + offset, initial_electron_position[1]])
        
        particles.append({
            'type': 'electron',
            'id': i,
            'pos': pos,
            'vel': np.array(initial_electron_velocity),
            'mass': 1.0,  # Reference mass
            'charge': -1.0,
            'spin': electron_spin,
            'field_memory': np.zeros(2),
        })
    
    # Generate neutrons
    for i in range(num_neutrons):
        # Place neutrons near protons
        offset = i * 5 if num_neutrons > 1 else 0
        pos = np.array([initial_proton_position[0] + offset + 5, initial_proton_position[1]])
        
        particles.append({
            'type': 'neutron',
            'id': i,
            'pos': pos,
            'vel': np.array([0.0, 0.0]),
            'mass': 1838.68, # Neutron mass relative to electron
            'charge': 0.0,
            'spin': neutron_spin,
            'field_memory': np.zeros(2),
        })
    
    return particles