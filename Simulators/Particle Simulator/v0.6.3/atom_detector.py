import numpy as np
from collections import defaultdict

class AtomDetector:
    """Detects stable atomic configurations in the particle simulation."""
    
    def __init__(self):
        # Hydrogen target parameters with 10% tolerance
        self.hydrogen_params = {
            'mean_radius': {
                'target': 216.86,
                'lower': 195.17,
                'upper': 238.55
            },
            'std_dev': {
                'target': 37.19,
                'lower': 33.47,
                'upper': 40.91
            },
            'bohr_ratio': {
                'target': 216.9,
                'lower': 195.21,
                'upper': 238.59
            }
        }
        
        # History of orbit data for stability assessment
        self.orbit_history = defaultdict(lambda: [])
        self.stability_window = 60  # frames to consider for stability
        self.element_detections = {}  # Detected elements
        self.harmonic_particles = set()  # Particles in harmonic state
        
    def update(self, particles, step):
        """Update atom detection stats for this simulation step."""
        # Reset harmonic tracking each update
        self.harmonic_particles = set()
        
        # Group particles by nuclei
        nuclei = [p for p in particles if p['type'] == 'proton']
        electrons = [p for p in particles if p['type'] == 'electron']
        
        # No detection possible without both types of particles
        if not nuclei or not electrons:
            return
            
        # For each nucleus, check if electrons are forming stable orbits
        for nucleus in nuclei:
            nucleus_id = nucleus['id']
            nucleus_pos = nucleus['pos']
            
            # Find electrons that might be orbiting this nucleus
            orbiting_electrons = []
            
            for electron in electrons:
                # Calculate orbital data
                e_pos = electron['pos']
                e_vel = electron['vel']
                
                # Handle boundary conditions for correct distance calculation
                from physics_core import calculate_displacement
                rel_pos = calculate_displacement(nucleus_pos, e_pos)
                distance = np.linalg.norm(rel_pos)
                
                # Check if electron is bound to this nucleus (within reasonable distance)
                if distance < 300:  # Arbitrary binding threshold
                    # Calculate angular momentum to determine if in orbit
                    rel_vel = e_vel - nucleus.get('vel', np.array([0.0, 0.0]))
                    
                    # Cross product magnitude in 2D
                    angular_momentum = np.cross(rel_pos, rel_vel)
                    
                    # Only consider electrons with significant angular momentum
                    if abs(angular_momentum) > 0.1:
                        orbiting_electrons.append({
                            'id': electron['id'],
                            'distance': distance,
                            'angular_momentum': angular_momentum
                        })
            
            # Record orbital data for this nucleus
            if orbiting_electrons:
                # Calculate mean orbital radius and standard deviation
                distances = [e['distance'] for e in orbiting_electrons]
                mean_radius = np.mean(distances)
                std_dev = np.std(distances)
                bohr_ratio = mean_radius / 0.0529  # Bohr radius in nm
                
                # Store data for stability assessment
                orbit_data = {
                    'step': step,
                    'mean_radius': mean_radius,
                    'std_dev': std_dev,
                    'bohr_ratio': bohr_ratio,
                    'electrons': [e['id'] for e in orbiting_electrons]
                }
                
                self.orbit_history[nucleus_id].append(orbit_data)
                
                # Keep only recent history
                if len(self.orbit_history[nucleus_id]) > self.stability_window:
                    self.orbit_history[nucleus_id].pop(0)
                
                # Check for stability
                self._check_stability(nucleus_id, nucleus, orbiting_electrons)
    
    def _check_stability(self, nucleus_id, nucleus, orbiting_electrons):
        """Check if orbital configuration is stable and matches known elements."""
        history = self.orbit_history[nucleus_id]
        
        if len(history) < self.stability_window // 2:
            return  # Not enough history
        
        # Calculate stability metrics over recent history
        recent = history[-self.stability_window//2:]
        
        mean_radii = [data['mean_radius'] for data in recent]
        std_devs = [data['std_dev'] for data in recent]
        
        # Stability is low variance in orbital parameters
        radius_stability = np.std(mean_radii) / np.mean(mean_radii) if np.mean(mean_radii) > 0 else 999
        width_stability = np.std(std_devs) / np.mean(std_devs) if np.mean(std_devs) > 0 else 999
        
        # Consider orbit stable if variance is low
        is_stable = radius_stability < 0.05 and width_stability < 0.1
        
        # If stable, check if it matches known element parameters
        if is_stable:
            # Get most recent measurements
            current = recent[-1]
            mean_radius = current['mean_radius']
            std_dev = current['std_dev']
            bohr_ratio = current['bohr_ratio']
            
            # Add electrons to harmonic particles
            for e_id in current['electrons']:
                self.harmonic_particles.add(e_id)
            
            # Consider nucleus in harmonic state too
            self.harmonic_particles.add(nucleus_id)
            
            # Check if it matches hydrogen parameters
            h_params = self.hydrogen_params
            is_hydrogen = (
                h_params['mean_radius']['lower'] <= mean_radius <= h_params['mean_radius']['upper'] and
                h_params['std_dev']['lower'] <= std_dev <= h_params['std_dev']['upper'] and
                h_params['bohr_ratio']['lower'] <= bohr_ratio <= h_params['bohr_ratio']['upper']
            )
            
            if is_hydrogen:
                # Record hydrogen detection
                self.element_detections[nucleus_id] = {
                    'element': 'Hydrogen',
                    'nucleus': nucleus_id,
                    'electrons': current['electrons'],
                    'metrics': {
                        'mean_radius': mean_radius,
                        'std_dev': std_dev,
                        'bohr_ratio': bohr_ratio
                    },
                    'match_quality': self._calculate_match_quality('hydrogen', mean_radius, std_dev, bohr_ratio)
                }
    
    def _calculate_match_quality(self, element, mean_radius, std_dev, bohr_ratio):
        """Calculate how closely the orbital parameters match the element's expected values."""
        if element == 'hydrogen':
            params = self.hydrogen_params
            
            # Calculate percentage differences from targets
            radius_diff = abs(mean_radius - params['mean_radius']['target']) / params['mean_radius']['target']
            std_diff = abs(std_dev - params['std_dev']['target']) / params['std_dev']['target']
            bohr_diff = abs(bohr_ratio - params['bohr_ratio']['target']) / params['bohr_ratio']['target']
            
            # Average the differences
            avg_diff = (radius_diff + std_diff + bohr_diff) / 3
            
            # Convert to a percentage match (0-100%)
            match_quality = 100 * (1 - min(avg_diff, 0.1) / 0.1)
            return round(match_quality, 1)
        
        return 0.0
    
    def get_element_info(self):
        """Get information about detected elements for the legend."""
        return self.element_detections
    
    def is_in_harmonic_state(self, particle_id):
        """Check if a particle is in a harmonic state."""
        return particle_id in self.harmonic_particles