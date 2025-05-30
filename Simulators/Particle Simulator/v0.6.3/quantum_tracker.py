"""
Quantum Tracker for DWARF Simulator.
Tracks the emergent quantum behavior in the vortex field.
"""
import numpy as np
import math

class TorsionAnalyzer:
    """
    Analyzes torsion effects in the electron trajectory.
    Used for spin-orbit coupling detection.
    """
    def __init__(self):
        """Initialize the torsion analyzer."""
        self.position_history = []
        self.velocity_history = []
        self.max_history = 100  # Keep last 100 points
        
    def add_point(self, position, velocity):
        """
        Add a new data point to the analyzer.
        
        Args:
            position: Position vector
            velocity: Velocity vector
        """
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        
        # Keep history within size limit
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            
    def analyze_spin_orbit_coupling(self):
        """
        Analyze the trajectory for signs of spin-orbit coupling.
        
        Returns:
            dict: Information about detected spin-orbit coupling
        """
        # Need at least 3 points for meaningful analysis
        if len(self.position_history) < 3:
            return {'detected': False}
            
        # Simple detection of orbital precession
        try:
            # Calculate orbital plane normal and rate of change
            normals = []
            for i in range(len(self.position_history) - 2):
                p0 = self.position_history[i]
                p1 = self.position_history[i+1]
                p2 = self.position_history[i+2]
                
                # Create two vectors in orbital plane
                v1 = p1 - p0
                v2 = p2 - p1
                
                # Cross product gives normal to plane
                normal = np.cross(v1, v2)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                    normals.append(normal)
            
            if len(normals) < 2:
                return {'detected': False}
                
            # Calculate angle between first and last normal vector
            first_normal = normals[0]
            last_normal = normals[-1]
            dot_product = np.clip(np.dot(first_normal, last_normal), -1.0, 1.0)
            angle = math.acos(dot_product)
            
            # Calculate precession rate
            precession_rate = angle / (len(normals) - 1)
            
            return {
                'detected': precession_rate > 0.01,  # Arbitrary threshold
                'angle': angle,
                'precession_rate': precession_rate
            }
            
        except (IndexError, ValueError, ZeroDivisionError):
            return {'detected': False}


class EmergentQuantumTracker:
    """
    Tracks emergent quantum behavior in classical electron trajectories.
    """
    def __init__(self):
        """Initialize the quantum tracker."""
        # Track position, velocity, energy
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.energy = 0.0
        
        # Quantum state information
        self.quantum_state = {
            'n': 1,  # Principal quantum number
            'l': 0,  # Angular momentum quantum number
            'ml': 0,  # Magnetic quantum number
            'm': 0,   # Alias for ml
            'j': 0.5, # Total angular momentum
            'energy': 0.0,
            'orbital_type': 's'
        }
        
        # Analyzer for spin-orbit effects
        self.torsion_analyzer = TorsionAnalyzer()
        
        # Track history for analysis
        self.position_history = []
        self.velocity_history = []
        self.energy_history = []
        self.max_history = 100
        
    def update(self, position, velocity, energy):
        """
        Update tracker with new electron state.
        
        Args:
            position: Position vector (x,y,z)
            velocity: Velocity vector (vx,vy,vz)
            energy: Energy value
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.energy = energy
        
        # Add to history
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        self.energy_history.append(energy)
        
        # Keep history within size limit
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.energy_history.pop(0)
            
        # Update torsion analyzer
        self.torsion_analyzer.add_point(position, velocity)
        
        # Update quantum state based on observed behavior
        self._update_quantum_state()
        
    def _update_quantum_state(self):
        """Update quantum state based on observed behavior."""
        if len(self.position_history) < 10:
            return
            
        # Calculate distance from origin (proton)
        r = np.linalg.norm(self.position)
        
        # Calculate approximate n (principal quantum number) based on energy
        # For hydrogen, E_n = -13.6/n^2 eV
        if self.energy < 0:  # Only if energy is negative (bound state)
            n_calculated = np.sqrt(-13.6 / self.energy)
            n = max(1, int(round(n_calculated)))
        else:
            n = 1  # Default to ground state if energy is positive
        
        # Calculate angular momentum
        # L = r × p
        p = self.velocity  # simplification - should be mass * velocity
        L_vec = np.cross(self.position, p)
        L_mag = np.linalg.norm(L_vec)
        
        # L ~ sqrt(l(l+1)) * hbar, approximately
        l_calculated = (round(L_mag / 0.5) - 1) / 2
        l = max(0, min(n-1, int(round(l_calculated))))  # l must be less than n
        
        # Estimate magnetic quantum number (ml)
        # Project L onto z-axis
        ml = int(round(L_vec[2] / max(0.1, L_mag) * l))
        ml = max(-l, min(l, ml))  # -l ≤ ml ≤ l
        
        # Update quantum state
        self.quantum_state['n'] = n
        self.quantum_state['l'] = l
        self.quantum_state['ml'] = ml
        self.quantum_state['m'] = ml  # Alias
        self.quantum_state['energy'] = self.energy
        
        # Determine j (total angular momentum) - simplified
        self.quantum_state['j'] = l + 0.5  # Assuming spin-up electron
        
        # Set orbital type name
        if l == 0:
            self.quantum_state['orbital_type'] = 's'
        elif l == 1:
            self.quantum_state['orbital_type'] = 'p'
        elif l == 2:
            self.quantum_state['orbital_type'] = 'd'
        elif l == 3:
            self.quantum_state['orbital_type'] = 'f'
        else:
            self.quantum_state['orbital_type'] = f'l={l}'
            
    def get_quantum_state_info(self):
        """
        Get the current quantum state information.
        
        Returns:
            dict: Current quantum state parameters
        """
        return self.quantum_state.copy()
        
    def get_trajectory_analysis(self):
        """
        Get trajectory analysis data.
        
        Returns:
            dict: Trajectory analysis information
        """
        # Need minimum number of points for analysis
        if len(self.position_history) < 10:
            return {}
            
        # Calculate average radius and its standard deviation
        radii = [np.linalg.norm(p) for p in self.position_history]
        avg_radius = sum(radii) / len(radii)
        std_radius = np.std(radii)
        
        # Calculate orbital period
        # This is a very simplified calculation - would need more sophisticated
        # approaches for accurate period detection
        if len(radii) > 20:
            # Look for repeating patterns in radius
            periods = []
            for i in range(5, len(radii)//2):
                correlation = np.correlate(radii[:-i], radii[i:], mode='valid')
                if len(correlation) > 0 and correlation[0] > 0.8 * np.max(correlation):
                    periods.append(i)
            
            orbital_period = min(periods) if periods else None
        else:
            orbital_period = None
            
        return {
            'avg_radius': avg_radius,
            'std_radius': std_radius,
            'stability': 1.0 - min(1.0, std_radius / max(1.0, avg_radius)),
            'orbital_period': orbital_period
        }