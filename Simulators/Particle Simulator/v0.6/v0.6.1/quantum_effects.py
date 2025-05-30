"""
Quantum effects module for DWARF simulator.
Implements wave-particle duality and quantum mechanical behavior.
"""
import numpy as np
from scipy.special import sph_harm
from scipy import constants

class QuantumState:
    """
    Represents the quantum state of an electron in the DWARF system.
    Handles wave function calculations and probabilistic distributions.
    """
    def __init__(self, n=1, l=0, m=0):
        """
        Initialize a quantum state with quantum numbers.
        
        Args:
            n (int): Principal quantum number
            l (int): Angular momentum quantum number
            m (int): Magnetic quantum number
        """
        self.n = n  # Principal quantum number
        self.l = l  # Angular momentum quantum number
        self.m = m  # Magnetic quantum number
        self.spin = 0.5  # Electron spin
        
        # Calculate energy level in eV (simplified Bohr model)
        self.energy = -constants.value('Rydberg constant times hc in eV') / (n * n)
        
    def wave_function(self, r, theta, phi):
        """
        Calculate the wave function value at a given point.
        
        Args:
            r (float): Radial distance from nucleus
            theta (float): Polar angle
            phi (float): Azimuthal angle
            
        Returns:
            complex: Value of wave function at the specified point
        """
        # This is a simplified hydrogen atom wave function
        # Real implementation would use appropriate radial functions
        
        # Radial component (simplified)
        a0 = constants.value('Bohr radius')
        radial = np.exp(-r/(self.n * a0)) * (2*r/(self.n * a0))**self.l
        
        # Angular component
        angular = sph_harm(self.m, self.l, phi, theta)
        
        # Complete wave function
        return radial * angular
    
    def probability_density(self, r, theta, phi):
        """
        Calculate the probability density at a given point.
        
        Args:
            r (float): Radial distance from nucleus
            theta (float): Polar angle
            phi (float): Azimuthal angle
            
        Returns:
            float: Probability density at the specified point
        """
        psi = self.wave_function(r, theta, phi)
        return np.abs(psi)**2
    
    def sample_position(self, num_samples=1):
        """
        Sample positions from the probability distribution.
        
        Args:
            num_samples (int): Number of positions to sample
            
        Returns:
            ndarray: Sampled positions in Cartesian coordinates
        """
        # This is a simplified approach - real implementation would use
        # rejection sampling or other techniques to sample from the actual distribution
        
        # For hydrogen atom in the ground state (n=1, l=0, m=0)
        # the distribution is spherically symmetric
        a0 = constants.value('Bohr radius')
        
        # Sample spherical coordinates
        r = a0 * np.random.exponential(scale=self.n, size=num_samples)
        theta = np.arccos(2.0 * np.random.random(num_samples) - 1.0)
        phi = 2.0 * np.pi * np.random.random(num_samples)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return np.column_stack((x, y, z))

class QuantizedTransitions:
    """
    Handles energy quantization and transitions between discrete energy states.
    """
    def __init__(self):
        self.available_states = {}  # Dictionary of available quantum states
        
    def generate_states(self, max_n=3):
        """
        Generate available quantum states up to max_n.
        
        Args:
            max_n (int): Maximum principal quantum number
        """
        # Clear existing states
        self.available_states = {}
        
        # Generate new states
        for n in range(1, max_n + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    state = QuantumState(n, l, m)
                    key = (n, l, m)
                    self.available_states[key] = state
    
    def calculate_transition_probability(self, initial_state, final_state):
        """
        Calculate the transition probability between two quantum states.
        
        Args:
            initial_state (QuantumState): Initial state of the electron
            final_state (QuantumState): Final state of the electron
            
        Returns:
            float: Transition probability
        """
        # Implement transition probability calculations based on selection rules
        # This is a simplified version - real implementation would use
        # dipole moment matrix elements
        
        # Selection rules for electric dipole transitions
        n1, l1, m1 = initial_state.n, initial_state.l, initial_state.m
        n2, l2, m2 = final_state.n, final_state.l, final_state.m
        
        # Selection rules for l
        if l2 != l1 + 1 and l2 != l1 - 1:
            return 0.0
            
        # Selection rules for m
        if m2 != m1 + 1 and m2 != m1 and m2 != m1 - 1:
            return 0.0
            
        # Simple probability based on energy difference
        energy_diff = abs(final_state.energy - initial_state.energy)
        return 1.0 / energy_diff  # Higher probability for smaller energy jumps

    def find_allowed_transitions(self, current_state):
        """
        Find all allowed transitions from the current state.
        
        Args:
            current_state (QuantumState): Current state of the electron
            
        Returns:
            list: List of possible quantum states and their transition probabilities
        """
        transitions = []
        
        for state_key, state in self.available_states.items():
            if state_key != (current_state.n, current_state.l, current_state.m):
                prob = self.calculate_transition_probability(current_state, state)
                if prob > 0:
                    transitions.append((state, prob))
                    
        return transitions
    
    def apply_uncertainty_principle(self, position, momentum, precision=1.0):
        """
        Apply the uncertainty principle to position and momentum.
        
        Args:
            position (ndarray): Position vector
            momentum (ndarray): Momentum vector
            precision (float): Precision factor
            
        Returns:
            tuple: Adjusted position and momentum
        """
        h_bar = constants.hbar
        
        # Calculate uncertainties based on precision
        position_uncertainty = precision * h_bar / (2 * np.linalg.norm(momentum) + 1e-10)
        momentum_uncertainty = precision * h_bar / (2 * np.linalg.norm(position) + 1e-10)
        
        # Apply random fluctuations within uncertainty bounds
        position_noise = position_uncertainty * np.random.normal(0, 1, size=position.shape)
        momentum_noise = momentum_uncertainty * np.random.normal(0, 1, size=momentum.shape)
        
        return position + position_noise, momentum + momentum_noise