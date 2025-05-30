"""
Spin-orbit coupling module for DWARF simulator.
Implements relativistic effects and spin-orbit interactions.
"""
import numpy as np
from scipy import constants

class SpinOrbitCoupling:
    """
    Handles spin-orbit coupling effects in the DWARF system.
    """
    def __init__(self):
        # Physical constants
        self.c = constants.c  # Speed of light
        self.e = constants.e  # Elementary charge
        self.m_e = constants.m_e  # Electron mass
        self.hbar = constants.hbar  # Reduced Planck constant
        
    def calculate_relativistic_mass(self, velocity):
        """
        Calculate the relativistic mass of an electron based on its velocity.
        
        Args:
            velocity (ndarray): Velocity vector of the electron
            
        Returns:
            float: Relativistic mass
        """
        v = np.linalg.norm(velocity)
        beta = v / self.c
        
        # Relativistic mass correction
        gamma = 1 / np.sqrt(1 - beta**2)
        return self.m_e * gamma
        
    def calculate_spin_orbit_energy(self, l, j, Z=1):
        """
        Calculate spin-orbit coupling energy correction.
        
        Args:
            l (int): Angular momentum quantum number
            j (float): Total angular momentum quantum number (l+s or l-s)
            Z (int): Atomic number
            
        Returns:
            float: Spin-orbit energy correction in electron volts
        """
        # Fine structure constant
        alpha = constants.alpha
        
        # Spin-orbit energy correction (eV)
        if l == 0:
            return 0  # No spin-orbit coupling for s orbitals (l=0)
            
        # Calculate the s dot l term from j, l, s where s=1/2
        s_dot_l = 0.5 * (j * (j + 1) - l * (l + 1) - 0.5 * (0.5 + 1))
        
        # Calculate the spin-orbit coupling energy
        rydberg_energy = constants.value('Rydberg constant times hc in eV')
        spin_orbit_energy = Z**4 * alpha**2 * rydberg_energy * s_dot_l / (l * (l + 0.5) * (l + 1))
        
        return spin_orbit_energy
        
    def calculate_magnetic_field(self, position, velocity):
        """
        Calculate the magnetic field experienced by the electron due to its motion.
        
        Args:
            position (ndarray): Position vector of the electron
            velocity (ndarray): Velocity vector of the electron
            
        Returns:
            ndarray: Magnetic field vector
        """
        r = np.linalg.norm(position)
        if r < 1e-15:
            return np.zeros(3)  # Avoid division by zero
        
        # Electric field from nucleus (assuming hydrogen)
        E_field = position * self.e / (4 * np.pi * constants.epsilon_0 * r**3)
        
        # Magnetic field in electron's rest frame (v × E/c²)
        B_field = np.cross(velocity, E_field) / self.c**2
        
        return B_field
        
    def calculate_spin_precession(self, spin_vector, magnetic_field, dt):
        """
        Calculate spin precession due to magnetic field.
        
        Args:
            spin_vector (ndarray): Spin vector of the electron
            magnetic_field (ndarray): Magnetic field vector
            dt (float): Time step
            
        Returns:
            ndarray: Updated spin vector
        """
        # Electron g-factor
        g_e = -2.00231930436  # Includes small anomalous magnetic moment correction
        
        # Electron magnetic moment
        mu = g_e * self.e * self.hbar / (2 * self.m_e)
        
        # Larmor frequency
        omega = mu * magnetic_field / self.hbar
        
        # Calculate spin precession
        if np.linalg.norm(omega) > 0:
            axis = omega / np.linalg.norm(omega)
            angle = np.linalg.norm(omega) * dt
            
            # Rotate spin vector around precession axis
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            dot_product = np.dot(spin_vector, axis)
            
            new_spin = (
                spin_vector * cos_angle + 
                np.cross(axis, spin_vector) * sin_angle +
                axis * dot_product * (1 - cos_angle)
            )
            
            # Maintain spin magnitude
            new_spin = new_spin * np.linalg.norm(spin_vector) / np.linalg.norm(new_spin)
            
            return new_spin
        else:
            return spin_vector
            
    def apply_orbital_precession(self, position, velocity, spin_vector, dt):
        """
        Apply orbital precession due to spin-orbit coupling.
        
        Args:
            position (ndarray): Position vector
            velocity (ndarray): Velocity vector
            spin_vector (ndarray): Spin vector
            dt (float): Time step
            
        Returns:
            tuple: Updated position and velocity
        """
        # Calculate angular momentum
        L = np.cross(position, self.m_e * velocity)
        
        # Calculate torque due to spin-orbit coupling
        # (simplified model - real implementation would be more complex)
        spin_orbit_torque = np.cross(spin_vector, L) * 0.001  # Scaling factor
        
        # Calculate change in angular momentum
        dL = spin_orbit_torque * dt
        
        # Apply change to velocity (to conserve angular momentum)
        dv = np.cross(position, dL) / (self.m_e * np.linalg.norm(position)**2)
        
        # Update velocity
        new_velocity = velocity + dv
        
        # Update position
        new_position = position + new_velocity * dt
        
        return new_position, new_velocity