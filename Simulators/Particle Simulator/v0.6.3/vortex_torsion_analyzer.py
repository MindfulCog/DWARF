"""
Analyzes torsion effects in the vortex field that correspond to
spin-orbit coupling in quantum mechanics.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

class VortexTorsionAnalyzer:
    """
    Analyzes torsion effects in the DWARF vortex model that give rise
    to emergent spin-orbit coupling.
    """
    def __init__(self, simulator, buffer_size=1000):
        """
        Initialize the torsion analyzer.
        
        Args:
            simulator: DWARF vortex simulator
            buffer_size: Maximum size of history buffers
        """
        self.simulator = simulator
        self.buffer_size = buffer_size
        
        # History buffers
        self.position_history = deque(maxlen=buffer_size)
        self.velocity_history = deque(maxlen=buffer_size)
        self.time_history = deque(maxlen=buffer_size)
        
        # Orbital plane normal vector (approximation of orbital angular momentum direction)
        self.orbital_plane_normal = np.array([0.0, 0.0, 1.0])
        
        # Effective spin vector
        self.effective_spin = np.array([0.0, 0.0, 1.0])
        
        # Precision threshold for calculations
        self.precision_threshold = 0.01
        
        # Fine structure constant (for energy corrections)
        self.alpha = 1 / 137.035999084  # Fine structure constant
        
    def record_state(self, time, vortex_field=None):
        """
        Record current state for torsion analysis.
        
        Args:
            time: Current simulation time
            vortex_field: Optional vortex field data
        """
        # Get current position and velocity
        position = self.simulator.get_electron_position()
        velocity = self.simulator.get_electron_velocity()
        
        # Convert to 3D if needed
        if len(position) == 2:
            position = np.append(position, 0.0)
            velocity = np.append(velocity, 0.0)
            
        # Add to history
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        self.time_history.append(time)
        
        # Update orbital plane normal if we have enough history
        if len(self.position_history) >= 3:
            self._update_orbital_plane()
            
    def _update_orbital_plane(self):
        """Update the orbital plane normal vector estimate."""
        # Need at least 3 points to define a plane
        if len(self.position_history) < 3:
            return
            
        # Get recent positions
        pos1 = self.position_history[-3]
        pos2 = self.position_history[-2]
        pos3 = self.position_history[-1]
        
        # Calculate vectors in the orbital plane
        v1 = pos2 - pos1
        v2 = pos3 - pos2
        
        # Calculate normal vector using cross product
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > self.precision_threshold:
            normal /= norm
            
            # Update orbital plane normal with smoothing
            smoothing_factor = 0.1
            self.orbital_plane_normal = (1 - smoothing_factor) * self.orbital_plane_normal + smoothing_factor * normal
            
            # Re-normalize
            self.orbital_plane_normal /= np.linalg.norm(self.orbital_plane_normal)
            
    def analyze_spin_orbit_coupling(self):
        """
        Analyze spin-orbit coupling from the vortex dynamics.
        
        Returns:
            dict: Spin-orbit coupling analysis
        """
        # Need sufficient history
        if len(self.position_history) < 10 or len(self.velocity_history) < 10:
            return {"detected": False}
            
        # Get recent position and velocity
        position = self.position_history[-1]
        velocity = self.velocity_history[-1]
        
        # Calculate orbital angular momentum
        L = np.cross(position, velocity)
        L_norm = np.linalg.norm(L)
        
        if L_norm < self.precision_threshold:
            return {"detected": False}
            
        # Normalize orbital angular momentum
        L_hat = L / L_norm
        
        # Update orbital plane normal with this measurement
        smoothing = 0.2
        self.orbital_plane_normal = (1 - smoothing) * self.orbital_plane_normal + smoothing * L_hat
        self.orbital_plane_normal /= np.linalg.norm(self.orbital_plane_normal)
        
        # Calculate electron "spin" direction from vortex field dynamics
        # This uses the vorticity information embedded in the velocity field
        self._update_effective_spin()
        
        # Calculate the angle between spin and orbital angular momentum
        spin_orbit_angle = self._calculate_angle(self.effective_spin, self.orbital_plane_normal)
        
        # Determine coupling type
        if spin_orbit_angle < 70:
            coupling_type = "parallel"  # j = l + 1/2
        elif spin_orbit_angle > 110:
            coupling_type = "antiparallel"  # j = l - 1/2
        else:
            coupling_type = "intermediate"
            
        # Estimate precession rate
        precession_rate = self._estimate_precession_rate()
        
        # Return analysis
        return {
            "detected": True,
            "angle": spin_orbit_angle,
            "coupling_type": coupling_type,
            "precession_rate": precession_rate,
            "L_magnitude": L_norm
        }
        
    def _update_effective_spin(self):
        """Update the effective spin vector of the electron."""
        # Need velocity history to calculate vorticity
        if len(self.velocity_history) < 5:
            return
            
        # Get recent velocities
        vel_history = list(self.velocity_history)[-5:]
        
        # Calculate finite-difference derivatives
        dv_dt = []
        for i in range(1, len(vel_history)):
            dv = vel_history[i] - vel_history[i-1]
            dv_dt.append(dv)
            
        # Calculate curl estimates from the velocity pattern
        if len(dv_dt) >= 2:
            # Use cross products to estimate rotation axis
            curl_estimate = np.cross(dv_dt[0], dv_dt[-1])
            
            # Normalize
            curl_norm = np.linalg.norm(curl_estimate)
            if curl_norm > self.precision_threshold:
                curl_estimate /= curl_norm
                
                # Update effective spin with smoothing
                smoothing = 0.1
                self.effective_spin = (1 - smoothing) * self.effective_spin + smoothing * curl_estimate
                self.effective_spin /= np.linalg.norm(self.effective_spin)
                
    def _calculate_angle(self, v1, v2):
        """
        Calculate angle between two vectors in degrees.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            float: Angle in degrees
        """
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)
        
    def _estimate_precession_rate(self):
        """
        Estimate the precession rate of the orbital plane.
        
        Returns:
            float: Precession rate in radians per time unit
        """
        # Need sufficient history
        if len(self.position_history) < 20:
            return 0.0
            
        # Look at changes in orbital plane normal over time
        normal_history = []
        
        # Calculate orbital plane normals for segments of the trajectory
        segment_size = 5
        for i in range(0, len(self.position_history) - segment_size, segment_size):
            segment = list(self.position_history)[i:i+segment_size]
            
            if len(segment) < 3:
                continue
                
            # Calculate normal vector for this segment
            v1 = segment[1] - segment[0]
            v2 = segment[2] - segment[1]
            normal = np.cross(v1, v2)
            
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > self.precision_threshold:
                normal /= norm
                normal_history.append(normal)
                
        # If we don't have enough normals, return 0
        if len(normal_history) < 2:
            return 0.0
            
        # Calculate average angular displacement between consecutive normals
        total_angle = 0.0
        count = 0
        
        for i in range(1, len(normal_history)):
            angle = self._calculate_angle(normal_history[i-1], normal_history[i])
            total_angle += np.radians(angle)
            count += 1
            
        if count == 0:
            return 0.0
            
        avg_angle = total_angle / count
        
        # Calculate time step between segments
        time_history = list(self.time_history)
        if len(time_history) < segment_size * 2:
            return 0.0
            
        time_step = (time_history[-1] - time_history[0]) / (len(normal_history) - 1)
        
        # Precession rate
        if time_step > 0:
            return avg_angle / time_step
        else:
            return 0.0
            
    def get_fine_structure_shifts(self):
        """
        Calculate energy shifts due to emergent fine structure effects.
        
        Returns:
            dict: Fine structure effects
        """
        # Analyze spin-orbit coupling
        coupling_info = self.analyze_spin_orbit_coupling()
        
        if not coupling_info["detected"]:
            return {"detected": False}
            
        # Get the base energy level
        energy = self.simulator.get_electron_energy()
        
        # Calculate estimated orbital quantum numbers
        r = np.linalg.norm(self.position_history[-1])
        n_effective = np.sqrt(-13.6 / energy) if energy < 0 else 0
        
        # Rough estimate of angular momentum quantum number
        l_estimate = min(int(n_effective) - 1, max(0, int(np.floor(n_effective/2))))
        
        # Determine j based on coupling type
        j_estimate = l_estimate + 0.5 if coupling_info["coupling_type"] == "parallel" else max(0.5, l_estimate - 0.5)
        
        # Calculate fine structure shift using the formula:
        # ΔE = -alpha^2(Z^2/n^3)(1/(j+1/2) - 0.75/n)
        # where alpha is the fine structure constant, Z is nuclear charge
        Z = 1  # Hydrogen
        
        if n_effective > 0 and l_estimate > 0:
            # Standard fine structure formula
            fine_structure_shift = -(self.alpha**2) * (Z**2 / (n_effective**3)) * (
                1 / (j_estimate + 0.5) - 0.75 / n_effective
            ) * 13.6  # Convert to eV
            
            # Check for numerical issues
            if np.isnan(fine_structure_shift) or np.isinf(fine_structure_shift):
                fine_structure_shift = 0
                
            # Scaling factor based on angle - maximum effect when aligned or anti-aligned
            angle_factor = abs(np.cos(np.radians(coupling_info["angle"])))
            fine_structure_shift *= angle_factor
        else:
            fine_structure_shift = 0
            
        # Total energy with fine structure included
        total_energy = energy + fine_structure_shift
        
        return {
            "detected": True,
            "fine_structure_shift": fine_structure_shift,
            "total_energy": total_energy,
            "estimated_n": n_effective,
            "estimated_l": l_estimate,
            "estimated_j": j_estimate
        }
        
    def visualize_spin_orbit_coupling(self):
        """
        Visualize the spin-orbit coupling effects.
        """
        # Check if we have enough data
        if len(self.position_history) < 20:
            print("Not enough data for visualization")
            return
            
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        positions = np.array(list(self.position_history))
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.5)
        
        # Get maximum radius for scaling
        max_radius = np.max(np.linalg.norm(positions, axis=1))
        
        # Plot current position
        current_pos = positions[-1]
        ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                  color='blue', s=50)
                  
        # Plot origin (nucleus)
        ax.scatter([0], [0], [0], color='red', s=100)
        
        # Plot orbital plane
        # Create a circle in the orbital plane
        theta = np.linspace(0, 2*np.pi, 100)
        radius = np.linalg.norm(current_pos)
        
        # Create vectors in the orbital plane
        if abs(self.orbital_plane_normal[2]) < 0.9:
            v1 = np.array([1, 0, -self.orbital_plane_normal[0]/self.orbital_plane_normal[2]])
        else:
            v1 = np.array([1, 0, 0])
            
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(self.orbital_plane_normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Scale vectors
        v1 = v1 * radius
        v2 = v2 * radius
        
        # Create circle points
        circle_points = np.array([np.cos(theta[i]) * v1 + np.sin(theta[i]) * v2 
                                for i in range(len(theta))])
        
        # Plot orbital plane circle
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
               'g-', alpha=0.7, label='Orbital Plane')
        
        # Plot orbital plane normal vector (angular momentum direction)
        L_vector = self.orbital_plane_normal * max_radius * 0.7
        ax.quiver(0, 0, 0, L_vector[0], L_vector[1], L_vector[2], 
                 color='green', label='Angular Momentum', length=1.0)
        
        # Plot effective spin vector
        spin_vector = self.effective_spin * max_radius * 0.7
        ax.quiver(0, 0, 0, spin_vector[0], spin_vector[1], spin_vector[2], 
                 color='red', label='Effective Spin', length=1.0)
        
        # Get coupling info
        coupling_info = self.analyze_spin_orbit_coupling()
        
        if coupling_info["detected"]:
            # Add text with coupling information
            angle = coupling_info["angle"]
            coupling_type = coupling_info["coupling_type"]
            
            ax.text2D(0.05, 0.95, f"Spin-Orbit Angle: {angle:.1f} degrees\n"
                               f"Coupling Type: {coupling_type}\n"
                               f"Precession Rate: {coupling_info['precession_rate']:.6f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
                    
            # Get fine structure info
            fine_structure = self.get_fine_structure_shifts()
            
            if fine_structure["detected"]:
                ax.text2D(0.05, 0.85, f"Fine Structure Shift: {fine_structure['fine_structure_shift']:.8f} eV\n"
                                   f"n ~ {fine_structure['estimated_n']:.1f}\n"
                                   f"j ~ {fine_structure['estimated_j']:.1f}",
                        transform=ax.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))
                        
            # Add arc showing the angle between vectors
            self._add_angle_arc(ax, L_vector, spin_vector, max_radius*0.3)
        
        ax.set_title('Emergent Spin-Orbit Coupling in DWARF Model')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_xlim([-max_radius, max_radius])
        ax.set_ylim([-max_radius, max_radius])
        ax.set_zlim([-max_radius, max_radius])
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def _add_angle_arc(self, ax, v1, v2, radius):
        """
        Add an arc showing the angle between two vectors.
        
        Args:
            ax: Matplotlib 3D axes
            v1: First vector
            v2: Second vector
            radius: Radius of the arc
        """
        # Calculate normal to both vectors
        normal = np.cross(v1, v2)
        
        # If normal is zero, vectors are parallel or antiparallel
        if np.linalg.norm(normal) < self.precision_threshold:
            return
            
        normal = normal / np.linalg.norm(normal)
        
        # Calculate angle between vectors
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Create arc
        t = np.linspace(0, angle, 20)
        
        # Normalize input vectors
        v1 = v1 / np.linalg.norm(v1) * radius
        v2 = v2 / np.linalg.norm(v2) * radius
        
        # Create rotation matrix around normal
        def rotation_matrix(axis, theta):
            """Generate rotation matrix for rotation around axis by theta."""
            axis = axis / np.linalg.norm(axis)
            a = np.cos(theta / 2)
            b, c, d = -axis * np.sin(theta / 2)
            return np.array([
                [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
            ])
            
        # Create arc points
        arc_points = np.array([np.dot(rotation_matrix(normal, t_i), v1) for t_i in t])
        
        # Plot arc
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
               'y-', linewidth=2, alpha=0.7)