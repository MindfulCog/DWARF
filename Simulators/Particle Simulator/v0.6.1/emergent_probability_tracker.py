"""
Tracks the electron's positions to build probability distributions.
Identifies orbital characteristics like node patterns and symmetry.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import deque
from scipy.stats import gaussian_kde

class EmergentProbabilityTracker:
    """
    Tracks and analyzes the emergent probability distributions of particles.
    """
    def __init__(self, simulator, buffer_size=10000, grid_resolution=20):
        """
        Initialize the probability tracker.
        
        Args:
            simulator: DWARF vortex simulator
            buffer_size: Maximum number of positions to track
            grid_resolution: Resolution of probability density grid
        """
        self.simulator = simulator
        self.buffer_size = buffer_size
        self.grid_resolution = grid_resolution
        
        # Position history
        self.position_history = deque(maxlen=buffer_size)
        
        # Probability density grid parameters
        self.grid_size = 3.0  # Grid extends to +/-grid_size
        self.last_density = None
        self.last_grid = None
        
        # Minimum positions needed for meaningful analysis
        self.min_positions = 100
        
        # Orbital characteristics
        self.node_pattern = None
        self.orbital_symmetry = None
        
    def record_position(self):
        """
        Record the current electron position.
        """
        position = self.simulator.get_electron_position()
        
        # Convert to 3D position (if needed)
        if len(position) == 2:
            position = np.append(position, 0.0)  # Add z=0 for 2D simulations
            
        self.position_history.append(position)
        
    def calculate_probability_density(self, recalculate=False):
        """
        Calculate the 3D probability density from position history.
        
        Args:
            recalculate: Force recalculation even if data already exists
            
        Returns:
            tuple: (grid, density) where grid is a tuple of (X,Y,Z) coordinate arrays
                  and density is the probability density array
        """
        # Check if we have enough data
        if len(self.position_history) < self.min_positions:
            return None, None
            
        # If we've already calculated this and no new data, return cached result
        if not recalculate and self.last_density is not None:
            return self.last_grid, self.last_density
        
        # Convert position history to numpy array for calculations
        positions = np.array(self.position_history)
        
        # Create 3D grid
        x = np.linspace(-self.grid_size, self.grid_size, self.grid_resolution)
        y = np.linspace(-self.grid_size, self.grid_size, self.grid_resolution)
        z = np.linspace(-self.grid_size, self.grid_size, self.grid_resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Calculate probability density on the grid
        density = np.zeros((self.grid_resolution, self.grid_resolution, self.grid_resolution))
        
        # Simple histogram approach
        for point in positions:
            # Find the closest grid point
            ix = int((point[0] + self.grid_size) / (2 * self.grid_size) * (self.grid_resolution - 1))
            iy = int((point[1] + self.grid_size) / (2 * self.grid_size) * (self.grid_resolution - 1))
            iz = int((point[2] + self.grid_size) / (2 * self.grid_size) * (self.grid_resolution - 1))
            
            # Ensure indices are within bounds
            ix = max(0, min(ix, self.grid_resolution - 1))
            iy = max(0, min(iy, self.grid_resolution - 1))
            iz = max(0, min(iz, self.grid_resolution - 1))
            
            # Increment count
            density[ix, iy, iz] += 1
            
        # Normalize density
        total = np.sum(density)
        if total > 0:
            density /= total
            
        # Cache results
        self.last_grid = (X, Y, Z)
        self.last_density = density
        
        return (X, Y, Z), density
        
    def detect_orbital_characteristics(self):
        """
        Detect orbital characteristics such as nodes and symmetry.
        
        Returns:
            dict: Orbital characteristics
        """
        if len(self.position_history) < self.min_positions:
            return {"detected": False}
            
        # Get the probability density
        grid, density = self.calculate_probability_density()
        
        if grid is None or density is None:
            return {"detected": False}
            
        # Analyze orbital characteristics
        characteristics = self._analyze_orbital_type(grid, density)
        
        return characteristics
        
    def _analyze_orbital_type(self, grid, density):
        """
        Analyze the probability density to determine orbital type.
        
        Args:
            grid: Grid coordinates (X,Y,Z)
            density: Probability density array
            
        Returns:
            dict: Orbital characteristics
        """
        # Threshold for density consideration
        threshold = np.max(density) * 0.1
        
        # Extract grid components
        X, Y, Z = grid
        
        # Calculate distance from origin for each grid point
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Calculate angular coordinates
        Theta = np.arccos(Z / np.maximum(R, 1e-10))
        Phi = np.arctan2(Y, X)
        
        # Find high-density regions
        high_density = density > threshold
        
        # Count number of significant nodes
        # A "node" is a low-density region surrounded by high-density regions
        # This is a simplified way to detect nodes - real analysis would be more sophisticated
        
        # Analyze radial distribution
        radial_bins = 15
        radial_values = np.linspace(0, self.grid_size, radial_bins)
        radial_density = np.zeros(radial_bins-1)
        
        for i in range(radial_bins-1):
            r_min = radial_values[i]
            r_max = radial_values[i+1]
            
            # Find grid points in this radial shell
            shell = (R >= r_min) & (R < r_max)
            
            # Calculate average density in this shell
            if np.sum(shell) > 0:
                radial_density[i] = np.mean(density[shell])
                
        # Count radial nodes by looking for local minima
        radial_nodes = 0
        for i in range(1, len(radial_density)-1):
            if (radial_density[i] < radial_density[i-1] and 
                radial_density[i] < radial_density[i+1] and
                radial_density[i] < np.max(radial_density) * 0.5):
                radial_nodes += 1
                
        # Analyze angular distribution
        angular_bins = 12
        angular_values = np.linspace(0, np.pi, angular_bins)
        angular_density = np.zeros(angular_bins-1)
        
        for i in range(angular_bins-1):
            theta_min = angular_values[i]
            theta_max = angular_values[i+1]
            
            # Find grid points in this angular slice
            slice_mask = (Theta >= theta_min) & (Theta < theta_max)
            
            # Calculate average density in this slice
            if np.sum(slice_mask) > 0:
                angular_density[i] = np.mean(density[slice_mask])
                
        # Count angular nodes
        angular_nodes = 0
        for i in range(1, len(angular_density)-1):
            if (angular_density[i] < angular_density[i-1] and 
                angular_density[i] < angular_density[i+1] and
                angular_density[i] < np.max(angular_density) * 0.5):
                angular_nodes += 1
                
        # Analyze azimuthal symmetry
        azimuthal_bins = 16
        azimuthal_values = np.linspace(-np.pi, np.pi, azimuthal_bins)
        azimuthal_density = np.zeros(azimuthal_bins-1)
        
        for i in range(azimuthal_bins-1):
            phi_min = azimuthal_values[i]
            phi_max = azimuthal_values[i+1]
            
            # Find grid points in this azimuthal slice
            slice_mask = (Phi >= phi_min) & (Phi < phi_max)
            
            # Calculate average density in this slice
            if np.sum(slice_mask) > 0:
                azimuthal_density[i] = np.mean(density[slice_mask])
                
        # Detect azimuthal symmetry by analyzing pattern
        azimuthal_symmetry_order = self._detect_azimuthal_symmetry(azimuthal_density)
        
        # Total nodes
        total_nodes = radial_nodes + angular_nodes
        
        # Determine orbital type based on nodes and symmetry
        orbital_type = "unknown"
        
        if total_nodes == 0 and azimuthal_symmetry_order == 0:
            orbital_type = "s-like"  # Spherical symmetry
        elif total_nodes <= 1 and azimuthal_symmetry_order == 1:
            orbital_type = "p-like"  # Single directional lobe
        elif (total_nodes <= 2 and azimuthal_symmetry_order == 2) or angular_nodes >= 1:
            orbital_type = "d-like"  # Multiple angular nodes
        elif total_nodes >= 3:
            orbital_type = "higher-order"
            
        # Dominant radius
        positions = np.array(self.position_history)
        radii = np.linalg.norm(positions, axis=1)
        dominant_radius = np.mean(radii)
        
        # Return characteristics
        return {
            "detected": True,
            "orbital_type": orbital_type,
            "node_count": total_nodes,
            "radial_nodes": radial_nodes,
            "angular_nodes": angular_nodes,
            "azimuthal_symmetry": azimuthal_symmetry_order,
            "dominant_radius": dominant_radius
        }
        
    def _detect_azimuthal_symmetry(self, azimuthal_density):
        """
        Detect symmetry order in the azimuthal density distribution.
        
        Args:
            azimuthal_density: Density distribution in azimuthal direction
            
        Returns:
            int: Symmetry order (0 for spherical, 1 for p-like, 2 for d-like, etc.)
        """
        # Normalize density
        if np.max(azimuthal_density) > 0:
            normalized = azimuthal_density / np.max(azimuthal_density)
        else:
            return 0
            
        # Calculate variance - low variance indicates spherical symmetry
        variance = np.var(normalized)
        if variance < 0.05:
            return 0  # Spherical symmetry
            
        # Count peaks
        peaks = 0
        for i in range(1, len(normalized)-1):
            if (normalized[i] > normalized[i-1] and 
                normalized[i] > normalized[i+1] and
                normalized[i] > 0.5):
                peaks += 1
                
        if peaks == 0:
            return 0  # No clear symmetry
        elif peaks <= 2:
            return 1  # p-like (two lobes)
        elif peaks <= 4:
            return 2  # d-like (four lobes)
        else:
            return peaks // 2  # Higher order
            
    def visualize_probability_cloud(self, mode='3d_surface'):
        """
        Visualize the electron probability cloud.
        
        Args:
            mode: Visualization mode ('3d_surface', '2d_slice', or 'radial')
        """
        if len(self.position_history) < self.min_positions:
            print("Not enough position data for visualization")
            return
            
        grid, density = self.calculate_probability_density()
        
        if grid is None or density is None:
            print("Could not calculate probability density")
            return
            
        X, Y, Z = grid
        
        if mode == '3d_surface':
            # Create 3D isosurfaces
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Find threshold for significant probability density
            max_density = np.max(density)
            threshold = max_density * 0.2
            
            # Construct points for the isosurface
            points = []
            values = []
            
            # Sample points for visualization
            sample_step = 2
            for i in range(0, self.grid_resolution, sample_step):
                for j in range(0, self.grid_resolution, sample_step):
                    for k in range(0, self.grid_resolution, sample_step):
                        if density[i, j, k] > threshold:
                            points.append([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                            values.append(density[i, j, k])
            
            if not points:
                print("No significant density points found for visualization")
                return
                
            points = np.array(points)
            values = np.array(values)
            
            # Normalize values for coloring
            normalized_values = values / max_density
            
            # Plot points with color based on density
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=normalized_values, cmap='viridis',
                s=50 * normalized_values, alpha=0.6
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Normalized Probability Density')
            
            # Add origin marker
            ax.scatter([0], [0], [0], color='red', s=100, marker='o')
            
            # Set equal aspect ratio
            max_range = self.grid_size
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Get orbital characteristics for title
            characteristics = self.detect_orbital_characteristics()
            
            ax.set_title(f"Emergent {characteristics.get('orbital_type', 'Unknown')} "
                       f"Probability Distribution")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        elif mode == '2d_slice':
            # Create 2D slices through the probability cloud
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Mid-point indices
            mid_x = self.grid_resolution // 2
            mid_y = self.grid_resolution // 2
            mid_z = self.grid_resolution // 2
            
            # Create slices
            xy_slice = density[:, :, mid_z]
            xz_slice = density[:, mid_y, :]
            yz_slice = density[mid_x, :, :]
            
            # Plot slices
            axes[0].imshow(xy_slice.T, origin='lower', extent=[-self.grid_size, self.grid_size, 
                                                           -self.grid_size, self.grid_size])
            axes[0].set_title('XY Plane (Z=0)')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            
            axes[1].imshow(xz_slice.T, origin='lower', extent=[-self.grid_size, self.grid_size, 
                                                           -self.grid_size, self.grid_size])
            axes[1].set_title('XZ Plane (Y=0)')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Z')
            
            axes[2].imshow(yz_slice.T, origin='lower', extent=[-self.grid_size, self.grid_size, 
                                                           -self.grid_size, self.grid_size])
            axes[2].set_title('YZ Plane (X=0)')
            axes[2].set_xlabel('Y')
            axes[2].set_ylabel('Z')
            
            characteristics = self.detect_orbital_characteristics()
            fig.suptitle(f"Emergent {characteristics.get('orbital_type', 'Unknown')} "
                       f"Probability Distribution Slices")
                       
        elif mode == 'radial':
            # Analyze radial probability distribution
            positions = np.array(self.position_history)
            radii = np.linalg.norm(positions, axis=1)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Radial histogram
            bins = np.linspace(0, 1.1 * np.max(radii), 50)
            ax1.hist(radii, bins=bins, density=True, alpha=0.7)
            ax1.set_title('Radial Probability Distribution')
            ax1.set_xlabel('Radius')
            ax1.set_ylabel('Probability Density')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Radial probability vs. expected quantum distribution
            # Calculate radial bins for analysis
            r_bins = np.linspace(0, 2 * np.mean(radii), 50)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            r_hist, _ = np.histogram(radii, bins=r_bins, density=True)
            
            # Plot actual radial distribution
            ax2.plot(r_centers, r_hist, 'b-', label='DWARF Model')
            
            # Get orbital characteristics
            characteristics = self.detect_orbital_characteristics()
            
            # Superimpose expected quantum distribution for this orbital type
            if characteristics["detected"]:
                # Calculate theoretically expected distributions based on orbital type
                if characteristics["orbital_type"] == "s-like":
                    # 1s hydrogen-like distribution: r^2 exp(-2r/a0)
                    a0 = np.mean(radii) / 2  # Estimate Bohr radius
                    r_theory = r_centers / a0
                    p_theory = r_theory**2 * np.exp(-2 * r_theory)
                    label = "1s Orbital"
                elif characteristics["orbital_type"] == "p-like":
                    # 2p hydrogen-like distribution: r^2 exp(-r/a0) (simplified)
                    a0 = np.mean(radii) / 2
                    r_theory = r_centers / a0
                    p_theory = r_theory**2 * np.exp(-r_theory)
                    label = "2p Orbital"
                else:
                    # Generic higher orbital
                    a0 = np.mean(radii) / 3
                    r_theory = r_centers / a0
                    p_theory = r_theory**2 * np.exp(-0.7 * r_theory)
                    label = f"{characteristics['orbital_type']} Orbital"
                
                # Normalize theoretical distribution to match actual peaks
                scale_factor = np.max(r_hist) / np.max(p_theory)
                p_theory *= scale_factor
                
                # Plot theoretical distribution
                ax2.plot(r_centers, p_theory, 'r--', label=f"Theoretical {label}")
                
                # Add node count to title
                node_info = ""
                if "node_count" in characteristics:
                    node_info = f" (Nodes: {characteristics['node_count']})"
                
                ax2.set_title(f"Comparison with Quantum {label}{node_info}")
            else:
                ax2.set_title("Radial Probability Distribution")
                
            ax2.set_xlabel('Radius')
            ax2.set_ylabel('Probability Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()