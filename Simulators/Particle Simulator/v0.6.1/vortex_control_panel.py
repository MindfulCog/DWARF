"""
Interactive control panel for DWARF vortex field visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

class VortexControlPanel:
    """
    Interactive control panel for visualizing and manipulating the DWARF vortex field.
    """
    def __init__(self, vortex_simulator):
        """
        Initialize the control panel.
        
        Args:
            vortex_simulator: DwarfVortexSimulator instance
        """
        self.simulator = vortex_simulator
        self.particles = self.simulator.particles
        
        # Control state
        self.show_field = True
        self.field_resolution = 30
        self.particle_sliders = {}
        
        # Set up the figure
        self.fig = plt.figure(figsize=(8, 8))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        # Main vortex field view
        self.ax_field = self.fig.add_subplot(self.gs[0, :])
        self.ax_field.set_title("DWARF Vortex Field")
        self.ax_field.set_aspect('equal')
        
        # Control panel
        self.ax_controls = self.fig.add_subplot(self.gs[1, 0])
        self.ax_controls.set_title("Controls")
        self.ax_controls.axis('off')
        
        # Info panel
        self.ax_info = self.fig.add_subplot(self.gs[1, 1])
        self.ax_info.set_title("Field Info")
        self.ax_info.axis('off')
        
        # Create sliders for each particle
        self._create_particle_controls()
        
        # Field visualization
        self.field_mesh = None
        self.particle_plots = {}
        self.trajectory_plot = None
        self.trajectory_points = []
        self.max_trajectory_points = 200
        
        plt.tight_layout()
        
    def _create_particle_controls(self):
        """Create interactive controls for particles."""
        # Get particle counts by type
        particle_counts = {}
        for p in self.particles:
            p_type = p['type']
            if p_type not in particle_counts:
                particle_counts[p_type] = 0
            p_type_idx = particle_counts[p_type]
            particle_counts[p_type] += 1
            
            # Create a unique label
            label = f"{p_type}_{p_type_idx}"
            p['ui_label'] = label
            
            # Set up slider position
            slider_y = 0.8 - (len(self.particle_sliders) * 0.15)
            
            # Create slider axis
            ax_slider = plt.axes([0.25, slider_y, 0.65, 0.05])
            
            # Get current spin value
            current_spin = p.get('spin', 0)
            
            # Create slider
            min_spin = current_spin * 0.1
            max_spin = current_spin * 3.0
            if min_spin == max_spin:
                min_spin = -1000000
                max_spin = 1000000
                
            slider = Slider(
                ax_slider, label, min_spin, max_spin,
                valinit=current_spin,
                valstep=current_spin * 0.01 if current_spin != 0 else 10000
            )
            
            # Add to collection
            self.particle_sliders[p['id']] = {
                'slider': slider,
                'particle': p
            }
            
            # Connect callback
            def update_spin(val, p_id=p['id']):
                self.simulator.modify_particle_spin(p_id, val)
            
            slider.on_changed(update_spin)
    
    def update(self, trajectory=None):
        """
        Update the visualization.
        
        Args:
            trajectory: List of electron positions (optional)
        """
        # Clear the plot
        self.ax_field.clear()
        
        # Get grid size and set up centered axis
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)
        half_grid = grid_size / 2
        self.ax_field.set_xlim(-half_grid, half_grid)
        self.ax_field.set_ylim(-half_grid, half_grid)
        self.ax_field.set_title("DWARF Vortex Field")
        self.ax_field.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_field.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
        # Update vortex field visualization if enabled
        if self.show_field:
            self._update_field_visualization()
            

        # Set up the axis
        grid_size = self.simulator.config.GRID_SIZE if hasattr(self.simulator.config, 'GRID_SIZE') else 100
        self.ax_field.set_xlim(0, grid_size)
        self.ax_field.set_ylim(0, grid_size)
        self.ax_field.set_title("DWARF Vortex Field")
        
        # Update vortex field visualization if enabled
        if self.show_field:
            self._update_field_visualization()
            
        # Update particle positions
        self._update_particle_visualization()
        
        # Update trajectory
        if trajectory:
            self._update_trajectory(trajectory)
            
        # Update info panel
        self._update_info_panel()
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
        
    def _update_field_visualization(self):
        """Update the vortex field visualization."""
        # Get grid size from simulator or use default
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)  # Use 2048 as default
        
        # Create a grid for the field
        # Instead of going from 0 to grid_size, center the visualization
        half_size = grid_size / 2
        x = np.linspace(-half_size, half_size, self.field_resolution)
        y = np.linspace(-half_size, half_size, self.field_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate field strength at each point
        Z = np.zeros_like(X)
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                pos = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.simulator.get_vortex_field_at(pos)
                
        # Normalize field for better visualization
        if np.max(Z) > 0:
            Z = Z / np.max(Z)
            
        # Create contour plot
        contour = self.ax_field.contourf(X, Y, Z, cmap='viridis', alpha=0.5, levels=20)
        
        # Add color bar if not already added
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            self.colorbar = plt.colorbar(contour, ax=self.ax_field)
            self.colorbar.set_label('Relative Field Strength')
        
        # Set axis limits to center the view
        self.ax_field.set_xlim(-half_size, half_size)
        self.ax_field.set_ylim(-half_size, half_size)
        
        # Add a center marker
        self.ax_field.plot(0, 0, 'k+', markersize=10)  # Black '+' marker at center
            
    def _update_particle_visualization(self):
        """Update the particle visualization."""
        # Get physics data
        physics_data = self.simulator.get_vortex_field_data()
        
        # Plot particles
        for particle_data in physics_data['particles']:
            p_type = particle_data['type']
            p_id = particle_data['id']
            pos = particle_data['pos']
            
            # Choose color and size based on particle type
            if p_type == 'proton':
                color = 'red'
                size = 100
            elif p_type == 'electron':
                color = 'blue'
                size = 50
            elif p_type == 'neutron':
                color = 'gray'
                size = 80
            else:
                color = 'purple'
                size = 60
                
            # Plot particle
            self.ax_field.scatter(pos[0], pos[1], color=color, s=size, 
                               label=f"{p_type} {p_id}" if p_id == 0 else None)
                               
            # Add velocity vector if available
            if 'vel' in particle_data and np.linalg.norm(particle_data['vel']) > 0:
                vel = particle_data['vel']
                vel_scaled = vel / np.linalg.norm(vel) * 5.0  # Scale for visibility
                self.ax_field.arrow(pos[0], pos[1], vel_scaled[0], vel_scaled[1],
                               head_width=1.0, head_length=1.5, fc=color, ec=color, alpha=0.7)
                               
            # Visualize spin if available
            if 'spin' in particle_data and particle_data['spin'] != 0:
                spin = particle_data['spin']
                spin_magnitude = abs(spin) / 100000  # Scale for visibility
                spin_direction = 1 if spin > 0 else -1
                
                # Draw spin as a circle
                circle = plt.Circle((pos[0], pos[1]), spin_magnitude, 
                                  fill=False, color=color, alpha=0.5)
                self.ax_field.add_artist(circle)
                
                # Add spin direction indicator
                if spin_direction > 0:
                    self.ax_field.text(pos[0], pos[1], '+', ha='center', va='center', 
                                   color=color, fontweight='bold')
                else:
                    self.ax_field.text(pos[0], pos[1], '-', ha='center', va='center', 
                                   color=color, fontweight='bold')
        
        # Add legend
        self.ax_field.legend(loc='upper right')
        
    def _update_trajectory(self, trajectory):
        """
        Update the electron trajectory visualization.
        
        Args:
            trajectory: List of electron positions
        """
        if not trajectory:
            return
            
        # Convert to numpy array if needed
        if isinstance(trajectory[0], np.ndarray):
            # For 3D trajectories, take only x and y
            if len(trajectory[0]) > 2:
                trajectory_points = np.array([p[:2] for p in trajectory])
            else:
                trajectory_points = np.array(trajectory)
        else:
            # Handle list of lists or similar
            trajectory_points = np.array(trajectory)
            
        # Plot trajectory
        self.ax_field.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                       'b-', alpha=0.5, linewidth=1)
        
    def _update_info_panel(self):
        """Update the information panel."""
        # Clear current info
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Get physics data
        physics_data = self.simulator.get_vortex_field_data()
        
        # Format info text
        info_text = []
        info_text.append(f"Step: {physics_data['step']}")
        info_text.append(f"Time: {physics_data['time']:.3f}")
        
        # Add particle info
        for particle_data in physics_data['particles']:
            p_type = particle_data['type']
            p_id = particle_data['id']
            
            info_text.append(f"\n{p_type.capitalize()} {p_id}:")
            
            # Position and velocity
            pos = particle_data['pos']
            info_text.append(f"  Pos: ({pos[0]:.1f}, {pos[1]:.1f})")
            
            if 'vel' in particle_data:
                vel = particle_data['vel']
                speed = np.linalg.norm(vel)
                info_text.append(f"  Speed: {speed:.2f}")
            
            # Angular momentum and curl
            if 'angular_momentum' in particle_data:
                info_text.append(f"  Ang.Mom: {particle_data['angular_momentum']:.2f}")
            
            if 'curl' in particle_data:
                info_text.append(f"  Curl: {particle_data['curl']:.2e}")
                
            # Add memory info if available
            if 'field_memory' in particle_data:
                memory_mag = np.linalg.norm(particle_data['field_memory'])
                info_text.append(f"  Memory: {memory_mag:.2e}")
                
        # Add distance info
        if 'distances' in physics_data:
            info_text.append("\nDistances:")
            for key, distance in physics_data['distances'].items():
                info_text.append(f"  {key}: {distance:.2f}")
                
        # Display text
        self.ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                       va='top', fontsize=9, family='monospace')