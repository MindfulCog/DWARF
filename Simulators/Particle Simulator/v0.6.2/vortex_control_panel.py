"""
Interactive control panel for DWARF vortex field visualization with fixed layout.
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
        
        # Set up the figure - taller to accommodate controls
        self.fig = plt.figure(figsize=(10, 14))
        
        # Standard particle colors for consistency
        self.particle_colors = {
            'proton': 'red',
            'electron': 'blue',
            'neutron': 'gray'
        }
        
        # Use GridSpec with clear vertical separation between components
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[6, 3, 2])
        
        # Main vortex field view (top 60%)
        self.ax_field = self.fig.add_subplot(self.gs[0])
        self.ax_field.set_title("DWARF Vortex Field")
        self.ax_field.set_aspect('equal')
        
        # Control panel (middle 30%)
        self.ax_controls = self.fig.add_subplot(self.gs[1])
        self.ax_controls.set_title("Particle Controls")
        self.ax_controls.axis('off')
        
        # Info panel (bottom 10%)
        self.ax_info = self.fig.add_subplot(self.gs[2])
        self.ax_info.set_title("Field Info")
        self.ax_info.axis('off')
        
        # Field visualization
        self.field_mesh = None
        self.particle_plots = {}
        self.trajectory_plot = None
        self.trajectory_points = []
        self.max_trajectory_points = 200
        
        # Create sliders for each particle
        self._create_particle_controls()
        
        # Apply tight layout with extra vertical padding
        plt.tight_layout(h_pad=3.0)
        
    def _create_particle_controls(self):
        """Create interactive controls for particles."""
        # Count particles by type for layout planning
        particle_types = {}
        for p in self.particles:
            p_type = p['type']
            if p_type not in particle_types:
                particle_types[p_type] = []
            particle_types[p_type].append(p)
        
        # Layout parameters within control axes
        slider_height = 0.15  # Relative to control axis height
        slider_spacing = 0.05
        current_y = 0.9  # Start near the top of control axis
        
        # Process each particle type
        for p_type, particles in particle_types.items():
            # Add a label for this particle type
            if particles:
                self.ax_controls.text(0.02, current_y, f"{p_type.capitalize()}s:",
                                    fontsize=12, fontweight='bold', 
                                    transform=self.ax_controls.transAxes)
                current_y -= 2 * slider_spacing
            
            # Add sliders for each particle of this type
            for i, p in enumerate(particles):
                # Create a unique label
                label = f"{p_type}_{i}"
                p['ui_label'] = label
                
                # Set up slider position - use axes coordinates
                rect = [0.25, current_y - slider_height/2, 0.65, slider_height]
                ax_slider = plt.axes(rect, transform=self.ax_controls.transAxes)
                
                # Get current spin value
                current_spin = p.get('spin', 0)
                
                # Create slider with better limits and step size
                if current_spin == 0:
                    min_spin = -1000000
                    max_spin = 1000000
                    step = 10000
                else:
                    # Make range proportional to current value
                    magnitude = abs(current_spin)
                    min_spin = -magnitude * 3
                    max_spin = magnitude * 3
                    step = magnitude / 100
                    
                slider = Slider(
                    ax_slider, label, min_spin, max_spin,
                    valinit=current_spin,
                    valstep=step,
                    color=self.particle_colors.get(p_type, 'purple')
                )
                
                # Add to collection
                self.particle_sliders[p.get('id', i)] = {
                    'slider': slider,
                    'particle': p
                }
                
                # Connect callback
                def update_spin(val, p_id=p.get('id', i)):
                    self.simulator.modify_particle_spin(p_id, val)
                    # Force update of the visualization
                    self.update()
                
                slider.on_changed(update_spin)
                
                # Move down for next slider
                current_y -= (slider_height + 2*slider_spacing)
    
    def update(self, trajectory=None):
        """
        Update the visualization.
        
        Args:
            trajectory: List of electron positions (optional)
        """
        # Clear the plot
        self.ax_field.clear()
        
        # Set up the axis - use centered coordinates
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)
        half_grid = grid_size / 2
        self.ax_field.set_xlim(-half_grid, half_grid)
        self.ax_field.set_ylim(-half_grid, half_grid)
        self.ax_field.set_title("DWARF Vortex Field")
        
        # Add grid lines
        self.ax_field.grid(True, alpha=0.3)
        self.ax_field.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        self.ax_field.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        
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
        
        # Force draw
        self.fig.canvas.draw_idle()
        
    def _update_field_visualization(self):
        """Update the vortex field visualization."""
        # Get grid size and center properly
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)
        half_size = grid_size / 2
        
        # Create a centered grid for the field
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
        
    def _update_particle_visualization(self):
        """Update the particle visualization."""
        # Get physics data
        physics_data = self.simulator.get_vortex_field_data()
        
        # Track if we've added a legend entry for each particle type
        legend_entries = set()
        
        # Plot particles
        for particle_data in physics_data['particles']:
            p_type = particle_data['type']
            p_id = particle_data.get('id', 0)
            pos = particle_data['pos']
            
            # Use consistent colors
            color = self.particle_colors.get(p_type, 'purple')
            
            # Determine sizes
            if p_type == 'proton':
                size = 150  # Make proton larger for visibility
            elif p_type == 'electron':
                size = 50
            elif p_type == 'neutron':
                size = 80
            else:
                size = 60
                
            # Add to legend only once per particle type
            label = None
            if p_type not in legend_entries:
                label = f"{p_type.capitalize()}"
                legend_entries.add(p_type)
                
            # Plot particle
            self.ax_field.scatter(pos[0], pos[1], color=color, s=size, 
                               label=label, edgecolors='white', linewidths=1)
                               
            # Add particle ID label for clarity
            self.ax_field.text(pos[0], pos[1] + size/4, f"{p_id}", 
                            ha='center', va='center', color='white',
                            fontweight='bold', fontsize=8)
                               
            # Add velocity vector if available
            if 'vel' in particle_data and np.linalg.norm(particle_data['vel']) > 0:
                vel = particle_data['vel']
                vel_norm = np.linalg.norm(vel)
                # Scale arrow length based on velocity magnitude
                arrow_scale = 20.0 / max(1.0, vel_norm)  
                vel_scaled = vel * arrow_scale
                self.ax_field.arrow(pos[0], pos[1], vel_scaled[0], vel_scaled[1],
                               head_width=5.0, head_length=10.0, fc=color, ec=color, alpha=0.7)
                               
            # Visualize spin if available
            if 'spin' in particle_data and particle_data['spin'] != 0:
                spin = particle_data['spin']
                # Use log scale for visualizing very large spin values
                spin_magnitude = 20.0 * np.log10(max(1, abs(spin) / 1000))
                spin_direction = 1 if spin > 0 else -1
                
                # Draw spin as a circle
                circle = plt.Circle((pos[0], pos[1]), spin_magnitude, 
                                  fill=False, color=color, alpha=0.5)
                self.ax_field.add_artist(circle)
                
                # Add spin direction indicator
                if spin_direction > 0:
                    self.ax_field.text(pos[0], pos[1], '+', ha='center', va='center', 
                                   color='white', fontweight='bold')
                else:
                    self.ax_field.text(pos[0], pos[1], '-', ha='center', va='center', 
                                   color='white', fontweight='bold')
        
        # Add legend if we have any entries
        if legend_entries:
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
            
        # Plot trajectory using electron's color
        self.ax_field.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                       color=self.particle_colors['electron'], alpha=0.5, linewidth=1)
        
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
            p_id = particle_data.get('id', 0)
            
            color = self.particle_colors.get(p_type, 'black')
            info_text.append(f"\n{p_type.capitalize()} {p_id}:")
            
            # Position and velocity
            pos = particle_data['pos']
            info_text.append(f"  Pos: ({pos[0]:.1f}, {pos[1]:.1f})")
            
            if 'vel' in particle_data:
                vel = particle_data['vel']
                speed = np.linalg.norm(vel)
                info_text.append(f"  Speed: {speed:.2f}")
            
            # Add spin info
            if 'spin' in particle_data:
                info_text.append(f"  Spin: {particle_data['spin']:.2e}")
            
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