"""
Interactive control panel for DWARF vortex field visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import matplotlib.gridspec as gridspec

class VortexControlPanel:
    """
    Interactive control panel for visualizing and manipulating the DWARF vortex field.
    This panel provides real-time control over particle properties and field visualization.
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
        
        # Consistent particle colors
        self.particle_colors = {
            'proton': 'red',
            'electron': 'blue',
            'neutron': 'gray'
        }
        
        # Set up the figure with proper sizing
        self.fig = plt.figure(figsize=(10, 12))
        
        # Use a GridSpec with clear separation between components
        self.gs = gridspec.GridSpec(4, 1, height_ratios=[6, 1.5, 1.5, 1])
        
        # Main vortex field view - only in top row
        self.ax_field = self.fig.add_subplot(self.gs[0])
        self.ax_field.set_title("DWARF Vortex Field")
        self.ax_field.set_aspect('equal')
        
        # Control panel - dedicated row with proper separation
        self.ax_controls = self.fig.add_subplot(self.gs[1])
        self.ax_controls.set_title("Particle Controls")
        self.ax_controls.axis('off')
        
        # Second control row for additional controls if needed
        self.ax_controls2 = self.fig.add_subplot(self.gs[2])
        self.ax_controls2.set_title("Additional Controls")
        self.ax_controls2.axis('off')
        
        # Info panel - bottom row
        self.ax_info = self.fig.add_subplot(self.gs[3])
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
        
        # Add zoom control
        self.zoom_factor = 1.0
        ax_zoom_in = plt.axes([0.85, 0.02, 0.06, 0.03])
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom +')
        self.btn_zoom_in.on_clicked(self._zoom_in)
        
        ax_zoom_out = plt.axes([0.92, 0.02, 0.06, 0.03])
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom -')
        self.btn_zoom_out.on_clicked(self._zoom_out)
        
        # Add field visibility toggle
        ax_toggle = plt.axes([0.1, 0.02, 0.15, 0.03])
        self.chk_field = CheckButtons(ax_toggle, ['Show Field'], [self.show_field])
        self.chk_field.on_clicked(self._toggle_field)
        
        # Apply tight layout with padding
        plt.tight_layout(h_pad=3.0)
        
        # Draw initial state - FIX: Call update at initialization
        self.update()
        
    def _spin_to_rpm(self, spin):
        """
        Convert spin value to RPM for display purposes.
        
        Args:
            spin: Raw spin value
            
        Returns:
            float: RPM value
        """
        # Convert to RPM (Reasonable scaling for display)
        return spin / 10000.0
        
    def _rpm_to_spin(self, rpm):
        """
        Convert RPM back to spin value.
        
        Args:
            rpm: RPM value
            
        Returns:
            float: Raw spin value
        """
        return rpm * 10000.0
    
    def _create_particle_controls(self):
        """Create interactive controls for particles."""
        # Organize particles by type - FIX: Create particle_types dict from particles
        particle_types = {'proton': [], 'electron': [], 'neutron': []}
        
        for p in self.particles:
            p_type = p['type']
            if p_type in particle_types:
                particle_types[p_type].append(p)
        
        # Layout parameters - THINNER SLIDERS
        slider_height = 0.1  # Reduced height for sliders
        slider_spacing = 0.04
        
        # Process each particle type with proper vertical spacing
        # Start from the top of the first control area
        current_ax = self.ax_controls
        current_y = 0.9
        slider_count = 0
        
        # Process protons first, then electrons, then neutrons
        for p_type in ['proton', 'electron', 'neutron']:
            particles = particle_types[p_type]
            
            # Skip if no particles of this type
            if not particles:
                continue
                
            # Add a label for this particle type
            current_ax.text(0.02, current_y, f"{p_type.capitalize()}s:",
                          fontsize=12, fontweight='bold',
                          transform=current_ax.transAxes)
            current_y -= slider_spacing * 2
            
            # Add sliders for each particle of this type
            for i, p in enumerate(particles):
                # Create a unique label
                label = f"{p_type}_{i}"
                p['ui_label'] = label
                
                # If we're running out of space in the first control area, move to the second
                if current_y < 0.2 and current_ax == self.ax_controls:
                    current_ax = self.ax_controls2
                    current_y = 0.9
                
                # Set up slider position within the appropriate axes
                ax_slider = plt.axes([0.25, current_y - slider_height/2, 0.65, slider_height], 
                                   transform=current_ax.transAxes)
                
                # Get current spin value and convert to RPM
                current_spin = p.get('spin', 0)
                current_rpm = self._spin_to_rpm(current_spin)
                
                # Create slider with better limits
                if current_rpm == 0:
                    min_rpm = -100
                    max_rpm = 100
                    step = 1
                else:
                    # Make range proportional to current value
                    magnitude = abs(current_rpm)
                    min_rpm = -magnitude * 3
                    max_rpm = magnitude * 3
                    step = magnitude / 100
                    
                slider = Slider(
                    ax_slider, f"{p_type.capitalize()} {i} (RPM)", 
                    min_rpm, max_rpm,
                    valinit=current_rpm,
                    valstep=step,
                    color=self.particle_colors.get(p_type, 'purple'),
                    valfmt='%0.1f RPM'  # Format as RPM with 1 decimal place
                )
                
                # Style the slider for better visibility
                slider.label.set_color(self.particle_colors.get(p_type, 'purple'))
                slider.valtext.set_color(self.particle_colors.get(p_type, 'purple'))
                
                # Add to collection
                self.particle_sliders[p.get('id', i)] = {
                    'slider': slider,
                    'particle': p
                }
                
                # Connect callback that converts RPM back to raw spin
                def update_spin(val, p_id=p.get('id', i)):
                    spin_value = self._rpm_to_spin(val)
                    self.simulator.modify_particle_spin(p_id, spin_value)
                    # Force update of the visualization
                    self.update()
                
                slider.on_changed(update_spin)
                
                # Update position for next slider
                current_y -= (slider_height + slider_spacing * 2)
                slider_count += 1
    
    def _zoom_in(self, event):
        """Zoom in on the vortex field."""
        self.zoom_factor = min(10.0, self.zoom_factor * 1.5)
        self.update()
        
    def _zoom_out(self, event):
        """Zoom out from the vortex field."""
        self.zoom_factor = max(0.1, self.zoom_factor / 1.5)
        self.update()
        
    def _toggle_field(self, label):
        """Toggle field visibility."""
        self.show_field = not self.show_field
        self.update()
    
    def update(self, trajectory=None):
        """
        Update the visualization.
        
        Args:
            trajectory: List of electron positions (optional)
        """
        # Clear the plot
        self.ax_field.clear()
        
        # Set up the axis with zoom factor
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)
        half_grid = grid_size / 2 / self.zoom_factor
        self.ax_field.set_xlim(-half_grid, half_grid)
        self.ax_field.set_ylim(-half_grid, half_grid)
        self.ax_field.set_title(f"DWARF Vortex Field (Zoom: {self.zoom_factor:.1f}x)")
        
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
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
        
    def _update_field_visualization(self):
        """Update the vortex field visualization."""
        # Get grid size from simulator or use default
        grid_size = getattr(self.simulator.config, 'GRID_SIZE', 2048)
        
        # Create a centered grid for the field, considering zoom
        half_size = grid_size / 2 / self.zoom_factor
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
        
        # Plot particles
        for particle_data in physics_data.get('particles', []):
            p_type = particle_data['type']
            p_id = particle_data.get('id', 0)
            pos = particle_data['pos']
            
            # Use consistent colors from the defined dictionary
            color = self.particle_colors.get(p_type, 'purple')
            
            # Determine sizes
            if p_type == 'proton':
                size = 100
            elif p_type == 'electron':
                size = 50
            elif p_type == 'neutron':
                size = 80
            else:
                size = 60
                
            # Plot particle with more visible outline
            self.ax_field.scatter(pos[0], pos[1], color=color, s=size, 
                               label=f"{p_type.capitalize()} {p_id}",
                               edgecolor='white', linewidth=1)
                               
            # Add velocity vector if available
            if 'vel' in particle_data and np.linalg.norm(particle_data['vel']) > 0:
                vel = particle_data['vel']
                vel_scaled = vel / np.linalg.norm(vel) * 5.0  # Scale for visibility
                self.ax_field.arrow(pos[0], pos[1], vel_scaled[0], vel_scaled[1],
                               head_width=1.0, head_length=1.5, fc=color, ec=color, alpha=0.7)
                               
            # Visualize spin if available
            if 'spin' in particle_data and particle_data['spin'] != 0:
                spin = particle_data['spin']
                rpm = self._spin_to_rpm(spin)
                spin_magnitude = min(20, abs(rpm) / 10)  # Cap visualization size
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
                
                # Show RPM text near particle
                self.ax_field.text(pos[0] + spin_magnitude + 1, pos[1], 
                                f"{rpm:.1f} RPM", color=color, fontsize=8)
        
        # Add legend with properly colored and labeled items
        handles, labels = self.ax_field.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_field.legend(by_label.values(), by_label.keys(), loc='upper right')
        
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
            
        # Plot trajectory with electron color
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
        info_text.append(f"Step: {physics_data.get('step', 0)}")
        info_text.append(f"Time: {physics_data.get('time', 0):.3f}")
        
        # Add particle info
        for particle_data in physics_data.get('particles', []):
            p_type = particle_data['type']
            p_id = particle_data.get('id', 0)
            
            info_text.append(f"\n{p_type.capitalize()} {p_id}:")
            
            # Position and velocity
            pos = particle_data.get('pos', [0, 0])
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
                
            # Add spin info with RPM conversion
            if 'spin' in particle_data:
                rpm = self._spin_to_rpm(particle_data['spin'])
                info_text.append(f"  Spin: {rpm:.1f} RPM")
                
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