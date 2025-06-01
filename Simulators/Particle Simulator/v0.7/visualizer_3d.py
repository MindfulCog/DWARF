from vispy import app, scene
import numpy as np
import cupy as cp

# No transforms import - we'll avoid using them directly

class Visualizer3D:
    """3D visualization of DWARF physics simulation"""
    
    def __init__(self, window_size=(1200, 800)):
        # Create a canvas with a 3D viewport
        self.canvas = scene.SceneCanvas(keys='interactive',
                                       size=window_size,
                                       bgcolor='black',
                                       title="DWARF Physics Simulator")
        
        # Create a view for the 3D viewport
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=60, elevation=30, azimuth=45)
        self.view.camera.center = (0, 0, 0)
        self.view.camera.distance = 15
        
        # Grid size and parameters
        self.grid = None
        self.particle_system = None
        
        # For coloring particles by type
        self.particle_colors = {
            'proton': (1.0, 0.2, 0.2, 0.8),    # Red
            'electron': (0.2, 0.6, 1.0, 0.8),  # Blue
            'neutron': (0.8, 0.8, 0.8, 0.8)    # Gray
        }
        
        # Particle markers
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        
        # Velocity vectors - use Line instead of Arrow
        self.velocity_vectors = scene.visuals.Line(parent=self.view.scene)
        self.velocity_arrows_visible = False
        
        # Spin vectors - use Line instead of Arrow 
        self.spin_vectors = scene.visuals.Line(parent=self.view.scene)
        self.spin_arrows_visible = False
        
        # Bonds visualization
        self.bond_lines = scene.visuals.Line(parent=self.view.scene, 
                                          width=3, 
                                          color=(0.8, 0.8, 0.2, 0.6),
                                          connect='segments')
        
        # Add simple axes (X,Y,Z)
        self.axes = scene.visuals.XYZAxis(parent=self.view.scene)
        
        # Grid lines to show boundaries
        grid_size = 10.0
        grid_lines = [
            # Bottom face
            [-grid_size/2, -grid_size/2, -grid_size/2], [grid_size/2, -grid_size/2, -grid_size/2],
            [grid_size/2, -grid_size/2, -grid_size/2], [grid_size/2, grid_size/2, -grid_size/2],
            [grid_size/2, grid_size/2, -grid_size/2], [-grid_size/2, grid_size/2, -grid_size/2],
            [-grid_size/2, grid_size/2, -grid_size/2], [-grid_size/2, -grid_size/2, -grid_size/2],
            # Top face
            [-grid_size/2, -grid_size/2, grid_size/2], [grid_size/2, -grid_size/2, grid_size/2],
            [grid_size/2, -grid_size/2, grid_size/2], [grid_size/2, grid_size/2, grid_size/2],
            [grid_size/2, grid_size/2, grid_size/2], [-grid_size/2, grid_size/2, grid_size/2],
            [-grid_size/2, grid_size/2, grid_size/2], [-grid_size/2, -grid_size/2, grid_size/2],
            # Connecting edges
            [-grid_size/2, -grid_size/2, -grid_size/2], [-grid_size/2, -grid_size/2, grid_size/2],
            [grid_size/2, -grid_size/2, -grid_size/2], [grid_size/2, -grid_size/2, grid_size/2],
            [grid_size/2, grid_size/2, -grid_size/2], [grid_size/2, grid_size/2, grid_size/2],
            [-grid_size/2, grid_size/2, -grid_size/2], [-grid_size/2, grid_size/2, grid_size/2]
        ]
        
        self.grid_visual = scene.visuals.Line(pos=grid_lines, 
                                           color=(0.5, 0.5, 0.5, 0.3), 
                                           connect='segments',
                                           parent=self.view.scene)
        
        # Vector field visualization (simple lines)
        self.vector_field_vis = scene.visuals.Line(parent=self.view.scene)
        
        # Selected particle
        self.selected_particle = None
        self.selection_marker = scene.visuals.Markers(parent=self.view.scene)
        
        # Vortex visualization
        self.vortex_markers = scene.visuals.Markers(parent=self.view.scene)
        
        # Event handling
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        
        # Visualizations toggles
        self.show_field_vectors = False
        self.show_vortices = False
        self.show_bonds = True
        
    def initialize(self, grid, particle_system):
        """Initialize visualization with grid and particles"""
        self.grid = grid
        self.particle_system = particle_system
        
    def update(self):
        """Update visualization with current state"""
        if not self.grid or not self.particle_system:
            return
            
        # Update particles
        self._update_particles()
        
        # Update bonds
        if self.show_bonds:
            self._update_bonds()
        else:
            self.bond_lines.visible = False
        
        # Update selected particle marker
        self._update_selection_marker()
        
        # Update fluid field visualization
        if self.show_field_vectors:
            self._update_field_vectors()
            self.vector_field_vis.visible = True
        else:
            self.vector_field_vis.visible = False
            
        # Update vortex visualization
        if self.show_vortices:
            self._update_vortices()
            self.vortex_markers.visible = True
        else:
            self.vortex_markers.visible = False
        
        # Update the canvas
        self.canvas.update()
    
    def _update_particles(self):
        """Update particle visualization"""
        if not self.particle_system.particles:
            return
            
        positions = []
        colors = []
        sizes = []
        
        for particle in self.particle_system.particles:
            positions.append(particle.position)
            
            # Color by particle type
            colors.append(self.particle_colors.get(particle.particle_type, (1, 1, 1, 1)))
            
            # Size based on mass
            sizes.append(max(5, particle.mass * 10))
        
        # Update the markers
        self.markers.set_data(pos=np.array(positions),
                            face_color=np.array(colors),
                            size=np.array(sizes))
        
        # Update velocity arrows if enabled
        if self.velocity_arrows_visible:
            starts = []
            ends = []
            colors = []
            
            for particle in self.particle_system.particles:
                starts.append(particle.position)
                # Scale velocity for visualization
                vel_scale = 0.5
                ends.append(particle.position + particle.velocity * vel_scale)
                colors.append(self.particle_colors.get(particle.particle_type, (1, 1, 1, 1)))
            
            if starts:  # Only update if we have particles
                # Use Line for arrows
                arrow_pos = []
                for i in range(len(starts)):
                    arrow_pos.append(starts[i])
                    arrow_pos.append(ends[i])
                
                self.velocity_vectors.set_data(pos=np.array(arrow_pos),
                                          color=np.array(colors * 2),
                                          connect='segments')
                self.velocity_vectors.visible = True
            else:
                self.velocity_vectors.visible = False
                
        # Update spin arrows if enabled
        if self.spin_arrows_visible:
            starts = []
            ends = []
            colors = []
            
            for particle in self.particle_system.particles:
                starts.append(particle.position)
                # Scale spin for visualization
                spin_scale = 0.3
                ends.append(particle.position + particle.spin * spin_scale)
                
                # Different color for spin
                spin_color = list(self.particle_colors.get(particle.particle_type, (1, 1, 1, 1)))
                # Modify the color slightly to differentiate from velocity
                spin_color[1] = min(1.0, spin_color[1] + 0.3)
                colors.append(tuple(spin_color))
            
            if starts:  # Only update if we have particles
                # Use Line for arrows
                arrow_pos = []
                for i in range(len(starts)):
                    arrow_pos.append(starts[i])
                    arrow_pos.append(ends[i])
                
                self.spin_vectors.set_data(pos=np.array(arrow_pos),
                                       color=np.array(colors * 2),
                                       connect='segments')
                self.spin_vectors.visible = True
            else:
                self.spin_vectors.visible = False
    
    def _update_bonds(self):
        """Update bond visualization"""
        if not self.particle_system.bonds:
            self.bond_lines.visible = False
            return
            
        # Collect bond line segments
        line_positions = []
        line_colors = []
        
        for bond in self.particle_system.bonds:
            p1, p2 = bond
            line_positions.extend([p1.position, p2.position])
            
            # Different colors for different bond types
            if p1.particle_type == 'proton' and p2.particle_type == 'electron':
                bond_color = (0.2, 0.8, 0.2, 0.7)  # Green for proton-electron bonds
            else:
                bond_color = (0.8, 0.8, 0.2, 0.7)  # Yellow for other bonds
                
            line_colors.extend([bond_color, bond_color])
        
        # Update the lines
        if line_positions:
            self.bond_lines.set_data(pos=np.array(line_positions),
                                   color=np.array(line_colors),
                                   connect='segments')
            self.bond_lines.visible = True
        else:
            self.bond_lines.visible = False
    
    def _update_selection_marker(self):
        """Update the marker for the selected particle"""
        if self.selected_particle is not None:
            # Create a highlighted marker for the selected particle
            self.selection_marker.set_data(pos=np.array([self.selected_particle.position]),
                                         face_color=None,
                                         edge_color=(1, 1, 0, 1),  # Yellow outline
                                         size=12,
                                         symbol='o')
            self.selection_marker.visible = True
        else:
            self.selection_marker.visible = False
    
    def _update_field_vectors(self):
        """Update vector field visualization"""
        step = 8  # Subsample for performance
        field_pos = []
        field_vectors = []
        field_colors = []
        
        for i in range(0, self.grid.base_resolution, step):
            for j in range(0, self.grid.base_resolution, step):
                for k in range(0, self.grid.base_resolution, step):
                    # Get world position
                    world_pos = self.grid.grid_to_world(np.array([i, j, k]))
                    
                    # Get field vector and convert to CPU
                    if i < self.grid.velocity_field.shape[0] and j < self.grid.velocity_field.shape[1] and k < self.grid.velocity_field.shape[2]:
                        vel_vec = cp.asnumpy(self.grid.velocity_field[i, j, k])
                    else:
                        vel_vec = np.zeros(3)
                    
                    # Skip very small vectors
                    magnitude = np.linalg.norm(vel_vec)
                    if magnitude < 0.01:
                        continue
                        
                    field_pos.append(world_pos)
                    
                    # Scale for visualization
                    scale = min(2.0, magnitude * 5)
                    field_vectors.append(vel_vec * scale / max(magnitude, 1e-6))
                    
                    # Color based on magnitude
                    intensity = min(1.0, magnitude / 0.5)
                    field_colors.append((0.3, 0.3 + intensity*0.7, 1.0, min(0.8, intensity*0.8)))
        
        # Update vector field visualization with Line segments
        if field_pos:
            # Create paired start-end points for each vector
            vector_lines = []
            vector_colors = []
            
            for i in range(len(field_pos)):
                start = field_pos[i]
                end = field_pos[i] + field_vectors[i]
                vector_lines.append(start)
                vector_lines.append(end)
                vector_colors.append(field_colors[i])
                vector_colors.append(field_colors[i])
            
            # Update the Lines visual
            self.vector_field_vis.set_data(pos=np.array(vector_lines),
                                        color=np.array(vector_colors),
                                        connect='segments')
            self.vector_field_vis.visible = True
        else:
            self.vector_field_vis.visible = False
    
    def _update_vortices(self):
        """Update vortex visualization"""
        # Find regions of high vorticity
        vortex_pos = []
        vortex_colors = []
        
        # Threshold for detecting vortices
        threshold = 1.5
        
        # Keep only the top N vortices for performance
        max_vortices = 500
        
        # Sample points with high vorticity
        step = 4
        vortex_candidates = []
        
        for i in range(step, self.grid.base_resolution-step, step):
            for j in range(step, self.grid.base_resolution-step, step):
                for k in range(step, self.grid.base_resolution-step, step):
                    mag = self.grid.vorticity_magnitude[i, j, k]
                    if mag > threshold:
                        world_pos = self.grid.grid_to_world(np.array([i, j, k]))
                        vortex_candidates.append((world_pos, float(cp.asnumpy(mag))))
        
        # Sort by vorticity magnitude
        vortex_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Use top N vortices
        for pos, mag in vortex_candidates[:max_vortices]:
            vortex_pos.append(pos)
            
            # Color based on vorticity magnitude
            intensity = min(1.0, mag / 3.0)
            color = (0.5, intensity, 0.8, min(0.7, intensity*0.7))
            vortex_colors.append(color)
        
        # Update vortex markers
        if vortex_pos:
            self.vortex_markers.set_data(pos=np.array(vortex_pos),
                                      face_color=np.array(vortex_colors),
                                      size=5)
            self.vortex_markers.visible = True
        else:
            self.vortex_markers.visible = False
    
    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'v':
            # Toggle velocity arrows
            self.velocity_arrows_visible = not self.velocity_arrows_visible
        elif event.key == 's':
            # Toggle spin arrows
            self.spin_arrows_visible = not self.spin_arrows_visible
        elif event.key == '1':
            # Toggle field vector visualization
            self.show_field_vectors = not self.show_field_vectors
        elif event.key == '3':
            # Toggle vortex visualization
            self.show_vortices = not self.show_vortices
        elif event.key == '4':
            # Toggle bonds visualization
            self.show_bonds = not self.show_bonds
        elif event.key == 'x' and self.selected_particle:
            # Set X-aligned spin for selected particle
            self.selected_particle.spin = np.array([1.0, 0.0, 0.0])
        elif event.key == 'y' and self.selected_particle:
            # Set Y-aligned spin for selected particle
            self.selected_particle.spin = np.array([0.0, 1.0, 0.0])
        elif event.key == 'z' and self.selected_particle:
            # Set Z-aligned spin for selected particle
            self.selected_particle.spin = np.array([0.0, 0.0, 1.0])
        elif event.key == 'c':
            # Reset camera
            self.view.camera.center = (0, 0, 0)
            self.view.camera.distance = 15
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            self.canvas.update()
    
    def _on_mouse_press(self, event):
        """Handle mouse press events"""
        # Use ray picking to select particles
        if event.button == 1:  # Left click
            # Simple selection method - just select the first particle for now
            if self.particle_system.particles:
                self.selected_particle = self.particle_system.particles[0]