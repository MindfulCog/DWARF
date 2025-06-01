from vispy import app, scene, visuals, gloo
from vispy.scene import ViewBox, PanZoomCamera, TurntableCamera
from vispy.scene.transforms import STTransform, MatrixTransform
from vispy.visuals.transforms import Transform
import numpy as np
import cupy as cp

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
        self.view.camera = TurntableCamera(fov=60, elevation=30, azimuth=45)
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
        
        # Velocity vectors
        self.velocity_vectors = scene.visuals.Arrow(parent=self.view.scene)
        self.velocity_arrows_visible = False
        
        # Spin vectors 
        self.spin_vectors = scene.visuals.Arrow(parent=self.view.scene)
        self.spin_arrows_visible = False
        
        # Bonds visualization
        self.bond_lines = scene.visuals.Line(parent=self.view.scene, 
                                           width=3, 
                                           color=(0.8, 0.8, 0.2, 0.6),
                                           connect='segments')
        
        # Axes to show coordinate system
        self.axes = scene.visuals.XYZAxis(parent=self.view.scene)
        
        # Grid lines to show boundaries
        grid_size = 10.0
        grid_lines = np.array([
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
        ])
        
        self.grid_visual = scene.visuals.Line(pos=grid_lines, 
                                           color=(0.5, 0.5, 0.5, 0.3), 
                                           connect='segments',
                                           parent=self.view.scene)
        
        # Fluid field visualization
        self.vector_field_vis = None
        self.fluid_state_vis = None
        self.state_volume = None
        
        # Selected particle
        self.selected_particle = None
        self.selection_marker = scene.visuals.Markers(parent=self.view.scene)
        
        # Vortex visualization
        self.vortex_points = []
        self.vortex_markers = scene.visuals.Markers(parent=self.view.scene)
        
        # Event handling
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        
        # Visualizations toggles
        self.show_field_vectors = False
        self.show_fluid_states = False
        self.show_vortices = False
        self.show_bonds = True
        
    def initialize(self, grid, particle_system):
        """Initialize visualization with grid and particles"""
        self.grid = grid
        self.particle_system = particle_system
        
        # Initialize vector field visualization
        step = 8  # Subsample for performance
        field_pos = []
        field_mag = []
        
        for i in range(0, grid.base_resolution, step):
            for j in range(0, grid.base_resolution, step):
                for k in range(0, grid.base_resolution, step):
                    # Convert grid to world position
                    world_pos = grid.grid_to_world(np.array([i, j, k]))
                    field_pos.append(world_pos)
                    field_mag.append(0.0)  # Initial magnitude
                    
        # Create vector field visualization
        self.vector_field_vis = scene.visuals.Arrows(parent=self.view.scene, 
                                                  arrow_type='stealth',
                                                  arrow_size=10,
                                                  color=(0.5, 0.5, 1.0, 0.6),
                                                  connect='segments')
        
        # Create fluid state visualization (volume rendering)
        self.state_volume = np.zeros((grid.base_resolution, grid.base_resolution, grid.base_resolution), dtype=np.float32)
        
        self.fluid_state_vis = scene.visuals.Volume(self.state_volume, 
                                                 parent=self.view.scene, 
                                                 method='mip',    # Maximum intensity projection
                                                 threshold=0.01,  # Minimum opacity threshold
                                                 emulate_texture=False)
        
        # Hide initially
        self.fluid_state_vis.visible = False
        self.vector_field_vis.visible = False
        
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
            
        # Update fluid state visualization
        if self.show_fluid_states:
            self._update_fluid_states()
            self.fluid_state_vis.visible = True
        else:
            self.fluid_state_vis.visible = False
            
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
                self.velocity_vectors.set_data(pos=np.array([starts, ends]).swapaxes(0, 1).reshape(-1, 3),
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
                spin_color[1] = min(1.0, spin_color[1] + 0.3)  # Add some green
                colors.append(tuple(spin_color))
            
            if starts:  # Only update if we have particles
                self.spin_vectors.set_data(pos=np.array([starts, ends]).swapaxes(0, 1).reshape(-1, 3),
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
        
        # Update vector field visualization
        if field_pos:
            starts = np.array(field_pos)
            ends = starts + np.array(field_vectors)
            
            combined = np.zeros((len(starts)*2, 3))
            combined[0::2] = starts
            combined[1::2] = ends
            
            self.vector_field_vis.set_data(pos=combined,
                                        color=np.repeat(field_colors, 2, axis=0),
                                        connect='segments')
    
    def _update_fluid_states(self):
        """Update fluid state visualization with adaptive resolution"""
        # Original state volume visualization
        state_array = cp.asnumpy(self.grid.state)
        step = 2  # Skip some cells for performance
        
        for i in range(0, self.grid.base_resolution, step):
            for j in range(0, self.grid.base_resolution, step):
                for k in range(0, self.grid.base_resolution, step):
                    state = state_array[i, j, k]
                    
                    # Only show compressed and vacuum states
                    if state == self.grid.COMPRESSED:
                        self.state_volume[i, j, k] = 1.0  # Compressed
                    elif state == self.grid.VACUUM:
                        self.state_volume[i, j, k] = 2.0  # Vacuum
                    else:
                        self.state_volume[i, j, k] = 0.0  # Uncompressed (invisible)
        
        # Now visualize refinement regions if available
        if hasattr(self.grid, 'refinement_regions'):
            for region_key, region_grid in self.grid.refinement_regions.items():
                base_i, base_j, base_k, level = region_key
                
                # Highlight refinement regions with a special color
                for i_offset in range(2):
                    for j_offset in range(2):
                        for k_offset in range(2):
                            i, j, k = base_i + i_offset, base_j + j_offset, base_k + k_offset
                            
                            if (0 <= i < self.grid.base_resolution and 
                                0 <= j < self.grid.base_resolution and 
                                0 <= k < self.grid.base_resolution):
                                # Mark refinement regions with a special value (3)
                                # This will be colored differently
                                self.state_volume[i, j, k] = 3.0
        
        # Update volume data
        self.fluid_state_vis.set_data(self.state_volume)
        
        # Scale to grid size
        scale_factor = self.grid.size / self.grid.base_resolution
        self.fluid_state_vis.transform = STTransform(
            scale=(scale_factor, scale_factor, scale_factor),
            translate=(-self.grid.size/2, -self.grid.size/2, -self.grid.size/2)
        )
        
        # Set custom colormap: transparent for uncompressed, red for compressed, 
        # purple for vacuum, green for refinement regions
        cmap = np.array([
            [0, 0, 0, 0],           # Transparent for uncompressed
            [1, 0.5, 0, 0.3],       # Orange/red for compressed
            [0.5, 0, 0.5, 0.3],     # Purple for vacuum
            [0, 0.8, 0.2, 0.5]      # Green for refinement regions
        ])
        self.fluid_state_vis.cmap = cmap
        self.fluid_state_vis.visible = True
    
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
        elif event.key == '2':
            # Toggle fluid state visualization
            self.show_fluid_states = not self.show_fluid_states
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
            # Get mouse position in scene coordinates
            pos = self.canvas.native.mapFromGlobal(event.pos)
            tr = self.canvas.transforms.get_transform()
            scene_pos = tr.map(pos)
            
            # Get view/camera transform
            view_tr = self.view.camera.transform
            
            # Create ray from camera to clicked point
            ray_dir = self.view.camera.transform.map([0, 0, 1]) - self.view.camera.transform.map([0, 0, 0])
            ray_ori = self.view.camera.transform.map([0, 0, 0])
            
            # Find closest particle to ray
            closest_particle = None
            closest_dist = float('inf')
            
            for particle in self.particle_system.particles:
                # Vector from ray origin to particle
                to_particle = particle.position - ray_ori
                
                # Project onto ray direction
                proj_dist = np.dot(to_particle, ray_dir)
                
                if proj_dist <= 0:
                    continue  # Behind camera
                    
                # Get closest point on ray to particle
                closest_point = ray_ori + proj_dist * ray_dir
                
                # Distance from particle to ray
                dist = np.linalg.norm(closest_point - particle.position)
                
                # Selection distance depends on particle size
                selection_radius = 0.5 + particle.mass * 0.2
                
                if dist < selection_radius and proj_dist < closest_dist:
                    closest_dist = proj_dist
                    closest_particle = particle
            
            # Update selection
            self.selected_particle = closest_particle