import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app, scene
import cupy as cp

class Visualizer3D:
    """3D visualization for particle system and fluid grid"""
    
    def __init__(self, window_size=(1200, 800)):
        self.window_size = window_size
        self.canvas = None
        self.view = None
        self.markers = None
        self.grid_visual = None
        self.grid_lines = None
        self.grid_vector_field = None
        self.memory_vectors = None
        self.particle_system = None
        self.grid = None
        self.show_vector_field = False
        self.show_memory_field = False
        self.show_grid = True
        self.show_potential = False
        self.show_vorticity = False
        self.active_layer = 0
        self.potential_visual = None
        self.vorticity_visual = None
        
    def initialize(self, grid, particle_system):
        """Initialize visualization with a grid and particle system"""
        print("Visualizer init: Starting...")
        self.grid = grid
        self.particle_system = particle_system
        
        # Create canvas
        print("Visualizer init: Creating canvas...")
        self.canvas = scene.SceneCanvas(keys='interactive', size=self.window_size, show=True, bgcolor='black')
        
        # Add a 3D viewport
        print("Visualizer init: Creating viewport...")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.fov = 60
        self.view.camera.distance = grid.size * 1.5
        
        # Make sure background is fully opaque to fix transparency issues
        self.view.canvas.bgcolor = (0, 0, 0, 1.0)
        
        # Create axes
        print("Visualizer init: Creating axes...")
        axes = scene.visuals.XYZAxis(parent=self.view.scene)
        
        # Create grid lines
        print("Visualizer init: Creating grid lines...")
        grid_range = grid.size / 2
        grid_points = np.zeros((12 * 2, 3), dtype=np.float32)
        line_index = 0
        
        # X-axis grid lines
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                grid_points[line_index] = [-grid_range, i * grid_range, j * grid_range]
                grid_points[line_index + 1] = [grid_range, i * grid_range, j * grid_range]
                line_index += 2
        
        # Y-axis grid lines
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                grid_points[line_index] = [i * grid_range, -grid_range, j * grid_range]
                grid_points[line_index + 1] = [i * grid_range, grid_range, j * grid_range]
                line_index += 2
        
        # Z-axis grid lines
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                grid_points[line_index] = [i * grid_range, j * grid_range, -grid_range]
                grid_points[line_index + 1] = [i * grid_range, j * grid_range, grid_range]
                line_index += 2
        
        grid_colors = np.ones((12 * 2, 4), dtype=np.float32) * 0.5  # Gray with alpha
        grid_colors[:, 3] = 0.3  # Low alpha for grid lines
        
        # Create line visual for grid
        print("Visualizer init: Creating grid visual...")
        self.grid_lines = scene.visuals.Line(pos=grid_points, color=grid_colors, 
                                            connect='segments', parent=self.view.scene)
        
        # Create scatter visual for particles
        print("Visualizer init: Creating markers...")
        self.markers = visuals.Markers(parent=self.view.scene)
        self.markers.set_gl_state('translucent', depth_test=True, cull_face=False)
        
        # Create vector field visual
        print("Visualizer init: Creating vector field...")
        self.grid_vector_field = visuals.Arrow(parent=self.view.scene)
        self.grid_vector_field.visible = False
        
        # Create memory field visual
        print("Visualizer init: Creating memory field...")
        self.memory_vectors = visuals.Arrow(parent=self.view.scene)
        self.memory_vectors.visible = False
        
        # Create empty scalar volume data for potential field
        # FIX: Use scalar data (1 channel) instead of RGBA (4 channels)
        print("Visualizer init: Creating potential volume...")
        empty_scalar_vol = np.zeros((grid.base_resolution, grid.base_resolution, 
                                    grid.base_resolution), dtype=np.float32)
        
        # Create potential field visual (scalar field colormap)
        # FIX: Provide scalar data and explicitly set cmap
        self.potential_visual = scene.visuals.Volume(
            empty_scalar_vol,  # scalar field (single channel)
            parent=self.view.scene,
            cmap='coolwarm',   # use a colormap
            method='mip'       # maximum intensity projection
        )
        self.potential_visual.visible = False
        
        # Create vorticity field visual
        print("Visualizer init: Creating vorticity volume...")
        self.vorticity_visual = scene.visuals.Volume(
            empty_scalar_vol,  # scalar field (single channel)
            parent=self.view.scene,
            cmap='viridis',    # use a different colormap
            method='mip'       # maximum intensity projection
        )
        self.vorticity_visual.visible = False
        
        # Setup UI for controls
        print("Visualizer init: Setting up UI...")
        self._setup_ui()
        
        # Setup keyboard handling
        print("Visualizer init: Setting up keyboard controls...")
        @self.canvas.events.key_press.connect
        def on_key_press(event):
            if event.key == 'p':
                self._add_particle("proton")
            elif event.key == 'e':
                self._add_particle("electron")
            elif event.key == 'n':
                self._add_particle("neutron")
            elif event.key in ['1', '2', '3', '4', '5']:
                self._toggle_visualization(int(event.key))
            elif event.key in ['x', 'y', 'z']:
                self._set_spin_direction(event.key)
        
        print("Visualizer init: Completed successfully")
    
    def _setup_ui(self):
        """Setup UI elements"""
        # Add visualization controls
        grid_label = scene.visuals.Text('Grid Visualization (1-5): '
                                      '\n 1: Grid Lines'
                                      '\n 2: Vector Field'
                                      '\n 3: Memory Field'
                                      '\n 4: Potential Field'
                                      '\n 5: Vorticity Field'
                                      '\n\nParticle Controls:'
                                      '\n p: Add proton'
                                      '\n e: Add electron'
                                      '\n n: Add neutron'
                                      '\n x/y/z: Set spin direction',
                                      color='white', pos=(10, 10), parent=self.canvas.scene)
                                      
    def _toggle_visualization(self, layer_num):
        """Toggle visualization layers"""
        if layer_num == 1:
            self.show_grid = not self.show_grid
            self.grid_lines.visible = self.show_grid
        elif layer_num == 2:
            self.show_vector_field = not self.show_vector_field
            self.grid_vector_field.visible = self.show_vector_field
            if self.show_vector_field:
                self._update_vector_field()
        elif layer_num == 3:
            self.show_memory_field = not self.show_memory_field
            self.memory_vectors.visible = self.show_memory_field
            if self.show_memory_field:
                self._update_memory_field()
        elif layer_num == 4:
            self.show_potential = not self.show_potential
            self.potential_visual.visible = self.show_potential
            if self.show_potential:
                self._update_potential_field()
        elif layer_num == 5:
            self.show_vorticity = not self.show_vorticity
            self.vorticity_visual.visible = self.show_vorticity
            if self.show_vorticity:
                self._update_vorticity_field()
    
    def _add_particle(self, particle_type):
        """Add a particle at the camera focus point"""
        from dwarf_particle import Proton, Electron, Neutron
        
        # Get position from camera center
        camera_pos = self.view.camera.center
        
        # Add a small random offset
        pos = camera_pos + (np.random.random(3) - 0.5) * 0.5
        
        # Create random spin and velocity
        spin = np.random.random(3) - 0.5
        spin = spin / np.linalg.norm(spin)
        vel = (np.random.random(3) - 0.5) * 0.2
        
        # Create the appropriate particle type
        if particle_type == "proton":
            particle = Proton(pos, vel, spin)
        elif particle_type == "electron":
            particle = Electron(pos, vel, spin)
        elif particle_type == "neutron":
            particle = Neutron(pos, vel, spin)
        else:
            return
            
        # Add to particle system
        self.particle_system.add(particle)
        print(f"Added {particle_type} at position {pos}")
        
    def _set_spin_direction(self, direction):
        """Set spin direction of selected particle"""
        if self.particle_system.selected_particle is not None:
            spin = np.zeros(3)
            if direction == 'x':
                spin[0] = 1.0
            elif direction == 'y':
                spin[1] = 1.0
            elif direction == 'z':
                spin[2] = 1.0
            
            self.particle_system.selected_particle.spin = spin
            print(f"Set spin direction to {direction}")
        else:
            print("No particle selected")
    
    def update(self):
        """Update visualization based on current state"""
        if self.particle_system is None or self.grid is None:
            return
            
        # Update particle positions
        self._update_particles()
        
        # Update vector field if visible
        if self.show_vector_field:
            self._update_vector_field()
            
        # Update memory field if visible
        if self.show_memory_field:
            self._update_memory_field()
            
        # Update potential field if visible
        if self.show_potential:
            self._update_potential_field()
            
        # Update vorticity field if visible
        if self.show_vorticity:
            self._update_vorticity_field()
    
    def _update_particles(self):
        """Update particle visualization"""
        if not self.particle_system.particles:
            return
            
        # Extract positions from particles
        positions = np.array([p.position for p in self.particle_system.particles])
        
        # To handle periodic boundary conditions, create "ghost" particles
        # that appear on the opposite side when a particle is near a boundary
        grid_size = self.grid.size
        half_grid = grid_size / 2
        boundary_threshold = grid_size * 0.1  # 10% buffer from boundary
        
        # Find particles close to boundaries
        ghost_particles = []
        
        # Create a visual representation of the periodic wrapping by adding
        # "ghost" particles at the edges
        for i, pos in enumerate(positions):
            for dim in range(3):
                if pos[dim] > half_grid - boundary_threshold:
                    # Particle is close to positive boundary, add ghost at negative side
                    ghost_pos = pos.copy()
                    ghost_pos[dim] -= grid_size
                    ghost_particles.append(ghost_pos)
                    
                if pos[dim] < -half_grid + boundary_threshold:
                    # Particle is close to negative boundary, add ghost at positive side
                    ghost_pos = pos.copy()
                    ghost_pos[dim] += grid_size
                    ghost_particles.append(ghost_pos)
        
        # Store count for monitoring
        self.particle_system.ghost_particle_count = len(ghost_particles)
        
        # Combine original and ghost particles
        all_positions = np.vstack([positions, ghost_particles]) if ghost_particles else positions
        
        # Create colors for particles (including ghost copies)
        colors = []
        for p in self.particle_system.particles:
            if p.particle_type == "proton":
                colors.append([1, 0, 0, 1])  # Red
            elif p.particle_type == "electron":
                colors.append([0, 0.7, 1, 1])  # Light Blue
            elif p.particle_type == "neutron":
                colors.append([0.7, 0.7, 0.7, 1])  # Gray
            else:
                colors.append([1, 1, 1, 1])  # White
                
        # Make ghost particles semi-transparent
        ghost_colors = []
        if ghost_particles:
            for p in self.particle_system.particles:
                ghost_count_per_particle = len(ghost_particles) // len(positions)
                remainder = len(ghost_particles) % len(positions)
                
                # Calculate how many ghost particles are associated with this real particle
                count = ghost_count_per_particle
                if p.id % len(positions) < remainder:
                    count += 1
                    
                for _ in range(count):
                    if p.particle_type == "proton":
                        ghost_colors.append([1, 0, 0, 0.3])  # Red with 30% opacity
                    elif p.particle_type == "electron":
                        ghost_colors.append([0, 0.7, 1, 0.3])  # Light Blue with 30% opacity
                    elif p.particle_type == "neutron":
                        ghost_colors.append([0.7, 0.7, 0.7, 0.3])  # Gray with 30% opacity
                    else:
                        ghost_colors.append([1, 1, 1, 0.3])  # White with 30% opacity
            
        # Combine all colors
        all_colors = np.vstack([colors, ghost_colors]) if ghost_colors else np.array(colors)
        
        # Create sizes for particles
        sizes = []
        for p in self.particle_system.particles:
            base_size = 12
            if p.particle_type == "proton":
                sizes.append(base_size * 1.5)
            elif p.particle_type == "electron":
                sizes.append(base_size * 0.8)
            elif p.particle_type == "neutron":
                sizes.append(base_size * 1.2)
            else:
                sizes.append(base_size)
                
        # Duplicate sizes for ghost particles
        ghost_sizes = []
        if ghost_particles:
            for p in self.particle_system.particles:
                ghost_count_per_particle = len(ghost_particles) // len(positions)
                remainder = len(ghost_particles) % len(positions)
                
                # Calculate how many ghost particles are associated with this real particle
                count = ghost_count_per_particle
                if p.id % len(positions) < remainder:
                    count += 1
                    
                for _ in range(count):
                    base_size = 12
                    if p.particle_type == "proton":
                        ghost_sizes.append(base_size * 1.5)
                    elif p.particle_type == "electron":
                        ghost_sizes.append(base_size * 0.8)
                    elif p.particle_type == "neutron":
                        ghost_sizes.append(base_size * 1.2)
                    else:
                        ghost_sizes.append(base_size)
            
        # Combine all sizes
        all_sizes = np.concatenate([sizes, ghost_sizes]) if ghost_sizes else np.array(sizes)
        
        # Update markers
        try:
            self.markers.set_data(
                all_positions, 
                face_color=all_colors, 
                size=all_sizes,
                edge_width=0,
                edge_color=None
            )
        except Exception as e:
            print(f"Error updating particles: {e}")
        
    def _update_vector_field(self):
        """Update velocity vector field visualization"""
        if self.grid.velocity_field is None:
            return
            
        try:
            # Sample the velocity field
            sample_step = max(1, self.grid.base_resolution // 16)  # Reduce samples for clarity
            grid_size = self.grid.size
            half_grid = grid_size / 2
            
            # Linear space for grid indices
            indices = np.arange(0, self.grid.base_resolution, sample_step)
            
            # Create grid for positions
            x, y, z = np.meshgrid(indices, indices, indices, indexing='ij')
            
            # Convert indices to physical positions
            scaling = grid_size / self.grid.base_resolution
            px = (x - self.grid.base_resolution / 2) * scaling
            py = (y - self.grid.base_resolution / 2) * scaling
            pz = (z - self.grid.base_resolution / 2) * scaling
            
            # Extract velocity vectors
            if hasattr(self.grid.velocity_field, 'get'):
                # CuPy array
                vel_field = cp.asnumpy(self.grid.velocity_field[::sample_step, ::sample_step, ::sample_step])
            else:
                # NumPy array
                vel_field = self.grid.velocity_field[::sample_step, ::sample_step, ::sample_step]
                
            # Calculate vector magnitudes
            magnitudes = np.sqrt(np.sum(vel_field**2, axis=3))
            
            # Normalize vectors for visualization
            max_magnitude = np.max(magnitudes) if magnitudes.size > 0 and np.max(magnitudes) > 0 else 1.0
            scale_factor = grid_size * 0.03 / max_magnitude  # Scale vectors for visibility
            
            # Flatten arrays for Arrow visual
            pos = np.column_stack([px.flatten(), py.flatten(), pz.flatten()])
            vel = vel_field.reshape(-1, 3) * scale_factor
            
            # Filter out small vectors to reduce clutter
            mask = np.sqrt(np.sum(vel**2, axis=1)) > grid_size * 0.001
            pos_filtered = pos[mask]
            vel_filtered = vel[mask]
            
            # Skip if no vectors to show
            if len(pos_filtered) == 0:
                return
                
            # Normalize magnitudes for coloring
            mag_filtered = np.sqrt(np.sum(vel_filtered**2, axis=1)) / scale_factor
            max_mag = np.max(mag_filtered) if len(mag_filtered) > 0 and np.max(mag_filtered) > 0 else 1.0
            
            # Create color map
            colors = np.zeros((len(pos_filtered), 4))
            if len(pos_filtered) > 0:
                # Blue to red color map
                colors[:, 0] = mag_filtered / max_mag  # Red
                colors[:, 2] = 1.0 - mag_filtered / max_mag  # Blue
                colors[:, 3] = 0.6  # Alpha
                
            # Update vector field
            arrows = np.hstack([pos_filtered, vel_filtered])
            self.grid_vector_field.set_data(arrows, arrow_color=colors, arrow_size=5.0)
        except Exception as e:
            print(f"Error updating vector field: {e}")
        
    def _update_memory_field(self):
        """Update memory field visualization"""
        if self.grid.memory_field is None:
            return
            
        try:
            # Similar to velocity field but with memory field
            sample_step = max(1, self.grid.base_resolution // 12)  # Reduce samples for clarity
            grid_size = self.grid.size
            half_grid = grid_size / 2
            
            # Linear space for grid indices
            indices = np.arange(0, self.grid.base_resolution, sample_step)
            
            # Create grid for positions
            x, y, z = np.meshgrid(indices, indices, indices, indexing='ij')
            
            # Convert indices to physical positions
            scaling = grid_size / self.grid.base_resolution
            px = (x - self.grid.base_resolution / 2) * scaling
            py = (y - self.grid.base_resolution / 2) * scaling
            pz = (z - self.grid.base_resolution / 2) * scaling
            
            # Extract memory field vectors
            if hasattr(self.grid.memory_field, 'get'):
                # CuPy array
                mem_field = cp.asnumpy(self.grid.memory_field[::sample_step, ::sample_step, ::sample_step])
            else:
                # NumPy array
                mem_field = self.grid.memory_field[::sample_step, ::sample_step, ::sample_step]
                
            # Calculate vector magnitudes
            magnitudes = np.sqrt(np.sum(mem_field**2, axis=3))
            
            # Normalize vectors for visualization
            max_magnitude = np.max(magnitudes) if magnitudes.size > 0 and np.max(magnitudes) > 0 else 1.0
            scale_factor = grid_size * 0.03 / max_magnitude  # Scale vectors for visibility
            
            # Flatten arrays for Arrow visual
            pos = np.column_stack([px.flatten(), py.flatten(), pz.flatten()])
            mem_vecs = mem_field.reshape(-1, 3) * scale_factor
            
            # Filter out small vectors
            mask = np.sqrt(np.sum(mem_vecs**2, axis=1)) > grid_size * 0.001
            pos_filtered = pos[mask]
            mem_filtered = mem_vecs[mask]
            
            # Skip if no vectors to show
            if len(pos_filtered) == 0:
                return
                
            # Normalize magnitudes for coloring
            mag_filtered = np.sqrt(np.sum(mem_filtered**2, axis=1)) / scale_factor
            max_mag = np.max(mag_filtered) if len(mag_filtered) > 0 and np.max(mag_filtered) > 0 else 1.0
            
            # Create color map
            colors = np.zeros((len(pos_filtered), 4))
            if len(pos_filtered) > 0:
                # Green to purple color map for memory field
                colors[:, 0] = mag_filtered / max_mag  # Red contribution
                colors[:, 1] = 0.6 * (1.0 - mag_filtered / max_mag)  # Green contribution
                colors[:, 2] = mag_filtered / max_mag  # Blue contribution
                colors[:, 3] = 0.7  # Alpha
                
            # Update memory field
            arrows = np.hstack([pos_filtered, mem_filtered])
            self.memory_vectors.set_data(arrows, arrow_color=colors, arrow_size=4.0)
        except Exception as e:
            print(f"Error updating memory field: {e}")
        
    def _update_potential_field(self):
        """Update scalar potential field visualization"""
        if not hasattr(self.grid, 'pressure_field'):
            return
            
        try:
            # Create a volume for the scalar field
            if hasattr(self.grid.pressure_field, 'get'):
                # CuPy array
                potential = cp.asnumpy(self.grid.pressure_field)
            else:
                # NumPy array
                potential = self.grid.pressure_field
                
            # Normalize the potential field
            min_val = np.min(potential)
            max_val = np.max(potential)
            if max_val > min_val:
                # FIX: Keep as scalar field, don't convert to RGBA
                normalized = (potential - min_val) / (max_val - min_val)
                
                # Update volume with scalar data
                # Note: No need to transpose since we're using scalar data now
                self.potential_visual.set_data(normalized)
        except Exception as e:
            print(f"Error updating potential field: {e}")
        
    def _update_vorticity_field(self):
        """Update vorticity field visualization"""
        if not hasattr(self.grid, 'vorticity_magnitude'):
            return
            
        try:
            # Create a volume for the vorticity field
            if hasattr(self.grid.vorticity_magnitude, 'get'):
                # CuPy array
                vorticity = cp.asnumpy(self.grid.vorticity_magnitude)
            else:
                # NumPy array
                vorticity = self.grid.vorticity_magnitude
                
            # Normalize the vorticity field
            min_val = np.min(vorticity)
            max_val = np.max(vorticity)
            if max_val > min_val:
                # FIX: Keep as scalar field, don't convert to RGBA
                normalized = (vorticity - min_val) / (max_val - min_val)
                
                # Update volume with scalar data
                # Note: No need to transpose since we're using scalar data now
                self.vorticity_visual.set_data(normalized)
        except Exception as e:
            print(f"Error updating vorticity field: {e}")