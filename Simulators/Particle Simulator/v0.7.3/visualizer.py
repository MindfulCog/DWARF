import numpy as np
import time
import threading
from queue import Queue
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class Visualizer:
    def __init__(self, grid, physics, state_queue=None, command_queue=None):
        """
        Initialize the 3D visualizer
        
        Parameters:
        -----------
        grid : AdaptiveFluidGrid
            Reference to the simulation grid
        physics : DWARFPhysics
            Reference to the physics engine
        state_queue : Queue
            Queue for receiving simulation states
        command_queue : Queue
            Queue for sending commands to the simulator
        """
        self.grid = grid
        self.physics = physics
        self.state_queue = state_queue
        self.command_queue = command_queue
        
        # Configuration
        self.show_particles = True
        self.show_velocity = True
        self.show_pressure = False
        self.show_vorticity = False
        self.show_energy = False
        self.show_memory = False
        self.show_grid = False
        self.auto_rotate = False
        
        # Visualization objects
        self.app = None
        self.window = None
        self.view = None
        self.particles_plot = None
        self.velocity_plot = None
        self.pressure_plot = None
        self.vorticity_plot = None
        self.grid_plot = None
        self.trajectories_plot = {}
        
        # Colormaps
        self.colormap_velocity = pg.ColorMap([0, 0.5, 1], [[0, 0, 128], [0, 255, 255], [255, 255, 0]])
        self.colormap_pressure = pg.ColorMap([0, 0.5, 1], [[0, 0, 255], [255, 255, 255], [255, 0, 0]])
        self.colormap_vorticity = pg.ColorMap([0, 0.33, 0.66, 1], [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]])
        
        # Thread control
        self.running = False
        
        print("Visualizer initialized")
    
    def setup_ui(self):
        """Set up the visualization UI"""
        # Create PyQtGraph app and window
        self.app = pg.mkQApp("DWARF Simulator")
        self.window = QtGui.QMainWindow()
        self.window.setWindowTitle('DWARF Simulator Visualization')
        self.window.resize(1024, 768)
        
        # Create central widget and layout
        central_widget = QtGui.QWidget()
        self.window.setCentralWidget(central_widget)
        layout = QtGui.QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create 3D view
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view, stretch=8)
        
        # Set up camera position
        self.view.setCameraPosition(distance=20, elevation=30, azimuth=45)
        
        # Add coordinate axes
        axes = gl.GLAxisItem()
        axes.setSize(x=10, y=10, z=10)
        self.view.addItem(axes)
        
        # Add grid planes
        gx = gl.GLGridItem()
        gx.setSize(x=10, y=10, z=0)
        gx.setSpacing(x=1, y=1, z=1)
        gx.translate(5, 5, 0)
        self.view.addItem(gx)
        
        gy = gl.GLGridItem()
        gy.setSize(x=10, y=0, z=10)
        gy.setSpacing(x=1, y=1, z=1)
        gy.rotate(90, 1, 0, 0)
        gy.translate(5, 0, 5)
        self.view.addItem(gy)
        
        gz = gl.GLGridItem()
        gz.setSize(x=0, y=10, z=10)
        gz.setSpacing(x=1, y=1, z=1)
        gz.rotate(90, 0, 1, 0)
        gz.translate(0, 5, 5)
        self.view.addItem(gz)
        
        # Create control panel
        control_panel = QtGui.QHBoxLayout()
        layout.addLayout(control_panel)
        
        # Add buttons and controls
        self.add_controls(control_panel)
        
        # Initialize visualization elements
        self.init_visualization()
        
        # Show the window
        self.window.show()
    
    def add_controls(self, layout):
        """Add control buttons to the UI"""
        # Display options
        display_group = QtGui.QGroupBox("Display Options")
        display_layout = QtGui.QVBoxLayout()
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Checkboxes for display options
        self.checkbox_particles = QtGui.QCheckBox("Show Particles")
        self.checkbox_particles.setChecked(self.show_particles)
        self.checkbox_particles.toggled.connect(lambda checked: self.toggle_display('particles', checked))
        display_layout.addWidget(self.checkbox_particles)
        
        self.checkbox_velocity = QtGui.QCheckBox("Show Velocity")
        self.checkbox_velocity.setChecked(self.show_velocity)
        self.checkbox_velocity.toggled.connect(lambda checked: self.toggle_display('velocity', checked))
        display_layout.addWidget(self.checkbox_velocity)
        
        self.checkbox_pressure = QtGui.QCheckBox("Show Pressure")
        self.checkbox_pressure.setChecked(self.show_pressure)
        self.checkbox_pressure.toggled.connect(lambda checked: self.toggle_display('pressure', checked))
        display_layout.addWidget(self.checkbox_pressure)
        
        self.checkbox_vorticity = QtGui.QCheckBox("Show Vorticity")
        self.checkbox_vorticity.setChecked(self.show_vorticity)
        self.checkbox_vorticity.toggled.connect(lambda checked: self.toggle_display('vorticity', checked))
        display_layout.addWidget(self.checkbox_vorticity)
        
        self.checkbox_memory = QtGui.QCheckBox("Show Memory Field")
        self.checkbox_memory.setChecked(self.show_memory)
        self.checkbox_memory.toggled.connect(lambda checked: self.toggle_display('memory', checked))
        display_layout.addWidget(self.checkbox_memory)
        
        self.checkbox_grid = QtGui.QCheckBox("Show Grid")
        self.checkbox_grid.setChecked(self.show_grid)
        self.checkbox_grid.toggled.connect(lambda checked: self.toggle_display('grid', checked))
        display_layout.addWidget(self.checkbox_grid)
        
        self.checkbox_autorotate = QtGui.QCheckBox("Auto Rotate")
        self.checkbox_autorotate.setChecked(self.auto_rotate)
        self.checkbox_autorotate.toggled.connect(lambda checked: setattr(self, 'auto_rotate', checked))
        display_layout.addWidget(self.checkbox_autorotate)
        
        # Particle controls
        particle_group = QtGui.QGroupBox("Spawn Particles")
        particle_layout = QtGui.QVBoxLayout()
        particle_group.setLayout(particle_layout)
        layout.addWidget(particle_group)
        
        # Buttons for adding particles
        btn_add_electron = QtGui.QPushButton("Add Electron")
        btn_add_electron.clicked.connect(lambda: self.spawn_particle('electron'))
        particle_layout.addWidget(btn_add_electron)
        
        btn_add_proton = QtGui.QPushButton("Add Proton")
        btn_add_proton.clicked.connect(lambda: self.spawn_particle('proton'))
        particle_layout.addWidget(btn_add_proton)
        
        btn_add_neutral = QtGui.QPushButton("Add Neutral")
        btn_add_neutral.clicked.connect(lambda: self.spawn_particle('neutral'))
        particle_layout.addWidget(btn_add_neutral)
        
        # Simulation controls
        sim_group = QtGui.QGroupBox("Simulation")
        sim_layout = QtGui.QVBoxLayout()
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Buttons for simulation control
        btn_pause = QtGui.QPushButton("Pause")
        btn_pause.clicked.connect(self.pause_simulation)
        sim_layout.addWidget(btn_pause)
        
        btn_resume = QtGui.QPushButton("Resume")
        btn_resume.clicked.connect(self.resume_simulation)
        sim_layout.addWidget(btn_resume)
        
        btn_reset = QtGui.QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_simulation)
        sim_layout.addWidget(btn_reset)
    
    def init_visualization(self):
        """Initialize visualization elements"""
        # Initialize particles scatter plot
        self.particles_plot = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=10)
        self.view.addItem(self.particles_plot)
        
        # Initialize velocity arrows
        self.velocity_plot = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0, 1, 1, 1), width=2, antialias=True)
        if self.show_velocity:
            self.view.addItem(self.velocity_plot)
        
        # Initialize pressure isosurface
        self.pressure_plot = gl.GLMeshItem(meshdata=None, smooth=True, shader='shaded', color=(1, 0, 0, 0.5))
        if self.show_pressure:
            self.view.addItem(self.pressure_plot)
        
        # Initialize vorticity tubes
        self.vorticity_plot = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 0, 0, 1), width=2, antialias=True)
        if self.show_vorticity:
            self.view.addItem(self.vorticity_plot)
        
        # Initialize grid lines
        self.grid_plot = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.5, 0.5, 0.5, 0.5), width=1, antialias=True)
        if self.show_grid:
            self.view.addItem(self.grid_plot)
    
    def run(self):
        """Run the visualizer in its own thread"""
        self.running = True
        
        # Set up UI in the Qt thread
        self.setup_ui()
        
        # Start the update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(33)  # ~30 FPS
        
        # Run the Qt application event loop
        self.app.exec_()
        
        # When app exits, mark as not running
        self.running = False
        print("Visualizer stopped")
    
    def stop(self):
        """Stop the visualizer"""
        if self.running and self.timer:
            self.timer.stop()
            self.running = False
            # Use a single shot timer to quit the app from the Qt thread
            QtCore.QTimer.singleShot(0, self.app.quit)
    
    def update_visualization(self):
        """Update the visualization based on current simulation state"""
        # Get the latest state without blocking
        state = None
        if self.state_queue and not self.state_queue.empty():
            while not self.state_queue.empty():
                state = self.state_queue.get(block=False)
                self.state_queue.task_done()
        
        # If we have a state, use it to update visualization
        if state:
            # Update particles
            if self.show_particles and state.particles:
                positions = np.array([p.position for p in state.particles])
                colors = np.array([
                    (1, 0, 0, 1) if p.charge > 0 else  # red for positive charge
                    (0, 0, 1, 1) if p.charge < 0 else  # blue for negative charge
                    (0.5, 0.5, 0.5, 1)                # gray for neutral
                    for p in state.particles
                ])
                sizes = np.array([max(5, p.radius * 20) for p in state.particles])
                
                # Update particle positions
                self.particles_plot.setData(pos=positions, color=colors, size=sizes)
                
                # Update trajectories for each particle
                for i, p in enumerate(state.particles):
                    particle_id = id(p)
                    
                    # Create trajectory plot if it doesn't exist
                    if particle_id not in self.trajectories_plot:
                        color = (1, 0, 0, 0.5) if p.charge > 0 else (0, 0, 1, 0.5) if p.charge < 0 else (0.5, 0.5, 0.5, 0.5)
                        self.trajectories_plot[particle_id] = gl.GLLinePlotItem(pos=np.array([p.position]), color=color, width=2, antialias=True)
                        self.view.addItem(self.trajectories_plot[particle_id])
                    
                    # Add current position to trajectory
                    if hasattr(p, 'history') and len(p.history) > 1:
                        # Use actual history from particle if available
                        trajectory_points = np.array(p.history[-30:])  # Show last 30 points
                    else:
                        # Otherwise use current position
                        trajectory_points = np.array([p.position])
                        
                    # Update the trajectory
                    self.trajectories_plot[particle_id].setData(pos=trajectory_points)
            
            # Update velocity field visualization
            if self.show_velocity and state.velocity_field is not None:
                # Create a downsampled grid of points for velocity vectors
                resolution = state.velocity_field.shape[1]
                skip = max(1, resolution // 10)
                
                # Create grid of sample points
                x = np.linspace(0, 10, resolution)
                y = np.linspace(0, 10, resolution)
                z = np.linspace(0, 10, resolution)
                X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
                
                # Sample velocity at these points
                vx = state.velocity_field[0, ::skip, ::skip, ::skip]
                vy = state.velocity_field[1, ::skip, ::skip, ::skip]
                vz = state.velocity_field[2, ::skip, ::skip, ::skip]
                
                # Calculate velocity magnitude
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                
                # Create line segments (start and end points of arrows)
                arrow_start = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
                arrow_vec = np.stack([vx.flatten(), vy.flatten(), vz.flatten()], axis=1)
                arrow_end = arrow_start + arrow_vec * 0.5  # Scale arrows
                
                # Filter out very small velocities
                mask = v_mag.flatten() > 0.1 * np.max(v_mag)
                if np.any(mask):
                    # Use valid start and end points
                    valid_starts = arrow_start[mask]
                    valid_ends = arrow_end[mask]
                    
                    # Interleave start and end points to create line segments
                    points = np.empty((valid_starts.shape[0] * 2, 3))
                    points[0::2] = valid_starts
                    points[1::2] = valid_ends
                    
                    # Compute colors based on velocity magnitude
                    colors = np.empty((valid_starts.shape[0] * 2, 4))
                    norm_v_mag = v_mag.flatten()[mask] / np.max(v_mag)
                    
                    for i in range(len(norm_v_mag)):
                        color = self.colormap_velocity.map(norm_v_mag[i])
                        colors[i*2] = (*color, 0.8)
                        colors[i*2+1] = (*color, 0.8)
                    
                    self.velocity_plot.setData(pos=points, color=colors)
                else:
                    # No valid velocities, show empty plot
                    self.velocity_plot.setData(pos=np.zeros((2, 3)))
            
            # Update pressure field visualization (isosurface)
            if self.show_pressure and state.pressure_field is not None:
                # For simplicity, we'll use a scatter plot to visualize pressure
                # In a real application, you might use isosurfaces for better visualization
                
                # Create a downsampled grid of points
                resolution = state.pressure_field.shape[0]
                skip = max(1, resolution // 8)
                
                # Create grid of sample points
                x = np.linspace(0, 10, resolution)
                y = np.linspace(0, 10, resolution)
                z = np.linspace(0, 10, resolution)
                X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
                
                # Sample pressure at these points
                P = state.pressure_field[::skip, ::skip, ::skip]
                
                # Normalize pressure for coloring
                p_min = np.min(P)
                p_max = np.max(P)
                if p_max > p_min:
                    P_norm = (P - p_min) / (p_max - p_min)
                else:
                    P_norm = np.zeros_like(P)
                
                # Create scatter points
                points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
                
                # Create colors based on pressure
                colors = np.empty((points.shape[0], 4))
                for i in range(len(P_norm.flatten())):
                    color = self.colormap_pressure.map(P_norm.flatten()[i])
                    colors[i] = (*color, 0.5)
                
                # Create a new scatter plot each time (not efficient, but simple)
                if hasattr(self, 'pressure_scatter') and self.pressure_scatter in self.view.items:
                    self.view.removeItem(self.pressure_scatter)
                    
                self.pressure_scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=5)
                self.view.addItem(self.pressure_scatter)
            
            # Update vorticity field visualization
            if self.show_vorticity and state.vorticity_field is not None:
                # For simplicity, visualize vorticity as directed lines along vorticity direction
                # In a real application, you might use streamlines or tubes
                
                # Create a downsampled grid of points
                resolution = state.vorticity_field.shape[1]
                skip = max(1, resolution // 6)
                
                # Create grid of sample points
                x = np.linspace(0, 10, resolution)
                y = np.linspace(0, 10, resolution)
                z = np.linspace(0, 10, resolution)
                X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
                
                # Sample vorticity at these points
                wx = state.vorticity_field[0, ::skip, ::skip, ::skip]
                wy = state.vorticity_field[1, ::skip, ::skip, ::skip]
                wz = state.vorticity_field[2, ::skip, ::skip, ::skip]
                
                # Calculate vorticity magnitude
                w_mag = np.sqrt(wx**2 + wy**2 + wz**2)
                
                # Create line segments (start and end points)
                line_start = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
                line_vec = np.stack([wx.flatten(), wy.flatten(), wz.flatten()], axis=1)
                line_end = line_start + line_vec * 0.5  # Scale lines
                
                # Filter out very small vorticity
                mask = w_mag.flatten() > 0.1 * np.max(w_mag)
                if np.any(mask):
                    # Use valid start and end points
                    valid_starts = line_start[mask]
                    valid_ends = line_end[mask]
                    
                    # Interleave start and end points to create line segments
                    points = np.empty((valid_starts.shape[0] * 2, 3))
                    points[0::2] = valid_starts
                    points[1::2] = valid_ends
                    
                    # Compute colors based on vorticity magnitude
                    colors = np.empty((valid_starts.shape[0] * 2, 4))
                    norm_w_mag = w_mag.flatten()[mask] / np.max(w_mag)
                    
                    for i in range(len(norm_w_mag)):
                        color = self.colormap_vorticity.map(norm_w_mag[i])
                        colors[i*2] = (*color, 0.8)
                        colors[i*2+1] = (*color, 0.8)
                    
                    self.vorticity_plot.setData(pos=points, color=colors)
                else:
                    # No valid vorticity, show empty plot
                    self.vorticity_plot.setData(pos=np.zeros((2, 3)))
            
            # Update memory field visualization
            if self.show_memory and state.memory_field is not None:
                # Visualize memory field as points with varying opacity
                resolution = state.memory_field.shape[0]
                skip = max(1, resolution // 8)
                
                # Create grid of sample points
                x = np.linspace(0, 10, resolution)
                y = np.linspace(0, 10, resolution)
                z = np.linspace(0, 10, resolution)
                X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
                
                # Sample memory at these points
                M = state.memory_field[::skip, ::skip, ::skip]
                
                # Normalize memory for coloring
                m_min = np.min(M)
                m_max = np.max(M)
                if m_max > m_min:
                    M_norm = (M - m_min) / (m_max - m_min)
                else:
                    M_norm = np.zeros_like(M)
                
                # Get points with non-zero memory
                threshold = 0.05
                mask = M_norm.flatten() > threshold
                
                if np.any(mask):
                    # Create scatter points
                    points = np.stack([X.flatten()[mask], Y.flatten()[mask], Z.flatten()[mask]], axis=1)
                    
                    # Create colors based on memory intensity
                    colors = np.empty((points.shape[0], 4))
                    for i in range(len(colors)):
                        intensity = M_norm.flatten()[mask][i]
                        colors[i] = (0.3, 0.5, 1.0, intensity * 0.7)
                    
                    # Create a new scatter plot each time (not efficient, but simple)
                    if hasattr(self, 'memory_scatter') and self.memory_scatter in self.view.items:
                        self.view.removeItem(self.memory_scatter)
                        
                    self.memory_scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=5)
                    self.view.addItem(self.memory_scatter)
            
            # Auto-rotate view if enabled
            if self.auto_rotate:
                current_rotation = self.view.opts['azimuth']
                self.view.opts['azimuth'] = (current_rotation + 0.5) % 360
                self.view.update()
    
    def toggle_display(self, element, show):
        """Toggle visibility of visualization elements"""
        if element == 'particles':
            self.show_particles = show
            if show and self.particles_plot not in self.view.items:
                self.view.addItem(self.particles_plot)
            elif not show and self.particles_plot in self.view.items:
                self.view.removeItem(self.particles_plot)
                
                # Also remove trajectories
                for particle_id, trajectory_plot in self.trajectories_plot.items():
                    if trajectory_plot in self.view.items:
                        self.view.removeItem(trajectory_plot)
                
        elif element == 'velocity':
            self.show_velocity = show
            if show and self.velocity_plot not in self.view.items:
                self.view.addItem(self.velocity_plot)
            elif not show and self.velocity_plot in self.view.items:
                self.view.removeItem(self.velocity_plot)
                
        elif element == 'pressure':
            self.show_pressure = show
            if not show and hasattr(self, 'pressure_scatter') and self.pressure_scatter in self.view.items:
                self.view.removeItem(self.pressure_scatter)
                
        elif element == 'vorticity':
            self.show_vorticity = show
            if show and self.vorticity_plot not in self.view.items:
                self.view.addItem(self.vorticity_plot)
            elif not show and self.vorticity_plot in self.view.items:
                self.view.removeItem(self.vorticity_plot)
                
        elif element == 'memory':
            self.show_memory = show
            if not show and hasattr(self, 'memory_scatter') and self.memory_scatter in self.view.items:
                self.view.removeItem(self.memory_scatter)
                
        elif element == 'grid':
            self.show_grid = show
            if show and self.grid_plot not in self.view.items:
                self.view.addItem(self.grid_plot)
            elif not show and self.grid_plot in self.view.items:
                self.view.removeItem(self.grid_plot)
    
    def spawn_particle(self, particle_type):
        """Spawn a new particle in the simulation"""
        if not self.command_queue:
            print("Cannot spawn particle: command queue not available")
            return
            
        # Randomly position in the middle 80% of the domain
        position = np.random.uniform(1.0, 9.0, size=3)
        
        # Set properties based on particle type
        if particle_type == 'electron':
            charge = -1.0
            mass = 1.0
            radius = 0.1
        elif particle_type == 'proton':
            charge = 1.0
            mass = 1836.0  # Proton/electron mass ratio
            radius = 0.1
        else:  # neutral
            charge = 0.0
            mass = 1.0
            radius = 0.15
        
        # Send command to add particle
        command = {
            "type": "add_particle",
            "params": {
                "position": position.tolist(),
                "particle_type": particle_type,
                "charge": charge,
                "mass": mass,
                "radius": radius
            }
        }
        
        self.command_queue.put(command)
        
    def pause_simulation(self):
        """Pause the simulation"""
        if self.command_queue:
            self.command_queue.put({"type": "pause"})
    
    def resume_simulation(self):
        """Resume the simulation"""
        if self.command_queue:
            self.command_queue.put({"type": "resume"})
    
    def reset_simulation(self):
        """Reset the simulation"""
        print("Reset not implemented yet")
        # Would require a full state reset - could be implemented by sending a reset command