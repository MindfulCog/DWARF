import sys
import numpy as np
import cupy as cp
import argparse
from vispy import app
import time
import psutil
import os

# Add explicit debugging prints
print("Loading modules...")

# Fixed imports with standardized lowercase naming
try:
    print("Importing dwarf_math...")
    from dwarf_math import dwarf_math
    print("Importing dwarf_particle...")
    from dwarf_particle import Particle, Proton, Electron, Neutron, ParticleSystem
    print("Importing adaptive_fluid_grid...")
    from adaptive_fluid_grid import adaptive_fluid_grid
    print("Importing dwarf_physics...")
    from dwarf_physics import dwarf_physics
    print("Importing bond_detector...")
    from bond_detector import BondDetector
    print("Importing data_logger...")
    from data_logger import DataLogger
    print("Importing visualizer_3d...")
    from visualizer_3d import Visualizer3D
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class dwarf_simulator:
    """Main simulator class for dwarf physics with debugging"""
    
    def __init__(self, config=None):
        """Initialize simulator with optional config"""
        print("Initializing simulator...")
        # Default configuration
        self.config = {
            'grid_resolution': 16,  # Reduce default resolution for debugging
            'grid_size': 10.0,
            'time_step': 0.01,
            'save_interval': 100,
            'window_size': (1200, 800),
            'seed': None,
            'initial_protons': 1,
            'initial_electrons': 1,
            'initial_neutrons': 0,
            'record_data': True,
            'use_gpu': True,
            'max_depth': 0,  # Reduce max depth for debugging
            'periodic_boundary': True
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
            
        # Set random seed if provided
        if self.config['seed'] is not None:
            np.random.seed(self.config['seed'])
            cp.random.seed(self.config['seed'])
            
        # Print GPU info
        if self.config['use_gpu']:
            try:
                print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()}")
                print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[1]/1024/1024/1024:.2f} GB total")
            except Exception as e:
                print(f"Error getting GPU info: {str(e)}")
                self.config['use_gpu'] = False
                print("Falling back to CPU mode")
        
        # Initialize components with explicit debugging
        print("Creating grid...")
        try:
            self.grid = adaptive_fluid_grid(
                base_resolution=self.config['grid_resolution'], 
                size=self.config['grid_size'],
                max_depth=self.config.get('max_depth', 0),
                use_gpu=self.config['use_gpu']
            )
            print("Grid created successfully")
        except Exception as e:
            print(f"Error creating grid: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        print("Creating particle system...")
        self.particle_system = ParticleSystem()
        
        print("Creating physics engine...")
        try:
            self.physics = dwarf_physics(
                grid=self.grid, 
                use_gpu=self.config['use_gpu']
            )
            print("Physics engine created successfully")
        except Exception as e:
            print(f"Error creating physics engine: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print("Setting up bond detector...")
        self.bond_detector = BondDetector(self.particle_system)
        self.physics.set_bond_detector(self.bond_detector)
        
        print("Setting up data logger...")
        self.logger = DataLogger() if self.config['record_data'] else None
        
        print("Setting up visualizer...")
        try:
            self.visualizer = Visualizer3D(window_size=self.config['window_size'])
            print("Visualizer created successfully")
        except Exception as e:
            print(f"Error creating visualizer: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.running = False
        self.selected_particle = None
        
        # Performance tracking
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        self.fps = 0
        self.performance_metrics = {
            'update_time': 0,
            'render_time': 0,
            'physics_time': 0
        }
        print("Simulator initialized")
        
    def initialize(self):
        """Initialize the simulation"""
        print("Starting initialization...")
        
        # Initialize grid
        print("Initializing grid...")
        start_time = time.time()
        try:
            self.grid.initialize()
            print(f"Grid initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error initializing grid: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize physics
        print("Initializing physics...")
        start_time = time.time()
        try:
            self.physics.initialize()
            print(f"Physics initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error initializing physics: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize logger if enabled
        if self.logger:
            print("Initializing logger...")
            self.logger.initialize()
        
        # Create initial particles
        print("Creating particles...")
        start_time = time.time()
        try:
            self._create_initial_particles()
            print(f"Particles created in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error creating particles: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize visualization
        print("Initializing visualizer...")
        start_time = time.time()
        try:
            self.visualizer.initialize(self.grid, self.particle_system)
            print(f"Visualizer initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error initializing visualizer: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Set running state
        self.running = True
        print("Initialization completed successfully!")
        
        # Print control instructions
        self._print_controls()
        
    def _print_controls(self):
        print("\nDWARF Simulator Controls:")
        print("  p: spawn proton")
        print("  e: spawn electron")
        print("  n: spawn neutron")
        print("  1-5: toggle visualization overlays")
        print("  x/y/z: set spin direction of selected particle")
        print("  r: reset simulation")
        print("  space: pause/resume simulation")
        print("  ESC: quit\n")
        
    def _create_initial_particles(self):
        # Create protons
        for _ in range(self.config['initial_protons']):
            pos = (np.random.random(3) - 0.5) * self.config['grid_size'] * 0.5
            spin = np.random.random(3) - 0.5
            spin = spin / np.linalg.norm(spin)
            vel = (np.random.random(3) - 0.5) * 0.2
            self.particle_system.add(Proton(pos, vel, spin))
        
        # Create electrons
        for _ in range(self.config['initial_electrons']):
            pos = (np.random.random(3) - 0.5) * self.config['grid_size'] * 0.5
            spin = np.random.random(3) - 0.5
            spin = spin / np.linalg.norm(spin)
            vel = (np.random.random(3) - 0.5) * 0.5
            self.particle_system.add(Electron(pos, vel, spin))
            
        # Create neutrons
        for _ in range(self.config['initial_neutrons']):
            pos = (np.random.random(3) - 0.5) * self.config['grid_size'] * 0.5
            spin = np.random.random(3) - 0.5
            spin = spin / np.linalg.norm(spin)
            vel = (np.random.random(3) - 0.5) * 0.1
            self.particle_system.add(Neutron(pos, vel, spin))
            
        print(f"Created {len(self.particle_system.particles)} particles: " +
              f"{self.config['initial_protons']} protons, " +
              f"{self.config['initial_electrons']} electrons, " +
              f"{self.config['initial_neutrons']} neutrons")
    
    def step(self):
        """Advance simulation by one time step"""
        print(f"Running step {self.step_count} at time {self.time:.3f}")
        
        try:
            # Start the physics timer
            physics_start = time.time()
            
            # Update physics
            self.physics.update(self.particle_system, self.config['time_step'])
            
            # Record physics time
            self.performance_metrics['physics_time'] = time.time() - physics_start
            
            # Start the grid update timer
            grid_start = time.time()
            
            # Update grid - capture number of active refinement regions
            active_regions = self.grid.update(self.particle_system.particles, self.config['time_step'])
            
            # Record grid update time
            self.performance_metrics['update_time'] = time.time() - grid_start
            
            # Log state if logger is enabled
            if self.logger and self.step_count % self.config['save_interval'] == 0:
                self.logger.log_state(self.time, self.particle_system, self.grid)
            
            # Start the render timer
            render_start = time.time()
            
            # Update visualization
            self.visualizer.update()
            
            # Record render time
            self.performance_metrics['render_time'] = time.time() - render_start
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / (current_time - self.last_frame_time)
                self.frame_count = 0
                self.last_frame_time = current_time
                
                # Print performance stats
                self._print_performance_stats()
            
            # Update simulation state
            self.time += self.config['time_step']
            self.step_count += 1
            
        except Exception as e:
            print(f"Error in step {self.step_count}: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Get GPU memory if available
        gpu_memory = "N/A"
        gpu_percent = "N/A"
        if self.config['use_gpu']:
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                total = mem_info[1]
                free = mem_info[0]
                used = (total - free) / (1024 * 1024)
                percent = (total - free) * 100 / total
                gpu_memory = f"{used:.1f} MB"
                gpu_percent = f"{percent:.1f}%"
            except:
                pass
        
        # Print formatted statistics
        print(f"Step: {self.step_count}, FPS: {self.fps:.1f}, " +
              f"Physics: {self.performance_metrics['physics_time']*1000:.1f}ms, " +
              f"Grid: {self.performance_metrics['update_time']*1000:.1f}ms, " +
              f"Render: {self.performance_metrics['render_time']*1000:.1f}ms, " +
              f"RAM: {memory_mb:.1f}MB, GPU: {gpu_percent}")
    
    def run(self, max_steps=None):
        """Run the simulation"""
        print("Setting up simulation run...")
        
        # Set up timer for animation
        def update(event):
            if self.running:
                try:
                    self.step()
                except Exception as e:
                    print(f"Error in update step: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                if max_steps and self.step_count >= max_steps:
                    self.running = False
                    self.finalize()
        
        # Create timer
        timer = app.Timer(interval=0.001, connect=update, start=True)
        print("Animation timer created")
        
        # Handle keyboard events for controls
        @self.visualizer.canvas.events.key_press.connect
        def on_key_press(event):
            print(f"Key pressed: {event.key}")
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
            elif event.key == 'r':
                self._reset_simulation()
            elif event.key == ' ':  # Space bar
                self._toggle_pause()
            elif event.key == 'escape':
                self.finalize()
                app.quit()
        
        print("Starting application run loop...")
        # Start the application
        if sys.flags.interactive == 0:
            app.run()
            
    def _add_particle(self, particle_type):
        """Add a particle at the camera focus point"""
        from dwarf_particle import Proton, Electron, Neutron
        
        # Get position from camera center
        camera_pos = self.visualizer.view.camera.center
        
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
            
    def _toggle_visualization(self, layer_num):
        """Toggle visualization layers"""
        self.visualizer._toggle_visualization(layer_num)
        
    def _reset_simulation(self):
        """Reset the simulation"""
        print("\nResetting simulation...\n")
        
        # Clear particles
        self.particle_system.particles.clear()
        self.particle_system.bonds.clear()
        if hasattr(self.particle_system, 'atom_groups'):
            self.particle_system.atom_groups.clear()
        
        # Reset grid
        print("Creating new grid...")
        self.grid = adaptive_fluid_grid(
            base_resolution=self.config['grid_resolution'], 
            size=self.config['grid_size'],
            max_depth=self.config.get('max_depth', 0),
            use_gpu=self.config['use_gpu']
        )
        self.grid.initialize()
        
        # Reset physics
        print("Creating new physics engine...")
        self.physics = dwarf_physics(
            grid=self.grid, 
            use_gpu=self.config['use_gpu']
        )
        self.physics.initialize()
        
        # Create new particles
        print("Creating new particles...")
        self._create_initial_particles()
        
        # Reset time and step count
        self.time = 0.0
        self.step_count = 0
        
    def _toggle_pause(self):
        """Toggle pause/resume simulation"""
        self.running = not self.running
        print(f"Simulation {'paused' if not self.running else 'resumed'}")
    
    def finalize(self):
        """Finalize the simulation"""
        print("\nFinalizing simulation...\n")
        
        if self.logger:
            print("Saving simulation data...")
            self.logger.finalize()
            print("Data saved successfully.")
        
        print(f"Simulation completed: {self.step_count} steps, {self.time:.2f} time units")
        print(f"Final particle count: {len(self.particle_system.particles)}")
        
        if hasattr(self.particle_system, 'bonds'):
            print(f"Bonds formed: {len(self.particle_system.bonds)}")
        
        if hasattr(self.particle_system, 'atom_groups'):
            if "hydrogen" in self.particle_system.atom_groups:
                print(f"Hydrogen atoms: {len(self.particle_system.atom_groups['hydrogen'])}")
            if "helium" in self.particle_system.atom_groups:
                print(f"Helium atoms: {len(self.particle_system.atom_groups['helium'])}")
        
        # Print performance statistics
        elapsed = time.time() - self.start_time
        print(f"\nTotal simulation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Average FPS: {self.step_count / max(1, elapsed):.2f}")
        
        # Free GPU memory
        if self.config['use_gpu']:
            print("Freeing GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU memory released.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='dwarf Physics Simulator (Debug Version)')
    
    parser.add_argument('--resolution', type=int, default=16,  # Reduce default for debugging
                        help='Grid resolution (default: 16)')
    parser.add_argument('--size', type=float, default=10.0,
                        help='Grid size in Bohr radii (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (default: 0.01)')
    parser.add_argument('--protons', type=int, default=1,
                        help='Initial number of protons (default: 1)')
    parser.add_argument('--electrons', type=int, default=1,
                        help='Initial number of electrons (default: 1)')
    parser.add_argument('--neutrons', type=int, default=0,
                        help='Initial number of neutrons (default: 0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable data logging')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--max-depth', type=int, default=0,  # Reduce for debugging
                        help='Maximum grid refinement depth (default: 0)')
    parser.add_argument('--no-periodic', action='store_true',
                        help='Disable periodic boundary conditions')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting debug version of dwarf Physics Simulator")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration from arguments
    config = {
        'grid_resolution': args.resolution,
        'grid_size': args.size,
        'time_step': args.dt,
        'initial_protons': args.protons,
        'initial_electrons': args.electrons,
        'initial_neutrons': args.neutrons,
        'seed': args.seed,
        'record_data': not args.no_log,
        'use_gpu': not args.cpu,
        'max_depth': args.max_depth,
        'periodic_boundary': not args.no_periodic
    }
    
    print(f"GPU Acceleration: {'Enabled' if not args.cpu else 'Disabled'}")
    print(f"Grid: {config['grid_resolution']}^3 cells, {config['grid_size']} Bohr radii")
    print(f"Particles: {config['initial_protons']} protons, {config['initial_electrons']} electrons, {config['initial_neutrons']} neutrons")
    print(f"Adaptive Grid: Max Depth {config['max_depth']} (up to {2**config['max_depth']}x refinement)")
    print(f"Periodic Boundary: {'Enabled' if config['periodic_boundary'] else 'Disabled'}")
    
    # Force a supported backend
    try:
        # Try different backends
        backends_to_try = ['pyqt5', 'pyglet', 'glfw']
        for backend in backends_to_try:
            try:
                print(f"Trying backend: {backend}")
                app.use_app(backend)
                print(f"Using backend: {backend}")
                break
            except Exception as e:
                print(f"Failed to use {backend}: {str(e)}")
    except Exception as e:
        print(f"Warning: Could not set preferred backend: {str(e)}")
    
    # Create and run the simulator
    print("Creating simulator instance...")
    simulator = dwarf_simulator(config)
    
    print("Initializing simulator...")
    simulator.initialize()
    
    # Show the canvas explicitly
    print("Showing visualization canvas...")
    simulator.visualizer.canvas.show()
    
    # Run the application
    print("Starting simulation run...")
    simulator.run()