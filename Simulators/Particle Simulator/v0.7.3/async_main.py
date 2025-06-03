import sys
import numpy as np
import cupy as cp
import argparse
from vispy import app
import time
import os

# Import components
from dwarf_math import dwarf_math
from dwarf_particle import Particle, Proton, Electron, Neutron, ParticleSystem
from adaptive_fluid_grid import adaptive_fluid_grid
from dwarf_physics import dwarf_physics
from bond_detector import BondDetector
from data_logger import DataLogger
from visualizer_3d import Visualizer3D

# Import async simulator
from async_dwarf_sim import AsyncSimulator


class dwarf_simulator:
    """Main simulator class for dwarf physics with async capabilities"""
    
    def debug_time_tracker(simulator):
        """Print time progress information"""
        print(f"Simulation time: {simulator.async_sim.time:.6f}, Steps: {simulator.async_sim.step_count}")

    
    def __init__(self, config=None):
        """Initialize simulator with optional config"""
        # Default configuration
        self.config = {
            'grid_resolution': 64,
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
            'max_depth': 2,
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
        
        # Initialize components
        self.grid = adaptive_fluid_grid(
            base_resolution=self.config['grid_resolution'], 
            size=self.config['grid_size'],
            max_depth=self.config.get('max_depth', 2),
            use_gpu=self.config['use_gpu']
        )
            
        self.particle_system = ParticleSystem()
        
        self.physics = dwarf_physics(
            grid=self.grid, 
            use_gpu=self.config['use_gpu']
        )
        
        self.bond_detector = BondDetector(self.particle_system)
        self.physics.set_bond_detector(self.bond_detector)
        
        self.logger = DataLogger() if self.config['record_data'] else None
        
        self.visualizer = Visualizer3D(window_size=self.config['window_size'])
        
        # Create async simulator
        self.async_sim = AsyncSimulator(
            physics_engine=self.physics,
            grid=self.grid,
            particle_system=self.particle_system,
            config=self.config
        )
        
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
        
    def initialize(self):
        """Initialize the simulation with vectorized operations"""
        # Initialize logger if enabled
        if self.logger:
            self.logger.initialize()
        
        # Create initial particles
        self._create_initial_particles()
        
        # Use the vectorized initialization for the grid if available
        if hasattr(self.grid, 'initialize_vectorized'):
            print("Using vectorized grid initialization...")
            start_time = time.time()
            self.grid.initialize_vectorized()
            print(f"Grid initialized in {time.time() - start_time:.2f} seconds")
        else:
            print("Using standard grid initialization...")
            start_time = time.time()
            self.grid.initialize()
            print(f"Grid initialized in {time.time() - start_time:.2f} seconds")
        
        # Initialize visualization (with the grid and particle system)
        self.visualizer.initialize(self.grid, self.particle_system)
        
        # Start the async simulation with explicit progress reporting
        print("Starting async simulator...")
        start_time = time.time()
        self.async_sim.start()
        print(f"Async simulation started in {time.time() - start_time:.2f} seconds")
        
        # Set running state
        self.running = True
        
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
    
    def run(self, max_steps=None):
        """Run the simulation"""
        # Set up timer for animation
        def update(event):
            if self.running:
                # Get the latest state from the async simulator
                latest_state = self.async_sim.get_latest_state()
                if latest_state:
                    # Update visualization based on latest state
                    try:
                        render_start = time.time()
                        
                        # Update the visualization
                        self.visualizer.update()
                        
                        self.async_sim.perf_metrics['render_time'] = time.time() - render_start
                    except Exception as e:
                        print(f"Error updating visualization: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Update frame counter
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                if elapsed >= 1.0:  # Update FPS every second
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_frame_time = current_time
                    
                    # Log state periodically
                    if self.logger and self.async_sim.step_count % self.config['save_interval'] == 0:
                        self.async_sim.log_state(self.logger)
        
        # Create timer for rendering
        timer = app.Timer(interval=1/60.0, connect=update, start=True)
        
        # Handle keyboard events for controls
        @self.visualizer.canvas.events.key_press.connect
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
            elif event.key == 'r':
                self._reset_simulation()
            elif event.key == ' ':  # Space bar
                self._toggle_pause()
            elif event.key == 'escape':
                self.finalize()
                app.quit()
        
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
        
        # Stop async simulator
        self.async_sim.stop()
        
        # Clear particles
        self.particle_system.particles.clear()
        self.particle_system.bonds.clear()
        if hasattr(self.particle_system, 'atom_groups'):
            self.particle_system.atom_groups.clear()
        
        # Reset grid
        self.grid = adaptive_fluid_grid(
            base_resolution=self.config['grid_resolution'], 
            size=self.config['grid_size'],
            max_depth=self.config.get('max_depth', 2),
            use_gpu=self.config['use_gpu']
        )
        
        # Reset physics
        self.physics = dwarf_physics(
            grid=self.grid, 
            use_gpu=self.config['use_gpu']
        )
        self.physics.set_bond_detector(self.bond_detector)
        
        # Create new particles
        self._create_initial_particles()
        
        # Reset time and step count
        self.time = 0.0
        self.step_count = 0
        
        # Create new async simulator
        self.async_sim = AsyncSimulator(
            physics_engine=self.physics,
            grid=self.grid,
            particle_system=self.particle_system,
            config=self.config
        )
        
        # Start the async simulator
        self.async_sim.start()
        
    def _toggle_pause(self):
        """Toggle pause/resume simulation"""
        self.running = not self.running
        print(f"Simulation {'paused' if not self.running else 'resumed'}")
    
    def finalize(self):
        """Finalize the simulation"""
        print("\nFinalizing simulation...\n")
        
        # Stop async simulator
        self.async_sim.stop()
        
        if self.logger:
            print("Saving simulation data...")
            self.logger.finalize()
            print("Data saved successfully.")
        
        # Print performance statistics
        elapsed = time.time() - self.start_time
        print(f"\nTotal simulation time: {elapsed:.2f} seconds")
        print(f"Average rendering FPS: {self.fps:.2f}")
        print(f"Average simulation FPS: {self.async_sim.perf_metrics['sim_fps']:.2f}")
        
        # Free GPU memory
        if self.config['use_gpu']:
            print("Freeing GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU memory released.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='dwarf Physics Simulator (Async Version)')
    
    parser.add_argument('--resolution', type=int, default=64,
                        help='Grid resolution (default: 64)')
    parser.add_argument('--size', type=float, default=10.0,
                        help='Grid size in Bohr radii (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (default: 0.01)')
    parser.add_argument('--protons', '--proton', type=int, default=1,
                        help='Initial number of protons (default: 1)')
    parser.add_argument('--electrons', '--electron', type=int, default=1,
                        help='Initial number of electrons (default: 1)')
    parser.add_argument('--neutrons', '--neutron', type=int, default=0,
                        help='Initial number of neutrons (default: 0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable data logging')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum grid refinement depth (default: 2)')
    parser.add_argument('--no-periodic', action='store_true',
                        help='Disable periodic boundary conditions')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting dwarf Physics Simulator (Async Version)")
    
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
    simulator = dwarf_simulator(config)
    simulator.initialize()
    simulator.run()