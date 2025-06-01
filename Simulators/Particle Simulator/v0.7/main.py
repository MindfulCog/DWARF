import sys
import numpy as np
import cupy as cp
import argparse
from vispy import app

# Fixed imports
from dwarf_math import DWARFMath
from dwarf_particle import Particle, Proton, Electron, Neutron, ParticleSystem
from adaptive_fluid_grid import AdaptiveFluidGrid  # Use adaptive grid
from dwarf_physics import DWARFPhysics
from bond_detector import BondDetector
from data_logger import DataLogger
from visualizer_3d import Visualizer3D

class DWARFSimulator:
    """Main simulator class for DWARF physics"""
    
    def __init__(self, config=None):
        """Initialize simulator with optional config"""
        # Default configuration
        self.config = {
            'grid_resolution': 128,
            'grid_size': 10.0,
            'time_step': 0.01,
            'save_interval': 100,  # Save data every 100 steps
            'window_size': (1200, 800),
            'seed': None,  # Random seed
            'initial_protons': 3,
            'initial_electrons': 3,
            'initial_neutrons': 1,
            'record_data': True,
            'use_gpu': True,  # GPU acceleration flag
            'max_depth': 2     # Maximum refinement depth for adaptive grid
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
            except:
                print("GPU info not available")
            
        # Initialize components
        self.grid = AdaptiveFluidGrid(
            base_resolution=self.config['grid_resolution'], 
            size=self.config['grid_size'],
            max_depth=self.config.get('max_depth', 2)
        )
        
        self.particle_system = ParticleSystem()
        
        self.physics = DWARFPhysics(
            grid=self.grid, 
            time_step=self.config['time_step']
        )
        
        self.bond_detector = BondDetector(self.particle_system)
        self.physics.set_bond_detector(self.bond_detector)
        
        self.logger = DataLogger() if self.config['record_data'] else None
        
        self.visualizer = Visualizer3D(window_size=self.config['window_size'])
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.running = False
        self.selected_particle = None
        
    def initialize(self):
        """Initialize the simulation"""
        # Initialize components
        self.grid.initialize()
        self.physics.initialize()
        
        if self.logger:
            self.logger.initialize()
        
        # Create initial particles
        self._create_initial_particles()
        
        # Initialize visualization
        self.visualizer.initialize(self.grid, self.particle_system)
        
        # Set running state
        self.running = True
        
    def _create_initial_particles(self):
        """Create initial set of particles"""
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
            vel = (np.random.random(3) - 0.5) * 0.5  # Electrons move faster
            self.particle_system.add(Electron(pos, vel, spin))
            
        # Create neutrons
        for _ in range(self.config['initial_neutrons']):
            pos = (np.random.random(3) - 0.5) * self.config['grid_size'] * 0.5
            spin = np.random.random(3) - 0.5
            spin = spin / np.linalg.norm(spin)
            vel = (np.random.random(3) - 0.5) * 0.1
            self.particle_system.add(Neutron(pos, vel, spin))
    
    def step(self):
        """Advance simulation by one time step"""
        # Update physics
        self.physics.update(self.particle_system, self.config['time_step'])
        
        # Update grid - capture number of active refinement regions
        active_regions = self.grid.update(self.particle_system.particles, self.config['time_step'])
        
        # Log state if logger is enabled
        if self.logger and self.step_count % self.config['save_interval'] == 0:
            self.logger.log_state(self.time, self.particle_system, self.grid)
            
            # Log refinement statistics if available
            if hasattr(self.grid, 'refinement_regions'):
                print(f"Active refinement regions: {len(self.grid.refinement_regions)}")
        
        # Update visualization
        self.visualizer.update()
        
        # Update simulation state
        self.time += self.config['time_step']
        self.step_count += 1
        
    def run(self, max_steps=None):
        """Run the simulation"""
        # Set up timer for animation
        def update(event):
            if self.running:
                self.step()
                
                if max_steps and self.step_count >= max_steps:
                    self.running = False
                    self.finalize()
        
        # Create timer
        timer = app.Timer(interval=0.02, connect=update, start=True)
        
        # Start the application
        if sys.flags.interactive == 0:
            app.run()
            
    def finalize(self):
        """Finalize the simulation"""
        if self.logger:
            print("Saving simulation data...")
            self.logger.finalize()
            print("Data saved successfully.")
        
        print(f"Simulation completed: {self.step_count} steps, {self.time:.2f} time units")
        print(f"Final particle count: {len(self.particle_system.particles)}")
        print(f"Bonds formed: {len(self.particle_system.bonds)}")
        
        if "hydrogen" in self.particle_system.atom_groups:
            print(f"Hydrogen atoms: {len(self.particle_system.atom_groups['hydrogen'])}")
        if "helium" in self.particle_system.atom_groups:
            print(f"Helium atoms: {len(self.particle_system.atom_groups['helium'])}")
            
        # Free GPU memory
        if self.config['use_gpu']:
            print("Freeing GPU memory...")
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU memory released.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DWARF Physics Simulator')
    
    parser.add_argument('--resolution', type=int, default=128,
                        help='Grid resolution (default: 128)')
    parser.add_argument('--size', type=float, default=10.0,
                        help='Grid size in Bohr radii (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (default: 0.01)')
    parser.add_argument('--protons', type=int, default=3,
                        help='Initial number of protons (default: 3)')
    parser.add_argument('--electrons', type=int, default=3,
                        help='Initial number of electrons (default: 3)')
    parser.add_argument('--neutrons', type=int, default=1,
                        help='Initial number of neutrons (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable data logging')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum grid refinement depth (default: 2)')
    
    return parser.parse_args()


if __name__ == "__main__":
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
        'max_depth': args.max_depth  # Add refinement depth parameter
    }
    
    print("Starting DWARF Physics Simulator")
    print(f"GPU Acceleration: {'Enabled' if not args.cpu else 'Disabled'}")
    print(f"Grid: {config['grid_resolution']}^3 cells, {config['grid_size']} Bohr radii")
    print(f"Particles: {config['initial_protons']} protons, {config['initial_electrons']} electrons, {config['initial_neutrons']} neutrons")
    print(f"Adaptive Grid: Max Depth {config['max_depth']} (up to {2**config['max_depth']}x refinement)")
    print("Controls:")
    print("  p: spawn proton")
    print("  e: spawn electron")
    print("  n: spawn neutron")
    print("  1-5: toggle visualization overlays")
    print("  x/y/z: set spin direction of selected particle")
    
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
    simulator = DWARFSimulator(config)
    simulator.initialize()
    
    # Show the canvas explicitly
    simulator.visualizer.canvas.show()
    
    # Run the application
    simulator.run()