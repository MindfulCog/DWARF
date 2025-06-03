import numpy as np
import cupy as cp
from gpu_memory_manager import GPUMemoryManager
from dwarf_math import DwarfMathGPU
from dwarf_physics import DwarfPhysicsGPU
from adaptive_fluid_grid import AdaptiveFluidGridGPU
from particle import Particle

class SimulationController:
    def __init__(self, use_gpu=True):
        # Check if GPU is available
        self.use_gpu = use_gpu and cp.cuda.is_available()
        
        if self.use_gpu:
            print("Using GPU acceleration")
            self.xp = cp 
            self.memory_manager = GPUMemoryManager()
        else:
            print("Using CPU (GPU not available or disabled)")
            self.xp = np
            self.memory_manager = None
        
        # Initialize physics, math, and grid components
        self.physics = DwarfPhysicsGPU(use_gpu=self.use_gpu)
        self.math = DwarfMathGPU(use_gpu=self.use_gpu)
        self.grid = AdaptiveFluidGridGPU(use_gpu=self.use_gpu)
        
        # Simulation parameters
        self.visualization_frequency = 10  # Visualize every 10 steps
        self.particles = []  # Will be populated with particle objects
        
        # Statistics tracking
        self.performance_stats = {
            'step_times': [],
            'physics_times': [],
            'field_update_times': [],
            'grid_sync_times': []
        }
        
    def initialize_simulation(self, num_particles=100, box_size=10.0):
        """Initialize simulation with random particles"""
        # Create particles with random positions, velocities, etc.
        self.particles = []
        
        import random
        for i in range(num_particles):
            particle = Particle()
            # Random position within box
            particle.position = np.array([
                random.uniform(-box_size/2, box_size/2),
                random.uniform(-box_size/2, box_size/2),
                random.uniform(-box_size/2, box_size/2)
            ])
            # Random velocity
            particle.velocity = np.array([
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0)
            ])
            # Random mass between 1 and 10
            particle.mass = random.uniform(1.0, 10.0)
            # Random charge between -1 and 1
            particle.charge = random.uniform(-1.0, 1.0)
            # Initialize force to zero
            particle.force = np.zeros(3)
            
            # Initialize spin properties for DWARF physics
            particle.spin = np.array([
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1)
            ])
            particle.torque = np.zeros(3)
            particle.moment_of_inertia = random.uniform(0.5, 2.0)
            
            self.particles.append(particle)
        
        # Initialize grid with random values
        self.grid.base_grid = self.xp.random.random(
            (self.grid.base_resolution, self.grid.base_resolution, self.grid.base_resolution, 3)
        ).astype(self.xp.float32)  # Vector field with 3 components
        
        # Initialize memory field
        self.grid.memory_field = self.xp.zeros_like(self.grid.base_grid)
        
        # Initial sync of hierarchical grids
        self.grid.sync_hierarchical_data()
        
        # Pre-transfer static data to GPU if using GPU
        if self.use_gpu:
            self.memory_manager.transfer_to_gpu("particles", self.particles)
        
    def run_simulation(self, num_steps, dt):
        """Run the simulation with GPU optimization"""
        import time
        
        for step in range(num_steps):
            step_start_time = time.time()
            
            # 1. Execute GPU-optimized physics calculations with CORRECT DWARF PHYSICS
            physics_start_time = time.time()
            self.physics.calculate_particle_interactions(self.particles, self.grid)
            self.physics.update_particles(self.particles, dt)
            physics_time = time.time() - physics_start_time
            
            # 2. Update grid fields - this part was correct in the original optimization
            field_start_time = time.time()
            self.math.update_memory_field(self.grid, dt)
            field_time = time.time() - field_start_time
            
            # 3. Identify regions for grid refinement
            self.grid.identify_refinement_regions()
            
            # 4. Synchronize hierarchical grid data
            grid_start_time = time.time()
            self.grid.sync_hierarchical_data()
            grid_time = time.time() - grid_start_time
            
            # 5. Transfer data to CPU only when needed (e.g., for visualization)
            if step % self.visualization_frequency == 0:
                if self.use_gpu:
                    # Need to transfer from GPU to CPU for visualization
                    cpu_particles = self.memory_manager.transfer_to_cpu("particles")
                    # Set grid to transfer on next sync
                    self.grid.needs_cpu_sync = True
                    self.grid.sync_hierarchical_data()
                    self.grid.needs_cpu_sync = False
                    
                    # Visualize simulation state
                    self.visualize(cpu_particles, self.grid)
                else:
                    # Already on CPU, just visualize
                    self.visualize(self.particles, self.grid)
            
            # Record performance stats
            step_time = time.time() - step_start_time
            self.performance_stats['step_times'].append(step_time)
            self.performance_stats['physics_times'].append(physics_time)
            self.performance_stats['field_update_times'].append(field_time)
            self.performance_stats['grid_sync_times'].append(grid_time)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{num_steps} completed in {step_time:.4f}s "
                     f"(Physics: {physics_time:.4f}s, Field: {field_time:.4f}s, Grid: {grid_time:.4f}s)")
    
    def visualize(self, particles, grid):
        """Visualize the current simulation state"""
        # This would typically use matplotlib or another visualization library
        # For simplicity, just print some summary statistics
        pos_array = np.array([p.position for p in particles])
        vel_array = np.array([p.velocity for p in particles])
        spin_array = np.array([p.spin for p in particles])
        
        print("\nDWARF Simulation Visualization")
        print(f"Particle position range: {pos_array.min():.2f} to {pos_array.max():.2f}")
        print(f"Particle velocity range: {vel_array.min():.2f} to {vel_array.max():.2f}")
        print(f"Particle spin range: {spin_array.min():.2f} to {spin_array.max():.2f}")
        print(f"Memory field range: {np.min(grid.memory_field):.2f} to {np.max(grid.memory_field):.2f}")
        print(f"Number of active refinement regions: {sum(len(regions) for regions in grid.refinement_regions)}")
        print("")