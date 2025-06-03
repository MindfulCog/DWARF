import numpy as np
import cupy as cp
import threading
import queue
import time
import copy
from concurrent.futures import ThreadPoolExecutor

class SimulationState:
    """Container for simulation state that can be shared between threads"""
    def __init__(self):
        self.particles = []
        self.velocity_field = None
        self.memory_field = None
        self.pressure_field = None
        self.vorticity_magnitude = None
        self.energy_density = None
        self.timestamp = 0.0
        self.step_count = 0
       
    
class AsyncSimulator:
    """Asynchronous simulator that runs GPU computation in a separate thread"""
    def update(event):
        if self.running:
            # Get the latest state from the async simulator
            latest_state = self.async_sim.get_latest_state()
            if latest_state:
                # Debug time
                if self.frame_count % 10 == 0:  # Print every 60 frames
                    print(f"Sim time: {latest_state.timestamp:.6f}, Steps: {latest_state.step_count}")

    def __init__(self, physics_engine, grid, particle_system, config):
        self.physics_engine = physics_engine
        self.grid = grid
        self.particle_system = particle_system
        self.config = config
        self.running = False
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # Simulation thread and state buffers
        self.sim_thread = None
        self.transfer_thread = None
        self.state_buffer = queue.Queue(maxsize=3)  # Limit buffer size
        self.latest_state = SimulationState()
        self.state_lock = threading.RLock()
        
        # Performance tracking
        self.perf_metrics = {
            'physics_time': 0.0,
            'transfer_time': 0.0,
            'render_time': 0.0,
            'fps': 0.0,
            'sim_fps': 0.0
        }
        self.last_metrics_update = time.time()
        self.frame_count = 0
        self.sim_frame_count = 0
        
        # Event flags for thread communication
        self.stop_event = threading.Event()
        
        # Thread pool for misc operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Setup CUDA streams for async operations if using GPU
        self.use_gpu = self.config.get('use_gpu', True)
        if self.use_gpu:
            self.streams = {
                'compute': cp.cuda.Stream(non_blocking=True),
                'transfer': cp.cuda.Stream(non_blocking=True),
            }
        
    def start(self):
        """Start simulation and transfer threads"""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        
        # Initialize the simulation before starting threads
        print("Initializing simulation state...")
        self._initialize_simulation()
        
        # Start threads
        print("Starting simulation thread...")
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        print("Starting transfer thread...")
        self.transfer_thread = threading.Thread(target=self._transfer_loop)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()
        
        print("Async simulation started")
        
    def stop(self):
        """Stop all simulation threads"""
        if not self.running:
            return
            
        print("Stopping async simulation...")
        self.stop_event.set()
        
        if self.sim_thread:
            self.sim_thread.join(timeout=2.0)
            
        if self.transfer_thread:
            self.transfer_thread.join(timeout=2.0)
            
        self.running = False
        print("Async simulation stopped")
        
    def _initialize_simulation(self):
        """Initialize the simulation with detailed progress reporting"""
        print("Initializing grid...")
        start_time = time.time()
        
        # Initialize the grid - this is likely the bottleneck
        try:
            # Break initialization into smaller, more measurable steps
            print("  Creating base fields...")
            self.grid._init_fields()  # Make sure fields are allocated
            
            print("  Creating initial vortex structure...")
            # Use vectorized operations instead of nested loops
            shape = self.grid.velocity_field.shape[:3]
            center = self.grid.base_resolution // 2
            
            # Create coordinate arrays (vectorized approach)
            x, y, z = np.meshgrid(
                np.arange(shape[0]) - center,
                np.arange(shape[1]) - center, 
                np.arange(shape[2]) - center,
                indexing='ij'
            )
            
            if self.use_gpu:
                x = cp.asarray(x)
                y = cp.asarray(y)
                z = cp.asarray(z)
                
            # Calculate distance from center (vectorized)
            r = self.grid.xp.sqrt(x*x + y*y + z*z)
            r_nonzero = self.grid.xp.maximum(r, 0.1)  # Avoid division by zero
            
            print("  Computing memory field...")
            # Memory field (vectorized)
            self.grid.memory_field[..., 0] = 0.1 * x / r_nonzero * self.grid.xp.exp(-0.02 * r)
            self.grid.memory_field[..., 1] = 0.1 * y / r_nonzero * self.grid.xp.exp(-0.02 * r)
            self.grid.memory_field[..., 2] = 0.1 * z / r_nonzero * self.grid.xp.exp(-0.02 * r)
            
            print("  Computing velocity field...")
            # Calculate theta (vectorized)
            theta = self.grid.xp.arctan2(y, x)
            
            # Velocity field (vectorized)
            self.grid.velocity_field[..., 0] = -0.05 * y / (r_nonzero) * self.grid.xp.exp(-0.01 * r)
            self.grid.velocity_field[..., 1] = 0.05 * x / (r_nonzero) * self.grid.xp.exp(-0.01 * r)
            self.grid.velocity_field[..., 2] = 0.02 * self.grid.xp.sin(2*theta) * self.grid.xp.exp(-0.01 * r)
            
            print("  Computing derived fields...")
            # Update derived fields
            self._update_pressure_field_vectorized()
            self._update_vorticity_field_vectorized()
            self._update_energy_density_vectorized()
            
        except Exception as e:
            print(f"Error in grid initialization: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"Grid initialized in {time.time() - start_time:.2f} seconds")
        
        # Initialize the physics engine
        print("Initializing physics engine...")
        start_time = time.time()
        try:
            self.physics_engine.initialize()
        except Exception as e:
            print(f"Error in physics initialization: {e}")
            import traceback
            traceback.print_exc()
        print(f"Physics initialized in {time.time() - start_time:.2f} seconds")
        
        # Create initial state
        print("Creating initial state snapshot...")
        initial_state = SimulationState()
        initial_state.particles = copy.deepcopy(self.particle_system.particles)
        
        # Store grid fields in initial state
        if self.use_gpu:
            print("Creating GPU snapshot (non-blocking)...")
            # Use streams for non-blocking copies
            with cp.cuda.Stream(non_blocking=True):
                initial_state.velocity_field = cp.asnumpy(self.grid.velocity_field)
                initial_state.memory_field = cp.asnumpy(self.grid.memory_field)
                initial_state.pressure_field = cp.asnumpy(self.grid.pressure_field)
                initial_state.vorticity_magnitude = cp.asnumpy(self.grid.vorticity_magnitude)
                initial_state.energy_density = cp.asnumpy(self.grid.energy_density)
        else:
            print("Creating CPU snapshot...")
            # Copy for CPU (numpy arrays)
            initial_state.velocity_field = np.copy(self.grid.velocity_field)
            initial_state.memory_field = np.copy(self.grid.memory_field)
            initial_state.pressure_field = np.copy(self.grid.pressure_field)
            initial_state.vorticity_magnitude = np.copy(self.grid.vorticity_magnitude)
            initial_state.energy_density = np.copy(self.grid.energy_density)
            
        initial_state.timestamp = self.time
        initial_state.step_count = self.step_count
        
        # Update latest state
        print("Updating latest state...")
        with self.state_lock:
            self.latest_state = initial_state
            self.time = initial_state.timestamp
            self.step_count = initial_state.step_count
        # Add to buffer (non-blocking, could drop if full)
        try:
            self.state_buffer.put_nowait(initial_state)
        except queue.Full:
            pass
        
        print("Initialization complete. Starting simulation threads...")
    
    def _update_pressure_field_vectorized(self):
        """Update pressure field using vectorized operations"""
        res = self.grid.base_resolution
        
        # Create shifted arrays for derivatives
        if self.use_gpu:
            # Using CuPy's roll function for periodic boundary
            vx_next = cp.roll(self.grid.velocity_field[..., 0], -1, axis=0)
            vx_prev = cp.roll(self.grid.velocity_field[..., 0], 1, axis=0)
            vy_next = cp.roll(self.grid.velocity_field[..., 1], -1, axis=1)
            vy_prev = cp.roll(self.grid.velocity_field[..., 1], 1, axis=1)
            vz_next = cp.roll(self.grid.velocity_field[..., 2], -1, axis=2)
            vz_prev = cp.roll(self.grid.velocity_field[..., 2], 1, axis=2)
        else:
            # Using NumPy's roll function
            vx_next = np.roll(self.grid.velocity_field[..., 0], -1, axis=0)
            vx_prev = np.roll(self.grid.velocity_field[..., 0], 1, axis=0)
            vy_next = np.roll(self.grid.velocity_field[..., 1], -1, axis=1)
            vy_prev = np.roll(self.grid.velocity_field[..., 1], 1, axis=1)
            vz_next = np.roll(self.grid.velocity_field[..., 2], -1, axis=2)
            vz_prev = np.roll(self.grid.velocity_field[..., 2], 1, axis=2)
        
        # Calculate divergence with central differences (vectorized)
        div_x = (vx_next - vx_prev) / 2.0
        div_y = (vy_next - vy_prev) / 2.0
        div_z = (vz_next - vz_prev) / 2.0
        
        # Calculate divergence
        div = div_x + div_y + div_z
        
        # Update pressure (negative divergence relationship)
        self.grid.pressure_field -= div * 0.1
        
        # Apply diffusion for stability (using a simple smoothing)
        if self.use_gpu:
            self.grid.pressure_field = cp.asarray(
                cp.asnumpy(self.grid.pressure_field), 
                dtype=cp.float32
            )
        else:
            self.grid.pressure_field = self.grid.pressure_field * 0.98

    def _update_vorticity_field_vectorized(self):
        """Update vorticity field using vectorized operations"""
        res = self.grid.base_resolution
        
        # Create shifted arrays for derivatives
        xp = self.grid.xp
        
        # Use roll for periodic boundary conditions
        vx = self.grid.velocity_field[..., 0]
        vy = self.grid.velocity_field[..., 1]
        vz = self.grid.velocity_field[..., 2]
        
        # X derivatives
        vx_next = xp.roll(vx, -1, axis=0)
        vx_prev = xp.roll(vx, 1, axis=0)
        
        # Y derivatives
        vy_next = xp.roll(vy, -1, axis=1)
        vy_prev = xp.roll(vy, 1, axis=1)
        
        # Z derivatives
        vz_next = xp.roll(vz, -1, axis=2)
        vz_prev = xp.roll(vz, 1, axis=2)
        
        # Calculate derivatives
        dvx_dy = (xp.roll(vx, -1, axis=1) - xp.roll(vx, 1, axis=1)) / 2.0
        dvx_dz = (xp.roll(vx, -1, axis=2) - xp.roll(vx, 1, axis=2)) / 2.0
        
        dvy_dx = (xp.roll(vy, -1, axis=0) - xp.roll(vy, 1, axis=0)) / 2.0
        dvy_dz = (xp.roll(vy, -1, axis=2) - xp.roll(vy, 1, axis=2)) / 2.0
        
        dvz_dx = (xp.roll(vz, -1, axis=0) - xp.roll(vz, 1, axis=0)) / 2.0
        dvz_dy = (xp.roll(vz, -1, axis=1) - xp.roll(vz, 1, axis=1)) / 2.0
        
        # Create temporary vorticity field
        vorticity = xp.zeros_like(self.grid.velocity_field)
        
        # Compute curl components
        vorticity[..., 0] = dvz_dy - dvy_dz  # x component
        vorticity[..., 1] = dvx_dz - dvz_dx  # y component
        vorticity[..., 2] = dvy_dx - dvx_dy  # z component
        
        # Calculate magnitude
        self.grid.vorticity_magnitude = xp.sqrt(
            vorticity[..., 0]**2 + vorticity[..., 1]**2 + vorticity[..., 2]**2
        )

    def _update_energy_density_vectorized(self):
        """Update energy density using vectorized operations"""
        # Calculate velocity squared (vectorized)
        velocity_squared = self.grid.xp.sum(self.grid.velocity_field**2, axis=3)
        
        # Kinetic energy (with constant density of 1.0)
        kinetic_energy = 0.5 * velocity_squared
        
        # Add pressure contribution
        pressure_energy = self.grid.pressure_field * 0.1
        
        # Update energy with damping factor
        self.grid.energy_density = self.grid.energy_density * 0.98 + (kinetic_energy + pressure_energy) * 0.02
    
    def _simulation_loop(self):
        """Main simulation loop running in a separate thread"""
        print("Simulation thread started")
        steps_completed = 0
        
        try:
            while not self.stop_event.is_set():
                # Simple, direct physics update
                self.physics_engine.update(self.particle_system, self.config['time_step'])
                self.grid.update(self.particle_system.particles, self.config['time_step'])
                
                # CRITICAL: Update time (make sure this executes)
                self.step_count += 1
                self.time += self.config['time_step']
                
                # Print progress every 100 steps
                steps_completed += 1
                if steps_completed % 100 == 0:
                    print(f"Simulation progress: step {self.step_count}, time: {self.time:.3f}")
                
                # Brief sleep to avoid CPU overload
                time.sleep(0.001)
        
        except Exception as e:
            print(f"ERROR in simulation loop: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("Simulation thread stopped")
    
    def _step_simulation(self):
        """Run a single simulation step"""
        
        self.physics_engine.update(self.particle_system, self.config['time_step'])
        
        # Update grid with the latest particle state
        self.grid.update(self.particle_system.particles, self.config['time_step'])
        
        # Update simulation state
        self.time += self.config['time_step']
        self.step_count += 1

    def _transfer_loop(self):
        """Transfer data from GPU to CPU in a separate thread"""
        print("Transfer thread started")
        
        try:
            while not self.stop_event.is_set():
                # Measure transfer time
                transfer_start = time.time()
                
                # Create a new state snapshot
                new_state = SimulationState()
                new_state.timestamp = self.time
                new_state.step_count = self.step_count
                
                # Deep copy particles to avoid thread safety issues
                new_state.particles = copy.deepcopy(self.particle_system.particles)
                
                # Non-blocking copy of GPU data
                if self.use_gpu:
                    # Using CUDA streams for non-blocking transfers
                    with cp.cuda.Stream(non_blocking=True):
                        # Asynchronous copy of grid fields
                        new_state.velocity_field = cp.asnumpy(self.grid.velocity_field)
                        new_state.memory_field = cp.asnumpy(self.grid.memory_field)
                        new_state.pressure_field = cp.asnumpy(self.grid.pressure_field)
                        new_state.vorticity_magnitude = cp.asnumpy(self.grid.vorticity_magnitude)
                        new_state.energy_density = cp.asnumpy(self.grid.energy_density)
                else:
                    # For CPU mode, just make a copy
                    new_state.velocity_field = np.copy(self.grid.velocity_field)
                    new_state.memory_field = np.copy(self.grid.memory_field)
                    new_state.pressure_field = np.copy(self.grid.pressure_field)
                    new_state.vorticity_magnitude = np.copy(self.grid.vorticity_magnitude)
                    new_state.energy_density = np.copy(self.grid.energy_density)
                
                # Update latest state (used for rendering)
                with self.state_lock:
                    self.latest_state = new_state
                    self.time = new_state.timestamp
                    self.step_count = new_state.step_count
                # Try to add to buffer queue (non-blocking)
                try:
                    self.state_buffer.put_nowait(new_state)
                except queue.Full:
                    # Buffer full, drop this frame
                    pass
                
                # Record transfer time
                self.perf_metrics['transfer_time'] = time.time() - transfer_start
                
                # Don't transfer too frequently to avoid overwhelming the render thread
                time.sleep(0.05)  # Max 20 transfers per second
        
        except Exception as e:
            print(f"Error in transfer thread: {e}")
            import traceback
            traceback.print_exc()
        
        print("Transfer thread stopped")
    
    def get_latest_state(self):
        """Get the latest state for rendering"""
        with self.state_lock:
            return self.latest_state
    
    def get_next_state(self, timeout=0.01):
        """Get the next state from the buffer, with optional timeout"""
        try:
            return self.state_buffer.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def log_state(self, logger):
        """Log current state using the provided logger"""
        if logger:
            with self.state_lock:
                current_time = self.latest_state.timestamp  # ✅ Use advancing time from simulation
            self.thread_pool.submit(logger.log_state, 
                               current_time,  # ✅ Pass the advancing time
                               self.particle_system, 
                               self.grid)