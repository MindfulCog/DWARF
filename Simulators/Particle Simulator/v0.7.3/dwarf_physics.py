import numpy as np
import cupy as cp
import time

class dwarf_physics:
    def __init__(self, constants=None, use_gpu=True, grid=None, **kwargs):
        # Check if GPU is available
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
        # Initialize physics constants
        if constants is None:
            constants = physics_constants()
        self.constants = constants
        
        # Store grid reference if provided
        self.grid = grid
        
        # Performance tracking
        self.last_calculation_time = 0.0
        
        # Additional parameters that might be passed
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Add bond detector reference
        self.bond_detector = None
        
    def initialize(self):
        """Initialize physics engine"""
        # This method can be expanded as needed
        pass
        
    def set_bond_detector(self, bond_detector):
        """Set the bond detector for the physics engine"""
        self.bond_detector = bond_detector
        
    def update(self, particle_system, dt):
        """Update all particles and physics for one time step"""
        start_time = time.time()
        
        # Calculate all forces and interactions
        self.calculate_particle_interactions(particle_system.particles, self.grid)
        
        # Update all particle positions and velocities with periodic boundaries
        self.update_particles(particle_system.particles, dt)
        
        # Update bond detector if available
        if self.bond_detector:
            self.bond_detector.update()
            
        # Store calculation time for performance metrics
        self.last_calculation_time = time.time() - start_time
        
    def calculate_particle_interactions(self, particles, grid=None):
        """GPU-accelerated particle interactions with correct dwarf physics"""
        # Use provided grid or fall back to stored grid
        if grid is None:
            grid = self.grid
            if grid is None:
                raise ValueError("No grid provided for particle interactions")
                
        num_particles = len(particles)
        
        if self.use_gpu and num_particles > 0:
            # Transfer particle data to GPU
            positions = np.array([p.position for p in particles])
            masses = np.array([p.mass for p in particles])
            charges = np.array([p.charge for p in particles])
            spins = np.array([p.spin for p in particles])
            
            positions_gpu = cp.asarray(positions)
            masses_gpu = cp.asarray(masses)
            charges_gpu = cp.asarray(charges)
            spins_gpu = cp.asarray(spins)
            
            # Preallocate forces and torques array
            forces_gpu = cp.zeros((num_particles, 3), dtype=cp.float32)
            torques_gpu = cp.zeros((num_particles, 3), dtype=cp.float32)
            
            # Define the grid size for periodic boundary calculations
            grid_size = grid.size  # Physical size of the grid
            
            # Create interaction kernel with CORRECT dwarf FORCE LAW (r^2.22)
            # and PERIODIC BOUNDARY CONDITIONS
            interaction_kernel_code = r'''
            extern "C" __global__ void dwarf_particle_forces(
                const float* positions, const float* masses, const float* charges,
                float* forces, int num_particles, float k_dwarf, float grid_size
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < num_particles) {
                    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
                    float xi = positions[i*3], yi = positions[i*3+1], zi = positions[i*3+2];
                    float qi = charges[i];
                    
                    for (int j = 0; j < num_particles; ++j) {
                        if (i == j) continue;
                        
                        float xj = positions[j*3], yj = positions[j*3+1], zj = positions[j*3+2];
                        float qj = charges[j];
                        
                        // Calculate distance with periodic boundary conditions
                        float dx = xj - xi;
                        float dy = yj - yi;
                        float dz = zj - zi;
                        
                        // Apply periodic boundary conditions
                        // This takes the shortest path across the periodic domain
                        if (dx > grid_size/2) dx -= grid_size;
                        else if (dx < -grid_size/2) dx += grid_size;
                        
                        if (dy > grid_size/2) dy -= grid_size;
                        else if (dy < -grid_size/2) dy += grid_size;
                        
                        if (dz > grid_size/2) dz -= grid_size;
                        else if (dz < -grid_size/2) dz += grid_size;
                        
                        float r_squared = dx*dx + dy*dy + dz*dz;
                        float r = sqrt(r_squared);
                        
                        if (r > 1e-10f) {
                            // CORRECT dwarf FORCE LAW: r^2.22 exponent
                            float r_222 = pow(r_squared, 1.11f);  // r^2.22 = (r^2)^1.11
                            float force_magnitude = k_dwarf * qi * qj / r_222;
                            
                            fx += force_magnitude * dx / r;
                            fy += force_magnitude * dy / r;
                            fz += force_magnitude * dz / r;
                        }
                    }
                    
                    forces[i*3] = fx;
                    forces[i*3+1] = fy;
                    forces[i*3+2] = fz;
                }
            }
            '''
            
            interaction_kernel = cp.RawKernel(interaction_kernel_code, 'dwarf_particle_forces')
            
            # Configure grid and block dimensions
            threads_per_block = 256
            blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block
            
            # Execute kernel with dwarf force constant
            k_dwarf = self.constants.k_dwarf  # dwarf force constant
            interaction_kernel((blocks_per_grid,), (threads_per_block,), 
                              (positions_gpu, masses_gpu, charges_gpu, forces_gpu, 
                               num_particles, k_dwarf, grid_size))
            
            # Calculate memory field gradient forces and torques
            memory_forces, memory_torques = self.calculate_memory_field_interactions(
                particles, grid, positions_gpu, spins_gpu)
            
            # Add memory field forces to particle forces
            forces_gpu += memory_forces
            
            # Transfer results back to CPU
            forces = cp.asnumpy(forces_gpu)
            torques = cp.asnumpy(torques_gpu + memory_torques)
            
            # Update particle forces and torques
            for i, particle in enumerate(particles):
                particle.force = forces[i]
                particle.torque = torques[i]
        else:
            # CPU fallback with correct dwarf physics and periodic boundaries
            grid_size = grid.size  # Physical size of the grid
            
            for i, p1 in enumerate(particles):
                # Reset forces and torques
                p1.force = np.zeros(3)
                p1.torque = np.zeros(3)
                
                # Particle-particle dwarf force (2.22 exponent)
                for j in range(i+1, num_particles):
                    p2 = particles[j]
                    force = self.calculate_dwarf_force(p1, p2, grid_size)
                    p1.force += force
                    p2.force -= force
                
                # Memory field gradient force
                memory_force = self.calculate_memory_field_force(p1, grid)
                p1.force += memory_force
                
                # Memory field torque (spin update)
                memory_torque = self.calculate_memory_torque(p1, grid)
                p1.torque += memory_torque
    
    def calculate_memory_field_interactions(self, particles, grid, positions_gpu=None, spins_gpu=None):
        """Calculate forces and torques from memory field gradients and curls (GPU-optimized)"""
        num_particles = len(particles)
        
        if self.use_gpu and num_particles > 0:
            # Transfer memory field to GPU if not already there
            if isinstance(grid.memory_field, np.ndarray):
                memory_field_gpu = cp.asarray(grid.memory_field)
            else:
                memory_field_gpu = grid.memory_field
                
            # Calculate gradient of memory field (fix for 4D field)
            dx = grid.cell_size
            
            # Handle each vector component separately
            grad_x = []
            grad_y = []
            grad_z = []
            
            # For each vector component, calculate its spatial gradient
            for i in range(memory_field_gpu.shape[3]):
                # Get gradient for this component with periodic boundary conditions
                gx, gy, gz = self.calculate_periodic_gradient(memory_field_gpu[..., i], dx)
                grad_x.append(gx)
                grad_y.append(gy)
                grad_z.append(gz)
            
            # Stack components
            grad_x = cp.stack(grad_x, axis=3)
            grad_y = cp.stack(grad_y, axis=3)
            grad_z = cp.stack(grad_z, axis=3)
            
            # Calculate curl of memory field with periodic boundaries
            curl_x, curl_y, curl_z = self.calculate_periodic_curl(grid, memory_field_gpu)
            
            # Use positions_gpu if provided, otherwise create from particles
            if positions_gpu is None:
                positions = np.array([p.position for p in particles])
                positions_gpu = cp.asarray(positions)
                
            if spins_gpu is None:
                spins = np.array([p.spin for p in particles])
                spins_gpu = cp.asarray(spins)
                
            # Preallocate force and torque arrays
            memory_forces = cp.zeros((num_particles, 3), dtype=cp.float32)
            memory_torques = cp.zeros((num_particles, 3), dtype=cp.float32)
            
            # CPU-based calculation for safety
            positions_cpu = cp.asnumpy(positions_gpu)
            spins_cpu = cp.asnumpy(spins_gpu)
            grad_x_cpu = cp.asnumpy(grad_x[..., 0])  # Use first component for simplicity
            grad_y_cpu = cp.asnumpy(grad_y[..., 0])
            grad_z_cpu = cp.asnumpy(grad_z[..., 0])
            curl_x_cpu = cp.asnumpy(curl_x)
            curl_y_cpu = cp.asnumpy(curl_y)
            curl_z_cpu = cp.asnumpy(curl_z)
            
            # Calculate forces and torques on CPU
            forces_cpu = np.zeros((num_particles, 3), dtype=np.float32)
            torques_cpu = np.zeros((num_particles, 3), dtype=np.float32)
            
            grid_size = grid.base_resolution
            half_size = grid_size / 2
            alpha = self.constants.alpha  # Memory field gradient force constant
            beta = self.constants.beta    # Memory field curl torque constant
            
            for i in range(num_particles):
                # Get particle position and spin
                px, py, pz = positions_cpu[i]
                sx, sy, sz = spins_cpu[i]
                
                # Convert position to grid indices with wrapping
                # First normalize to [0, grid_size] range
                px_wrapped = ((px / grid.cell_size) + half_size) % grid_size
                py_wrapped = ((py / grid.cell_size) + half_size) % grid_size
                pz_wrapped = ((pz / grid.cell_size) + half_size) % grid_size
                
                # Convert to integer indices with proper interpolation
                x = int(px_wrapped)
                y = int(py_wrapped)
                z = int(pz_wrapped)
                
                # Ensure within bounds (should be anyway due to wrapping)
                x = x % grid_size
                y = y % grid_size
                z = z % grid_size
                
                # Get gradient vector
                gx = grad_x_cpu[x, y, z]
                gy = grad_y_cpu[x, y, z]
                gz = grad_z_cpu[x, y, z]
                
                # Get curl vector
                cx = curl_x_cpu[x, y, z]
                cy = curl_y_cpu[x, y, z]
                cz = curl_z_cpu[x, y, z]
                
                # Calculate force
                forces_cpu[i, 0] = alpha * gx
                forces_cpu[i, 1] = alpha * gy
                forces_cpu[i, 2] = alpha * gz
                
                # Calculate torque
                torques_cpu[i, 0] = beta * (sy * cz - sz * cy)
                torques_cpu[i, 1] = beta * (sz * cx - sx * cz)
                torques_cpu[i, 2] = beta * (sx * cy - sy * cx)
            
            # Transfer back to GPU
            memory_forces = cp.asarray(forces_cpu)
            memory_torques = cp.asarray(torques_cpu)
            
            return memory_forces, memory_torques
        else:
            # CPU implementation
            memory_forces = np.zeros((num_particles, 3))
            memory_torques = np.zeros((num_particles, 3))
            
            for i, particle in enumerate(particles):
                # Calculate memory field force and torque
                force = self.calculate_memory_field_force(particle, grid)
                torque = self.calculate_memory_torque(particle, grid)
                
                memory_forces[i] = force
                memory_torques[i] = torque
                
            return memory_forces, memory_torques
    
    def calculate_periodic_gradient(self, field, dx):
        """Calculate gradient with periodic boundary conditions"""
        if self.use_gpu:
            # Use CuPy's gradient function but ensure periodicity
            # We need to pad the field to handle the boundaries
            padded_field = cp.pad(field, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            
            # Calculate gradient on padded field
            gx_padded, gy_padded, gz_padded = cp.gradient(padded_field, dx)
            
            # Extract the central part (excluding the padding)
            gx = gx_padded[1:-1, 1:-1, 1:-1]
            gy = gy_padded[1:-1, 1:-1, 1:-1]
            gz = gz_padded[1:-1, 1:-1, 1:-1]
            
            return gx, gy, gz
        else:
            # Use NumPy with similar approach
            padded_field = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            gx_padded, gy_padded, gz_padded = np.gradient(padded_field, dx)
            
            gx = gx_padded[1:-1, 1:-1, 1:-1]
            gy = gy_padded[1:-1, 1:-1, 1:-1]
            gz = gz_padded[1:-1, 1:-1, 1:-1]
            
            return gx, gy, gz
    
    def calculate_periodic_curl(self, grid, field):
        """Calculate curl with periodic boundary conditions"""
        dx = grid.cell_size
        
        if self.use_gpu:
            # Pad each vector component to handle periodicity
            field_x = field[..., 0]
            field_y = field[..., 1]
            field_z = field[..., 2]
            
            padded_x = cp.pad(field_x, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            padded_y = cp.pad(field_y, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            padded_z = cp.pad(field_z, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            
            # Calculate derivatives with periodic boundaries
            _, dFx_dy_padded, dFx_dz_padded = cp.gradient(padded_x, dx)
            dFy_dx_padded, _, dFy_dz_padded = cp.gradient(padded_y, dx)
            dFz_dx_padded, dFz_dy_padded, _ = cp.gradient(padded_z, dx)
            
            # Extract central parts (excluding padding)
            dFx_dy = dFx_dy_padded[1:-1, 1:-1, 1:-1]
            dFx_dz = dFx_dz_padded[1:-1, 1:-1, 1:-1]
            dFy_dx = dFy_dx_padded[1:-1, 1:-1, 1:-1]
            dFy_dz = dFy_dz_padded[1:-1, 1:-1, 1:-1]
            dFz_dx = dFz_dx_padded[1:-1, 1:-1, 1:-1]
            dFz_dy = dFz_dy_padded[1:-1, 1:-1, 1:-1]
            
            # Compute curl components
            curl_x = dFy_dz - dFz_dy
            curl_y = dFz_dx - dFx_dz
            curl_z = dFx_dy - dFy_dx
            
            return curl_x, curl_y, curl_z
        else:
            # NumPy implementation
            field_x = field[..., 0]
            field_y = field[..., 1]
            field_z = field[..., 2]
            
            padded_x = np.pad(field_x, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            padded_y = np.pad(field_y, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            padded_z = np.pad(field_z, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            
            # Calculate derivatives with periodic boundaries
            _, dFx_dy_padded, dFx_dz_padded = np.gradient(padded_x, dx)
            dFy_dx_padded, _, dFy_dz_padded = np.gradient(padded_y, dx)
            dFz_dx_padded, dFz_dy_padded, _ = np.gradient(padded_z, dx)
            
            # Extract central parts
            dFx_dy = dFx_dy_padded[1:-1, 1:-1, 1:-1]
            dFx_dz = dFx_dz_padded[1:-1, 1:-1, 1:-1]
            dFy_dx = dFy_dx_padded[1:-1, 1:-1, 1:-1]
            dFy_dz = dFy_dz_padded[1:-1, 1:-1, 1:-1]
            dFz_dx = dFz_dx_padded[1:-1, 1:-1, 1:-1]
            dFz_dy = dFz_dy_padded[1:-1, 1:-1, 1:-1]
            
            # Compute curl components
            curl_x = dFy_dz - dFz_dy
            curl_y = dFz_dx - dFx_dz
            curl_z = dFx_dy - dFy_dx
            
            return curl_x, curl_y, curl_z
    
    def calculate_dwarf_force(self, p1, p2, grid_size):
        """Calculate force between two particles using dwarf force law with periodic boundaries"""
        direction = p2.position - p1.position
        
        # Apply periodic boundary conditions
        for i in range(3):
            if direction[i] > grid_size/2:
                direction[i] -= grid_size
            elif direction[i] < -grid_size/2:
                direction[i] += grid_size
                
        distance_squared = np.sum(direction ** 2)
        distance = np.sqrt(distance_squared)
        
        if distance < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        
        # Normalize direction
        direction = direction / distance
        
        # dwarf force with r^2.22
        k_dwarf = self.constants.k_dwarf
        force_magnitude = k_dwarf * p1.charge * p2.charge / (distance ** 2.22)
        
        return force_magnitude * direction
    
    def calculate_memory_field_force(self, particle, grid):
        """Calculate force on particle from memory field gradient with periodic boundaries"""
        # Get particle position in grid coordinates with wrapping
        pos = particle.position
        grid_pos = self.get_grid_position(pos, grid)
        
        # Calculate memory field gradient at particle position
        gradient = self.interpolate_gradient(grid_pos, grid.memory_field, grid.cell_size, grid.base_resolution)
        
        # Calculate force from gradient
        alpha = self.constants.alpha
        force = alpha * gradient
        
        return force
    
    def calculate_memory_torque(self, particle, grid):
        """Calculate torque on particle from memory field curl with periodic boundaries"""
        # Get particle position in grid coordinates with wrapping
        pos = particle.position
        grid_pos = self.get_grid_position(pos, grid)
        
        # Calculate memory field curl at particle position
        curl = self.interpolate_curl(grid_pos, grid.memory_field, grid.cell_size, grid.base_resolution)
        
        # Calculate torque as cross product of spin and curl
        beta = self.constants.beta
        torque = beta * np.cross(particle.spin, curl)
        
        return torque
    
    def get_grid_position(self, position, grid):
        """Convert world position to grid position with periodic wrapping"""
        grid_size = grid.base_resolution
        half_grid = grid_size / 2
        
        # Map position to grid coordinates with wrapping
        x = (position[0] / grid.cell_size + half_grid) % grid_size
        y = (position[1] / grid.cell_size + half_grid) % grid_size
        z = (position[2] / grid.cell_size + half_grid) % grid_size
        
        return np.array([x, y, z])
    
    def interpolate_gradient(self, grid_pos, field, dx, grid_size):
        """Interpolate gradient at a position in the grid with periodic boundaries"""
        # Calculate gradient of the entire field
        gradient = np.zeros(3)
        
        # Handle each vector component separately
        for i in range(field.shape[3]):
            # Calculate gradient with periodic boundaries
            padded_field = np.pad(field[..., i], ((1, 1), (1, 1), (1, 1)), mode='wrap')
            gx_padded, gy_padded, gz_padded = np.gradient(padded_field, dx)
            
            # Extract central parts
            gx = gx_padded[1:-1, 1:-1, 1:-1]
            gy = gy_padded[1:-1, 1:-1, 1:-1]
            gz = gz_padded[1:-1, 1:-1, 1:-1]
            
            # Interpolate gradient at the specific position
            x, y, z = int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2])
            
            # Ensure periodic boundaries for indices
            x = x % grid_size
            y = y % grid_size
            z = z % grid_size
            
            # Add component's gradient at this position
            gradient[i] = gx[x, y, z]
        
        return gradient
    
    def interpolate_curl(self, grid_pos, field, dx, grid_size):
        """Interpolate curl at a position in the grid with periodic boundaries"""
        # Calculate curl components for vector field with periodic boundaries
        field_x = field[..., 0]
        field_y = field[..., 1]
        field_z = field[..., 2]
        
        # Pad fields for periodic boundaries
        padded_x = np.pad(field_x, ((1, 1), (1, 1), (1, 1)), mode='wrap')
        padded_y = np.pad(field_y, ((1, 1), (1, 1), (1, 1)), mode='wrap')
        padded_z = np.pad(field_z, ((1, 1), (1, 1), (1, 1)), mode='wrap')
        
        # Calculate derivatives
        _, dFx_dy_padded, dFx_dz_padded = np.gradient(padded_x, dx)
        dFy_dx_padded, _, dFy_dz_padded = np.gradient(padded_y, dx)
        dFz_dx_padded, dFz_dy_padded, _ = np.gradient(padded_z, dx)
        
        # Extract central parts
        dFx_dy = dFx_dy_padded[1:-1, 1:-1, 1:-1]
        dFx_dz = dFx_dz_padded[1:-1, 1:-1, 1:-1]
        dFy_dx = dFy_dx_padded[1:-1, 1:-1, 1:-1]
        dFy_dz = dFy_dz_padded[1:-1, 1:-1, 1:-1]
        dFz_dx = dFz_dx_padded[1:-1, 1:-1, 1:-1]
        dFz_dy = dFz_dy_padded[1:-1, 1:-1, 1:-1]
        
        # Compute curl components
        curl_x = dFy_dz - dFz_dy
        curl_y = dFz_dx - dFx_dz
        curl_z = dFx_dy - dFy_dx
        
        # Interpolate at the specific position with periodic wrapping
        x = int(grid_pos[0]) % grid_size
        y = int(grid_pos[1]) % grid_size
        z = int(grid_pos[2]) % grid_size
        
        curl = np.zeros(3)
        curl[0] = curl_x[x, y, z]
        curl[1] = curl_y[x, y, z]
        curl[2] = curl_z[x, y, z]
        
        return curl
    
    def update_particles(self, particles, dt):
        """Update particle positions, velocities, and spins with periodic boundaries"""
        try:
            if self.grid is None:
                raise ValueError("Grid is required for periodic boundary handling")
                
            grid_size = self.grid.size  # Physical size of grid
            
            for particle in particles:
                # Update velocity using force (v = v + F/m * dt)
                particle.velocity += particle.force / particle.mass * dt
                
                # Update position using velocity (p = p + v * dt)
                particle.position += particle.velocity * dt
                
                # Apply periodic boundary conditions
                for i in range(3):
                    if particle.position[i] > grid_size/2:
                        particle.position[i] -= grid_size
                    elif particle.position[i] < -grid_size/2:
                        particle.position[i] += grid_size
                
                # Update spin using torque (ω = ω + τ/I * dt)
                particle.spin += particle.torque / particle.rotational_inertia * dt
                
                # Normalize spin to keep it as a unit vector
                spin_magnitude = np.linalg.norm(particle.spin)
                if spin_magnitude > 0:
                    particle.spin = particle.spin / spin_magnitude
                    
        except Exception as e:
            print(f"Error in particle update: {e}")
            import traceback
            traceback.print_exc()


class physics_constants:
    def __init__(self):
        # dwarf force constant (replaces Coulomb constant)
        self.k_dwarf = 8.9875517923e9  # Force constant for dwarf's r^2.22 law
        
        # Memory field interaction constants
        self.alpha = 1.0  # Memory field gradient force constant
        self.beta = 0.1   # Memory field curl torque constant

# For backward compatibility
dwarf_physics_gpu = dwarf_physics