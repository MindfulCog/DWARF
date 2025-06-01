import numpy as np
import cupy as cp

class DWARFPhysics:
    def __init__(self, constants=None, use_gpu=True, grid=None, **kwargs):
        # Check if GPU is available
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
        # Initialize physics constants
        if constants is None:
            constants = PhysicsConstants()
        self.constants = constants
        
        # Store grid reference if provided
        self.grid = grid
        
        # Additional parameters that might be passed
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def calculate_particle_interactions(self, particles, grid=None):
        """GPU-accelerated particle interactions with correct DWARF physics"""
        # Use provided grid or fall back to stored grid
        if grid is None:
            grid = self.grid
            if grid is None:
                raise ValueError("No grid provided for particle interactions")
                
        num_particles = len(particles)
        
        if self.use_gpu:
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
            
            # Create interaction kernel with CORRECT DWARF FORCE LAW (r^2.22)
            interaction_kernel = cp.RawKernel(r'''
            extern "C" __global__ void dwarf_particle_forces(
                const float* positions, const float* masses, const float* charges,
                float* forces, int num_particles, float k_dwarf
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
                        
                        float dx = xj - xi;
                        float dy = yj - yi;
                        float dz = zj - zi;
                        
                        float r_squared = dx*dx + dy*dy + dz*dz;
                        float r = sqrt(r_squared);
                        
                        if (r > 1e-10f) {  // Avoid self-interaction singularity
                            // CORRECT DWARF FORCE LAW: r^2.22 exponent
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
            ''', 'dwarf_particle_forces')
            
            # Configure grid and block dimensions
            threads_per_block = 256
            blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block
            
            # Execute kernel with DWARF force constant
            k_dwarf = self.constants.k_dwarf  # DWARF force constant
            interaction_kernel((blocks_per_grid,), (threads_per_block,), 
                              (positions_gpu, masses_gpu, charges_gpu, forces_gpu, 
                               num_particles, k_dwarf))
            
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
            # CPU fallback with correct DWARF physics
            for i, p1 in enumerate(particles):
                # Reset forces and torques
                p1.force = np.zeros(3)
                p1.torque = np.zeros(3)
                
                # Particle-particle DWARF force (2.22 exponent)
                for j in range(i+1, num_particles):
                    p2 = particles[j]
                    force = self.calculate_dwarf_force(p1, p2)
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
        
        if self.use_gpu:
            # Transfer memory field to GPU if not already there
            if isinstance(grid.memory_field, np.ndarray):
                memory_field_gpu = cp.asarray(grid.memory_field)
            else:
                memory_field_gpu = grid.memory_field
                
            # Calculate gradient of memory field
            dx = grid.cell_size
            grad_x, grad_y, grad_z = cp.gradient(memory_field_gpu, dx)
            
            # Calculate curl of memory field
            curl_x, curl_y, curl_z = self.calculate_curl(grid, memory_field_gpu)
            
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
            
            # Create kernel for memory field interactions
            memory_interaction_kernel = cp.RawKernel(r'''
            extern "C" __global__ void memory_field_interactions(
                const float* positions, const float* spins,
                const float* grad_x, const float* grad_y, const float* grad_z,
                const float* curl_x, const float* curl_y, const float* curl_z,
                float* forces, float* torques,
                int num_particles, int grid_size, float alpha, float beta, float dx
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < num_particles) {
                    // Get particle position and spin
                    float px = positions[i*3];
                    float py = positions[i*3+1];
                    float pz = positions[i*3+2];
                    
                    float sx = spins[i*3];
                    float sy = spins[i*3+1];
                    float sz = spins[i*3+2];
                    
                    // Convert position to grid indices
                    int x = (int)((px / dx) + (grid_size / 2));
                    int y = (int)((py / dx) + (grid_size / 2));
                    int z = (int)((pz / dx) + (grid_size / 2));
                    
                    // Ensure within bounds
                    if (x >= 0 && x < grid_size && y >= 0 && y < grid_size && z >= 0 && z < grid_size) {
                        // Calculate linear index
                        int idx = x * grid_size * grid_size + y * grid_size + z;
                        
                        // Get gradient vector at particle position
                        float gx = grad_x[idx];
                        float gy = grad_y[idx];
                        float gz = grad_z[idx];
                        
                        // Get curl vector at particle position
                        float cx = curl_x[idx];
                        float cy = curl_y[idx];
                        float cz = curl_z[idx];
                        
                        // Calculate memory field force (gradient-based)
                        forces[i*3] = alpha * gx;
                        forces[i*3+1] = alpha * gy;
                        forces[i*3+2] = alpha * gz;
                        
                        // Calculate memory field torque (curl-based)
                        // torque = beta * (spin × curl)
                        torques[i*3] = beta * (sy * cz - sz * cy);
                        torques[i*3+1] = beta * (sz * cx - sx * cz);
                        torques[i*3+2] = beta * (sx * cy - sy * cx);
                    }
                }
            }
            ''', 'memory_field_interactions')
            
            # Configure grid and block dimensions
            threads_per_block = 256
            blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block
            
            # Prepare memory field data on GPU
            grid_size = grid.base_resolution
            
            # Execute kernel
            alpha = self.constants.alpha  # Memory field force constant
            beta = self.constants.beta    # Memory field torque constant
            dx = grid.cell_size
            
            memory_interaction_kernel((blocks_per_grid,), (threads_per_block,), 
                                    (positions_gpu, spins_gpu,
                                     grad_x, grad_y, grad_z,
                                     curl_x, curl_y, curl_z,
                                     memory_forces, memory_torques,
                                     num_particles, grid_size, alpha, beta, dx))
            
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
    
    def calculate_curl(self, grid, field):
        """Calculate the curl of a field on the grid"""
        dx = grid.cell_size
        
        if self.use_gpu:
            # Use CuPy's gradient function
            grad_y, grad_x, _ = cp.gradient(field[..., 0], dx)  # dF_x/dy, dF_x/dx
            _, grad_y, _ = cp.gradient(field[..., 1], dx)       # dF_y/dx, dF_y/dy
            _, _, grad_z = cp.gradient(field[..., 2], dx)       # dF_z/dx, dF_z/dy
            
            curl_x = grad_z - grad_y      # dF_z/dy - dF_y/dz
            curl_y = grad_x - grad_z      # dF_x/dz - dF_z/dx
            curl_z = grad_y - grad_x      # dF_y/dx - dF_x/dy
            
            return curl_x, curl_y, curl_z
        else:
            # NumPy implementation
            grad_y, grad_x, _ = np.gradient(field[..., 0], dx)
            _, grad_y, _ = np.gradient(field[..., 1], dx)
            _, _, grad_z = np.gradient(field[..., 2], dx)
            
            curl_x = grad_z - grad_y
            curl_y = grad_x - grad_z
            curl_z = grad_y - grad_x
            
            return curl_x, curl_y, curl_z
    
    def calculate_dwarf_force(self, p1, p2):
        """Calculate force between two particles using DWARF force law (CPU fallback)"""
        direction = p2.position - p1.position
        distance_squared = np.sum(direction ** 2)
        distance = np.sqrt(distance_squared)
        
        if distance < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        
        # Normalize direction
        direction = direction / distance
        
        # DWARF force with r^2.22
        k_dwarf = self.constants.k_dwarf
        force_magnitude = k_dwarf * p1.charge * p2.charge / (distance ** 2.22)
        
        return force_magnitude * direction
    
    def calculate_memory_field_force(self, particle, grid):
        """Calculate force on particle from memory field gradient (CPU fallback)"""
        # Get particle position in grid coordinates
        pos = particle.position
        grid_pos = self.get_grid_position(pos, grid)
        
        # Calculate memory field gradient at particle position
        gradient = self.interpolate_gradient(grid_pos, grid.memory_field, grid.cell_size)
        
        # Calculate force from gradient
        alpha = self.constants.alpha
        force = alpha * gradient
        
        return force
    
    def calculate_memory_torque(self, particle, grid):
        """Calculate torque on particle from memory field curl (CPU fallback)"""
        # Get particle position in grid coordinates
        pos = particle.position
        grid_pos = self.get_grid_position(pos, grid)
        
        # Calculate memory field curl at particle position
        curl = self.interpolate_curl(grid_pos, grid.memory_field, grid.cell_size)
        
        # Calculate torque as cross product of spin and curl
        beta = self.constants.beta
        torque = beta * np.cross(particle.spin, curl)
        
        return torque
    
    def get_grid_position(self, position, grid):
        """Convert world position to grid position"""
        grid_center = np.array([grid.base_resolution / 2] * 3)
        return grid_center + position / grid.cell_size
    
    def interpolate_gradient(self, grid_pos, field, dx):
        """Interpolate gradient at a position in the grid"""
        # Calculate gradient of the entire field
        gradient = np.gradient(field, dx)
        
        # Interpolate gradient at the specific position
        # This is a simplified interpolation - in production you'd want trilinear interpolation
        x, y, z = np.floor(grid_pos).astype(int)
        return np.array([gradient[0][x, y, z], gradient[1][x, y, z], gradient[2][x, y, z]])
    
    def interpolate_curl(self, grid_pos, field, dx):
        """Interpolate curl at a position in the grid"""
        # Calculate curl components
        grad_y_x, grad_x_x, _ = np.gradient(field[..., 0], dx)
        _, grad_y_y, _ = np.gradient(field[..., 1], dx)
        _, _, grad_z_z = np.gradient(field[..., 2], dx)
        
        # Calculate curl vector
        curl_x = grad_z_z - grad_y_y
        curl_y = grad_x_x - grad_z_z
        curl_z = grad_y_y - grad_x_x
        
        # Interpolate at the specific position
        x, y, z = np.floor(grid_pos).astype(int)
        return np.array([curl_x[x, y, z], curl_y[x, y, z], curl_z[x, y, z]])
    
    def update_particles(self, particles, dt):
        """Update particle positions, velocities, and spins"""
        if self.use_gpu:
            # Transfer particle data to GPU
            positions = np.array([p.position for p in particles])
            velocities = np.array([p.velocity for p in particles])
            forces = np.array([p.force for p in particles])
            masses = np.array([p.mass for p in particles])
            spins = np.array([p.spin for p in particles])
            torques = np.array([p.torque for p in particles])
            moments = np.array([p.moment_of_inertia for p in particles])
            
            positions_gpu = cp.asarray(positions)
            velocities_gpu = cp.asarray(velocities)
            forces_gpu = cp.asarray(forces)
            masses_gpu = cp.asarray(masses)
            spins_gpu = cp.asarray(spins)
            torques_gpu = cp.asarray(torques)
            moments_gpu = cp.asarray(moments)
            
            # Create particle update kernel
            particle_update_kernel = cp.RawKernel(r'''
            extern "C" __global__ void update_particles(
                float* positions, float* velocities, float* spins,
                const float* forces, const float* torques,
                const float* masses, const float* moments, 
                int num_particles, float dt
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < num_particles) {
                    // Update velocity using force (v = v + F/m * dt)
                    float inv_mass = 1.0f / masses[i];
                    velocities[i*3] += forces[i*3] * inv_mass * dt;
                    velocities[i*3+1] += forces[i*3+1] * inv_mass * dt;
                    velocities[i*3+2] += forces[i*3+2] * inv_mass * dt;
                    
                    // Update position using velocity (p = p + v * dt)
                    positions[i*3] += velocities[i*3] * dt;
                    positions[i*3+1] += velocities[i*3+1] * dt;
                    positions[i*3+2] += velocities[i*3+2] * dt;
                    
                    // Update spin using torque (ω = ω + τ/I * dt)
                    float inv_moment = 1.0f / moments[i];
                    spins[i*3] += torques[i*3] * inv_moment * dt;
                    spins[i*3+1] += torques[i*3+1] * inv_moment * dt;
                    spins[i*3+2] += torques[i*3+2] * inv_moment * dt;
                }
            }
            ''', 'update_particles')
            
            # Configure grid and block dimensions
            num_particles = len(particles)
            threads_per_block = 256
            blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block
            
            # Execute kernel
            particle_update_kernel((blocks_per_grid,), (threads_per_block,), 
                                   (positions_gpu, velocities_gpu, spins_gpu, 
                                    forces_gpu, torques_gpu, masses_gpu, moments_gpu, 
                                    num_particles, dt))
            
            # Transfer results back to CPU
            new_positions = cp.asnumpy(positions_gpu)
            new_velocities = cp.asnumpy(velocities_gpu)
            new_spins = cp.asnumpy(spins_gpu)
            
            # Update particle objects
            for i, particle in enumerate(particles):
                particle.position = new_positions[i]
                particle.velocity = new_velocities[i]
                particle.spin = new_spins[i]
        else:
            # CPU fallback
            for particle in particles:
                # Update velocity using force (v = v + F/m * dt)
                particle.velocity += particle.force / particle.mass * dt
                
                # Update position using velocity (p = p + v * dt)
                particle.position += particle.velocity * dt
                
                # Update spin using torque (ω = ω + τ/I * dt)
                particle.spin += particle.torque / particle.moment_of_inertia * dt

class PhysicsConstants:
    def __init__(self):
        # DWARF force constant (replaces Coulomb constant)
        self.k_dwarf = 8.9875517923e9  # Force constant for DWARF's r^2.22 law
        
        # Memory field interaction constants
        self.alpha = 1.0  # Memory field gradient force constant
        self.beta = 0.1   # Memory field curl torque constant

# For backward compatibility
DwarfPhysicsGPU = DWARFPhysics