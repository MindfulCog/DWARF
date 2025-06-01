import cupy as cp
import numpy as np

class DWARFMath:
    """Implementation of DWARF 2.22 gradient-based force calculations"""
    
    def __init__(self, saturation_limit=1.0, memory_decay=0.95, damping_factor=0.3):
        self.saturation_limit = saturation_limit
        self.memory_decay = memory_decay
        self.damping_factor = damping_factor
        self.memory_field = None
        self.memory_curl = None
        
    def initialize_memory_field(self, grid_size):
        """Initialize the memory field with zeros"""
        # Vector field (3 components) + scalar potential (1 component)
        self.memory_field = cp.zeros((grid_size, grid_size, grid_size, 4), dtype=cp.float32)
        # Store curl of the memory field
        self.memory_curl = cp.zeros((grid_size, grid_size, grid_size, 3), dtype=cp.float32)
        
    def calculate_force(self, particle1, particle2):
        """Calculate DWARF 2.22 force between two particles"""
        # These operations stay on CPU as they involve individual particles
        r_vec = particle2.position - particle1.position
        r = np.linalg.norm(r_vec)
        
        # Avoid division by zero with a small epsilon
        epsilon = 1e-10
        
        # Base direction and normalized distance
        if r > epsilon:
            r_hat = r_vec / r
        else:
            # Random direction if particles are too close
            r_hat = np.random.randn(3)
            r_hat = r_hat / np.linalg.norm(r_hat)
            r = epsilon
        
        # DWARF 2.22 inverse power law force - EXACTLY 2.22 power law
        # This is the core of DWARF theory
        force_magnitude = particle1.charge * particle2.charge / (r**2.22)
        
        # Soft repulsion to prevent overlaps (stronger at very short distances)
        # This is critical for stability and prevents particles from collapsing
        if r < 0.5:
            # Increased repulsion at close range - rises rapidly as r approaches 0
            soft_repulsion = 0.5 * (0.5 - r)**4
            # Direction is always repulsive, regardless of charges
            force_magnitude += soft_repulsion * np.sign(force_magnitude) if force_magnitude != 0 else 1.0
            
        # Apply hyperbolic tangent saturation for smooth force limiting
        # This prevents infinities and allows stable simulation
        force_magnitude = self.saturation_limit * np.tanh(force_magnitude / self.saturation_limit)
            
        # Base force vector
        force = force_magnitude * r_hat
        
        # Apply spin-based modification to force
        # Spin alignment factor (-1 to 1)
        spin_alignment = np.dot(particle1.spin, particle2.spin)
        
        # Spin effect depends on alignment (aligned or anti-aligned)
        spin_modifier = 1.0 + 0.2 * abs(spin_alignment) * np.sign(spin_alignment)
        
        # Final force with spin influence
        return force * spin_modifier
    
    def calculate_memory_field_force(self, particle, grid):
        """Calculate force on particle from memory field gradient - adaptive grid aware"""
        # Get grid cell containing particle at base resolution
        grid_pos = grid.world_to_grid(particle.position)
        i, j, k = np.clip(np.floor(grid_pos).astype(int), 0, grid.base_resolution - 2)
        
        # Check if this region has higher resolution data
        if hasattr(grid, 'refinement_regions'):
            for region_key, region_grid in grid.refinement_regions.items():
                base_i, base_j, base_k, level = region_key
                
                # If particle is in this region, use the higher resolution data
                if (base_i <= i < base_i+2 and 
                    base_j <= j < base_j+2 and 
                    base_k <= k < base_k+2):
                    
                    # Calculate position in refinement grid
                    ref_factor = 2**level
                    ref_cell_size = grid.cell_size / ref_factor
                    
                    # Get refined position
                    sub_x = (particle.position[0] + grid.size/2 - base_i*grid.cell_size) / ref_cell_size
                    sub_y = (particle.position[1] + grid.size/2 - base_j*grid.cell_size) / ref_cell_size
                    sub_z = (particle.position[2] + grid.size/2 - base_k*grid.cell_size) / ref_cell_size
                    
                    # Get integer position in refinement grid
                    si = int(min(max(0, sub_x), region_grid['velocity'].shape[0]-2))
                    sj = int(min(max(0, sub_y), region_grid['velocity'].shape[1]-2))
                    sk = int(min(max(0, sub_z), region_grid['velocity'].shape[2]-2))
                    
                    # Get memory field from refinement region
                    # For simplicity, we'll just use the velocity as the memory field
                    # In a full implementation, you'd have a memory field in the refinement regions
                    field_000 = cp.asnumpy(region_grid['velocity'][si, sj, sk])
                    field_100 = cp.asnumpy(region_grid['velocity'][si+1, sj, sk])
                    field_010 = cp.asnumpy(region_grid['velocity'][si, sj+1, sk])
                    field_001 = cp.asnumpy(region_grid['velocity'][si, sj, sk+1])
                    
                    # Calculate gradient at refinement resolution
                    dx_grad = (field_100 - field_000) / ref_cell_size
                    dy_grad = (field_010 - field_000) / ref_cell_size
                    dz_grad = (field_001 - field_000) / ref_cell_size
                    
                    # Rest of calculation same as before
                    gradient_tensor = np.stack([dx_grad, dy_grad, dz_grad], axis=0)
                    divergence = np.trace(gradient_tensor)
                    curl_contribution = gradient_tensor - gradient_tensor.T
                    field_vector = field_000
                    
                    # Force components
                    divergence_force = particle.charge * divergence * particle.velocity * 0.1
                    curl_force = np.dot(curl_contribution, particle.velocity) * 0.2
                    field_force = field_vector * particle.charge * 0.15
                    
                    # Total force
                    total_force = divergence_force + curl_force + field_force
                    
                    # Apply saturation
                    force_mag = np.linalg.norm(total_force)
                    if force_mag > self.saturation_limit:
                        total_force = total_force * (self.saturation_limit / force_mag)
                        
                    return total_force
        
        # If we're here, we're using the base grid
        # Original calculation from base grid
        field_000 = cp.asnumpy(self.memory_field[i, j, k][:3])
        field_100 = cp.asnumpy(self.memory_field[i+1, j, k][:3])
        field_010 = cp.asnumpy(self.memory_field[i, j+1, k][:3])
        field_001 = cp.asnumpy(self.memory_field[i, j, k+1][:3])
        
        # Calculate finite difference gradient (3x3 tensor)
        dx_grad = (field_100 - field_000) / grid.cell_size
        dy_grad = (field_010 - field_000) / grid.cell_size
        dz_grad = (field_001 - field_000) / grid.cell_size
        gradient_tensor = np.stack([dx_grad, dy_grad, dz_grad], axis=0)
        
        # Calculate div(v) - divergence of memory field
        divergence = np.trace(gradient_tensor)
        
        # Calculate curl contribution (antisymmetric part of gradient)
        curl_contribution = gradient_tensor - gradient_tensor.T
        
        # Get field at particle position
        field_vector = field_000
        field_strength = np.linalg.norm(field_vector)
        
        # Combined force from field gradient
        # Divergence term - particles are pushed/pulled by field compression/expansion
        divergence_force = particle.charge * divergence * particle.velocity * 0.1
        
        # Curl term - particles follow field lines
        curl_force = np.dot(curl_contribution, particle.velocity) * 0.2
        
        # Direct field interaction - particles are pulled by the memory field vector
        # This is crucial for DWARF orbit formation
        field_force = field_vector * particle.charge * 0.15
        
        # Total force from all memory field effects
        total_force = divergence_force + curl_force + field_force
        
        # Apply saturation to prevent excessive forces
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.saturation_limit:
            total_force = total_force * (self.saturation_limit / force_mag)
            
        return total_force
        
    def calculate_memory_torque(self, particle, grid):
        """Calculate torque on particle from memory field curl"""
        # Get grid cell containing particle
        grid_pos = grid.world_to_grid(particle.position)
        i, j, k = np.clip(np.floor(grid_pos).astype(int), 0, grid.base_resolution - 2)
        
        # First check if we have higher-resolution data in a refinement region
        if hasattr(grid, 'refinement_regions'):
            for region_key, region_grid in grid.refinement_regions.items():
                base_i, base_j, base_k, level = region_key
                
                # If particle is in this region, use higher resolution data
                if (base_i <= i < base_i+2 and 
                    base_j <= j < base_j+2 and 
                    base_k <= k < base_k+2):
                    
                    # Calculate position in refinement grid
                    ref_factor = 2**level
                    ref_cell_size = grid.cell_size / ref_factor
                    
                    # Get refined position
                    sub_x = (particle.position[0] + grid.size/2 - base_i*grid.cell_size) / ref_cell_size
                    sub_y = (particle.position[1] + grid.size/2 - base_j*grid.cell_size) / ref_cell_size
                    sub_z = (particle.position[2] + grid.size/2 - base_k*grid.cell_size) / ref_cell_size
                    
                    # Get integer position in refinement grid
                    si = int(min(max(0, sub_x), region_grid['vorticity'].shape[0]-1))
                    sj = int(min(max(0, sub_y), region_grid['vorticity'].shape[1]-1))
                    sk = int(min(max(0, sub_z), region_grid['vorticity'].shape[2]-1))
                    
                    # Get curl from refinement region
                    curl = cp.asnumpy(region_grid['vorticity'][si, sj, sk])
                    curl_magnitude = np.linalg.norm(curl)
                    
                    if curl_magnitude > 1e-10:
                        # Normalize curl vector
                        curl_direction = curl / curl_magnitude
                        
                        # Torque tries to align spin with curl
                        # Strength depends on curl magnitude and particle properties
                        torque_strength = 0.02 * curl_magnitude * abs(particle.charge)
                        torque = np.cross(particle.spin, curl_direction) * torque_strength
                        
                        return torque
                    else:
                        return np.zeros(3)
        
        # If we get here, use the base grid curl
        # Get curl at particle position - convert to CPU for particle interaction
        curl = cp.asnumpy(self.memory_curl[i, j, k])
        curl_magnitude = np.linalg.norm(curl)
        
        if curl_magnitude > 1e-10:
            # Normalize curl vector
            curl_direction = curl / curl_magnitude
            
            # Torque tries to align spin with curl
            # Strength depends on curl magnitude and particle properties
            torque_strength = 0.02 * curl_magnitude * abs(particle.charge)
            torque = np.cross(particle.spin, curl_direction) * torque_strength
            
            return torque
        else:
            return np.zeros(3)
        
    def update_memory_field(self, particles, grid, dt):
        """Update memory field based on particle movements"""
        # Decay existing field based on fluid state
        for i in range(grid.base_resolution):
            for j in range(grid.base_resolution):
                for k in range(grid.base_resolution):
                    # Get state-specific decay rate
                    # FIX: Convert CuPy array to Python integer before using as dictionary key
                    state_idx = int(cp.asnumpy(grid.state[i, j, k]))
                    decay_rate = grid.field_decay[state_idx]
                    self.memory_field[i, j, k] *= decay_rate ** dt
        
        # Add new contributions from particles
        for particle in particles:
            # Map particle position to grid
            grid_pos = grid.world_to_grid(particle.position)
            
            # Ensure position is within grid bounds
            if (grid_pos < 0).any() or (grid_pos >= grid.base_resolution).any():
                continue
                
            # Use trilinear interpolation to update surrounding cells
            i, j, k = np.floor(grid_pos).astype(int)
            
            if (i >= 0 and i < grid.base_resolution-1 and 
                j >= 0 and j < grid.base_resolution-1 and 
                k >= 0 and k < grid.base_resolution-1):
                
                # Fractional position within cell
                dx = grid_pos[0] - i
                dy = grid_pos[1] - j
                dz = grid_pos[2] - k
                
                # Weights for trilinear interpolation
                w000 = (1-dx)*(1-dy)*(1-dz)
                w001 = (1-dx)*(1-dy)*dz
                w010 = (1-dx)*dy*(1-dz)
                w011 = (1-dx)*dy*dz
                w100 = dx*(1-dy)*(1-dz)
                w101 = dx*(1-dy)*dz
                w110 = dx*dy*(1-dz)
                w111 = dx*dy*dz
                
                # Field contribution from particle: [vx, vy, vz, charge]
                field_contribution = np.array([
                    particle.velocity[0],
                    particle.velocity[1], 
                    particle.velocity[2],
                    particle.charge
                ])
                
                # Add spin contribution to memory field - cross product of spin and velocity
                # This is crucial for DWARF vortex formation
                spin_contribution = np.cross(particle.spin, particle.velocity)
                field_contribution[:3] += spin_contribution * 0.1
                
                # Scale by particle mass and time
                # Heavier particles leave stronger memory trails
                contribution_strength = dt * particle.mass / 500.0
                
                # Apply saturation to contribution
                contribution_mag = np.linalg.norm(field_contribution)
                if contribution_mag > self.saturation_limit:
                    field_contribution = field_contribution * (self.saturation_limit / contribution_mag)
                
                # Convert to GPU for field operations
                field_contribution_gpu = cp.asarray(field_contribution, dtype=cp.float32)
                contribution_strength_gpu = cp.float32(contribution_strength)
                
                # Apply weighted contribution to surrounding cells
                # This creates a smooth field without grid artifacts
                self.memory_field[i, j, k] += field_contribution_gpu * w000 * contribution_strength_gpu
                self.memory_field[i, j, k+1] += field_contribution_gpu * w001 * contribution_strength_gpu
                self.memory_field[i, j+1, k] += field_contribution_gpu * w010 * contribution_strength_gpu
                self.memory_field[i, j+1, k+1] += field_contribution_gpu * w011 * contribution_strength_gpu
                self.memory_field[i+1, j, k] += field_contribution_gpu * w100 * contribution_strength_gpu
                self.memory_field[i+1, j, k+1] += field_contribution_gpu * w101 * contribution_strength_gpu
                self.memory_field[i+1, j+1, k] += field_contribution_gpu * w110 * contribution_strength_gpu
                self.memory_field[i+1, j+1, k+1] += field_contribution_gpu * w111 * contribution_strength_gpu
        
        # Apply spatial smoothing (diffusion) to memory field
        self.apply_field_smoothing(grid)
            
        # Apply field damping based on state
        for i in range(grid.base_resolution):
            for j in range(grid.base_resolution):
                for k in range(grid.base_resolution):
                    # State-specific damping - get from CPU grid state
                    # FIX: Convert CuPy array to Python integer before using as dictionary key
                    state_idx = int(cp.asnumpy(grid.state[i, j, k]))
                    state_damping = self.damping_factor * grid.viscosity[state_idx]
                    self.memory_field[i, j, k] *= (1.0 - state_damping * dt)
        
        # Calculate curl of the memory field
        self.calculate_curl(grid)
        
    def apply_field_smoothing(self, grid):
        """Apply spatial smoothing to memory field to allow natural vortex formation"""
        # Create a smoothed copy of the field
        smoothed_field = cp.copy(self.memory_field)
        
        # Simple box-blur smoothing for internal cells - implemented on GPU
        for i in range(1, grid.base_resolution-1):
            for j in range(1, grid.base_resolution-1):
                for k in range(1, grid.base_resolution-1):
                    # Skip cells in vacuum state - preserve vortex structures
                    # FIX: Convert CuPy array to Python integer before comparison
                    state = int(cp.asnumpy(grid.state[i, j, k]))
                    if state == grid.VACUUM:
                        continue
                        
                    # Average with neighbors (box filter) - only vector part (first 3 components)
                    neighbor_sum = cp.zeros(4, dtype=cp.float32)
                    count = 0
                    
                    # 6-neighborhood (faces only)
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]
                    
                    for ni, nj, nk in neighbors:
                        neighbor_sum[:3] += self.memory_field[ni, nj, nk][:3]
                        count += 1
                    
                    # Weighted average - 80% original, 20% neighbors
                    smoothing_factor = 0.2
                    if count > 0:
                        smoothed_field[i, j, k][:3] = (
                            (1 - smoothing_factor) * self.memory_field[i, j, k][:3] +
                            smoothing_factor * (neighbor_sum[:3] / count)
                        )
                    
        # Update the field with smoothed version
        self.memory_field = smoothed_field
        
    def calculate_curl(self, grid):
        """Calculate curl of the memory field"""
        dx = dy = dz = grid.cell_size
        
        # Reset curl field
        self.memory_curl.fill(0)
        
        # Calculate curl components using central differences - implemented on GPU
        for i in range(1, grid.base_resolution-1):
            for j in range(1, grid.base_resolution-1):
                for k in range(1, grid.base_resolution-1):
                    # Get field vector components for surrounding cells
                    vx = self.memory_field[:, :, :, 0]  # x-component
                    vy = self.memory_field[:, :, :, 1]  # y-component
                    vz = self.memory_field[:, :, :, 2]  # z-component
                    
                    # Calculate derivatives using central differences
                    dvz_dy = (vz[i, j+1, k] - vz[i, j-1, k]) / (2*dy)
                    dvy_dz = (vy[i, j, k+1] - vy[i, j, k-1]) / (2*dz)
                    
                    dvx_dz = (vx[i, j, k+1] - vx[i, j, k-1]) / (2*dz)
                    dvz_dx = (vz[i+1, j, k] - vz[i-1, j, k]) / (2*dx)
                    
                    dvy_dx = (vy[i+1, j, k] - vy[i-1, j, k]) / (2*dx)
                    dvx_dy = (vx[i, j+1, k] - vx[i, j-1, k]) / (2*dy)
                    
                    # Curl components: ∇ × v
                    self.memory_curl[i, j, k, 0] = dvz_dy - dvy_dz  # x-component
                    self.memory_curl[i, j, k, 1] = dvx_dz - dvz_dx  # y-component
                    self.memory_curl[i, j, k, 2] = dvy_dx - dvx_dy  # z-component
                    
    def calculate_field_energy(self, grid):
        """Calculate total energy stored in memory field"""
        # Energy proportional to square of field magnitude - computed on GPU
        energy = 0.0
        
        # Calculate energy components on GPU
        field_energy = cp.sum(self.memory_field[:, :, :, :3]**2)
        charge_energy = cp.sum(self.memory_field[:, :, :, 3]**2)
        curl_energy = cp.sum(self.memory_curl[:, :, :]**2) * 0.5
        
        # Total energy - bring result back to CPU
        energy = cp.asnumpy(field_energy + charge_energy + curl_energy)
                    
        return energy * (grid.cell_size**3)  # Scale by cell volume