import numpy as np
import cupy as cp
import time

class adaptive_fluid_grid:
    """Adaptive fluid grid with memory field for dwarf physics simulation"""
    
    def __init__(self, base_resolution=64, size=10.0, max_depth=2, use_gpu=True):
        """Initialize the adaptive grid
        
        Parameters:
        -----------
        base_resolution : int
            Base resolution of the grid
        size : float
            Physical size of the grid in appropriate units
        max_depth : int
            Maximum refinement depth
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.base_resolution = base_resolution
        self.size = size  # Physical size of the grid
        self.cell_size = size / base_resolution  # Size of each cell
        self.max_depth = max_depth  # Maximum refinement level
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
        # Refinement regions
        self.refinement_regions = []  # List of higher-resolution regions
        
        # Initialize fields
        self._init_fields()
        
        # Performance tracking
        self.update_time = 0.0
        
    def _init_fields(self):
        """Initialize all grid fields with proper sizing"""
        # Create base resolution grid
        shape = (self.base_resolution, self.base_resolution, self.base_resolution)
        
        # Initialize fluid state (0=vacuum, 1=uncompressed, 2=compressed)
        self.state = self.xp.ones(shape, dtype=self.xp.int32)
        
        # Initialize velocity field (vector field)
        self.velocity_field = self.xp.zeros(shape + (3,), dtype=self.xp.float32)
        
        # Initialize memory field (vector field that can be affected by particles)
        self.memory_field = self.xp.zeros(shape + (3,), dtype=self.xp.float32)
        
        # Initialize pressure field (scalar field)
        self.pressure_field = self.xp.zeros(shape, dtype=self.xp.float32)
        
        # Initialize vorticity magnitude (scalar field)
        self.vorticity_magnitude = self.xp.zeros(shape, dtype=self.xp.float32)
        
        # Initialize energy density
        self.energy_density = self.xp.zeros(shape, dtype=self.xp.float32)
        
    def initialize(self):
        """Initialize the grid with some interesting structure"""
        # Create a simple initial condition
        # This is just an example - can be customized based on what you want
        shape = self.velocity_field.shape[:3]
        center = self.xp.array(shape) // 2
        
        # Create a central vortex
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    x = i - center[0]
                    y = j - center[1]
                    z = k - center[2]
                    
                    r = self.xp.sqrt(x*x + y*y + z*z)
                    if r > 0:
                        # Create a radial component
                        self.memory_field[i, j, k, 0] = 0.1 * x / r * self.xp.exp(-0.02 * r)
                        self.memory_field[i, j, k, 1] = 0.1 * y / r * self.xp.exp(-0.02 * r)
                        self.memory_field[i, j, k, 2] = 0.1 * z / r * self.xp.exp(-0.02 * r)
                        
                        # Create a circular component (vortex)
                        theta = self.xp.arctan2(y, x)
                        phi = self.xp.arctan2(z, self.xp.sqrt(x*x + y*y))
                        
                        self.velocity_field[i, j, k, 0] = -0.05 * y / (r + 1) * self.xp.exp(-0.01 * r)
                        self.velocity_field[i, j, k, 1] = 0.05 * x / (r + 1) * self.xp.exp(-0.01 * r)
                        self.velocity_field[i, j, k, 2] = 0.02 * self.xp.sin(2*theta) * self.xp.exp(-0.01 * r)
                        
        # Calculate initial pressure from velocity field
        self._update_pressure_field()
        
        # Calculate initial vorticity
        self._update_vorticity_field()
        
        # Calculate initial energy density
        self._update_energy_density()
    
    def initialize_vectorized(self):
        """Initialize the grid with vectorized operations for better performance"""
        print("Initializing grid with vectorized operations...")
        
        # Get grid shape
        shape = self.velocity_field.shape[:3]
        center = self.xp.array(shape) // 2
        
        # Create coordinate arrays (vectorized approach)
        print("  Creating coordinate mesh...")
        x = self.xp.arange(shape[0])[:, None, None] - center[0]
        y = self.xp.arange(shape[1])[None, :, None] - center[1]
        z = self.xp.arange(shape[2])[None, None, :] - center[2]
        
        # Broadcast to full shape
        x = self.xp.broadcast_to(x, shape)
        y = self.xp.broadcast_to(y, shape)
        z = self.xp.broadcast_to(z, shape)
        
        # Calculate distance from center (vectorized)
        print("  Computing radial distance...")
        r = self.xp.sqrt(x*x + y*y + z*z)
        r_nonzero = self.xp.maximum(r, 0.1)  # Avoid division by zero
        
        print("  Computing memory field...")
        # Memory field (vectorized)
        self.memory_field[..., 0] = 0.1 * x / r_nonzero * self.xp.exp(-0.02 * r)
        self.memory_field[..., 1] = 0.1 * y / r_nonzero * self.xp.exp(-0.02 * r)
        self.memory_field[..., 2] = 0.1 * z / r_nonzero * self.xp.exp(-0.02 * r)
        
        print("  Computing velocity field...")
        # Calculate theta (vectorized)
        theta = self.xp.arctan2(y, x)
        
        # Velocity field (vectorized)
        self.velocity_field[..., 0] = -0.05 * y / (r_nonzero) * self.xp.exp(-0.01 * r)
        self.velocity_field[..., 1] = 0.05 * x / (r_nonzero) * self.xp.exp(-0.01 * r)
        self.velocity_field[..., 2] = 0.02 * self.xp.sin(2*theta) * self.xp.exp(-0.01 * r)
        
        print("  Computing pressure field...")
        # Use vectorized operations for the pressure field
        self._update_pressure_field_vectorized()
        
        print("  Computing vorticity field...")
        # Use vectorized operations for the vorticity field
        self._update_vorticity_field_vectorized()
        
        print("  Computing energy density...")
        # Use vectorized operations for energy density
        self._update_energy_density_vectorized()
        
        print("Grid initialization complete.")
        
    def update(self, particles, dt):
        """Update the grid state based on particle interactions and fluid dynamics"""
        start_time = time.time()
        
        # Update grid with particle influences
        self._apply_particle_effects(particles, dt)
        
        # Update fluid dynamics using NavierStokes with periodic boundary
        self._update_fluid_dynamics(dt)
        
        # Update refined regions based on particles and activity
        active_regions = self._update_refinement_regions(particles)
        
        # Update derived quantities
        self._update_pressure_field_vectorized()
        self._update_vorticity_field_vectorized()
        self._update_energy_density_vectorized()
        
        # Record update time for performance tracking
        self.update_time = time.time() - start_time
        
        return active_regions
        
    def _apply_particle_effects(self, particles, dt):
        """Apply the effects of particles to the grid"""
        if not particles:
            return
            
        # Get the resolution
        res = self.base_resolution
        
        # For each particle, apply its effect to the grid
        for particle in particles:
            # Get particle position in grid coordinates
            grid_pos = self._world_to_grid(particle.position)
            
            # Check if the particle is within the grid
            if not (0 <= grid_pos[0] < res and 0 <= grid_pos[1] < res and 0 <= grid_pos[2] < res):
                # Apply periodic boundary to grid position
                grid_pos = grid_pos % res
            
            # Get integer indices and interpolation weights
            i, j, k = int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2])
            
            # Calculate weights for trilinear interpolation
            fx = grid_pos[0] - i
            fy = grid_pos[1] - j
            fz = grid_pos[2] - k
            
            # Define the radius of influence
            influence_radius = 3  # cells
            
            # Apply particle's influence to the grid within its radius
            for di in range(-influence_radius, influence_radius+1):
                for dj in range(-influence_radius, influence_radius+1):
                    for dk in range(-influence_radius, influence_radius+1):
                        # Calculate distance from particle to this cell in grid units
                        dist_squared = di*di + dj*dj + dk*dk
                        
                        # Skip cells outside the influence radius
                        if dist_squared > influence_radius*influence_radius:
                            continue
                            
                        # Calculate influence weight based on distance
                        weight = self.xp.exp(-dist_squared / (2.0 * influence_radius/2.5))
                        
                        # Calculate grid indices with periodic boundary
                        gi = (i + di) % res
                        gj = (j + dj) % res
                        gk = (k + dk) % res
                        
                        # Particle velocity contribution to fluid velocity
                        # The effect depends on the particle type
                        vel_factor = 0.1  # Base velocity influence factor
                        if particle.particle_type == "proton":
                            vel_factor *= 1.5
                            # Protons compress the fluid
                            self.state[gi, gj, gk] = 2  # Compressed
                        elif particle.particle_type == "electron":
                            vel_factor *= 0.8
                            # Electrons leave the state uncompressed
                            self.state[gi, gj, gk] = 1  # Uncompressed
                        elif particle.particle_type == "neutron":
                            vel_factor *= 1.2
                            # Neutrons don't affect state directly
                        
                        # FIX: Convert numpy array to CuPy array if using GPU
                        particle_vel = particle.velocity
                        if self.use_gpu:
                            # Convert numpy array to CuPy array
                            particle_vel = cp.asarray(particle_vel)
                        
                        # Add particle's velocity to the fluid scaled by weight and type
                        self.velocity_field[gi, gj, gk] += particle_vel * weight * vel_factor * dt
                        
                        # Memory field update based on particle spin
                        # Particles with spin create a kind of "memory" in the field
                        memory_factor = 0.05  # Base memory influence factor
                        if particle.particle_type == "proton":
                            memory_factor *= 2.0
                        elif particle.particle_type == "electron":
                            memory_factor *= 1.5
                        elif particle.particle_type == "neutron":
                            memory_factor *= 1.0
                            
                        # FIX: Convert numpy array to CuPy array if using GPU
                        particle_spin = particle.spin
                        if self.use_gpu:
                            # Convert numpy array to CuPy array
                            particle_spin = cp.asarray(particle_spin)
                            
                        # Update memory field based on spin and velocity
                        spin_contribution = particle_spin * weight * memory_factor
                        vel_contribution = particle_vel * weight * memory_factor * 0.2
                        
                        # Combine both contributions
                        self.memory_field[gi, gj, gk] += (spin_contribution + vel_contribution) * dt
                        
                        # Optional - add energy from particle to grid
                        energy_factor = 0.02
                        ke = particle.get_kinetic_energy()
                        if self.use_gpu:
                            ke = self.xp.array(ke)
                        self.energy_density[gi, gj, gk] += (
                            ke * weight * energy_factor * dt
                        )
    
    def _update_fluid_dynamics(self, dt):
        """Update fluid dynamics using NavierStokes with periodic boundary"""
        # Parameters
        viscosity = 0.01
        
        # Create temporary copy for advection
        if self.use_gpu:
            velocity_prev = cp.copy(self.velocity_field)
        else:
            velocity_prev = np.copy(self.velocity_field)
            
        # Get grid resolution
        res = self.base_resolution
        
        # Update velocity field using simplified Navier-Stokes
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    # Skip vacuum cells
                    if self.state[i, j, k] == 0:
                        continue
                    
                    # Get velocity at current cell
                    vel = self.velocity_field[i, j, k]
                    
                    # Apply advection - follow the velocity field backwards
                    # to find where the fluid came from
                    pos_i = float(i) - vel[0] * dt
                    pos_j = float(j) - vel[1] * dt
                    pos_k = float(k) - vel[2] * dt
                    
                    # Apply periodic boundary conditions to tracing position
                    pos_i = pos_i % res
                    pos_j = pos_j % res
                    pos_k = pos_k % res
                    
                    # Get integer indices for interpolation
                    i0 = int(pos_i)
                    j0 = int(pos_j)
                    k0 = int(pos_k)
                    
                    # FIX: Handle edge case where pos_i/j/k is exactly at res
                    # This can happen due to floating point precision and cause index errors
                    i0 = i0 % res  # Ensure within bounds
                    j0 = j0 % res
                    k0 = k0 % res
                    
                    # Get next indices with periodic boundary
                    i1 = (i0 + 1) % res
                    j1 = (j0 + 1) % res
                    k1 = (k0 + 1) % res
                    
                    # Calculate interpolation weights
                    s1 = pos_i - i0
                    s0 = 1 - s1
                    t1 = pos_j - j0
                    t0 = 1 - t1
                    u1 = pos_k - k0
                    u0 = 1 - u1
                    
                    # Perform trilinear interpolation to get velocity
                    v000 = velocity_prev[i0, j0, k0]
                    v001 = velocity_prev[i0, j0, k1]
                    v010 = velocity_prev[i0, j1, k0]
                    v011 = velocity_prev[i0, j1, k1]
                    v100 = velocity_prev[i1, j0, k0]
                    v101 = velocity_prev[i1, j0, k1]
                    v110 = velocity_prev[i1, j1, k0]
                    v111 = velocity_prev[i1, j1, k1]
                    
                    # Interpolate along x
                    v00 = s0 * v000 + s1 * v100
                    v01 = s0 * v001 + s1 * v101
                    v10 = s0 * v010 + s1 * v110
                    v11 = s0 * v011 + s1 * v111
                    
                    # Interpolate along y
                    v0 = t0 * v00 + t1 * v10
                    v1 = t0 * v01 + t1 * v11
                    
                    # Final interpolation along z
                    v = u0 * v0 + u1 * v1
                    
                    # Update velocity with advected value and apply damping/viscosity
                    self.velocity_field[i, j, k] = v * (1.0 - viscosity * dt)
        
        # Apply pressure forces based on pressure gradients
        self._apply_pressure_forces(dt)
                    
    def _apply_pressure_forces(self, dt):
        """Apply forces based on pressure gradients"""
        res = self.base_resolution
        
        # Create temporary pressure for gradient calculation
        pressure = self.pressure_field
        
        # Calculate pressure gradient
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    # Calculate pressure gradient with periodic boundary
                    i_next = (i + 1) % res
                    i_prev = (i - 1) % res
                    j_next = (j + 1) % res
                    j_prev = (j - 1) % res
                    k_next = (k + 1) % res
                    k_prev = (k - 1) % res
                    
                    # Compute gradient with central differences
                    grad_p = self.xp.zeros(3, dtype=self.xp.float32)
                    grad_p[0] = (pressure[i_next, j, k] - pressure[i_prev, j, k]) / 2.0
                    grad_p[1] = (pressure[i, j_next, k] - pressure[i, j_prev, k]) / 2.0
                    grad_p[2] = (pressure[i, j, k_next] - pressure[i, j, k_prev]) / 2.0
                    
                    # Apply force based on gradient (flow from high to low pressure)
                    self.velocity_field[i, j, k] -= grad_p * dt * 0.1
                    
    def _update_refinement_regions(self, particles):
        """Update adaptive grid refinement regions based on particle positions and activity"""
        # In a basic implementation, we create refinement regions around particles
        # where there is interesting activity
        
        # Clear existing refinement regions
        old_regions = len(self.refinement_regions)
        self.refinement_regions = []
        
        # Don't refine if no particles or maximum depth is 0
        if not particles or self.max_depth <= 0:
            return 0
            
        # Identify regions with significant activity
        # Here we use a simple heuristic based on particle positions
        for particle in particles:
            # Convert particle position to grid coordinates
            grid_pos = self._world_to_grid(particle.position)
            
            # Create a refinement region around the particle
            region_size = 8  # Size in grid cells
            
            # Create a region that obeys periodic boundary
            region_min = np.array([
                int(grid_pos[0] - region_size/2) % self.base_resolution,
                int(grid_pos[1] - region_size/2) % self.base_resolution,
                int(grid_pos[2] - region_size/2) % self.base_resolution
            ])
            
            region_max = np.array([
                int(grid_pos[0] + region_size/2) % self.base_resolution,
                int(grid_pos[1] + region_size/2) % self.base_resolution,
                int(grid_pos[2] + region_size/2) % self.base_resolution
            ])
            
            # Check if this overlaps with an existing region
            overlap = False
            for existing_region in self.refinement_regions:
                if self._regions_overlap(region_min, region_max, existing_region[0], existing_region[1]):
                    overlap = True
                    break
                    
            if not overlap:
                self.refinement_regions.append((region_min, region_max, 1))  # Level 1 refinement
                
        # Return the number of active refinement regions
        return len(self.refinement_regions)
                
    def _regions_overlap(self, min1, max1, min2, max2):
        """Check if two regions overlap, considering periodic boundary"""
        # This is a simplified check that doesn't fully account for wrapping
        # In a full implementation, this would need to handle cases where a region
        # wraps around the grid boundaries
        
        # Simple check for non-overlapping regions
        for i in range(3):
            if max1[i] < min2[i] and max1[i] + self.base_resolution > min2[i]:
                return False
            if max2[i] < min1[i] and max2[i] + self.base_resolution > min1[i]:
                return False
                
        return True
                
    def _update_pressure_field(self):
        """Update pressure field based on velocity divergence"""
        res = self.base_resolution
        
        # Calculate divergence of velocity field
        div = self.xp.zeros((res, res, res), dtype=self.xp.float32)
        
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    # Calculate indices with periodic boundary
                    i_next = (i + 1) % res
                    i_prev = (i - 1) % res
                    j_next = (j + 1) % res
                    j_prev = (j - 1) % res
                    k_next = (k + 1) % res
                    k_prev = (k - 1) % res
                    
                    # Calculate divergence with central differences
                    div_x = (self.velocity_field[i_next, j, k, 0] - self.velocity_field[i_prev, j, k, 0]) / 2.0
                    div_y = (self.velocity_field[i, j_next, k, 1] - self.velocity_field[i, j_prev, k, 1]) / 2.0
                    div_z = (self.velocity_field[i, j, k_next, 2] - self.velocity_field[i, j, k_prev, 2]) / 2.0
                    
                    div[i, j, k] = div_x + div_y + div_z
                    
        # Pressure is related to the negative divergence
        # High divergence (sources) reduces pressure, low divergence (sinks) increases pressure
        self.pressure_field -= div * 0.1
        
        # Diffuse pressure for stability
        self._diffuse_field(self.pressure_field, 0.05)
    
    def _update_pressure_field_vectorized(self):
        """Update pressure field using vectorized operations"""
        res = self.base_resolution
        
        # Create shifted arrays for derivatives
        # Use roll for periodic boundary conditions
        vx_next = self.xp.roll(self.velocity_field[..., 0], -1, axis=0)
        vx_prev = self.xp.roll(self.velocity_field[..., 0], 1, axis=0)
        vy_next = self.xp.roll(self.velocity_field[..., 1], -1, axis=1)
        vy_prev = self.xp.roll(self.velocity_field[..., 1], 1, axis=1)
        vz_next = self.xp.roll(self.velocity_field[..., 2], -1, axis=2)
        vz_prev = self.xp.roll(self.velocity_field[..., 2], 1, axis=2)
        
        # Calculate divergence with central differences (vectorized)
        div_x = (vx_next - vx_prev) / 2.0
        div_y = (vy_next - vy_prev) / 2.0
        div_z = (vz_next - vz_prev) / 2.0
        
        # Calculate divergence
        div = div_x + div_y + div_z
        
        # Update pressure (negative divergence relationship)
        self.pressure_field -= div * 0.1
        
        # Apply simple diffusion for smoothing
        self.pressure_field = self.pressure_field * 0.98
    
    def _update_vorticity_field(self):
        """Update vorticity field based on curl of velocity"""
        res = self.base_resolution
        
        # Calculate curl of velocity field (vorticity)
        vorticity = self.xp.zeros((res, res, res, 3), dtype=self.xp.float32)
        
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    # Calculate indices with periodic boundary
                    i_next = (i + 1) % res
                    i_prev = (i - 1) % res
                    j_next = (j + 1) % res
                    j_prev = (j - 1) % res
                    k_next = (k + 1) % res
                    k_prev = (k - 1) % res
                    
                    # Calculate derivatives with central differences
                    dvx_dy = (self.velocity_field[i, j_next, k, 0] - self.velocity_field[i, j_prev, k, 0]) / 2.0
                    dvx_dz = (self.velocity_field[i, j, k_next, 0] - self.velocity_field[i, j, k_prev, 0]) / 2.0
                    
                    dvy_dx = (self.velocity_field[i_next, j, k, 1] - self.velocity_field[i_prev, j, k, 1]) / 2.0
                    dvy_dz = (self.velocity_field[i, j, k_next, 1] - self.velocity_field[i, j, k_prev, 1]) / 2.0
                    
                    dvz_dx = (self.velocity_field[i_next, j, k, 2] - self.velocity_field[i_prev, j, k, 2]) / 2.0
                    dvz_dy = (self.velocity_field[i, j_next, k, 2] - self.velocity_field[i, j_prev, k, 2]) / 2.0
                    
                    # Compute curl components
                    vorticity[i, j, k, 0] = dvz_dy - dvy_dz  # x component
                    vorticity[i, j, k, 1] = dvx_dz - dvz_dx  # y component
                    vorticity[i, j, k, 2] = dvy_dx - dvx_dy  # z component
                    
        # Calculate the magnitude of vorticity
        self.vorticity_magnitude = self.xp.sqrt(
            vorticity[..., 0]**2 + vorticity[..., 1]**2 + vorticity[..., 2]**2
        )
    
    def _update_vorticity_field_vectorized(self):
        """Update vorticity field using vectorized operations"""
        # X derivatives
        vx = self.velocity_field[..., 0]
        vy = self.velocity_field[..., 1]
        vz = self.velocity_field[..., 2]
        
        # Calculate derivatives using roll for periodic boundary
        dvx_dy = (self.xp.roll(vx, -1, axis=1) - self.xp.roll(vx, 1, axis=1)) / 2.0
        dvx_dz = (self.xp.roll(vx, -1, axis=2) - self.xp.roll(vx, 1, axis=2)) / 2.0
        
        dvy_dx = (self.xp.roll(vy, -1, axis=0) - self.xp.roll(vy, 1, axis=0)) / 2.0
        dvy_dz = (self.xp.roll(vy, -1, axis=2) - self.xp.roll(vy, 1, axis=2)) / 2.0
        
        dvz_dx = (self.xp.roll(vz, -1, axis=0) - self.xp.roll(vz, 1, axis=0)) / 2.0
        dvz_dy = (self.xp.roll(vz, -1, axis=1) - self.xp.roll(vz, 1, axis=1)) / 2.0
        
        # Compute curl components
        vorticity_x = dvz_dy - dvy_dz
        vorticity_y = dvx_dz - dvz_dx
        vorticity_z = dvy_dx - dvx_dy
        
        # Calculate magnitude
        self.vorticity_magnitude = self.xp.sqrt(vorticity_x**2 + vorticity_y**2 + vorticity_z**2)
        
    def _update_energy_density(self):
        """Update energy density field"""
        # Calculate kinetic energy from velocity squared
        velocity_squared = self.xp.sum(self.velocity_field**2, axis=3)
        
        # Kinetic energy density is 0.5 * density * velocity^2
        # Here we assume constant density of 1.0
        kinetic_energy = 0.5 * velocity_squared
        
        # Add energy contribution from pressure
        pressure_energy = self.pressure_field * 0.1
        
        # Update energy density (with some damping to prevent excessive buildup)
        self.energy_density = self.energy_density * 0.98 + (kinetic_energy + pressure_energy) * 0.02
    
    def _update_energy_density_vectorized(self):
        """Update energy density using vectorized operations"""
        # Calculate velocity squared (vectorized)
        velocity_squared = self.xp.sum(self.velocity_field**2, axis=3)
        
        # Kinetic energy (with constant density of 1.0)
        kinetic_energy = 0.5 * velocity_squared
        
        # Add pressure contribution
        pressure_energy = self.pressure_field * 0.1
        
        # Update energy with damping factor
        self.energy_density = self.energy_density * 0.98 + (kinetic_energy + pressure_energy) * 0.02
        
    def _diffuse_field(self, field, diffusion_rate):
        """Apply diffusion to a scalar field with periodic boundary"""
        res = self.base_resolution
        
        # Create a temporary copy of the field
        if self.use_gpu:
            temp = cp.copy(field)
        else:
            temp = np.copy(field)
            
        # Apply diffusion using Laplacian
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    # Calculate indices with periodic boundary
                    i_next = (i + 1) % res
                    i_prev = (i - 1) % res
                    j_next = (j + 1) % res
                    j_prev = (j - 1) % res
                    k_next = (k + 1) % res
                    k_prev = (k - 1) % res
                    
                    # Calculate Laplacian using central differences
                    laplacian = (
                        temp[i_next, j, k] + temp[i_prev, j, k] +
                        temp[i, j_next, k] + temp[i, j_prev, k] +
                        temp[i, j, k_next] + temp[i, j, k_prev] - 6 * temp[i, j, k]
                    )
                    
                    # Apply diffusion
                    field[i, j, k] += diffusion_rate * laplacian
    
    def _world_to_grid(self, position):
        """Convert world position to grid coordinates"""
        # Map position from [-grid_size/2, grid_size/2] to [0, base_resolution]
        grid_x = (position[0] + self.size/2) / self.cell_size
        grid_y = (position[1] + self.size/2) / self.cell_size
        grid_z = (position[2] + self.size/2) / self.cell_size
        
        return np.array([grid_x, grid_y, grid_z])
    
    def get_state_counts(self):
        """Get the counts of different fluid states"""
        if self.use_gpu:
            state_array = cp.asnumpy(self.state)
        else:
            state_array = self.state
            
        # Count each state
        vacuum = np.sum(state_array == 0)
        uncompressed = np.sum(state_array == 1)
        compressed = np.sum(state_array == 2)
        
        return {
            'vacuum': int(vacuum),
            'uncompressed': int(uncompressed),
            'compressed': int(compressed)
        }
    
    def get_total_energy(self):
        """Calculate total energy in the grid"""
        if self.use_gpu:
            energy_sum = float(cp.sum(self.energy_density))
        else:
            energy_sum = float(np.sum(self.energy_density))
            
        return energy_sum