import numpy as np
import cupy as cp
from collections import defaultdict

class AdaptiveFluidGrid:
    """Adaptive multi-resolution fluid grid for the DWARF physics simulator"""
    
    # Fluid states constants
    UNCOMPRESSED = 0
    COMPRESSED = 1
    VACUUM = 2
    VORTEX = 3
    
    def __init__(self, base_resolution=128, size=10.0, max_depth=2):
        """Initialize adaptive fluid grid
        
        Args:
            base_resolution: Base grid resolution (coarsest level)
            size: Physical grid size
            max_depth: Maximum refinement depth
        """
        self.base_resolution = base_resolution
        self.size = size
        self.max_depth = max_depth
        self.cell_size = size / base_resolution
        
        # Compatibility property for old code
        self._resolution = base_resolution
        
        # Initialize grid fields
        self.velocity_field = None
        self.pressure_field = None
        self.density_field = None
        self.temperature_field = None
        self.energy_density = None
        self.vorticity_field = None
        self.vorticity_magnitude = None
        self.dilatation_field = None
        self.activity_map = None
        self.state = None
        
        # Physical parameters for different fluid states
        self.viscosity = {
            self.UNCOMPRESSED: 0.01,
            self.COMPRESSED: 0.02,
            self.VACUUM: 0.001,
            self.VORTEX: 0.005
        }
        
        self.diffusion = {
            self.UNCOMPRESSED: 0.01,
            self.COMPRESSED: 0.005,
            self.VACUUM: 0.1,
            self.VORTEX: 0.02
        }
        
        self.field_decay = {
            self.UNCOMPRESSED: 0.98,
            self.COMPRESSED: 0.99,
            self.VACUUM: 0.80,
            self.VORTEX: 0.95
        }
        
        # Refinement regions - store as dictionary with keys (i,j,k,level)
        # where i,j,k is base grid cell and level is refinement level
        self.refinement_regions = {}
        
    @property
    def resolution(self):
        """Compatibility property for code expecting the old grid interface"""
        return self.base_resolution
        
    def initialize(self):
        """Initialize grid fields"""
        # Initialize base grid fields on GPU
        self.velocity_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution, 3), dtype=cp.float32)
        self.pressure_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.density_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.temperature_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.energy_density = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.vorticity_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution, 3), dtype=cp.float32)
        self.vorticity_magnitude = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.dilatation_field = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        self.activity_map = cp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.float32)
        
        # Initialize fluid state
        self.state = cp.ones((self.base_resolution, self.base_resolution, self.base_resolution), dtype=cp.int32) * self.UNCOMPRESSED
        
        # Initialize refinement regions (empty at start)
        self.refinement_regions = {}
        
    def world_to_grid(self, position):
        """Convert world position to grid coordinates"""
        # Shift to grid space (centered at grid center)
        grid_pos = position + self.size / 2.0
        # Convert to cell coordinates
        grid_pos = grid_pos / self.cell_size
        return grid_pos
    
    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world position"""
        # Convert from cell coordinates
        world_pos = grid_pos * self.cell_size
        # Shift from grid space to world space
        world_pos = world_pos - self.size / 2.0
        return world_pos
    
    def update_activity_map(self, particles):
        """Update activity map based on particle positions"""
        # Reset activity map
        self.activity_map.fill(0)
        
        # Activity radius (in grid cells)
        activity_radius = 3
        
        # Add activity for each particle
        for particle in particles:
            # Convert particle position to grid coordinates
            grid_pos = self.world_to_grid(particle.position)
            
            # Get integer position and ensure within grid bounds
            i, j, k = np.clip(np.floor(grid_pos).astype(int), 
                             0, self.base_resolution-1)
            
            # Add activity around particle
            radius = activity_radius
            
            # Set activity in a sphere around particle
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    for dk in range(-radius, radius+1):
                        # Check if within activity radius and grid bounds
                        dist_sq = di**2 + dj**2 + dk**2
                        if dist_sq > radius**2:
                            continue
                            
                        ni, nj, nk = i + di, j + dj, k + dk
                        
                        if (0 <= ni < self.base_resolution and
                            0 <= nj < self.base_resolution and
                            0 <= nk < self.base_resolution):
                            
                            # Activity falls off with square of distance
                            activity_value = 1.0 / (1.0 + 0.2*dist_sq) 
                            
                            # Add activity from this particle
                            self.activity_map[ni, nj, nk] += activity_value
        
        # Also add activity in regions of high vorticity
        if hasattr(self, 'vorticity_magnitude') and self.vorticity_magnitude is not None:
            # Threshold for adding activity based on vorticity
            vorticity_threshold = 0.5
            
            # Find regions of high vorticity
            high_vorticity = self.vorticity_magnitude > vorticity_threshold
            
            # Increase activity in these regions
            self.activity_map = self.activity_map + high_vorticity.astype(cp.float32) * 0.5
            
        # Normalize activity map to range [0, 1]
        max_activity = cp.max(self.activity_map)
        if max_activity > 0:
            self.activity_map = self.activity_map / max_activity
    
    def manage_refinement_regions(self):
        """Manage creation and removal of refinement regions"""
        # Threshold for adding refinement
        activity_threshold = 0.3
        
        # Find potential regions for refinement
        high_activity_cells = []
        
        # Sample at coarser stride for efficiency
        stride = 2
        for i in range(0, self.base_resolution, stride):
            for j in range(0, self.base_resolution, stride):
                for k in range(0, self.base_resolution, stride):
                    if self.activity_map[i, j, k] > activity_threshold:
                        high_activity_cells.append((i, j, k))
        
        # Limit number of refined regions (for performance)
        max_regions = 32
        if len(high_activity_cells) > max_regions:
            # Sort by activity and keep highest
            high_activity_cells.sort(
                key=lambda cell: float(cp.asnumpy(self.activity_map[cell[0], cell[1], cell[2]])),
                reverse=True
            )
            high_activity_cells = high_activity_cells[:max_regions]
        
        # Track regions to remove
        regions_to_remove = set(self.refinement_regions.keys())
        
        # Add or update refinement regions
        for i, j, k in high_activity_cells:
            # Make sure i,j,k are even (for consistent refinement boundaries)
            i, j, k = i - i % 2, j - j % 2, k - k % 2
            
            # Determine refinement level (for now, just use level 1)
            # In a more advanced implementation, you could use higher levels
            # based on activity magnitude and available resources
            level = 1
            
            # Create region key tuple
            region_key = (i, j, k, level)
            
            # Remove from regions to remove list (we're keeping it)
            if region_key in regions_to_remove:
                regions_to_remove.remove(region_key)
            
            # If region already exists, update data from coarse grid
            if region_key in self.refinement_regions:
                self.update_refinement_from_base(region_key)
            else:
                # Create new refinement region
                self.create_refinement_region(i, j, k, level)
        
        # Remove inactive regions
        for region_key in regions_to_remove:
            del self.refinement_regions[region_key]
    
    def create_refinement_region(self, base_i, base_j, base_k, level):
        """Create a refinement region covering the specified base grid cells"""
        # Calculate refinement factor based on level
        ref_factor = 2**level
        
        # Size of refinement region (covering 2x2x2 base cells)
        ref_size = 2 * ref_factor
        
        # Create refinement region data fields
        region_grid = {
            'velocity': cp.zeros((ref_size, ref_size, ref_size, 3), dtype=cp.float32),
            'pressure': cp.zeros((ref_size, ref_size, ref_size), dtype=cp.float32),
            'density': cp.zeros((ref_size, ref_size, ref_size), dtype=cp.float32),
            'temperature': cp.zeros((ref_size, ref_size, ref_size), dtype=cp.float32),
            'energy': cp.zeros((ref_size, ref_size, ref_size), dtype=cp.float32),
            'vorticity': cp.zeros((ref_size, ref_size, ref_size, 3), dtype=cp.float32),
            'state': cp.ones((ref_size, ref_size, ref_size), dtype=cp.int32) * self.UNCOMPRESSED
        }
        
        # Add to refinement regions
        region_key = (base_i, base_j, base_k, level)
        self.refinement_regions[region_key] = region_grid
        
        # Initialize from base grid
        self.update_refinement_from_base(region_key)
        
        # Return the newly created region
        return region_grid
    
    def update_refinement_from_base(self, region_key):
        """Update refinement region data from base grid"""
        base_i, base_j, base_k, level = region_key
        ref_region = self.refinement_regions[region_key]
        
        # Calculate refinement factor
        ref_factor = 2**level
        
        # Make sure we don't exceed grid bounds
        if (base_i + 1 >= self.base_resolution or
            base_j + 1 >= self.base_resolution or
            base_k + 1 >= self.base_resolution):
            return
            
        # Get data from base grid
        base_vel = self.velocity_field[base_i:base_i+2, base_j:base_j+2, base_k:base_k+2]
        base_pressure = self.pressure_field[base_i:base_i+2, base_j:base_j+2, base_k:base_k+2]
        base_density = self.density_field[base_i:base_i+2, base_j:base_j+2, base_k:base_k+2]
        base_state = self.state[base_i:base_i+2, base_j:base_j+2, base_k:base_k+2]
        
        # Interpolate to refinement region (simple trilinear interpolation)
        ref_size = 2 * ref_factor
        
        for ri in range(ref_size):
            for rj in range(ref_size):
                for rk in range(ref_size):
                    # Calculate fractional position in base grid cell
                    fx = ri / ref_factor
                    fy = rj / ref_factor
                    fz = rk / ref_factor
                    
                    # Get base grid cell indices
                    bi = int(fx)
                    bj = int(fy)
                    bk = int(fz)
                    
                    # Calculate interpolation weights
                    wx = fx - bi
                    wy = fy - bj
                    wz = fz - bk
                    
                    # Clamp to prevent out of bounds access
                    bi = min(bi, 1)
                    bj = min(bj, 1)
                    bk = min(bk, 1)
                    
                    # Interpolate velocity
                    v000 = base_vel[bi, bj, bk]
                    v100 = base_vel[min(bi+1, 1), bj, bk] if bi<1 else v000
                    v010 = base_vel[bi, min(bj+1, 1), bk] if bj<1 else v000
                    v110 = base_vel[min(bi+1, 1), min(bj+1, 1), bk] if bi<1 and bj<1 else v000
                    v001 = base_vel[bi, bj, min(bk+1, 1)] if bk<1 else v000
                    v101 = base_vel[min(bi+1, 1), bj, min(bk+1, 1)] if bi<1 and bk<1 else v000
                    v011 = base_vel[bi, min(bj+1, 1), min(bk+1, 1)] if bj<1 and bk<1 else v000
                    v111 = base_vel[min(bi+1, 1), min(bj+1, 1), min(bk+1, 1)] if bi<1 and bj<1 and bk<1 else v000
                    
                    # Trilinear interpolation
                    ref_region['velocity'][ri, rj, rk] = (
                        (1-wx)*(1-wy)*(1-wz) * v000 +
                        wx*(1-wy)*(1-wz) * v100 +
                        (1-wx)*wy*(1-wz) * v010 +
                        wx*wy*(1-wz) * v110 +
                        (1-wx)*(1-wy)*wz * v001 +
                        wx*(1-wy)*wz * v101 +
                        (1-wx)*wy*wz * v011 +
                        wx*wy*wz * v111
                    )
                    
                    # For scalar fields like pressure, use same interpolation
                    # But simplified here to nearest neighbor for state
                    ref_region['state'][ri, rj, rk] = base_state[min(bi, 1), min(bj, 1), min(bk, 1)]
        
        # Return the updated region
        return ref_region
    
    def update_base_from_refinement(self, region_key):
        """Update base grid data from refinement region"""
        base_i, base_j, base_k, level = region_key
        ref_region = self.refinement_regions[region_key]
        
        # Calculate refinement factor
        ref_factor = 2**level
        
        # Make sure we don't exceed grid bounds
        if (base_i + 1 >= self.base_resolution or
            base_j + 1 >= self.base_resolution or
            base_k + 1 >= self.base_resolution):
            return
            
        # For each base grid cell, average the corresponding refined cells
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # Get refined cell range for this base cell
                    ri_start = bi * ref_factor
                    rj_start = bj * ref_factor
                    rk_start = bk * ref_factor
                    
                    # Get velocity in refinement region
                    ref_vel = ref_region['velocity'][ri_start:ri_start+ref_factor,
                                                   rj_start:rj_start+ref_factor,
                                                   rk_start:rk_start+ref_factor]
                    
                    # Average and assign to base grid
                    avg_vel = cp.mean(ref_vel, axis=(0, 1, 2))
                    self.velocity_field[base_i+bi, base_j+bj, base_k+bk] = avg_vel
                    
                    # Same for other fields like pressure, density, etc.
                    ref_pressure = ref_region['pressure'][ri_start:ri_start+ref_factor,
                                                        rj_start:rj_start+ref_factor,
                                                        rk_start:rk_start+ref_factor]
                    avg_pressure = cp.mean(ref_pressure)
                    self.pressure_field[base_i+bi, base_j+bj, base_k+bk] = avg_pressure
    
    def sync_hierarchical_data(self):
        """Synchronize data between resolution levels"""
        # First, update refinement regions from base grid
        for region_key in self.refinement_regions:
            self.update_refinement_from_base(region_key)
            
        # Then update base grid from refinement regions
        for region_key in self.refinement_regions:
            self.update_base_from_refinement(region_key)
            
    def update_fields_from_particles(self, particles, dt):
        """Update grid fields based on particles"""
        # Reset density field
        self.density_field.fill(0.0)
        
        # Update base grid from particles
        for particle in particles:
            # Convert particle position to grid coordinates
            grid_pos = self.world_to_grid(particle.position)
            
            # Get integer position and ensure within grid bounds
            i, j, k = np.floor(grid_pos).astype(int)
            
            if (0 <= i < self.base_resolution-1 and
                0 <= j < self.base_resolution-1 and
                0 <= k < self.base_resolution-1):
                
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
                
                # Update density field with particle mass
                self.density_field[i, j, k] += particle.mass * w000
                self.density_field[i, j, k+1] += particle.mass * w001
                self.density_field[i, j+1, k] += particle.mass * w010
                self.density_field[i, j+1, k+1] += particle.mass * w011
                self.density_field[i+1, j, k] += particle.mass * w100
                self.density_field[i+1, j, k+1] += particle.mass * w101
                self.density_field[i+1, j+1, k] += particle.mass * w110
                self.density_field[i+1, j+1, k+1] += particle.mass * w111
                
                # Update velocity field with particle momentum (mass * velocity)
                particle_velocity = cp.asarray(particle.velocity)
                momentum = particle_velocity * particle.mass
                
                self.velocity_field[i, j, k] += momentum * w000
                self.velocity_field[i, j, k+1] += momentum * w001
                self.velocity_field[i, j+1, k] += momentum * w010
                self.velocity_field[i, j+1, k+1] += momentum * w011
                self.velocity_field[i+1, j, k] += momentum * w100
                self.velocity_field[i+1, j, k+1] += momentum * w101
                self.velocity_field[i+1, j+1, k] += momentum * w110
                self.velocity_field[i+1, j+1, k+1] += momentum * w111
                
                # Also check if particle is in any refinement region
                # and update the refinement region fields
                for region_key in self.refinement_regions:
                    base_i, base_j, base_k, level = region_key
                    
                    # Check if particle is in this region
                    if (base_i <= i < base_i+2 and
                        base_j <= j < base_j+2 and
                        base_k <= k < base_k+2):
                        
                        # Calculate position in refinement grid
                        ref_factor = 2**level
                        ref_cell_size = self.cell_size / ref_factor
                        
                        # Get refined position
                        sub_x = (particle.position[0] + self.size/2 - base_i*self.cell_size) / ref_cell_size
                        sub_y = (particle.position[1] + self.size/2 - base_j*self.cell_size) / ref_cell_size
                        sub_z = (particle.position[2] + self.size/2 - base_k*self.cell_size) / ref_cell_size
                        
                        # Get integer position in refinement grid
                        ri = int(sub_x)
                        rj = int(sub_y)
                        rk = int(sub_z)
                        
                        # Ensure within bounds
                        ref_size = 2 * ref_factor
                        if (0 <= ri < ref_size-1 and
                            0 <= rj < ref_size-1 and
                            0 <= rk < ref_size-1):
                            
                            # Fractional position within refined cell
                            rdx = sub_x - ri
                            rdy = sub_y - rj
                            rdz = sub_z - rk
                            
                            # Weights for trilinear interpolation
                            rw000 = (1-rdx)*(1-rdy)*(1-rdz)
                            rw001 = (1-rdx)*(1-rdy)*rdz
                            rw010 = (1-rdx)*rdy*(1-rdz)
                            rw011 = (1-rdx)*rdy*rdz
                            rw100 = rdx*(1-rdy)*(1-rdz)
                            rw101 = rdx*(1-rdy)*rdz
                            rw110 = rdx*rdy*(1-rdz)
                            rw111 = rdx*rdy*rdz
                            
                            # Update refinement region fields
                            ref_region = self.refinement_regions[region_key]
                            
                            # Update density
                            ref_region['density'][ri, rj, rk] += particle.mass * rw000
                            ref_region['density'][ri, rj, rk+1] += particle.mass * rw001
                            ref_region['density'][ri, rj+1, rk] += particle.mass * rw010
                            ref_region['density'][ri, rj+1, rk+1] += particle.mass * rw011
                            ref_region['density'][ri+1, rj, rk] += particle.mass * rw100
                            ref_region['density'][ri+1, rj, rk+1] += particle.mass * rw101
                            ref_region['density'][ri+1, rj+1, rk] += particle.mass * rw110
                            ref_region['density'][ri+1, rj+1, rk+1] += particle.mass * rw111
                            
                            # Update velocity
                            ref_region['velocity'][ri, rj, rk] += momentum * rw000
                            ref_region['velocity'][ri, rj, rk+1] += momentum * rw001
                            ref_region['velocity'][ri, rj+1, rk] += momentum * rw010
                            ref_region['velocity'][ri, rj+1, rk+1] += momentum * rw011
                            ref_region['velocity'][ri+1, rj, rk] += momentum * rw100
                            ref_region['velocity'][ri+1, rj, rk+1] += momentum * rw101
                            ref_region['velocity'][ri+1, rj+1, rk] += momentum * rw110
                            ref_region['velocity'][ri+1, rj+1, rk+1] += momentum * rw111
                            
        # Normalize velocity field by density
        # Avoid division by zero with small epsilon
        epsilon = 1e-10
        
        # For base grid
        for i in range(self.base_resolution):
            for j in range(self.base_resolution):
                for k in range(self.base_resolution):
                    if self.density_field[i, j, k] > epsilon:
                        self.velocity_field[i, j, k] /= self.density_field[i, j, k]
        
        # For refinement regions
        for region_key in self.refinement_regions:
            ref_region = self.refinement_regions[region_key]
            ref_size = ref_region['density'].shape[0]
            
            for i in range(ref_size):
                for j in range(ref_size):
                    for k in range(ref_size):
                        if ref_region['density'][i, j, k] > epsilon:
                            ref_region['velocity'][i, j, k] /= ref_region['density'][i, j, k]

    def calculate_vorticity(self):
        """Calculate vorticity fields (curl of velocity)"""
        # Grid cell size
        dx = dy = dz = self.cell_size
        
        # Calculate vorticity on base grid
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Get velocity components
                    vx = self.velocity_field[:, :, :, 0]
                    vy = self.velocity_field[:, :, :, 1]
                    vz = self.velocity_field[:, :, :, 2]
                    
                    # Calculate partial derivatives using central differences
                    dvz_dy = (vz[i, j+1, k] - vz[i, j-1, k]) / (2*dy)
                    dvy_dz = (vy[i, j, k+1] - vy[i, j, k-1]) / (2*dz)
                    
                    dvx_dz = (vx[i, j, k+1] - vx[i, j, k-1]) / (2*dz)
                    dvz_dx = (vz[i+1, j, k] - vz[i-1, j, k]) / (2*dx)
                    
                    dvy_dx = (vy[i+1, j, k] - vy[i-1, j, k]) / (2*dx)
                    dvx_dy = (vx[i, j+1, k] - vx[i, j-1, k]) / (2*dy)
                    
                    # Vorticity = curl(v) = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
                    self.vorticity_field[i, j, k, 0] = dvz_dy - dvy_dz
                    self.vorticity_field[i, j, k, 1] = dvx_dz - dvz_dx
                    self.vorticity_field[i, j, k, 2] = dvy_dx - dvx_dy
                    
                    # Calculate vorticity magnitude
                    self.vorticity_magnitude[i, j, k] = cp.sqrt(
                        self.vorticity_field[i, j, k, 0]**2 +
                        self.vorticity_field[i, j, k, 1]**2 +
                        self.vorticity_field[i, j, k, 2]**2
                    )
                    
                    # Calculate dilatation (divergence of velocity)
                    dvx_dx = (vx[i+1, j, k] - vx[i-1, j, k]) / (2*dx)
                    dvy_dy = (vy[i, j+1, k] - vy[i, j-1, k]) / (2*dy)
                    dvz_dz = (vz[i, j, k+1] - vz[i, j, k-1]) / (2*dz)
                    
                    self.dilatation_field[i, j, k] = dvx_dx + dvy_dy + dvz_dz
        
        # Calculate vorticity in refinement regions
        for region_key in self.refinement_regions:
            ref_region = self.refinement_regions[region_key]
            ref_size = ref_region['velocity'].shape[0]
            ref_factor = 2**region_key[3]
            ref_dx = self.cell_size / ref_factor
            
            # Only calculate for interior cells
            for i in range(1, ref_size-1):
                for j in range(1, ref_size-1):
                    for k in range(1, ref_size-1):
                        # Get velocity components
                        vx = ref_region['velocity'][:, :, :, 0]
                        vy = ref_region['velocity'][:, :, :, 1]
                        vz = ref_region['velocity'][:, :, :, 2]
                        
                        # Calculate partial derivatives (same method as base grid)
                        dvz_dy = (vz[i, j+1, k] - vz[i, j-1, k]) / (2*ref_dx)
                        dvy_dz = (vy[i, j, k+1] - vy[i, j, k-1]) / (2*ref_dx)
                        
                        dvx_dz = (vx[i, j, k+1] - vx[i, j, k-1]) / (2*ref_dx)
                        dvz_dx = (vz[i+1, j, k] - vz[i-1, j, k]) / (2*ref_dx)
                        
                        dvy_dx = (vy[i+1, j, k] - vy[i-1, j, k]) / (2*ref_dx)
                        dvx_dy = (vx[i, j+1, k] - vx[i, j-1, k]) / (2*ref_dx)
                        
                        # Calculate vorticity
                        ref_region['vorticity'][i, j, k, 0] = dvz_dy - dvy_dz
                        ref_region['vorticity'][i, j, k, 1] = dvx_dz - dvz_dx
                        ref_region['vorticity'][i, j, k, 2] = dvy_dx - dvx_dy

    def calculate_pressure(self):
        """Calculate pressure field using a simple equation of state"""
        # Simple equation of state: P = rho * T
        # where T is temperature (defaulting to 1.0 if not simulated)
        for i in range(self.base_resolution):
            for j in range(self.base_resolution):
                for k in range(self.base_resolution):
                    self.pressure_field[i, j, k] = self.density_field[i, j, k]
                    
                    # Add pressure contribution from vorticity
                    if i > 0 and j > 0 and k > 0 and i < self.base_resolution-1 and j < self.base_resolution-1 and k < self.base_resolution-1:
                        vorticity_pressure = 0.1 * self.vorticity_magnitude[i, j, k]**2
                        self.pressure_field[i, j, k] += vorticity_pressure
                        
        # Calculate pressure in refinement regions
        for region_key in self.refinement_regions:
            ref_region = self.refinement_regions[region_key]
            ref_size = ref_region['density'].shape[0]
            
            for i in range(ref_size):
                for j in range(ref_size):
                    for k in range(ref_size):
                        ref_region['pressure'][i, j, k] = ref_region['density'][i, j, k]
                        
                        # Add vorticity contribution if vorticity has been calculated
                        if i > 0 and j > 0 and k > 0 and i < ref_size-1 and j < ref_size-1 and k < ref_size-1:
                            if 'vorticity' in ref_region:
                                vort_magnitude = cp.sqrt(
                                    ref_region['vorticity'][i, j, k, 0]**2 +
                                    ref_region['vorticity'][i, j, k, 1]**2 +
                                    ref_region['vorticity'][i, j, k, 2]**2
                                )
                                vorticity_pressure = 0.1 * vort_magnitude**2
                                ref_region['pressure'][i, j, k] += vorticity_pressure

    def update_fluid_states(self):
        """Update fluid state based on density, pressure, and vorticity"""
        # Constants for state transitions
        density_threshold = 0.1
        pressure_threshold = 0.5
        vorticity_threshold = 0.5
        
        # Update base grid states
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    density = self.density_field[i, j, k]
                    pressure = self.pressure_field[i, j, k]
                    
                    if density < density_threshold:
                        # Low density -> vacuum state
                        self.state[i, j, k] = self.VACUUM
                    elif pressure > pressure_threshold:
                        # High pressure -> compressed state
                        self.state[i, j, k] = self.COMPRESSED
                    elif i > 0 and j > 0 and k > 0 and self.vorticity_magnitude[i, j, k] > vorticity_threshold:
                        # High vorticity -> vortex state
                        self.state[i, j, k] = self.VORTEX
                    else:
                        # Default -> uncompressed state
                        self.state[i, j, k] = self.UNCOMPRESSED
        
        # Update refinement region states
        for region_key in self.refinement_regions:
            ref_region = self.refinement_regions[region_key]
            ref_size = ref_region['density'].shape[0]
            
            for i in range(1, ref_size-1):
                for j in range(1, ref_size-1):
                    for k in range(1, ref_size-1):
                        density = ref_region['density'][i, j, k]
                        pressure = ref_region['pressure'][i, j, k]
                        
                        if density < density_threshold:
                            ref_region['state'][i, j, k] = self.VACUUM
                        elif pressure > pressure_threshold:
                            ref_region['state'][i, j, k] = self.COMPRESSED
                        elif 'vorticity' in ref_region:
                            vort_mag = cp.sqrt(cp.sum(ref_region['vorticity'][i, j, k]**2))
                            if vort_mag > vorticity_threshold:
                                ref_region['state'][i, j, k] = self.VORTEX
                            else:
                                ref_region['state'][i, j, k] = self.UNCOMPRESSED
                        else:
                            ref_region['state'][i, j, k] = self.UNCOMPRESSED

    def apply_state_specific_dynamics(self, dt):
        """Apply specific dynamics based on fluid state"""
        # Apply on base grid - simple version for now
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Get state at this cell
                    state = self.state[i, j, k]
                    
                    # Get appropriate parameters for this state
                    viscosity = self.viscosity[int(cp.asnumpy(state))]
                    
                    # Apply viscosity diffusion to velocity
                    if viscosity > 0:
                        # Simple diffusion using neighboring cells
                        neighbors_velocity = (
                            self.velocity_field[i-1, j, k] +
                            self.velocity_field[i+1, j, k] +
                            self.velocity_field[i, j-1, k] +
                            self.velocity_field[i, j+1, k] +
                            self.velocity_field[i, j, k-1] +
                            self.velocity_field[i, j, k+1]
                        ) / 6.0
                        
                        # Update velocity with diffusion
                        self.velocity_field[i, j, k] = (
                            (1 - viscosity * dt) * self.velocity_field[i, j, k] +
                            viscosity * dt * neighbors_velocity
                        )
        
        # Apply to refinement regions
        for region_key in self.refinement_regions:
            ref_region = self.refinement_regions[region_key]
            ref_size = ref_region['velocity'].shape[0]
            
            for i in range(1, ref_size-1):
                for j in range(1, ref_size-1):
                    for k in range(1, ref_size-1):
                        # Get state
                        state = int(cp.asnumpy(ref_region['state'][i, j, k]))
                        
                        # Apply viscosity
                        viscosity = self.viscosity[state]
                        
                        if viscosity > 0:
                            # Calculate average neighbor velocity
                            neighbors_velocity = (
                                ref_region['velocity'][i-1, j, k] +
                                ref_region['velocity'][i+1, j, k] +
                                ref_region['velocity'][i, j-1, k] +
                                ref_region['velocity'][i, j+1, k] +
                                ref_region['velocity'][i, j, k-1] +
                                ref_region['velocity'][i, j, k+1]
                            ) / 6.0
                            
                            # Apply diffusion
                            ref_region['velocity'][i, j, k] = (
                                (1 - viscosity * dt) * ref_region['velocity'][i, j, k] +
                                viscosity * dt * neighbors_velocity
                            )

    def smooth_refinement_region(self, region_key):
        """Apply spatial smoothing to a refinement region"""
        ref_region = self.refinement_regions[region_key]
        ref_size = ref_region['velocity'].shape[0]
        
        # Create a copy for smoothing
        smoothed_velocity = cp.copy(ref_region['velocity'])
        
        # Smooth interior cells
        for i in range(1, ref_size-1):
            for j in range(1, ref_size-1):
                for k in range(1, ref_size-1):
                    # Skip vacuum cells
                    if ref_region['state'][i, j, k] == self.VACUUM:
                        continue
                    
                    # Simple box blur smoothing
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]
                    
                    neighbor_sum = cp.zeros(3, dtype=cp.float32)
                    count = 0
                    
                    for ni, nj, nk in neighbors:
                        neighbor_sum += ref_region['velocity'][ni, nj, nk]
                        count += 1
                    
                    smoothing_factor = 0.2
                    if count > 0:
                        smoothed_velocity[i, j, k] = (
                            (1 - smoothing_factor) * ref_region['velocity'][i, j, k] +
                            smoothing_factor * (neighbor_sum / count)
                        )
        
        # Update with smoothed version
        ref_region['velocity'] = smoothed_velocity
        
    def apply_field_smoothing(self):
        """Apply spatial smoothing to fluid grid fields"""
        # Create smoothed copy of velocity field
        smoothed_velocity = cp.copy(self.velocity_field)
        
        # Simple box-blur smoothing for internal cells - executed on GPU
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Skip cells in vacuum state - preserve vortex structures
                    state = int(cp.asnumpy(self.state[i, j, k]))
                    if state == self.VACUUM:
                        continue
                        
                    # Average with neighbors (box filter)
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]
                    
                    neighbor_sum = cp.zeros(3, dtype=cp.float32)
                    count = 0
                    
                    for ni, nj, nk in neighbors:
                        neighbor_sum += self.velocity_field[ni, nj, nk]
                        count += 1
                    
                    # Weighted average - 80% original, 20% neighbors
                    smoothing_factor = 0.2
                    if count > 0:
                        smoothed_velocity[i, j, k] = (
                            (1 - smoothing_factor) * self.velocity_field[i, j, k] +
                            smoothing_factor * (neighbor_sum / count)
                        )
        
        # Update fields with smoothed versions
        self.velocity_field = smoothed_velocity
        
        # Also smooth each refinement region
        for region_key in self.refinement_regions:
            self.smooth_refinement_region(region_key)

    def get_total_energy(self):
        """Calculate total energy in fluid grid"""
        # Sum base grid energy on GPU, transfer result to CPU
        base_energy = float(cp.sum(self.energy_density).get()) * (self.cell_size**3)
        
        # Add energy from refinement regions
        refinement_energy = 0.0
        for region_key, region_grid in self.refinement_regions.items():
            base_i, base_j, base_k, level = region_key
            ref_factor = 2**level
            ref_cell_size = self.cell_size / ref_factor
            
            # Calculate velocity energy
            velocity_energy = cp.sum(cp.sum(region_grid['velocity']**2, axis=3)) * 0.5
            
            # Calculate energy from curl/vorticity
            vorticity_energy = cp.sum(cp.sum(region_grid['vorticity']**2, axis=3)) * 0.25
            
            # Calculate energy from pressure
            pressure_energy = cp.sum(region_grid['pressure']) * 0.5
            
            # Total energy in this refinement region
            region_energy = float(cp.sum(velocity_energy + vorticity_energy + pressure_energy).get())
            region_energy *= (ref_cell_size**3)  # Scale by cell volume
            
            refinement_energy += region_energy
        
        return base_energy + refinement_energy

    def update(self, particles, dt):
        """Main update function for the adaptive fluid grid"""
        # Update activity map and manage refinement regions
        self.update_activity_map(particles)
        self.manage_refinement_regions()
        
        # Update fields from particles
        self.update_fields_from_particles(particles, dt)
        
        # Apply spatial smoothing
        self.apply_field_smoothing()
        
        # Calculate vorticity and curl
        self.calculate_vorticity()
        
        # Calculate pressure field
        self.calculate_pressure()
        
        # Update fluid states
        self.update_fluid_states()
        
        # Apply state-specific dynamics
        self.apply_state_specific_dynamics(dt)
        
        # Sync data between resolution levels
        self.sync_hierarchical_data()
        
        # Return the number of active refinement regions (for monitoring)
        return len(self.refinement_regions)