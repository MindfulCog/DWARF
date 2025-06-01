import cupy as cp
import numpy as np
from collections import defaultdict

class AdaptiveFluidGrid:
    """3D fluid grid with adaptive resolution for field calculations"""
    
    def __init__(self, base_resolution=128, size=10.0, max_depth=2):
        self.base_resolution = base_resolution
        self.size = size
        self.cell_size = size / base_resolution
        self.max_depth = max_depth  # Maximum refinement levels
        
        # Initialize fields on GPU - base resolution grid
        self.velocity_field = cp.zeros((base_resolution, base_resolution, base_resolution, 3), dtype=cp.float32)
        self.pressure_field = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.float32)
        self.charge_density = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.float32)
        self.vorticity = cp.zeros((base_resolution, base_resolution, base_resolution, 3), dtype=cp.float32)
        self.energy_density = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.float32)
        self.vorticity_magnitude = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.float32)
        
        # Activity map to track where higher resolution is needed
        self.activity_map = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.float32)
        
        # Refinement threshold
        self.refinement_threshold = 0.3  # Activity level that triggers refinement
        
        # State mask for the 3 fluid states (DWARF theory)
        self.state = cp.zeros((base_resolution, base_resolution, base_resolution), dtype=cp.int8)
        
        # Hierarchical grid storage - Dictionary mapping (i,j,k,level) to sub-grids
        self.refinement_regions = {}
        
        # State constants - the three fluid states in DWARF theory
        self.UNCOMPRESSED = 0  # Default fluid state
        self.COMPRESSED = 1    # High pressure state
        self.VACUUM = 2        # High vorticity state (toroidal)
        
        # State-specific properties (keep on CPU for easy access)
        self.viscosity = {
            self.UNCOMPRESSED: 0.5,  # Medium viscosity
            self.COMPRESSED: 0.8,    # High viscosity in compressed state
            self.VACUUM: 0.1         # Low viscosity in vacuum state
        }
        
        self.field_decay = {
            self.UNCOMPRESSED: 0.95,  # Medium decay
            self.COMPRESSED: 0.98,    # Slow decay in compressed state
            self.VACUUM: 0.99         # Very slow decay in vacuum state
        }
        
        self.state_transition_thresholds = {
            'pressure_for_compression': 1.5,   # Threshold for transitioning to compressed state
            'vorticity_for_vacuum': 2.0,       # Threshold for transitioning to vacuum state
            'recovery_threshold': 0.5          # Threshold for returning to uncompressed state
        }
        
    def initialize(self):
        """Initialize grid to default state"""
        # Default state is uncompressed (0)
        self.state.fill(self.UNCOMPRESSED)
        self.activity_map.fill(0)
        self.refinement_regions.clear()
        
    def world_to_grid(self, position):
        """Convert world position to grid coordinates at base resolution"""
        # Keep on CPU for individual particle calculations
        grid_pos = (position + self.size/2) / self.cell_size
        return np.clip(grid_pos, 0, self.base_resolution-1)
    
    def grid_to_world(self, grid_position):
        """Convert grid coordinates to world position"""
        # Keep on CPU for individual position calculations
        return grid_position * self.cell_size - self.size/2
        
    def update_activity_map(self, particles):
        """Update activity map based on particle positions and field properties"""
        # Decay existing activity
        self.activity_map *= 0.95
        
        # Add activity around particles
        for particle in particles:
            grid_pos = self.world_to_grid(particle.position)
            i, j, k = np.floor(grid_pos).astype(int)
            
            # Skip if outside grid
            if not (0 <= i < self.base_resolution-1 and 
                    0 <= j < self.base_resolution-1 and 
                    0 <= k < self.base_resolution-1):
                continue
            
            # Add activity based on particle mass and charge
            activity_radius = max(3, int(particle.mass * 2))
            activity_strength = abs(particle.charge) * 0.8 + 0.2
            
            # Create activity zone around particle
            for di in range(-activity_radius, activity_radius+1):
                for dj in range(-activity_radius, activity_radius+1):
                    for dk in range(-activity_radius, activity_radius+1):
                        ni, nj, nk = i+di, j+dj, k+dk
                        
                        if not (0 <= ni < self.base_resolution and 
                                0 <= nj < self.base_resolution and 
                                0 <= nk < self.base_resolution):
                            continue
                        
                        # Distance-based falloff
                        distance = np.sqrt(di**2 + dj**2 + dk**2)
                        if distance > activity_radius:
                            continue
                            
                        falloff = 1.0 - (distance / activity_radius)
                        self.activity_map[ni, nj, nk] = max(
                            cp.asnumpy(self.activity_map[ni, nj, nk]), 
                            activity_strength * falloff
                        )
                        
        # Enhance activity where vorticity or charge is high
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # High vorticity regions should be active
                    vorticity_factor = min(1.0, cp.asnumpy(self.vorticity_magnitude[i, j, k]) / 2.0)
                    
                    # High charge density regions should be active
                    charge_factor = min(1.0, abs(cp.asnumpy(self.charge_density[i, j, k])))
                    
                    # Set activity based on field properties
                    self.activity_map[i, j, k] = max(
                        cp.asnumpy(self.activity_map[i, j, k]),
                        vorticity_factor * 0.8,
                        charge_factor * 0.6
                    )
        
        # Add special activity for potential hydrogen bonds
        self.enhance_potential_bonds(particles)
    
    def enhance_potential_bonds(self, particles):
        """Add high activity in regions where hydrogen bonds might form"""
        protons = [p for p in particles if p.particle_type == "proton"]
        electrons = [p for p in particles if p.particle_type == "electron"]
        
        # Check each proton-electron pair
        for proton in protons:
            for electron in electrons:
                distance = np.linalg.norm(proton.position - electron.position)
                
                # If they're within potential bonding range
                if distance < 3.0:  # Suitable range for hydrogen formation
                    # Get midpoint between particles
                    midpoint = (proton.position + electron.position) / 2
                    grid_pos = self.world_to_grid(midpoint)
                    
                    # Set high activity for refined simulation
                    i, j, k = np.floor(grid_pos).astype(int)
                    if (0 <= i < self.base_resolution-1 and 
                        0 <= j < self.base_resolution-1 and 
                        0 <= k < self.base_resolution-1):
                        # Mark this as very high activity - ensure highest resolution
                        bond_radius = max(3, int(1.5 * distance / self.cell_size))
                        
                        # Create high-resolution zone around potential bond
                        for di in range(-bond_radius, bond_radius+1):
                            for dj in range(-bond_radius, bond_radius+1):
                                for dk in range(-bond_radius, bond_radius+1):
                                    ni, nj, nk = i+di, j+dj, k+dk
                                    
                                    if not (0 <= ni < self.base_resolution and 
                                            0 <= nj < self.base_resolution and 
                                            0 <= nk < self.base_resolution):
                                        continue
                                    
                                    # Distance-based falloff
                                    dist = np.sqrt(di**2 + dj**2 + dk**2)
                                    if dist > bond_radius:
                                        continue
                                        
                                    # Highest activity near potential bonds
                                    self.activity_map[ni, nj, nk] = 1.0
                    
                    # Also enhance orbital plane for this pair
                    self.enhance_orbital_region(proton, electron)
    
    def enhance_orbital_region(self, proton, electron):
        """Enhance resolution in the orbital plane between proton-electron pairs"""
        # Calculate orbital plane normal based on relative velocity
        r_vec = electron.position - proton.position
        v_rel = electron.velocity - proton.velocity
        
        if np.linalg.norm(r_vec) < 1e-6 or np.linalg.norm(v_rel) < 1e-6:
            return
            
        # Orbital angular momentum direction (perpendicular to orbital plane)
        orbital_normal = np.cross(r_vec, v_rel)
        if np.linalg.norm(orbital_normal) > 1e-6:
            orbital_normal = orbital_normal / np.linalg.norm(orbital_normal)
        else:
            return
            
        # Create a high-resolution disk in the orbital plane
        center = (proton.position + electron.position) / 2
        radius = np.linalg.norm(r_vec) * 1.5
        
        # Generate points in the orbital plane
        grid_center = self.world_to_grid(center)
        max_grid_radius = int(radius / self.cell_size) + 1
        
        # Create two orthogonal vectors in the orbital plane
        if abs(orbital_normal[2]) < 0.9:
            v1 = np.cross(orbital_normal, [0, 0, 1])
        else:
            v1 = np.cross(orbital_normal, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(orbital_normal, v1)
        
        # Enhance resolution in this orbital plane
        # Will create a disk of high-resolution cells
        for r in range(max_grid_radius):
            circumference = 2 * np.pi * r
            if circumference < 1:
                points = 1
            else:
                points = int(circumference * 1.5)
                
            for p in range(points):
                angle = 2 * np.pi * p / points
                # Point in orbital plane
                offset = r * (v1 * np.cos(angle) + v2 * np.sin(angle))
                point = center + offset * self.cell_size
                
                grid_pos = self.world_to_grid(point)
                i, j, k = np.floor(grid_pos).astype(int)
                
                if (0 <= i < self.base_resolution and 
                    0 <= j < self.base_resolution and 
                    0 <= k < self.base_resolution):
                    self.activity_map[i, j, k] = max(
                        cp.asnumpy(self.activity_map[i, j, k]), 
                        0.9
                    )
    
    def manage_refinement_regions(self):
        """Update refinement regions based on activity map"""
        # First, clear outdated regions
        regions_to_remove = []
        for region_key in self.refinement_regions.keys():
            base_i, base_j, base_k, level = region_key
            
            # Skip if outside grid boundaries
            if not (0 <= base_i < self.base_resolution and 
                    0 <= base_j < self.base_resolution and 
                    0 <= base_k < self.base_resolution):
                regions_to_remove.append(region_key)
                continue
            
            # Check if this region is still active enough for refinement
            if cp.asnumpy(self.activity_map[base_i, base_j, base_k]) < self.refinement_threshold:
                regions_to_remove.append(region_key)
        
        # Remove inactive regions
        for key in regions_to_remove:
            del self.refinement_regions[key]
            
        # Now create new regions where needed
        for i in range(0, self.base_resolution-1, 2):
            for j in range(0, self.base_resolution-1, 2):
                for k in range(0, self.base_resolution-1, 2):
                    if cp.asnumpy(self.activity_map[i, j, k]) >= self.refinement_threshold:
                        # This cell needs refinement
                        region_key = (i, j, k, 1)  # Level 1 refinement
                        
                        if region_key not in self.refinement_regions:
                            # Create a new high-resolution sub-grid for this region
                            sub_res = 4  # 2x resolution in each dimension
                            self.refinement_regions[region_key] = {
                                'velocity': cp.zeros((sub_res, sub_res, sub_res, 3), dtype=cp.float32),
                                'pressure': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                'charge': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                'vorticity': cp.zeros((sub_res, sub_res, sub_res, 3), dtype=cp.float32),
                                'vorticity_mag': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                'state': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.int8)
                            }
                            
                            # Initialize with interpolated data from base grid
                            self.initialize_refinement_region(region_key)
                            
                        # Check for further refinement (level 2)
                        if self.max_depth >= 2 and cp.asnumpy(self.activity_map[i, j, k]) >= 0.7:
                            region_key_l2 = (i, j, k, 2)  # Level 2 refinement
                            
                            if region_key_l2 not in self.refinement_regions:
                                # Create level 2 refinement (4x base resolution)
                                sub_res = 8  # 4x resolution in each dimension
                                self.refinement_regions[region_key_l2] = {
                                    'velocity': cp.zeros((sub_res, sub_res, sub_res, 3), dtype=cp.float32),
                                    'pressure': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                    'charge': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                    'vorticity': cp.zeros((sub_res, sub_res, sub_res, 3), dtype=cp.float32),
                                    'vorticity_mag': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.float32),
                                    'state': cp.zeros((sub_res, sub_res, sub_res), dtype=cp.int8)
                                }
                                
                                # Initialize with interpolated data
                                self.initialize_refinement_region(region_key_l2)
    
    def initialize_refinement_region(self, region_key):
        """Initialize a refinement region with interpolated data from parent grid"""
        base_i, base_j, base_k, level = region_key
        
        if level == 1:
            # Interpolate from base grid
            parent_grid = self
            parent_data = {
                'velocity': parent_grid.velocity_field[
                    base_i:base_i+2, base_j:base_j+2, base_k:base_k+2],
                'pressure': parent_grid.pressure_field[
                    base_i:base_i+2, base_j:base_j+2, base_k:base_k+2],
                'charge': parent_grid.charge_density[
                    base_i:base_i+2, base_j:base_j+2, base_k:base_k+2],
                'state': parent_grid.state[
                    base_i:base_i+2, base_j:base_j+2, base_k:base_k+2]
            }
        else:
            # Interpolate from level-1 grid
            parent_key = (base_i, base_j, base_k, level-1)
            if parent_key not in self.refinement_regions:
                # Parent doesn't exist - create it first
                self.initialize_refinement_region(parent_key)
            
            parent_data = self.refinement_regions[parent_key]
        
        # Get destination grid
        dest_grid = self.refinement_regions[region_key]
        dest_res = 2 ** (level + 1)  # Resolution of this subgrid (2^(level+1))
        
        # Perform trilinear interpolation to fill refined grid
        # For simplicity, we'll just use nearest-neighbor here, but could be improved
        for i in range(dest_res):
            for j in range(dest_res):
                for k in range(dest_res):
                    # Map to parent coordinates
                    pi = min(int(i / 2), parent_data['velocity'].shape[0] - 1)
                    pj = min(int(j / 2), parent_data['velocity'].shape[1] - 1)
                    pk = min(int(k / 2), parent_data['velocity'].shape[2] - 1)
                    
                    # Copy data from parent
                    dest_grid['velocity'][i, j, k] = parent_data['velocity'][pi, pj, pk]
                    dest_grid['pressure'][i, j, k] = parent_data['pressure'][pi, pj, pk]
                    dest_grid['charge'][i, j, k] = parent_data['charge'][pi, pj, pk]
                    dest_grid['state'][i, j, k] = parent_data['state'][pi, pj, pk]
    
    def update_fields_from_particles(self, particles, dt):
        """Update fields based on particle positions and properties"""
        # Reset fields on GPU
        self.charge_density.fill(0)
        self.velocity_field.fill(0)
        
        # Distribute particle properties to grid using trilinear interpolation
        for particle in particles:
            # Map particle position to grid (on CPU)
            grid_pos = self.world_to_grid(particle.position)
            
            # Ensure position is within grid bounds
            if (grid_pos < 0).any() or (grid_pos >= self.base_resolution).any():
                continue
                
            # Use trilinear interpolation to update surrounding cells
            i, j, k = np.floor(grid_pos).astype(int)
            
            if (i >= 0 and i < self.base_resolution-1 and 
                j >= 0 and j < self.base_resolution-1 and 
                k >= 0 and k < self.base_resolution-1):
                
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
                
                # Update charge density - transfer data to GPU
                charge_contribution = cp.float32(particle.charge)
                
                # Update the grid on GPU
                self.charge_density[i, j, k] += charge_contribution * w000
                self.charge_density[i, j, k+1] += charge_contribution * w001
                self.charge_density[i, j+1, k] += charge_contribution * w010
                self.charge_density[i, j+1, k+1] += charge_contribution * w011
                self.charge_density[i+1, j, k] += charge_contribution * w100
                self.charge_density[i+1, j, k+1] += charge_contribution * w101
                self.charge_density[i+1, j+1, k] += charge_contribution * w110
                self.charge_density[i+1, j+1, k+1] += charge_contribution * w111
                
                # Update velocity field (weighted by mass) - transfer to GPU
                velocity_contribution = cp.asarray(particle.velocity * (particle.mass / 100.0), dtype=cp.float32)
                
                for m in range(3):  # For each velocity component
                    self.velocity_field[i, j, k, m] += velocity_contribution[m] * w000
                    self.velocity_field[i, j, k+1, m] += velocity_contribution[m] * w001
                    self.velocity_field[i, j+1, k, m] += velocity_contribution[m] * w010
                    self.velocity_field[i, j+1, k+1, m] += velocity_contribution[m] * w011
                    self.velocity_field[i+1, j, k, m] += velocity_contribution[m] * w100
                    self.velocity_field[i+1, j, k+1, m] += velocity_contribution[m] * w101
                    self.velocity_field[i+1, j+1, k, m] += velocity_contribution[m] * w110
                    self.velocity_field[i+1, j+1, k+1, m] += velocity_contribution[m] * w111
                    
                # Add spin contribution to velocity field (creates vorticity)
                # This is crucial for the DWARF vacuum state formation
                spin_contribution = cp.asarray(np.cross(particle.spin, particle.velocity) * 0.05, dtype=cp.float32)
                
                for m in range(3):
                    self.velocity_field[i, j, k, m] += spin_contribution[m] * w000
                    self.velocity_field[i, j, k+1, m] += spin_contribution[m] * w001
                    self.velocity_field[i, j+1, k, m] += spin_contribution[m] * w010
                    self.velocity_field[i, j+1, k+1, m] += spin_contribution[m] * w011
                    self.velocity_field[i+1, j, k, m] += spin_contribution[m] * w100
                    self.velocity_field[i+1, j, k+1, m] += spin_contribution[m] * w101
                    self.velocity_field[i+1, j+1, k, m] += spin_contribution[m] * w110
                    self.velocity_field[i+1, j+1, k+1, m] += spin_contribution[m] * w111
                    
            # Now update any refinement regions this particle may affect
            self.update_refinement_regions_for_particle(particle)
    
    def update_refinement_regions_for_particle(self, particle):
        """Update refinement regions affected by this particle"""
        grid_pos = self.world_to_grid(particle.position)
        i, j, k = np.floor(grid_pos).astype(int)
        
        # Check each refinement region
        for region_key, region_grid in self.refinement_regions.items():
            base_i, base_j, base_k, level = region_key
            
            # Skip if particle is not in this region
            if not (base_i <= i < base_i+2 and 
                    base_j <= j < base_j+2 and 
                    base_k <= k < base_k+2):
                continue
                
            # Calculate refined position within this region
            ref_factor = 2**level
            ref_cell_size = self.cell_size / ref_factor
            
            # Get subgrid coordinates
            sub_x = (particle.position[0] + self.size/2 - base_i*self.cell_size) / ref_cell_size
            sub_y = (particle.position[1] + self.size/2 - base_j*self.cell_size) / ref_cell_size
            sub_z = (particle.position[2] + self.size/2 - base_k*self.cell_size) / ref_cell_size
            
            # Ensure within bounds of refinement region
            sub_x = max(0, min(sub_x, ref_factor*2-1))
            sub_y = max(0, min(sub_y, ref_factor*2-1))
            sub_z = max(0, min(sub_z, ref_factor*2-1))
            
            # Get integer and fractional parts
            si, sj, sk = int(sub_x), int(sub_y), int(sub_z)
            dx, dy, dz = sub_x - si, sub_y - sj, sub_z - sk
            
            # Skip if outside refinement region
            if not (0 <= si < ref_factor*2-1 and 
                    0 <= sj < ref_factor*2-1 and 
                    0 <= sk < ref_factor*2-1):
                continue
            
            # Weights for trilinear interpolation
            w000 = (1-dx)*(1-dy)*(1-dz)
            w001 = (1-dx)*(1-dy)*dz
            w010 = (1-dx)*dy*(1-dz)
            w011 = (1-dx)*dy*dz
            w100 = dx*(1-dy)*(1-dz)
            w101 = dx*(1-dy)*dz
            w110 = dx*dy*(1-dz)
            w111 = dx*dy*dz
            
            # Update charge density in refinement region
            charge_contribution = cp.float32(particle.charge)
            
            region_grid['charge'][si, sj, sk] += charge_contribution * w000
            region_grid['charge'][si, sj, sk+1] += charge_contribution * w001
            region_grid['charge'][si, sj+1, sk] += charge_contribution * w010
            region_grid['charge'][si, sj+1, sk+1] += charge_contribution * w011
            region_grid['charge'][si+1, sj, sk] += charge_contribution * w100
            region_grid['charge'][si+1, sj, sk+1] += charge_contribution * w101
            region_grid['charge'][si+1, sj+1, sk] += charge_contribution * w110
            region_grid['charge'][si+1, sj+1, sk+1] += charge_contribution * w111
            
            # Update velocity field in refinement region
            velocity_contribution = cp.asarray(particle.velocity * (particle.mass / 100.0), dtype=cp.float32)
            
            for m in range(3):  # For each velocity component
                region_grid['velocity'][si, sj, sk, m] += velocity_contribution[m] * w000
                region_grid['velocity'][si, sj, sk+1, m] += velocity_contribution[m] * w001
                region_grid['velocity'][si, sj+1, sk, m] += velocity_contribution[m] * w010
                region_grid['velocity'][si, sj+1, sk+1, m] += velocity_contribution[m] * w011
                region_grid['velocity'][si+1, sj, sk, m] += velocity_contribution[m] * w100
                region_grid['velocity'][si+1, sj, sk+1, m] += velocity_contribution[m] * w101
                region_grid['velocity'][si+1, sj+1, sk, m] += velocity_contribution[m] * w110
                region_grid['velocity'][si+1, sj+1, sk+1, m] += velocity_contribution[m] * w111
            
            # Add spin contribution
            spin_contribution = cp.asarray(np.cross(particle.spin, particle.velocity) * 0.05, dtype=cp.float32)
            
            for m in range(3):
                region_grid['velocity'][si, sj, sk, m] += spin_contribution[m] * w000
                region_grid['velocity'][si, sj, sk+1, m] += spin_contribution[m] * w001
                region_grid['velocity'][si, sj+1, sk, m] += spin_contribution[m] * w010
                region_grid['velocity'][si, sj+1, sk+1, m] += spin_contribution[m] * w011
                region_grid['velocity'][si+1, sj, sk, m] += spin_contribution[m] * w100
                region_grid['velocity'][si+1, sj, sk+1, m] += spin_contribution[m] * w101
                region_grid['velocity'][si+1, sj+1, sk, m] += spin_contribution[m] * w110
                region_grid['velocity'][si+1, sj+1, sk+1, m] += spin_contribution[m] * w111
    
    def apply_field_smoothing(self):
        """Apply spatial smoothing to fluid grid fields"""
        # Create smoothed copy of velocity field
        smoothed_velocity = cp.copy(self.velocity_field)
        
        # Simple box-blur smoothing for internal cells - executed on GPU
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Skip cells in vacuum state - preserve vortex structures
                    # FIX: Convert CuPy array to Python integer before comparison
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
        for region_key, region_grid in self.refinement_regions.items():
            self.smooth_refinement_region(region_key)
    
    def smooth_refinement_region(self, region_key):
        """Apply smoothing to a refinement region"""
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['velocity'].shape[0]  # Resolution of this region
        
        # Create smoothed copy
        smoothed_velocity = cp.copy(region_grid['velocity'])
        
        # Apply smoothing
        for i in range(1, ref_res-1):
            for j in range(1, ref_res-1):
                for k in range(1, ref_res-1):
                    # Skip vacuum cells
                    state = int(cp.asnumpy(region_grid['state'][i, j, k]))
                    if state == self.VACUUM:
                        continue
                    
                    # Average with neighbors
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]
                    
                    neighbor_sum = cp.zeros(3, dtype=cp.float32)
                    count = 0
                    
                    for ni, nj, nk in neighbors:
                        neighbor_sum += region_grid['velocity'][ni, nj, nk]
                        count += 1
                    
                    # Weighted average - 80% original, 20% neighbors
                    smoothing_factor = 0.2
                    if count > 0:
                        smoothed_velocity[i, j, k] = (
                            (1 - smoothing_factor) * region_grid['velocity'][i, j, k] +
                            smoothing_factor * (neighbor_sum / count)
                        )
        
        # Update with smoothed velocity
        region_grid['velocity'] = smoothed_velocity
    
    def calculate_vorticity(self):
        """Calculate vorticity from velocity field (curl of velocity)"""
        # Calculate on GPU using central differences
        dx = dy = dz = self.cell_size
        
        # For each internal cell (not on the boundary)
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Get velocity components for surrounding cells
                    vx = self.velocity_field[:, :, :, 0]  # x-component of velocity
                    vy = self.velocity_field[:, :, :, 1]  # y-component of velocity
                    vz = self.velocity_field[:, :, :, 2]  # z-component of velocity
                    
                    # Calculate derivatives using central difference
                    dvz_dy = (vz[i, j+1, k] - vz[i, j-1, k]) / (2*dy)
                    dvy_dz = (vy[i, j, k+1] - vy[i, j, k-1]) / (2*dz)
                    
                    dvx_dz = (vx[i, j, k+1] - vx[i, j, k-1]) / (2*dz)
                    dvz_dx = (vz[i+1, j, k] - vz[i-1, j, k]) / (2*dx)
                    
                    dvy_dx = (vy[i+1, j, k] - vy[i-1, j, k]) / (2*dx)
                    dvx_dy = (vx[i, j+1, k] - vx[i, j-1, k]) / (2*dy)
                    
                    # Vorticity is the curl of velocity
                    self.vorticity[i, j, k, 0] = dvz_dy - dvy_dz  # x-component 
                    self.vorticity[i, j, k, 1] = dvx_dz - dvz_dx  # y-component
                    self.vorticity[i, j, k, 2] = dvy_dx - dvx_dy  # z-component
                    
                    # Calculate vorticity magnitude
                    self.vorticity_magnitude[i, j, k] = cp.sqrt(
                        self.vorticity[i, j, k, 0]**2 + 
                        self.vorticity[i, j, k, 1]**2 + 
                        self.vorticity[i, j, k, 2]**2
                    )
        
        # Also calculate vorticity for each refinement region
        for region_key, region_grid in self.refinement_regions.items():
            self.calculate_refinement_vorticity(region_key)
    
    def calculate_refinement_vorticity(self, region_key):
        """Calculate vorticity for a refinement region"""
        base_i, base_j, base_k, level = region_key
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['velocity'].shape[0]  # Resolution of this region
        
        # Cell size for this refinement level
        ref_factor = 2**level
        ref_cell_size = self.cell_size / ref_factor
        
        # Calculate vorticity for each cell in the refinement region
        for i in range(1, ref_res-1):
            for j in range(1, ref_res-1):
                for k in range(1, ref_res-1):
                    # Get velocity components
                    vx = region_grid['velocity'][:, :, :, 0]
                    vy = region_grid['velocity'][:, :, :, 1]
                    vz = region_grid['velocity'][:, :, :, 2]
                    
                    # Calculate derivatives
                    dvz_dy = (vz[i, j+1, k] - vz[i, j-1, k]) / (2*ref_cell_size)
                    dvy_dz = (vy[i, j, k+1] - vy[i, j, k-1]) / (2*ref_cell_size)
                    
                    dvx_dz = (vx[i, j, k+1] - vx[i, j, k-1]) / (2*ref_cell_size)
                    dvz_dx = (vz[i+1, j, k] - vz[i-1, j, k]) / (2*ref_cell_size)
                    
                    dvy_dx = (vy[i+1, j, k] - vy[i-1, j, k]) / (2*ref_cell_size)
                    dvx_dy = (vx[i, j+1, k] - vx[i, j-1, k]) / (2*ref_cell_size)
                    
                    # Vorticity components
                    region_grid['vorticity'][i, j, k, 0] = dvz_dy - dvy_dz
                    region_grid['vorticity'][i, j, k, 1] = dvx_dz - dvz_dx
                    region_grid['vorticity'][i, j, k, 2] = dvy_dx - dvx_dy
                    
                    # Magnitude
                    region_grid['vorticity_mag'][i, j, k] = cp.sqrt(
                        region_grid['vorticity'][i, j, k, 0]**2 + 
                        region_grid['vorticity'][i, j, k, 1]**2 + 
                        region_grid['vorticity'][i, j, k, 2]**2
                    )
    
    def calculate_pressure(self):
        """Calculate pressure field from fluid state"""
        # Reset pressure field
        self.pressure_field.fill(0)
        
        # Calculate on GPU
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # Pressure from charge density (absolute value for compression effect)
                    charge_pressure = cp.abs(self.charge_density[i, j, k]) * 0.5
                    
                    # Pressure from velocity divergence
                    vx = self.velocity_field[:, :, :, 0]
                    vy = self.velocity_field[:, :, :, 1]
                    vz = self.velocity_field[:, :, :, 2]
                    
                    dx = dy = dz = self.cell_size
                    
                    dvx_dx = (vx[i+1, j, k] - vx[i-1, j, k]) / (2*dx)
                    dvy_dy = (vy[i, j+1, k] - vy[i, j-1, k]) / (2*dy)
                    dvz_dz = (vz[i, j, k+1] - vz[i, j, k-1]) / (2*dz)
                    
                    # Negative divergence contributes to pressure (compression)
                    divergence = dvx_dx + dvy_dy + dvz_dz
                    div_pressure = cp.maximum(0, -divergence * 0.5)
                    
                    # State-specific pressure scaling
                    state_factor = 1.0
                    # FIX: Convert CuPy array to Python integer before comparison
                    state = int(cp.asnumpy(self.state[i, j, k]))
                    if state == self.COMPRESSED:
                        state_factor = 1.5  # Higher pressure retention in compressed state
                    elif state == self.VACUUM:
                        state_factor = 0.5  # Lower pressure in vacuum state
                    
                    # Total pressure
                    self.pressure_field[i, j, k] = (charge_pressure + div_pressure) * state_factor
                    
                    # Calculate energy density (kinetic + potential)
                    vel_squared = cp.sum(self.velocity_field[i, j, k]**2)
                    vorticity_squared = cp.sum(self.vorticity[i, j, k]**2)
                    
                    # Energy density from velocities, vorticity and pressure
                    self.energy_density[i, j, k] = (
                        0.5 * vel_squared +                # Kinetic energy
                        0.25 * vorticity_squared +         # Rotational energy
                        0.5 * self.pressure_field[i, j, k] # Potential energy
                    )
        
        # Calculate pressure for each refinement region
        for region_key, region_grid in self.refinement_regions.items():
            self.calculate_refinement_pressure(region_key)
    
    def calculate_refinement_pressure(self, region_key):
        """Calculate pressure for a refinement region"""
        base_i, base_j, base_k, level = region_key
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['velocity'].shape[0]  # Resolution of this region
        
        # Cell size for this refinement level
        ref_factor = 2**level
        ref_cell_size = self.cell_size / ref_factor
        
        # Calculate pressure for each cell
        for i in range(1, ref_res-1):
            for j in range(1, ref_res-1):
                for k in range(1, ref_res-1):
                    # Charge contribution
                    charge_pressure = cp.abs(region_grid['charge'][i, j, k]) * 0.5
                    
                    # Divergence contribution
                    vx = region_grid['velocity'][:, :, :, 0]
                    vy = region_grid['velocity'][:, :, :, 1]
                    vz = region_grid['velocity'][:, :, :, 2]
                    
                    dvx_dx = (vx[i+1, j, k] - vx[i-1, j, k]) / (2*ref_cell_size)
                    dvy_dy = (vy[i, j+1, k] - vy[i, j-1, k]) / (2*ref_cell_size)
                    dvz_dz = (vz[i, j, k+1] - vz[i, j, k-1]) / (2*ref_cell_size)
                    
                    divergence = dvx_dx + dvy_dy + dvz_dz
                    div_pressure = cp.maximum(0, -divergence * 0.5)
                    
                    # State-specific scaling
                    state_factor = 1.0
                    state = int(cp.asnumpy(region_grid['state'][i, j, k]))
                    if state == self.COMPRESSED:
                        state_factor = 1.5
                    elif state == self.VACUUM:
                        state_factor = 0.5
                    
                    # Set pressure
                    region_grid['pressure'][i, j, k] = (charge_pressure + div_pressure) * state_factor
    
    def update_fluid_states(self):
        """Update fluid state based on pressure and vorticity"""
        # Update states based on thresholds
        vorticity_threshold = self.state_transition_thresholds['vorticity_for_vacuum']
        pressure_threshold = self.state_transition_thresholds['pressure_for_compression']
        recovery_threshold = self.state_transition_thresholds['recovery_threshold']
        
        for i in range(self.base_resolution):
            for j in range(self.base_resolution):
                for k in range(self.base_resolution):
                    # Current state
                    # FIX: Convert CuPy array to Python integer before comparison
                    current_state = int(cp.asnumpy(self.state[i, j, k]))
                    
                    # State transition logic
                    if self.vorticity_magnitude[i, j, k] > vorticity_threshold:
                        # High vorticity creates vacuum (toroidal) state
                        new_state = self.VACUUM
                    elif self.pressure_field[i, j, k] > pressure_threshold:
                        # High pressure creates compressed state
                        new_state = self.COMPRESSED
                    elif (self.pressure_field[i, j, k] < recovery_threshold and 
                          self.vorticity_magnitude[i, j, k] < recovery_threshold):
                        # Low pressure and vorticity return to uncompressed state
                        new_state = self.UNCOMPRESSED
                    else:
                        # No change
                        new_state = current_state
                        
                    self.state[i, j, k] = new_state
        
        # Update states in refinement regions
        for region_key, region_grid in self.refinement_regions.items():
            self.update_refinement_states(region_key)
    
    def update_refinement_states(self, region_key):
        """Update fluid states in a refinement region"""
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['vorticity_mag'].shape[0]
        
        # Get thresholds
        vorticity_threshold = self.state_transition_thresholds['vorticity_for_vacuum']
        pressure_threshold = self.state_transition_thresholds['pressure_for_compression']
        recovery_threshold = self.state_transition_thresholds['recovery_threshold']
        
        # Update states
        for i in range(ref_res):
            for j in range(ref_res):
                for k in range(ref_res):
                    current_state = int(cp.asnumpy(region_grid['state'][i, j, k]))
                    
                    if region_grid['vorticity_mag'][i, j, k] > vorticity_threshold:
                        new_state = self.VACUUM
                    elif region_grid['pressure'][i, j, k] > pressure_threshold:
                        new_state = self.COMPRESSED
                    elif (region_grid['pressure'][i, j, k] < recovery_threshold and
                          region_grid['vorticity_mag'][i, j, k] < recovery_threshold):
                        new_state = self.UNCOMPRESSED
                    else:
                        new_state = current_state
                    
                    region_grid['state'][i, j, k] = new_state
    
    def apply_state_specific_dynamics(self, dt):
        """Apply different dynamics based on fluid state"""
        dt_gpu = cp.float32(dt)
        
        for i in range(1, self.base_resolution-1):
            for j in range(1, self.base_resolution-1):
                for k in range(1, self.base_resolution-1):
                    # FIX: Convert CuPy array to Python integer before using as dictionary key
                    cell_state = int(cp.asnumpy(self.state[i, j, k]))
                    
                    # Apply appropriate viscosity/damping (state-specific)
                    # Get viscosity value from CPU dict based on state
                    viscosity = cp.float32(self.viscosity[cell_state])
                    self.velocity_field[i, j, k] *= (1.0 - viscosity * dt_gpu)
                    
                    # In vacuum state (toroidal), enhance vorticity
                    if cell_state == self.VACUUM:
                        # Add boost to vorticity - this maintains the toroidal state
                        vorticity_direction = self.vorticity[i, j, k]
                        vorticity_mag = cp.linalg.norm(vorticity_direction)
                        if vorticity_mag > 0:
                            vorticity_direction = vorticity_direction / vorticity_mag
                            self.velocity_field[i, j, k] += vorticity_direction * 0.02 * dt_gpu
                    
                    # In compressed state, pressure gradient affects velocity
                    elif cell_state == self.COMPRESSED:
                        # Calculate pressure gradient
                        pressure_gradient = cp.array([
                            self.pressure_field[i+1, j, k] - self.pressure_field[i-1, j, k],
                            self.pressure_field[i, j+1, k] - self.pressure_field[i, j-1, k],
                            self.pressure_field[i, j, k+1] - self.pressure_field[i, j, k-1]
                        ]) / (2 * self.cell_size)
                        
                        # Velocity affected by pressure gradient
                        self.velocity_field[i, j, k] -= pressure_gradient * 0.08 * dt_gpu
        
        # Apply state-specific dynamics to refinement regions
        for region_key, region_grid in self.refinement_regions.items():
            self.apply_refinement_state_dynamics(region_key, dt)
    
    def apply_refinement_state_dynamics(self, region_key, dt):
        """Apply state-specific dynamics to a refinement region"""
        base_i, base_j, base_k, level = region_key
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['velocity'].shape[0]
        
        # Cell size for this refinement level
        ref_factor = 2**level
        ref_cell_size = self.cell_size / ref_factor
        
        dt_gpu = cp.float32(dt)
        
        # Apply state-specific dynamics
        for i in range(1, ref_res-1):
            for j in range(1, ref_res-1):
                for k in range(1, ref_res-1):
                    cell_state = int(cp.asnumpy(region_grid['state'][i, j, k]))
                    
                    # Apply viscosity
                    viscosity = cp.float32(self.viscosity[cell_state])
                    region_grid['velocity'][i, j, k] *= (1.0 - viscosity * dt_gpu)
                    
                    # State-specific effects
                    if cell_state == self.VACUUM:
                        # Enhance vorticity
                        vorticity_direction = region_grid['vorticity'][i, j, k]
                        vorticity_mag = cp.linalg.norm(vorticity_direction)
                        if vorticity_mag > 0:
                            vorticity_direction = vorticity_direction / vorticity_mag
                            region_grid['velocity'][i, j, k] += vorticity_direction * 0.02 * dt_gpu
                    
                    elif cell_state == self.COMPRESSED:
                        # Pressure effects
                        pressure_gradient = cp.array([
                            region_grid['pressure'][i+1, j, k] - region_grid['pressure'][i-1, j, k],
                            region_grid['pressure'][i, j+1, k] - region_grid['pressure'][i, j-1, k],
                            region_grid['pressure'][i, j, k+1] - region_grid['pressure'][i, j, k-1]
                        ]) / (2 * ref_cell_size)
                        
                        region_grid['velocity'][i, j, k] -= pressure_gradient * 0.08 * dt_gpu
    
    def sync_hierarchical_data(self):
        """Synchronize data between different resolution levels"""
        # Step 1: Push data up from refinement regions to base grid
        for region_key, region_grid in self.refinement_regions.items():
            base_i, base_j, base_k, level = region_key
            ref_res = region_grid['velocity'].shape[0]  # Resolution of this region
            
            # Skip if outside base grid bounds
            if not (0 <= base_i < self.base_resolution-1 and 
                    0 <= base_j < self.base_resolution-1 and 
                    0 <= base_k < self.base_resolution-1):
                continue
            
            # For each cell in the base grid that overlaps this region
            for i_offset in range(2):
                for j_offset in range(2):
                    for k_offset in range(2):
                        i, j, k = base_i + i_offset, base_j + j_offset, base_k + k_offset
                        
                        # Skip if outside base grid bounds
                        if not (0 <= i < self.base_resolution and 
                                0 <= j < self.base_resolution and 
                                0 <= k < self.base_resolution):
                            continue
                        
                        # Calculate start/end indices in the refinement region for this base cell
                        start_i = i_offset * (ref_res // 2)
                        start_j = j_offset * (ref_res // 2)
                        start_k = k_offset * (ref_res // 2)
                        end_i = start_i + (ref_res // 2)
                        end_j = start_j + (ref_res // 2)
                        end_k = start_k + (ref_res // 2)
                        
                        # Average the refinement region values for this base cell
                        # Velocity
                        avg_velocity = cp.mean(
                            region_grid['velocity'][start_i:end_i, start_j:end_j, start_k:end_k], 
                            axis=(0, 1, 2)
                        )
                        self.velocity_field[i, j, k] = avg_velocity
                        
                        # Pressure
                        avg_pressure = cp.mean(
                            region_grid['pressure'][start_i:end_i, start_j:end_j, start_k:end_k]
                        )
                        self.pressure_field[i, j, k] = avg_pressure
                        
                        # Charge
                        avg_charge = cp.mean(
                            region_grid['charge'][start_i:end_i, start_j:end_j, start_k:end_k]
                        )
                        self.charge_density[i, j, k] = avg_charge
                        
                        # State (mode)
                        state_counts = cp.bincount(region_grid['state'][start_i:end_i, start_j:end_j, start_k:end_k].flatten())
                        self.state[i, j, k] = cp.argmax(state_counts)
        
        # Step 2: Push data down from parent levels to child levels where needed
        # This ensures consistency at boundaries between different resolution levels
        
        # First, update level 1 regions from base grid
        for region_key, region_grid in sorted(self.refinement_regions.items(), key=lambda x: x[0][3]):
            base_i, base_j, base_k, level = region_key
            
            if level == 1:
                # This is a level 1 region, update boundaries from base grid
                self.update_refinement_boundaries_from_base(region_key)
            else:
                # This is a higher-level region, update from parent level
                parent_key = (base_i, base_j, base_k, level-1)
                if parent_key in self.refinement_regions:
                    self.update_refinement_boundaries_from_parent(region_key, parent_key)
    
    def update_refinement_boundaries_from_base(self, region_key):
        """Update boundaries of a level 1 refinement region from base grid"""
        base_i, base_j, base_k, level = region_key
        region_grid = self.refinement_regions[region_key]
        ref_res = region_grid['velocity'].shape[0]
        
        # Refinement factor
        ref_factor = 2
        
        # Update boundaries (just an example for the x=0 face)
        # For a complete implementation, you'd need to do this for all 6 faces
        
        # X = 0 face
        i_base = base_i
        if i_base > 0:
            # Get values from base grid
            for j_offset in range(ref_res):
                j_base = base_j + (j_offset // ref_factor)
                j_frac = (j_offset % ref_factor) / ref_factor
                
                for k_offset in range(ref_res):
                    k_base = base_k + (k_offset // ref_factor)
                    k_frac = (k_offset % ref_factor) / ref_factor
                    
                    # Skip if outside base grid
                    if not (0 <= j_base < self.base_resolution and 
                            0 <= k_base < self.base_resolution):
                        continue
                    
                    # Interpolate from base grid to refinement boundary
                    j_lerp = j_frac
                    k_lerp = k_frac
                    
                    # Get base grid values - first handle the simple case where we're exactly on grid points
                    if j_lerp == 0 and k_lerp == 0:
                        velocity = self.velocity_field[i_base-1, j_base, k_base]
                        pressure = self.pressure_field[i_base-1, j_base, k_base]
                        charge = self.charge_density[i_base-1, j_base, k_base]
                        state = self.state[i_base-1, j_base, k_base]
                    else:
                        # Need to interpolate
                        # (simplified - you'd need proper trilinear interpolation here)
                        velocity = self.velocity_field[i_base-1, j_base, k_base]
                        pressure = self.pressure_field[i_base-1, j_base, k_base]
                        charge = self.charge_density[i_base-1, j_base, k_base]
                        state = self.state[i_base-1, j_base, k_base]
                    
                    # Set values at refinement boundary
                    region_grid['velocity'][0, j_offset, k_offset] = velocity
                    region_grid['pressure'][0, j_offset, k_offset] = pressure
                    region_grid['charge'][0, j_offset, k_offset] = charge
                    region_grid['state'][0, j_offset, k_offset] = state
        
        # Similar for other boundaries...
    
    def update_refinement_boundaries_from_parent(self, region_key, parent_key):
        """Update boundaries of a refinement region from its parent region"""
        # This is a simplified implementation - a full implementation would handle all boundaries
        # and do proper interpolation between resolution levels
        child_grid = self.refinement_regions[region_key]
        parent_grid = self.refinement_regions[parent_key]
        
        child_res = child_grid['velocity'].shape[0]
        parent_res = parent_grid['velocity'].shape[0]
        
        # Update X=0 boundary as an example
        for j in range(child_res):
            for k in range(child_res):
                # Map to parent coordinates
                p_j = min(j // 2, parent_res - 1)
                p_k = min(k // 2, parent_res - 1)
                
                # Copy data from parent
                child_grid['velocity'][0, j, k] = parent_grid['velocity'][0, p_j, p_k]
                child_grid['pressure'][0, j, k] = parent_grid['pressure'][0, p_j, p_k]
                child_grid['charge'][0, j, k] = parent_grid['charge'][0, p_j, p_k]
                child_grid['state'][0, j, k] = parent_grid['state'][0, p_j, p_k]
    
    def update(self, particles, dt):
        """Main update function for the adaptive fluid grid"""
        # Update activity map and manage refinement regions
        self.update_activity_map(particles)
        self.manage_refinement_regions()
        
        # Update fields from particles
        self.update_fields_from_particles(particles, dt)
        
        # Apply spatial smoothing
        self.apply_field_smoothing

# This file contains the completion of methods that were cut off

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