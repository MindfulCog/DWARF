import numpy as np
import cupy as cp

class AdaptiveFluidGrid:
    def __init__(self, size=64, max_level=3, use_gpu=True, cell_size=1.0, **kwargs):
        # Handle all possible parameter name variations for compatibility
        # The **kwargs will catch any other parameters that might be passed in
        
        # Check if GPU is available
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
        # Grid parameters - CONVERT ALL DIMENSIONS TO INT!
        self.base_resolution = int(size)  # Convert to int to prevent float issues
        self.size = int(size)  # Alias for backward compatibility
        self.max_level = int(max_level)  # Ensure integer
        self.needs_cpu_sync = False  # Flag to control CPU syncing
        self.cell_size = float(cell_size)  # Ensure this is float
        
        # Initialize grids - NOW WITH INTEGER DIMENSIONS
        self.base_grid = self.xp.zeros((self.base_resolution, self.base_resolution, self.base_resolution, 3), 
                                      dtype=self.xp.float32)  # Vector field with 3 components
        
        # Initialize memory fields (also as vector field)
        self.memory_field = self.xp.zeros_like(self.base_grid)
        self.state = self.xp.zeros((self.base_resolution, self.base_resolution, self.base_resolution), 
                                   dtype=self.xp.int32)
        
        # Initialize decay rates for different states
        self.field_decay = np.array([0.99, 0.95, 0.9, 0.85, 0.8], dtype=np.float32)
        
        # Initialize refined grids with increasing resolution
        self.refined_grids = []
        for level in range(self.max_level):
            level_resolution = self.base_resolution * (2 ** (level + 1))
            grid = self.xp.zeros((level_resolution, level_resolution, level_resolution, 3), dtype=self.xp.float32)
            self.refined_grids.append(grid)
        
        # Grid metadata for refinement criteria
        self.refinement_threshold = 0.5
        self.refinement_regions = [[] for _ in range(self.max_level)]
        
    def sync_hierarchical_data(self):
        """GPU-accelerated hierarchical grid synchronization"""
        if self.use_gpu:
            # Ensure all grids are on GPU
            if isinstance(self.base_grid, np.ndarray):
                self.base_grid = cp.asarray(self.base_grid)
            
            refined_grids_gpu = []
            for grid in self.refined_grids:
                if isinstance(grid, np.ndarray):
                    grid = cp.asarray(grid)
                refined_grids_gpu.append(grid)
            
            # Process each refinement level
            for level in range(self.max_level):
                refinement_factor = 2 ** (level + 1)
                
                # Create out grid for this level
                out_grid = refined_grids_gpu[level]
                
                # Check if we have explicit refinement regions
                if len(self.refinement_regions[level]) > 0:
                    # Process only specific regions
                    for region in self.refinement_regions[level]:
                        # Extract region bounds
                        x_start, x_end = region['x_range']
                        y_start, y_end = region['y_range']
                        z_start, z_end = region['z_range']
                        
                        # Calculate corresponding refined grid indices
                        rx_start = x_start * refinement_factor
                        rx_end = x_end * refinement_factor
                        ry_start = y_start * refinement_factor
                        ry_end = y_end * refinement_factor
                        rz_start = z_start * refinement_factor
                        rz_end = z_end * refinement_factor
                        
                        # GPU kernel to update this region
                        grid_dim = (32, 32, 1)
                        block_dim = ((rx_end - rx_start + 31) // 32, 
                                    (ry_end - ry_start + 31) // 32, 
                                    1)
                        
                        trilinear_interp_kernel = cp.RawKernel(r'''
                        extern "C" __global__ void trilinear_interp(
                            const float* base_grid, float* refined_grid,
                            int base_res, int refined_res, int vector_components,
                            int rx_start, int ry_start, int rz_start,
                            int rx_end, int ry_end, int rz_end,
                            float refactor_inv
                        ) {
                            int rx = blockIdx.x * blockDim.x + threadIdx.x + rx_start;
                            int ry = blockIdx.y * blockDim.y + threadIdx.y + ry_start;
                            
                            if (rx >= rx_end || ry >= ry_end) return;
                            
                            for (int rz = rz_start; rz < rz_end; rz++) {
                                // Convert to base grid coordinates
                                float x = rx * refactor_inv;
                                float y = ry * refactor_inv;
                                float z = rz * refactor_inv;
                                
                                // Integer and fractional parts
                                int x0 = (int)x;
                                int y0 = (int)y;
                                int z0 = (int)z;
                                
                                // Ensure within bounds
                                if (x0 >= base_res-1) x0 = base_res-2;
                                if (y0 >= base_res-1) y0 = base_res-2;
                                if (z0 >= base_res-1) z0 = base_res-2;
                                
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;
                                
                                float dx = x - x0;
                                float dy = y - y0;
                                float dz = z - z0;
                                
                                for (int c = 0; c < vector_components; c++) {
                                    // Calculate base grid indices with component offset
                                    int c000 = ((x0*base_res*base_res + y0*base_res + z0)*vector_components) + c;
                                    int c001 = ((x0*base_res*base_res + y0*base_res + z1)*vector_components) + c;
                                    int c010 = ((x0*base_res*base_res + y1*base_res + z0)*vector_components) + c;
                                    int c011 = ((x0*base_res*base_res + y1*base_res + z1)*vector_components) + c;
                                    int c100 = ((x1*base_res*base_res + y0*base_res + z0)*vector_components) + c;
                                    int c101 = ((x1*base_res*base_res + y0*base_res + z1)*vector_components) + c;
                                    int c110 = ((x1*base_res*base_res + y1*base_res + z0)*vector_components) + c;
                                    int c111 = ((x1*base_res*base_res + y1*base_res + z1)*vector_components) + c;
                                
                                    // Trilinear interpolation
                                    float c00 = base_grid[c000] * (1-dx) + base_grid[c100] * dx;
                                    float c01 = base_grid[c001] * (1-dx) + base_grid[c101] * dx;
                                    float c10 = base_grid[c010] * (1-dx) + base_grid[c110] * dx;
                                    float c11 = base_grid[c011] * (1-dx) + base_grid[c111] * dx;
                                    
                                    float c0 = c00 * (1-dy) + c10 * dy;
                                    float c1 = c01 * (1-dy) + c11 * dy;
                                    
                                    float result = c0 * (1-dz) + c1 * dz;
                                    
                                    // Write to refined grid
                                    int refined_idx = ((rx*refined_res*refined_res + ry*refined_res + rz)*vector_components) + c;
                                    refined_grid[refined_idx] = result;
                                }
                            }
                        }
                        ''', 'trilinear_interp')
                        
                        refined_res = self.base_resolution * refinement_factor
                        refactor_inv = 1.0 / refinement_factor
                        vector_components = self.base_grid.shape[3]
                        
                        # Reshape to contiguous 1D arrays for CUDA
                        base_grid_flat = self.base_grid.reshape(-1)
                        refined_grid_flat = out_grid.reshape(-1)
                        
                        trilinear_interp_kernel(grid_dim, block_dim, 
                                              (base_grid_flat, refined_grid_flat, 
                                               self.base_resolution, refined_res, vector_components,
                                               rx_start, ry_start, rz_start,
                                               rx_end, ry_end, rz_end,
                                               refactor_inv))
                else:
                    # Process entire grid
                    refined_res = self.base_resolution * refinement_factor
                    vector_components = self.base_grid.shape[3]
                    
                    # Create a grid of coordinates for the refined grid
                    x = cp.linspace(0, self.base_resolution-1, refined_res, dtype=cp.float32)
                    y = cp.linspace(0, self.base_resolution-1, refined_res, dtype=cp.float32)
                    z = cp.linspace(0, self.base_resolution-1, refined_res, dtype=cp.float32)
                    
                    # Create kernel to process entire grid
                    full_interp_kernel = cp.RawKernel(r'''
                    extern "C" __global__ void full_grid_interp(
                        const float* base_grid, float* refined_grid,
                        const float* x_coords, const float* y_coords, const float* z_coords,
                        int base_res, int refined_res, int vector_components
                    ) {
                        int rx = blockIdx.x * blockDim.x + threadIdx.x;
                        int ry = blockIdx.y * blockDim.y + threadIdx.y;
                        int rz = blockIdx.z * blockDim.z + threadIdx.z;
                        
                        if (rx >= refined_res || ry >= refined_res || rz >= refined_res) return;
                        
                        // Get interpolation coordinates
                        float x = x_coords[rx];
                        float y = y_coords[ry];
                        float z = z_coords[rz];
                        
                        // Integer and fractional parts
                        int x0 = (int)x;
                        int y0 = (int)y;
                        int z0 = (int)z;
                        
                        // Ensure within bounds
                        if (x0 >= base_res-1) x0 = base_res-2;
                        if (y0 >= base_res-1) y0 = base_res-2;
                        if (z0 >= base_res-1) z0 = base_res-2;
                        
                        int x1 = x0 + 1;
                        int y1 = y0 + 1;
                        int z1 = z0 + 1;
                        
                        float dx = x - x0;
                        float dy = y - y0;
                        float dz = z - z0;
                        
                        for (int c = 0; c < vector_components; c++) {
                            // Calculate base grid indices with component offset
                            int c000 = ((x0*base_res*base_res + y0*base_res + z0)*vector_components) + c;
                            int c001 = ((x0*base_res*base_res + y0*base_res + z1)*vector_components) + c;
                            int c010 = ((x0*base_res*base_res + y1*base_res + z0)*vector_components) + c;
                            int c011 = ((x0*base_res*base_res + y1*base_res + z1)*vector_components) + c;
                            int c100 = ((x1*base_res*base_res + y0*base_res + z0)*vector_components) + c;
                            int c101 = ((x1*base_res*base_res + y0*base_res + z1)*vector_components) + c;
                            int c110 = ((x1*base_res*base_res + y1*base_res + z0)*vector_components) + c;
                            int c111 = ((x1*base_res*base_res + y1*base_res + z1)*vector_components) + c;
                        
                            // Trilinear interpolation
                            float c00 = base_grid[c000] * (1-dx) + base_grid[c100] * dx;
                            float c01 = base_grid[c001] * (1-dx) + base_grid[c101] * dx;
                            float c10 = base_grid[c010] * (1-dx) + base_grid[c110] * dx;
                            float c11 = base_grid[c011] * (1-dx) + base_grid[c111] * dx;
                            
                            float c0 = c00 * (1-dy) + c10 * dy;
                            float c1 = c01 * (1-dy) + c11 * dy;
                            
                            float result = c0 * (1-dz) + c1 * dz;
                            
                            // Write to refined grid
                            int refined_idx = ((rx*refined_res*refined_res + ry*refined_res + rz)*vector_components) + c;
                            refined_grid[refined_idx] = result;
                        }
                    }
                    ''', 'full_grid_interp')
                    
                    # Configure grid and block dimensions
                    block_dim = (8, 8, 8)
                    grid_dim = (
                        (refined_res + block_dim[0] - 1) // block_dim[0],
                        (refined_res + block_dim[1] - 1) // block_dim[1],
                        (refined_res + block_dim[2] - 1) // block_dim[2]
                    )
                    
                    # Reshape to contiguous 1D arrays for CUDA
                    base_grid_flat = self.base_grid.reshape(-1)
                    refined_grid_flat = out_grid.reshape(-1)
                    
                    # Execute kernel
                    full_interp_kernel(grid_dim, block_dim, 
                                     (base_grid_flat, refined_grid_flat, 
                                      x, y, z,
                                      self.base_resolution, refined_res, vector_components))
                
                # Update refined_grids_gpu
                refined_grids_gpu[level] = out_grid
            
            # Update class attributes
            self.refined_grids = refined_grids_gpu
            
            # Sync back to CPU if needed
            if self.needs_cpu_sync:
                self.base_grid = cp.asnumpy(self.base_grid)
                self.refined_grids = [cp.asnumpy(grid) for grid in self.refined_grids]
        else:
            # CPU fallback implementation
            for level in range(self.max_level):
                refinement_factor = 2 ** (level + 1)
                refined_res = self.base_resolution * refinement_factor
                vector_components = self.base_grid.shape[3]
                
                # Initialize output grid
                refined_grid = np.zeros((refined_res, refined_res, refined_res, vector_components), dtype=np.float32)
                
                # Simple CPU trilinear interpolation
                for rx in range(refined_res):
                    for ry in range(refined_res):
                        for rz in range(refined_res):
                            # Convert to base grid coordinates
                            x = rx / refinement_factor
                            y = ry / refinement_factor
                            z = rz / refinement_factor
                            
                            # Simple trilinear interpolation
                            x0, y0, z0 = int(x), int(y), int(z)
                            
                            # Ensure within bounds
                            x0 = min(x0, self.base_resolution - 2)
                            y0 = min(y0, self.base_resolution - 2)
                            z0 = min(z0, self.base_resolution - 2)
                            
                            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                            
                            dx, dy, dz = x - x0, y - y0, z - z0
                            
                            # Trilinear interpolation for each component
                            for c in range(vector_components):
                                c00 = self.base_grid[x0, y0, z0, c] * (1 - dx) + self.base_grid[x1, y0, z0, c] * dx
                                c01 = self.base_grid[x0, y0, z1, c] * (1 - dx) + self.base_grid[x1, y0, z1, c] * dx
                                c10 = self.base_grid[x0, y1, z0, c] * (1 - dx) + self.base_grid[x1, y1, z0, c] * dx
                                c11 = self.base_grid[x0, y1, z1, c] * (1 - dx) + self.base_grid[x1, y1, z1, c] * dx
                                
                                c0 = c00 * (1 - dy) + c10 * dy
                                c1 = c01 * (1 - dy) + c11 * dy
                                
                                refined_grid[rx, ry, rz, c] = c0 * (1 - dz) + c1 * dz
                
                self.refined_grids[level] = refined_grid
    
    def identify_refinement_regions(self):
        """Identify regions that need refinement based on field values"""
        # Use magnitude of memory field vector for thresholding
        if self.use_gpu:
            memory_field_gpu = cp.asarray(self.memory_field)
            # Calculate magnitude of vector field
            magnitude = cp.sqrt(cp.sum(memory_field_gpu**2, axis=3))
            threshold = self.refinement_threshold
            
            # Find regions above threshold
            mask = magnitude > threshold
            
            # Fetch mask to CPU for processing
            mask_cpu = cp.asnumpy(mask)
        else:
            # CPU calculation
            magnitude = np.sqrt(np.sum(self.memory_field**2, axis=3))
            threshold = self.refinement_threshold
            mask_cpu = magnitude > threshold
        
        # Identify contiguous regions
        from scipy.ndimage import label, find_objects
        
        labeled_array, num_features = label(mask_cpu)
        
        # Extract regions
        regions = []
        for region in find_objects(labeled_array):
            x_range = (region[0].start, region[0].stop)
            y_range = (region[1].start, region[1].stop)
            z_range = (region[2].start, region[2].stop)
            
            regions.append({
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range
            })
        
        # Distribute regions across refinement levels
        self.refinement_regions = [[] for _ in range(self.max_level)]
        
        # Simple strategy: larger regions get lower refinement levels
        regions.sort(key=lambda r: (r['x_range'][1] - r['x_range'][0]) * 
                               (r['y_range'][1] - r['y_range'][0]) * 
                               (r['z_range'][1] - r['z_range'][0]),
                     reverse=True)
        
        for i, region in enumerate(regions):
            level = min(i % self.max_level, self.max_level - 1)
            self.refinement_regions[level].append(region)

# For backward compatibility
AdaptiveFluidGridGPU = AdaptiveFluidGrid