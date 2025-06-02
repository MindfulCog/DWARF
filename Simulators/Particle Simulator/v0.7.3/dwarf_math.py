import numpy as np
import cupy as cp

# Standardized lowercase naming
class dwarf_math:
    def __init__(self, use_gpu=True):
        # Check if GPU is available
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        # Initialize other parameters
        
    def update_memory_field(self, grid, dt):
        """GPU-accelerated memory field update"""
        if self.use_gpu:
            # Transfer to GPU if not already there
            memory_field_gpu = cp.asarray(grid.memory_field)
            state_gpu = cp.asarray(grid.state)
            field_decay_gpu = cp.asarray(grid.field_decay)
            
            # Vectorized approach using fused operations
            # Extract indices based on state
            state_indices = cp.asnumpy(state_gpu).astype(int)
            
            # Create decay factors array based on states
            decay_factors = cp.array([grid.field_decay[int(s)] for s in state_indices.flatten()])
            decay_factors = decay_factors.reshape(grid.base_resolution, grid.base_resolution, grid.base_resolution)
            
            # Apply decay operation
            memory_field_gpu *= decay_factors ** dt
            
            # Transfer back to CPU if needed
            if grid.needs_cpu_sync:
                grid.memory_field = cp.asnumpy(memory_field_gpu)
            else:
                grid.memory_field = memory_field_gpu
        else:
            # CPU fallback
            for i in range(grid.base_resolution):
                for j in range(grid.base_resolution):
                    for k in range(grid.base_resolution):
                        state_idx = int(grid.state[i, j, k])
                        decay_rate = grid.field_decay[state_idx]
                        grid.memory_field[i, j, k] *= decay_rate ** dt
    
    def calculate_gradient(self, field, dx=1.0):
        """Calculate gradient of a field"""
        if self.use_gpu:
            field_gpu = cp.asarray(field)
            gradient = cp.gradient(field_gpu, dx)
            if hasattr(field, 'needs_cpu_sync') and field.needs_cpu_sync:
                return [cp.asnumpy(g) for g in gradient]
            return gradient
        else:
            return np.gradient(field, dx)
    
    def calculate_divergence(self, vector_field, dx=1.0):
        """Calculate divergence of a vector field"""
        if self.use_gpu:
            vector_field_gpu = [cp.asarray(component) for component in vector_field]
            # Calculate partial derivatives
            dudx = cp.gradient(vector_field_gpu[0], dx)[0]
            dvdy = cp.gradient(vector_field_gpu[1], dx)[1]
            dwdz = cp.gradient(vector_field_gpu[2], dx)[2]
            
            divergence = dudx + dvdy + dwdz
            
            if hasattr(vector_field[0], 'needs_cpu_sync') and vector_field[0].needs_cpu_sync:
                return cp.asnumpy(divergence)
            return divergence
        else:
            # CPU fallback
            dudx = np.gradient(vector_field[0], dx)[0]
            dvdy = np.gradient(vector_field[1], dx)[1]
            dwdz = np.gradient(vector_field[2], dx)[2]
            return dudx + dvdy + dwdz

# For backward compatibility, but using the lowercase version
dwarf_math_gpu = dwarf_math