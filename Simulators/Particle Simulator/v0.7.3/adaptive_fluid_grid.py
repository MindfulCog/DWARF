import numpy as np
import time
import logging
from numba import jit, cuda, prange
from typing import Tuple, List, Optional
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', category=Warning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptiveFluidGrid")

class AdaptiveFluidGridOptimized:
    """
    High-performance vectorized adaptive fluid grid for DWARF physics simulation.
    Completely removes object wrappers and uses pure NumPy/Numba acceleration.
    """
    
    def __init__(self, base_resolution=64, size=10.0, max_depth=2, use_gpu=True):
        """Initialize the optimized adaptive grid"""
        self.base_resolution = base_resolution
        self.size = size
        self.cell_size = size / base_resolution
        self.max_depth = max_depth
        
        # GPU acceleration setup
        self.use_gpu = use_gpu and cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Core grid arrays - all vectorized, no object wrappers
        shape = (base_resolution, base_resolution, base_resolution)
        
        # Fluid state arrays
        self.state = np.ones(shape, dtype=np.int32)  # 0=vacuum, 1=uncompressed, 2=compressed
        self.velocity_field = np.zeros(shape + (3,), dtype=np.float32)
        self.memory_field = np.zeros(shape + (3,), dtype=np.float32)
        self.pressure_field = np.zeros(shape, dtype=np.float32)
        self.density_field = np.zeros(shape, dtype=np.float32)
        self.temperature_field = np.ones(shape, dtype=np.float32) * 300.0  # Room temperature
        self.vorticity_magnitude = np.zeros(shape, dtype=np.float32)
        self.energy_density = np.zeros(shape, dtype=np.float32)
        
        # Particle tracking arrays
        self.particle_count = np.zeros(shape, dtype=np.int32)
        self.particle_mass = np.zeros(shape, dtype=np.float32)
        
        # Performance tracking
        self.update_time = 0.0
        self.stats = {
            'particle_mapping_time': 0.0,
            'fluid_dynamics_time': 0.0,
            'vectorized_ops_time': 0.0,
            'total_time': 0.0
        }
        
        # Initialize coordinate meshes for vectorized operations
        self._setup_coordinate_meshes()
        
        logger.info(f"Optimized grid initialized: {shape}, GPU: {self.use_gpu}")
    
    def _setup_coordinate_meshes(self):
        """Pre-compute coordinate meshes for vectorized operations"""
        res = self.base_resolution
        
        # Create coordinate arrays
        x = np.arange(res, dtype=np.float32)
        y = np.arange(res, dtype=np.float32)
        z = np.arange(res, dtype=np.float32)
        
        # Create meshgrids
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center coordinates
        center = res // 2
        self.X_centered = self.X - center
        self.Y_centered = self.Y - center
        self.Z_centered = self.Z - center
        
        # Radial distance from center
        self.R = np.sqrt(self.X_centered**2 + self.Y_centered**2 + self.Z_centered**2)
        self.R_safe = np.maximum(self.R, 0.1)  # Avoid division by zero
    
    @jit(nopython=True, parallel=True)
    def _map_particles_to_grid_numba(self, positions, velocities, masses, spins, particle_types):
        """Ultra-fast particle to grid mapping using Numba JIT compilation"""
        res = self.base_resolution
        cell_size = self.cell_size
        
        # Reset arrays
        particle_count = np.zeros((res, res, res), dtype=np.int32)
        particle_mass = np.zeros((res, res, res), dtype=np.float32)
        velocity_contrib = np.zeros((res, res, res, 3), dtype=np.float32)
        memory_contrib = np.zeros((res, res, res, 3), dtype=np.float32)
        
        # Parallel particle processing
        for p in prange(len(positions)):
            # Convert world position to grid coordinates
            gx = int((positions[p, 0] + self.size/2) / cell_size)
            gy = int((positions[p, 1] + self.size/2) / cell_size)
            gz = int((positions[p, 2] + self.size/2) / cell_size)
            
            # Apply periodic boundary conditions
            gx = gx % res
            gy = gy % res
            gz = gz % res
            
            # Influence radius based on particle type
            influence = 2 if particle_types[p] == 0 else 3  # proton vs others
            
            # Apply particle influence to surrounding cells
            for di in range(-influence, influence + 1):
                for dj in range(-influence, influence + 1):
                    for dk in range(-influence, influence + 1):
                        # Calculate grid indices with periodic boundary
                        gi = (gx + di) % res
                        gj = (gy + dj) % res
                        gk = (gz + dk) % res
                        
                        # Calculate distance-based weight
                        dist_sq = di*di + dj*dj + dk*dk
                        if dist_sq <= influence*influence:
                            weight = np.exp(-dist_sq / (2.0 * influence/2.5))
                            
                            # Update grid properties
                            particle_count[gi, gj, gk] += 1
                            particle_mass[gi, gj, gk] += masses[p] * weight
                            
                            # Velocity contribution
                            for k in range(3):
                                velocity_contrib[gi, gj, gk, k] += velocities[p, k] * weight
                            
                            # Memory field contribution from spin
                            for k in range(3):
                                memory_contrib[gi, gj, gk, k] += spins[p, k] * weight * 0.1
        
        return particle_count, particle_mass, velocity_contrib, memory_contrib
    
    def update_with_particles(self, particles, dt):
        """Main update function - fully vectorized"""
        start_time = time.time()
        
        # Extract particle data into numpy arrays for vectorized processing
        n_particles = len(particles)
        positions = np.zeros((n_particles, 3), dtype=np.float32)
        velocities = np.zeros((n_particles, 3), dtype=np.float32)
        masses = np.zeros(n_particles, dtype=np.float32)
        spins = np.zeros((n_particles, 3), dtype=np.float32)
        particle_types = np.zeros(n_particles, dtype=np.int32)
        
        # Vectorized data extraction
        for i, p in enumerate(particles):
            positions[i] = [p.position[0], p.position[1], p.position[2]]
            velocities[i] = [p.velocity[0], p.velocity[1], p.velocity[2]]
            masses[i] = p.mass
            spins[i] = [p.spin[0], p.spin[1], p.spin[2]]
            particle_types[i] = 0 if p.particle_type == "proton" else 1
        
        map_start = time.time()
        
        # Ultra-fast particle mapping using Numba
        if self.use_gpu and cuda.is_available():
            # GPU acceleration path
            self._map_particles_gpu(positions, velocities, masses, spins, particle_types)
        else:
            # CPU with Numba JIT
            count, mass, vel_contrib, mem_contrib = self._map_particles_to_grid_numba(
                positions, velocities, masses, spins, particle_types
            )
            
            self.particle_count = count
            self.particle_mass = mass
            self.velocity_field = vel_contrib
            self.memory_field = mem_contrib
        
        self.stats['particle_mapping_time'] = time.time() - map_start
        
        # Update fluid properties using vectorized operations
        fluid_start = time.time()
        self._update_fluid_dynamics_vectorized(dt)
        self.stats['fluid_dynamics_time'] = time.time() - fluid_start
        
        # Update derived quantities
        vec_start = time.time()
        self._update_derived_fields_vectorized()
        self.stats['vectorized_ops_time'] = time.time() - vec_start
        
        self.update_time = time.time() - start_time
        self.stats['total_time'] = self.update_time
        
        logger.debug(f"Grid update completed in {self.update_time*1000:.2f}ms")
        return self.update_time
    
    def _map_particles_gpu(self, positions, velocities, masses, spins, particle_types):
        """GPU-accelerated particle mapping using CuPy/CUDA"""
        try:
            import cupy as cp
            
            # Transfer data to GPU
            gpu_positions = cp.asarray(positions)
            gpu_velocities = cp.asarray(velocities)
            gpu_masses = cp.asarray(masses)
            gpu_spins = cp.asarray(spins)
            gpu_types = cp.asarray(particle_types)
            
            # GPU kernel for particle mapping
            # This would be a custom CUDA kernel for maximum performance
            # For now, use CuPy operations
            res = self.base_resolution
            
            # Convert to grid coordinates
            grid_coords = ((gpu_positions + self.size/2) / self.cell_size).astype(cp.int32) % res
            
            # Use advanced indexing for fast accumulation
            # This is a simplified version - full implementation would use custom kernels
            count_gpu = cp.zeros((res, res, res), dtype=cp.int32)
            mass_gpu = cp.zeros((res, res, res), dtype=cp.float32)
            
            # Transfer results back
            self.particle_count = cp.asnumpy(count_gpu)
            self.particle_mass = cp.asnumpy(mass_gpu)
            
        except ImportError:
            logger.warning("CuPy not available, falling back to CPU")
            self.use_gpu = False
            # Fallback to CPU version
            count, mass, vel_contrib, mem_contrib = self._map_particles_to_grid_numba(
                positions, velocities, masses, spins, particle_types
            )
            self.particle_count = count
            self.particle_mass = mass
    
    def _update_fluid_dynamics_vectorized(self, dt):
        """Update fluid dynamics using pure vectorized operations"""
        
        # Calculate density from particle mass
        cell_volume = self.cell_size ** 3
        self.density_field = self.particle_mass / cell_volume
        
        # Pressure from ideal gas law (vectorized)
        R_gas = 8.314  # Gas constant
        self.pressure_field = self.density_field * R_gas * self.temperature_field
        
        # Velocity field advection using vectorized operations
        # Create shifted arrays for derivatives
        vx = self.velocity_field[:, :, :, 0]
        vy = self.velocity_field[:, :, :, 1]
        vz = self.velocity_field[:, :, :, 2]
        
        # Pressure gradient (vectorized central differences)
        px_grad = np.roll(self.pressure_field, -1, axis=0) - np.roll(self.pressure_field, 1, axis=0)
        py_grad = np.roll(self.pressure_field, -1, axis=1) - np.roll(self.pressure_field, 1, axis=1)
        pz_grad = np.roll(self.pressure_field, -1, axis=2) - np.roll(self.pressure_field, 1, axis=2)
        
        # Apply forces (vectorized)
        force_scale = dt * 0.1
        mask = self.density_field > 0  # Only apply to non-empty cells
        
        self.velocity_field[mask, 0] -= px_grad[mask] * force_scale / (2 * self.cell_size)
        self.velocity_field[mask, 1] -= py_grad[mask] * force_scale / (2 * self.cell_size)
        self.velocity_field[mask, 2] -= pz_grad[mask] * force_scale / (2 * self.cell_size)
        
        # Apply viscosity (vectorized diffusion)
        viscosity = 0.01
        self._apply_diffusion_vectorized(self.velocity_field, viscosity * dt)
    
    def _apply_diffusion_vectorized(self, field, diffusion_rate):
        """Apply diffusion using vectorized Laplacian operator"""
        if len(field.shape) == 4:  # Vector field
            for component in range(field.shape[3]):
                self._diffuse_scalar_vectorized(field[:, :, :, component], diffusion_rate)
        else:  # Scalar field
            self._diffuse_scalar_vectorized(field, diffusion_rate)
    
    def _diffuse_scalar_vectorized(self, field, diffusion_rate):
        """Vectorized scalar diffusion using 6-point Laplacian stencil"""
        # Create shifted arrays for Laplacian
        laplacian = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        )
        
        # Apply diffusion
        field += diffusion_rate * laplacian
    
    def _update_derived_fields_vectorized(self):
        """Update all derived fields using vectorized operations"""
        
        # Vorticity calculation (curl of velocity field)
        vx, vy, vz = self.velocity_field[:, :, :, 0], self.velocity_field[:, :, :, 1], self.velocity_field[:, :, :, 2]
        
        # Compute curl components using central differences
        dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * self.cell_size)
        dvx_dz = (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / (2 * self.cell_size)
        dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * self.cell_size)
        dvy_dz = (np.roll(vy, -1, axis=2) - np.roll(vy, 1, axis=2)) / (2 * self.cell_size)
        dvz_dx = (np.roll(vz, -1, axis=0) - np.roll(vz, 1, axis=0)) / (2 * self.cell_size)
        dvz_dy = (np.roll(vz, -1, axis=1) - np.roll(vz, 1, axis=1)) / (2 * self.cell_size)
        
        # Vorticity magnitude
        curl_x = dvz_dy - dvy_dz
        curl_y = dvx_dz - dvz_dx
        curl_z = dvy_dx - dvx_dy
        
        self.vorticity_magnitude = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        
        # Energy density (vectorized)
        velocity_squared = np.sum(self.velocity_field**2, axis=3)
        kinetic_energy = 0.5 * self.density_field * velocity_squared
        pressure_energy = self.pressure_field * 0.1
        
        self.energy_density = self.energy_density * 0.98 + (kinetic_energy + pressure_energy) * 0.02
    
    def initialize_vectorized(self):
        """Initialize grid with interesting structure using vectorized operations"""
        logger.info("Initializing grid with vectorized operations...")
        
        # Create initial vortex structure using pre-computed meshes
        r = self.R_safe
        
        # Memory field (radial pattern)
        decay = np.exp(-0.02 * r)
        self.memory_field[:, :, :, 0] = 0.1 * self.X_centered / r * decay
        self.memory_field[:, :, :, 1] = 0.1 * self.Y_centered / r * decay
        self.memory_field[:, :, :, 2] = 0.1 * self.Z_centered / r * decay
        
        # Velocity field (circular motion)
        theta = np.arctan2(self.Y_centered, self.X_centered)
        velocity_decay = np.exp(-0.01 * r)
        
        self.velocity_field[:, :, :, 0] = -0.05 * self.Y_centered / r * velocity_decay
        self.velocity_field[:, :, :, 1] = 0.05 * self.X_centered / r * velocity_decay
        self.velocity_field[:, :, :, 2] = 0.02 * np.sin(2*theta) * velocity_decay
        
        # Update derived fields
        self._update_derived_fields_vectorized()
        
        logger.info("Grid initialization complete.")
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        return {
            'total_update_time': self.update_time,
            'particle_mapping_efficiency': self.stats['particle_mapping_time'] / self.update_time if self.update_time > 0 else 0,
            'fluid_dynamics_efficiency': self.stats['fluid_dynamics_time'] / self.update_time if self.update_time > 0 else 0,
            'vectorization_efficiency': self.stats['vectorized_ops_time'] / self.update_time if self.update_time > 0 else 0,
            'using_gpu': self.use_gpu,
            'grid_resolution': self.base_resolution,
            **self.stats
        }
    
    def get_state_counts(self):
        """Get counts of different fluid states"""
        return {
            'vacuum': int(np.sum(self.state == 0)),
            'uncompressed': int(np.sum(self.state == 1)),
            'compressed': int(np.sum(self.state == 2))
        }
    
    def get_total_energy(self):
        """Calculate total energy in the grid"""
        return float(np.sum(self.energy_density))
    
    def get_field_data(self, field_name):
        """Get field data for visualization or analysis"""
        field_map = {
            'velocity': self.velocity_field,
            'pressure': self.pressure_field,
            'density': self.density_field,
            'vorticity': self.vorticity_magnitude,
            'energy': self.energy_density,
            'memory': self.memory_field,
            'temperature': self.temperature_field
        }
        
        return field_map.get(field_name, None)