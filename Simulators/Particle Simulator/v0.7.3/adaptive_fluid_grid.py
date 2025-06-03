"""
Adaptive Fluid Grid module for the particle simulator.
This version uses vectorized operations with CuPy/NumPy for high performance.
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    print("CuPy not available - falling back to NumPy only (CPU)")
    cp = None
    HAS_CUPY = False

class AdaptiveFluidGrid:
    """
    A 3D adaptive fluid grid that simulates fluid dynamics.
    This version is optimized for performance using vectorized operations.
    """
    
    def __init__(self, base_resolution=16, size=10.0, use_gpu=True, max_depth=0, 
                 periodic=False, **kwargs):
        """
        Initialize the adaptive fluid grid.
        
        Args:
            base_resolution: Base resolution of the grid (int)
            size: Physical size of the domain (float for cubic domain or tuple of 3 floats)
            use_gpu: Whether to use GPU acceleration if available (bool)
            max_depth: Maximum refinement depth for adaptive grid (int)
            periodic: Whether to use periodic boundary conditions (bool)
            **kwargs: Additional parameters for backward compatibility
        """
        # Initialize basic parameters
        self.base_resolution = base_resolution
        self.max_depth = max_depth
        self.periodic = periodic
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Handle domain size
        if isinstance(size, (int, float)):
            # If a single value is provided, create a cubic domain
            self.domain_size = (float(size), float(size), float(size))
            self.size = float(size)  # For backward compatibility
        else:
            # Otherwise use the provided dimensions
            self.domain_size = size
            self.size = max(size)  # For backward compatibility
        
        # Calculate cell size
        self.cell_size = min(self.domain_size) / base_resolution
        
        # Set physical parameters before initializing fields
        self.viscosity = 0.1
        self.density = 1.0
        self.gravity = 9.81
        self.vorticity_confinement = 0.1
        
        # Determine if we can use GPU
        self.use_gpu = use_gpu and HAS_CUPY
        
        # Array module to use (cupy or numpy)
        self.xp = cp if self.use_gpu else np
        
        # Simulation control
        self.enable_vorticity = True
        self.enable_energy = True
        self.debug_timing = False  # Set to False in production for better performance
        
        # Adaptive grid structure (compatibility with original)
        self.subgrids = {}  # Dictionary to store refined subgrids
        self.refined_cells = set()  # Set of cells that have been refined
        
        # Finally initialize fields
        self._initialize_fields()
        
        # Calculate derived fields like vorticity_magnitude
        self._update_derived_fields()
    
    def _initialize_fields(self):
        """Initialize all fluid fields with zeros."""
        res = self.base_resolution
        xp = self.xp
        
        # Main fields (always created)
        self.velocity_field = xp.zeros((3, res, res, res), dtype=xp.float32)
        self.pressure_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Optional fields (created on demand)
        self.vorticity_field = xp.zeros((3, res, res, res), dtype=xp.float32)
        self.energy_field = xp.zeros((res, res, res), dtype=xp.float32)
        self.temperature_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Memory field for simulation state persistence (required by async_dwarf_sim.py)
        self.memory_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Derived fields required by async_dwarf_sim.py
        self.vorticity_magnitude = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Visualization fields
        self.visualization_buffer = xp.zeros((res, res, res), dtype=xp.float32)
        
        # If density is variable, initialize as constant field
        if isinstance(self.density, (int, float)):
            self.density = xp.ones((res, res, res), dtype=xp.float32) * self.density
    
    #----- Compatibility methods and properties -----#
    
    # Alias for backward compatibility - async_dwarf_sim.py calls this method
    def _init_fields(self):
        """Alias for _initialize_fields to maintain backward compatibility."""
        return self._initialize_fields()
    
    @property
    def energy_density(self):
        """Alias for energy_field for compatibility."""
        return self.energy_field
    
    @energy_density.setter
    def energy_density(self, value):
        self.energy_field = value
    
    @property
    def velocity(self):
        """Alias for velocity_field for compatibility."""
        return self.velocity_field
    
    @velocity.setter
    def velocity(self, value):
        self.velocity_field = value
    
    @property
    def pressure(self):
        """Alias for pressure_field for compatibility."""
        return self.pressure_field
    
    @pressure.setter
    def pressure(self, value):
        self.pressure_field = value
    
    @property
    def vorticity(self):
        """Alias for vorticity_field for compatibility."""
        return self.vorticity_field
    
    @vorticity.setter
    def vorticity(self, value):
        self.vorticity_field = value
    
    # Handle missing attribute access with dynamic fallbacks
    def __getattr__(self, name):
        """
        Dynamic attribute fallback for compatibility.
        
        This method is called when an attribute lookup fails.
        It implements various fallback strategies to maintain compatibility.
        """
        # Common field name mappings
        field_mappings = {
            # Core fields
            'velocity_grid': 'velocity_field',
            'pressure_grid': 'pressure_field',
            'vorticity_grid': 'vorticity_field',
            'energy_grid': 'energy_field',
            
            # Derived fields
            'kinetic_energy': 'energy_field',
            'potential_energy': 'energy_field',
            'temperature': 'temperature_field',
            
            # Other possible variations
            'density_field': 'density',
            'memory_grid': 'memory_field',
            'vis_buffer': 'visualization_buffer',
        }
        
        # Check for mapped fields
        if name in field_mappings and hasattr(self, field_mappings[name]):
            return getattr(self, field_mappings[name])
        
        # For fields that might be missing but can be derived or defaulted
        if name.endswith('_field') or name.endswith('_grid'):
            base_name = name.rsplit('_', 1)[0]
            
            # Try common variations
            for suffix in ['_field', '_grid', '']:
                if hasattr(self, base_name + suffix):
                    return getattr(self, base_name + suffix)
        
        # If we can't find a good fallback, raise the normal AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _update_derived_fields(self):
        """Update derived fields like vorticity_magnitude from base fields."""
        # Calculate vorticity magnitude from vorticity field
        if hasattr(self, 'vorticity_field'):
            # |ω| = sqrt(ωx² + ωy² + ωz²)
            self.vorticity_magnitude = self.xp.sqrt(
                self.vorticity_field[0]**2 + 
                self.vorticity_field[1]**2 + 
                self.vorticity_field[2]**2
            )
    
    def initialize(self):
        """
        Initialize or reinitialize the fluid grid.
        Called by async_main.py to reset simulation state.
        """
        # Reset main fields
        res = self.base_resolution
        xp = self.xp
        
        # Re-initialize main fields
        self.velocity_field = xp.zeros((3, res, res, res), dtype=xp.float32)
        self.pressure_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Reset optional fields
        self.vorticity_field = xp.zeros((3, res, res, res), dtype=xp.float32)
        self.energy_field = xp.zeros((res, res, res), dtype=xp.float32)
        self.temperature_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Reset memory field
        self.memory_field = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Reset derived fields
        self.vorticity_magnitude = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Reset visualization buffer
        self.visualization_buffer = xp.zeros((res, res, res), dtype=xp.float32)
        
        # Reset adaptive grid structures
        self.subgrids = {}
        self.refined_cells = set()
        
        # Initialize density field if needed
        if isinstance(self.density, (int, float)):
            self.density = xp.ones((res, res, res), dtype=xp.float32) * self.density
        
        # Pre-compute any constants if needed
        if hasattr(self, 'pre_compute_constants') and callable(self.pre_compute_constants):
            self.pre_compute_constants()
        
        return True
    
    def update(self, dt):
        """
        Update the fluid state by advancing one timestep.
        
        Args:
            dt: The timestep size (float)
        """
        if self.debug_timing:
            import time
            start_time = time.time()
            
        # Main update step
        self._update_fluid_dynamics(dt)
        
        # Update subgrids if using adaptive refinement
        if self.max_depth > 0:
            self._update_subgrids(dt)
        
        # Project velocity to enforce divergence-free condition
        self._project_velocity()
        
        # Update derived fields like vorticity_magnitude
        self._update_derived_fields()
        
        if self.debug_timing:
            elapsed = time.time() - start_time
            print(f"Total grid update time: {elapsed:.6f} seconds")
        
        return True
    
    def _update_subgrids(self, dt):
        """
        Update all refined subgrids.
        Placeholder implementation - adapt based on your original code.
        """
        # This would need to be implemented based on your refinement strategy
        # For now, it's a placeholder to maintain compatibility
        pass
    
    def _apply_pressure_forces(self, dt):
        """
        Apply forces based on pressure gradients using GPU/NumPy vector operations.
        This replaces triple-nested loops with vectorized gradient computation.
        """
        # Get references to necessary fields
        pressure = self.pressure_field         # shape (res, res, res)
        rho = self.density                     # scalar or (res, res, res)
        cell_size = self.cell_size             # grid spacing
        vfield = self.velocity_field           # shape (3, res, res, res)

        # Compute pressure gradients across entire field
        try:
            if self.use_gpu:
                # CuPy gradient on GPU - computes dP/dx, dP/dy, dP/dz in one operation
                dPdx, dPdy, dPdz = cp.gradient(pressure, cell_size, edge_order=2)
            else:
                # NumPy gradient on CPU
                dPdx, dPdy, dPdz = np.gradient(pressure, cell_size, edge_order=2)
                
        except Exception as e:
            # Fall back to NumPy if operation fails
            import numpy as np
            if self.debug_timing:
                print(f"Falling back to NumPy for pressure gradient: {e}")
                
            p_cpu = cp.asnumpy(pressure) if isinstance(pressure, cp.ndarray) else pressure
            dPdx_cpu, dPdy_cpu, dPdz_cpu = np.gradient(p_cpu, cell_size, edge_order=2)
            
            # Convert back to CuPy if needed
            if self.use_gpu:
                dPdx = cp.asarray(dPdx_cpu)
                dPdy = cp.asarray(dPdy_cpu)
                dPdz = cp.asarray(dPdz_cpu)
            else:
                dPdx, dPdy, dPdz = dPdx_cpu, dPdy_cpu, dPdz_cpu

        # Compute pressure force multiplier: -dt/rho
        inv_rho = -dt * (1.0 / rho)

        # Update velocity field in single vectorized operations
        vfield[0] += inv_rho * dPdx  # vx += -dt/rho * dP/dx
        vfield[1] += inv_rho * dPdy  # vy += -dt/rho * dP/dy
        vfield[2] += inv_rho * dPdz  # vz += -dt/rho * dP/dz

        # Handle periodic boundary conditions if enabled
        if self.periodic:
            self._apply_periodic_boundary()

    def _update_vorticity_field(self, dt):
        """
        Compute vorticity field (curl of velocity) and apply vorticity confinement.
        Uses vectorized operations instead of triple-nested loops.
        """
        # Get references
        vfield = self.velocity_field   # shape (3, res, res, res)
        cell_size = self.cell_size
        xp = self.xp
        
        # Get velocity components
        vx, vy, vz = vfield[0], vfield[1], vfield[2]
        
        try:
            # Compute gradients of velocity components
            dvx_dx, dvx_dy, dvx_dz = xp.gradient(vx, cell_size, edge_order=2)
            dvy_dx, dvy_dy, dvy_dz = xp.gradient(vy, cell_size, edge_order=2)
            dvz_dx, dvz_dy, dvz_dz = xp.gradient(vz, cell_size, edge_order=2)
            
            # Compute curl components: ω = ∇ × v
            omega_x = dvy_dz - dvz_dy  # ωx = ∂vy/∂z - ∂vz/∂y
            omega_y = dvz_dx - dvx_dz  # ωy = ∂vz/∂x - ∂vx/∂z
            omega_z = dvx_dy - dvy_dx  # ωz = ∂vx/∂y - ∂vy/∂x
            
            # Store vorticity field
            self.vorticity_field[0] = omega_x
            self.vorticity_field[1] = omega_y
            self.vorticity_field[2] = omega_z
            
            # Update vorticity magnitude for async_dwarf_sim.py compatibility
            self.vorticity_magnitude = xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            # === Optional: Apply vorticity confinement (if enabled) ===
            if hasattr(self, 'vorticity_confinement') and self.vorticity_confinement > 0:
                # Compute gradient of vorticity magnitude
                dvort_dx, dvort_dy, dvort_dz = xp.gradient(self.vorticity_magnitude, cell_size, edge_order=2)
                
                # Normalize the gradient
                epsilon = 1e-10  # Small constant to avoid division by zero
                norm = xp.sqrt(dvort_dx**2 + dvort_dy**2 + dvort_dz**2) + epsilon
                
                N_x = dvort_dx / norm
                N_y = dvort_dy / norm
                N_z = dvort_dz / norm
                
                # Compute vorticity confinement force: f = ε(N × ω)
                # Cross product of N and ω
                force_x = self.vorticity_confinement * (N_y * omega_z - N_z * omega_y)
                force_y = self.vorticity_confinement * (N_z * omega_x - N_x * omega_z)
                force_z = self.vorticity_confinement * (N_x * omega_y - N_y * omega_x)
                
                # Apply forces to velocity field
                vfield[0] += dt * force_x
                vfield[1] += dt * force_y
                vfield[2] += dt * force_z
                
        except Exception as e:
            if self.debug_timing:
                print(f"Vorticity calculation error: {e}")
            # If there's an error, leave vorticity field unchanged

    def _update_energy_density(self, dt):
        """
        Update the energy density field based on velocity and other properties.
        Uses vectorized operations instead of triple-nested loops.
        """
        # Get references
        vfield = self.velocity_field  # shape (3, res, res, res)
        xp = self.xp
        
        try:
            # Compute kinetic energy density: KE = 0.5 * ρ * |v|²
            # |v|² = vx² + vy² + vz²
            v_squared = vfield[0]**2 + vfield[1]**2 + vfield[2]**2
            
            # If density is a field, use it, otherwise use constant density
            if isinstance(self.density, (xp.ndarray, np.ndarray)):
                # If density is an array
                kinetic_energy = 0.5 * self.density * v_squared
            else:
                # Default density as scalar
                kinetic_energy = 0.5 * self.density * v_squared
            
            # Update energy field
            self.energy_field = kinetic_energy
            
            # Optional: Add potential energy if gravity is present
            if hasattr(self, 'gravity') and self.gravity is not None:
                # Simple gravitational potential energy
                # Assuming y-axis is up direction
                y_coords = xp.linspace(0, self.domain_size[1], self.base_resolution)
                y_grid = xp.tile(y_coords.reshape(1, -1, 1), (self.base_resolution, 1, self.base_resolution))
                
                # PE = m * g * h
                if isinstance(self.density, (xp.ndarray, np.ndarray)):
                    mass_field = self.density * (self.cell_size**3)  # mass per cell
                    potential_energy = mass_field * self.gravity * y_grid
                else:
                    # Constant density
                    potential_energy = self.density * (self.cell_size**3) * self.gravity * y_grid
                    
                self.energy_field += potential_energy
                
            # Update memory field based on energy field for async_dwarf_sim.py compatibility
            # This is a reasonable assumption - memory can track energy distribution
            self.memory_field = 0.1 * self.memory_field + 0.9 * self.energy_field
                
        except Exception as e:
            if self.debug_timing:
                print(f"Energy density calculation error: {e}")
            # If there's an error, leave energy field unchanged

    def _update_fluid_dynamics(self, dt):
        """
        Main fluid dynamics update method that calls all vectorized components.
        Controls the simulation pipeline and applies all physical forces.
        """
        # Record timing for performance analysis
        if self.debug_timing:
            import time
            timing = {}
            total_start = time.time()
        
        # 1. Apply pressure forces (this updates velocity based on pressure gradient)
        if self.debug_timing:
            start = time.time()
            
        self._apply_pressure_forces(dt)
        
        if self.debug_timing:
            timing['pressure_forces'] = time.time() - start
        
        # 2. Apply viscosity (if enabled)
        if hasattr(self, 'viscosity') and self.viscosity > 0:
            if self.debug_timing:
                start = time.time()
                
            self._apply_viscosity(dt)
            
            if self.debug_timing:
                timing['viscosity'] = time.time() - start
        
        # 3. Update vorticity field and apply vorticity confinement
        if self.enable_vorticity:
            if self.debug_timing:
                start = time.time()
                
            self._update_vorticity_field(dt)
            
            if self.debug_timing:
                timing['vorticity'] = time.time() - start
        
        # 4. Apply external forces (gravity, etc.)
        if self.debug_timing:
            start = time.time()
            
        self._apply_external_forces(dt)
        
        if self.debug_timing:
            timing['external_forces'] = time.time() - start
        
        # 5. Update energy field
        if self.enable_energy:
            if self.debug_timing:
                start = time.time()
                
            self._update_energy_density(dt)
            
            if self.debug_timing:
                timing['energy'] = time.time() - start
        
        # 6. Apply boundary conditions to velocity field
        if self.debug_timing:
            start = time.time()
            
        if not self.periodic:
            self._apply_boundary_conditions()
        else:
            self._apply_periodic_boundary()
        
        if self.debug_timing:
            timing['boundary_conditions'] = time.time() - start
            timing['total'] = time.time() - total_start
            print(f"Fluid dynamics update timing (s): {timing}")

    def _apply_external_forces(self, dt):
        """
        Apply external forces like gravity to the velocity field.
        Vectorized implementation.
        """
        # Apply gravity if defined
        if hasattr(self, 'gravity') and self.gravity is not None:
            # Assuming gravity points in negative y-direction
            grav_force = -self.gravity * dt
            
            # Apply to entire y-component of velocity field
            self.velocity_field[1] += grav_force
        
        # Apply any other external forces if defined
        if hasattr(self, 'external_forces') and self.external_forces is not None:
            # Assuming external_forces is a tuple/list of (fx, fy, fz) or an array of shape (3, res, res, res)
            if isinstance(self.external_forces, (list, tuple)):
                self.velocity_field[0] += self.external_forces[0] * dt
                self.velocity_field[1] += self.external_forces[1] * dt
                self.velocity_field[2] += self.external_forces[2] * dt
            else:
                self.velocity_field += self.external_forces * dt

    def _apply_boundary_conditions(self):
        """
        Apply boundary conditions to velocity field.
        Vectorized implementation for no-slip boundaries.
        """
        # Example: No-slip boundary conditions (velocity = 0 at boundaries)
        res = self.base_resolution
        
        # X-boundaries (left and right walls)
        self.velocity_field[0, 0, :, :] = 0  # vx = 0 at x = 0
        self.velocity_field[0, res-1, :, :] = 0  # vx = 0 at x = res-1
        
        # Y-boundaries (floor and ceiling)
        self.velocity_field[1, :, 0, :] = 0  # vy = 0 at y = 0
        self.velocity_field[1, :, res-1, :] = 0  # vy = 0 at y = res-1
        
        # Z-boundaries (front and back walls)
        self.velocity_field[2, :, :, 0] = 0  # vz = 0 at z = 0
        self.velocity_field[2, :, :, res-1] = 0  # vz = 0 at z = res-1

    def _apply_periodic_boundary(self):
        """
        Apply periodic boundary conditions to the velocity field.
        """
        # For a periodic domain, we need to ensure the values wrap around properly
        # This is a simplified implementation - you might need to adjust based on 
        # how your original code handles periodic boundaries
        res = self.base_resolution
        
        # For each component of velocity
        for i in range(3):
            # Copy the values from the opposite sides to ensure continuity
            # Left/right boundaries
            self.velocity_field[i, 0, :, :] = self.velocity_field[i, res-2, :, :]
            self.velocity_field[i, res-1, :, :] = self.velocity_field[i, 1, :, :]
            
            # Top/bottom boundaries
            self.velocity_field[i, :, 0, :] = self.velocity_field[i, :, res-2, :]
            self.velocity_field[i, :, res-1, :] = self.velocity_field[i, :, 1, :]
            
            # Front/back boundaries
            self.velocity_field[i, :, :, 0] = self.velocity_field[i, :, :, res-2]
            self.velocity_field[i, :, :, res-1] = self.velocity_field[i, :, :, 1]

    def _apply_viscosity(self, dt):
        """
        Apply viscous diffusion to velocity field.
        Vectorized implementation using Laplacian operator.
        """
        # Get references
        vfield = self.velocity_field
        visc = self.viscosity
        dx = self.cell_size
        xp = self.xp
        
        try:
            # For each velocity component, apply the diffusion equation:
            # dv/dt = ν∇²v
            # where ν is kinematic viscosity and ∇²v is the Laplacian of v
            
            # Apply Laplacian to each velocity component
            for i in range(3):  # x, y, z components
                # Get component
                v_comp = vfield[i]
                
                # Pad for handling boundaries
                padded = xp.pad(v_comp, 1, mode='edge')
                
                # Compute stencil terms
                vijk = padded[1:-1, 1:-1, 1:-1]  # center
                vip1 = padded[2:, 1:-1, 1:-1]    # i+1
                vim1 = padded[0:-2, 1:-1, 1:-1]  # i-1
                vjp1 = padded[1:-1, 2:, 1:-1]    # j+1
                vjm1 = padded[1:-1, 0:-2, 1:-1]  # j-1
                vkp1 = padded[1:-1, 1:-1, 2:]    # k+1
                vkm1 = padded[1:-1, 1:-1, 0:-2]  # k-1
                
                # Compute Laplacian
                laplacian = (vip1 + vim1 + vjp1 + vjm1 + vkp1 + vkm1 - 6*vijk) / (dx*dx)
                
                # Update velocity: v += ν∇²v*dt
                vfield[i] += visc * dt * laplacian
                
        except Exception as e:
            if self.debug_timing:
                print(f"Viscosity application failed: {e}")
            
            # Simple alternative diffusion
            try:
                for i in range(3):
                    if self.use_gpu:
                        # GPU implementation
                        from cupyx.scipy import ndimage
                        vfield[i] = ndimage.gaussian_filter(vfield[i], sigma=visc*dt/dx)
                    else:
                        # CPU implementation
                        from scipy import ndimage
                        vfield[i] = ndimage.gaussian_filter(vfield[i], sigma=visc*dt/dx)
            except Exception:
                # If even that fails, just skip viscosity for this step
                pass
    
    def _project_velocity(self):
        """
        Project velocity field to make it divergence-free.
        Uses the Helmholtz decomposition and solves a Poisson equation.
        """
        # Get references
        vfield = self.velocity_field
        dx = self.cell_size
        res = self.base_resolution
        xp = self.xp
        
        try:
            # 1. Compute divergence of velocity field
            dvx_dx, _, _ = xp.gradient(vfield[0], dx, edge_order=2)
            _, dvy_dy, _ = xp.gradient(vfield[1], dx, edge_order=2)
            _, _, dvz_dz = xp.gradient(vfield[2], dx, edge_order=2)
            
            divergence = dvx_dx + dvy_dy + dvz_dz
            
            # 2. Solve Poisson equation: ∇²p = div(v) using Jacobi iteration
            # Initialize pressure field (or reuse existing)
            p = xp.zeros_like(divergence)
            
            # Simple Jacobi iteration - in production code, use a more efficient solver
            iterations = min(20, res)  # Adjust based on resolution
            for _ in range(iterations):
                # Pad for stencil computation
                padded = xp.pad(p, 1, mode='edge')
                
                # Apply Laplacian stencil
                pijk = padded[1:-1, 1:-1, 1:-1]  # center
                pip1 = padded[2:, 1:-1, 1:-1]    # i+1
                pim1 = padded[0:-2, 1:-1, 1:-1]  # i-1
                pjp1 = padded[1:-1, 2:, 1:-1]    # j+1
                pjm1 = padded[1:-1, 0:-2, 1:-1]  # j-1
                pkp1 = padded[1:-1, 1:-1, 2:]    # k+1
                pkm1 = padded[1:-1, 1:-1, 0:-2]  # k-1
                
                # Update pressure (Jacobi iteration)
                p = (pip1 + pim1 + pjp1 + pjm1 + pkp1 + pkm1 - dx*dx*divergence) / 6.0
            
            # 3. Compute gradient of pressure
            dp_dx, dp_dy, dp_dz = xp.gradient(p, dx, edge_order=2)
            
            # 4. Subtract pressure gradient from velocity
            vfield[0] -= dp_dx
            vfield[1] -= dp_dy
            vfield[2] -= dp_dz
            
            # Store pressure field for visualization or other uses
            self.pressure_field = p
            
        except Exception as e:
            if self.debug_timing:
                print(f"Velocity projection failed: {e}")
    
    # Additional helper methods and utilities
    
    def add_source(self, position, strength, radius):
        """
        Add a source (or sink) of fluid at the specified position.
        
        Args:
            position: (x,y,z) position of source
            strength: Positive for source, negative for sink
            radius: Radius of influence
        """
        res = self.base_resolution
        xp = self.xp
        
        # Convert position to grid coordinates
        x, y, z = [int(p / self.domain_size[i] * res) for i, p in enumerate(position)]
        
        # Ensure coordinates are within grid
        x = max(0, min(res-1, x))
        y = max(0, min(res-1, y))
        z = max(0, min(res-1, z))
        
        # Create distance field from source point
        X, Y, Z = xp.meshgrid(
            xp.arange(res),
            xp.arange(res),
            xp.arange(res),
            indexing='ij'
        )
        
        # Compute squared distances
        dist_sq = (X - x)**2 + (Y - y)**2 + (Z - z)**2
        
        # Normalize radius to grid units
        radius_grid = radius / self.cell_size
        
        # Create falloff field based on distance
        falloff = xp.exp(-dist_sq / (2 * radius_grid**2))
        
        # Apply source to pressure field
        self.pressure_field += strength * falloff
    
    def add_velocity(self, position, direction, strength, radius):
        """
        Add velocity at the specified position.
        
        Args:
            position: (x,y,z) position
            direction: (dx,dy,dz) direction vector (will be normalized)
            strength: Strength of the velocity addition
            radius: Radius of influence
        """
        res = self.base_resolution
        xp = self.xp
        
        # Convert position to grid coordinates
        x, y, z = [int(p / self.domain_size[i] * res) for i, p in enumerate(position)]
        
        # Ensure coordinates are within grid
        x = max(0, min(res-1, x))
        y = max(0, min(res-1, y))
        z = max(0, min(res-1, z))
        
        # Normalize direction
        dx, dy, dz = direction
        magnitude = xp.sqrt(dx**2 + dy**2 + dz**2)
        if magnitude > 0:
            dx, dy, dz = dx/magnitude, dy/magnitude, dz/magnitude
        
        # Create distance field from source point
        X, Y, Z = xp.meshgrid(
            xp.arange(res),
            xp.arange(res),
            xp.arange(res),
            indexing='ij'
        )
        
        # Compute squared distances
        dist_sq = (X - x)**2 + (Y - y)**2 + (Z - z)**2
        
        # Normalize radius to grid units
        radius_grid = radius / self.cell_size
        
        # Create falloff field based on distance
        falloff = xp.exp(-dist_sq / (2 * radius_grid**2))
        
        # Apply velocity
        self.velocity_field[0] += strength * dx * falloff
        self.velocity_field[1] += strength * dy * falloff
        self.velocity_field[2] += strength * dz * falloff
    
    def get_velocity_at_position(self, position):
        """
        Get interpolated velocity at any position in the domain.
        
        Args:
            position: (x,y,z) position in world space
            
        Returns:
            (vx,vy,vz) interpolated velocity at position
        """
        # Convert position to normalized grid coordinates [0,1]
        normalized_pos = [
            p / self.domain_size[i] for i, p in enumerate(position)
        ]
        
        # Clamp to domain boundaries
        normalized_pos = [max(0.0, min(0.999, p)) for p in normalized_pos]
        
        # Convert to grid indices and fractional offsets
        res = self.base_resolution
        indices = [int(p * res) for p in normalized_pos]
        fractions = [(p * res) - i for p, i in zip(normalized_pos, indices)]
        
        # Get indices
        i, j, k = indices
        fi, fj, fk = fractions
        
        # Perform trilinear interpolation
        c000 = self.velocity_field[:, i, j, k]
        c001 = self.velocity_field[:, i, j, k+1] if k+1 < res else c000
        c010 = self.velocity_field[:, i, j+1, k] if j+1 < res else c000
        c011 = self.velocity_field[:, i, j+1, k+1] if j+1 < res and k+1 < res else c000
        c100 = self.velocity_field[:, i+1, j, k] if i+1 < res else c000
        c101 = self.velocity_field[:, i+1, j, k+1] if i+1 < res and k+1 < res else c000
        c110 = self.velocity_field[:, i+1, j+1, k] if i+1 < res and j+1 < res else c000
        c111 = self.velocity_field[:, i+1, j+1, k+1] if i+1 < res and j+1 < res and k+1 < res else c000
        
        # Linear interpolation in x direction
        c00 = c000 * (1 - fi) + c100 * fi
        c01 = c001 * (1 - fi) + c101 * fi
        c10 = c010 * (1 - fi) + c110 * fi
        c11 = c011 * (1 - fi) + c111 * fi
        
        # Linear interpolation in y direction
        c0 = c00 * (1 - fj) + c10 * fj
        c1 = c01 * (1 - fj) + c11 * fj
        
        # Linear interpolation in z direction
        result = c0 * (1 - fk) + c1 * fk
        
        # Convert to Python tuple for compatibility
        return tuple(float(v) if isinstance(v, (np.ndarray, cp.ndarray)) else float(v) for v in result)
    
    # Methods for adaptive refinement - placeholders that would need to be implemented
    
    def refine_cell(self, i, j, k):
        """Placeholder for cell refinement logic"""
        cell_key = (i, j, k)
        if cell_key not in self.refined_cells and len(self.subgrids) < 100:  # Limit refinement
            self.refined_cells.add(cell_key)
            # Create a refined grid for this cell (details would depend on original implementation)
            # self.subgrids[cell_key] = ...
    
    def coarsen_cell(self, i, j, k):
        """Placeholder for cell coarsening logic"""
        cell_key = (i, j, k)
        if cell_key in self.refined_cells:
            self.refined_cells.remove(cell_key)
            if cell_key in self.subgrids:
                del self.subgrids[cell_key]

# For backward compatibility with existing import statements
adaptive_fluid_grid = AdaptiveFluidGrid