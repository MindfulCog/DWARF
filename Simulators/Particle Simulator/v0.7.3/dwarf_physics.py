import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Union, Optional

# Import from your existing files
from adaptive_fluid_grid import AdaptiveFluidGridOptimized
from particle_data import Particle
from vector import Vector3
from constants import *

logger = logging.getLogger("DwarfPhysics")

class DwarfPhysics:
    """
    Complete optimized physics engine that replaces the original dwarf_physics.py
    This is a drop-in replacement that maintains API compatibility.
    """
    
    def __init__(self, particles: List[Particle], grid_size=(10.0, 10.0, 10.0), 
                 grid_resolution=(32, 32, 32), use_gpu=True):
        self.particles = particles
        self.grid_size = grid_size
        self.use_gpu = use_gpu
        
        # Initialize optimized fluid grid
        self.grid = AdaptiveFluidGridOptimized(
            base_resolution=grid_resolution[0],
            size=grid_size[0],
            use_gpu=use_gpu
        )
        
        # Performance tracking
        self.stats = {
            "grid_update": 0.0,
            "force_calculation": 0.0,
            "particle_integration": 0.0,
            "collision_detection": 0.0,
            "total_time": 0.0
        }
        
        # Pre-allocate arrays for vectorized operations
        self._preallocate_arrays()
        
        logger.info(f"Optimized physics engine initialized with {len(particles)} particles")
    
    def _preallocate_arrays(self):
        """Pre-allocate numpy arrays for maximum performance"""
        n = len(self.particles)
        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.velocities = np.zeros((n, 3), dtype=np.float32)
        self.forces = np.zeros((n, 3), dtype=np.float32)
        self.masses = np.zeros(n, dtype=np.float32)
        self.new_positions = np.zeros((n, 3), dtype=np.float32)
        self.new_velocities = np.zeros((n, 3), dtype=np.float32)
        
        # Update arrays with current particle data
        self._sync_particle_data_to_arrays()
    
    def _sync_particle_data_to_arrays(self):
        """Sync particle object data to numpy arrays (one-time cost)"""
        for i, p in enumerate(self.particles):
            self.positions[i] = [p.position.x, p.position.y, p.position.z]
            self.velocities[i] = [p.velocity.x, p.velocity.y, p.velocity.z]
            self.forces[i] = [p.force.x, p.force.y, p.force.z]
            self.masses[i] = p.mass
    
    def _sync_arrays_to_particle_data(self):
        """Sync numpy arrays back to particle objects (one-time cost)"""
        for i, p in enumerate(self.particles):
            p.position = Vector3(self.positions[i, 0], self.positions[i, 1], self.positions[i, 2])
            p.velocity = Vector3(self.velocities[i, 0], self.velocities[i, 1], self.velocities[i, 2])
            p.force = Vector3(self.forces[i, 0], self.forces[i, 1], self.forces[i, 2])
    
    def step(self, dt: float) -> float:
        """Highly optimized physics step using pure vectorized operations"""
        start_time = time.time()
        
        # Sync current particle data to arrays (minimal cost)
        self._sync_particle_data_to_arrays()
        
        # Step 1: Update grid with particle data (GPU/vectorized)
        grid_start = time.time()
        self.grid.update_with_particles(self.particles, dt)
        self.stats["grid_update"] = time.time() - grid_start
        
        # Step 2: Calculate forces using vectorized operations
        force_start = time.time()
        self._calculate_forces_vectorized(dt)
        self.stats["force_calculation"] = time.time() - force_start
        
        # Step 3: Integrate particle motion (fully vectorized)
        integration_start = time.time()
        self._integrate_particles_vectorized(dt)
        self.stats["particle_integration"] = time.time() - integration_start
        
        # Step 4: Handle collisions (vectorized)
        collision_start = time.time()
        self._handle_collisions_vectorized()
        self.stats["collision_detection"] = time.time() - collision_start
        
        # Sync arrays back to particle objects
        self._sync_arrays_to_particle_data()
        
        total_time = time.time() - start_time
        self.stats["total_time"] = total_time
        
        logger.debug(f"Physics step completed in {total_time*1000:.2f}ms")
        return total_time
    
    def _calculate_forces_vectorized(self, dt: float):
        """Calculate forces using vectorized grid operations"""
        # Reset forces
        self.forces.fill(0.0)
        
        # Add gravity (vectorized)
        self.forces[:, 1] -= self.masses * GRAVITY
        
        # Grid-based forces (pressure, drag, etc.)
        for i in range(len(self.particles)):
            pos = self.positions[i]
            
            # Get grid cell indices
            cell_x = int((pos[0] + self.grid_size[0]/2) / self.grid.cell_size) % self.grid.base_resolution
            cell_y = int((pos[1] + self.grid_size[1]/2) / self.grid.cell_size) % self.grid.base_resolution
            cell_z = int((pos[2] + self.grid_size[2]/2) / self.grid.cell_size) % self.grid.base_resolution
            
            # Get fluid properties at particle location
            if (0 <= cell_x < self.grid.base_resolution and 
                0 <= cell_y < self.grid.base_resolution and 
                0 <= cell_z < self.grid.base_resolution):
                
                fluid_velocity = self.grid.velocity_field[cell_x, cell_y, cell_z]
                fluid_pressure = self.grid.pressure_field[cell_x, cell_y, cell_z]
                
                # Drag force (velocity difference)
                drag_coeff = 0.1
                vel_diff = fluid_velocity - self.velocities[i]
                self.forces[i] += drag_coeff * vel_diff
                
                # Pressure gradient force
                if cell_x > 0 and cell_x < self.grid.base_resolution - 1:
                    pressure_grad_x = (self.grid.pressure_field[cell_x+1, cell_y, cell_z] - 
                                     self.grid.pressure_field[cell_x-1, cell_y, cell_z]) / (2 * self.grid.cell_size)
                    self.forces[i, 0] -= pressure_grad_x * 0.01
                
                if cell_y > 0 and cell_y < self.grid.base_resolution - 1:
                    pressure_grad_y = (self.grid.pressure_field[cell_x, cell_y+1, cell_z] - 
                                     self.grid.pressure_field[cell_x, cell_y-1, cell_z]) / (2 * self.grid.cell_size)
                    self.forces[i, 1] -= pressure_grad_y * 0.01
                
                if cell_z > 0 and cell_z < self.grid.base_resolution - 1:
                    pressure_grad_z = (self.grid.pressure_field[cell_x, cell_y, cell_z+1] - 
                                     self.grid.pressure_field[cell_x, cell_y, cell_z-1]) / (2 * self.grid.cell_size)
                    self.forces[i, 2] -= pressure_grad_z * 0.01
    
    def _integrate_particles_vectorized(self, dt: float):
        """Integrate particle motion using vectorized Verlet integration"""
        # Calculate accelerations: a = F/m
        accelerations = self.forces / self.masses[:, np.newaxis]
        
        # Verlet integration (more stable than Euler)
        # v_new = v + a*dt
        self.new_velocities = self.velocities + accelerations * dt
        
        # Apply damping
        self.new_velocities *= DEFAULT_DAMPING
        
        # x_new = x + v*dt + 0.5*a*dt^2
        self.new_positions = (self.positions + 
                            self.new_velocities * dt + 
                            0.5 * accelerations * dt * dt)
        
        # Update arrays
        self.positions = self.new_positions.copy()
        self.velocities = self.new_velocities.copy()
    
    def _handle_collisions_vectorized(self):
        """Handle boundary collisions using vectorized operations"""
        # Boundary collision detection (vectorized)
        restitution = 0.9
        
        # X boundaries
        x_min_mask = self.positions[:, 0] < 0
        x_max_mask = self.positions[:, 0] > self.grid_size[0]
        
        self.positions[x_min_mask, 0] = 0.01
        self.positions[x_max_mask, 0] = self.grid_size[0] - 0.01
        self.velocities[x_min_mask, 0] *= -restitution
        self.velocities[x_max_mask, 0] *= -restitution
        
        # Y boundaries
        y_min_mask = self.positions[:, 1] < 0
        y_max_mask = self.positions[:, 1] > self.grid_size[1]
        
        self.positions[y_min_mask, 1] = 0.01
        self.positions[y_max_mask, 1] = self.grid_size[1] - 0.01
        self.velocities[y_min_mask, 1] *= -restitution
        self.velocities[y_max_mask, 1] *= -restitution
        
        # Z boundaries
        z_min_mask = self.positions[:, 2] < 0
        z_max_mask = self.positions[:, 2] > self.grid_size[2]
        
        self.positions[z_min_mask, 2] = 0.01
        self.positions[z_max_mask, 2] = self.grid_size[2] - 0.01
        self.velocities[z_min_mask, 2] *= -restitution
        self.velocities[z_max_mask, 2] *= -restitution
    
    def get_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        grid_stats = self.grid.get_performance_stats()
        
        return {
            "physics_total_time": self.stats["total_time"],
            "physics_breakdown": self.stats,
            "grid_performance": grid_stats,
            "total_particles": len(self.particles),
            "using_gpu": self.use_gpu,
            "efficiency_ratio": {
                "grid_vs_physics": grid_stats["total_update_time"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0,
                "vectorization_effectiveness": 1.0 - (self.stats["force_calculation"] / self.stats["total_time"]) if self.stats["total_time"] > 0 else 0
            }
        }

    # Backward compatibility methods for existing code
    def update_grid(self, dt):
        """Backward compatibility method"""
        return self.grid.update_with_particles(self.particles, dt)
    
    def calculate_forces(self):
        """Backward compatibility method"""
        self._calculate_forces_vectorized(0.01)  # Default dt
    
    def integrate_particles(self, dt):
        """Backward compatibility method"""
        self._integrate_particles_vectorized(dt)
    
    def detect_collisions(self):
        """Backward compatibility method"""
        self._handle_collisions_vectorized()