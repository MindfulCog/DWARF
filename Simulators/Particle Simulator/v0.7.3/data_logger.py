import os
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from particle_data import Particle
# Import the optimized grid but keep the same interface

from adaptive_fluid_grid import AdaptiveFluidGridOptimized as AdaptiveFluidGrid


from constants import *

logger = logging.getLogger("DataLogger")

class DataLogger:
    """
    Complete optimized data logger that replaces the original data_logger.py
    This maintains backward compatibility while using optimized serialization.
    """
    
    def __init__(self, log_dir="logs", visualize=False, enable_performance_tracking=True,
                 enable_field_logging=True, compression_level=6):
        self.log_dir = Path(log_dir)
        self.visualize = visualize
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_field_logging = enable_field_logging
        self.compression_level = compression_level
        
        # Create session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        
        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "steps").mkdir(exist_ok=True)
        if self.enable_field_logging:
            (self.session_dir / "fields").mkdir(exist_ok=True)
        if self.enable_performance_tracking:
            (self.session_dir / "performance").mkdir(exist_ok=True)
        
        # File paths
        self.stats_file = self.session_dir / "stats.json"
        self.particle_file = self.session_dir / "particles.npz"
        self.grid_file = self.session_dir / "grid.npz"
        self.metadata_file = self.session_dir / "metadata.json"
        self.performance_file = self.session_dir / "performance" / "performance_log.json"
        
        # Performance tracking
        self.performance_data = []
        self.last_log_time = 0.0
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize metadata
        self._initialize_metadata()
        
        logger.info(f"Data logger initialized: {self.session_dir}")
    
    def _initialize_metadata(self):
        """Initialize session metadata"""
        metadata = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "data_logger_version": "2.0_optimized_compatible",
            "features": {
                "performance_tracking": self.enable_performance_tracking,
                "field_logging": self.enable_field_logging,
                "compression": self.compression_level > 0,
                "visualization": self.visualize
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _serialize_particle(self, particle: Particle) -> Dict:
        """Serialize a particle to a dictionary (backward compatibility)"""
        return {
            "id": particle.id,
            "position": [particle.position.x, particle.position.y, particle.position.z],
            "velocity": [particle.velocity.x, particle.velocity.y, particle.velocity.z],
            "force": [particle.force.x, particle.force.y, particle.force.z],
            "mass": particle.mass,
            "temperature": particle.temperature,
            "material_type": getattr(particle, 'material_type', 1)
        }
    
    def _serialize_particles_optimized(self, particles: List[Particle]) -> Dict[str, np.ndarray]:
        """Optimized particle serialization using vectorized operations"""
        n_particles = len(particles)
        
        # Pre-allocate arrays
        data = {
            "ids": np.zeros(n_particles, dtype=np.int32),
            "positions": np.zeros((n_particles, 3), dtype=np.float32),
            "velocities": np.zeros((n_particles, 3), dtype=np.float32),
            "forces": np.zeros((n_particles, 3), dtype=np.float32),
            "spins": np.zeros((n_particles, 3), dtype=np.float32),
            "masses": np.zeros(n_particles, dtype=np.float32),
            "temperatures": np.zeros(n_particles, dtype=np.float32),
            "types": np.zeros(n_particles, dtype='U10')  # String array for particle types
        }
        
        # Vectorized data extraction
        for i, p in enumerate(particles):
            data["ids"][i] = p.id
            data["positions"][i] = [p.position.x, p.position.y, p.position.z]
            data["velocities"][i] = [p.velocity.x, p.velocity.y, p.velocity.z]
            data["forces"][i] = [p.force.x, p.force.y, p.force.z]
            # Handle spin if it exists
            if hasattr(p, 'spin'):
                data["spins"][i] = [p.spin.x, p.spin.y, p.spin.z]
            data["masses"][i] = p.mass
            data["temperatures"][i] = getattr(p, 'temperature', 300.0)
            data["types"][i] = getattr(p, 'particle_type', 'unknown')
        
        return data
    
    def _serialize_grid(self, grid: AdaptiveFluidGrid) -> Dict[str, Any]:
        """Serialize grid data (handles both old and new grid types)"""
        
        # Check if this is the new optimized grid
        if hasattr(grid, 'base_resolution'):
            # New optimized grid
            return {
                "resolution": [grid.base_resolution] * 3,
                "size": [grid.size] * 3,
                "density": getattr(grid, 'density_field', np.zeros((grid.base_resolution,) * 3)),
                "pressure": getattr(grid, 'pressure_field', np.zeros((grid.base_resolution,) * 3)),
                "velocity_x": getattr(grid, 'velocity_field', np.zeros((grid.base_resolution,) * 3 + (3,)))[:,:,:,0],
                "velocity_y": getattr(grid, 'velocity_field', np.zeros((grid.base_resolution,) * 3 + (3,)))[:,:,:,1],
                "velocity_z": getattr(grid, 'velocity_field', np.zeros((grid.base_resolution,) * 3 + (3,)))[:,:,:,2],
                "temperature": getattr(grid, 'temperature_field', np.ones((grid.base_resolution,) * 3) * 300),
                "particle_count": getattr(grid, 'particle_count', np.zeros((grid.base_resolution,) * 3)),
                "active_mask": getattr(grid, 'state', np.ones((grid.base_resolution,) * 3)) > 0
            }
        
        # Handle original grid format (backward compatibility)
        elif hasattr(grid, 'resolution'):
            nx, ny, nz = int(grid.resolution.x), int(grid.resolution.y), int(grid.resolution.z)
            
            # Extract data from original grid format
            density = np.zeros((nx, ny, nz))
            pressure = np.zeros((nx, ny, nz))
            velocity_x = np.zeros((nx, ny, nz))
            velocity_y = np.zeros((nx, ny, nz))
            velocity_z = np.zeros((nx, ny, nz))
            particle_count = np.zeros((nx, ny, nz))
            active_mask = np.zeros((nx, ny, nz), dtype=bool)
            
            # If grid has array versions, use them
            if hasattr(grid, 'density_array'):
                density = grid.density_array.copy()
            if hasattr(grid, 'pressure_array'):
                pressure = grid.pressure_array.copy()
            if hasattr(grid, 'velocity_x_array'):
                velocity_x = grid.velocity_x_array.copy()
                velocity_y = grid.velocity_y_array.copy()
                velocity_z = grid.velocity_z_array.copy()
            if hasattr(grid, 'particle_count_array'):
                particle_count = grid.particle_count_array.copy()
            if hasattr(grid, 'active_mask'):
                active_mask = grid.active_mask.copy()
            
            return {
                "resolution": [nx, ny, nz],
                "size": [grid.size.x, grid.size.y, grid.size.z],
                "density": density,
                "pressure": pressure,
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "velocity_z": velocity_z,
                "temperature": np.ones((nx, ny, nz)) * 300,  # Default temperature
                "particle_count": particle_count,
                "active_mask": active_mask
            }
        
        else:
            # Fallback for unknown grid types
            return {
                "resolution": [32, 32, 32],
                "size": [10.0, 10.0, 10.0],
                "density": np.zeros((32, 32, 32)),
                "pressure": np.zeros((32, 32, 32)),
                "velocity_x": np.zeros((32, 32, 32)),
                "velocity_y": np.zeros((32, 32, 32)),
                "velocity_z": np.zeros((32, 32, 32)),
                "temperature": np.ones((32, 32, 32)) * 300,
                "particle_count": np.zeros((32, 32, 32)),
                "active_mask": np.zeros((32, 32, 32), dtype=bool)
            }
    
    def log_step(self, step: int, particles: List[Particle], 
                 grid: AdaptiveFluidGrid, stats: Dict):
        """Synchronous step logging (backward compatible)"""
        start_time = time.time()
        
        # Create step-specific files
        step_file = self.session_dir / "steps" / f"step_{step:06d}.npz"
        
        try:
            # Serialize data using optimized methods
            particle_data = self._serialize_particles_optimized(particles)
            grid_data = self._serialize_grid(grid)
            
            # Combine all data for efficient storage
            save_data = {
                # Particle data
                **{f"particles_{k}": v for k, v in particle_data.items()},
                # Grid data
                **{f"grid_{k}": v for k, v in grid_data.items()}
            }
            
            # Save with compression
            if self.compression_level > 0:
                np.savez_compressed(step_file, **save_data)
            else:
                np.savez(step_file, **save_data)
            
            # Log performance data
            if self.enable_performance_tracking:
                self._log_performance_data(step, stats, time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Failed to log step {step}: {e}")
        
        self.last_log_time = time.time() - start_time
    
    def _log_performance_data(self, step: int, stats: Dict, log_time: float):
        """Log performance data"""
        perf_entry = {
            "step": step,
            "timestamp": time.time(),
            "log_time": log_time,
            "stats": stats
        }
        
        self.performance_data.append(perf_entry)
        
        # Write performance data every 10 steps to avoid excessive I/O
        if step % 10 == 0:
            self._write_performance_data()
    
    def _write_performance_data(self):
        """Write accumulated performance data to file"""
        if not self.enable_performance_tracking:
            return
            
        try:
            # Ensure performance directory exists
            self.performance_file.parent.mkdir(exist_ok=True)
            
            # Load existing data if file exists
            existing_data = []
            if self.performance_file.exists():
                try:
                    with open(self.performance_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            # Append new data
            all_data = existing_data + self.performance_data
            
            # Write back to file
            with open(self.performance_file, 'w') as f:
                json.dump(all_data, f, indent=2, default=str)
            
            # Clear accumulated data
            self.performance_data.clear()
            
        except Exception as e:
            logger.error(f"Failed to write performance data: {e}")
    
    def log_final_stats(self, stats: Dict, particles: List[Particle], 
                       grid: AdaptiveFluidGrid):
        """Log final simulation statistics and state"""
        
        # Final performance data
        if self.performance_data:
            self._write_performance_data()
        
        # Final state
        final_file = self.session_dir / "final_state.npz"
        particle_data = self._serialize_particles_optimized(particles)
        grid_data = self._serialize_grid(grid)
        
        save_data = {
            **{f"particles_{k}": v for k, v in particle_data.items()},
            **{f"grid_{k}": v for k, v in grid_data.items()}
        }
        
        np.savez_compressed(final_file, **save_data)
        
        # Final statistics
        stats_file = self.session_dir / "final_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Create summary files for backward compatibility
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Summary report
        self._generate_summary_report(stats)
        
        logger.info(f"Final simulation data logged to {self.session_dir}")
    
    def _generate_summary_report(self, stats: Dict):
        """Generate a human-readable summary report"""
        report_file = self.session_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DWARF SIMULATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Basic simulation info
            f.write("SIMULATION PARAMETERS:\n")
            f.write("-" * 25 + "\n")
            config = stats.get("configuration", stats.get("config", {}))
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Performance summary
            if "performance_summary" in stats:
                perf = stats["performance_summary"]
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Steps completed: {stats.get('current_step', 'Unknown')}\n")
                f.write(f"Average step time: {perf.get('avg_wall_time', 0)*1000:.2f} ms\n")
                f.write(f"Average physics time: {perf.get('avg_physics_time', 0)*1000:.2f} ms\n")
                f.write(f"Average efficiency: {perf.get('avg_efficiency', 0):.1%}\n")
                f.write(f"Effective FPS: {1.0/perf.get('avg_wall_time', 1):.1f}\n")
    
    def _generate_visualization(self, step: int, particles: List[Particle], 
                              grid: AdaptiveFluidGrid, step_dir: str):
        """Generate visualization data for the current step (backward compatibility)"""
        if not self.visualize:
            return
        # Placeholder for visualization generation
        # This would implement visualization data generation
        pass
    
    def get_session_info(self) -> Dict:
        """Get information about the current logging session"""
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "features": {
                "performance_tracking": self.enable_performance_tracking,
                "field_logging": self.enable_field_logging,
                "compression": self.compression_level > 0,
                "visualization": self.visualize
            },
            "files": {
                "metadata": str(self.metadata_file),
                "performance": str(self.performance_file),
                "stats": str(self.stats_file)
            }
        }
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)