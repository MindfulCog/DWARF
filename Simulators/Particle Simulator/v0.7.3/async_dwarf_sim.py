import asyncio
import time
import logging
from typing import List, Dict, Any, Optional

# Import optimized components but maintain original API
from particle_data import Particle
from dwarf_physics import DwarfPhysics  # This now uses the optimized version
from particle_initializer import ParticleInitializer
from data_logger import DataLogger
from constants import *

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsyncDwarfSim")

class AsyncDwarfSim:
    """
    Complete optimized AsyncDwarfSim that maintains backward compatibility
    while using all the new optimized components under the hood.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Enhanced default configuration with optimizations
        self.config = {
            "num_particles": 5000,  # Increased default due to better performance
            "grid_size": (10.0, 10.0, 10.0),
            "grid_resolution": (64, 64, 64),  # Higher resolution possible now
            "time_step": 0.005,  # Smaller timestep for stability
            "max_steps": 2000,
            "log_interval": 50,
            "seed": 42,
            "initialization_mode": "cluster",
            "use_gpu": True,
            "performance_monitoring": True,
            "adaptive_timestep": True,
            "max_timestep": 0.01,
            "min_timestep": 0.001,
            "target_fps": 60
        }
        
        # Override with user config
        if config:
            self.config.update(config)
        
        # Simulation state
        self.particles: List[Particle] = []
        self.physics: Optional[DwarfPhysics] = None
        self.data_logger: Optional[DataLogger] = None
        self.running = False
        self.paused = False
        self.current_step = 0
        
        # Performance tracking
        self.performance_history = []
        self.last_step_time = 0.0
        self.total_sim_time = 0.0
        self.stats = {}
        self.fps_counter = 0
        self.fps_timer = time.time()
        
        # Adaptive timestep variables
        self.current_timestep = self.config["time_step"]
        self.stability_metric = 0.0
        
        logger.info(f"Optimized simulation initialized (backward compatible)")
    
    def initialize(self):
        """Initialize the simulation components (backward compatible)"""
        logger.info("Initializing optimized simulation...")
        start_time = time.time()
        
        # Initialize particles with vectorized operations
        initializer = ParticleInitializer(seed=self.config["seed"])
        self.particles = initializer.create_particles(
            self.config["num_particles"], 
            self.config["grid_size"],
            mode=self.config["initialization_mode"]
        )
        
        # Initialize optimized physics engine (uses new optimized version)
        self.physics = DwarfPhysics(
            self.particles,
            grid_size=self.config["grid_size"],
            grid_resolution=self.config["grid_resolution"],
            use_gpu=self.config["use_gpu"]
        )
        
        # Initialize the grid with interesting structure
        self.physics.grid.initialize_vectorized()
        
        # Initialize data logger
        self.data_logger = DataLogger(
            log_dir="logs_optimized",
            enable_performance_tracking=self.config["performance_monitoring"]
        )
        
        init_time = time.time() - start_time
        logger.info(f"Simulation initialized with {len(self.particles)} particles in {init_time:.3f} seconds")
        
        return init_time
    
    async def run(self):
        """Run the optimized simulation with adaptive performance"""
        if not self.particles:
            self.initialize()
        
        logger.info("Starting optimized simulation...")
        self.running = True
        self.paused = False
        self.current_step = 0
        self.total_sim_time = 0.0
        self.fps_timer = time.time()
        
        try:
            while self.running and self.current_step < self.config["max_steps"]:
                if not self.paused:
                    # Execute simulation step
                    step_stats = await self.step()
                    
                    # Adaptive timestep adjustment
                    if self.config.get("adaptive_timestep", False):
                        self._adjust_timestep(step_stats)
                    
                    # Performance monitoring
                    if self.config.get("performance_monitoring", False):
                        self._update_performance_metrics(step_stats)
                    
                    # Logging at specified intervals
                    if self.current_step % self.config["log_interval"] == 0:
                        await self._log_step_data()
                        self._log_performance_info()
                    
                    # FPS calculation
                    self._update_fps_counter()
                
                # Yield control to other async tasks
                await asyncio.sleep(0)
            
            # Final logging
            await self._finalize_simulation()
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
    
    async def step(self) -> Dict:
        """Execute one optimized simulation step"""
        step_start_time = time.time()
        
        # Physics step with current adaptive timestep
        physics_time = self.physics.step(self.current_timestep)
        
        # Calculate stability metrics for adaptive timestep
        self.stability_metric = self._calculate_stability_metric()
        
        # Update step counter and timing
        self.current_step += 1
        wall_time = time.time() - step_start_time
        self.last_step_time = wall_time
        self.total_sim_time += wall_time
        
        # Collect comprehensive stats
        step_stats = {
            "step": self.current_step,
            "wall_time": wall_time,
            "physics_time": physics_time,
            "timestep": self.current_timestep,
            "stability_metric": self.stability_metric,
            "physics_breakdown": self.physics.get_stats(),
            "simulation_time": self.current_step * self.current_timestep,
            "efficiency": physics_time / wall_time if wall_time > 0 else 0
        }
        
        self.stats = step_stats
        return step_stats
    
    def _adjust_timestep(self, step_stats: Dict):
        """Adaptive timestep adjustment based on stability and performance"""
        target_step_time = 1.0 / self.config.get("target_fps", 60)
        actual_step_time = step_stats["wall_time"]
        
        # Performance-based adjustment
        if actual_step_time < target_step_time * 0.5:
            # Running too fast, can increase timestep
            self.current_timestep = min(self.current_timestep * 1.05, self.config.get("max_timestep", 0.01))
        elif actual_step_time > target_step_time * 2.0:
            # Running too slow, decrease timestep
            self.current_timestep = max(self.current_timestep * 0.95, self.config.get("min_timestep", 0.001))
        
        # Stability-based adjustment
        if self.stability_metric > 1.0:  # Unstable
            self.current_timestep = max(self.current_timestep * 0.9, self.config.get("min_timestep", 0.001))
        elif self.stability_metric < 0.1:  # Very stable
            self.current_timestep = min(self.current_timestep * 1.02, self.config.get("max_timestep", 0.01))
    
    def _calculate_stability_metric(self) -> float:
        """Calculate a metric indicating simulation stability"""
        if not self.particles:
            return 0.0
        
        # Simple stability metric based on average particle velocity
        total_velocity = 0.0
        for p in self.particles:
            total_velocity += p.velocity.magnitude()
        
        avg_velocity = total_velocity / len(self.particles)
        
        # Normalize to a 0-2 scale where 1 is "normal"
        stability = min(avg_velocity / 10.0, 2.0)
        return stability
    
    def _update_performance_metrics(self, step_stats: Dict):
        """Update rolling performance metrics"""
        self.performance_history.append(step_stats)
        
        # Keep only last 100 steps for rolling average
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    async def _log_step_data(self):
        """Log step data asynchronously"""
        if self.data_logger:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.data_logger.log_step,
                self.current_step,
                self.particles,
                self.physics.grid,
                self.stats
            )
    
    def _log_performance_info(self):
        """Log performance information"""
        if self.performance_history:
            recent_stats = self.performance_history[-10:]  # Last 10 steps
            avg_wall_time = sum(s["wall_time"] for s in recent_stats) / len(recent_stats)
            avg_physics_time = sum(s["physics_time"] for s in recent_stats) / len(recent_stats)
            avg_efficiency = sum(s["efficiency"] for s in recent_stats) / len(recent_stats)
            
            logger.info(
                f"Step {self.current_step}/{self.config['max_steps']} | "
                f"Wall: {avg_wall_time*1000:.1f}ms | "
                f"Physics: {avg_physics_time*1000:.1f}ms | "
                f"Efficiency: {avg_efficiency:.1%} | "
                f"Timestep: {self.current_timestep:.4f} | "
                f"FPS: {self.fps_counter:.1f}"
            )
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:  # Update every second
            self.fps_counter = self.config["log_interval"] / (current_time - self.fps_timer)
            self.fps_timer = current_time
    
    async def _finalize_simulation(self):
        """Finalize simulation and save results"""
        logger.info(f"Simulation completed in {self.total_sim_time:.2f} seconds")
        
        if self.data_logger:
            final_stats = self.get_stats()
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.data_logger.log_final_stats,
                final_stats,
                self.particles,
                self.physics.grid
            )
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True
        logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation"""
        self.paused = False
        logger.info("Simulation resumed")
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        logger.info("Simulation stopped")
    
    def get_stats(self) -> Dict:
        """Get comprehensive performance and simulation statistics"""
        base_stats = {
            "current_step": self.current_step,
            "max_steps": self.config["max_steps"],
            "last_step_time": self.last_step_time,
            "total_sim_time": self.total_sim_time,
            "current_timestep": self.current_timestep,
            "stability_metric": self.stability_metric,
            "configuration": self.config
        }
        
        if self.performance_history:
            perf_stats = {
                "performance_summary": {
                    "avg_wall_time": sum(s["wall_time"] for s in self.performance_history) / len(self.performance_history),
                    "avg_physics_time": sum(s["physics_time"] for s in self.performance_history) / len(self.performance_history),
                    "avg_efficiency": sum(s["efficiency"] for s in self.performance_history) / len(self.performance_history),
                    "steps_recorded": len(self.performance_history)
                }
            }
            base_stats.update(perf_stats)
        
        if self.physics:
            base_stats.update({"physics_stats": self.physics.get_stats()})
        
        return base_stats
    
    def get_particle_data(self) -> List[Particle]:
        """Get current particle data"""
        return self.particles
    
    def get_grid_data(self):
        """Get current grid data"""
        return self.physics.grid if self.physics else None