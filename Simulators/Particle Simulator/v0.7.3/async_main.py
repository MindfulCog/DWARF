import asyncio
import time
import logging
import argparse
import json
import os
from pathlib import Path

from async_dwarf_sim import AsyncDwarfSim  # Uses optimized version
from constants import *

# Setup enhanced logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging"""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

logger = logging.getLogger("AsyncMain")

async def run_simulation(config: dict) -> dict:
    """Run the optimized simulation with comprehensive monitoring"""
    logger.info(f"Starting DWARF simulation with optimized backend")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Performance tracking
    total_start_time = time.time()
    
    # Create and initialize simulation
    sim = AsyncDwarfSim(config)
    sim.initialize()  # Synchronous initialization for backward compatibility
    
    # Run simulation
    try:
        await sim.run()
        
        # Get final statistics
        final_stats = sim.get_stats()
        
        # Calculate total runtime
        total_runtime = time.time() - total_start_time
        final_stats["total_runtime"] = total_runtime
        
        # Performance summary
        if "performance_summary" in final_stats:
            perf = final_stats["performance_summary"]
            logger.info("\n" + "="*60)
            logger.info("SIMULATION PERFORMANCE SUMMARY")
            logger.info("="*60)
            logger.info(f"Total Runtime: {total_runtime:.2f} seconds")
            logger.info(f"Simulation Steps: {final_stats['current_step']}")
            logger.info(f"Average Step Time: {perf['avg_wall_time']*1000:.2f} ms")
            logger.info(f"Average Physics Time: {perf['avg_physics_time']*1000:.2f} ms")
            logger.info(f"Average Efficiency: {perf['avg_efficiency']:.1%}")
            logger.info(f"Effective FPS: {1.0/perf['avg_wall_time']:.1f}")
            logger.info(f"GPU Acceleration: {'Enabled' if config.get('use_gpu', False) else 'Disabled'}")
            logger.info("="*60)
        
        return final_stats
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sim.stop()
        return sim.get_stats()
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

def load_config(config_path: str) -> dict:
    """Load configuration from file with validation"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return {}

async def main():
    """Enhanced main entry point with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="DWARF Particle Simulator (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config.json",
                        help="Configuration file path")
    parser.add_argument("--particles", type=int, default=5000,
                        help="Number of particles to simulate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of simulation steps")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Grid resolution (cubic)")
    parser.add_argument("--timestep", type=float, default=0.005,
                        help="Simulation timestep")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    
    # Initialization modes
    parser.add_argument("--init-mode", choices=["random", "structured", "cluster"],
                        default="cluster", help="Particle initialization mode")
    
    # Performance options
    parser.add_argument("--target-fps", type=int, default=60,
                        help="Target simulation FPS for adaptive timestep")
    parser.add_argument("--adaptive-timestep", action="store_true", default=True,
                        help="Enable adaptive timestep")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str,
                        help="Log file path")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Logging interval (steps)")
    
    # Output options
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for simulation logs")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load base configuration
    config = load_config(args.config)
    
    # Override with command-line arguments
    cli_config = {
        "num_particles": args.particles,
        "max_steps": args.steps,
        "grid_resolution": (args.resolution, args.resolution, args.resolution),
        "time_step": args.timestep,
        "initialization_mode": args.init_mode,
        "target_fps": args.target_fps,
        "adaptive_timestep": args.adaptive_timestep,
        "log_interval": args.log_interval,
        "log_dir": args.log_dir
    }
    
    # GPU configuration
    if args.gpu:
        cli_config["use_gpu"] = True
    elif args.no_gpu:
        cli_config["use_gpu"] = False
    
    # Update config with CLI overrides
    config.update(cli_config)
    
    # Run simulation
    results = await run_simulation(config)
    
    # Save simulation results
    results_file = f"{args.log_dir}/simulation_results.json"
    os.makedirs(args.log_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Simulation results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())