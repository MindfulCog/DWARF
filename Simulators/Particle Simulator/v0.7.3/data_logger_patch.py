def patch_data_logger(logger):
    """Apply patches to fix data logging issues"""
    original_log_state = logger.log_state
    
    def log_state_with_debug(time, particle_system, grid):
        try:
            print(f"Logging state at time {time:.3f}...")
            original_log_state(time, particle_system, grid)
            print(f"State logged successfully. Particles: {len(particle_system.particles)}")
        except Exception as e:
            print(f"Error logging state: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Replace log_state method
    logger.log_state = log_state_with_debug
    return logger