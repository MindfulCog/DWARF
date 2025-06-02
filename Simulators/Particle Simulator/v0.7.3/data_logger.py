import numpy as np
import cupy as cp
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import psutil

class DataLogger:
    """Log and save simulation data"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.particle_data = []
        self.field_data = []
        self.energy_data = []
        self.state_data = []
        self.performance_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create subdirectory for this session
        self.session_dir = os.path.join(log_dir, f"session_{self.session_id}")
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
        
        # Create subdirectory for plots
        self.plot_dir = os.path.join(self.session_dir, "plots")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        # Initialize periodic boundary counting
        self.ghost_particle_counts = []
            
        # Performance tracking
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize logger"""
        # Create metadata file with timestamp
        with open(os.path.join(self.session_dir, "metadata.txt"), "w") as f:
            f.write(f"DWARF Physics Simulation Session\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session ID: {self.session_id}\n")
            
            # Check for CuPy and document GPU info
            try:
                import cupy as cp
                device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                f.write(f"GPU: {device_props['name'].decode()}\n")
                mem_info = cp.cuda.runtime.memGetInfo()
                f.write(f"GPU Memory: {mem_info[0]/1024/1024/1024:.2f} GB free / {mem_info[1]/1024/1024/1024:.2f} GB total\n")
            except:
                f.write("GPU: Not available\n")
                
            # Log CPU info
            f.write(f"CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count()} logical cores\n")
            f.write(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
        
    def log_state(self, time, particle_system, grid):
        """Log current state of simulation"""
        try:
            # Performance metrics
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get GPU usage if available
            gpu_memory_used = 0
            gpu_memory_percent = 0
            try:
                import cupy as cp
                memory_info = cp.cuda.runtime.memGetInfo()
                device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                total_memory = memory_info[1]
                free_memory = memory_info[0]
                gpu_memory_used = (total_memory - free_memory) / (1024 * 1024)  # MB
                gpu_memory_percent = (total_memory - free_memory) * 100.0 / total_memory
            except:
                pass
            
            # Log performance data
            self.performance_data.append({
                'time': time,
                'elapsed_time': elapsed,
                'memory_usage_mb': memory_usage,
                'cpu_percent': cpu_percent,
                'gpu_memory_used_mb': gpu_memory_used,
                'gpu_memory_percent': gpu_memory_percent,
                'grid_update_time': getattr(grid, 'update_time', 0.0),
                'refinement_regions': len(getattr(grid, 'refinement_regions', [])),
                'particle_count': len(particle_system.particles)
            })
            
            # Count ghost particles from periodic boundary visualization
            ghost_count = 0
            if hasattr(particle_system, 'ghost_particle_count'):
                ghost_count = particle_system.ghost_particle_count
                
            # Log ghost particle count for boundary analysis
            self.ghost_particle_counts.append({
                'time': time,
                'ghost_particles': ghost_count,
                'real_particles': len(particle_system.particles)
            })
            
            # Log particle data
            for particle in particle_system.particles:
                angular_momentum = particle.calculate_angular_momentum()
                angular_momentum_mag = np.linalg.norm(angular_momentum)
                
                kinetic_energy = particle.get_kinetic_energy()
                
                self.particle_data.append({
                    'time': time,
                    'id': particle.id,
                    'type': particle.particle_type,
                    'x': particle.position[0],
                    'y': particle.position[1],
                    'z': particle.position[2],
                    'vx': particle.velocity[0],
                    'vy': particle.velocity[1],
                    'vz': particle.velocity[2],
                    'spin_x': particle.spin[0],
                    'spin_y': particle.spin[1],
                    'spin_z': particle.spin[2],
                    'angular_momentum_magnitude': angular_momentum_mag,
                    'kinetic_energy': kinetic_energy,
                    'bonded': len(particle.bonded_with) > 0,
                    'bond_type': particle.bond_type
                })
            
            # Log field data (sampled at lower resolution)
            sample_step = 16  # Sample every 16 cells for storage efficiency
            for i in range(0, grid.base_resolution, sample_step):
                for j in range(0, grid.base_resolution, sample_step):
                    for k in range(0, grid.base_resolution, sample_step):
                        # Make sure indices are within bounds
                        if i < grid.velocity_field.shape[0] and j < grid.velocity_field.shape[1] and k < grid.velocity_field.shape[2]:
                            # Transfer GPU data to CPU for logging
                            if hasattr(grid.velocity_field, 'get'):
                                # If it's a CuPy array
                                field_mag = float(cp.linalg.norm(grid.velocity_field[i, j, k]).get())
                                vorticity_mag = float(grid.vorticity_magnitude[i, j, k].get())
                                state = int(cp.asnumpy(grid.state[i, j, k]))
                                pressure = float(cp.asnumpy(grid.pressure_field[i, j, k]))
                                energy_density = float(cp.asnumpy(grid.energy_density[i, j, k]))
                            else:
                                # If it's a NumPy array
                                field_mag = float(np.linalg.norm(grid.velocity_field[i, j, k]))
                                vorticity_mag = float(grid.vorticity_magnitude[i, j, k])
                                state = int(grid.state[i, j, k])
                                pressure = float(grid.pressure_field[i, j, k])
                                energy_density = float(grid.energy_density[i, j, k])
                            
                            self.field_data.append({
                                'time': time,
                                'i': i,
                                'j': j,
                                'k': k,
                                'field_magnitude': field_mag,
                                'vorticity_magnitude': vorticity_mag,
                                'state': state,
                                'pressure': pressure,
                                'energy_density': energy_density
                            })
            
            # Log energy data
            total_ke = sum(p.get_kinetic_energy() for p in particle_system.particles)
            field_energy = grid.get_total_energy()  # This already returns a CPU value
            
            self.energy_data.append({
                'time': time,
                'kinetic_energy': total_ke,
                'field_energy': field_energy,
                'total_energy': total_ke + field_energy,
                'particle_count': len(particle_system.particles),
                'proton_count': len(particle_system.get_particles_by_type("proton")),
                'electron_count': len(particle_system.get_particles_by_type("electron")),
                'neutron_count': len(particle_system.get_particles_by_type("neutron")),
                'bond_count': len(particle_system.bonds),
                'hydrogen_count': len(particle_system.atom_groups.get("hydrogen", [])),
                'helium_count': len(particle_system.atom_groups.get("helium", []))
            })
            
            # Log fluid state data
            state_counts = grid.get_state_counts()  # This returns a CPU dictionary
            
            self.state_data.append({
                'time': time,
                'uncompressed': state_counts['uncompressed'],
                'compressed': state_counts['compressed'],
                'vacuum': state_counts['vacuum']
            })
            
            # Every 100 log entries, save data to disk
            if len(self.energy_data) % 100 == 0:
                self.save_logs()
                
            # Print a status message
            print(f"Logged state at time {time:.3f}... Particles: {len(particle_system.particles)}")
            
            return True
            
        except Exception as e:
            print(f"Error logging state: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_plots(self):
        """Generate analysis plots"""
        if not self.energy_data:
            return
            
        # Convert to DataFrames for easier plotting
        df_energy = pd.DataFrame(self.energy_data)
        df_states = pd.DataFrame(self.state_data)
        df_performance = pd.DataFrame(self.performance_data)
        df_ghost = pd.DataFrame(self.ghost_particle_counts) if self.ghost_particle_counts else None
        
        # Plot 1: Energy over time
        plt.figure(figsize=(12, 6))
        plt.plot(df_energy['time'], df_energy['kinetic_energy'], label='Kinetic Energy')
        plt.plot(df_energy['time'], df_energy['field_energy'], label='Field Energy')
        plt.plot(df_energy['time'], df_energy['total_energy'], label='Total Energy')
        plt.xlabel('Simulation Time')
        plt.ylabel('Energy')
        plt.title('Energy Distribution Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'energy_over_time.png'))
        plt.close()
        
        # Plot 2: Particle counts over time
        plt.figure(figsize=(12, 6))
        plt.plot(df_energy['time'], df_energy['proton_count'], label='Protons')
        plt.plot(df_energy['time'], df_energy['electron_count'], label='Electrons')
        plt.plot(df_energy['time'], df_energy['neutron_count'], label='Neutrons')
        plt.xlabel('Simulation Time')
        plt.ylabel('Count')
        plt.title('Particle Counts Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'particle_counts.png'))
        plt.close()
        
        # Plot 3: Bond and atom formation over time
        plt.figure(figsize=(12, 6))
        plt.plot(df_energy['time'], df_energy['bond_count'], label='Bonds')
        plt.plot(df_energy['time'], df_energy['hydrogen_count'], label='Hydrogen Atoms')
        plt.plot(df_energy['time'], df_energy['helium_count'], label='Helium Atoms')
        plt.xlabel('Simulation Time')
        plt.ylabel('Count')
        plt.title('Bond and Atom Formation Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'atom_formation.png'))
        plt.close()
        
        # Plot 4: Fluid state distribution over time
        plt.figure(figsize=(12, 6))
        plt.plot(df_states['time'], df_states['uncompressed'], label='Uncompressed')
        plt.plot(df_states['time'], df_states['compressed'], label='Compressed')
        plt.plot(df_states['time'], df_states['vacuum'], label='Vacuum')
        plt.xlabel('Simulation Time')
        plt.ylabel('Cell Count')
        plt.title('Fluid State Distribution Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'fluid_states.png'))
        plt.close()
        
        # Plot 5: Performance metrics
        plt.figure(figsize=(12, 8))
        
        # CPU usage subplot
        plt.subplot(3, 1, 1)
        plt.plot(df_performance['elapsed_time'], df_performance['cpu_percent'], 'b-')
        plt.title('CPU Utilization')
        plt.ylabel('CPU %')
        plt.grid(True)
        
        # Memory usage subplot
        plt.subplot(3, 1, 2)
        plt.plot(df_performance['elapsed_time'], df_performance['memory_usage_mb'], 'g-', label='System RAM')
        if 'gpu_memory_used_mb' in df_performance and df_performance['gpu_memory_used_mb'].max() > 0:
            plt.plot(df_performance['elapsed_time'], df_performance['gpu_memory_used_mb'], 'r-', label='GPU Memory')
            plt.legend()
        plt.title('Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        
        # Grid update time subplot
        plt.subplot(3, 1, 3)
        plt.plot(df_performance['elapsed_time'], df_performance['grid_update_time'] * 1000, 'c-')
        plt.title('Grid Update Time')
        plt.ylabel('Time (ms)')
        plt.xlabel('Elapsed Time (seconds)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'performance_metrics.png'))
        plt.close()
        
        # Plot 6: Ghost particle analysis (periodic boundary visualization)
        if df_ghost is not None and not df_ghost.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(df_ghost['time'], df_ghost['ghost_particles'] / df_ghost['real_particles'], 'b-')
            plt.title('Periodic Boundary Activity')
            plt.ylabel('Ghost / Real Particle Ratio')
            plt.xlabel('Simulation Time')
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_dir, 'boundary_activity.png'))
            plt.close()
    
    def save_logs(self):
        """Save logged data to files"""
        # Save particle data
        if self.particle_data:
            df_particles = pd.DataFrame(self.particle_data)
            df_particles.to_csv(os.path.join(self.session_dir, "particles.csv"), index=False)
        
        # Save energy data
        if self.energy_data:
            df_energy = pd.DataFrame(self.energy_data)
            df_energy.to_csv(os.path.join(self.session_dir, "energy.csv"), index=False)
        
        # Save field data (can be large, so use chunks)
        if self.field_data:
            df_field = pd.DataFrame(self.field_data)
            df_field.to_csv(os.path.join(self.session_dir, "field_data.csv"), index=False)
        
        # Save state data
        if self.state_data:
            df_state = pd.DataFrame(self.state_data)
            df_state.to_csv(os.path.join(self.session_dir, "state_data.csv"), index=False)
            
        # Save performance data
        if self.performance_data:
            df_performance = pd.DataFrame(self.performance_data)
            df_performance.to_csv(os.path.join(self.session_dir, "performance.csv"), index=False)
            
        # Save ghost particle data (periodic boundary activity)
        if self.ghost_particle_counts:
            df_ghost = pd.DataFrame(self.ghost_particle_counts)
            df_ghost.to_csv(os.path.join(self.session_dir, "boundary_activity.csv"), index=False)
        
        # Generate plots
        self.generate_plots()
        
    def finalize(self):
        """Finalize logging and generate summary"""
        self.save_logs()
        
        # Update metadata with end time
        with open(os.path.join(self.session_dir, "metadata.txt"), "a") as f:
            end_time = time.time()
            elapsed = end_time - self.start_time
            
            f.write(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
            
            if self.energy_data:
                df_energy = pd.DataFrame(self.energy_data)
                final_record = df_energy.iloc[-1]
                
                f.write("\nFinal Statistics:\n")
                f.write(f"Total particles: {int(final_record['particle_count'])}\n")
                f.write(f"Protons: {int(final_record['proton_count'])}\n")
                f.write(f"Electrons: {int(final_record['electron_count'])}\n")
                f.write(f"Neutrons: {int(final_record['neutron_count'])}\n")
                f.write(f"Bonds formed: {int(final_record['bond_count'])}\n")
                f.write(f"Hydrogen atoms: {int(final_record['hydrogen_count'])}\n")
                f.write(f"Helium atoms: {int(final_record['helium_count'])}\n")
                
            # Add performance summary
            if self.performance_data:
                df_perf = pd.DataFrame(self.performance_data)
                
                f.write("\nPerformance Summary:\n")
                f.write(f"Average CPU usage: {df_perf['cpu_percent'].mean():.1f}%\n")
                f.write(f"Peak memory usage: {df_perf['memory_usage_mb'].max():.1f} MB\n")
                if 'gpu_memory_used_mb' in df_perf and df_perf['gpu_memory_used_mb'].max() > 0:
                    f.write(f"Peak GPU memory usage: {df_perf['gpu_memory_used_mb'].max():.1f} MB\n")
                f.write(f"Average grid update time: {df_perf['grid_update_time'].mean() * 1000:.2f} ms\n")
                
            print(f"Simulation data saved to {self.session_dir}")