import numpy as np
import cupy as cp
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DataLogger:
    """Log and save simulation data"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.particle_data = []
        self.field_data = []
        self.energy_data = []
        self.state_data = []
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
        
    def log_state(self, time, particle_system, grid):
        """Log current state of simulation"""
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
                'angular_velocity_x': particle.angular_velocity[0],
                'angular_velocity_y': particle.angular_velocity[1],
                'angular_velocity_z': particle.angular_velocity[2],
                'angular_momentum_magnitude': angular_momentum_mag,
                'kinetic_energy': kinetic_energy,
                'bonded': len(particle.bonded_with) > 0,
                'bond_type': particle.bond_type
            })
        
        # Log field data (sampled at lower resolution)
        sample_step = 16  # Sample every 16 cells for storage efficiency
        for i in range(0, grid.resolution, sample_step):
            for j in range(0, grid.resolution, sample_step):
                for k in range(0, grid.resolution, sample_step):
                    # Transfer GPU data to CPU for logging
                    # Field vector magnitude
                    field_mag = float(cp.linalg.norm(grid.velocity_field[i, j, k]).get())
                    
                    # Vorticity magnitude
                    vorticity_mag = float(grid.vorticity_magnitude[i, j, k].get())
                    
                    # State
                    state = int(cp.asnumpy(grid.state[i, j, k]))
                    
                    # Pressure
                    pressure = float(cp.asnumpy(grid.pressure_field[i, j, k]))
                    
                    # Energy density
                    energy_density = float(cp.asnumpy(grid.energy_density[i, j, k]))
                    
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
    
    def generate_plots(self):
        """Generate analysis plots"""
        if not self.energy_data:
            return
            
        # Convert to DataFrames for easier plotting
        df_energy = pd.DataFrame(self.energy_data)
        df_states = pd.DataFrame(self.state_data)
        
        # Plot 1: Energy over time
        plt.figure(figsize=(10, 5))
        plt.plot(df_energy['time'], df_energy['kinetic_energy'], label='Kinetic Energy')
        plt.plot(df_energy['time'], df_energy['field_energy'], label='Field Energy')
        plt.plot(df_energy['time'], df_energy['total_energy'], label='Total Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Distribution Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'energy_over_time.png'))
        plt.close()
        
        # Plot 2: Particle counts over time
        plt.figure(figsize=(10, 5))
        plt.plot(df_energy['time'], df_energy['proton_count'], label='Protons')
        plt.plot(df_energy['time'], df_energy['electron_count'], label='Electrons')
        plt.plot(df_energy['time'], df_energy['neutron_count'], label='Neutrons')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Particle Counts Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'particle_counts.png'))
        plt.close()
        
        # Plot 3: Bond and atom formation over time
        plt.figure(figsize=(10, 5))
        plt.plot(df_energy['time'], df_energy['bond_count'], label='Bonds')
        plt.plot(df_energy['time'], df_energy['hydrogen_count'], label='Hydrogen Atoms')
        plt.plot(df_energy['time'], df_energy['helium_count'], label='Helium Atoms')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Bond and Atom Formation Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'atom_formation.png'))
        plt.close()
        
        # Plot 4: Fluid state distribution over time
        plt.figure(figsize=(10, 5))
        plt.plot(df_states['time'], df_states['uncompressed'], label='Uncompressed')
        plt.plot(df_states['time'], df_states['compressed'], label='Compressed')
        plt.plot(df_states['time'], df_states['vacuum'], label='Vacuum')
        plt.xlabel('Time')
        plt.ylabel('Cell Count')
        plt.title('Fluid State Distribution Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, 'fluid_states.png'))
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
        
        # Generate plots
        self.generate_plots()
        
    def finalize(self):
        """Finalize logging and generate summary"""
        self.save_logs()
        
        # Update metadata with end time
        with open(os.path.join(self.session_dir, "metadata.txt"), "a") as f:
            f.write(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
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