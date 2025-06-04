import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os

class DwarfPhysicsEngine:
    """
    DWARF 2D Physics Engine with fully dynamic particles and proper field interactions
    Console version with matplotlib plotting
    """
    
    def __init__(self):
        # === SIMULATION PARAMETERS ===
        
        # Time integration - adjusted for better dynamics
        self.dt = 1e-15  # seconds (1 femtosecond)
        self.time = 0.0
        self.step_count = 0
        
        # Domain
        self.domain_size = 100.0  # Bohr radii
        
        # === PARTICLE PROPERTIES ===
        
        # Proton (now dynamic!)
        self.proton_mass = 1836.0
        self.proton_position = np.array([self.domain_size / 2, self.domain_size / 2])
        self.proton_velocity = np.array([0.0, 0.0])  # starts at rest but can move
        self.proton_spin_rpm = 81918  # creates wake field
        self.proton_charge = +1.0
        
        # Electron (dynamic with initial velocity to get motion started)
        self.electron_mass = 1.0
        self.electron_position = self.proton_position + np.array([5.0, 0.0])
        self.electron_velocity = np.array([0.0, 0.5])  # larger initial velocity
        self.electron_spin_rpm = 0.0  # can be induced
        self.electron_charge = -1.0
        
        # === DWARF FORCE PARAMETERS ===
        
        # 1. DWARF Force (2.22 Gradient Force) - adjusted for better interaction
        self.K_dwarf = 5e4  # reduced for stability but still significant
        self.force_exponent = 2.22
        
        # 2. Memory Field - improved implementation
        self.tau_m_base = 1e-13  # base memory time (100 fs)
        self.memory_field_strength = 1.0
        
        # 3. Wake Field - proper implementation
        self.A_wake = 1.0  # wake amplitude
        self.L_wake = 3.0  # wake length scale (Bohr radii)
        self.wake_decay_rate = 0.1  # how fast wake decays
        
        # 4. Viscosity (drag) - very light drag
        self.mu = 0.0001
        
        # 5. Soft-Core Repulsion - adjusted
        self.K_repel = 1e5
        self.repel_n = 6
        self.core_radius = 1.0  # minimum approach distance
        
        # === WAKE FIELD TRACKING ===
        self.proton_wake_trail = []
        self.electron_wake_trail = []
        self.max_wake_points = 200
        
        # === DATA LOGGING ===
        self.log_data = {
            'time': [],
            'proton_x': [], 'proton_y': [], 'proton_vx': [], 'proton_vy': [],
            'electron_x': [], 'electron_y': [], 'electron_vx': [], 'electron_vy': [],
            'electron_ke': [], 'proton_ke': [], 'total_ke': [],
            'electron_pe': [], 'total_energy': [],
            'distance_r': [],
            'F_dwarf_e_x': [], 'F_dwarf_e_y': [],
            'F_dwarf_p_x': [], 'F_dwarf_p_y': [],
            'F_memory_e': [], 'F_memory_p': [],
            'F_drag_e_x': [], 'F_drag_e_y': [],
            'F_drag_p_x': [], 'F_drag_p_y': [],
            'F_repel_e_x': [], 'F_repel_e_y': [],
            'F_repel_p_x': [], 'F_repel_p_y': [],
            'F_total_e_x': [], 'F_total_e_y': [],
            'F_total_p_x': [], 'F_total_p_y': [],
            'angular_momentum': [],
            'wake_amplitude_e': [], 'wake_amplitude_p': []
        }
        
    def update_wake_trails(self):
        """Update wake trails for both particles based on their motion"""
        
        # Add current positions to wake trails
        proton_wake_point = {
            'position': self.proton_position.copy(),
            'velocity': self.proton_velocity.copy(),
            'time': self.time,
            'strength': self.proton_spin_rpm / 81918.0  # normalized strength
        }
        
        electron_wake_point = {
            'position': self.electron_position.copy(),
            'velocity': self.electron_velocity.copy(),
            'time': self.time,
            'strength': np.linalg.norm(self.electron_velocity) * 0.5  # velocity-based strength
        }
        
        self.proton_wake_trail.append(proton_wake_point)
        self.electron_wake_trail.append(electron_wake_point)
        
        # Limit trail length
        if len(self.proton_wake_trail) > self.max_wake_points:
            self.proton_wake_trail.pop(0)
        if len(self.electron_wake_trail) > self.max_wake_points:
            self.electron_wake_trail.pop(0)
    
    def calculate_wake_field_at_position(self, position, exclude_particle=None):
        """Calculate total wake field at a given position from all wake sources"""
        total_wake = 0.0
        
        # Wake from proton trail
        if exclude_particle != 'proton':
            for wake_point in self.proton_wake_trail:
                r_wake = np.linalg.norm(position - wake_point['position'])
                if r_wake > 0:
                    # Time since this wake point was created
                    age = self.time - wake_point['time']
                    
                    # Wake amplitude with spatial and temporal decay
                    spatial_decay = np.exp(-r_wake / self.L_wake)
                    temporal_decay = np.exp(-age / self.tau_m_base)
                    
                    wake_contribution = (wake_point['strength'] * self.A_wake * 
                                       spatial_decay * temporal_decay)
                    total_wake += wake_contribution
        
        # Wake from electron trail  
        if exclude_particle != 'electron':
            for wake_point in self.electron_wake_trail:
                r_wake = np.linalg.norm(position - wake_point['position'])
                if r_wake > 0:
                    age = self.time - wake_point['time']
                    
                    spatial_decay = np.exp(-r_wake / self.L_wake)
                    temporal_decay = np.exp(-age / self.tau_m_base)
                    
                    wake_contribution = (wake_point['strength'] * self.A_wake * 
                                       spatial_decay * temporal_decay)
                    total_wake += wake_contribution
        
        return total_wake
    
    def calculate_memory_field_at_position(self, position):
        """Calculate memory field based on accumulated wake history"""
        memory_strength = 0.0
        
        # Memory accumulates from all previous wake activity
        for wake_point in self.proton_wake_trail + self.electron_wake_trail:
            r_memory = np.linalg.norm(position - wake_point['position'])
            if r_memory > 0:
                age = self.time - wake_point['time']
                
                # Memory field with different decay profile than wake
                tau_m_r = self.tau_m_base * (1.0 + 0.1 * r_memory)  # space-dependent memory time
                memory_decay = np.exp(-age / tau_m_r)
                
                memory_contribution = wake_point['strength'] * memory_decay / (1.0 + r_memory * 0.1)
                memory_strength += memory_contribution
        
        return memory_strength
    
    def calculate_dwarf_force(self, pos1, pos2, charge1, charge2):
        """Calculate DWARF force between two particles"""
        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        
        if r < 1e-10:
            return np.array([0.0, 0.0]), 0.0
            
        r_hat = r_vec / r
        
        # DWARF force magnitude with charge interaction
        F_dwarf_mag = self.K_dwarf * abs(charge1 * charge2) / (r ** self.force_exponent)
        
        # Attractive if opposite charges, repulsive if same charges
        if charge1 * charge2 < 0:
            F_dwarf = F_dwarf_mag * r_hat  # attractive (toward other particle)
        else:
            F_dwarf = -F_dwarf_mag * r_hat  # repulsive (away from other particle)
        
        return F_dwarf, F_dwarf_mag
    
    def calculate_drag_force(self, velocity):
        """Calculate viscosity drag force"""
        return -self.mu * velocity
    
    def calculate_repulsion_force(self, pos1, pos2):
        """Calculate soft-core repulsion between particles"""
        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        
        if r < 1e-10:
            return np.array([0.0, 0.0]), 0.0
        
        if r > self.core_radius:
            return np.array([0.0, 0.0]), 0.0
            
        r_hat = r_vec / r
        F_repel_mag = self.K_repel / (r ** self.repel_n)
        F_repel = -F_repel_mag * r_hat  # repulsive (away from other particle)
        
        return F_repel, F_repel_mag
    
    def step(self):
        """Advance simulation by one time step with full particle interactions"""
        
        # Update wake trails first
        self.update_wake_trails()
        
        # Calculate distance between particles
        r_vec = self.electron_position - self.proton_position
        r = np.linalg.norm(r_vec)
        
        # === FORCES ON ELECTRON ===
        
        # 1. DWARF force from proton
        F_dwarf_e, F_dwarf_e_mag = self.calculate_dwarf_force(
            self.proton_position, self.electron_position, 
            self.proton_charge, self.electron_charge
        )
        
        # 2. Wake field at electron position (from proton)
        wake_field_e = self.calculate_wake_field_at_position(
            self.electron_position, exclude_particle='electron'
        )
        
        # 3. Memory field at electron position
        memory_field_e = self.calculate_memory_field_at_position(self.electron_position)
        
        # 4. Drag on electron
        F_drag_e = self.calculate_drag_force(self.electron_velocity)
        
        # 5. Repulsion from proton
        F_repel_e, _ = self.calculate_repulsion_force(
            self.proton_position, self.electron_position
        )
        
        # === FORCES ON PROTON ===
        
        # 1. DWARF force from electron (Newton's 3rd law)
        F_dwarf_p = -F_dwarf_e
        
        # 2. Wake field at proton position (from electron)
        wake_field_p = self.calculate_wake_field_at_position(
            self.proton_position, exclude_particle='proton'
        )
        
        # 3. Memory field at proton position
        memory_field_p = self.calculate_memory_field_at_position(self.proton_position)
        
        # 4. Drag on proton
        F_drag_p = self.calculate_drag_force(self.proton_velocity)
        
        # 5. Repulsion from electron
        F_repel_p = -F_repel_e
        
        # === COMBINE FORCES WITH FIELD MODULATION ===
        
        # Memory and wake fields modulate the DWARF forces
        F_dwarf_e_modulated = F_dwarf_e * (1.0 + memory_field_e * 0.1) * (1.0 + wake_field_e * 0.1)
        F_dwarf_p_modulated = F_dwarf_p * (1.0 + memory_field_p * 0.1) * (1.0 + wake_field_p * 0.1)
        
        # Total forces
        F_total_e = F_dwarf_e_modulated + F_drag_e + F_repel_e
        F_total_p = F_dwarf_p_modulated + F_drag_p + F_repel_p
        
        # === UPDATE MOTION FOR BOTH PARTICLES ===
        
        # Update velocities
        acc_e = F_total_e / self.electron_mass
        acc_p = F_total_p / self.proton_mass
        
        self.electron_velocity += acc_e * self.dt
        self.proton_velocity += acc_p * self.dt
        
        # Update positions
        self.electron_position += self.electron_velocity * self.dt
        self.proton_position += self.proton_velocity * self.dt
        
        # Update time
        self.time += self.dt
        self.step_count += 1
        
        # === CALCULATE ENERGIES ===
        
        # Kinetic energies
        ke_electron = 0.5 * self.electron_mass * np.dot(self.electron_velocity, self.electron_velocity)
        ke_proton = 0.5 * self.proton_mass * np.dot(self.proton_velocity, self.proton_velocity)
        total_ke = ke_electron + ke_proton
        
        # Potential energy (DWARF interaction)
        if r > 0:
            potential_energy = -self.K_dwarf * abs(self.proton_charge * self.electron_charge) / (r ** (self.force_exponent - 1))
        else:
            potential_energy = 0
            
        total_energy = total_ke + potential_energy
        
        # Angular momentum (relative to center of mass)
        com = (self.proton_mass * self.proton_position + self.electron_mass * self.electron_position) / (self.proton_mass + self.electron_mass)
        
        r_e_com = self.electron_position - com
        r_p_com = self.proton_position - com
        
        L_e = self.electron_mass * (r_e_com[0] * self.electron_velocity[1] - r_e_com[1] * self.electron_velocity[0])
        L_p = self.proton_mass * (r_p_com[0] * self.proton_velocity[1] - r_p_com[1] * self.proton_velocity[0])
        total_L = L_e + L_p
        
        # === LOG DATA ===
        self.log_data['time'].append(self.time)
        self.log_data['proton_x'].append(self.proton_position[0])
        self.log_data['proton_y'].append(self.proton_position[1])
        self.log_data['proton_vx'].append(self.proton_velocity[0])
        self.log_data['proton_vy'].append(self.proton_velocity[1])
        self.log_data['electron_x'].append(self.electron_position[0])
        self.log_data['electron_y'].append(self.electron_position[1])
        self.log_data['electron_vx'].append(self.electron_velocity[0])
        self.log_data['electron_vy'].append(self.electron_velocity[1])
        self.log_data['electron_ke'].append(ke_electron)
        self.log_data['proton_ke'].append(ke_proton)
        self.log_data['total_ke'].append(total_ke)
        self.log_data['electron_pe'].append(potential_energy)
        self.log_data['total_energy'].append(total_energy)
        self.log_data['distance_r'].append(r)
        self.log_data['F_dwarf_e_x'].append(F_dwarf_e[0])
        self.log_data['F_dwarf_e_y'].append(F_dwarf_e[1])
        self.log_data['F_dwarf_p_x'].append(F_dwarf_p[0])
        self.log_data['F_dwarf_p_y'].append(F_dwarf_p[1])
        self.log_data['F_memory_e'].append(memory_field_e)
        self.log_data['F_memory_p'].append(memory_field_p)
        self.log_data['F_drag_e_x'].append(F_drag_e[0])
        self.log_data['F_drag_e_y'].append(F_drag_e[1])
        self.log_data['F_drag_p_x'].append(F_drag_p[0])
        self.log_data['F_drag_p_y'].append(F_drag_p[1])
        self.log_data['F_repel_e_x'].append(F_repel_e[0])
        self.log_data['F_repel_e_y'].append(F_repel_e[1])
        self.log_data['F_repel_p_x'].append(F_repel_p[0])
        self.log_data['F_repel_p_y'].append(F_repel_p[1])
        self.log_data['F_total_e_x'].append(F_total_e[0])
        self.log_data['F_total_e_y'].append(F_total_e[1])
        self.log_data['F_total_p_x'].append(F_total_p[0])
        self.log_data['F_total_p_y'].append(F_total_p[1])
        self.log_data['angular_momentum'].append(total_L)
        self.log_data['wake_amplitude_e'].append(wake_field_e)
        self.log_data['wake_amplitude_p'].append(wake_field_p)
        
        return {
            'electron_pos': self.electron_position.copy(),
            'proton_pos': self.proton_position.copy(),
            'electron_vel': self.electron_velocity.copy(),
            'proton_vel': self.proton_velocity.copy(),
            'forces': {
                'dwarf_e': F_dwarf_e,
                'dwarf_p': F_dwarf_p,
                'drag_e': F_drag_e,
                'drag_p': F_drag_p,
                'repel_e': F_repel_e,
                'repel_p': F_repel_p,
                'total_e': F_total_e,
                'total_p': F_total_p
            },
            'physics': {
                'distance': r,
                'ke_electron': ke_electron,
                'ke_proton': ke_proton,
                'total_ke': total_ke,
                'potential_energy': potential_energy,
                'total_energy': total_energy,
                'angular_momentum': total_L,
                'memory_field_e': memory_field_e,
                'memory_field_p': memory_field_p,
                'wake_field_e': wake_field_e,
                'wake_field_p': wake_field_p
            }
        }
    
    def save_log_data(self, filename=None):
        """Save all logged data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dwarf_dynamic_sim_{timestamp}.csv"
        
        headers = list(self.log_data.keys())
        num_steps = len(self.log_data['time'])
        if num_steps == 0:
            print("No data to save")
            return None
            
        rows = []
        for i in range(num_steps):
            row = [self.log_data[key][i] for key in headers]
            rows.append(row)
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            
            print(f"Data saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving data: {e}")
            return None
    
    def plot_results(self, save_plots=True):
        """Create matplotlib plots of the simulation results"""
        if len(self.log_data['time']) == 0:
            print("No data to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('DWARF 2D Physics Simulation Results', fontsize=16)
            
            # 1. Particle trajectories
            ax1 = axes[0, 0]
            ax1.plot(self.log_data['proton_x'], self.log_data['proton_y'], 'r-', label='Proton', linewidth=2)
            ax1.plot(self.log_data['electron_x'], self.log_data['electron_y'], 'b-', label='Electron', linewidth=2)
            ax1.plot(self.log_data['proton_x'][0], self.log_data['proton_y'][0], 'ro', markersize=8, label='Proton Start')
            ax1.plot(self.log_data['electron_x'][0], self.log_data['electron_y'][0], 'bo', markersize=8, label='Electron Start')
            ax1.set_xlabel('X Position (Bohr)')
            ax1.set_ylabel('Y Position (Bohr)')
            ax1.set_title('Particle Trajectories')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # 2. Distance vs time
            ax2 = axes[0, 1]
            ax2.plot(self.log_data['time'], self.log_data['distance_r'], 'g-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Distance (Bohr)')
            ax2.set_title('Inter-particle Distance')
            ax2.grid(True, alpha=0.3)
            
            # 3. Energy vs time
            ax3 = axes[0, 2]
            ax3.plot(self.log_data['time'], self.log_data['total_ke'], 'b-', label='Kinetic Energy', linewidth=2)
            ax3.plot(self.log_data['time'], self.log_data['electron_pe'], 'r-', label='Potential Energy', linewidth=2)
            ax3.plot(self.log_data['time'], self.log_data['total_energy'], 'k-', label='Total Energy', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Energy')
            ax3.set_title('Energy Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Memory and wake fields
            ax4 = axes[1, 0]
            ax4.plot(self.log_data['time'], self.log_data['F_memory_e'], 'c-', label='Memory Field (Electron)', linewidth=2)
            ax4.plot(self.log_data['time'], self.log_data['F_memory_p'], 'm-', label='Memory Field (Proton)', linewidth=2)
            ax4.plot(self.log_data['time'], self.log_data['wake_amplitude_e'], 'b--', label='Wake Field (Electron)', linewidth=2)
            ax4.plot(self.log_data['time'], self.log_data['wake_amplitude_p'], 'r--', label='Wake Field (Proton)', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Field Strength')
            ax4.set_title('Memory & Wake Fields')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Angular momentum
            ax5 = axes[1, 1]
            ax5.plot(self.log_data['time'], self.log_data['angular_momentum'], 'purple', linewidth=2)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Angular Momentum')
            ax5.set_title('Angular Momentum')
            ax5.grid(True, alpha=0.3)
            
            # 6. Velocities
            ax6 = axes[1, 2]
            ax6.plot(self.log_data['time'], [np.sqrt(vx**2 + vy**2) for vx, vy in zip(self.log_data['electron_vx'], self.log_data['electron_vy'])], 
                    'b-', label='Electron Speed', linewidth=2)
            ax6.plot(self.log_data['time'], [np.sqrt(vx**2 + vy**2) for vx, vy in zip(self.log_data['proton_vx'], self.log_data['proton_vy'])], 
                    'r-', label='Proton Speed', linewidth=2)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Speed')
            ax6.set_title('Particle Speeds')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"dwarf_simulation_plots_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Plots saved to {plot_filename}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")


def main():
    """Main function to run the DWARF 2D dynamic simulator"""
    
    print("🌟 DWARF 2D Dynamic Physics Simulator 🌟")
    print("=" * 50)
    
    # Create physics engine
    physics_engine = DwarfPhysicsEngine()
    
    print(f"Initial setup:")
    print(f"  Proton: position=({physics_engine.proton_position[0]:.1f}, {physics_engine.proton_position[1]:.1f}), velocity=({physics_engine.proton_velocity[0]:.3f}, {physics_engine.proton_velocity[1]:.3f})")
    print(f"  Electron: position=({physics_engine.electron_position[0]:.1f}, {physics_engine.electron_position[1]:.1f}), velocity=({physics_engine.electron_velocity[0]:.3f}, {physics_engine.electron_velocity[1]:.3f})")
    print(f"  Initial distance: {np.linalg.norm(physics_engine.electron_position - physics_engine.proton_position):.3f} Bohr")
    print()
    
    # Run simulation
    print("Running DWARF dynamic physics simulation...")
    num_steps = 10000
    
    for i in range(num_steps):
        step_data = physics_engine.step()
        
        if i % 1000 == 0:
            physics = step_data['physics']
            print(f"Step {i:5d}: Distance={physics['distance']:.3f} Bohr, "
                  f"Memory_e={physics['memory_field_e']:.3f}, Wake_e={physics['wake_field_e']:.3f}, "
                  f"Total_E={physics['total_energy']:.2e}")
    
    print()
    print("Simulation complete!")
    
    # Final state
    final_distance = np.linalg.norm(physics_engine.electron_position - physics_engine.proton_position)
    print(f"Final distance: {final_distance:.3f} Bohr")
    print(f"Final electron position: ({physics_engine.electron_position[0]:.2f}, {physics_engine.electron_position[1]:.2f})")
    print(f"Final proton position: ({physics_engine.proton_position[0]:.2f}, {physics_engine.proton_position[1]:.2f})")
    print(f"Final electron velocity: ({physics_engine.electron_velocity[0]:.3e}, {physics_engine.electron_velocity[1]:.3e})")
    print(f"Final proton velocity: ({physics_engine.proton_velocity[0]:.3e}, {physics_engine.proton_velocity[1]:.3e})")
    
    # Save data and create plots
    filename = physics_engine.save_log_data()
    print(f"Data saved to {filename}")
    
    # Create analysis plots
    print("Creating analysis plots...")
    physics_engine.plot_results(save_plots=True)
    
    print("\n🎯 Analysis Summary:")
    print(f"   Total steps: {physics_engine.step_count}")
    print(f"   Total time: {physics_engine.time:.2e} seconds")
    print(f"   Wake trail lengths: Proton={len(physics_engine.proton_wake_trail)}, Electron={len(physics_engine.electron_wake_trail)}")
    print(f"   Memory field active: {len(physics_engine.proton_wake_trail + physics_engine.electron_wake_trail) > 0}")


if __name__ == "__main__":
    main()