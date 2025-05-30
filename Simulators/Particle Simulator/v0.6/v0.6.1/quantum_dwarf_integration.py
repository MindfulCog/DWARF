"""
Integration module that connects emergent quantum effects with the DWARF vortex model.
"""
import numpy as np
import matplotlib.pyplot as plt
from vortex_resonance_analyzer import VortexResonanceAnalyzer
from emergent_probability_tracker import EmergentProbabilityTracker
from vortex_torsion_analyzer import VortexTorsionAnalyzer

class EmergentQuantumTracker:
    """
    Tracks emergent quantum-like effects in the DWARF vortex model.
    """
    def __init__(self, vortex_simulator):
        """
        Initialize the quantum tracker.
        
        Args:
            vortex_simulator: The DWARF vortex simulator instance
        """
        self.vortex_simulator = vortex_simulator
        
        # Create analyzers to track different quantum-like emergent behaviors
        self.resonance_analyzer = VortexResonanceAnalyzer(vortex_simulator)
        self.probability_tracker = EmergentProbabilityTracker(vortex_simulator)
        self.torsion_analyzer = VortexTorsionAnalyzer(vortex_simulator)
        
        # Simulation time tracking
        self.time = 0.0
        
        # Current identified quantum-like state
        self.current_quantum_state = {
            "n": 1,
            "l": 0,
            "j": 0.5,
            "orbital_type": "s-like",
            "energy": 0.0
        }
        
    def update(self, dt, vortex_field=None):
        """
        Update the quantum tracking with new simulation data.
        
        Args:
            dt: Time step
            vortex_field: Optional vortex field data
        """
        # Update time
        self.time += dt
        
        # Record state in all analyzers
        self.resonance_analyzer.record_state(self.time)
        self.probability_tracker.record_position()
        self.torsion_analyzer.record_state(self.time, vortex_field)
        
        # Periodically update quantum state assessment
        if len(self.resonance_analyzer.energy_history) % 50 == 0:
            self._update_quantum_state()
            
    def _update_quantum_state(self):
        """
        Update the assessment of the current quantum-like state.
        """
        # Analyze resonances (energy quantization)
        resonance_info = self.resonance_analyzer.analyze_resonances()
        
        # Only update if we have meaningful data
        if resonance_info["resonances_detected"] and resonance_info["resonances"]:
            # Get the most recent/strongest resonance
            strongest_resonance = max(resonance_info["resonances"], 
                                    key=lambda r: r["duration"])
            
            # Update quantum numbers
            n_effective = strongest_resonance["effective_n"]
            n = round(n_effective)
            
            # Detect orbital characteristics (angular momentum)
            orbital_info = self.probability_tracker.detect_orbital_characteristics()
            
            if orbital_info["detected"]:
                # Map the detected orbital type to quantum numbers
                l = 0  # Default for s-orbital
                
                if orbital_info["orbital_type"] == "p-like":
                    l = 1
                elif orbital_info["orbital_type"] == "d-like":
                    l = 2
                    
                # Get approximate node count as additional input
                node_count = orbital_info.get("node_count", 0)
                
                # Refine l based on nodes if needed
                if node_count > l:
                    l = min(n-1, node_count)
                    
                # Update orbital type name if needed
                if l == 0:
                    orbital_type = "s-like"
                elif l == 1:
                    orbital_type = "p-like"
                elif l == 2:
                    orbital_type = "d-like"
                else:
                    orbital_type = "higher-angular-momentum"
            else:
                # Default values if orbital detection fails
                l = 0
                orbital_type = "unknown"
                
            # Get spin-orbit coupling info
            coupling_info = self.torsion_analyzer.analyze_spin_orbit_coupling()
            
            if coupling_info["detected"]:
                # Determine j value based on spin-orbit angle
                j = l + 0.5 if coupling_info["angle"] < 90 else max(0.5, l - 0.5)
            else:
                j = l + 0.5  # Default to aligned spin
                
            # Get fine structure info
            fine_structure_info = self.torsion_analyzer.get_fine_structure_shifts()
            
            if fine_structure_info["detected"]:
                total_energy = fine_structure_info["total_energy"]
            else:
                total_energy = strongest_resonance["energy"]
                
            # Update current quantum state
            self.current_quantum_state = {
                "n": n,
                "l": l,
                "j": j,
                "orbital_type": orbital_type,
                "energy": total_energy,
                "resonance_energy": strongest_resonance["energy"],
                "fine_structure_shift": fine_structure_info.get("fine_structure_shift", 0) 
                                      if fine_structure_info.get("detected", False) else 0
            }
            
    def get_quantum_state_info(self):
        """
        Get information about the current emergent quantum-like state.
        
        Returns:
            dict: Information about the current state
        """
        return self.current_quantum_state
        
    def visualize_quantum_effects(self, mode='all'):
        """
        Visualize the emergent quantum-like effects.
        
        Args:
            mode: Visualization mode ('all', 'resonances', 'probability', or 'spin_orbit')
        """
        if mode == 'all' or mode == 'resonances':
            self.resonance_analyzer.visualize_resonances()
            
        if mode == 'all' or mode == 'probability':
            self.probability_tracker.visualize_probability_cloud('3d_surface')
            self.probability_tracker.visualize_probability_cloud('radial')
            
        if mode == 'all' or mode == 'spin_orbit':
            self.torsion_analyzer.visualize_spin_orbit_coupling()
            
    def create_dashboard(self):
        """
        Create a comprehensive dashboard showing all emergent quantum effects.
        """
        print("Creating dashboard of emergent quantum effects...")
        
        # Analyze current state
        quantum_state = self.get_quantum_state_info()
        resonance_info = self.resonance_analyzer.analyze_resonances()
        orbital_info = self.probability_tracker.detect_orbital_characteristics()
        coupling_info = self.torsion_analyzer.analyze_spin_orbit_coupling()
        
        # Set up the figure
        fig = plt.figure(figsize=(15, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3)
        
        # 1. Resonance/Energy panel
        ax1 = fig.add_subplot(gs[0, 0])
        time_array = np.array(self.resonance_analyzer.time_history)
        energy_array = np.array(self.resonance_analyzer.energy_history)
        
        ax1.plot(time_array, energy_array, 'b-', label='Energy')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Energy')
        ax1.set_title('Emergent Energy Quantization')
        
        # Add horizontal lines for detected resonances
        if resonance_info["resonances_detected"]:
            quantized_states = self.resonance_analyzer.get_quantized_states_info()
            if quantized_states["states_detected"]:
                for state in quantized_states["states"]:
                    ax1.axhline(y=state["energy"], color='r', linestyle='--', alpha=0.5)
                    ax1.text(time_array[-1], state["energy"], f"n≈{state['n']}", 
                             ha='right', va='bottom')
        
        # 2. Probability cloud slice
        ax2 = fig.add_subplot(gs[0, 1])
        # Show a slice of the probability distribution if available
        if self.probability_tracker.last_density is not None:
            mid_idx = self.probability_tracker.grid_resolution // 2
            z_slice = self.probability_tracker.last_density[mid_idx, :, :]
            extent = [-self.probability_tracker.grid_size, self.probability_tracker.grid_size, 
                     -self.probability_tracker.grid_size, self.probability_tracker.grid_size]
            ax2.imshow(z_slice.T, extent=extent, origin='lower', cmap='viridis')
            ax2.set_title(f'Emergent {orbital_info["orbital_type"] if orbital_info["detected"] else "?"} Orbital Pattern')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
        else:
            ax2.text(0.5, 0.5, "Insufficient data for probability distribution", 
                    ha='center', va='center')
            ax2.set_title('Probability Distribution')
            
        # 3. Spin-Orbit visualization
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        
        # Plot recent trajectory
        if len(self.torsion_analyzer.position_history) > 10:
            positions = np.array(self.torsion_analyzer.position_history[-50:])
            ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.5)
            
            # Plot current position
            current_pos = positions[-1]
            ax3.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                       color='blue', s=30)
                       
            # Plot spin and orbital vectors
            r = np.max(np.linalg.norm(positions, axis=1))
            
            # Spin vector
            spin_scale = r * 0.5
            spin_vector = self.torsion_analyzer.effective_spin * spin_scale
            
            ax3.quiver(current_pos[0], current_pos[1], current_pos[2],
                      spin_vector[0], spin_vector[1], spin_vector[2],
                      color='red', label='Spin')
                      
            # Orbital angular momentum
            orbital_scale = r * 0.5
            orbital_vector = self.torsion_analyzer.orbital_plane_normal * orbital_scale
            
            ax3.quiver(0, 0, 0,
                      orbital_vector[0], orbital_vector[1], orbital_vector[2],
                      color='green', label='Orbital')
                      
            # Nucleus
            ax3.scatter([0], [0], [0], color='black', s=50)
            
            # Set axis limits
            ax3.set_xlim(-r, r)
            ax3.set_ylim(-r, r)
            ax3.set_zlim(-r, r)
            
            ax3.set_title('Emergent Spin-Orbit Coupling')
            ax3.legend()
        else:
            ax3.text(0, 0, 0, "Insufficient data", ha='center', va='center')