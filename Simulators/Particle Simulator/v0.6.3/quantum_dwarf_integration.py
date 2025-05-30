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