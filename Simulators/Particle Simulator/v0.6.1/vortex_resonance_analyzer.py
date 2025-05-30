"""
Analyzes energy resonances in the DWARF vortex model.
Identifies and characterizes quantized energy states.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class VortexResonanceAnalyzer:
    """
    Analyzes resonances (quantized energy states) in the DWARF vortex model.
    """
    def __init__(self, simulator, buffer_size=10000):
        """
        Initialize the resonance analyzer.
        
        Args:
            simulator: The DWARF vortex simulator
            buffer_size: Maximum size of history buffers
        """
        self.simulator = simulator
        self.buffer_size = buffer_size
        
        # Energy and time history
        self.energy_history = []
        self.time_history = []
        self.radius_history = []
        
        # Detected resonances
        self.resonances = []
        
        # Resonance detection parameters
        self.energy_stability_threshold = 0.01  # Energy variation tolerance
        self.min_resonance_duration = 10  # Minimum duration for resonance detection
        self.detection_window = 100  # Data points to consider for detection
        
    def record_state(self, time):
        """
        Record current energy and radius for resonance analysis.
        
        Args:
            time: Current simulation time
        """
        # Get current electron data
        position = self.simulator.get_electron_position()
        energy = self.simulator.get_electron_energy()
        
        # Calculate radius (distance from origin)
        radius = np.linalg.norm(position)
        
        # Add to history
        self.energy_history.append(energy)
        self.time_history.append(time)
        self.radius_history.append(radius)
        
        # Limit buffer size
        if len(self.energy_history) > self.buffer_size:
            self.energy_history.pop(0)
            self.time_history.pop(0)
            self.radius_history.pop(0)
            
    def analyze_resonances(self, min_duration=None):
        """
        Analyze energy history to detect resonant energy levels.
        
        Args:
            min_duration: Minimum resonance duration (optional)
            
        Returns:
            dict: Resonance analysis results
        """
        # Use default or specified minimum duration
        min_duration = min_duration or self.min_resonance_duration
        
        if len(self.energy_history) < self.detection_window:
            # Not enough data for analysis
            return {"resonances_detected": False, "resonances": []}
            
        # Process energy history to detect stable energy periods
        current_resonances = self._detect_stable_energy_periods(min_duration)
        
        # Add any new resonances to the full list
        for res in current_resonances:
            if not any(self._is_same_resonance(res, existing) for existing in self.resonances):
                self.resonances.append(res)
                
        # Return resonance info
        return {
            "resonances_detected": len(current_resonances) > 0,
            "resonance_count": len(self.resonances),
            "resonances": self.resonances
        }
        
    def _detect_stable_energy_periods(self, min_duration):
        """
        Detect periods of stable energy in the history.
        
        Args:
            min_duration: Minimum duration for a valid resonance
            
        Returns:
            list: Detected resonance events
        """
        # Create a sliding window over the energy history
        resonances = []
        window_size = min_duration
        
        # Need enough history for a meaningful analysis
        if len(self.energy_history) < 2 * window_size:
            return []
            
        # Analyze energy stability in sliding windows
        start_idx = 0
        in_resonance = False
        resonance_start_idx = 0
        resonance_energy = 0
        
        for i in range(window_size, len(self.energy_history)):
            # Get the window of energies to analyze
            window = self.energy_history[i-window_size:i]
            
            # Calculate energy statistics in this window
            mean_energy = np.mean(window)
            std_dev = np.std(window)
            
            # Check if energy is stable (standard deviation below threshold)
            is_stable = std_dev < self.energy_stability_threshold * abs(mean_energy)
            
            if is_stable and not in_resonance:
                # Start of a resonance
                in_resonance = True
                resonance_start_idx = i - window_size
                resonance_energy = mean_energy
            elif not is_stable and in_resonance:
                # End of a resonance
                in_resonance = False
                resonance_end_idx = i - 1
                resonance_duration = self.time_history[resonance_end_idx] - self.time_history[resonance_start_idx]
                
                # Only record if duration is significant
                if resonance_duration >= min_duration:
                    # Calculate average radius during this resonance
                    avg_radius = np.mean(self.radius_history[resonance_start_idx:resonance_end_idx])
                    
                    # Calculate effective quantum number based on energy (-13.6/n^2)
                    effective_n = np.sqrt(-13.6 / resonance_energy) if resonance_energy < 0 else 0
                    
                    # Create resonance record
                    resonance = {
                        "start_time": self.time_history[resonance_start_idx],
                        "end_time": self.time_history[resonance_end_idx],
                        "duration": resonance_duration,
                        "energy": resonance_energy,
                        "effective_n": effective_n,
                        "radius": avg_radius,
                        "angular_momentum": 0  # To be calculated if needed
                    }
                    
                    resonances.append(resonance)
                    
        # Check if we're still in a resonance at the end of the data
        if in_resonance:
            resonance_end_idx = len(self.energy_history) - 1
            resonance_duration = self.time_history[resonance_end_idx] - self.time_history[resonance_start_idx]
            
            if resonance_duration >= min_duration:
                # Calculate the same metrics as above
                avg_radius = np.mean(self.radius_history[resonance_start_idx:resonance_end_idx])
                effective_n = np.sqrt(-13.6 / resonance_energy) if resonance_energy < 0 else 0
                
                resonance = {
                    "start_time": self.time_history[resonance_start_idx],
                    "end_time": self.time_history[resonance_end_idx],
                    "duration": resonance_duration,
                    "energy": resonance_energy,
                    "effective_n": effective_n,
                    "radius": avg_radius,
                    "angular_momentum": 0
                }
                
                resonances.append(resonance)
                
        return resonances
        
    def _is_same_resonance(self, res1, res2, energy_tolerance=0.05):
        """
        Check if two resonances are effectively the same state.
        
        Args:
            res1: First resonance
            res2: Second resonance
            energy_tolerance: Energy difference tolerance
            
        Returns:
            bool: True if resonances represent the same state
        """
        # Check if energies are close enough to be the same state
        energy_diff = abs(res1["energy"] - res2["energy"])
        rel_diff = energy_diff / abs(max(res1["energy"], res2["energy"], 1e-10))
        
        return rel_diff < energy_tolerance
        
    def get_quantized_states_info(self):
        """
        Extract information about detected quantized energy states.
        
        Returns:
            dict: Information about quantized states
        """
        if not self.resonances:
            return {"states_detected": False, "states": []}
            
        # Group resonances by similar energy levels
        grouped_resonances = self._group_similar_resonances()
        
        # Extract state information from each group
        states = []
        for group in grouped_resonances:
            if not group:
                continue
                
            # Calculate average state properties
            energies = [res["energy"] for res in group]
            n_values = [res["effective_n"] for res in group]
            durations = [res["duration"] for res in group]
            
            # Create state record
            state = {
                "energy": np.mean(energies),
                "n": np.round(np.mean(n_values), 1),
                "occurrences": len(group),
                "avg_duration": np.mean(durations)
            }
            
            states.append(state)
            
        return {
            "states_detected": len(states) > 0,
            "states": sorted(states, key=lambda s: s["energy"])
        }
        
    def _group_similar_resonances(self, energy_tolerance=0.05):
        """
        Group resonances with similar energies.
        
        Args:
            energy_tolerance: Energy difference tolerance
            
        Returns:
            list: Groups of similar resonances
        """
        if not self.resonances:
            return []
            
        # Start with each resonance in its own group
        groups = [[res] for res in self.resonances]
        
        # Iteratively merge groups with similar energies
        merged = True
        while merged:
            merged = False
            
            for i in range(len(groups)):
                if not groups[i]:
                    continue
                    
                group1_energy = np.mean([res["energy"] for res in groups[i]])
                
                for j in range(i + 1, len(groups)):
                    if not groups[j]:
                        continue
                        
                    group2_energy = np.mean([res["energy"] for res in groups[j]])
                    
                    energy_diff = abs(group1_energy - group2_energy)
                    rel_diff = energy_diff / abs(max(group1_energy, group2_energy, 1e-10))
                    
                    if rel_diff < energy_tolerance:
                        # Merge groups
                        groups[i].extend(groups[j])
                        groups[j] = []
                        merged = True
                        
        # Remove empty groups
        return [group for group in groups if group]
        
    def visualize_resonances(self):
        """
        Visualize detected energy resonances.
        """
        if not self.energy_history:
            print("No energy data available for visualization")
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot energy history
        plt.plot(self.time_history, self.energy_history, 'b-', linewidth=1, label='Energy')
        
        # Highlight resonance regions
        resonance_info = self.analyze_resonances()
        if resonance_info["resonances_detected"]:
            for res in resonance_info["resonances"]:
                plt.axvspan(res["start_time"], res["end_time"], 
                           color='yellow', alpha=0.3)
                plt.axhline(y=res["energy"], color='r', linestyle='--', 
                           alpha=0.5, linewidth=1)
                plt.text(res["end_time"], res["energy"], 
                        f"n~{res['effective_n']:.1f}", fontsize=10)
                
        # Get quantized states
        states_info = self.get_quantized_states_info()
        if states_info["states_detected"]:
            # Add dashed lines for each detected state
            for state in states_info["states"]:
                plt.axhline(y=state["energy"], color='green', linestyle='-.',
                           alpha=0.7, linewidth=2)
                
                # Add state label on the right
                plt.text(max(self.time_history) * 1.01, state["energy"], 
                        f"n~{state['n']}", fontsize=12,
                        verticalalignment='center')
        
        plt.title('Emergent Energy Quantization in DWARF Vortex Model')
        plt.xlabel('Time')
        plt.ylabel('Energy (eV)')
        plt.grid(True, alpha=0.3)
        
        # Show resonance statistics in a text box
        if resonance_info["resonances"]:
            info_text = f"Detected {len(resonance_info['resonances'])} resonances\n"
            if states_info["states"]:
                info_text += f"Quantized states: {len(states_info['states'])}\n"
                for i, state in enumerate(sorted(states_info["states"], key=lambda s: s["energy"])):
                    info_text += f"State {i+1}: n~{state['n']}, E~{state['energy']:.4f} eV\n"
            
            plt.figtext(0.02, 0.02, info_text, fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()