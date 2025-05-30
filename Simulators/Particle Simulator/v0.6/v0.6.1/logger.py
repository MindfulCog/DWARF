"""
Comprehensive logger for DWARF simulator.
Provides logging functionality for simulation steps, events, and quantum metrics.

Date: 2025-05-30
"""
import os
import json
import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import warnings

class SimulationLogger:
    """
    Main logger class for DWARF simulator.
    Handles all logging operations for the simulation.
    """
    def __init__(self, log_dir="logs", simulation_name=None, config=None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Base directory for logs
            simulation_name: Name of this simulation run
            config: SimulationConfig instance
        """
        self.log_dir = log_dir
        self.config = config
        
        # Generate timestamp and simulation name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if simulation_name:
            self.simulation_name = f"{simulation_name}_{self.timestamp}"
        else:
            self.simulation_name = f"dwarf_sim_{self.timestamp}"
            
        # Create log directory
        self.run_log_dir = os.path.join(log_dir, self.simulation_name)
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # Initialize log file paths
        self.step_log_path = os.path.join(self.run_log_dir, "simulation_steps.csv")
        self.event_log_path = os.path.join(self.run_log_dir, "events.txt")
        self.config_log_path = os.path.join(self.run_log_dir, "configuration.json")
        self.quantum_log_path = os.path.join(self.run_log_dir, "quantum_metrics.csv")
        self.resonance_log_path = os.path.join(self.run_log_dir, "resonances.csv")
        
        # Create step log file with header
        with open(self.step_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 'Time', 'Position_X', 'Position_Y', 'Position_Z',
                'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Energy'
            ])
            
        # Create quantum log file with header
        with open(self.quantum_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 'Time', 'Energy', 'N_Value', 'L_Value', 'J_Value',
                'Orbital_Type', 'Resonance_Strength', 'Node_Count',
                'Spin_Orbit_Angle', 'Fine_Structure_Shift'
            ])
            
        # Create resonance log file with header
        with open(self.resonance_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time_Start', 'Time_End', 'Duration', 'Energy',
                'Effective_N', 'Radius', 'Angular_Momentum'
            ])
            
        # Log configuration if provided
        if config:
            self._log_configuration(config)
            
        # Statistics tracking
        self.stats = defaultdict(list)
        
        # Tracking variables
        self.last_step = -1
        self.last_quantum_update = -1
        self.quantum_update_interval = 10  # Update quantum metrics every 10 steps
        self.detailed_interval = 100  # Detailed logging every 100 steps
        
        # Performance tracking
        self.start_time = datetime.datetime.now()
        
        print(f"Logger initialized. Logs will be saved to {self.run_log_dir}")
        
    def log_step(self, step, time, position, velocity, energy):
        """
        Log a simulation step.
        
        Args:
            step: Current simulation step
            time: Current simulation time
            position: Electron position (array)
            velocity: Electron velocity (array)
            energy: System energy
        """
        # Update tracking variables
        self.last_step = step
        
        # Track statistics
        self.stats["energy"].append(energy)
        self.stats["position_radius"].append(np.linalg.norm(position))
        self.stats["velocity_magnitude"].append(np.linalg.norm(velocity))
        
        # Write to CSV log
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, time, 
                position[0], position[1], position[2],
                velocity[0], velocity[1], velocity[2],
                energy
            ])
            
    def log_event(self, event_type, message):
        """
        Log a simulation event.
        
        Args:
            event_type: Type of event
            message: Event description
        """
        with open(self.event_log_path, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {event_type}: {message}\n")
            
    def log_quantum_metrics(self, step, time, quantum_tracker):
        """
        Log quantum metrics from the quantum tracker.
        
        Args:
            step: Current simulation step
            time: Current simulation time
            quantum_tracker: The EmergentQuantumTracker instance
        """
        # Only log if quantum tracker exists and it's time for an update
        if not quantum_tracker or step - self.last_quantum_update < self.quantum_update_interval:
            return
            
        self.last_quantum_update = step
        
        # Get quantum state information
        state = quantum_tracker.get_quantum_state_info()
        
        # Get resonance info
        resonance_info = quantum_tracker.resonance_analyzer.analyze_resonances()
        resonance_strength = 0
        if resonance_info.get("resonances_detected", False) and resonance_info.get("resonances", []):
            # Use the duration of the most recent resonance as a measure of strength
            resonance_strength = resonance_info["resonances"][-1].get("duration", 0)
            
            # Log newly detected resonances
            for resonance in resonance_info["resonances"]:
                self.log_resonance(resonance)
            
        # Get orbital characteristics
        orbital_info = quantum_tracker.probability_tracker.detect_orbital_characteristics()
        node_count = orbital_info.get("node_count", 0) if orbital_info.get("detected", False) else 0
        
        # Get spin-orbit coupling info
        coupling_info = quantum_tracker.torsion_analyzer.analyze_spin_orbit_coupling()
        spin_orbit_angle = coupling_info.get("angle", 0) if coupling_info.get("detected", False) else 0
        
        # Get energy and fine structure
        energy = state.get("energy", 0)
        fine_structure_shift = state.get("fine_structure_shift", 0)
        
        # Log to quantum metrics file
        with open(self.quantum_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                time, 
                energy, 
                state.get("n", 0), 
                state.get("l", 0), 
                state.get("j", 0),
                state.get("orbital_type", "unknown"), 
                resonance_strength, 
                node_count,
                spin_orbit_angle, 
                fine_structure_shift
            ])
            
        # Track quantum statistics
        self.stats["quantum_energy"].append(energy)
        self.stats["n_value"].append(state.get("n", 0))
        self.stats["l_value"].append(state.get("l", 0))
        self.stats["spin_orbit_angle"].append(spin_orbit_angle)
        self.stats["node_count"].append(node_count)
        
        # Log detailed information periodically
        if step % self.detailed_interval == 0:
            self._log_detailed_quantum(step, time, quantum_tracker)
            
    def log_resonance(self, resonance):
        """
        Log a detected resonance.
        
        Args:
            resonance: Resonance information dictionary
        """
        # Check if the resonance has been logged before
        # This is a simple check to avoid duplicate entries
        if not hasattr(self, 'logged_resonances'):
            self.logged_resonances = set()
            
        # Create a unique identifier for this resonance
        res_id = f"{resonance['start_time']:.4f}_{resonance['energy']:.4f}"
        
        if res_id not in self.logged_resonances:
            self.logged_resonances.add(res_id)
            
            with open(self.resonance_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    resonance.get("start_time", 0),
                    resonance.get("end_time", 0),
                    resonance.get("duration", 0),
                    resonance.get("energy", 0),
                    resonance.get("effective_n", 0),
                    resonance.get("radius", 0),
                    resonance.get("angular_momentum", 0)
                ])
                
            # Add to event log
            self.log_event("RESONANCE", 
                        f"Detected resonance with n≈{resonance.get('effective_n', 0):.2f}, " + 
                        f"energy={resonance.get('energy', 0):.4f}")
                
    def _log_detailed_quantum(self, step, time, quantum_tracker):
        """
        Log detailed quantum metrics information.
        
        Args:
            step: Current simulation step
            time: Current simulation time
            quantum_tracker: The EmergentQuantumTracker instance
        """
        # Create a detailed log file if needed
        detailed_dir = os.path.join(self.run_log_dir, "detailed")
        os.makedirs(detailed_dir, exist_ok=True)
        
        detailed_path = os.path.join(detailed_dir, f"quantum_detailed_{step}.json")
        
        # Get detailed information from all analyzers
        resonance_info = quantum_tracker.resonance_analyzer.analyze_resonances()
        orbital_info = quantum_tracker.probability_tracker.detect_orbital_characteristics()
        coupling_info = quantum_tracker.torsion_analyzer.analyze_spin_orbit_coupling()
        fine_structure = quantum_tracker.torsion_analyzer.get_fine_structure_shifts()
        
        # Create detailed log entry
        entry = {
            "step": step,
            "time": time,
            "quantum_state": quantum_tracker.get_quantum_state_info(),
            "resonance_analysis": {
                "detected": resonance_info.get("resonances_detected", False),
                "count": resonance_info.get("resonance_count", 0),
                "resonances": [
                    {
                        "energy": r.get("energy", 0),
                        "effective_n": r.get("effective_n", 0),
                        "duration": r.get("duration", 0)
                    }
                    for r in resonance_info.get("resonances", [])
                ]
            },
            "orbital_analysis": orbital_info if orbital_info.get("detected", False) else {"detected": False},
            "spin_orbit_analysis": coupling_info if coupling_info.get("detected", False) else {"detected": False},
            "fine_structure_analysis": fine_structure if fine_structure.get("detected", False) else {"detected": False}
        }
        
        # Write to JSON log file
        with open(detailed_path, 'w') as f:
            json.dump(entry, f, indent=2)
            
    def _log_configuration(self, config):
        """
        Log the simulation configuration.
        
        Args:
            config: SimulationConfig instance
        """
        # Convert config to dict
        config_dict = {}
        
        # Get all attributes that don't start with _ (non-private)
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                attr_value = getattr(config, attr_name)
                
                # Skip methods and complex objects
                if not callable(attr_value) and isinstance(attr_value, (int, float, str, bool, list, dict, tuple)):
                    # Handle numpy arrays
                    if isinstance(attr_value, (list, tuple)) and attr_value and hasattr(attr_value[0], 'tolist'):
                        attr_value = [item.tolist() if hasattr(item, 'tolist') else item for item in attr_value]
                        
                    config_dict[attr_name] = attr_value
        
        # Write to JSON file
        with open(self.config_log_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def generate_statistics_report(self):
        """
        Generate a report of collected statistics.
        
        Returns:
            str: Statistics report
        """
        report = []
        report.append("DWARF Simulator Statistics Report")
        report.append("===============================")
        report.append(f"Simulation: {self.simulation_name}")
        report.append(f"Duration: {self._get_elapsed_time()}")
        report.append(f"Steps: {self.last_step + 1}\n")
        
        # Basic simulation statistics
        if self.stats["energy"]:
            report.append("Core Simulation Statistics:")
            report.append("--------------------------")
            report.append(f"Energy:")
            report.append(f"  Mean: {np.mean(self.stats['energy']):.6f}")
            report.append(f"  Std Dev: {np.std(self.stats['energy']):.6f}")
            report.append(f"  Min: {np.min(self.stats['energy']):.6f}")
            report.append(f"  Max: {np.max(self.stats['energy']):.6f}\n")
            
            report.append(f"Orbital Radius:")
            report.append(f"  Mean: {np.mean(self.stats['position_radius']):.6f}")
            report.append(f"  Std Dev: {np.std(self.stats['position_radius']):.6f}")
            report.append(f"  Min: {np.min(self.stats['position_radius']):.6f}")
            report.append(f"  Max: {np.max(self.stats['position_radius']):.6f}\n")
        
        # Quantum statistics if available
        if self.stats["quantum_energy"]:
            report.append("Quantum Metrics Statistics:")
            report.append("--------------------------")
            
            report.append(f"Quantum Energy:")
            report.append(f"  Mean: {np.mean(self.stats['quantum_energy']):.6f} eV")
            report.append(f"  Std Dev: {np.std(self.stats['quantum_energy']):.6f} eV")
            report.append(f"  Min: {np.min(self.stats['quantum_energy']):.6f} eV")
            report.append(f"  Max: {np.max(self.stats['quantum_energy']):.6f} eV\n")
            
            # Quantum number statistics
            n_values, n_counts = np.unique(self.stats["n_value"], return_counts=True)
            report.append("N Value Distribution:")
            for n, count in zip(n_values, n_counts):
                percentage = 100 * count / len(self.stats["n_value"])
                report.append(f"  n={n}: {count} occurrences ({percentage:.1f}%)")
                
            # L value statistics
            l_values, l_counts = np.unique(self.stats["l_value"], return_counts=True)
            report.append("\nL Value Distribution:")
            for l, count in zip(l_values, l_counts):
                percentage = 100 * count / len(self.stats["l_value"])
                report.append(f"  l={l}: {count} occurrences ({percentage:.1f}%)")
                
            # Node count statistics
            report.append("\nNode Count Statistics:")
            report.append(f"  Mean: {np.mean(self.stats['node_count']):.2f}")
            report.append(f"  Max: {np.max(self.stats['node_count'])}")
            
            # Spin-orbit statistics
            report.append("\nSpin-Orbit Coupling Statistics:")
            report.append(f"  Mean angle: {np.mean(self.stats['spin_orbit_angle']):.2f}°")
            aligned = 100 * np.mean(np.array(self.stats['spin_orbit_angle']) < 90)
            anti_aligned = 100 * np.mean(np.array(self.stats['spin_orbit_angle']) >= 90)
            report.append(f"  Aligned percentage: {aligned:.1f}%")
            report.append(f"  Anti-aligned percentage: {anti_aligned:.1f}%")
        
        return "\n".join(report)
        
    def save_statistics_report(self):
        """
        Save statistics report to file.
        """
        report = self.generate_statistics_report()
        
        report_path = os.path.join(self.run_log_dir, "statistics_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"Statistics report saved to {report_path}")
        
        # Also create plots of key statistics
        self._create_statistics_plots()
        
    def _create_statistics_plots(self):
        """
        Create plots of key statistics.
        """
        # Create plots directory
        plots_dir = os.path.join(self.run_log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Energy plot
        if self.stats["energy"]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.stats["energy"])
            plt.title("Energy over Time")
            plt.xlabel("Step")
            plt.ylabel("Energy")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "energy_history.png"), dpi=150)
            plt.close()
            
        # Quantum metrics plots
        if self.stats["quantum_energy"]:
            # N value distribution
            plt.figure(figsize=(8, 6))
            n_values, n_counts = np.unique(self.stats["n_value"], return_counts=True)
            plt.bar(n_values, n_counts)
            plt.title("Distribution of Quantum Number n")
            plt.xlabel("n Value")
            plt.ylabel("Count")
            plt.xticks(n_values)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "n_value_distribution.png"), dpi=150)
            plt.close()
            
            # Spin-orbit angle history
            plt.figure(figsize=(10, 6))
            plt.plot(self.stats["spin_orbit_angle"])
            plt.title("Spin-Orbit Angle over Time")
            plt.xlabel("Measurement")
            plt.ylabel("Angle (degrees)")
            plt.axhline(y=90, color='r', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "spin_orbit_angle.png"), dpi=150)
            plt.close()
            
    def _get_elapsed_time(self):
        """
        Get elapsed time since logger was initialized.
        
        Returns:
            str: Formatted elapsed time
        """
        elapsed = datetime.datetime.now() - self.start_time
        seconds = elapsed.total_seconds()
        
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
            
    def save_checkpoint(self, vortex_simulator, quantum_tracker=None, step=None):
        """
        Save a simulation checkpoint.
        
        Args:
            vortex_simulator: The DWARF vortex simulator
            quantum_tracker: The quantum tracker (optional)
            step: Current simulation step (optional)
        """
        # Create checkpoints directory
        ckpt_dir = os.path.join(self.run_log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Use current step if not specified
        if step is None:
            step = self.last_step
            
        # Save vortex simulator state
        try:
            ckpt_path = os.path.join(ckpt_dir, f"simulator_state_{step}.pkl")
            with open(ckpt_path, 'wb') as f:
                pickle.dump(vortex_simulator, f)
                
            # Save quantum tracker state if provided
            if quantum_tracker:
                q_ckpt_path = os.path.join(ckpt_dir, f"quantum_state_{step}.pkl")
                with open(q_ckpt_path, 'wb') as f:
                    pickle.dump(quantum_tracker, f)
                    
            self.log_event("CHECKPOINT", f"Saved checkpoint at step {step}")
        except Exception as e:
            self.log_event("ERROR", f"Failed to save checkpoint: {str(e)}")
            warnings.warn(f"Failed to save checkpoint: {str(e)}")
            
    def load_checkpoint(self, step, vortex_simulator_class=None):
        """
        Load a simulation checkpoint.
        
        Args:
            step: Step to load checkpoint from
            vortex_simulator_class: Class to use for creating vortex simulator (if needed)
            
        Returns:
            tuple: (vortex_simulator, quantum_tracker) or None if failed
        """
        ckpt_dir = os.path.join(self.run_log_dir, "checkpoints")
        
        # Check if checkpoint exists
        ckpt_path = os.path.join(ckpt_dir, f"simulator_state_{step}.pkl")
        q_ckpt_path = os.path.join(ckpt_dir, f"quantum_state_{step}.pkl")
        
        if not os.path.exists(ckpt_path):
            warnings.warn(f"Checkpoint at step {step} not found")
            return None
            
        try:
            # Load vortex simulator
            with open(ckpt_path, 'rb') as f:
                vortex_simulator = pickle.load(f)
                
            # Load quantum tracker if it exists
            quantum_tracker = None
            if os.path.exists(q_ckpt_path):
                with open(q_ckpt_path, 'rb') as f:
                    quantum_tracker = pickle.load(f)
                    
            self.log_event("CHECKPOINT", f"Loaded checkpoint from step {step}")
            return vortex_simulator, quantum_tracker
        except Exception as e:
            self.log_event("ERROR", f"Failed to load checkpoint: {str(e)}")
            warnings.warn(f"Failed to load checkpoint: {str(e)}")
            return None

# Create a global logger instance with default settings
# This allows log_step to be imported directly
global_logger = SimulationLogger()

# For backward compatibility - these functions use the global logger
def log_step(step, time, position, velocity, energy):
    """
    Log a simulation step using the global logger.
    
    Args:
        step: Current simulation step
        time: Current simulation time
        position: Electron position (array)
        velocity: Electron velocity (array)
        energy: System energy
    """
    global_logger.log_step(step, time, position, velocity, energy)

def log_event(event_type, message):
    """
    Log an event using the global logger.
    
    Args:
        event_type: Type of event
        message: Event description
    """
    global_logger.log_event(event_type, message)

def get_logger():
    """
    Get the global logger instance.
    
    Returns:
        SimulationLogger: The global logger
    """
    return global_logger

def create_new_logger(log_dir="logs", simulation_name=None, config=None):
    """
    Create a new logger instance.
    
    Args:
        log_dir: Base directory for logs
        simulation_name: Name of this simulation run
        config: SimulationConfig instance
        
    Returns:
        SimulationLogger: A new logger instance
    """
    global global_logger
    global_logger = SimulationLogger(log_dir, simulation_name, config)
    return global_logger