"""
Hybrid logger module that supports both v0.5 style flat file logging
and the newer enhanced hierarchical logging.

This combines the best of both approaches for backward compatibility.
"""
import os
import csv
import json
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Global logger instance for enhanced style
_LOGGER = None

# Global variables for v0.5 style
log_writers = None
log_files = None

def create_new_logger(log_dir="logs", simulation_name=None, config=None):
    """
    Create a new logger.
    
    Args:
        log_dir: Directory to store logs
        simulation_name: Name of the simulation
        config: SimulationConfig instance
        
    Returns:
        DWARFLogger: New logger instance
    """
    global _LOGGER
    _LOGGER = DWARFLogger(log_dir, simulation_name, config)
    return _LOGGER

def get_logger():
    """
    Get the current logger instance.
    
    Returns:
        DWARFLogger: Current logger instance
    """
    global _LOGGER
    return _LOGGER

class FlushableWriter:
    """Wrapper for CSV writer that provides flushing capability"""
    def __init__(self, file):
        self.file = file
        self.writer = csv.writer(file)
        
    def writerow(self, row):
        return self.writer.writerow(row)
        
    def writerows(self, rows):
        return self.writer.writerows(rows)
        
    def flush(self):
        self.file.flush()

def init_loggers(log_dir='logs'):
    """
    Initialize v0.5 style loggers and return writers and file handles.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        tuple: (writers, files)
    """
    global log_writers, log_files
    
    os.makedirs(log_dir, exist_ok=True)
    writers = {}
    files = {}
    
    # Define the filenames and headers
    log_files = {
        'positions': ('positions.csv', ['step', 'id', 'x', 'y']),
        'velocities': ('velocities.csv', ['step', 'id', 'vx', 'vy']),
        'distance': ('distances.csv', ['step', 'id1', 'id2', 'distance']),
        'memory': ('memory.csv', ['step', 'id', 'memory_x', 'memory_y']),
        'curl': ('curl.csv', ['step', 'id', 'curl']),
        'angular_momentum': ('angular_momentum.csv', ['step', 'id', 'L']),
        'memory_gradient': ('memory_gradient.csv', ['step', 'id', 'grad_x', 'grad_y']),
        'memory_vector': ('memory_vector.csv', ['step', 'id', 'vec_x', 'vec_y']),
        'net_force': ('net_force.csv', ['step', 'id', 'force_x', 'force_y']),
    }
    
    for key, (fname, headers) in log_files.items():
        path = os.path.join(log_dir, fname)
        f = open(path, mode='w', newline='')
        writer = FlushableWriter(f)
        writer.writerow(headers)
        writers[key] = writer
        files[key] = f
    
    # Store in module variables
    log_writers = writers
    log_files = files
    
    # Also create the enhanced logger (with v0.5 compatibility)
    create_new_logger(log_dir)
    
    return writers, files

def log_step(step, time=None, position=None, velocity=None, energy=None):
    """
    Log a simulation step - dual logging to both systems.
    
    Args:
        step: Current simulation step
        time: Current simulation time
        position: Position vector
        velocity: Velocity vector
        energy: Energy value
    """
    # Log to enhanced logger if it exists
    logger = get_logger()
    if logger and position is not None and velocity is not None:
        logger.log_step(step, time, position, velocity, energy)

def log_particle_state(step, particles, writers=None):
    """
    Log particle state at current simulation step.
    
    Args:
        step: Simulation step
        particles: List of particle dictionaries
        writers: Dictionary of CSV writers (optional)
    """
    # Use global writers if not provided
    global log_writers
    if writers is None:
        writers = log_writers
    
    # Skip if no writers available
    if not writers:
        return
        
    for p in particles:
        pid = p.get('id', None)
        # Position
        if 'positions' in writers:
            writers['positions'].writerow([step, pid, float(p['pos'][0]), float(p['pos'][1])])
        # Velocity
        if 'velocities' in writers:
            writers['velocities'].writerow([step, pid, float(p['vel'][0]), float(p['vel'][1])])
        # Distance (e.g. electron to proton)
        if 'distance' in writers and p['type']=='electron':
            # assume distance from origin (proton at id 0 at origin)
            d = float((p['pos']**2).sum()**0.5)
            writers['distance'].writerow([step, pid, 0, d])
        # Memory
        if 'memory' in writers and 'field_memory' in p:
            writers['memory'].writerow([step, pid, float(p['field_memory'][0]), float(p['field_memory'][1])])
        # Curl
        if 'curl' in writers and 'curl' in p:
            writers['curl'].writerow([step, pid, float(p['curl'])])
        # Angular momentum
        if 'angular_momentum' in writers and 'angular_momentum' in p:
            writers['angular_momentum'].writerow([step, pid, float(p['angular_momentum'])])
        # Memory Gradient
        if 'memory_gradient' in writers and 'memory_gradient' in p:
            writers['memory_gradient'].writerow([step, pid, float(p['memory_gradient'][0]), float(p['memory_gradient'][1])])
        # Memory Vector
        if 'memory_vector' in writers and 'memory_vector' in p:
            writers['memory_vector'].writerow([step, pid, float(p['memory_vector'][0]), float(p['memory_vector'][1])])
        # Net Force
        if 'net_force' in writers and 'net_force' in p:
            writers['net_force'].writerow([step, pid, float(p['net_force'][0]), float(p['net_force'][1])])
    
    # Flush all writers to ensure data is written immediately
    for writer in writers.values():
        writer.flush()

class DWARFLogger:
    """Logger class for DWARF simulation."""
    def __init__(self, log_dir, simulation_name=None, config=None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store logs
            simulation_name: Name of the simulation
            config: SimulationConfig instance
        """
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Generate log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if simulation_name:
            self.log_base = f"{log_dir}/{simulation_name}_{timestamp}"
        else:
            self.log_base = f"{log_dir}/dwarf_sim_{timestamp}"
            
        # Initialize log files
        self.event_log = f"{self.log_base}_events.log"
        self.data_log = f"{self.log_base}_data.csv"
        self.data_fields = None
        
        print(f"Logger initialized. Logs will be saved to {self.log_base}")
        
        # Store configuration
        if config:
            self.config = config
            # Save config to file
            with open(f"{self.log_base}_config.json", 'w') as f:
                # Convert config to dict
                config_dict = self._config_to_dict(config)
                json.dump(config_dict, f, indent=2)
                
        # Initialize event log
        self.log_event("INIT", "Logger initialized")
        
    def _config_to_dict(self, config):
        """Convert config object to dictionary."""
        if config is None:
            return {}
            
        config_dict = {}
        # Get all non-private attributes
        for attr in dir(config):
            if not attr.startswith('_') and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    config_dict[attr] = value.tolist()
                else:
                    # Try to convert to simple type
                    try:
                        json.dumps(value)
                        config_dict[attr] = value
                    except (TypeError, OverflowError):
                        config_dict[attr] = str(value)
                        
        return config_dict
        
    def log_event(self, event_type, message):
        """
        Log an event to the event log.
        
        Args:
            event_type: Type of event
            message: Event message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.event_log, 'a') as f:
            f.write(f"{timestamp} | {event_type:15s} | {message}\n")
            
    def log_data(self, data_dict):
        """
        Log data to the CSV log.
        
        Args:
            data_dict: Dictionary of data to log
        """
        # Initialize data fields if not already done
        if self.data_fields is None:
            self.data_fields = sorted(data_dict.keys())
            with open(self.data_log, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data_fields)
                writer.writeheader()
                
        # Ensure all fields are present
        for field in self.data_fields:
            if field not in data_dict:
                data_dict[field] = None
                
        # Write data to CSV
        with open(self.data_log, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data_fields)
            writer.writerow(data_dict)
            
    def log_step(self, step, time, position, velocity, energy=None):
        """
        Log a simulation step.
        
        Args:
            step: Current simulation step
            time: Current simulation time
            position: Current position vector
            velocity: Current velocity vector
            energy: Current energy (optional)
        """
        data = {
            'step': step,
            'time': time if time is not None else step,
            'pos_x': position[0],
            'pos_y': position[1],
            'pos_z': position[2] if len(position) > 2 else 0.0,
            'vel_x': velocity[0],
            'vel_y': velocity[1],
            'vel_z': velocity[2] if len(velocity) > 2 else 0.0,
            'speed': np.linalg.norm(velocity),
        }
        
        if energy is not None:
            data['energy'] = energy
            
        self.log_data(data)
        
    def log_quantum_metrics(self, step, time, quantum_tracker):
        """
        Log quantum metrics from the quantum tracker.
        
        Args:
            step: Current simulation step
            time: Current simulation time
            quantum_tracker: EmergentQuantumTracker instance
        """
        # Get quantum state info
        state = quantum_tracker.get_quantum_state_info()
        
        if not state:
            return
            
        # Log key quantum metrics
        data = {
            'step': step,
            'time': time,
            'quantum_n': state.get('n', 0),
            'quantum_l': state.get('l', 0),
            'quantum_j': state.get('j', 0),
            'quantum_energy': state.get('energy', 0.0),
            'orbital_type': state.get('orbital_type', 'unknown')
        }
        
        # Add additional quantum data if available
        if 'fine_structure_shift' in state:
            data['fine_structure'] = state['fine_structure_shift']
            
        # Log spin-orbit coupling if available
        coupling_info = quantum_tracker.torsion_analyzer.analyze_spin_orbit_coupling()
        if coupling_info.get('detected', False):
            data['spin_orbit_angle'] = coupling_info.get('angle', 0.0)
            data['spin_orbit_precession'] = coupling_info.get('precession_rate', 0.0)
        
        # Write to a separate quantum metrics log
        quantum_log = f"{self.log_base}_quantum.csv"
        
        if not os.path.exists(quantum_log):
            with open(quantum_log, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(data.keys()))
                writer.writeheader()
                writer.writerow(data)
        else:
            with open(quantum_log, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(data.keys()))
                writer.writerow(data)
                
    def save_checkpoint(self, vortex_simulator, quantum_tracker, step):
        """
        Save a simulation checkpoint.
        
        Args:
            vortex_simulator: The DWARF vortex simulator
            quantum_tracker: The quantum tracker
            step: Current step number
        """
        checkpoint_dir = f"{self.log_base}_checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint_file = f"{checkpoint_dir}/checkpoint_{step}.pkl"
        
        try:
            checkpoint_data = {
                'step': step,
                'vortex_simulator_state': vortex_simulator.__dict__,
                'quantum_tracker_state': quantum_tracker.__dict__
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            self.log_event("CHECKPOINT", f"Saved checkpoint at step {step}")
        except Exception as e:
            self.log_event("ERROR", f"Failed to save checkpoint: {str(e)}")
            
    def save_statistics_report(self):
        """
        Save a statistical report of the simulation.
        """
        try:
            # Load data from CSV
            data = []
            with open(self.data_log, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
                    
            if not data:
                self.log_event("WARNING", "No data to generate statistics report")
                return
                
            # Convert string values to numeric
            for row in data:
                for key, value in row.items():
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
                        
            # Calculate basic statistics
            stats = {}
            numeric_keys = []
            
            # Find numeric columns
            if data:
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        numeric_keys.append(key)
                        
            # Calculate min, max, mean, std for numeric columns
            for key in numeric_keys:
                values = [float(row[key]) for row in data if key in row and row[key] is not None]
                if values:
                    stats[key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }
                    
            # Save statistics to file
            stats_file = f"{self.log_base}_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
            # Generate some basic plots
            self._generate_statistics_plots(data, numeric_keys)
            
            self.log_event("STATS", "Generated statistics report")
        except Exception as e:
            self.log_event("ERROR", f"Failed to generate statistics: {str(e)}")
            
    def _generate_statistics_plots(self, data, numeric_keys):
        """
        Generate basic plots from simulation data.
        
        Args:
            data: List of data dictionaries
            numeric_keys: List of numeric column names
        """
        try:
            # Create plots directory
            plots_dir = f"{self.log_base}_plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            # Time series plots for key metrics
            time_series_keys = ['energy', 'speed', 'pos_x', 'pos_y']
            if 'time' in numeric_keys:
                x = [row['time'] for row in data if 'time' in row]
                
                for key in time_series_keys:
                    if key in numeric_keys:
                        plt.figure(figsize=(10, 6))
                        y = [row[key] for row in data if key in row]
                        plt.plot(x, y)
                        plt.title(f"{key} vs Time")
                        plt.xlabel("Time")
                        plt.ylabel(key)
                        plt.grid(True)
                        plt.savefig(f"{plots_dir}/{key}_vs_time.png")
                        plt.close()
                        
            # Position plot
            if 'pos_x' in numeric_keys and 'pos_y' in numeric_keys:
                plt.figure(figsize=(10, 10))
                x = [row['pos_x'] for row in data if 'pos_x' in row]
                y = [row['pos_y'] for row in data if 'pos_y' in row]
                plt.plot(x, y)
                plt.title("Trajectory")
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.grid(True)
                plt.axis('equal')
                plt.savefig(f"{plots_dir}/trajectory.png")
                plt.close()
                
        except Exception as e:
            self.log_event("ERROR", f"Failed to generate plots: {str(e)}")

def log_physics_data(step, time, physics_data):
    """
    Log comprehensive physics data to the log file.
    
    Args:
        step: Current simulation step
        time: Current simulation time
        physics_data: Dictionary of physics data
    """
    logger = get_logger()
    if not logger:
        return
        
    # Log basic step info
    log_data = {
        'step': step,
        'time': time,
    }
    
    # Add particle data
    if 'particles' in physics_data:
        for particle in physics_data['particles']:
            p_type = particle['type']
            p_id = particle.get('id', 0)
            p_key = f"{p_type}_{p_id}"
            
            # Position and velocity
            if 'pos' in particle:
                log_data[f"{p_key}_pos_x"] = particle['pos'][0]
                log_data[f"{p_key}_pos_y"] = particle['pos'][1]
            
            if 'vel' in particle:
                log_data[f"{p_key}_vel_x"] = particle['vel'][0]
                log_data[f"{p_key}_vel_y"] = particle['vel'][1]
                log_data[f"{p_key}_speed"] = np.linalg.norm(particle['vel'])
            
            # Physics metrics
            if 'angular_momentum' in particle:
                log_data[f"{p_key}_ang_momentum"] = particle['angular_momentum']
            
            if 'curl' in particle:
                log_data[f"{p_key}_curl"] = particle['curl']
                
            if 'spin' in particle:
                log_data[f"{p_key}_spin"] = particle['spin']
            
            if 'field_memory' in particle:
                log_data[f"{p_key}_memory_x"] = particle['field_memory'][0]
                log_data[f"{p_key}_memory_y"] = particle['field_memory'][1]
                log_data[f"{p_key}_memory_mag"] = np.linalg.norm(particle['field_memory'])
            
            if 'memory_gradient' in particle:
                log_data[f"{p_key}_memory_grad_x"] = particle['memory_gradient'][0]
                log_data[f"{p_key}_memory_grad_y"] = particle['memory_gradient'][1]
                log_data[f"{p_key}_memory_grad_mag"] = np.linalg.norm(particle['memory_gradient'])
            
            if 'memory_vector' in particle:
                log_data[f"{p_key}_memory_vec_x"] = particle['memory_vector'][0]
                log_data[f"{p_key}_memory_vec_y"] = particle['memory_vector'][1]
                log_data[f"{p_key}_memory_vec_mag"] = np.linalg.norm(particle['memory_vector'])
            
            if 'net_force' in particle:
                log_data[f"{p_key}_force_x"] = particle['net_force'][0]
                log_data[f"{p_key}_force_y"] = particle['net_force'][1]
                log_data[f"{p_key}_force_mag"] = np.linalg.norm(particle['net_force'])
    
    # Add distance data
    if 'distances' in physics_data:
        for key, distance in physics_data['distances'].items():
            log_data[f"distance_{key}"] = distance
    
    # Write to CSV - use physics-specific log file
    physics_log = f"{logger.log_base}_physics.csv"
    
    # Check if file exists, if not create header
    if not os.path.exists(physics_log):
        with open(physics_log, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(log_data.keys()))
            writer.writeheader()
            writer.writerow(log_data)
    else:
        with open(physics_log, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(log_data.keys()))
            writer.writerow(log_data)
            
    # Also log to flat files for compatibility
    global log_writers
    if log_writers is not None:
        log_particle_state(step, physics_data['particles'], log_writers)