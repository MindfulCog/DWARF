"""
Configuration setup script for DWARF Simulator.
Updates the config.py file with user-specified parameters.
"""
import os
import sys
import re
from datetime import datetime

def prompt_for_value(name, default_value, description=None, value_type=None):
    """
    Prompt the user for a value with a default option.
    
    Args:
        name: Name of the parameter
        default_value: Default value to use
        description: Optional description of the parameter
        value_type: Type of the value (int, float, etc.)
        
    Returns:
        The user-provided or default value
    """
    prompt_str = f"{name}"
    if description:
        prompt_str += f" ({description})"
    prompt_str += f" [{default_value}]: "
    
    response = input(prompt_str).strip()
    
    if not response:
        return default_value
        
    try:
        if value_type:
            return value_type(response)
        else:
            # Try to infer the type from the default value
            if isinstance(default_value, int):
                return int(response)
            elif isinstance(default_value, float):
                return float(response)
            else:
                return response
    except ValueError:
        print(f"Invalid input, using default: {default_value}")
        return default_value

def read_current_config(file_path="config.py"):
    """
    Read the current configuration from file.
    
    Args:
        file_path: Path to the config file
        
    Returns:
        dict: Current configuration values
    """
    config = {}
    
    if not os.path.exists(file_path):
        return config
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Extract class attributes
    class_match = re.search(r'class SimulationConfig.*?:(.*?)(?:class|\Z)', content, re.DOTALL)
    if not class_match:
        return config
        
    class_content = class_match.group(1)
    
    # Extract individual attributes
    # Look for self.attribute = value
    attr_pattern = re.compile(r'self\.(\w+)\s*=\s*([^#\n]+)')
    for match in attr_pattern.finditer(class_content):
        attr_name = match.group(1)
        attr_value = match.group(2).strip()
        
        try:
            # Handle different value types
            if attr_value.startswith('[') and attr_value.endswith(']'):
                # List value
                list_value = eval(attr_value)
                config[attr_name] = list_value
            elif attr_value.startswith('{') and attr_value.endswith('}'):
                # Dict value
                dict_value = eval(attr_value)
                config[attr_name] = dict_value
            else:
                # Try direct evaluation
                config[attr_name] = eval(attr_value)
        except (SyntaxError, NameError):
            # If eval fails, keep as string
            config[attr_name] = attr_value
    
    # Also check for constants at module level
    constants_pattern = re.compile(r'^(\w+)\s*=\s*([^#\n]+)', re.MULTILINE)
    for match in constants_pattern.finditer(content):
        const_name = match.group(1)
        if const_name != 'SimulationConfig' and not const_name.startswith('__'):
            const_value = match.group(2).strip()
            
            try:
                # Handle different value types
                config[const_name] = eval(const_value)
            except (SyntaxError, NameError):
                # If eval fails, keep as string
                config[const_name] = const_value
    
    return config

def write_updated_config(config, file_path="config.py"):
    """
    Write the updated configuration to file.
    
    Args:
        config: Dictionary of configuration values
        file_path: Path to write the config file
    """
    # Backup the existing file if it exists
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    # Map some class attributes to global constants that physics_core expects
    constants = {
        'GRID_SIZE': config.get('GRID_SIZE', 512),
        # Map time_step to DT constant that physics_core expects
        'DT': config.get('time_step', 1e-3),
        # Map global_drag to GLOBAL_DRAG constant
        'GLOBAL_DRAG': config.get('global_drag', 0.990),
        # Map saturation_limit to SATURATION_LIMIT constant
        'SATURATION_LIMIT': config.get('saturation_limit', 5.0),
        # Add missing DWARF_FORCE_EXPONENT constant if needed
        'DWARF_FORCE_EXPONENT': config.get('force_exponent', 2.22),
        # Add any other constants physics_core.py expects
        'MEMORY_DECAY': config.get('memory_decay', 0.995)
    }
    
    # Extract the class attributes (excluding those already as constants)
    class_attrs = {}
    for key, value in config.items():
        if key not in constants and not key.isupper():
            class_attrs[key] = value
    
    # Generate the new config file content
    content = [
        "\"\"\"",
        "Simulation configuration for DWARF Simulator.",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\"\"\"",
        "import numpy as np",
        ""
    ]
    
    # Add constants
    for key, value in constants.items():
        content.append(f"{key} = {repr(value)}")
    
    content.append("")
    content.append("class SimulationConfig:")
    content.append("    \"\"\"")
    content.append("    Configuration parameters for the DWARF simulation.")
    content.append("    \"\"\"")
    content.append("    def __init__(self):")
    
    # Add class attributes, also duplicating the constants inside the class
    # This ensures both import styles work
    for key, value in constants.items():
        # Convert constant name to lowercase for class attribute if not already
        class_key = key.lower() if key.isupper() else key
        content.append(f"        self.{class_key} = {repr(value)}")
    
    # Add remaining class attributes
    for key, value in class_attrs.items():
        content.append(f"        self.{key} = {repr(value)}")
    
    # Write to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"Updated configuration saved to {file_path}")
def setup_config():
    """
    Interactive setup for simulation configuration.
    """
    print("DWARF Simulator Configuration Setup")
    print("==================================")
    print("Press Enter to keep the current value, or type a new value.")
    
    # Read current configuration
    current_config = read_current_config()
    
    # Set up defaults for missing parameters
    defaults = {
        'GRID_SIZE': 2048,
        'time_step': 0.002,
        'max_steps': 10000,
        'viz_interval': 10,
        'global_drag': 0.001,
        'saturation_limit': 10000000,
        'memory_decay': 0.999,
        'num_protons': 1,
        'num_electrons': 1,
        'num_neutrons': 0,
        'proton_spin': 1000000,
        'electron_spin': -1000000,
        'neutron_spin': 0,
        'initial_electron_position': [1024.0, 1024.0],  # Center of grid by default
        'initial_electron_velocity': [0.0, 5.0],
        'initial_proton_position': [1024.0, 1024.0],    # Center of grid by default
    }
    
    # Use existing values if available, otherwise use defaults
    config = {}
    for key, default in defaults.items():
        config[key] = current_config.get(key, default)
    
    # Add any other values from current config
    for key, value in current_config.items():
        if key not in config:
            config[key] = value
    
    # Prompt for each parameter
    print("\nBasic Simulation Parameters:")
    config['GRID_SIZE'] = prompt_for_value("Grid Size", config['GRID_SIZE'], 
                                       "Size of simulation grid", int)
    config['max_steps'] = prompt_for_value("Total Steps", config['max_steps'], 
                                       "Maximum simulation steps", int)
    config['time_step'] = prompt_for_value("Time Step (DT)", config['time_step'], 
                                       "Simulation time increment", float)
    config['viz_interval'] = prompt_for_value("Visualization Interval", config['viz_interval'],
                                         "Steps between visualization updates", int)
    
    print("\nPhysics Parameters:")
    config['global_drag'] = prompt_for_value("Global Drag Force", config['global_drag'], 
                                       "Dampening factor", float)
    config['saturation_limit'] = prompt_for_value("Saturation Limit", config['saturation_limit'], 
                                       "Maximum field memory value", float)
    config['memory_decay'] = prompt_for_value("Memory Decay", config['memory_decay'], 
                                       "Rate of field memory decay (0-1)", float)
    
    print("\nParticle Configuration:")
    # Protons
    config['num_protons'] = prompt_for_value("Number of Protons", config['num_protons'], 
                                       "Number of protons in simulation", int)
    if config['num_protons'] > 0:
        config['proton_spin'] = prompt_for_value("Proton Spin", config['proton_spin'], 
                                       "Spin value for protons", float)
        
        # Initial proton position - default to center of grid
        center = config['GRID_SIZE'] / 2
        print("\nProton Initial Position:")
        x = prompt_for_value("  X", config.get('initial_proton_position', [center, center])[0], 
                           "X coordinate", float)
        y = prompt_for_value("  Y", config.get('initial_proton_position', [center, center])[1], 
                           "Y coordinate", float)
        config['initial_proton_position'] = [x, y]
    
    # Electrons
    config['num_electrons'] = prompt_for_value("Number of Electrons", config['num_electrons'], 
                                       "Number of electrons in simulation", int)
    if config['num_electrons'] > 0:
        config['electron_spin'] = prompt_for_value("Electron Spin", config['electron_spin'], 
                                       "Spin value for electrons", float)
                                       
        # Initial electron position and velocity
        center = config['GRID_SIZE'] / 2
        print("\nElectron Initial Position:")
        x = prompt_for_value("  X", config.get('initial_electron_position', [center, center])[0], 
                           "X coordinate", float)
        y = prompt_for_value("  Y", config.get('initial_electron_position', [center, center])[1], 
                           "Y coordinate", float)
        config['initial_electron_position'] = [x, y]
        
        print("\nElectron Initial Velocity:")
        vx = prompt_for_value("  Vx", config.get('initial_electron_velocity', [0.0, 5.0])[0], 
                            "X velocity", float)
        vy = prompt_for_value("  Vy", config.get('initial_electron_velocity', [0.0, 5.0])[1], 
                            "Y velocity", float)
        config['initial_electron_velocity'] = [vx, vy]
    
    # Neutrons
    config['num_neutrons'] = prompt_for_value("Number of Neutrons", config['num_neutrons'], 
                                       "Number of neutrons in simulation", int)
    if config['num_neutrons'] > 0:
        config['neutron_spin'] = prompt_for_value("Neutron Spin", config['neutron_spin'], 
                                       "Spin value for neutrons", float)
    
    # Write the updated configuration
    write_updated_config(config)
    
    print("\nConfiguration setup complete!")
    print("You can now run the simulation with: python main.py")

if __name__ == "__main__":
    setup_config()