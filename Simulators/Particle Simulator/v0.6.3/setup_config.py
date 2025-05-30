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
    
    # Evaluate and normalize all values to avoid string references
    # Get actual numeric values for all constants
    constants = {
        'GRID_SIZE': float(config.get('grid_size', config.get('GRID_SIZE', 2048))),
        'DT': float(config.get('time_step', config.get('dt', config.get('DT', 0.001)))),
        'DWARF_FORCE_EXPONENT': float(config.get('force_exponent', config.get('dwarf_force_exponent', 2.0))),
        'GLOBAL_DRAG': float(config.get('global_drag', 0.985)),
        'DRAG_RATE': float(config.get('global_drag', 0.985)),  # Same as GLOBAL_DRAG
        'SATURATION_LIMIT': float(config.get('saturation_limit', 15.0)),
        'MEMORY_DECAY': float(config.get('memory_decay', 0.9998)),
        'BOUNDARY_MODE': 'wrap',  # String needs quotes
        'BOUNDARY_WRAP': 'wrap',  # String needs quotes
        'BOUNDARY_DAMP': 'damp',  # String needs quotes
        'VORTEX_FIELD_ENABLED': True,  # Boolean
        'STABLE_ORBIT_WIDTH': float(config.get('stable_orbit_width', 0.15)),
        'SPIN_ORBIT_SCALE': float(config.get('spin_orbit_scale', 0.001))
    }
    
    # Remove duplicates and ensure consistent keys
    class_attrs = {
        'grid_size': constants['GRID_SIZE'],
        'time_step': constants['DT'],
        'force_exponent': constants['DWARF_FORCE_EXPONENT'],
        'global_drag': constants['GLOBAL_DRAG'],
        'drag_rate': constants['DRAG_RATE'],
        'saturation_limit': constants['SATURATION_LIMIT'],
        'memory_decay': constants['MEMORY_DECAY'],
        'boundary_mode': constants['BOUNDARY_MODE'],
        'vortex_field_enabled': constants['VORTEX_FIELD_ENABLED'],
        'stable_orbit_width': constants['STABLE_ORBIT_WIDTH'],
        'spin_orbit_scale': constants['SPIN_ORBIT_SCALE'],
    }
    
    # Add other class attributes from config
    for key, value in config.items():
        if key.lower() not in [k.lower() for k in class_attrs.keys()] and not key.isupper():
            # Make sure we have actual values, not strings referring to other values
            if isinstance(value, str) and ('float(' in value or value.isupper()):
                # Try to evaluate expressions like 'float(X)' or 'CONSTANT'
                try:
                    # For safety, only evaluate simple expressions
                    if value.startswith('float(') and value.endswith(')'):
                        inner_value = value[6:-1]  # Extract value between float( and )
                        if inner_value in constants:
                            value = float(constants[inner_value])
                        else:
                            value = float(config.get(inner_value.lower(), 0.0))
                    elif value in constants:
                        value = constants[value]
                except (ValueError, TypeError, KeyError):
                    pass
            class_attrs[key] = value
    
    # Generate the new config file content
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content = [
        "\"\"\"",
        f"Simulation configuration for DWARF Simulator.",
        f"Generated on {timestamp}",
        "\"\"\"",
        "import numpy as np",
        ""
    ]
    
    # Add constants
    for key, value in constants.items():
        # Format the value correctly based on type
        if isinstance(value, str):
            content.append(f"{key} = '{value}'")
        elif isinstance(value, bool):
            content.append(f"{key} = {str(value)}")
        else:
            content.append(f"{key} = {value}")
    
    content.append("")
    content.append("class SimulationConfig:")
    content.append("    \"\"\"")
    content.append("    Configuration parameters for the DWARF simulation.")
    content.append("    \"\"\"")
    content.append("    def __init__(self):")
    
    # Add class attributes
    for key, value in class_attrs.items():
        # Format the value correctly based on type
        if isinstance(value, str):
            content.append(f"        self.{key} = '{value}'")
        elif isinstance(value, bool):
            content.append(f"        self.{key} = {str(value)}")
        elif isinstance(value, (list, tuple)):
            content.append(f"        self.{key} = {value}")
        else:
            content.append(f"        self.{key} = {value}")
    
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