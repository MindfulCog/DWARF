import argparse
import json
import sys
import os
import matplotlib.pyplot as plt
import time

from dwarf_vortex_simulator import DwarfVortexSimulator
from vortex_control_panel import VortexControlPanel

def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def default_config():
    """Create a default configuration."""
    return {
        'width': 800,
        'height': 600,
        'dt': 0.01,
        'memory_decay': 0.995,
        'field_resolution': 30,
        'saturation_limit': 5.0,
        'global_drag': 0.001,
        'use_multiprocessing': True,
        'track_energy': True,
        'particles': [
            {
                'type': 'proton',
                'x': 300,
                'y': 300,
                'vx': 0,
                'vy': 0.5,
                'mass': 1.0,
                'spin': 0.8
            },
            {
                'type': 'electron',
                'x': 500,
                'y': 300,
                'vx': 0,
                'vy': -5.0,
                'mass': 0.1,
                'spin': -0.8
            }
        ]
    }

def create_scenario(scenario_name):
    """Create a configuration for a specific scenario."""
    if scenario_name == 'proton_electron':
        return {
            'width': 800,
            'height': 600,
            'dt': 0.01,
            'memory_decay': 0.995,
            'field_resolution': 30,
            'saturation_limit': 5.0,
            'global_drag': 0.001,
            'use_multiprocessing': True,
            'track_energy': True,
            'particles': [
                {
                    'type': 'proton',
                    'x': 300,
                    'y': 300,
                    'vx': 0,
                    'vy': 0.5,
                    'mass': 1.0,
                    'spin': 0.8
                },
                {
                    'type': 'electron',
                    'x': 500,
                    'y': 300,
                    'vx': 0,
                    'vy': -5.0,
                    'mass': 0.1,
                    'spin': -0.8
                }
            ]
        }
    elif scenario_name == 'neutron_field':
        return {
            'width': 800,
            'height': 600,
            'dt': 0.01,
            'memory_decay': 0.997,
            'field_resolution': 40,
            'saturation_limit': 10.0,
            'global_drag': 0.0005,
            'use_multiprocessing': True,
            'track_energy': True,
            'particles': [
                {
                    'type': 'neutron',
                    'x': 400,
                    'y': 300,
                    'vx': 0,
                    'vy': 0,
                    'mass': 1.0,
                    'spin': 0.1
                },
                {
                    'type': 'proton',
                    'x': 300,
                    'y': 300,
                    'vx': 0,
                    'vy': 0.8,
                    'mass': 1.0,
                    'spin': 0.8
                },
                {
                    'type': 'electron',
                    'x': 500,
                    'y': 300,
                    'vx': 0,
                    'vy': -8.0,
                    'mass': 0.1,
                    'spin': -0.8
                }
            ]
        }
    elif scenario_name == 'dual_vortex':
        return {
            'width': 800,
            'height': 600,
            'dt': 0.01,
            'memory_decay': 0.998,
            'field_resolution': 40,
            'saturation_limit': 8.0,
            'global_drag': 0.0001,
            'use_multiprocessing': True,
            'track_energy': True,
            'particles': [
                {
                    'type': 'proton',
                    'x': 300,
                    'y': 200,
                    'vx': 0.5,
                    'vy': 0,
                    'mass': 1.0,
                    'spin': 0.9
                },
                {
                    'type': 'electron',
                    'x': 500,
                    'y': 200,
                    'vx': -5.0,
                    'vy': 0,
                    'mass': 0.1,
                    'spin': -0.9
                },
                {
                    'type': 'proton',
                    'x': 300,
                    'y': 400,
                    'vx': 0.5,
                    'vy': 0,
                    'mass': 1.0,
                    'spin': 0.9
                },
                {
                    'type': 'electron',
                    'x': 500,
                    'y': 400,
                    'vx': -5.0,
                    'vy': 0,
                    'mass': 0.1,
                    'spin': -0.9
                }
            ]
        }
    elif scenario_name == 'vortex_field':
        return {
            'width': 800,
            'height': 600,
            'dt': 0.01,
            'memory_decay': 0.999,
            'field_resolution': 40,
            'saturation_limit': 8.0,
            'global_drag': 0.0001,
            'use_multiprocessing': True,
            'track_energy': True,
            'particles': [
                {
                    'type': 'proton',
                    'x': 250,
                    'y': 300,
                    'vx': 0,
                    'vy': 0.5,
                    'mass': 1.0,
                    'spin': 1.0
                },
                {
                    'type': 'proton',
                    'x': 550,
                    'y': 300,
                    'vx': 0,
                    'vy': -0.5,
                    'mass': 1.0,
                    'spin': -1.0
                },
                {
                    'type': 'electron',
                    'x': 400,
                    'y': 300,
                    'vx': 0,
                    'vy': 0,
                    'mass': 0.1,
                    'spin': 0.0
                }
            ]
        }
    else:
        return default_config()

def main():
    """Main entry point for the DWARF vortex simulator."""
    parser = argparse.ArgumentParser(description='DWARF Vortex Simulator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--scenario', type=str, 
                      choices=['proton_electron', 'neutron_field', 'dual_vortex', 'vortex_field'],
                      help='Predefined scenario to run')
    parser.add_argument('--no-ui', action='store_true', help='Run without UI controls')
    parser.add_argument('--record', action='store_true', help='Record frames to output directory')
    parser.add_argument('--output', type=str, default='frames', help='Output directory for recordings')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.scenario:
        config = create_scenario(args.scenario)
    else:
        config = default_config()
    
    # Add recording settings if specified
    if args.record:
        config['record'] = True
        config['record_dir'] = args.output
    
    # Create simulator
    start_time = time.time()
    print("Initializing DWARF Vortex Simulator...")
    simulator = DwarfVortexSimulator(config)
    
    # Initialize visualization
    simulator.start_visualization()
    
    # Create control panel unless --no-ui is specified
    if not args.no_ui:
        control_panel = VortexControlPanel(simulator, simulator.fig)
    
    print(f"Initialization complete in {time.time() - start_time:.2f} seconds")
    print("Starting simulation...")
    print("Controls:")
    print("  P - Pause/resume simulation")
    print("  F - Toggle field visibility")
    print("  R - Reset simulation")
    print("  S - Save screenshot")
    print("  E - Toggle energy tracking")
    print("  +/- - Adjust time step")
    print("  Q - Quit")
    
    # Show the plot (this will block until the window is closed)
    plt.show()

if __name__ == '__main__':
    main()