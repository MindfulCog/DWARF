import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import json
from typing import Dict, List, Tuple, Optional, Union
import sys

def load_energy_data(file_path: str) -> Dict[str, List[float]]:
    """Load energy data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate required keys
        required_keys = ['kinetic_energy', 'potential_energy', 'total_energy']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing required keys in data file. Expected: {required_keys}")
        
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading energy data: {e}")
        sys.exit(1)

def plot_energy_data(data: Dict[str, List[float]], output_path: Optional[str] = None, 
                     show_plot: bool = True, plot_title: str = "Energy Conservation") -> None:
    """Plot energy data and optionally save to file."""
    # Validate data lengths
    data_lengths = [len(data[key]) for key in ['kinetic_energy', 'potential_energy', 'total_energy']]
    if len(set(data_lengths)) > 1:
        print(f"Warning: Energy data arrays have different lengths: {data_lengths}")
    
    # Create figure with a reasonable size
    plt.figure(figsize=(12, 8))
    
    # Plot time steps
    time_steps = range(min(data_lengths))
    
    # Plot energy values
    plt.plot(time_steps, data['kinetic_energy'][:len(time_steps)], 'b-', linewidth=2, label='Kinetic Energy')
    plt.plot(time_steps, data['potential_energy'][:len(time_steps)], 'r-', linewidth=2, label='Potential Energy')
    plt.plot(time_steps, data['total_energy'][:len(time_steps)], 'g-', linewidth=2, label='Total Energy')
    
    # Calculate energy conservation metrics
    if len(data['total_energy']) > 1:
        initial_energy = data['total_energy'][0]
        final_energy = data['total_energy'][-1]
        max_energy = max(data['total_energy'])
        min_energy = min(data['total_energy'])
        energy_drift = (max_energy - min_energy) / initial_energy * 100 if initial_energy != 0 else float('inf')
        
        # Add energy conservation info to plot
        conservation_text = (
            f"Initial Energy: {initial_energy:.3e}\n"
            f"Final Energy: {final_energy:.3e}\n"
            f"Change: {(final_energy - initial_energy):.3e}\n"
            f"Drift: {energy_drift:.2f}%"
        )
        plt.annotate(conservation_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    verticalalignment='top')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(plot_title, fontsize=16)
    plt.legend(fontsize=12)
    
    # Improve appearance
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def analyze_energy_conservation(data: Dict[str, List[float]]) -> Dict[str, float]:
    """Analyze energy conservation metrics in the simulation."""
    total_energy = data['total_energy']
    
    if not total_energy:
        return {"error": "No energy data available"}
    
    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    
    # Calculate metrics
    metrics = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "absolute_change": final_energy - initial_energy,
        "relative_change": (final_energy - initial_energy) / initial_energy if initial_energy != 0 else float('inf'),
        "max_energy": max(total_energy),
        "min_energy": min(total_energy),
        "max_fluctuation": (max(total_energy) - min(total_energy)) / initial_energy if initial_energy != 0 else float('inf')
    }
    
    return metrics

def print_energy_analysis(metrics: Dict[str, float]) -> None:
    """Print energy conservation analysis in a formatted way."""
    if "error" in metrics:
        print(f"Analysis Error: {metrics['error']}")
        return
    
    print("\n" + "="*50)
    print(" ENERGY CONSERVATION ANALYSIS ")
    print("="*50)
    
    print(f"Initial Energy: {metrics['initial_energy']:.6e}")
    print(f"Final Energy:   {metrics['final_energy']:.6e}")
    print(f"Absolute Change: {metrics['absolute_change']:.6e}")
    print(f"Relative Change: {metrics['relative_change']:.6%}")
    print("-"*50)
    print(f"Maximum Energy: {metrics['max_energy']:.6e}")
    print(f"Minimum Energy: {metrics['min_energy']:.6e}")
    print(f"Maximum Fluctuation: {metrics['max_fluctuation']:.6%}")
    
    # Evaluate conservation quality
    if metrics['max_fluctuation'] < 0.001:
        quality = "Excellent"
    elif metrics['max_fluctuation'] < 0.01:
        quality = "Good"
    elif metrics['max_fluctuation'] < 0.05:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print("-"*50)
    print(f"Conservation Quality: {quality}")
    print("="*50 + "\n")

def batch_process_energy_files(directory: str, output_dir: Optional[str] = None) -> None:
    """Process all energy data files in a directory."""
    energy_files = glob.glob(os.path.join(directory, "*energy*.json"))
    
    if not energy_files:
        print(f"No energy data files found in {directory}")
        return
    
    print(f"Found {len(energy_files)} energy data files.")
    
    for file_path in energy_files:
        try:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            # Load data
            data = load_energy_data(file_path)
            
            # Generate output path if needed
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
            else:
                output_path = None
            
            # Plot data
            plot_title = f"Energy Conservation - {file_name}"
            plot_energy_data(data, output_path, show_plot=False, plot_title=plot_title)
            
            # Analyze and print metrics
            metrics = analyze_energy_conservation(data)
            print_energy_analysis(metrics)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze energy conservation in particle simulations")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single file parser
    file_parser = subparsers.add_parser("file", help="Process a single energy data file")
    file_parser.add_argument("file_path", type=str, help="Path to the energy data JSON file")
    file_parser.add_argument("-o", "--output", type=str, help="Output path for the plot image")
    file_parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    
    # Batch processing parser
    batch_parser = subparsers.add_parser("batch", help="Process all energy files in a directory")
    batch_parser.add_argument("directory", type=str, help="Directory containing energy data files")
    batch_parser.add_argument("-o", "--output-dir", type=str, help="Output directory for plot images")
    
    args = parser.parse_args()
    
    if args.command == "file":
        # Process single file
        data = load_energy_data(args.file_path)
        
        # Plot energy data
        plot_energy_data(data, args.output, show_plot=not args.no_show)
        
        # Analyze and print energy conservation metrics
        metrics = analyze_energy_conservation(data)
        print_energy_analysis(metrics)
        
    elif args.command == "batch":
        # Batch process files
        batch_process_energy_files(args.directory, args.output_dir)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()