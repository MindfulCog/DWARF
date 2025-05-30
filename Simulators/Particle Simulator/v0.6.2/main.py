"""
Main module for DWARF simulator with emergent quantum effects tracking.
This version integrates quantum-like analysis while maintaining DWARF vortex physics primacy.

Date: 2025-05-30
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Import DWARF simulator wrapper
from dwarf_vortex_simulator import DwarfVortexSimulator
from config import SimulationConfig

# Import emergent quantum analyzers
from vortex_resonance_analyzer import VortexResonanceAnalyzer
from emergent_probability_tracker import EmergentProbabilityTracker
from vortex_torsion_analyzer import VortexTorsionAnalyzer
from quantum_dwarf_integration import EmergentQuantumTracker

# Import vortex control panel
from vortex_control_panel import VortexControlPanel

# Import logger
from logger import create_new_logger, get_logger

# Standard colors for consistent visualization across all components
PARTICLE_COLORS = {
    'proton': 'red',
    'electron': 'blue',
    'neutron': 'gray'
}

def setup_visualization():
    """
    Set up visualization environment with figure and subplots.
    
    Returns:
        tuple: (fig, axes_dict) containing the figure and dictionary of axes
    """
    plt.ion()  # Enable interactive mode
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplot layout
    axes = {}
    
    # Main DWARF vortex field view
    axes['vortex'] = fig.add_subplot(221, projection='3d')
    axes['vortex'].set_title('DWARF Vortex Field')
    axes['vortex'].set_xlabel('X')
    axes['vortex'].set_ylabel('Y')
    axes['vortex'].set_zlabel('Z')
    
    # Energy history view
    axes['energy'] = fig.add_subplot(222)
    axes['energy'].set_title('Energy Evolution')
    axes['energy'].set_xlabel('Time')
    axes['energy'].set_ylabel('Energy')
    
    # Emergent probability view
    axes['probability'] = fig.add_subplot(223, projection='3d')
    axes['probability'].set_title('Emergent Probability Distribution')
    axes['probability'].set_xlabel('X')
    axes['probability'].set_ylabel('Y')
    axes['probability'].set_zlabel('Z')
    
    # Quantum state info view
    axes['info'] = fig.add_subplot(224)
    axes['info'].set_title('Emergent Quantum State')
    axes['info'].axis('off')  # No axes for text display
    
    plt.tight_layout()
    
    return fig, axes

def update_visualization(fig, axes, vortex_simulator, quantum_tracker, 
                         trajectory, energy_history, time_history, vortex_panel=None):
    """
    Update all visualization components.
    
    Args:
        fig: Figure object
        axes: Dictionary of axes
        vortex_simulator: The DWARF vortex simulator
        quantum_tracker: The emergent quantum tracker
        trajectory: List of electron positions
        energy_history: List of energy values
        time_history: List of time values
        vortex_panel: Optional vortex control panel
    """
    # Update vortex field visualization
    vortex_ax = axes['vortex']
    vortex_ax.clear()
    
    # Plot trajectory
    trajectory_array = np.array(trajectory)
    vortex_ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                  c=PARTICLE_COLORS['electron'], alpha=0.5)
    
    # Plot current position
    current_pos = trajectory[-1]
    vortex_ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                     c=PARTICLE_COLORS['electron'], s=50)
    
    # Plot proton at origin - use consistent colors
    vortex_ax.scatter([0], [0], [0], c=PARTICLE_COLORS['proton'], s=100)
    
    # Optional: Add vortex field visualization here
    # This would depend on your specific vortex_simulator implementation
    
    # Set consistent scale
    max_range = max(1.0, np.max(np.abs(trajectory_array)) * 1.1)
    vortex_ax.set_xlim([-max_range, max_range])
    vortex_ax.set_ylim([-max_range, max_range])
    vortex_ax.set_zlim([-max_range, max_range])
    
    vortex_ax.set_title('DWARF Vortex Field')
    
    # Update energy history
    energy_ax = axes['energy']
    energy_ax.clear()
    
    energy_ax.plot(time_history, energy_history, 'g-')
    
    # Highlight resonance regions if detected
    resonance_info = quantum_tracker.resonance_analyzer.analyze_resonances(min_duration=20)
    if resonance_info["resonances_detected"]:
        for res in resonance_info["resonances"]:
            energy_ax.axhspan(res["energy"] - 0.05, res["energy"] + 0.05, 
                           alpha=0.2, color='yellow')
            energy_ax.text(res["end_time"], res["energy"], 
                        f"n~{res['effective_n']:.1f}", fontsize=9)
    
    energy_ax.set_title('Energy Evolution')
    energy_ax.set_xlabel('Time')
    energy_ax.set_ylabel('Energy')
    energy_ax.grid(True, alpha=0.3)
    
    # Update probability distribution
    prob_ax = axes['probability']
    prob_ax.clear()
    
    # Only try to visualize probability if we have enough data
    if len(trajectory) > 100:
        # Get grid and density from probability tracker
        grid, density = quantum_tracker.probability_tracker.calculate_probability_density()
        
        if grid is not None and density is not None:
            X, Y, Z = grid
            
            # Find indices where density is above threshold for visualization
            threshold = np.max(density) * 0.1
            high_density = density > threshold
            
            # Extract points with significant probability
            points = []
            densities = []
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        if density[i, j, k] > threshold:
                            points.append([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                            densities.append(density[i, j, k])
            
            if points:
                points = np.array(points)
                densities = np.array(densities)
                
                # Normalize densities for coloring
                densities = densities / np.max(densities)
                
                # Plot probability clouds as points with electron's color
                prob_ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=densities, cmap='Blues', alpha=0.3, s=10*densities)
    
    # Plot proton at origin - use consistent colors
    prob_ax.scatter([0], [0], [0], c=PARTICLE_COLORS['proton'], s=100)
    
    # Set consistent scale
    prob_ax.set_xlim([-max_range, max_range])
    prob_ax.set_ylim([-max_range, max_range])
    prob_ax.set_zlim([-max_range, max_range])
    
    # Get orbital info for title
    orbital_info = quantum_tracker.probability_tracker.detect_orbital_characteristics()
    orbital_type = orbital_info["orbital_type"] if orbital_info["detected"] else "unknown"
    
    prob_ax.set_title(f'Emergent {orbital_type} Probability Distribution')
    
    # Add legend to probability plot
    prob_ax.scatter([], [], [], c=PARTICLE_COLORS['proton'], s=100, label='Proton')
    prob_ax.scatter([], [], [], c=PARTICLE_COLORS['electron'], s=50, label='Electron Probability')
    prob_ax.legend()
    
    # Update quantum state info
    info_ax = axes['info']
    info_ax.clear()
    info_ax.axis('off')
    
    # Get quantum state info
    state = quantum_tracker.get_quantum_state_info()
    
    # Display information as text
    if state:
        info_text = (
            f"Emergent Quantum-Like State\n"
            f"==========================\n"
            f"Principal quantum number (n) ~ {state['n']}\n"
            f"Angular momentum (l) ~ {state['l']}\n"
            f"Total angular momentum (j) ~ {state.get('j', 'N/A')}\n"
            f"Orbital type: {state['orbital_type']}\n\n"
            f"Energy: {state['energy']:.6f} eV\n"
        )
        
        # Add fine structure info if available
        if 'fine_structure_shift' in state and abs(state['fine_structure_shift']) > 0:
            info_text += (
                f"Fine structure shift: {state['fine_structure_shift']:.8f} eV\n"
                f"Resonance energy: {state['resonance_energy']:.6f} eV\n"
            )
        
        # Add relativistic effects info
        coupling_info = quantum_tracker.torsion_analyzer.analyze_spin_orbit_coupling()
        if coupling_info["detected"]:
            info_text += (
                f"\nEmergent Relativistic Effects\n"
                f"===========================\n"
                f"Spin-orbit angle: {coupling_info['angle']:.1f} degrees\n"
                f"Coupling type: {coupling_info['coupling_type']}\n"
                f"Precession rate: {coupling_info['precession_rate']:.6f} rad/time\n"
            )
            
        # Add vortex field statistics
        physics_data = vortex_simulator.get_vortex_field_data()
        if physics_data and 'particles' in physics_data:
            info_text += (
                f"\nDWARF Core Physics\n"
                f"=================\n"
            )
            
            # Add electron info
            for particle in physics_data['particles']:
                if particle['type'] == 'electron':
                    if 'angular_momentum' in particle:
                        info_text += f"Angular momentum: {particle['angular_momentum']:.4f}\n"
                    if 'curl' in particle:
                        info_text += f"Curl: {particle['curl']:.4e}\n"
                    
                    # Show memory stats
                    if 'field_memory' in particle:
                        memory_mag = np.linalg.norm(particle['field_memory'])
                        info_text += f"Memory magnitude: {memory_mag:.4e}\n"
            
            # Add distance info
            if 'distances' in physics_data:
                for key, distance in physics_data['distances'].items():
                    if 'proton' in key and 'electron' in key:
                        info_text += f"P-E distance: {distance:.4f}\n"
            
        info_ax.text(0.02, 0.98, info_text, fontsize=10, va='top', 
                   family='monospace', transform=info_ax.transAxes)
    
    plt.draw()
    plt.pause(0.01)
    
    # Update vortex control panel if provided
    if vortex_panel:
        vortex_panel.update(trajectory)

def save_visualization(fig, frame_count, output_dir="output"):
    """
    Save visualization to file.
    
    Args:
        fig: Figure object
        frame_count: Current frame number
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save figure
    filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    fig.savefig(filename, dpi=150)

def run_simulation(save_frames=False):
    """
    Run the DWARF simulator with emergent quantum effects analysis.
    
    Args:
        save_frames: Whether to save visualization frames
    """
    print("Starting DWARF simulation with emergent quantum effects tracking...")
    
    # Create config and simulator
    config = SimulationConfig()
    
    # Create a new logger with the simulation config
    logger = create_new_logger(
        log_dir="logs", 
        simulation_name="quantum_dwarf", 
        config=config
    )
    
    # Log simulation start
    logger.log_event("SIMULATION", "Starting DWARF simulation with quantum tracking")
    
    vortex_simulator = DwarfVortexSimulator(config)
    
    # Create emergent quantum tracker
    quantum_tracker = EmergentQuantumTracker(vortex_simulator)
    
    # Setup visualization
    fig, axes = setup_visualization()
    
    # Create vortex control panel in a separate figure
    vortex_panel = VortexControlPanel(vortex_simulator)
    
    # Simulation parameters
    dt = config.time_step
    max_steps = config.max_steps
    viz_interval = config.viz_interval
    log_interval = config.log_interval if hasattr(config, 'log_interval') else 10
    
    # Track trajectory and energy
    trajectory = []
    energy_history = []
    time_history = []
    current_time = 0.0
    frame_count = 0
    
    # Report initial conditions
    print(f"Initial position: {vortex_simulator.get_electron_position()}")
    print(f"Initial velocity: {vortex_simulator.get_electron_velocity()}")
    print(f"Time step: {dt}, Max steps: {max_steps}")
    
    # Log initial state
    logger.log_event("CONFIG", f"Time step: {dt}, Max steps: {max_steps}")
    
    # Run the simulation
    start_time = time.time()
    try:
        for step in range(max_steps):
            # Update current time
            current_time += dt
            
            # Update the vortex simulator - this runs the core DWARF physics
            vortex_field = vortex_simulator.step(dt)
            
            # Record current state
            position = vortex_simulator.get_electron_position()
            energy = vortex_simulator.get_electron_energy()
            velocity = vortex_simulator.get_electron_velocity()
            
            # Log step data (core physics already uses the global logger)
            # Additional quantum logging
            if step % log_interval == 0:
                logger.log_quantum_metrics(step, current_time, quantum_tracker)
            
            trajectory.append(position.copy())
            energy_history.append(energy)
            time_history.append(current_time)
            
            # Keep trajectory at a manageable size
            max_trajectory_points = 500
            if len(trajectory) > max_trajectory_points:
                trajectory.pop(0)
                energy_history.pop(0)
                time_history.pop(0)
            
            # Track emergent quantum effects WITHOUT modifying DWARF physics
            quantum_tracker.update(dt, vortex_field)
            
            # Periodically update visualization
            if step % viz_interval == 0:
                update_visualization(fig, axes, vortex_simulator, quantum_tracker, 
                                   trajectory, energy_history, time_history, vortex_panel)
                
                # Save frame if requested
                if save_frames:
                    save_visualization(fig, frame_count)
                    frame_count += 1
                
                # Report progress
                if step % (viz_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    state = quantum_tracker.get_quantum_state_info()
                    progress = step / max_steps * 100
                    
                    # Use ASCII-compatible formatting
                    status_msg = (f"Progress: {progress:.1f}% (Step {step}/{max_steps}), " + 
                          f"Time: {elapsed:.1f}s, " +
                          f"Energy: {energy:.4f}, " +
                          f"Emergent state: n~{state.get('n', '?')}, " +
                          f"type: {state.get('orbital_type', 'unknown')}")
                    
                    print(status_msg)
                    logger.log_event("STATUS", status_msg)
            
            # Save checkpoint periodically
            if step > 0 and step % 1000 == 0:
                logger.save_checkpoint(vortex_simulator, quantum_tracker, step)
        
        # Log final statistics
        logger.save_statistics_report()
        
        # Final analysis after simulation completes
        print("\nSimulation completed. Running final analysis...")
        logger.log_event("SIMULATION", "Simulation completed, running final analysis")
        
        # Create comprehensive visualizations of the emergent quantum effects
        plt.ioff()  # Turn off interactive mode
        
        # Visualize resonances
        quantum_tracker.resonance_analyzer.visualize_resonances()
        
        # Visualize probability cloud
        quantum_tracker.probability_tracker.visualize_probability_cloud(mode='2d_slice')
        quantum_tracker.probability_tracker.visualize_probability_cloud(mode='radial')
        
        # Visualize spin-orbit coupling
        quantum_tracker.torsion_analyzer.visualize_spin_orbit_coupling()
        
        # Show final quantum state
        final_state = quantum_tracker.get_quantum_state_info()
        print("\nFinal emergent quantum-like state:")
        print(f"  n ~ {final_state['n']}")
        print(f"  l ~ {final_state['l']}")
        print(f"  j ~ {final_state.get('j', 'N/A')}")
        print(f"  Orbital type: {final_state['orbital_type']}")
        print(f"  Energy: {final_state['energy']:.6f} eV")
        
        if 'fine_structure_shift' in final_state:
            print(f"  Fine structure shift: {final_state['fine_structure_shift']:.8f} eV")
        
        # Log final state
        logger.log_event("FINAL_STATE", f"n~{final_state['n']}, l~{final_state['l']}, "
                       f"type: {final_state['orbital_type']}, E={final_state['energy']:.6f} eV")
        
        plt.show(block=True)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        logger.log_event("INTERRUPTED", "Simulation interrupted by user")
        # Still log final statistics
        logger.save_statistics_report()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        logger.log_event("ERROR", f"Simulation error: {str(e)}")
    
    print(f"Simulation finished. Total runtime: {time.time() - start_time:.2f} seconds.")
    logger.log_event("FINISHED", f"Total runtime: {time.time() - start_time:.2f} seconds")

def create_animation(output_dir="output", output_file="dwarf_simulation.mp4"):
    """
    Create animation from saved frames.
    
    Args:
        output_dir: Directory with saved frames
        output_file: Output animation file
    """
    import glob
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    
    # Create logger
    logger = get_logger()
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found.")
        logger.log_event("ERROR", f"Output directory {output_dir} not found")
        return
        
    # Get list of frames
    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    
    if not frames:
        print("No frames found.")
        logger.log_event("ERROR", "No animation frames found")
        return
        
    print(f"Creating animation from {len(frames)} frames...")
    logger.log_event("ANIMATION", f"Creating animation from {len(frames)} frames")
    
    # Create figure for animation
    fig = plt.figure(figsize=(12, 9))
    
    # Create animation
    def update(frame):
        plt.clf()
        plt.imshow(plt.imread(frames[frame]))
        plt.axis('off')
        return [plt.gca()]
        
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50)
    
    # Save animation
    try:
        writer = FFMpegWriter(fps=20, bitrate=5000)
        ani.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        logger.log_event("ANIMATION", f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        logger.log_event("ERROR", f"Animation error: {str(e)}")
        print("If FFmpeg is not installed, try: pip install ffmpeg-python")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='DWARF Simulator with Emergent Quantum Effects')
    parser.add_argument('--save-frames', action='store_true', help='Save visualization frames')
    parser.add_argument('--create-animation', action='store_true', help='Create animation from saved frames')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    
    args = parser.parse_args()
    
    if args.create_animation:
        create_animation()
    else:
        run_simulation(save_frames=args.save_frames)