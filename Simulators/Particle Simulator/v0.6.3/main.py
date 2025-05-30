"""
Main entry point for DWARF Vortex Simulator.
Includes visualization and simulation control.
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from dwarf_vortex_simulator import DwarfVortexSimulator
from config import SimulationConfig
from quantum_tracker import EmergentQuantumTracker
from vortex_control_panel import VortexControlPanel

# Standard colors for particles
PARTICLE_COLORS = {
    'proton': 'red',
    'electron': 'blue',
    'neutron': 'gray'
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DWARF Vortex Simulator')
    parser.add_argument('--save-frames', action='store_true',
                      help='Save animation frames as PNG files')
    parser.add_argument('--steps', type=int, default=None,
                      help='Number of simulation steps')
    parser.add_argument('--vortex-panel', action='store_true',
                      help='Show vortex control panel')
                      
    return parser.parse_args()

def create_plots():
    """Create the visualization plots."""
    fig = plt.figure(figsize=(18, 10))
    
    # Vortex field visualization (3D)
    vortex_ax = fig.add_subplot(2, 3, 1, projection='3d')
    vortex_ax.set_title("DWARF Vortex Field")
    
    # Electron probability visualization (3D)
    prob_ax = fig.add_subplot(2, 3, 2, projection='3d')
    prob_ax.set_title("p-like Probability Distribution")
    
    # Energy level visualization
    energy_ax = fig.add_subplot(2, 3, 3)
    energy_ax.set_title("Energy Level")
    
    # Orbital angular momentum and spin
    spin_ax = fig.add_subplot(2, 3, 4)
    spin_ax.set_title("Angular Momentum and Spin")
    
    # Superposition visualization
    state_ax = fig.add_subplot(2, 3, 5)
    state_ax.set_title("Quantum State")
    
    # Information panel
    info_ax = fig.add_subplot(2, 3, 6)
    info_ax.set_title("Quantum Information")
    info_ax.axis('off')
    
    plt.tight_layout()
    
    return fig, (vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax)

def update_visualization(frame, vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax, 
                       vortex_simulator, quantum_tracker, config, trajectory_points):
    """
    Update the visualization for the current frame.
    
    Args:
        frame: Frame number
        vortex_ax, prob_ax, etc.: Matplotlib axes
        vortex_simulator: DwarfVortexSimulator instance
        quantum_tracker: EmergentQuantumTracker instance
        config: SimulationConfig instance
        trajectory_points: List to store trajectory points
    """
    # Print progress for debugging
    print(f"Updating frame {frame}...")
    
    # Clear axes
    vortex_ax.clear()
    prob_ax.clear()
    energy_ax.clear()
    spin_ax.clear()
    state_ax.clear()
    info_ax.clear()
    
    # Set titles
    vortex_ax.set_title("DWARF Vortex Field")
    prob_ax.set_title("p-like Probability Distribution")
    energy_ax.set_title("Energy Level")
    spin_ax.set_title("Angular Momentum and Spin")
    state_ax.set_title("Quantum State")
    info_ax.set_title("Quantum Information")
    
    # Set up info panel
    info_ax.axis('off')
    
    # Run simulation step
    step_dt = config.time_step
    # Run multiple physical steps per visualization frame
    for _ in range(config.viz_interval):
        vortex_simulator.step(step_dt)
    
    # Get electron position and update trajectory
    current_pos = vortex_simulator.get_electron_position()
    trajectory_points.append(current_pos.copy())
    
    # Keep trajectory limited to avoid growing indefinitely
    max_trajectory = 100
    if len(trajectory_points) > max_trajectory:
        trajectory_points.pop(0)
    
    # Update quantum tracker with electron state
    quantum_tracker.update(current_pos, vortex_simulator.get_electron_velocity(), 
                         vortex_simulator.get_electron_energy())
    
    # Get quantum state information
    quantum_info = quantum_tracker.get_quantum_state_info()
    
    # Visualize vortex field with trajectory
    vortex_ax.set_xlim([-500, 500])
    vortex_ax.set_ylim([-500, 500])
    vortex_ax.set_zlim([-10, 10])
    
    # Plot trajectory
    if len(trajectory_points) > 1:
        trajectory = np.array(trajectory_points)
        vortex_ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.5)
    
    # Plot current position (electron)
    vortex_ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                    c=PARTICLE_COLORS['electron'], s=50)
    
    # Plot proton at origin
    vortex_ax.scatter([0], [0], [0], c=PARTICLE_COLORS['proton'], s=100)
    
    # Visualize probability distribution
    if quantum_info:
        prob_ax.set_xlim([-500, 500])
        prob_ax.set_ylim([-500, 500])
        prob_ax.set_zlim([-10, 10])
        
        # Plot proton at origin
        prob_ax.scatter([0], [0], [0], c=PARTICLE_COLORS['proton'], s=100)
        
        # Plot probability cloud based on quantum state
        n = quantum_info.get('n', 1)
        l = quantum_info.get('l', 0)
        m = quantum_info.get('m', 0)
        
        # Generate simplified probability distribution for visualization
        r = np.linspace(0, 400, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Simple approximation of hydrogen-like wavefunction probability
        if l == 0:  # s orbital
            prob = np.exp(-r_grid / (50 * n**2))
            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)
            z = np.zeros_like(x)
        elif l == 1:  # p orbital
            prob = r_grid * np.exp(-r_grid / (50 * n**2)) * np.abs(np.cos(theta_grid - m * np.pi/2))**2
            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)
            z = np.zeros_like(x)
        elif l == 2:  # d orbital
            prob = (r_grid**2) * np.exp(-r_grid / (50 * n**2)) * np.abs(np.cos(2*theta_grid - m * np.pi/3))**2
            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)
            z = np.zeros_like(x)
        else:
            # Higher orbitals - simplified visualization
            prob = (r_grid**l) * np.exp(-r_grid / (50 * n**2))
            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)
            z = np.zeros_like(x)
        
        # Scale probability for visualization
        prob = prob / np.max(prob) * 10
        
        # Plot as points with size proportional to probability
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if prob[i, j] > 0.1:
                    prob_ax.scatter([x[i, j]], [y[i, j]], [z[i, j]], 
                                  s=prob[i, j]*5, alpha=0.7, c=PARTICLE_COLORS['electron'])
        
        # Energy level visualization
        energy = quantum_info.get('energy', 0.0)
        ground_energy = -13.6 
        n_levels = 5
        
        # Draw energy levels
        for i in range(1, n_levels+1):
            level_energy = ground_energy / (i**2)
            energy_ax.axhline(y=level_energy, color='gray', linestyle='--', alpha=0.5)
            energy_ax.text(0.1, level_energy + 0.5, f"n={i}", fontsize=10)
        
        # Draw current energy
        energy_ax.axhline(y=energy, color='red', linestyle='-', linewidth=2)
        energy_ax.text(0.8, energy + 0.5, f"{energy:.2f} eV", fontsize=10)
        
        energy_ax.set_ylim([ground_energy*1.2, 0])
        energy_ax.set_xlim([0, 1])
        energy_ax.set_ylabel("Energy (eV)")
        energy_ax.set_xticks([])
        
        # Angular momentum visualization
        l_value = quantum_info.get('l', 0)
        m_value = quantum_info.get('m', 0)
        j_value = quantum_info.get('j', l_value)
        
        spin_ax.set_xlim([-2, 2])
        spin_ax.set_ylim([-2, 2])
        spin_ax.arrow(0, 0, 0, l_value, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
        spin_ax.text(0.1, l_value, f"L = {l_value}", fontsize=10)
        
        # Draw spin
        spin_ax.arrow(0, 0, 0.5, 0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
        spin_ax.text(0.6, 0.6, "S = 1/2", fontsize=10)
        
        # Draw total angular momentum
        spin_ax.arrow(0, 0, 0, j_value, head_width=0.15, head_length=0.1, fc='purple', ec='purple')
        spin_ax.text(-0.9, j_value, f"J = {j_value}", fontsize=10)
        
        # Information display
        orbital_type = quantum_info.get('orbital_type', 's')
        info_text = [
            f"Quantum Numbers:",
            f"n = {n} (principal)",
            f"l = {l} (orbital angular momentum)",
            f"m = {m} (magnetic)",
            f"j = {j_value} (total angular momentum)",
            f"",
            f"Orbital: {n}{orbital_type}",
            f"Energy: {energy:.2f} eV",
        ]
        
        # Add trajectory analysis
        trajectory_info = quantum_tracker.get_trajectory_analysis()
        if trajectory_info:
            info_text.extend(["", "Trajectory Analysis:"])
            for key, value in trajectory_info.items():
                if value is not None:
                    info_text.append(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Display text
        info_ax.text(0.05, 0.95, '\n'.join(info_text), va='top', transform=info_ax.transAxes)
    
    return vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax

def run_vortex_control_panel(config):
    """Run the vortex control panel."""
    print("Starting vortex control panel...")
    vortex_simulator = DwarfVortexSimulator(config)
    panel = VortexControlPanel(vortex_simulator)
    plt.show()

def run_simulation(save_frames=False, max_steps=None):
    """
    Run the DWARF vortex simulation with visualization.
    
    Args:
        save_frames: Whether to save animation frames
        max_steps: Maximum number of simulation steps
    """
    # Create configuration
    print("Initializing simulation...")
    config = SimulationConfig()
    
    # Set up logging with absolute path to ensure proper directory
    import os
    from logger import create_new_logger
    
    # Create the logs directory in a known location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Initializing logger at: {log_dir}")
    logger = create_new_logger(log_dir=log_dir, simulation_name="dwarf_sim", config=config)
    logger.log_event("SIMULATION", "Starting DWARF simulation")
    
    # Create simulator
    vortex_simulator = DwarfVortexSimulator(config)
    
    # Create quantum tracker
    quantum_tracker = EmergentQuantumTracker()
    
    # Create plots
    print("Creating visualization...")
    fig, (vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax) = create_plots()
    
    # Store trajectory points
    trajectory_points = []
    
    # Set up animation
    print("Setting up animation...")
    steps = max_steps if max_steps is not None else config.max_steps
    interval = 50  # ms between animation frames
    
    # Add text to display simulation info
    fig.text(0.02, 0.02, f"DWARF Vortex Simulator\nElectron in Hydrogen-like Atom", fontsize=10)
    
    # Create the animation
    print("Starting animation loop...")
    anim = FuncAnimation(
        fig, update_visualization, frames=range(steps),
        fargs=(vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax, 
              vortex_simulator, quantum_tracker, config, trajectory_points),
        interval=interval, blit=False, repeat=False
    )
    
    # Save frames if requested
    if save_frames:
        print("Saving animation frames...")
        os.makedirs('frames', exist_ok=True)
        for i in range(steps):
            filename = f'frames/frame_{i:04d}.png'
            print(f"Saving {filename}...")
            # Save the updated figure
            plt.savefig(filename)
            # Update visualization for next frame
            update_visualization(i, vortex_ax, prob_ax, energy_ax, spin_ax, state_ax, info_ax,
                              vortex_simulator, quantum_tracker, config, trajectory_points)
        print("All frames saved.")
    else:
        # Show the animation
        print("Displaying animation. Close the window to end the simulation.")
        plt.show()
    
    # Final message
    print("Simulation complete!")
    
    # Generate statistics report
    logger.save_statistics_report()
    
    return vortex_simulator, quantum_tracker

# Main entry point
if __name__ == "__main__":
    print("DWARF Simulator starting...")
    args = parse_arguments()
    
    # Run either the control panel or the simulation based on args
    if args.vortex_panel:
        print("Running vortex control panel...")
        config = SimulationConfig()
        run_vortex_control_panel(config)
    else:
        print("Running main simulation...")
        run_simulation(save_frames=args.save_frames, max_steps=args.steps)
        
    print("DWARF Simulator finished.")