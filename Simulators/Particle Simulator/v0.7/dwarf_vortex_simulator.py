import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Any, Optional
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

from physics_core import calculate_forces, apply_memory_field, calculate_field_at_point
from particle_types import Particle, Proton, Electron, Neutron

class DwarfVortexSimulator:
    """
    Main simulator class for the DWARF (Dynamic Wake Accretion in Relativistic Fluids) theory.
    Simulates particle interactions based on vortex dynamics, memory field effects, and spin coupling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DWARF vortex simulator.
        
        Args:
            config: Dictionary containing simulation configuration parameters
        """
        # Simulation parameters
        self.dt = config.get('dt', 0.01)
        self.width = config.get('width', 800)
        self.height = config.get('height', 600)
        self.memory_decay = config.get('memory_decay', 0.995)
        self.field_resolution = config.get('field_resolution', 30)
        self.saturation_limit = config.get('saturation_limit', 5.0)
        self.global_drag = config.get('global_drag', 0.001)
        
        # DWARF-specific parameters
        self.force_exponent = 2.22  # Characteristic DWARF exponent
        
        # Initialize particles
        self.particles = []
        self._initialize_particles(config.get('particles', []))
        
        # Initialize memory field
        # Field dimensions: x, y, density, curl_x, curl_y
        self.field = np.zeros((self.field_resolution, self.field_resolution, 5))
        
        # Visualization properties
        self.fig = None
        self.ax = None
        self.field_plot = None
        self.particles_plot = []
        self.trails = []
        self.paused = False
        self.show_field = True
        
        # Recording properties
        self.recording = config.get('record', False)
        self.record_dir = config.get('record_dir', 'frames')
        self.frame_count = 0
        if self.recording:
            os.makedirs(self.record_dir, exist_ok=True)
        
        # Performance optimization
        self.use_multiprocessing = config.get('use_multiprocessing', False)
        
        # Energy tracking
        self.track_energy = config.get('track_energy', True)
        self.kinetic_energy = []
        self.potential_energy = []
        self.field_energy = []
        self.total_energy = []
        
    def _initialize_particles(self, particle_configs: List[Dict[str, Any]]) -> None:
        """Initialize particles based on provided configuration."""
        for p_config in particle_configs:
            particle_type = p_config.get('type', 'particle').lower()
            
            # Common parameters
            x = p_config.get('x', self.width / 2)
            y = p_config.get('y', self.height / 2)
            vx = p_config.get('vx', 0.0)
            vy = p_config.get('vy', 0.0)
            mass = p_config.get('mass', 1.0)
            spin = p_config.get('spin', 1.0)
            
            # Create specific particle type
            if particle_type == 'proton':
                self.particles.append(Proton(x, y, vx, vy, mass, spin))
            elif particle_type == 'electron':
                self.particles.append(Electron(x, y, vx, vy, mass, spin))
            elif particle_type == 'neutron':
                self.particles.append(Neutron(x, y, vx, vy, mass, spin))
            else:
                self.particles.append(Particle(x, y, vx, vy, mass, spin))
    
    def update(self) -> None:
        """Update the simulation state for one time step."""
        if self.paused:
            return
        
        # Calculate forces between particles using DWARF physics
        if self.use_multiprocessing and len(self.particles) > multiprocessing.cpu_count():
            self._calculate_forces_parallel()
        else:
            self._calculate_forces_sequential()
        
        # Update particle positions and velocities
        for particle in self.particles:
            # Velocity Verlet integration for better numerical stability
            particle.x += particle.vx * self.dt + 0.5 * particle.ax * self.dt**2
            particle.y += particle.vy * self.dt + 0.5 * particle.ay * self.dt**2
            
            # Store old acceleration for Verlet integration
            ax_old, ay_old = particle.ax, particle.ay
            
            # Update velocity
            particle.vx += 0.5 * (ax_old + particle.ax) * self.dt
            particle.vy += 0.5 * (ay_old + particle.ay) * self.dt
            
            # Update particle trail
            particle.add_trail_point(particle.x, particle.y)
            
            # Update field based on particle movement
            self._update_field_from_particle(particle)
        
        # Apply memory decay
        self.field[:,:,0:3] *= self.memory_decay
        
        # Calculate energy if tracking is enabled
        if self.track_energy:
            self._calculate_system_energy()
    
    def _calculate_forces_sequential(self) -> None:
        """Calculate forces on all particles sequentially."""
        for i, particle in enumerate(self.particles):
            # Reset acceleration
            particle.ax = 0
            particle.ay = 0
            
            # Apply forces from other particles
            for j, other in enumerate(self.particles):
                if i != j:
                    fx, fy = calculate_forces(
                        particle, other, 
                        force_exponent=self.force_exponent, 
                        saturation_limit=self.saturation_limit
                    )
                    particle.ax += fx / particle.mass
                    particle.ay += fy / particle.mass
            
            # Apply memory field effects
            field_fx, field_fy = apply_memory_field(particle, self.field)
            particle.ax += field_fx / particle.mass
            particle.ay += field_fy / particle.mass
            
            # Apply global drag (dampening)
            particle.ax -= self.global_drag * particle.vx
            particle.ay -= self.global_drag * particle.vy
    
    def _calculate_forces_parallel(self) -> None:
        """Calculate forces on all particles using multiprocessing."""
        try:
            # Define function to calculate forces for a single particle
            def calc_forces_for_particle(idx):
                particle = self.particles[idx]
                ax, ay = 0, 0
                
                # Apply forces from other particles
                for j, other in enumerate(self.particles):
                    if idx != j:
                        fx, fy = calculate_forces(
                            particle, other,
                            force_exponent=self.force_exponent,
                            saturation_limit=self.saturation_limit
                        )
                        ax += fx / particle.mass
                        ay += fy / particle.mass
                
                # Apply memory field effects
                field_fx, field_fy = apply_memory_field(particle, self.field)
                ax += field_fx / particle.mass
                ay += field_fy / particle.mass
                
                # Apply global drag
                ax -= self.global_drag * particle.vx
                ay -= self.global_drag * particle.vy
                
                return ax, ay
            
            # Execute in parallel
            num_processes = min(multiprocessing.cpu_count(), len(self.particles))
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(executor.map(calc_forces_for_particle, range(len(self.particles))))
            
            # Update accelerations
            for i, (ax, ay) in enumerate(results):
                self.particles[i].ax = ax
                self.particles[i].ay = ay
                
        except Exception as e:
            print(f"Multiprocessing error: {e}. Falling back to sequential calculation.")
            self._calculate_forces_sequential()
    
    def _update_field_from_particle(self, particle: Particle) -> None:
        """Update the memory field based on particle movement."""
        # Convert particle position to field grid
        grid_x = int(particle.x * self.field_resolution / self.width)
        grid_y = int(particle.y * self.field_resolution / self.height)
        
        # Ensure within bounds
        if 0 <= grid_x < self.field_resolution and 0 <= grid_y < self.field_resolution:
            # Increase field density based on particle properties
            intensity = particle.mass * abs(particle.spin) * 0.1
            
            # Add to field density
            self.field[grid_y, grid_x, 0] += intensity
            
            # Add to field curl components based on spin and velocity
            spin_factor = 0.05 * particle.spin
            self.field[grid_y, grid_x, 3] += spin_factor * particle.vy
            self.field[grid_y, grid_x, 4] += spin_factor * -particle.vx
            
            # Add to field vector components
            velocity_factor = 0.1
            self.field[grid_y, grid_x, 1] += velocity_factor * particle.vx
            self.field[grid_y, grid_x, 2] += velocity_factor * particle.vy
            
            # Apply saturation limit
            self.field[grid_y, grid_x, :] = np.clip(
                self.field[grid_y, grid_x, :], 
                -self.saturation_limit, 
                self.saturation_limit
            )
    
    def _calculate_system_energy(self) -> None:
        """Calculate and store system energy components."""
        # Reset energy values
        ke = 0.0  # Kinetic energy
        pe = 0.0  # Potential energy
        fe = 0.0  # Field energy
        
        # Calculate kinetic energy
        for particle in self.particles:
            ke += 0.5 * particle.mass * (particle.vx**2 + particle.vy**2)
        
        # Calculate potential energy from particle interactions
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i < j:  # Avoid double counting
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    r = np.sqrt(dx**2 + dy**2 + 1.0)  # Add softening
                    
                    # Use DWARF force exponent for potential
                    if self.force_exponent != 1.0:  # Avoid division by zero
                        pe -= p1.mass * p2.mass / ((self.force_exponent - 1.0) * r**(self.force_exponent - 1.0))
                    else:
                        pe -= p1.mass * p2.mass * np.log(r)
        
        # Calculate field energy (sum of squared field components)
        fe = np.sum(self.field[:,:,0]**2) * 0.5  # Density energy
        fe += np.sum(self.field[:,:,1]**2 + self.field[:,:,2]**2) * 0.2  # Vector energy
        fe += np.sum(self.field[:,:,3]**2 + self.field[:,:,4]**2) * 0.3  # Curl energy
        
        # Store energy values
        self.kinetic_energy.append(ke)
        self.potential_energy.append(pe)
        self.field_energy.append(fe)
        self.total_energy.append(ke + pe + fe)
    
    def start_visualization(self) -> None:
        """Initialize and start the visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('DWARF Vortex Simulator')
        self.ax.set_xlabel('X position')
        self.ax.set_ylabel('Y position')
        
        # Create initial field visualization
        if self.show_field:
            self._update_field_visualization()
        
        # Create initial particle visualization
        for particle in self.particles:
            particle_plot, = self.ax.plot(
                [particle.x], [particle.y], 
                marker='o', 
                markersize=max(4, particle.mass/2), 
                color=particle.color
            )
            self.particles_plot.append(particle_plot)
            
            # Initialize trail
            trail_plot, = self.ax.plot([], [], '-', linewidth=1, alpha=0.5, color=particle.color)
            self.trails.append(trail_plot)
        
        # Register key event handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Start animation
        self.animation = FuncAnimation(
            self.fig, 
            self._animation_step, 
            interval=20, 
            blit=False
        )
    
    def _update_field_visualization(self) -> None:
        """Update the field visualization."""
        if self.show_field:
            # Create grid for field visualization
            x = np.linspace(0, self.width, self.field_resolution)
            y = np.linspace(0, self.height, self.field_resolution)
            X, Y = np.meshgrid(x, y)
            
            # Get field components
            U = self.field[:,:,1]  # x-component
            V = self.field[:,:,2]  # y-component
            C = self.field[:,:,0]  # density
            
            # Clear previous field plots
            if hasattr(self, 'field_plot') and self.field_plot is not None:
                for collection in self.field_plot.collections:
                    collection.remove()
            
            # Plot field with streamlines and color based on density
            self.field_plot = self.ax.streamplot(
                X, Y, U, V, 
                density=1.5, 
                color=C, 
                linewidth=1, 
                cmap='viridis',
                arrowsize=0.5
            )
    
    def _animation_step(self, frame: int) -> List:
        """Animation step function for updating visualization."""
        # Update simulation state
        self.update()
        
        # Update field visualization if needed
        if frame % 5 == 0 and self.show_field:  # Update field less frequently for performance
            self._update_field_visualization()
        
        # Update particle and trail positions
        for i, particle in enumerate(self.particles):
            if i < len(self.particles_plot):  # Check to avoid index errors
                self.particles_plot[i].set_data([particle.x], [particle.y])
                
                # Update trail with gradient effect
                x_trail, y_trail = particle.get_trail()
                self.trails[i].set_data(x_trail, y_trail)
                
                # Set trail color with gradient effect (optional enhancement)
                if len(x_trail) > 1:
                    points = np.array([x_trail, y_trail]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    # Create gradient colors
                    alpha_values = np.linspace(0.1, 1.0, len(segments))
                    colors = np.ones((len(segments), 4))
                    
                    # Extract RGB from particle color and create alpha gradient
                    if isinstance(particle.color, str):
                        rgb = plt.cm.colors.to_rgb(particle.color)
                    else:
                        rgb = particle.color[:3]
                    
                    colors[:, 0] = rgb[0]
                    colors[:, 1] = rgb[1]
                    colors[:, 2] = rgb[2]
                    colors[:, 3] = alpha_values
        
        # Save frame if recording
        if self.recording:
            self.save_frame()
            
        # Return updated artists
        return self.particles_plot + self.trails
    
    def save_frame(self) -> None:
        """Save the current frame as an image."""
        if self.recording:
            frame_path = os.path.join(self.record_dir, f"frame_{self.frame_count:05d}.png")
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            self.frame_count += 1
    
    def _on_key_press(self, event) -> None:
        """Handle key press events."""
        if event.key == 'p':
            # Toggle pause
            self.paused = not self.paused
            status = "Paused" if self.paused else "Running"
            print(f"Simulation {status}")
            
        elif event.key == 'f':
            # Toggle field visibility
            self.show_field = not self.show_field
            if not self.show_field and hasattr(self, 'field_plot') and self.field_plot is not None:
                for collection in self.field_plot.collections:
                    collection.remove()
                self.field_plot = None
            print(f"Field visualization {'hidden' if not self.show_field else 'visible'}")
            
        elif event.key == 'r':
            # Reset simulation
            self.reset()
            print("Simulation reset")
            
        elif event.key == 'q':
            # Quit simulation
            print("Exiting simulation")
            plt.close(self.fig)
            sys.exit(0)
            
        elif event.key == 's':
            # Save screenshot
            filename = f"dwarf_sim_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Screenshot saved as {filename}")
            
        elif event.key == '+' or event.key == '=':
            # Increase time step
            self.dt *= 1.1
            print(f"Time step increased to {self.dt:.5f}")
            
        elif event.key == '-' or event.key == '_':
            # Decrease time step
            self.dt *= 0.9
            print(f"Time step decreased to {self.dt:.5f}")
            
        elif event.key == 'e':
            # Toggle energy tracking
            self.track_energy = not self.track_energy
            print(f"Energy tracking {'enabled' if self.track_energy else 'disabled'}")
    
    def toggle_field_visibility(self) -> None:
        """Toggle the visibility of the field visualization."""
        self.show_field = not self.show_field
    
    def reset(self) -> None:
        """Reset the simulation state."""
        # Reset field
        self.field[:,:,:] = 0
        
        # Reset particles
        for particle in self.particles:
            particle.reset_trail()
        
        # Reset energy tracking
        if self.track_energy:
            self.kinetic_energy = []
            self.potential_energy = []
            self.field_energy = []
            self.total_energy = []
    
    def set_memory_decay(self, value: float) -> None:
        """Set memory decay parameter."""
        self.memory_decay = float(value)
    
    def set_global_drag(self, value: float) -> None:
        """Set global drag parameter."""
        self.global_drag = float(value)
    
    def set_saturation_limit(self, value: float) -> None:
        """Set saturation limit parameter."""
        self.saturation_limit = float(value)
    
    def set_particle_spin(self, particle_index: int, value: float) -> None:
        """Set spin value for a specific particle."""
        if 0 <= particle_index < len(self.particles):
            self.particles[particle_index].spin = float(value)
    
    def plot_energy(self) -> None:
        """Plot energy conservation data."""
        if not self.track_energy or len(self.kinetic_energy) == 0:
            print("No energy data available to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Time steps
        steps = range(len(self.kinetic_energy))
        
        # Plot energy components
        plt.plot(steps, self.kinetic_energy, 'b-', label='Kinetic Energy')
        plt.plot(steps, self.potential_energy, 'r-', label='Potential Energy')
        plt.plot(steps, self.field_energy, 'g-', label='Field Energy')
        plt.plot(steps, self.total_energy, 'k--', label='Total Energy')
        
        plt.xlabel('Time Step')
        plt.ylabel('Energy')
        plt.title('Energy Conservation in DWARF Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Display or save
        if not self.recording:
            plt.show()
        else:
            plt.savefig(os.path.join(self.record_dir, "energy_plot.png"), dpi=300)
            plt.close()