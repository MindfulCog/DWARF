import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Tuple, Optional, Dict, Union, Any
import time
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

# Constants
G = 6.67430e-11  # Gravitational constant
SOFTENING = 1.0  # Softening parameter to prevent division by zero

# Screen dimensions
WIDTH, HEIGHT = 800, 600
BG_COLOR = (0, 0, 0)
FPS = 60

# Particle class
class Particle:
    def __init__(self, x: float, y: float, mass: float, vx: float = 0, vy: float = 0, 
                color: Tuple[int, int, int] = (255, 255, 255)):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.ax = 0
        self.ay = 0
        self.color = color
        self.radius = int(max(2, min(10, mass / 5e10)))
        self.trail = []
        self.trail_length = 50

    def update_position(self, dt: float) -> None:
        """Update particle position based on velocity and acceleration."""
        # Use Velocity Verlet integration for better numerical stability
        self.x += self.vx * dt + 0.5 * self.ax * dt**2
        self.y += self.vy * dt + 0.5 * self.ay * dt**2
        
        # Store old acceleration for Verlet integration
        old_ax, old_ay = self.ax, self.ay
        
        # New velocity will be updated after acceleration is recalculated
        self.vx += 0.5 * (old_ax + self.ax) * dt
        self.vy += 0.5 * (old_ay + self.ay) * dt
        
        # Add current position to the trail
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)

    def calculate_force(self, other: 'Particle', softening: float = SOFTENING) -> Tuple[float, float]:
        """Calculate gravitational force between two particles."""
        dx = other.x - self.x
        dy = other.y - self.y
        dist_squared = dx**2 + dy**2 + softening**2
        dist = np.sqrt(dist_squared)
        
        # Calculate force magnitude (F = G * m1 * m2 / r^2)
        force = G * self.mass * other.mass / dist_squared
        
        # Calculate force components
        fx = force * dx / dist
        fy = force * dy / dist
        
        return fx, fy

    def draw(self, screen) -> None:
        """Draw the particle and its trail on the screen."""
        # Draw trail with gradient effect
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = int(255 * i / len(self.trail))
                trail_color = (min(self.color[0], alpha), 
                              min(self.color[1], alpha), 
                              min(self.color[2], alpha))
                pygame.draw.line(screen, trail_color, self.trail[i-1], self.trail[i], 1)
        
        # Draw particle
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def calculate_acceleration_for_particle(particles: List[Particle], i: int) -> Tuple[float, float]:
    """Calculate net acceleration on a single particle."""
    particle = particles[i]
    ax, ay = 0, 0
    
    for j, other in enumerate(particles):
        if i != j:  # Skip self
            fx, fy = particle.calculate_force(other)
            ax += fx / particle.mass
            ay += fy / particle.mass
            
    return ax, ay


def update_accelerations(particles: List[Particle], use_multiprocessing: bool = False) -> None:
    """Update accelerations of all particles."""
    # Only use multiprocessing if beneficial (more particles than CPU cores)
    if use_multiprocessing and len(particles) > multiprocessing.cpu_count():
        try:
            # Calculate accelerations in parallel
            num_processes = multiprocessing.cpu_count()
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                func = partial(calculate_acceleration_for_particle, particles)
                results = list(executor.map(func, range(len(particles))))
            
            # Update particles with calculated accelerations
            for i, (ax, ay) in enumerate(results):
                particles[i].ax = ax
                particles[i].ay = ay
        except Exception as e:
            print(f"Multiprocessing error: {e}. Falling back to sequential calculation.")
            # Fall back to sequential calculation if multiprocessing fails
            for i, particle in enumerate(particles):
                ax, ay = calculate_acceleration_for_particle(particles, i)
                particle.ax = ax
                particle.ay = ay
    else:
        # Sequential calculation
        for i, particle in enumerate(particles):
            ax, ay = calculate_acceleration_for_particle(particles, i)
            particle.ax = ax
            particle.ay = ay


def generate_random_particles(num_particles: int, width: int, height: int) -> List[Particle]:
    """Generate random particles within the given dimensions."""
    particles = []
    for _ in range(num_particles):
        # Random position
        x = np.random.uniform(100, width - 100)
        y = np.random.uniform(100, height - 100)
        
        # Random mass (between 1e10 and 1e12)
        mass = np.random.uniform(1e10, 1e12)
        
        # Random velocity
        vx = np.random.uniform(-50, 50)
        vy = np.random.uniform(-50, 50)
        
        # Random color - brighter colors for better visibility
        color = (np.random.randint(150, 255),
                np.random.randint(150, 255),
                np.random.randint(150, 255))
        
        particles.append(Particle(x, y, mass, vx, vy, color))
    
    return particles


def generate_orbit_system(width: int, height: int) -> List[Particle]:
    """Generate a simple orbital system with a central mass and orbiting particles."""
    particles = []
    
    # Central mass
    central_mass = 1e13
    center_x, center_y = width // 2, height // 2
    particles.append(Particle(center_x, center_y, central_mass, 0, 0, (255, 215, 0)))  # Gold color
    
    # Orbiting particles
    num_orbiting = 5
    for i in range(num_orbiting):
        angle = 2 * np.pi * i / num_orbiting
        orbit_radius = 150
        x = center_x + orbit_radius * np.cos(angle)
        y = center_y + orbit_radius * np.sin(angle)
        
        # Calculate orbital velocity for a circular orbit
        # v = sqrt(G * M / r)
        mass = 1e11
        orbit_speed = np.sqrt(G * central_mass / orbit_radius)
        
        # Velocity components perpendicular to radius
        vx = orbit_speed * np.sin(angle)
        vy = -orbit_speed * np.cos(angle)
        
        # Different colors for each orbiting particle
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (0, 255, 255)]
        
        particles.append(Particle(x, y, mass, vx, vy, colors[i % len(colors)]))
    
    return particles


def generate_collision_system(width: int, height: int) -> List[Particle]:
    """Generate a system where two clusters of particles will collide."""
    particles = []
    
    # First cluster
    for _ in range(10):
        x = np.random.uniform(width * 0.2, width * 0.3)
        y = np.random.uniform(height * 0.4, height * 0.6)
        mass = np.random.uniform(5e10, 8e10)
        vx = 30  # Moving right
        vy = np.random.uniform(-5, 5)
        color = (200, 100, 100)  # Reddish
        particles.append(Particle(x, y, mass, vx, vy, color))
    
    # Second cluster
    for _ in range(10):
        x = np.random.uniform(width * 0.7, width * 0.8)
        y = np.random.uniform(height * 0.4, height * 0.6)
        mass = np.random.uniform(5e10, 8e10)
        vx = -30  # Moving left
        vy = np.random.uniform(-5, 5)
        color = (100, 100, 200)  # Bluish
        particles.append(Particle(x, y, mass, vx, vy, color))
    
    return particles


def generate_galaxy_system(width: int, height: int) -> List[Particle]:
    """Generate a spiral galaxy-like system."""
    particles = []
    
    # Central black hole
    central_mass = 5e13
    center_x, center_y = width // 2, height // 2
    particles.append(Particle(center_x, center_y, central_mass, 0, 0, (255, 255, 255)))
    
    # Spiral arms
    num_particles = 100
    a = 10  # Spiral parameter
    for i in range(num_particles):
        # Spiral equation in polar coordinates
        theta = 4 * np.pi * i / num_particles
        r = a * theta
        
        # Convert to Cartesian coordinates
        x = center_x + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        
        # Keep particles within screen bounds
        if 0 <= x < width and 0 <= y < height:
            # Mass and orbital velocity
            mass = np.random.uniform(1e9, 5e10)
            
            # Calculate orbital velocity (tangential to radius)
            orbit_speed = 0.8 * np.sqrt(G * central_mass / r)
            
            # Velocity components perpendicular to radius
            vx = orbit_speed * np.sin(theta)
            vy = -orbit_speed * np.cos(theta)
            
            # Color based on distance from center (blue to red)
            ratio = i / num_particles
            color = (int(255 * ratio), int(70 * (1-ratio)), int(255 * (1-ratio)))
            
            particles.append(Particle(x, y, mass, vx, vy, color))
    
    return particles


def save_screenshot(screen, frame_count: int, output_dir: str) -> None:
    """Save the current simulation frame as an image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pygame.image.save(screen, os.path.join(output_dir, f"frame_{frame_count:05d}.png"))


def plot_energy(kinetic_energy: List[float], potential_energy: List[float], total_energy: List[float], step: int) -> np.ndarray:
    """Plot energy values and return as a numpy array."""
    fig, ax = plt.subplots(figsize=(8, 6))
    time_steps = range(0, step + 1)
    
    ax.plot(time_steps, kinetic_energy, 'b-', label='Kinetic Energy')
    ax.plot(time_steps, potential_energy, 'r-', label='Potential Energy')
    ax.plot(time_steps, total_energy, 'g-', label='Total Energy')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True)
    
    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot_image = np.array(canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return plot_image


def calculate_energy(particles: List[Particle]) -> Tuple[float, float]:
    """Calculate kinetic and potential energy of the system."""
    kinetic_energy = 0
    potential_energy = 0
    
    # Calculate kinetic energy
    for particle in particles:
        v_squared = particle.vx**2 + particle.vy**2
        kinetic_energy += 0.5 * particle.mass * v_squared
    
    # Calculate potential energy
    for i, particle1 in enumerate(particles):
        for j, particle2 in enumerate(particles):
            if i < j:  # Avoid double counting
                dx = particle2.x - particle1.x
                dy = particle2.y - particle1.y
                dist = np.sqrt(dx**2 + dy**2 + SOFTENING**2)
                potential_energy -= G * particle1.mass * particle2.mass / dist
    
    return kinetic_energy, potential_energy


def run_simulation(scenario: str = 'orbit', num_particles: int = 20, record: bool = False, 
                  output_dir: str = 'output', use_multiprocessing: bool = False,
                  show_energy: bool = False, timestep: float = 0.1) -> None:
    """Run the particle simulation."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-Body Particle Simulator")
    clock = pygame.time.Clock()
    
    # Load font for displaying information
    try:
        font = pygame.font.SysFont('Arial', 18)
    except pygame.error:
        # Fallback to default font if Arial is not available
        font = pygame.font.Font(None, 18)
    
    # Initialize particles based on scenario
    if scenario == 'random':
        particles = generate_random_particles(num_particles, WIDTH, HEIGHT)
    elif scenario == 'orbit':
        particles = generate_orbit_system(WIDTH, HEIGHT)
    elif scenario == 'collision':
        particles = generate_collision_system(WIDTH, HEIGHT)
    elif scenario == 'galaxy':
        particles = generate_galaxy_system(WIDTH, HEIGHT)
    else:
        print(f"Unknown scenario: {scenario}. Defaulting to 'orbit'")
        particles = generate_orbit_system(WIDTH, HEIGHT)
    
    # Metrics for energy conservation
    kinetic_energy_history = []
    potential_energy_history = []
    total_energy_history = []
    
    frame_count = 0
    running = True
    paused = False
    dt = timestep  # Time step in seconds
    
    # Create energy plot surface if needed
    energy_plot_surface = None
    
    # For FPS calculation
    fps_update_interval = 10  # Update FPS display every 10 frames
    fps_values = []
    
    # Controls info
    controls_visible = False
    
    # Create a semi-transparent surface for controls overlay
    controls_surface = pygame.Surface((WIDTH//2, HEIGHT//3), pygame.SRCALPHA)
    controls_surface.fill((20, 20, 20, 200))  # Semi-transparent dark background
    
    # Control info text
    control_lines = [
        "Controls:",
        "ESC - Exit simulation",
        "SPACE - Pause/Resume",
        "S - Save screenshot (if recording enabled)",
        "C - Toggle controls display",
        "R - Reset simulation with same scenario",
        "E - Toggle energy display (if enabled)",
        "+ / - - Increase/decrease time step"
    ]
    
    start_time = time.time()
    
    try:
        while running:
            frame_start_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s and record:
                        save_screenshot(screen, frame_count, output_dir)
                    elif event.key == pygame.K_c:
                        controls_visible = not controls_visible
                    elif event.key == pygame.K_r:
                        # Reset simulation with same parameters
                        if scenario == 'random':
                            particles = generate_random_particles(num_particles, WIDTH, HEIGHT)
                        elif scenario == 'orbit':
                            particles = generate_orbit_system(WIDTH, HEIGHT)
                        elif scenario == 'collision':
                            particles = generate_collision_system(WIDTH, HEIGHT)
                        elif scenario == 'galaxy':
                            particles = generate_galaxy_system(WIDTH, HEIGHT)
                        
                        # Reset energy history
                        kinetic_energy_history = []
                        potential_energy_history = []
                        total_energy_history = []
                    elif event.key == pygame.K_e and show_energy:
                        # Toggle energy plot visibility
                        show_energy = not show_energy
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        # Increase time step
                        dt *= 1.1
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        # Decrease time step
                        dt *= 0.9
            
            if not paused:
                # Calculate energy before update
                if show_energy:
                    ke, pe = calculate_energy(particles)
                    kinetic_energy_history.append(ke)
                    potential_energy_history.append(pe)
                    total_energy_history.append(ke + pe)
                    
                    # Create or update energy plot
                    if frame_count % 10 == 0:  # Update plot every 10 frames
                        energy_plot = plot_energy(kinetic_energy_history, potential_energy_history, 
                                                total_energy_history, len(kinetic_energy_history)-1)
                        energy_plot_surface = pygame.surfarray.make_surface(energy_plot)
                
                # Update accelerations for all particles
                update_accelerations(particles, use_multiprocessing)
                
                # Update positions based on new accelerations
                for particle in particles:
                    particle.update_position(dt)
                
                frame_count += 1
            
            # Clear the screen
            screen.fill(BG_COLOR)
            
            # Draw all particles
            for particle in particles:
                particle.draw(screen)
            
            # Calculate and store current FPS
            frame_time = time.time() - frame_start_time
            if frame_time > 0:
                current_fps = 1 / frame_time
                fps_values.append(current_fps)
                if len(fps_values) > fps_update_interval:
                    fps_values.pop(0)
            
            # Calculate average FPS over the last few frames
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
            
            # Display simulation info
            run_time = time.time() - start_time
            hours, remainder = divmod(run_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            info_text = (f"Scenario: {scenario} | Particles: {len(particles)} | "
                        f"FPS: {int(avg_fps)} | Time step: {dt:.3f} | "
                        f"Runtime: {time_str} | {'Paused' if paused else 'Running'}")
            info_surface = font.render(info_text, True, (255, 255, 255))
            screen.blit(info_surface, (10, 10))
            
            # Draw energy plot if enabled
            if show_energy and energy_plot_surface is not None:
                # Scale the plot to fit in the corner
                scale_factor = 0.3
                plot_width = int(WIDTH * scale_factor)
                plot_height = int(HEIGHT * scale_factor)
                scaled_plot = pygame.transform.scale(energy_plot_surface, (plot_width, plot_height))
                screen.blit(scaled_plot, (WIDTH - plot_width - 10, 10))
            
            # Display controls if visible
            if controls_visible:
                screen.blit(controls_surface, (WIDTH//4, HEIGHT//3))
                
                for i, line in enumerate(control_lines):
                    control_text = font.render(line, True, (255, 255, 255))
                    screen.blit(control_text, (WIDTH//4 + 10, HEIGHT//3 + 10 + (i * 20)))
            
            # Save frame if recording
            if record and not paused:
                save_screenshot(screen, frame_count, output_dir)
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(FPS)
    
    except Exception as e:
        print(f"Error in simulation: {e}")
        
    finally:
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-body particle simulator')
    parser.add_argument('--scenario', type=str, default='orbit', 
                      choices=['random', 'orbit', 'collision', 'galaxy'],
                      help='Simulation scenario')
    parser.add_argument('--particles', type=int, default=20, 
                      help='Number of particles (for random scenario)')
    parser.add_argument('--record', action='store_true', 
                      help='Record frames to output directory')
    parser.add_argument('--output', type=str, default='output', 
                      help='Output directory for recorded frames')
    parser.add_argument('--multiprocessing', action='store_true', 
                      help='Use multiprocessing for acceleration calculations')
    parser.add_argument('--energy', action='store_true', 
                      help='Show energy conservation plot')
    parser.add_argument('--timestep', type=float, default=0.1,
                      help='Simulation time step (smaller for more accuracy)')
                      
    args = parser.parse_args()
    
    run_simulation(scenario=args.scenario, num_particles=args.particles,
                  record=args.record, output_dir=args.output,
                  use_multiprocessing=args.multiprocessing,
                  show_energy=args.energy, timestep=args.timestep)