import sys
import time
import numpy as np
from vispy import app, gloo
from config import GRID_SIZE, DT
from physics_core import simulate_step, BOUNDARY_WRAP, BOUNDARY_REFLECT, BOUNDARY_DAMP
import physics_core
from particle_types import generate_default_particles
from logger import init_loggers
from atom_detector import AtomDetector

# Use the shaders from visualizer_gpu.py (they work better)
VERT_SHADER = """
attribute vec2 a_position;
attribute vec4 a_color;
attribute float a_size;
varying vec4 v_color;
void main() {
    v_color = a_color;
    gl_Position = vec4(2.0 * a_position - 1.0, 0.0, 1.0);
    gl_PointSize = a_size;
}
"""

FRAG_SHADER = """
varying vec4 v_color;
void main() {
    float r = 0.0, delta = 0.0, alpha = 1.0;
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    r = dot(cxy, cxy);
    if (r > 1.0) {
        discard;
    }
    gl_FragColor = v_color;
}
"""

# Add a shader for glowing harmonic particles
GLOW_FRAG_SHADER = """
varying vec4 v_color;
void main() {
    float r = 0.0, delta = 0.0, alpha = 1.0;
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    r = dot(cxy, cxy);
    if (r > 1.0) {
        discard;
    }
    // Gold glow effect for harmonic particles
    vec4 glow_color = vec4(1.0, 0.843, 0.0, v_color.a); // Gold color
    vec4 mixed_color = mix(v_color, glow_color, 0.7);
    gl_FragColor = mixed_color;
    
    // Add glow effect
    float glow = 0.5 * max(0.0, 1.0 - r);
    gl_FragColor.rgb += glow * vec3(1.0, 0.8, 0.0);
}
"""

# Add shader for shell circles
CIRCLE_VERT_SHADER = """
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;
void main() {
    gl_Position = vec4(2.0 * a_position - 1.0, 0.0, 1.0);
    v_color = a_color;
}
"""

CIRCLE_FRAG_SHADER = """
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"""

# Add shaders for text rendering using simple rectangular quads
TEXT_RECT_VERT = """
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
"""

TEXT_RECT_FRAG = """
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"""

class ParticleSimulator(app.Canvas):
    def __init__(self, particles):
        app.Canvas.__init__(self, title='DWARF Particle Simulator', keys='interactive', size=(800, 800))
        self.particles = particles
        # Initialize loggers: returns dict of csv.writer and open file handles
        self.log_writers, self.log_files = init_loggers('logs')
        self.step = 0
        self.frame_count = 0
        self.particle_radius = 6.0  # Default particle size radius
        
        # Target shell radius from hydrogen parameters (as defined in AtomDetector)
        self.target_shell_radius = 216.86  # From the provided hydrogen parameters
        self.shell_width = 37.19          # Standard deviation/shell width
        self.dynamic_legend_info = None

        # Initialize atom detector
        self.atom_detector = AtomDetector()

        # Debug flags
        self.debug = False
        self.show_fps = True
        self.last_time = time.time()

        # Set up simple legend rendering
        self.legend_lines = []
        self.legend_program = gloo.Program(TEXT_RECT_VERT, TEXT_RECT_FRAG)
        
        # Enable vortex field physics in the physics core
        # This does not add artificial forces, but rather ensures the
        # physics engine is using the proper vortex field calculations
        physics_core.VORTEX_FIELD_ENABLED = True
        self.show_field_debug = False

        # Compile shaders
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.glow_program = gloo.Program(VERT_SHADER, GLOW_FRAG_SHADER)
        self.circle_program = gloo.Program(CIRCLE_VERT_SHADER, CIRCLE_FRAG_SHADER)
        
        # Initialize particle data
        self.apply_particle_data()
        
        # Set up GL state
        gloo.set_state(clear_color='black', blend=True, 
                      blend_func=('src_alpha', 'one_minus_src_alpha'))
        
        # Create a timer that calls on_timer at 60 fps
        print("Starting simulation timer...")
        self.timer = app.Timer(interval=1.0/60.0, connect=self.on_timer, start=True)
        
        print("Showing canvas...")
        self.show()
        print("Canvas shown, waiting for events...")

    def apply_particle_data(self):
        """Update particle data for visualization with support for harmonic state."""
        try:
            # Split particles into regular and harmonic states
            regular_particles = []
            harmonic_particles = []
            
            for p in self.particles:
                if self.atom_detector.is_in_harmonic_state(p['id']):
                    harmonic_particles.append(p)
                else:
                    regular_particles.append(p)
            
            # Handle regular particles
            if regular_particles:
                n = len(regular_particles)
                pos = np.array([p['pos'] / GRID_SIZE for p in regular_particles], dtype=np.float32)
                size = np.array([self.particle_radius * 2 for _ in regular_particles], dtype=np.float32)
                color = np.zeros((n, 4), dtype=np.float32)

                for i, p in enumerate(regular_particles):
                    ptype = p.get('type', 'unknown')
                    if ptype == 'proton':
                        color[i] = [1, 0, 0, 1]  # red
                    elif ptype == 'electron':
                        color[i] = [0, 0, 1, 1]  # blue
                    elif ptype == 'neutron':
                        color[i] = [0.5, 0.5, 0.5, 1]  # grey
                    else:
                        color[i] = [1, 1, 1, 1]  # white default

                self.program['a_position'] = pos
                self.program['a_color'] = color
                self.program['a_size'] = size
            
            # Handle harmonic particles
            if harmonic_particles:
                n = len(harmonic_particles)
                pos = np.array([p['pos'] / GRID_SIZE for p in harmonic_particles], dtype=np.float32)
                size = np.array([self.particle_radius * 2.5 for _ in harmonic_particles], dtype=np.float32)
                color = np.zeros((n, 4), dtype=np.float32)

                for i, p in enumerate(harmonic_particles):
                    ptype = p.get('type', 'unknown')
                    if ptype == 'proton':
                        color[i] = [1, 0, 0, 1]  # red but will be mixed with gold
                    elif ptype == 'electron':
                        color[i] = [0, 0, 1, 1]  # blue but will be mixed with gold
                    elif ptype == 'neutron':
                        color[i] = [0.5, 0.5, 0.5, 1]  # grey but will be mixed with gold
                    else:
                        color[i] = [1, 1, 1, 1]  # white default

                self.glow_program['a_position'] = pos
                self.glow_program['a_color'] = color
                self.glow_program['a_size'] = size
            
        except Exception as e:
            print(f"Error updating particle data: {e}")
            import traceback
            traceback.print_exc()
    
    def update_legend(self):
        """Update the legend with current particle information and detected elements."""
        legend_lines = []
        
        # Count particle types
        particle_counts = {
            'proton': 0,
            'electron': 0,
            'neutron': 0
        }
        
        # Track proton and electron spins for display
        proton_spins = []
        electron_spins = []
        
        for p in self.particles:
            ptype = p.get('type', 'unknown')
            if ptype in particle_counts:
                particle_counts[ptype] += 1
                
            # Collect particle spins for display
            if ptype == 'proton':
                proton_spins.append(p['spin'])
            elif ptype == 'electron':
                electron_spins.append(p['spin'])
        
        # Add particle counts to legend
        legend_lines.append("DWARF Simulator - Active Ions:")
        
        if particle_counts['proton'] > 0:
            legend_lines.append(f"Protons: {particle_counts['proton']}")
            # Display each proton's spin value
            for i, spin in enumerate(proton_spins):
                legend_lines.append(f"  Proton {i} spin: {int(spin)}")
            
        if particle_counts['electron'] > 0:
            legend_lines.append(f"Electrons: {particle_counts['electron']}")
            # Display each electron's spin value
            for i, spin in enumerate(electron_spins):
                legend_lines.append(f"  Electron {i} spin: {int(spin)}")
            
        if particle_counts['neutron'] > 0:
            legend_lines.append(f"Neutrons: {particle_counts['neutron']}")
        
        # Add shell information if available
        if self.dynamic_legend_info:
            legend_lines.append("")
            legend_lines.append(self.dynamic_legend_info)
            legend_lines.append(f"Perfect hydrogen: {round(self.target_shell_radius, 1)} units")
            legend_lines.append(f"Ref. spin: 133000")
        
        # Add detected elements to legend
        element_info = self.atom_detector.get_element_info()
        if element_info:
            legend_lines.append("")
            legend_lines.append("Detected Elements:")
            
            for nucleus_id, info in element_info.items():
                element_name = info['element']
                match_quality = info['match_quality']
                legend_lines.append(f"{element_name} (Match: {match_quality}%)")
                
                # Show metrics
                metrics = info['metrics']
                radius = round(metrics['mean_radius'], 2)
                std_dev = round(metrics['std_dev'], 2)
                legend_lines.append(f"  Radius: {radius} ± {std_dev}")
        
        # Show harmonic state info
        harmonic_count = len(self.atom_detector.harmonic_particles)
        if harmonic_count > 0:
            legend_lines.append("")
            legend_lines.append(f"Harmonic Particles: {harmonic_count}")
        
        # Show vortex field status
        legend_lines.append("")
        legend_lines.append(f"Vortex physics: {'Enabled' if physics_core.VORTEX_FIELD_ENABLED else 'Disabled'}")
        
        # Store the legend lines
        self.legend_lines = legend_lines

    def calculate_expected_orbital_radius(self, proton):
        """Calculate the expected orbital radius based on proton's current spin"""
        # Reference values that produce our target hydrogen
        base_spin = 133000  # Reference spin that gives 216.86 units
        current_spin = proton['spin']
        
        # Scale the orbital radius proportionally to spin^(1/4)
        # This power relationship approximates the DWARF vortex physics
        spin_ratio = abs(current_spin / base_spin)
        expected_radius = self.target_shell_radius * (spin_ratio ** 0.25)
        
        return expected_radius

    def create_circle_vertices(self, center_pos, radius, num_segments=60):
        """Create vertices for a circle with given center and radius."""
        # Convert to normalized device coordinates
        center_x = center_pos[0] / GRID_SIZE
        center_y = center_pos[1] / GRID_SIZE
        radius_normalized = radius / GRID_SIZE
        
        # Generate circle points
        angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
        x = center_x + radius_normalized * np.cos(angles)
        y = center_y + radius_normalized * np.sin(angles)
        
        # Create vertex array
        vertices = np.zeros((num_segments, 2), dtype=np.float32)
        vertices[:, 0] = x
        vertices[:, 1] = y
        
        return vertices
    
    def draw_proton_shells(self):
        """Draw visual shell circles around protons to show orbital boundaries."""
        # Find all protons
        protons = [p for p in self.particles if p['type'] == 'proton']
        
        for proton in protons:
            # Calculate dynamic radius based on current proton spin
            expected_radius = self.calculate_expected_orbital_radius(proton)
            
            # Draw inner shell boundary (lower bound of expected orbital)
            inner_radius = expected_radius - self.shell_width
            inner_vertices = self.create_circle_vertices(proton['pos'], inner_radius)
            inner_colors = np.ones((len(inner_vertices), 4), dtype=np.float32) * np.array([0.7, 0.1, 0.1, 0.5], dtype=np.float32)
            self.circle_program['a_position'] = inner_vertices
            self.circle_program['a_color'] = inner_colors
            self.circle_program.draw('line_loop')
            
            # Draw perfect hydrogen orbital shell (FIXED target radius - stays white)
            perfect_vertices = self.create_circle_vertices(proton['pos'], self.target_shell_radius)
            perfect_colors = np.ones((len(perfect_vertices), 4), dtype=np.float32) * np.array([0.9, 0.9, 0.9, 0.8], dtype=np.float32)
            self.circle_program['a_position'] = perfect_vertices
            self.circle_program['a_color'] = perfect_colors
            self.circle_program.draw('line_loop')
            
            # Draw outer shell boundary (upper bound of expected orbital)
            outer_radius = expected_radius + self.shell_width
            outer_vertices = self.create_circle_vertices(proton['pos'], outer_radius)
            outer_colors = np.ones((len(outer_vertices), 4), dtype=np.float32) * np.array([0.3, 0.3, 1.0, 0.5], dtype=np.float32)
            self.circle_program['a_position'] = outer_vertices
            self.circle_program['a_color'] = outer_colors
            self.circle_program.draw('line_loop')
            
            # Add to legend if the current spin differs significantly from the base spin
            if abs(proton['spin'] - 133000) > 1000:
                ratio = round(expected_radius / self.target_shell_radius, 2)
                direction = "larger" if ratio > 1 else "smaller"
                self.dynamic_legend_info = f"Current shell: {round(expected_radius, 1)} units ({ratio}x {direction})"
            else:
                self.dynamic_legend_info = None

    def draw_field_debug(self):
        """Draw debug visualization of the vortex field."""
        if not self.show_field_debug:
            return
            
        # Get all protons
        protons = [p for p in self.particles if p['type'] == 'proton']
        if not protons:
            return
            
        for proton in protons:
            # Draw field strength at different distances
            samples = 20
            for i in range(1, samples + 1):
                radius = i * (self.target_shell_radius * 2) / samples
                
                field_vertices = self.create_circle_vertices(proton['pos'], radius, 40)
                
                # Calculate field strength - approximate visualization
                # In real physics_core, this would be calculated properly
                field_strength = abs(np.sin(np.pi * radius / self.target_shell_radius))
                
                # Show field intensity with color gradient
                alpha = field_strength * 0.3
                if radius < self.target_shell_radius:
                    # Inside - green/blue gradient
                    field_colors = np.ones((len(field_vertices), 4), dtype=np.float32) * np.array([0.0, field_strength, 1.0-field_strength, alpha], dtype=np.float32)
                else:
                    # Outside - yellow/red gradient
                    field_colors = np.ones((len(field_vertices), 4), dtype=np.float32) * np.array([1.0, field_strength, 0.0, alpha], dtype=np.float32)
                
                self.circle_program['a_position'] = field_vertices
                self.circle_program['a_color'] = field_colors
                self.circle_program.draw('line_loop')

    def draw_legend(self):
        """Draw the legend text as colored rectangles since text rendering is problematic."""
        if not self.legend_lines:
            return
            
        # Draw a semi-transparent black background for the legend
        background_vertices = np.array([
            [0.52, -1.0],  # bottom left
            [1.0, -1.0],   # bottom right
            [0.52, 1.0],   # top left
            [1.0, 1.0]     # top right
        ], dtype=np.float32)
        
        background_colors = np.ones((4, 4), dtype=np.float32) * np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
        
        self.legend_program['a_position'] = background_vertices
        self.legend_program['a_color'] = background_colors
        self.legend_program.draw('triangle_strip')
        
        # Draw colored dots for each legend line as simple indicators
        y_step = 0.05
        y_pos = 0.9
        x_pos = 0.55
        
        for i, line in enumerate(self.legend_lines):
            # Skip empty lines
            if not line:
                y_pos -= y_step
                continue
                
            # Create a small indicator dot
            dot_size = 0.01
            dot_vertices = np.array([
                [x_pos, y_pos],
                [x_pos + dot_size, y_pos],
                [x_pos, y_pos - dot_size],
                [x_pos + dot_size, y_pos - dot_size]
            ], dtype=np.float32)
            
            # Choose color based on line content (simple heuristic)
            dot_color = [1.0, 1.0, 1.0, 1.0]  # Default white
            
            if "Proton" in line:
                dot_color = [1.0, 0.0, 0.0, 1.0]  # Red for protons
            elif "Electron" in line:
                dot_color = [0.0, 0.0, 1.0, 1.0]  # Blue for electrons
            elif "Neutron" in line:
                dot_color = [0.5, 0.5, 0.5, 1.0]  # Grey for neutrons
            elif "Hydrogen" in line or "Elements" in line:
                dot_color = [1.0, 0.843, 0.0, 1.0]  # Gold for elements/hydrogen
            elif "shell" in line:
                dot_color = [0.0, 1.0, 0.0, 1.0]  # Green for shell info
                
            dot_colors = np.ones((4, 4), dtype=np.float32) * np.array(dot_color, dtype=np.float32)
            
            self.legend_program['a_position'] = dot_vertices
            self.legend_program['a_color'] = dot_colors
            self.legend_program.draw('triangle_strip')
            
            # Move down for the next line
            y_pos -= y_step

    def on_draw(self, event):
        if self.debug:
            print(f"Draw event at step {self.step}")
        gloo.clear()
        
        # Draw field debug visualization first (if enabled)
        self.draw_field_debug()
        
        # Draw shell circles around protons (so they're behind particles)
        self.draw_proton_shells()
        
        # Update particle visualization
        self.apply_particle_data()
        
        # Draw regular particles
        if len(self.particles) > 0 and not all(self.atom_detector.is_in_harmonic_state(p['id']) for p in self.particles):
            self.program.draw('points')
        
        # Draw harmonic particles with glow effect
        if any(self.atom_detector.is_in_harmonic_state(p['id']) for p in self.particles):
            self.glow_program.draw('points')
        
        # Draw the legend (simplified version)
        self.draw_legend()

    def on_timer(self, event):
        # Calculate FPS
        if self.show_fps and self.frame_count % 60 == 0:
            now = time.time()
            elapsed = now - self.last_time
            fps = 60 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}, Step: {self.step}")
            self.last_time = now
        
        # Run simulation step - physics_core handles the vortex field interactions
        simulate_step(self.step, self.particles, self.log_writers)
        
        # Update atom detector
        self.atom_detector.update(self.particles, self.step)
        
        # Update legend
        self.update_legend()
        
        self.step += 1
        self.frame_count += 1
        
        # Debug output
        if self.debug and self.step % 60 == 0:
            for i, p in enumerate(self.particles):
                print(f"Particle {i} ({p['type']}): pos={p['pos']}, vel={p['vel']}")
            
            if self.atom_detector.harmonic_particles:
                print(f"Harmonic particles: {self.atom_detector.harmonic_particles}")
                
            if self.atom_detector.get_element_info():
                print(f"Detected elements: {self.atom_detector.get_element_info()}")
        
        # Report to console when harmonic state is achieved
        element_info = self.atom_detector.get_element_info()
        if element_info and self.frame_count % 60 == 0:
            for nucleus_id, info in element_info.items():
                print(f"Detected {info['element']} with {info['match_quality']}% match at step {self.step}")
                
        # Print legend to console periodically
        if self.frame_count % 120 == 0:
            print("\n=== DWARF Simulator Status ===")
            for line in self.legend_lines:
                print(line)
            print("============================\n")
        
        # Request canvas update
        self.update()

    def on_key_press(self, event):
        if event.key == '1':
            physics_core.BOUNDARY_MODE = BOUNDARY_WRAP
            print("Boundary mode: WRAP (toroidal/periodic)")
        elif event.key == '2':
            physics_core.BOUNDARY_MODE = BOUNDARY_REFLECT
            print("Boundary mode: REFLECT (energy conserving)")
        elif event.key == '3':
            physics_core.BOUNDARY_MODE = BOUNDARY_DAMP
            print("Boundary mode: DAMP (energy loss on collision)")
        elif event.key == 'b':
            self.debug = not self.debug
            print(f"Debug output {'enabled' if self.debug else 'disabled'}")
        elif event.key == 'f':
            self.show_fps = not self.show_fps
            print(f"FPS display {'enabled' if self.show_fps else 'disabled'}")
        elif event.key == 'r':
            print("Resetting simulation...")
            self.step = 0
            self.particles = generate_default_particles()
            for i, p in enumerate(self.particles):
                p['id'] = i
            self.atom_detector = AtomDetector()  # Reset the detector
        elif event.key == 'v':
            # Toggle vortex field
            physics_core.VORTEX_FIELD_ENABLED = not physics_core.VORTEX_FIELD_ENABLED
            print(f"Vortex field physics {'enabled' if physics_core.VORTEX_FIELD_ENABLED else 'disabled'}")
        elif event.key == 'g':
            # Toggle field visualization
            self.show_field_debug = not self.show_field_debug
            print(f"Field visualization {'enabled' if self.show_field_debug else 'disabled'}")
            
        # Proton spin controls - Arrow Keys
        elif event.key == 'Up':
            # Increase proton spin by 2%
            for p in self.particles:
                if p['type'] == 'proton':
                    p['spin'] *= 1.02  # Fine adjustment
                    print(f"Proton spin increased to {int(p['spin'])}")
        elif event.key == 'Down':
            # Decrease proton spin by 2%
            for p in self.particles:
                if p['type'] == 'proton':
                    p['spin'] /= 1.02  # Fine adjustment
                    print(f"Proton spin decreased to {int(p['spin'])}")
        elif event.key == 'Right':
            # Increase proton spin by 10%
            for p in self.particles:
                if p['type'] == 'proton':
                    p['spin'] *= 1.1  # Coarse adjustment
                    print(f"Proton spin increased to {int(p['spin'])}")
        elif event.key == 'Left':
            # Decrease proton spin by 10%
            for p in self.particles:
                if p['type'] == 'proton':
                    p['spin'] /= 1.1  # Coarse adjustment
                    print(f"Proton spin decreased to {int(p['spin'])}")
                    
        # Electron spin controls - WASD keys
        elif event.key == 'w':
            # Increase electron spin by 2%
            for p in self.particles:
                if p['type'] == 'electron':
                    p['spin'] *= 1.02  # Fine adjustment
                    print(f"Electron spin increased to {int(p['spin'])}")
        elif event.key == 's':
            # Decrease electron spin by 2%
            for p in self.particles:
                if p['type'] == 'electron':
                    p['spin'] /= 1.02  # Fine adjustment
                    print(f"Electron spin decreased to {int(p['spin'])}")
        elif event.key == 'd':
            # Increase electron spin by 10%
            for p in self.particles:
                if p['type'] == 'electron':
                    p['spin'] *= 1.1  # Coarse adjustment
                    print(f"Electron spin increased to {int(p['spin'])}")
        elif event.key == 'a':
            # Decrease electron spin by 10%
            for p in self.particles:
                if p['type'] == 'electron':
                    p['spin'] /= 1.1  # Coarse adjustment
                    print(f"Electron spin decreased to {int(p['spin'])}")

    def on_close(self, event):
        # Close all log files when the application closes
        print("Closing application and log files...")
        for file_handle in self.log_files.values():
            file_handle.close()

if __name__ == '__main__':
    print("Legend ▶ red = proton, blue = electron, grey = neutron")
    print("Controls:")
    print("  1/2/3 - Change boundary mode")
    print("  r - Reset simulation")
    print("  b - Toggle debug output")
    print("  f - Toggle FPS display")
    print("  v - Toggle vortex field physics")
    print("  g - Toggle field visualization")
    print("  Spin Controls:")
    print("    Arrow Keys - Adjust proton spin:")
    print("      ↑/↓ - Fine adjustment (±2%)")
    print("      ←/→ - Coarse adjustment (±10%)")
    print("    WASD Keys - Adjust electron spin:")
    print("      W/S - Fine adjustment (±2%)")
    print("      A/D - Coarse adjustment (±10%)")
    
    # Generate particles
    particles = generate_default_particles()
    
    # Assign particle IDs to ensure proper logging
    for i, p in enumerate(particles):
        p['id'] = i
    
    # Print initial positions
    print("Initial particle positions:")
    for i, p in enumerate(particles):
        print(f"Particle {i} ({p['type']}): pos={p['pos']}, vel={p['vel']}, spin={int(p['spin'])}")
    
    try:
        # Initialize particle simulator
        print("Initializing simulator...")
        canvas = ParticleSimulator(particles)
        
        # Run the application with debug info
        print("Starting vispy app...")
        app.run()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()