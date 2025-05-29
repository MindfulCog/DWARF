### main.py
import sys
import time  # Standard Python time module
import numpy as np
from vispy import app, gloo
from config import GRID_SIZE, DT
from physics_core import simulate_step, BOUNDARY_WRAP, BOUNDARY_REFLECT, BOUNDARY_DAMP
import physics_core
from particle_types import generate_default_particles
from logger import init_loggers

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

class ParticleSimulator(app.Canvas):
    def __init__(self, particles):
        app.Canvas.__init__(self, title='DWARF Particle Simulator', keys='interactive', size=(800, 800))
        self.particles = particles
        # Initialize loggers: returns dict of csv.writer and open file handles
        self.log_writers, self.log_files = init_loggers('logs')
        self.step = 0
        self.frame_count = 0
        self.particle_radius = 6.0  # Default particle size radius

        # Debug flags
        self.debug = False
        self.show_fps = True
        self.last_time = time.time()  # Use standard time module

        # Compile shaders
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.apply_particle_data()  # Initialize visualization data
        
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
        """Update particle data for visualization (from visualizer_gpu.py)"""
        try:
            n = len(self.particles)
            pos = np.array([p['pos'] / GRID_SIZE for p in self.particles], dtype=np.float32)
            size = np.array([self.particle_radius * 2 for _ in self.particles], dtype=np.float32)
            color = np.zeros((n, 4), dtype=np.float32)

            for i, p in enumerate(self.particles):
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
        except Exception as e:
            print(f"Error updating particle data: {e}")
            import traceback
            traceback.print_exc()

    def on_draw(self, event):
        if self.debug:
            print(f"Draw event at step {self.step}")
        gloo.clear()
        self.apply_particle_data()
        self.program.draw('points')

    def on_timer(self, event):
        # Calculate FPS
        if self.show_fps and self.frame_count % 60 == 0:
            now = time.time()  # Use standard time module
            elapsed = now - self.last_time
            fps = 60 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}, Step: {self.step}")
            self.last_time = now
        
        # Run simulation step
        simulate_step(self.step, self.particles, self.log_writers)
        self.step += 1
        self.frame_count += 1
        
        # Debug output
        if self.debug and self.step % 60 == 0:
            for i, p in enumerate(self.particles):
                print(f"Particle {i} ({p['type']}): pos={p['pos']}, vel={p['vel']}")
        
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
        elif event.key == 'd':
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

    def on_close(self, event):
        # Close all log files when the application closes
        print("Closing application and log files...")
        for file_handle in self.log_files.values():
            file_handle.close()

if __name__ == '__main__':
    print("Legend ▶ red = proton, blue = electron, grey = neutron")
    
    # Generate particles
    particles = generate_default_particles()
    
    # Assign particle IDs to ensure proper logging
    for i, p in enumerate(particles):
        p['id'] = i
    
    # Print initial positions
    print("Initial particle positions:")
    for i, p in enumerate(particles):
        print(f"Particle {i} ({p['type']}): pos={p['pos']}, vel={p['vel']}")
    
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