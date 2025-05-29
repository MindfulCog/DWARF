### main.py
import sys
import numpy as np
import vispy.app
import vispy.gloo as gloo
from vispy.app import Timer
from physics_core import simulate_step
from particle_types import generate_default_particles
from logger import init_loggers

class ParticleSimulator(vispy.app.Canvas):
    def __init__(self, particles):
        super().__init__(keys='interactive', size=(800, 800), title='DWARF Particle Simulator')
        self.particles = particles
        # Initialize loggers: returns dict of csv.writer and open file handles
        self.log_writers, self.log_files = init_loggers('logs')
        self.step = 0

        # Compile shaders
        self.program = gloo.Program(self.vertex_shader(), self.fragment_shader())
        self.timer = Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear('black')
        self.update_visualizer()

    def on_timer(self, event):
        # simulate_step handles physics + logging internally
        simulate_step(self.step, self.particles, self.log_writers)
        self.step += 1
        self.update()

    def update_visualizer(self):
        pos_data = np.array([p['pos'] for p in self.particles], dtype=np.float32)
        size_data = np.array([p.get('size', 6.0) for p in self.particles], dtype=np.float32)
        # Ensure color is RGB
        color_data = np.array([p.get('color', (1.0,1.0,1.0,1.0))[:3] for p in self.particles], dtype=np.float32)

        self.program['a_position'] = pos_data
        self.program['a_color']    = color_data
        self.program['a_size']     = size_data
        self.program.draw('points')

    def vertex_shader(self):
        return '''
        attribute vec2 a_position;
        attribute vec3 a_color;
        attribute float a_size;
        varying vec3 v_color;
        void main() {
            gl_Position = vec4((a_position / 400.0) - 1.0, 0.0, 1.0);
            gl_PointSize = a_size;
            v_color = a_color;
        }
        '''

    def fragment_shader(self):
        return '''
        varying vec3 v_color;
        void main() {
            gl_FragColor = vec4(v_color, 1.0);
        }
        '''

if __name__ == '__main__':
    print("Legend ▶ red = proton, blue = electron, grey = neutron")
    particles = generate_default_particles()
    canvas = ParticleSimulator(particles)
    vispy.app.run()

### visualizer_gpu.py
from vispy import app, gloo
import numpy as np
from physics_core import simulate_step  # moved to top-level import

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

class ParticleVisualizer(app.Canvas):
    def __init__(self, particles, grid_size, particle_radius, dt=1e-14):
        super().__init__(title='DWARF Atomic Simulator', keys='interactive', size=(800, 800))
        self.particles = particles
        self.grid_size = grid_size
        self.particle_radius = particle_radius
        self.dt = dt
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.apply_particle_data()

        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.timer = app.Timer(interval=0, connect=self.on_timer, start=True)
        self.show()

    def apply_particle_data(self):
        n = len(self.particles)
        pos = np.array([p['pos'] / self.grid_size for p in self.particles], dtype=np.float32)
        size = np.array([self.particle_radius * 2 for _ in self.particles], dtype=np.float32)
        color = np.zeros((n, 4), dtype=np.float32)

        for i, p in enumerate(self.particles):
            ptype = p.get('type', 'unknown')
            if ptype == 'proton':
                color[i] = [1, 0, 0, 1]
            elif ptype == 'electron':
                color[i] = [0, 0, 1, 1]
            elif ptype == 'neutron':
                color[i] = [0.5, 0.5, 0.5, 1]
            else:
                color[i] = [1, 1, 1, 1]

        self.program['a_position'] = pos
        self.program['a_color']    = color
        self.program['a_size']     = size

    def on_draw(self, event):
        gloo.clear()
        self.apply_particle_data()
        self.program.draw('points')

    def on_timer(self, event):
        # Use imported simulate_step
        simulate_step(self.particles, self.dt)
        self.update()
