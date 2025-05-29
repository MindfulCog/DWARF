
from vispy import app, gloo
import numpy as np

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
        app.Canvas.__init__(self, title='DWARF Atomic Simulator', keys='interactive', size=(800, 800))
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

    def on_draw(self, event):
        gloo.clear()
        self.apply_particle_data()
        self.program.draw('points')

    def on_timer(self, event):
        from physics_core import simulate_step
        simulate_step(self.particles, self.dt)
        self.update()
