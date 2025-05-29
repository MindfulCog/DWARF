import sys
import numpy as np
import vispy.app
import vispy.gloo as gloo
from vispy.app import Timer
from physics_core import simulate_step
from particle_types import generate_default_particles
from logger import open_logs, log_particle_state

class ParticleSimulator(vispy.app.Canvas):
    def __init__(self, particles):
        vispy.app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        self.particles = particles
        self.logs = open_logs()
        self.step = 0

        self.program = gloo.Program(self.vertex_shader(), self.fragment_shader())
        self.timer = Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_draw(self, event):
        gloo.clear('black')
        self.update_visualizer()

    def on_timer(self, event):
        simulate_step(self.step, self.particles, self.logs)
        self.step += 1
        self.update()

    def update_visualizer(self):
        pos_data = np.array([p['pos'] for p in self.particles], dtype=np.float32)
        size_data = np.array([p.get('size', 5.0) for p in self.particles], dtype=np.float32)
        
        # FIXED: Ensure color_data has only 3 channels (RGB), not 4
        color_data = np.array([p.get('color', [1.0, 1.0, 1.0])[:3] for p in self.particles], dtype=np.float32)

        self.program['a_position'] = pos_data
        self.program['a_color'] = color_data
        self.program['a_size'] = size_data
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
    canvas = ParticleSimulator(generate_default_particles())
    vispy.app.run()
