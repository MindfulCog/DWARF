
import numpy as np
from config import DT, GLOBAL_DRAG
from integrator import integrate
from logger import log_particle_state

def apply_fluid_drag(p):
    p['vel'] *= GLOBAL_DRAG

def simulate_step(step, particles, logs):
    for p in particles:
        apply_fluid_drag(p)
    integrate(particles, DT)
    log_particle_state(step, particles, logs)

