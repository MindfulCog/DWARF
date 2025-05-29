
import numpy as np

def integrate(particles, dt):
    for p in particles:
        p['pos'] += p['vel'] * dt
