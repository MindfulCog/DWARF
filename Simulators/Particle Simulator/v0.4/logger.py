import csv
import os
import time

class FlushableWriter:
    """Wrapper for CSV writer that provides flushing capability"""
    def __init__(self, file):
        self.file = file
        self.writer = csv.writer(file)
        
    def writerow(self, row):
        return self.writer.writerow(row)
        
    def writerows(self, rows):
        return self.writer.writerows(rows)
        
    def flush(self):
        self.file.flush()

def init_loggers(log_dir='logs'):
    """Initialize loggers and return writers and file handles"""
    os.makedirs(log_dir, exist_ok=True)
    writers = {}
    files = {}
    
    # Define the filenames and headers - ADDING THE MISSING ONES
    log_files = {
        'positions': ('positions.csv', ['step', 'id', 'x', 'y']),
        'velocities': ('velocities.csv', ['step', 'id', 'vx', 'vy']),
        'distance': ('distances.csv', ['step', 'id1', 'id2', 'distance']),
        'memory': ('memory.csv', ['step', 'id', 'memory_x', 'memory_y']),
        'curl': ('curl.csv', ['step', 'id', 'curl']),
        'angular_momentum': ('angular_momentum.csv', ['step', 'id', 'L']),
        # Add the missing log files
        'memory_gradient': ('memory_gradient.csv', ['step', 'id', 'grad_x', 'grad_y']),
        'memory_vector': ('memory_vector.csv', ['step', 'id', 'vec_x', 'vec_y']),
        'net_force': ('net_force.csv', ['step', 'id', 'force_x', 'force_y']),
    }
    
    for key, (fname, headers) in log_files.items():
        path = os.path.join(log_dir, fname)
        f = open(path, mode='w', newline='')
        writer = FlushableWriter(f)
        writer.writerow(headers)
        writers[key] = writer
        files[key] = f
    
    return writers, files

def log_step(step, particles, log_writers):
    """Log particle state at current simulation step"""
    log_particle_state(step, particles, log_writers)

def log_particle_state(step, particles, writers):
    for p in particles:
        pid = p.get('id', None)
        # Position
        if 'positions' in writers:
            writers['positions'].writerow([step, pid, float(p['pos'][0]), float(p['pos'][1])])
        # Velocity
        if 'velocities' in writers:
            writers['velocities'].writerow([step, pid, float(p['vel'][0]), float(p['vel'][1])])
        # Distance (e.g. electron to proton)
        if 'distance' in writers and p['type']=='electron':
            # assume distance from origin (proton at id 0 at origin)
            d = float((p['pos']**2).sum()**0.5)
            writers['distance'].writerow([step, pid, 0, d])
        # Memory
        if 'memory' in writers and 'field_memory' in p:
            writers['memory'].writerow([step, pid, float(p['field_memory'][0]), float(p['field_memory'][1])])
        # Curl
        if 'curl' in writers and 'curl' in p:
            writers['curl'].writerow([step, pid, float(p['curl'])])
        # Angular momentum
        if 'angular_momentum' in writers and 'angular_momentum' in p:
            writers['angular_momentum'].writerow([step, pid, float(p['angular_momentum'])])
        
        # Add logging for the missing values
        # Memory Gradient
        if 'memory_gradient' in writers and 'memory_gradient' in p:
            writers['memory_gradient'].writerow([step, pid, float(p['memory_gradient'][0]), float(p['memory_gradient'][1])])
        # Memory Vector
        if 'memory_vector' in writers and 'memory_vector' in p:
            writers['memory_vector'].writerow([step, pid, float(p['memory_vector'][0]), float(p['memory_vector'][1])])
        # Net Force
        if 'net_force' in writers and 'net_force' in p:
            writers['net_force'].writerow([step, pid, float(p['net_force'][0]), float(p['net_force'][1])])
    
    # Flush all writers to ensure data is written immediately
    for writer in writers.values():
        writer.flush()
    