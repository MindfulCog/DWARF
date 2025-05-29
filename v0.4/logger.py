import csv
import os

def open_logs(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    writers = {}
    # Define the filenames and headers
    files = {
        'positions': ('positions.csv', ['step', 'id', 'x', 'y']),
        'velocities': ('velocities.csv', ['step', 'id', 'vx', 'vy']),
        'distance': ('distances.csv', ['step', 'id1', 'id2', 'distance']),
        'memory': ('memory.csv', ['step', 'id', 'memory_x', 'memory_y']),
        'curl': ('curl.csv', ['step', 'id', 'curl']),
        'angular_momentum': ('angular_momentum.csv', ['step', 'id', 'L']),
    }
    for key, (fname, headers) in files.items():
        path = os.path.join(log_dir, fname)
        f = open(path, mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        writers[key] = writer
    return writers


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
