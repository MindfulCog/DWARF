import csv
import os

def open_logs():
    os.makedirs("logs", exist_ok=True)
    logs = {
        'positions': open_log_file('logs/positions.csv', ['step', 'id', 'x_m', 'y_m']),
        'velocities': open_log_file('logs/velocities.csv', ['step', 'id', 'vx_m/s', 'vy_m/s']),
        'distances': open_log_file('logs/distances.csv', ['step', 'p1', 'p2', 'distance_m']),
        'memory': open_log_file('logs/memory.csv', ['step', 'id', 'memory_x', 'memory_y']),
        'curl': open_log_file('logs/curl.csv', ['step', 'id', 'curl']),
        'angular_momentum': open_log_file('logs/angular_momentum.csv', ['step', 'id', 'angular_momentum']),
    }
    return logs

def open_log_file(filename, headers):
    file = open(filename, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(headers)
    return writer

def log_particle_state(step, particles, logs):
    for pid, p in enumerate(particles):
        x, y = p['pos']
        vx, vy = p['vel']
        logs['positions'].writerow([step, pid, x, y])
        logs['velocities'].writerow([step, pid, vx, vy])
        if 'memory' in p:
            logs['memory'].writerow([step, pid, *p['memory']])
        if 'curl' in p:
            logs['curl'].writerow([step, pid, p['curl']])
        if 'angular_momentum' in p:
            logs['angular_momentum'].writerow([step, pid, p['angular_momentum']])
