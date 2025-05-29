import numpy as np
from config import GRID_SIZE, DT, DWARF_FORCE_EXPONENT, GLOBAL_DRAG
from logger import log_step

# Boundary handling constants
BOUNDARY_WRAP = 'wrap'      # Particle wraps around to opposite side (no energy loss)
BOUNDARY_REFLECT = 'reflect'  # Particle bounces off walls (conserves energy)
BOUNDARY_DAMP = 'damp'      # Particle bounces with energy loss
BOUNDARY_MODE = BOUNDARY_WRAP  # Default mode

def extract_pos(p):
    """Extract position from particle regardless of representation"""
    if isinstance(p, dict) and 'pos' in p:
        return p['pos']
    elif hasattr(p, 'pos'):
        return p.pos
    return None

def simulate_step(step, particles, logs=None):
    """Main simulation step function that updates particle states"""
    update_memory(particles)
    apply_forces(particles)
    apply_spin_effects(particles)  # Apply spin effects
    update_positions(particles)
    
    # Add initial velocity if particles aren't moving
    if step == 0:
        initialize_velocities(particles)
    
    if logs is not None:
        log_step(step, particles, logs)

def initialize_velocities(particles):
    """Make sure particles have some initial velocity if they're all static"""
    all_zero = True
    for p in particles:
        if isinstance(p, dict) and 'vel' in p:
            if not np.all(p['vel'] == 0):
                all_zero = False
                break
    
    if all_zero and len(particles) > 1:
        print("Adding initial velocities...")
        # Give electrons some initial velocity
        for p in particles:
            if p['type'] == 'electron':
                center = np.array([GRID_SIZE/2, GRID_SIZE/2])
                direction = p['pos'] - center
                perpendicular = np.array([-direction[1], direction[0]])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                p['vel'] = perpendicular * 30.0

def update_memory(particles):
    # Simple memory field update implementation
    for p in particles:
        if isinstance(p, dict):
            if 'field_memory' not in p:
                p['field_memory'] = np.array([0.0, 0.0])
            
            # Record particle's movement in memory
            if 'vel' in p:
                memory_decay = 0.95
                p['field_memory'] = p['field_memory'] * memory_decay + p['vel'] * 0.05
                
                # Create memory gradient (rate of change of field memory)
                if 'prev_field_memory' not in p:
                    p['prev_field_memory'] = p['field_memory'].copy()
                    p['memory_gradient'] = np.array([0.0, 0.0])
                else:
                    p['memory_gradient'] = (p['field_memory'] - p['prev_field_memory']) / DT
                    p['prev_field_memory'] = p['field_memory'].copy()
                
                # Create memory vector (direction and magnitude of memory)
                memory_mag = np.linalg.norm(p['field_memory'])
                if memory_mag > 0:
                    p['memory_vector'] = p['field_memory'] / memory_mag * memory_mag**2
                else:
                    p['memory_vector'] = np.array([0.0, 0.0])

def apply_forces(particles):
    for i, p in enumerate(particles):
        pos_p = extract_pos(p)
        if pos_p is None:
            continue

        total_force = np.array([0.0, 0.0])
        charge_p = p.get('charge', 0) if isinstance(p, dict) else 0

        for j, other in enumerate(particles):
            if i == j:
                continue
            pos_o = extract_pos(other)
            if pos_o is None:
                continue
                
            charge_o = other.get('charge', 0) if isinstance(other, dict) else 0
            
            # Handle boundary wrap-around for force calculations when in WRAP mode
            rel_pos = calculate_displacement(pos_p, pos_o)
            r_squared = np.sum(rel_pos**2)
            r = np.sqrt(r_squared) + 1e-9
            
            coulomb_factor = charge_p * charge_o / r_squared
            force_direction = rel_pos / r
            force = force_direction * coulomb_factor * 100000
            total_force += force

        if isinstance(p, dict) and 'vel' in p:
            mass = p.get('mass', 1.0)
            p['vel'] += total_force * DT / mass
            p['vel'] *= GLOBAL_DRAG
            
            # Store net force for logging
            p['net_force'] = total_force

def calculate_displacement(pos1, pos2):
    """Calculate displacement vector considering boundary conditions"""
    disp = pos2 - pos1
    
    if BOUNDARY_MODE == BOUNDARY_WRAP:
        # For wrap mode, use shortest path across boundaries
        for i in range(len(disp)):
            if disp[i] > GRID_SIZE/2:
                disp[i] -= GRID_SIZE
            elif disp[i] < -GRID_SIZE/2:
                disp[i] += GRID_SIZE
                
    return disp

def apply_spin_effects(particles):
    """Apply effects from particle spin"""
    for p in particles:
        if not isinstance(p, dict) or 'spin' not in p:
            continue
            
        # Skip particles with zero spin
        if p['spin'] == 0:
            continue
            
        # Calculate curl based on spin
        spin = p['spin']
        pos = p.get('pos', np.array([0, 0]))
        vel = p.get('vel', np.array([0, 0]))
        
        # Set curl value (vorticity)
        p['curl'] = spin / 1000000.0  # Scale down for reasonable values
        
        # Generate spin-induced force (vortex effect)
        for other in particles:
            if p is other:
                continue
                
            other_pos = other.get('pos', None)
            if other_pos is None:
                continue
                
            # Vector from p to other, considering boundary conditions
            r_vec = calculate_displacement(pos, other_pos)
            r_mag = np.linalg.norm(r_vec) + 1e-9
            
            if r_mag > 100:  # Only affect nearby particles
                continue
                
            # Create perpendicular vector (for rotational effect)
            perp_vec = np.array([-r_vec[1], r_vec[0]])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            
            # Spin force decreases with distance
            spin_force = perp_vec * (p['curl'] / (r_mag**2 + 1))
            
            # Apply the spin-induced force to the other particle
            if 'vel' in other:
                other['vel'] += spin_force * DT
                # Update net force
                if 'net_force' in other:
                    other['net_force'] += spin_force
                else:
                    other['net_force'] = spin_force
                
        # Calculate angular momentum (L = r × p)
        center = np.array([GRID_SIZE/2, GRID_SIZE/2])
        r_vec = calculate_displacement(center, pos)
        p_vec = p.get('mass', 1.0) * vel
        
        # Cross product in 2D: r_x*p_y - r_y*p_x
        angular_momentum = r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0]
        p['angular_momentum'] = angular_momentum

def update_positions(particles):
    for p in particles:
        if isinstance(p, dict) and 'pos' in p and 'vel' in p:
            velocity_impact = p['vel'] * DT
            p['pos'] += velocity_impact
            
            # Apply boundary conditions based on selected mode
            apply_boundary_conditions(p)
            
        elif hasattr(p, 'pos') and hasattr(p, 'vel'):
            p.pos += p.vel * DT
            
def apply_boundary_conditions(p):
    """Apply appropriate boundary conditions to a particle"""
    if BOUNDARY_MODE == BOUNDARY_WRAP:
        # Toroidal/wrap-around boundaries (periodic)
        for i in range(2):
            if p['pos'][i] < 0:
                p['pos'][i] = p['pos'][i] % GRID_SIZE
            elif p['pos'][i] >= GRID_SIZE:
                p['pos'][i] = p['pos'][i] % GRID_SIZE
    
    elif BOUNDARY_MODE == BOUNDARY_REFLECT:
        # Perfect reflection (energy conserving)
        for i in range(2):
            if p['pos'][i] < 0:
                p['pos'][i] = -p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i]  # Perfectly reflect velocity
            elif p['pos'][i] >= GRID_SIZE:
                p['pos'][i] = 2*GRID_SIZE - p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i]  # Perfectly reflect velocity
    
    elif BOUNDARY_MODE == BOUNDARY_DAMP:
        # Damping reflection (energy loss)
        for i in range(2):
            if p['pos'][i] < 0:
                p['pos'][i] = -p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i] * 0.9  # 10% energy loss
            elif p['pos'][i] >= GRID_SIZE:
                p['pos'][i] = 2*GRID_SIZE - p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i] * 0.9  # 10% energy loss