import numpy as np
from config import GRID_SIZE, DT, DWARF_FORCE_EXPONENT, GLOBAL_DRAG, SATURATION_LIMIT
from logger import log_step

# Boundary handling constants
BOUNDARY_WRAP = 'wrap'      # Particle wraps around to opposite side (no energy loss)
BOUNDARY_REFLECT = 'reflect'  # Particle bounces off walls (conserves energy)
BOUNDARY_DAMP = 'damp'      # Particle bounces with energy loss
BOUNDARY_MODE = BOUNDARY_WRAP  # Default mode

# Flag to enable/disable vortex field physics
VORTEX_FIELD_ENABLED = True

# Physical constants - independent of timestep
MEMORY_DECAY_RATE = 0.018      # Per time unit (not per step)
MEMORY_ACCUMULATION_RATE = 0.05  # Per time unit  
DRAG_RATE = -np.log(GLOBAL_DRAG)  # Convert per-step drag to continuous rate

def extract_pos(p):
    """Extract position from particle regardless of representation"""
    if isinstance(p, dict) and 'pos' in p:
        return p['pos']
    elif hasattr(p, 'pos'):
        return p.pos
    return None

def simulate_step(step, particles, logs=None):
    """Main simulation step function that updates particle states"""
    try:
        if step % 100 == 0:
            print(f"Simulation step {step}")
        
        # Add initial velocity if particles aren't moving
        if step == 0:
            initialize_velocities(particles)
            
        update_memory(particles)
        
        # Enhanced vortex field interactions in core physics 
        # when VORTEX_FIELD_ENABLED is true
        apply_forces(particles)
        apply_spin_effects(particles)
        update_positions(particles)
        
        if logs is not None:
            log_step(step, particles, logs)
            
        return True
    except Exception as e:
        print(f"Error in simulation step {step}: {e}")
        import traceback
        traceback.print_exc()
        return False

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
                if np.linalg.norm(perpendicular) > 0:
                    perpendicular = perpendicular / np.linalg.norm(perpendicular)
                    p['vel'] = perpendicular * 30.0  # Physical velocity, independent of DT
                    print(f"Set velocity for electron: {p['vel']}")

def update_memory(particles):
    """Update particle memory fields with proper time-scaling"""
    for p in particles:
        if isinstance(p, dict):
            if 'field_memory' not in p:
                p['field_memory'] = np.array([0.0, 0.0])
            
            # Record particle's movement in memory - TIMESTEP SCALED
            if 'vel' in p:
                # Time-scaled memory decay (e^(-rate*dt) is the physical decay)
                memory_decay = np.exp(-MEMORY_DECAY_RATE * DT)
                
                # Time-scaled memory accumulation
                memory_accumulation = MEMORY_ACCUMULATION_RATE * DT
                
                p['field_memory'] = p['field_memory'] * memory_decay + p['vel'] * memory_accumulation
                
                # Create memory gradient (rate of change) - already physical
                if 'prev_field_memory' not in p:
                    p['prev_field_memory'] = p['field_memory'].copy()
                    p['memory_gradient'] = np.array([0.0, 0.0])
                else:
                    # div by DT makes this a true derivative (rate of change per unit time)
                    p['memory_gradient'] = (p['field_memory'] - p['prev_field_memory']) / max(DT, 1e-10)
                    p['prev_field_memory'] = p['field_memory'].copy()
                
                # Create memory vector - physical equation independent of timestep
                memory_mag = np.linalg.norm(p['field_memory'])
                if memory_mag > 1e-10:
                    try:
                        safe_exponent = min(DWARF_FORCE_EXPONENT, 3.0)  # Cap exponent for stability
                        p['memory_vector'] = p['field_memory'] / memory_mag * (memory_mag**safe_exponent)
                    except (OverflowError, FloatingPointError):
                        print(f"Warning: Numerical overflow in memory vector calculation: {memory_mag}")
                        p['memory_vector'] = p['field_memory'] * 10.0
                else:
                    p['memory_vector'] = np.array([0.0, 0.0])
                    
            # Apply saturation limit - physical limit independent of timestep
            memory_mag = np.linalg.norm(p['field_memory'])
            if memory_mag > SATURATION_LIMIT:
                p['field_memory'] = p['field_memory'] / memory_mag * SATURATION_LIMIT

def apply_forces(particles):
    """Apply forces between particles with physical time scaling"""
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
            r = np.sqrt(r_squared) + 1e-9  # Prevent division by zero
            
            # Base Coulomb force - physical force law independent of timestep
            coulomb_factor = charge_p * charge_o / r_squared
            force_direction = rel_pos / r
            force = force_direction * coulomb_factor * 100000  # Force scaling constant
            
            # Apply DWARF scaling - physical force law with r^2.22 dependency 
            try:
                if DWARF_FORCE_EXPONENT != 2.0:
                    # Adjust for DWARF exponent difference from standard inverse square law
                    dwarf_adjustment = r**(2.0 - DWARF_FORCE_EXPONENT)
                    # Cap the adjustment factor for numerical stability
                    dwarf_adjustment = np.clip(dwarf_adjustment, 0.1, 10.0)
                    force = force * dwarf_adjustment
            except (OverflowError, FloatingPointError):
                print(f"Warning: Numerical issue in force calculation at distance {r}")
            
            total_force += force

            # Memory coupling - with physical time scaling
            if 'memory_vector' in p and 'memory_vector' in other:
                try:
                    # Physical memory interaction rate - independent of timestep
                    memory_coupling = 0.001 * np.linalg.norm(other['memory_vector'])
                    memory_force = force_direction * memory_coupling
                    total_force += memory_force
                except (OverflowError, FloatingPointError, ValueError):
                    print("Warning: Numerical issue in memory coupling")

        if isinstance(p, dict) and 'vel' in p:
            mass = max(p.get('mass', 1.0), 0.1)  # Ensure mass is never too small
            
            # Store net force for logging
            p['net_force'] = total_force
            
            # Update velocity - F = ma, so dv = F/m * dt
            p['vel'] += total_force * DT / mass
            
            # Apply global drag with proper time scaling
            p['vel'] *= np.exp(-DRAG_RATE * DT)  # Physical decay rate e^(-drag_rate * dt)

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
    """Apply effects from particle spin with physical time scaling"""
    for p in particles:
        if not isinstance(p, dict) or 'spin' not in p:
            continue
            
        # Skip particles with zero spin
        if p['spin'] == 0:
            continue
            
        # Calculate curl based on spin - physical value independent of timestep
        spin = p['spin']
        pos = p.get('pos', np.array([0, 0]))
        vel = p.get('vel', np.array([0, 0]))
        
        # Set curl value (vorticity) - physical property
        p['curl'] = spin / 1000000.0  # Scale down for reasonable values
        
        # Apply curl-induced drift (DWARF turbulence effect) - WITH TIME SCALING
        if 'vel' in p and np.linalg.norm(p['vel']) > 1e-10:
            try:
                tangent_vector = np.array([-p['vel'][1], p['vel'][0]])
                tangent_norm = np.linalg.norm(tangent_vector)
                if tangent_norm > 1e-10:
                    # Physical drift rate, scaled by DT
                    drift_rate = 0.1  # Per time unit
                    tangent_drift = tangent_vector / tangent_norm * p['curl'] * drift_rate * DT
                    p['vel'] += tangent_drift
            except (ValueError, FloatingPointError):
                pass  # Ignore errors in curl calculations
        
        # Generate spin-induced force (vortex effect)
        # This is the key DWARF physics that creates orbital shells naturally
        for other in particles:
            if p is other:
                continue
                
            other_pos = other.get('pos', None)
            if other_pos is None:
                continue
                
            # Vector from p to other, considering boundary conditions
            r_vec = calculate_displacement(pos, other_pos)
            r_mag = np.linalg.norm(r_vec) + 1e-9
            
            # DWARF vortex field has enhanced range when enabled
            max_range = 500 if VORTEX_FIELD_ENABLED else 100
            if r_mag > max_range:
                continue
                
            # Skip low-spin interactions for performance unless they're close
            if abs(p['curl']) < 0.01 and r_mag > 50:
                continue
                
            # Create perpendicular vector (for rotational effect)
            try:
                perp_vec = np.array([-r_vec[1], r_vec[0]])
                perp_vec = perp_vec / max(np.linalg.norm(perp_vec), 1e-10)
                
                # Physical spin force law uses DWARF_FORCE_EXPONENT
                # This is where shell formation naturally occurs in the DWARF model
                spin_force_magnitude = p['curl'] / (r_mag**DWARF_FORCE_EXPONENT)
                
                # Enhanced vortex physics when enabled
                if VORTEX_FIELD_ENABLED and p['type'] == 'proton' and other['type'] == 'electron':
                    # This is where mass-resonance creates natural shells!
                    
                    # Mass ratio - key to quantum orbital shell formation
                    # Standard proton:electron mass ratio is ~1836:1
                    mass_ratio = p['mass'] / other['mass']
                    
                    # Calculate resonant wave coupling in vortex field
                    # The 0.5 exponent reflects quantum wave mechanics in DWARF model
                    resonance_factor = (mass_ratio / 1836.0) ** 0.5
                    
                    # Combine with normalized spin value
                    spin_normalized = abs(p['spin'] / 133000)  # Normalize to hydrogen reference
                    
                    # Fine structure factor - reflects how spin and mass create quantized shells
                    fine_structure = 1.0 + 0.007 * np.sin(r_mag / (10 * np.pi))
                    
                    # Combined resonance effect - creates natural orbital shells at specific radii
                    # This is pure physics - no artificial well forcing!
                    combined_factor = spin_normalized ** 0.25 * resonance_factor * fine_structure
                    spin_force_magnitude *= combined_factor
                    
                    # The true DWARF vortex physics - electrons naturally find stable orbits
                    # based on their wave-mechanical resonance with proton's spin field
                    
                    # Record shell formation data for visualization
                    if 'shell_data' not in p:
                        p['shell_data'] = {'expected_radius': 0, 'resonant_factor': 0}
                    
                    # Calculate expected shell radius based on this physics
                    # For visualization only - doesn't affect actual forces
                    base_radius = 216.86  # Hydrogen reference
                    p['shell_data']['expected_radius'] = base_radius * (spin_normalized ** 0.25)
                    p['shell_data']['resonant_factor'] = resonance_factor * fine_structure
                
                spin_force = perp_vec * spin_force_magnitude
                
                # Add memory coupling with physical rate
                if 'field_memory' in other:
                    memory_coupling = 0.1 * np.linalg.norm(other['field_memory'])
                    spin_force = spin_force * (1.0 + memory_coupling)
                
                # Apply the spin-induced force - dv = F * dt / m
                if 'vel' in other:
                    other_mass = max(other.get('mass', 1.0), 0.1)
                    other['vel'] += spin_force * DT / other_mass  # Proper mass scaling
                    
                    # Update net force for logging
                    if 'net_force' in other:
                        other['net_force'] += spin_force * other_mass  # F = ma
                    else:
                        other['net_force'] = spin_force * other_mass
            except (OverflowError, FloatingPointError, ValueError):
                print(f"Warning: Numerical issue in spin force calculation at distance {r_mag}")
                
        # Calculate angular momentum (L = r × p) - physical quantity
        try:
            center = np.array([GRID_SIZE/2, GRID_SIZE/2])
            r_vec = calculate_displacement(center, pos)
            p_vec = p.get('mass', 1.0) * vel
            
            # Cross product in 2D: r_x*p_y - r_y*p_x
            angular_momentum = r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0]
            p['angular_momentum'] = angular_momentum
        except (ValueError, FloatingPointError):
            p['angular_momentum'] = 0.0

def update_positions(particles):
    """Update particle positions based on velocities"""
    for p in particles:
        if isinstance(p, dict) and 'pos' in p and 'vel' in p:
            # This is already timestep-scaled correctly: dx = v * dt
            velocity_impact = p['vel'] * DT
            p['pos'] += velocity_impact
            
            # Apply boundary conditions
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
        # Damping reflection with proper physical damping
        damping_factor = 0.9  # Fixed reflection energy loss
        for i in range(2):
            if p['pos'][i] < 0:
                p['pos'][i] = -p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i] * damping_factor  # Energy loss on reflection
            elif p['pos'][i] >= GRID_SIZE:
                p['pos'][i] = 2*GRID_SIZE - p['pos'][i]  # Reflect position
                p['vel'][i] = -p['vel'][i] * damping_factor  # Energy loss on reflection