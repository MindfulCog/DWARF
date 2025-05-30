"""
Physics core for the DWARF Simulator.
Implements the core physics engine for particle interactions and field dynamics.
"""
import numpy as np
import time
import sys

# Try to import constants from config, with fallbacks
try:
    from config import GRID_SIZE, DT, DWARF_FORCE_EXPONENT, GLOBAL_DRAG, SATURATION_LIMIT, MEMORY_DECAY
    from config import BOUNDARY_MODE, BOUNDARY_WRAP, BOUNDARY_DAMP, VORTEX_FIELD_ENABLED
except ImportError as e:
    print(f"Warning: Could not import all constants from config.py: {e}")
    try:
        from config import SimulationConfig
        config = SimulationConfig()
        
        # Create globals for compatibility
        GRID_SIZE = getattr(config, 'GRID_SIZE', 2048)
        DT = getattr(config, 'time_step', 0.001)
        DWARF_FORCE_EXPONENT = getattr(config, 'force_exponent', 2.0)
        GLOBAL_DRAG = getattr(config, 'global_drag', 0.985)
        SATURATION_LIMIT = getattr(config, 'saturation_limit', 15.0)
        MEMORY_DECAY = getattr(config, 'memory_decay', 0.9998)
        BOUNDARY_MODE = getattr(config, 'boundary_mode', 'wrap')
        BOUNDARY_WRAP = 'wrap'
        BOUNDARY_DAMP = 'damp'
        VORTEX_FIELD_ENABLED = getattr(config, 'vortex_field_enabled', True)
        
        print("Successfully loaded constants from SimulationConfig")
    except Exception as config_err:
        print(f"Error: Could not load simulation constants! Using defaults: {config_err}")
        # Default fallback values
        GRID_SIZE = 2048
        DT = 0.001
        DWARF_FORCE_EXPONENT = 2.0
        GLOBAL_DRAG = 0.985
        SATURATION_LIMIT = 15.0
        MEMORY_DECAY = 0.9998
        BOUNDARY_MODE = 'wrap'
        BOUNDARY_WRAP = 'wrap'
        BOUNDARY_DAMP = 'damp'
        VORTEX_FIELD_ENABLED = True

# Ensure all physics constants are the right type
GRID_SIZE = float(GRID_SIZE)
DT = float(DT)
DWARF_FORCE_EXPONENT = float(DWARF_FORCE_EXPONENT)
GLOBAL_DRAG = float(GLOBAL_DRAG)
SATURATION_LIMIT = float(SATURATION_LIMIT)
MEMORY_DECAY = float(MEMORY_DECAY)

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
        print(f"Beginning simulation step {step}")
        current_time = float(step) * float(DT)  # Calculate time for logging
        
        # Add initial velocity if particles aren't moving
        if step == 0:
            initialize_velocities(particles)
            
        update_memory(particles)
        apply_forces(particles)
        apply_spin_effects(particles)
        update_positions(particles)
        
        # Proper logging with better error handling
        if logs is not None:
            try:
                # Use the logger object directly instead of importing a function
                physics_data = {
                    'step': step,
                    'time': current_time,
                    'particles': []
                }
                
                # Copy particle data
                for p in particles:
                    particle_data = {
                        'type': p.get('type', 'unknown'),
                        'id': p.get('id', 0)
                    }
                    
                    # Include all numerical attributes
                    for key, value in p.items():
                        if key not in ['type', 'id']:
                            if isinstance(value, np.ndarray):
                                # Handle 2D vectors
                                if len(value) == 2:
                                    particle_data[f"{key}_x"] = float(value[0])
                                    particle_data[f"{key}_y"] = float(value[1])
                                else:
                                    # Log magnitude for other vectors
                                    particle_data[key] = float(np.linalg.norm(value))
                            else:
                                # Copy scalar values directly
                                try:
                                    particle_data[key] = float(value)
                                except (TypeError, ValueError):
                                    pass  # Skip non-numeric values
                    
                    physics_data['particles'].append(particle_data)
                
                # Calculate distances for physics data
                physics_data['distances'] = {}
                for i, p1 in enumerate(particles):
                    for j, p2 in enumerate(particles):
                        if i >= j:  # Skip self and avoid duplicates
                            continue
                            
                        key = f"{p1['type']}_{p1.get('id', i)}_to_{p2['type']}_{p2.get('id', j)}"
                        r_vec = calculate_displacement(p1['pos'], p2['pos'])
                        physics_data['distances'][key] = float(np.linalg.norm(r_vec))
                
                # Log the physics data directly using the logger instance
                logs.log_data(physics_data)
                
                # Also log in v0.5 style
                from logger import log_particle_state
                log_particle_state(step, particles)
            except Exception as e:
                print(f"Error in simulation logging: {e}")
                import traceback
                traceback.print_exc()
            
        print(f"Completed simulation step {step}")
        return True
    except Exception as e:
        print(f"Error in simulation step {step}: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_velocities(particles):
    """Make sure particles have some initial velocity if they're all static"""
    print("Initializing velocities...")
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
                    p['vel'] = perpendicular * 30.0
                    print(f"Set velocity for electron: {p['vel']}")

def update_memory(particles):
    """Update particle memory fields"""
    print("Updating memory fields...")
    
    # Ensure DWARF_FORCE_EXPONENT is a float
    force_exponent = float(DWARF_FORCE_EXPONENT)
    memory_decay = float(MEMORY_DECAY)
    saturation_limit = float(SATURATION_LIMIT)
    
    for p in particles:
        if isinstance(p, dict):
            if 'field_memory' not in p:
                p['field_memory'] = np.array([0.0, 0.0])
            
            # Record particle's movement in memory using global MEMORY_DECAY
            if 'vel' in p:
                memory_update_rate = 1.0 - memory_decay
                p['field_memory'] = p['field_memory'] * memory_decay + p['vel'] * memory_update_rate
                
                # Create memory gradient (rate of change)
                if 'prev_field_memory' not in p:
                    p['prev_field_memory'] = p['field_memory'].copy()
                    p['memory_gradient'] = np.array([0.0, 0.0])
                else:
                    p['memory_gradient'] = (p['field_memory'] - p['prev_field_memory']) / max(float(DT), 1e-10)
                    p['prev_field_memory'] = p['field_memory'].copy()
                
                # Create memory vector with correct DWARF_FORCE_EXPONENT
                memory_mag = np.linalg.norm(p['field_memory'])
                if memory_mag > 1e-10:
                    # Use configured exponent but with safety check
                    try:
                        safe_exponent = min(force_exponent, 3.0)  # Cap exponent for stability
                        p['memory_vector'] = p['field_memory'] / memory_mag * (memory_mag**safe_exponent)
                    except (OverflowError, FloatingPointError, TypeError) as e:
                        print(f"Warning: Error in memory vector calculation: {e}")
                        print(f"Memory mag: {memory_mag}, Exponent: {force_exponent}, Type: {type(force_exponent)}")
                        p['memory_vector'] = p['field_memory'] * 10.0  # Fallback calculation
                else:
                    p['memory_vector'] = np.array([0.0, 0.0])
                    
            # Apply saturation limit
            memory_mag = np.linalg.norm(p['field_memory'])
            if memory_mag > saturation_limit:
                p['field_memory'] = p['field_memory'] / memory_mag * saturation_limit

def apply_forces(particles):
    """Apply forces between particles"""
    print("Applying forces...")
    # Ensure constants are floats
    dt = float(DT)
    force_exponent = float(DWARF_FORCE_EXPONENT)
    global_drag = float(GLOBAL_DRAG)
    
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
            
            # Base Coulomb force
            coulomb_factor = charge_p * charge_o / r_squared
            force_direction = rel_pos / r
            
            # Apply force using DWARF_FORCE_EXPONENT
            try:
                # Scale force using proper exponent
                if force_exponent != 2.0:
                    # Adjust for DWARF exponent difference from standard inverse square law
                    dwarf_adjustment = r**(2.0 - force_exponent)
                    # Cap the adjustment factor for numerical stability
                    dwarf_adjustment = np.clip(dwarf_adjustment, 0.1, 10.0)
                    force = force_direction * coulomb_factor * dwarf_adjustment * 100000
                else:
                    force = force_direction * coulomb_factor * 100000
            except (OverflowError, FloatingPointError, TypeError) as e:
                print(f"Warning: Numerical issue in force calculation at distance {r}: {e}")
                force = force_direction * coulomb_factor * 100000  # Fallback to normal inverse square
            
            total_force += force

            # Memory coupling - simplify for stability
            if 'memory_vector' in p and 'memory_vector' in other:
                try:
                    # Simple scaled memory interaction
                    memory_coupling = min(0.01, 0.001 * np.linalg.norm(other['memory_vector']))
                    memory_force = force_direction * memory_coupling
                    total_force += memory_force
                except (OverflowError, FloatingPointError, ValueError):
                    print("Warning: Numerical issue in memory coupling")

        if isinstance(p, dict) and 'vel' in p:
            mass = max(p.get('mass', 1.0), 0.1)  # Ensure mass is never too small
            
            # Store net force for logging
            p['net_force'] = total_force
            
            # Update velocity
            p['vel'] += total_force * dt / mass
            
            # Apply global drag using GLOBAL_DRAG
            p['vel'] *= global_drag

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
    print("Applying spin effects...")
    # Ensure constants are floats
    dt = float(DT)
    force_exponent = float(DWARF_FORCE_EXPONENT)
    
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
        
        # Apply curl-induced drift (DWARF turbulence effect) - new feature
        if 'vel' in p and np.linalg.norm(p['vel']) > 1e-10:
            try:
                tangent_vector = np.array([-p['vel'][1], p['vel'][0]])
                tangent_norm = np.linalg.norm(tangent_vector)
                if tangent_norm > 1e-10:
                    # Small-scale curl effect
                    tangent_drift = tangent_vector / tangent_norm * p['curl'] * 0.001
                    p['vel'] += tangent_drift
            except (ValueError, FloatingPointError):
                pass  # Ignore errors in curl calculations
        
        # Generate spin-induced force (vortex effect)
        if VORTEX_FIELD_ENABLED:  # Only if vortex field is enabled
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
                try:
                    perp_vec = np.array([-r_vec[1], r_vec[0]])
                    perp_vec = perp_vec / max(np.linalg.norm(perp_vec), 1e-10)
                    
                    # Basic spin force calculation - using correct exponent
                    spin_force_mag = abs(p['curl']) / (r_mag**force_exponent + 1)
                    spin_force = perp_vec * spin_force_mag
                    
                    # Add memory coupling if available - simplified
                    if 'field_memory' in other:
                        memory_coupling = min(1.0, 0.1 * np.linalg.norm(other['field_memory']))
                        spin_force = spin_force * (1.0 + memory_coupling)
                    
                    # Apply the spin-induced force to the other particle
                    if 'vel' in other:
                        other['vel'] += spin_force * dt
                        # Update net force
                        if 'net_force' in other:
                            other['net_force'] += spin_force
                        else:
                            other['net_force'] = spin_force
                except (OverflowError, FloatingPointError, ValueError, TypeError) as e:
                    print(f"Warning: Numerical issue in spin force calculation at distance {r_mag}: {e}")
                    
        # Calculate angular momentum (L = r × p)
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
    print("Updating positions...")
    dt = float(DT)  # Ensure DT is a float
    
    for p in particles:
        if isinstance(p, dict) and 'pos' in p and 'vel' in p:
            velocity_impact = p['vel'] * dt
            p['pos'] += velocity_impact
            
            # Apply boundary conditions based on selected mode
            apply_boundary_conditions(p)
            
        elif hasattr(p, 'pos') and hasattr(p, 'vel'):
            p.pos += p.vel * dt
            
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