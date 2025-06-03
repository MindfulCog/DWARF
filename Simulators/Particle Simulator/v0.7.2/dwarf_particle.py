import numpy as np

class Particle:
    """Base class for all particles with spin dynamics"""
    
    def __init__(self, position, velocity=None, spin=None, mass=1.0, charge=0.0):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity, dtype=np.float64)
        self.spin = np.array([0, 0, 1]) if spin is None else np.array(spin, dtype=np.float64)
        self.normalize_spin()
        self.mass = mass
        self.charge = charge
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        # Rotational inertia (moment of inertia) for spin dynamics
        self.rotational_inertia = mass * 0.2  # Scales with mass
        self.angular_velocity = np.zeros(3)    # Angular velocity vector
        self.id = id(self)
        self.bonded_with = []
        self.trajectory = []  # Store recent positions for visualization
        self.max_trajectory_length = 100  # Maximum trajectory points
        self.bond_type = None  # For tracking atom type
        self.stable_orbit_counter = 0  # For bond detection
        
    def normalize_spin(self):
        """Ensure spin is a unit vector"""
        magnitude = np.linalg.norm(self.spin)
        if magnitude > 0:
            self.spin = self.spin / magnitude
            
    def update_position(self, dt):
        """Update position based on velocity"""
        self.position += self.velocity * dt
        
        # Store position in trajectory
        self.trajectory.append(np.copy(self.position))
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
            
    def update_velocity(self, dt):
        """Update velocity based on force"""
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        
    def update_spin(self, dt):
        """Update spin direction based on torque"""
        # Update angular velocity based on torque and rotational inertia
        angular_acceleration = self.torque / self.rotational_inertia
        self.angular_velocity += angular_acceleration * dt
        
        # Apply damping to angular velocity (prevents excessive spin)
        damping = 0.98
        self.angular_velocity *= damping
        
        # Calculate change in spin direction
        delta_angle = np.linalg.norm(self.angular_velocity) * dt
        
        if delta_angle > 0:
            # Axis of rotation is the normalized angular velocity
            axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
            
            # Rotate the spin vector around this axis using Rodrigues' rotation formula
            cos_theta = np.cos(delta_angle)
            sin_theta = np.sin(delta_angle)
            cross_product = np.cross(axis, self.spin)
            dot_product = np.dot(axis, self.spin)
            
            self.spin = (self.spin * cos_theta + 
                        cross_product * sin_theta + 
                        axis * dot_product * (1 - cos_theta))
            
            self.normalize_spin()
        
    def calculate_angular_momentum(self):
        """Calculate angular momentum of the particle"""
        return self.rotational_inertia * self.angular_velocity
        
    def reset_forces(self):
        """Reset forces and torques to zero"""
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        
    def apply_force(self, force, point=None):
        """Apply force to particle"""
        self.force += force
        
        # Calculate torque if point of application is provided
        if point is not None:
            r = point - self.position
            torque = np.cross(r, force)
            self.torque += torque
            
    def apply_torque(self, torque):
        """Apply torque directly to particle"""
        self.torque += torque
        
    def get_kinetic_energy(self):
        """Calculate kinetic energy of the particle"""
        # Translational kinetic energy
        translational = 0.5 * self.mass * np.sum(self.velocity**2)
        
        # Rotational kinetic energy
        rotational = 0.5 * self.rotational_inertia * np.sum(self.angular_velocity**2)
        
        return translational + rotational
        
    def set_spin(self, new_spin):
        """Set spin to a new direction (used for manual control)"""
        self.spin = np.array(new_spin, dtype=np.float64)
        self.normalize_spin()

class Proton(Particle):
    """Proton implementation"""
    
    def __init__(self, position, velocity=None, spin=None):
        super().__init__(
            position=position, 
            velocity=velocity, 
            spin=spin, 
            mass=1836.0,  # Mass in units of electron mass
            charge=1.0     # Charge in units of elementary charge
        )
        self.particle_type = "proton"
        # Protons have higher rotational inertia
        self.rotational_inertia = self.mass * 0.2

class Electron(Particle):
    """Electron implementation"""
    
    def __init__(self, position, velocity=None, spin=None):
        super().__init__(
            position=position, 
            velocity=velocity, 
            spin=spin, 
            mass=1.0,      # Mass in units of electron mass
            charge=-1.0    # Charge in units of elementary charge
        )
        self.particle_type = "electron"
        # Electrons have lower rotational inertia (spin more easily)
        self.rotational_inertia = self.mass * 0.05
        
class Neutron(Particle):
    """Neutron implementation"""
    
    def __init__(self, position, velocity=None, spin=None):
        super().__init__(
            position=position, 
            velocity=velocity, 
            spin=spin, 
            mass=1839.0,   # Mass in units of electron mass (slightly more than proton)
            charge=0.0     # Neutral charge
        )
        self.particle_type = "neutron"
        # Neutrons have similar rotational inertia to protons
        self.rotational_inertia = self.mass * 0.2

class ParticleSystem:
    """Manages a collection of particles"""
    
    def __init__(self):
        self.particles = []
        self.bonds = []
        self.atom_groups = {}  # {atom_type: [(nucleus, [electrons]), ...]}
        
    def add(self, particle):
        """Add a particle to the system"""
        self.particles.append(particle)
        return particle
        
    def remove(self, particle):
        """Remove a particle from the system"""
        if particle in self.particles:
            self.particles.remove(particle)
            
            # Remove any bonds involving this particle
            self.bonds = [b for b in self.bonds if particle not in b]
            
            # Remove from bonded_with lists of other particles
            for p in self.particles:
                if particle in p.bonded_with:
                    p.bonded_with.remove(particle)
            
            # Update atom groups
            for atom_type in list(self.atom_groups.keys()):
                updated_atoms = []
                for nucleus, electrons in self.atom_groups[atom_type]:
                    if particle == nucleus:
                        continue  # Remove this atom
                    elif particle in electrons:
                        electrons.remove(particle)
                        if electrons:  # Keep if still has electrons
                            updated_atoms.append((nucleus, electrons))
                    else:
                        updated_atoms.append((nucleus, electrons))
                
                if updated_atoms:
                    self.atom_groups[atom_type] = updated_atoms
                else:
                    del self.atom_groups[atom_type]
            
    def get_particles_by_type(self, particle_type):
        """Get all particles of a specific type"""
        return [p for p in self.particles if p.particle_type == particle_type]
    
    def get_unbonded_particles(self, particle_type=None):
        """Get particles that are not part of any bond"""
        if particle_type:
            return [p for p in self.particles if p.particle_type == particle_type and not p.bonded_with]
        else:
            return [p for p in self.particles if not p.bonded_with]
            
    def create_bond(self, particle1, particle2, bond_type):
        """Create a bond between particles"""
        if (particle1, particle2) not in self.bonds and (particle2, particle1) not in self.bonds:
            self.bonds.append((particle1, particle2))
            
            # Add reference to each other
            particle1.bonded_with.append(particle2)
            particle2.bonded_with.append(particle1)
            particle1.bond_type = bond_type
            particle2.bond_type = bond_type
            
            # Track as atom
            # For hydrogen: one proton + one electron
            if bond_type == "hydrogen":
                nucleus = particle1 if particle1.particle_type == "proton" else particle2
                electron = particle2 if particle2.particle_type == "electron" else particle1
                
                if "hydrogen" not in self.atom_groups:
                    self.atom_groups["hydrogen"] = []
                    
                self.atom_groups["hydrogen"].append((nucleus, [electron]))
                
            # For helium: was already hydrogen, now adding second electron
            elif bond_type == "helium":
                nucleus = particle1 if particle1.particle_type == "proton" else particle2
                electron = particle2 if particle2.particle_type == "electron" else particle1
                
                # Find existing hydrogen atom with this nucleus
                if "hydrogen" in self.atom_groups:
                    for i, (h_nucleus, h_electrons) in enumerate(self.atom_groups["hydrogen"]):
                        if h_nucleus == nucleus:
                            # Remove from hydrogen
                            del self.atom_groups["hydrogen"][i]
                            
                            # Add to helium
                            if "helium" not in self.atom_groups:
                                self.atom_groups["helium"] = []
                                
                            h_electrons.append(electron)
                            self.atom_groups["helium"].append((nucleus, h_electrons))
                            break
            
    def update_atom_type(self, nucleus, old_type, new_type):
        """Update atom from one type to another"""
        if old_type in self.atom_groups and new_type not in self.atom_groups:
            self.atom_groups[new_type] = []
            
        # Find the atom with this nucleus
        for i, (atom_nucleus, electrons) in enumerate(self.atom_groups[old_type]):
            if atom_nucleus == nucleus:
                # Move to new type
                self.atom_groups[new_type].append((atom_nucleus, electrons))
                del self.atom_groups[old_type][i]
                
                # Update bond types
                atom_nucleus.bond_type = new_type
                for electron in electrons:
                    electron.bond_type = new_type
                    
                break
                
        # Clean up empty lists
        if old_type in self.atom_groups and not self.atom_groups[old_type]:
            del self.atom_groups[old_type]