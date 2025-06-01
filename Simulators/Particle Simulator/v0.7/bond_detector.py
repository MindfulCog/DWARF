import numpy as np

class BondDetector:
    """Detects and manages atomic bonds between particles"""
    
    def __init__(self, particle_system):
        self.particle_system = particle_system
        
        # Bond detection parameters
        self.bond_parameters = {
            'hydrogen': {
                'radius_target': 1.0,       # Target orbital radius for electron in hydrogen
                'radius_tolerance': 0.1,     # Allowed deviation (10%)
                'velocity_tolerance': 0.2,   # Relative velocity tolerance
                'stability_period': 50,      # Frames required to confirm bond
                'angular_momentum_min': 0.5  # Minimum angular momentum for stable orbit
            },
            'helium': {
                'radius_target': 1.0,
                'radius_tolerance': 0.15,
                'velocity_tolerance': 0.25,
                'stability_period': 70,
                'angular_momentum_min': 0.5,
                'electrons_required': 2
            }
            # More complex atoms can be added here
        }
        
        # Tracking potential bonds
        self.potential_bonds = {}  # {(particle1_id, particle2_id): stability_counter}
        self.potential_atoms = {}  # {(nucleus_id, [electron_ids]): stability_counter}
        
    def check_hydrogen_bond(self, proton, electron):
        """Check if a proton and electron form a hydrogen atom"""
        # Calculate relative position and distance
        r_vec = electron.position - proton.position
        distance = np.linalg.norm(r_vec)
        
        # Check if distance is within tolerance of target radius
        hydrogen_params = self.bond_parameters['hydrogen']
        target_radius = hydrogen_params['radius_target']
        radius_tolerance = hydrogen_params['radius_tolerance']
        
        if abs(distance - target_radius) > radius_tolerance * target_radius:
            return False
            
        # Check relative velocity
        rel_vel = electron.velocity - proton.velocity
        rel_speed = np.linalg.norm(rel_vel)
        
        # For stable orbit, velocity should be mostly perpendicular to position vector
        r_hat = r_vec / distance
        radial_vel = np.dot(rel_vel, r_hat)  # Velocity component along radius
        
        # Check if radial velocity is small compared to total relative velocity
        if rel_speed > 0 and abs(radial_vel) > hydrogen_params['velocity_tolerance'] * rel_speed:
            return False
            
        # Calculate angular momentum
        angular_momentum = np.cross(r_vec, electron.mass * rel_vel)
        angular_momentum_mag = np.linalg.norm(angular_momentum)
        
        # Check if angular momentum is sufficient
        if angular_momentum_mag < hydrogen_params['angular_momentum_min']:
            return False
            
        # All criteria met for potential hydrogen bond
        return True
        
    def check_helium_bond(self, proton, electrons):
        """Check if a proton and two electrons form a helium atom"""
        if len(electrons) != 2:
            return False
            
        helium_params = self.bond_parameters['helium']
        
        # Check each electron's orbit
        for electron in electrons:
            r_vec = electron.position - proton.position
            distance = np.linalg.norm(r_vec)
            
            # Check distance
            if abs(distance - helium_params['radius_target']) > helium_params['radius_tolerance'] * helium_params['radius_target']:
                return False
                
            # Check velocity
            rel_vel = electron.velocity - proton.velocity
            rel_speed = np.linalg.norm(rel_vel)
            r_hat = r_vec / distance if distance > 0 else np.array([1, 0, 0])
            radial_vel = np.dot(rel_vel, r_hat)
            
            if rel_speed > 0 and abs(radial_vel) > helium_params['velocity_tolerance'] * rel_speed:
                return False
                
            # Check angular momentum
            angular_momentum = np.cross(r_vec, electron.mass * rel_vel)
            if np.linalg.norm(angular_momentum) < helium_params['angular_momentum_min']:
                return False
        
        # Check if electrons have opposite spins (simplified Pauli exclusion)
        spin_alignment = np.dot(electrons[0].spin, electrons[1].spin)
        if spin_alignment > -0.5:  # Should be close to -1 for antiparallel spins
            return False
            
        # All criteria met for potential helium bond
        return True
        
    def update(self):
        """Main update function to detect and manage bonds"""
        # Get particles by type
        protons = self.particle_system.get_particles_by_type("proton")
        electrons = self.particle_system.get_particles_by_type("electron")
        
        # Check for potential new hydrogen bonds
        for proton in protons:
            # Skip protons that are already fully bonded
            if len(proton.bonded_with) >= 2:  # Max 2 electrons per proton (He)
                continue
                
            # Check unbonded electrons
            for electron in electrons:
                # Skip electrons that are already bonded
                if electron in proton.bonded_with or electron.bonded_with:
                    continue
                    
                # Check if they form a hydrogen bond
                if self.check_hydrogen_bond(proton, electron):
                    bond_key = (proton.id, electron.id)
                    
                    # Increment stability counter
                    if bond_key in self.potential_bonds:
                        self.potential_bonds[bond_key] += 1
                    else:
                        self.potential_bonds[bond_key] = 1
                        
                    # Check if bond is stable enough to confirm
                    if self.potential_bonds[bond_key] >= self.bond_parameters['hydrogen']['stability_period']:
                        # Create bond
                        self.particle_system.create_bond(proton, electron, "hydrogen")
                        
                        # Remove from potential bonds
                        del self.potential_bonds[bond_key]
                        
                        print(f"Hydrogen atom formed! Proton {proton.id} + Electron {electron.id}")
                else:
                    # Reset stability counter if conditions not met
                    bond_key = (proton.id, electron.id)
                    if bond_key in self.potential_bonds:
                        self.potential_bonds[bond_key] = max(0, self.potential_bonds[bond_key] - 2)
                        
        # Check for potential helium atoms (1 proton + 2 electrons)
        for proton in protons:
            bonded_electrons = [p for p in proton.bonded_with if p.particle_type == "electron"]
            
            # If already has one electron, look for a second
            if len(bonded_electrons) == 1:
                for electron in electrons:
                    # Skip electrons that are already bonded
                    if electron.bonded_with or electron in bonded_electrons:
                        continue
                        
                    # Check if adding this electron would form helium
                    potential_helium_electrons = bonded_electrons + [electron]
                    
                    if self.check_helium_bond(proton, potential_helium_electrons):
                        atom_key = (proton.id, tuple(e.id for e in potential_helium_electrons))
                        
                        # Increment stability counter
                        if atom_key in self.potential_atoms:
                            self.potential_atoms[atom_key] += 1
                        else:
                            self.potential_atoms[atom_key] = 1
                            
                        # Check if atom is stable enough to confirm
                        if self.potential_atoms[atom_key] >= self.bond_parameters['helium']['stability_period']:
                            # Add bond between proton and second electron
                            self.particle_system.create_bond(proton, electron, "helium")
                            
                            # Update atom type from hydrogen to helium
                            self.particle_system.update_atom_type(proton, "hydrogen", "helium")
                            
                            # Remove from potential atoms
                            del self.potential_atoms[atom_key]
                            
                            print(f"Helium atom formed! Proton {proton.id} + 2 Electrons")
                    else:
                        # Reset stability counter if conditions not met
                        atom_key = (proton.id, tuple(e.id for e in potential_helium_electrons))
                        if atom_key in self.potential_atoms:
                            self.potential_atoms[atom_key] = max(0, self.potential_atoms[atom_key] - 2)