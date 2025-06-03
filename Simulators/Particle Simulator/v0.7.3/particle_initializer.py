import numpy as np
import random
from typing import List, Tuple
from particle_data import Particle
from vector import Vector3
from constants import *

class ParticleInitializer:
    """Optimized particle initialization with vectorized operations"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def create_particles(self, num_particles: int, grid_size: Tuple[float, float, float], 
                        mode: str = "random") -> List[Particle]:
        """Create particles using vectorized initialization"""
        
        if mode == "random":
            return self._create_random_particles(num_particles, grid_size)
        elif mode == "structured":
            return self._create_structured_particles(num_particles, grid_size)
        elif mode == "cluster":
            return self._create_clustered_particles(num_particles, grid_size)
        else:
            return self._create_random_particles(num_particles, grid_size)
    
    def _create_random_particles(self, num_particles: int, grid_size: Tuple[float, float, float]) -> List[Particle]:
        """Create randomly distributed particles using vectorized operations"""
        
        # Vectorized position generation
        positions = np.random.uniform(-np.array(grid_size)/2, np.array(grid_size)/2, 
                                     (num_particles, 3)).astype(np.float32)
        
        # Vectorized velocity generation (Maxwell-Boltzmann-like distribution)
        velocities = np.random.normal(0, 0.5, (num_particles, 3)).astype(np.float32)
        
        # Vectorized particle type assignment
        particle_types = np.random.choice(["proton", "electron", "neutron"], 
                                         num_particles, 
                                         p=[0.4, 0.4, 0.2])
        
        # Vectorized mass assignment based on type
        masses = np.where(particle_types == "proton", 1.67e-27,
                 np.where(particle_types == "electron", 9.11e-31, 1.675e-27))
        
        # Create particle list
        particles = []
        for i in range(num_particles):
            p = Particle(i)
            p.position = Vector3(positions[i, 0], positions[i, 1], positions[i, 2])
            p.velocity = Vector3(velocities[i, 0], velocities[i, 1], velocities[i, 2])
            p.mass = float(masses[i])
            p.particle_type = particle_types[i]
            
            # Add some spin for interesting dynamics
            spin_magnitude = np.random.uniform(0, 0.1)
            spin_direction = np.random.normal(0, 1, 3)
            spin_direction = spin_direction / np.linalg.norm(spin_direction)
            p.spin = Vector3(spin_direction[0] * spin_magnitude,
                           spin_direction[1] * spin_magnitude,
                           spin_direction[2] * spin_magnitude)
            
            particles.append(p)
        
        return particles
    
    def _create_structured_particles(self, num_particles: int, grid_size: Tuple[float, float, float]) -> List[Particle]:
        """Create particles in a structured lattice"""
        
        # Calculate lattice dimensions
        particles_per_dim = int(np.ceil(num_particles ** (1/3)))
        actual_particles = particles_per_dim ** 3
        
        # Create lattice positions
        x = np.linspace(-grid_size[0]/2, grid_size[0]/2, particles_per_dim)
        y = np.linspace(-grid_size[1]/2, grid_size[1]/2, particles_per_dim)
        z = np.linspace(-grid_size[2]/2, grid_size[2]/2, particles_per_dim)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Take only the requested number of particles
        positions = positions[:num_particles]
        
        # Add small random perturbations
        perturbations = np.random.normal(0, 0.1, positions.shape)
        positions += perturbations
        
        # Create particles
        particles = []
        for i in range(len(positions)):
            p = Particle(i)
            p.position = Vector3(positions[i, 0], positions[i, 1], positions[i, 2])
            
            # Give initial rotation velocity
            center = np.array([0, 0, 0])
            r_vec = positions[i] - center
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag > 0:
                # Circular velocity around center
                angular_velocity = 0.1
                tangent = np.cross([0, 0, 1], r_vec)
                tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0, 0])
                velocity = tangent * angular_velocity
                p.velocity = Vector3(velocity[0], velocity[1], velocity[2])
            
            p.mass = 1.0
            p.particle_type = "proton" if i % 3 == 0 else ("electron" if i % 3 == 1 else "neutron")
            
            particles.append(p)
        
        return particles
    
    def _create_clustered_particles(self, num_particles: int, grid_size: Tuple[float, float, float]) -> List[Particle]:
        """Create particles in multiple clusters"""
        
        num_clusters = max(1, num_particles // 100)  # Roughly 100 particles per cluster
        particles = []
        particles_per_cluster = num_particles // num_clusters
        
        for cluster_id in range(num_clusters):
            # Random cluster center
            cluster_center = np.random.uniform(-np.array(grid_size)/3, np.array(grid_size)/3, 3)
            
            # Particles for this cluster
            cluster_particles = particles_per_cluster
            if cluster_id == num_clusters - 1:  # Last cluster gets remainder
                cluster_particles = num_particles - len(particles)
            
            # Generate cluster positions (Gaussian distribution around center)
            cluster_positions = np.random.normal(cluster_center, 0.5, (cluster_particles, 3))
            
            # Create particles for this cluster
            for i in range(cluster_particles):
                p = Particle(len(particles))
                p.position = Vector3(cluster_positions[i, 0], cluster_positions[i, 1], cluster_positions[i, 2])
                
                # Velocity towards cluster center with some randomness
                to_center = cluster_center - cluster_positions[i]
                center_velocity = to_center * 0.1
                random_velocity = np.random.normal(0, 0.2, 3)
                final_velocity = center_velocity + random_velocity
                
                p.velocity = Vector3(final_velocity[0], final_velocity[1], final_velocity[2])
                p.mass = 1.0
                p.particle_type = ["proton", "electron", "neutron"][i % 3]
                
                particles.append(p)
        
        return particles