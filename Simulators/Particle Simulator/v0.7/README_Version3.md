# DWARF Vortex Simulator

## Overview

The DWARF (Dynamic Wake Accretion in Relativistic Fluids) Vortex Simulator is a specialized physics simulation tool that explores a novel approach to particle interactions. Unlike traditional N-body simulators that use standard Newtonian gravity, this simulator implements the DWARF theory which models gravitational-like phenomena as emergent properties of vortex dynamics in a continuous field with memory.

## Key Concepts in DWARF Theory

- **Memory Field**: A field that "remembers" the passage of particles, creating persistent patterns
- **Force Exponent**: DWARF uses a characteristic 2.22 power law exponent (vs. 2.0 in Newton's gravity)
- **Spin Coupling**: Particle spin creates wake patterns that influence other particles
- **Vortex Formation**: Interacting particles create vortex structures in the memory field

## Features

- **Particle Types**:
  - Protons: Positive spin and charge
  - Electrons: Negative spin and charge
  - Neutrons: Neutral charge with small spin

- **Interactive Controls**:
  - Adjust particle spin values
  - Control memory field decay rate
  - Set global drag and saturation parameters
  - Toggle field visualization
  - Reset simulation

- **Visualization**:
  - Real-time memory field display with streamlines
  - Particle trails showing movement history
  - Field density and curl visualization

- **Performance Features**:
  - Multiprocessing support for faster calculations
  - Optimized physics calculations
  - Velocity Verlet integration for numerical stability

- **Analysis Tools**:
  - Energy conservation tracking
  - Frame recording for creating animations
  - Screenshot capability

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- (Optional) Multiprocessing library for performance

## Installation

1. Ensure you have Python 3.7 or newer installed
2. Install required packages:
   ```bash
   pip install numpy matplotlib