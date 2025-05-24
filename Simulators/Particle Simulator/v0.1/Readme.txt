# DWARF Particle Simulation (3D PyTorch Version)

This repository contains a 3D simulation of emergent particle-like structures based on the DWARF theory — Dynamic Wake Accretion in Relativistic Fluids. The simulation evolves a velocity and density field using modified fluid dynamics equations and tracks energy, enstrophy, and structural behavior over time.

## 🔧 Requirements

To run this simulation, you’ll need Python 3.10+ and the following packages:

```bash
pip install torch numpy matplotlib tqdm
```

For optional 3D visualization:
```bash
pip install pyvista
```

If you want to render tracer particles or field slices into a video:
```bash
# Requires ffmpeg installed and added to PATH
# Download: https://ffmpeg.org/download.html
```

## 📂 Files

- `dwarf_particle_simulation.py` — Main simulation script
- `rho_slice_*.png` — Density slice output images
- `tracers_*.png` — Tracer particle 2D visualizations
- `energy_log.npy`, `enstrophy_log.npy`, `radius_log.npy`, `peak_density_log.npy` — Logged diagnostics
- `README.md` — This file

## 🚀 Running the Simulation

```bash
python dwarf_particle_simulation.py
```

## 📊 Outputs

- **Density Slices**: Cross-sections of the 3D field showing density evolution
- **Tracers**: Visual indicator of velocity field flow behavior
- **Diagnostic Logs**: Tracks total energy, enstrophy, RMS radius, and peak density over time

## 🧠 Scientific Purpose
This simulation explores whether DWARF theory can:
- Support emergent, stable, coherent structures (field quanta)
- Exhibit properties analogous to mass, spin, and charge
- Provide a unified framework without invoking spacetime curvature

## 📌 Next Development Goals
- Add multi-vortex interaction tests
- Simulate particle-like collisions
- Visualize full 3D density blobs using `pyvista`
- Export 3D tracer motion
- Add electromagnetic-like coupling terms

---