***Currently in development not yet tested***


# 🌀 DWARF Binary Wake Collision Simulator

Welcome to the **DWARF Binary Wake Collision Simulator**, a visualization and physics demo designed to showcase the emergent behavior of wake interference between two rotating mass analogs — inspired by merging black holes.

This simulation is part of the **DWARF** project (Dynamic Wake Accretion in Relativistic Fluids), a new approach to gravitational dynamics using fluid-like field interactions instead of spacetime curvature.

---

## 🌌 Overview

This tool simulates:
- Two orbiting mass sources (black hole analogs)
- Rotating wakes that dynamically interfere
- Tracer particles that reveal fluidic field behavior and emergent structure during merger
- Export of particle trajectories to `.csv` files for Blender visualization

---

## 🔧 How to Use

### 1. Run the Simulation

Execute the simulation script:

```bash
python dwarf_binary_collision_simulation.py
```

This generates a series of files:
```
tracer_0.csv
tracer_1.csv
...
tracer_499.csv
```

Each CSV file contains the 3D position of a tracer particle at every timestep.

---

### 2. Visualize in Blender

1. Open Blender and go to the **Scripting** tab.
2. Place the CSV files and the script `import_tracer_curves_to_blender.py` in the same directory.
3. Load the script in Blender and click **Run Script**.
4. Each tracer path will appear as a 3D curve in the scene.
5. Add glow effects, geometry modifiers, or volumetrics to create stunning visuals.

---

## 📦 Files in This Repo

| File | Description |
|------|-------------|
| `dwarf_binary_collision_simulation.py` | Python script to generate tracer data |
| `import_tracer_curves_to_blender.py` | Blender import script for .csv curves |
| `README.md` | You're reading it! |

---

## 🧙‍♂️ Tips

- Use **curve bevels** and **emission shaders** in Blender for glowing tracer paths.
- Animate a camera around the scene for dynamic fly-throughs.
- Use volumetric fog to simulate gravitational energy release.
- Add sound effects or background music for cinematic exports.

---

## 🤯 Why This Matters

This simulation visualizes how a field-based theory like DWARF might explain gravitational wave dynamics, information retention, and energy distribution **without invoking spacetime curvature** or **dark matter**.

It’s open-source, modifiable, and educational. Perfect for:
- Researchers
- Sci-fi fans
- Blender artists
- Chaos enthusiasts

---

## 🪐 License

MIT License — go forth and simulate.
