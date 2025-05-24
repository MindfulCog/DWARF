import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def export_all_3d(sim, output_dir="output/3d", blender_export=True, pyvista_export=True, mpl_export=True):
    os.makedirs(output_dir, exist_ok=True)

    # Export with matplotlib
    if mpl_export:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = sim.positions[0]
        y = sim.positions[1]
        z = np.zeros_like(x)  # Flat ocean layer
        disp = np.linalg.norm(sim.positions - sim.tracers, axis=0)

        p = ax.scatter(x, y, z, c=disp, cmap='plasma', s=10)
        ax.set_title("3D Tidal Bulge (Matplotlib)")
        fig.colorbar(p, ax=ax, label="Displacement")
        ax.set_xlim(-sim.earth_radius * 1.5, sim.earth_radius * 1.5)
        ax.set_ylim(-sim.earth_radius * 1.5, sim.earth_radius * 1.5)
        ax.set_zlim(-sim.earth_radius * 0.1, sim.earth_radius * 0.1)
        plt.tight_layout()
        mpl_path = os.path.join(output_dir, "tidal_3d_mpl.png")
        plt.savefig(mpl_path)
        plt.close()
        print(f"Matplotlib 3D plot saved to: {mpl_path}")

    # Export for Blender as .obj
    if blender_export:
        obj_path = os.path.join(output_dir, "tidal_bulge.obj")
        with open(obj_path, "w") as f:
            for i in range(len(sim.positions[0])):
                x, y = sim.positions[:, i]
                z = 0.0
                f.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
        print(f"OBJ mesh exported for Blender at: {obj_path}")

    # PyVista interactive preview placeholder
    if pyvista_export:
        try:
            import pyvista as pv
            cloud = pv.PolyData(np.c_[sim.positions[0], sim.positions[1], np.zeros_like(sim.positions[0])])
            plotter = pv.Plotter()
            plotter.add_mesh(cloud, render_points_as_spheres=True, scalars=np.linalg.norm(sim.positions - sim.tracers, axis=0), cmap="plasma")
            plotter.show(screenshot=os.path.join(output_dir, "pyvista_preview.png"))
            print("PyVista preview rendered and saved.")
        except ImportError:
            print("PyVista not installed. Skipping interactive export.")
