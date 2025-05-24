import numpy as np
import matplotlib.pyplot as plt

def generate_mock_fundy_mask(grid_shape=(100, 50), water_level=0):
    # Simulated elevation: gradient down toward the right edge (bay inlet)
    x = np.linspace(-1, 1, grid_shape[1])
    y = np.linspace(-1, 1, grid_shape[0])
    xv, yv = np.meshgrid(x, y)
    elevation = -5 + 5 * (xv**2 + yv**2)  # Parabolic bowl, negative in center
    mask = (elevation < water_level).astype(int)
    return mask, elevation

def save_fundy_mask(output_dir="data"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    mask, elevation = generate_mock_fundy_mask()
    np.save(f"{output_dir}/fundy_mask.npy", mask)
    np.save(f"{output_dir}/fundy_elevation.npy", elevation)
    plt.imshow(mask, cmap="Blues")
    plt.title("Simulated Bay of Fundy Ocean Mask")
    plt.savefig(f"{output_dir}/fundy_mask_preview.png")
    plt.close()
    print(f"✅ Mask and elevation saved to {output_dir}")

if __name__ == "__main__":
    save_fundy_mask()
