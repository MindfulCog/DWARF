import numpy as np
from vispy import app, scene

# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', show=True, size=(800, 600))
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Create some scalar test data (16x16x16 volume)
vol_data = np.zeros((16, 16, 16), dtype=np.float32)

# Add a simple central sphere pattern
for i in range(16):
    for j in range(16):
        for k in range(16):
            # Distance from center
            x, y, z = i-7.5, j-7.5, k-7.5
            r = np.sqrt(x**2 + y**2 + z**2)
            # Simple sphere pattern with smooth falloff
            vol_data[i, j, k] = np.exp(-0.2 * r)

print("Creating volume visual...")
# Create a Volume visual with a colormap
volume = scene.visuals.Volume(vol_data, cmap='viridis', method='mip', parent=view.scene)
print("Volume created successfully!")

# Add an XYZ axis for orientation
scene.visuals.XYZAxis(parent=view.scene)

@canvas.events.key_press.connect
def on_key_press(event):
    if event.key == 'escape':
        app.quit()
    print(f"Key pressed: {event.key}")

print("Starting app... Press ESC to quit")
app.run()