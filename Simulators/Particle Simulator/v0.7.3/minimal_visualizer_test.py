import numpy as np
from vispy import app, scene

# Create a window with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Create a simple volume
vol_data = np.random.rand(32, 32, 32)  # Simple 3D scalar data

# Create Volume visual
volume = scene.visuals.Volume(vol_data, cmap='viridis', method='mip', parent=view.scene)

# Run the app
print("Starting app...")
app.run()