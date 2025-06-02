def fix_visualizer_rendering():
    """Apply patches to fix rendering issues in the visualizer"""
    import sys
    
    # Import the Visualizer3D class
    from visualizer_3d import Visualizer3D
    
    # Create a copy of the modules dictionary to avoid modification during iteration
    modules_copy = dict(sys.modules)
    
    # Find visualizer instances
    for module_name, module in modules_copy.items():
        # Find visualizer instances
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, Visualizer3D):
                    # Fix transparency issues
                    if hasattr(attr, 'view') and hasattr(attr.view, 'canvas'):
                        # Set background alpha to 1.0 (fully opaque)
                        attr.view.canvas.bgcolor = (0, 0, 0, 1.0)
                    
                    # Fix particle visibility
                    if hasattr(attr, 'markers'):
                        attr.markers.set_gl_state('translucent', depth_test=True, cull_face=False)
                        
                    print("Applied visualization fixes to", attr_name)
            except:
                pass  # Skip any attributes that can't be accessed