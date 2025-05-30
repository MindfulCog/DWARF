"""
Small diagnostic script to identify the simulator class name in physics_core.py
"""

def check_physics_core():
    # Print all classes defined in physics_core
    import inspect
    import physics_core
    
    classes = []
    for name, obj in inspect.getmembers(physics_core):
        if inspect.isclass(obj) and obj.__module__ == physics_core.__name__:
            classes.append(name)
    
    print("Classes defined in physics_core.py:")
    for class_name in classes:
        print(f"- {class_name}")
    
    print("\nPlease use one of these class names in main.py")

if __name__ == "__main__":
    check_physics_core()