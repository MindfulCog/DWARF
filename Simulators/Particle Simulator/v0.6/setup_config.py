
from pathlib import Path

def get_int(prompt, default):
    try:
        return int(input(prompt + f" [{default}]: ") or default)
    except ValueError:
        return default

def get_float(prompt, default):
    try:
        return float(input(prompt + f" [{default}]: ") or default)
    except ValueError:
        return default

GRID_SIZE = 256
TOTAL_STEPS = get_int("Number of steps?", 10000)
DT = get_float("Timestep (DT)?", 0.01)
NUM_PROTONS = get_int("Number of protons?", 1)
NUM_ELECTRONS = get_int("Number of electrons?", 1)
NUM_NEUTRONS = get_int("Number of neutrons?", 0)

PARTICLE_TYPES = {
    "proton": {
        "mass": 16000,
        "color": "gold",
        "spin_rpm": 82373,
        "drag": 0.02,
        "radius": 1.0
    },
    "electron": {
        "mass": 1,
        "color": "cyan",
        "spin_rpm": 120000,
        "drag": 0.01,
        "radius": 0.5
    },
    "neutron": {
        "mass": 16000,
        "color": "gray",
        "spin_rpm": 0,
        "drag": 0.02,
        "radius": 1.0
    }
}

DWARF_FORCE_EXPONENT = 2.22
MEMORY_DECAY = 0.995
SATURATION_LIMIT = 1e-3

config_text = f"""# Auto-generated config

GRID_SIZE = {GRID_SIZE}
TOTAL_STEPS = {TOTAL_STEPS}
DT = {DT}
PROTON_SPIN = {PARTICLE_TYPES['proton']['spin_rpm']}
ELECTRON_SPIN = {PARTICLE_TYPES['electron']['spin_rpm']}
NUM_PROTONS = {NUM_PROTONS}
NUM_ELECTRONS = {NUM_ELECTRONS}
NUM_NEUTRONS = {NUM_NEUTRONS}

DWARF_FORCE_EXPONENT = {DWARF_FORCE_EXPONENT}
MEMORY_DECAY = {MEMORY_DECAY}
SATURATION_LIMIT = {SATURATION_LIMIT}

PARTICLE_TYPES = {{
    "proton": {PARTICLE_TYPES['proton']},
    "electron": {PARTICLE_TYPES['electron']},
    "neutron": {PARTICLE_TYPES['neutron']}
}}
"""

Path("config.py").write_text(config_text)
print("✔ config.py written successfully.")
