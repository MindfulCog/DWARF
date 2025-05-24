from skyfield.api import load
import numpy as np

# Load ephemeris data
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']
moon = eph['moon']
sun = eph['sun']

# Returns Moon and Sun 2D Cartesian positions (in meters) at given UTC time.
def get_positions_utc(year, month, day, hour, minute, second):
    t = ts.utc(year, month, day, hour, minute, second)

    # Positions relative to Earth
    moon_pos = earth.at(t).observe(moon).apparent().ecliptic_position().au
    sun_pos = earth.at(t).observe(sun).apparent().ecliptic_position().au

    # Convert from AU to meters
    au_to_m = 1.496e11
    moon_vec = np.array(moon_pos) * au_to_m
    sun_vec  = np.array(sun_pos)  * au_to_m


    return moon_vec[:2], sun_vec[:2]  # Return 2D projection for DWARF

# Example usage:
# moon_xy, sun_xy = get_positions_utc(2025, 5, 1, 12, 0, 0)
