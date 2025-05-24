
import bpy
import csv
import os

# Set your tracer CSV folder path here
csv_folder = bpy.path.abspath("//")

# Import a single tracer file and create a curve
def import_tracer(file_path, tracer_name):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        coords = [(float(row['x']), float(row['y']), float(row['z'])) for row in reader]

    # Create a new curve
    curve_data = bpy.data.curves.new(name=tracer_name, type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(coords)-1)

    for i, coord in enumerate(coords):
        x, y, z = coord
        polyline.points[i].co = (x, y, z, 1)

    curve_obj = bpy.data.objects.new(tracer_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

# Auto-import all CSVs in the directory
for file_name in os.listdir(csv_folder):
    if file_name.endswith(".csv") and file_name.startswith("tracer_"):
        file_path = os.path.join(csv_folder, file_name)
        import_tracer(file_path, file_name[:-4])
