import os
import numpy as np


def read_airfoil_coordinates(airfoil_path, airfoil_name):
    file_path = os.path.join(airfoil_path, airfoil_name)

    with open(file_path, 'r') as f:
        #print(f)
        lines = f.readlines()
        lines = lines[1:]

    x_coords = []
    y_coords = []
    for line in lines:
        if not line.strip():
            continue
        x, y = line.split()
        x_coords.append(float(x))
        y_coords.append(float(y))
    return np.array(x_coords), np.array(y_coords)