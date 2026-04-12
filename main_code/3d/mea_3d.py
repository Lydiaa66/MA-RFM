import numpy as np


def generate_cube_surface(x_min, x_max, y_min, y_max, z_min, z_max, n):
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    z = np.linspace(z_min, z_max, n)
    
    face1 = np.array([[x_min, yi, zi] for yi in y for zi in z])
    face2 = np.array([[x_max, yi, zi] for yi in y for zi in z])
    face3 = np.array([[xi, y_min, zi] for xi in x for zi in z])
    face4 = np.array([[xi, y_max, zi] for xi in x for zi in z])
    face5 = np.array([[xi, yi, z_min] for xi in x for yi in y])
    face6 = np.array([[xi, yi, z_max] for xi in x for yi in y])
    
    points = np.vstack((face1, face2, face3, face4, face5, face6))
    return points
