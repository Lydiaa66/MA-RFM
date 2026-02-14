import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Arc
import os

rcParams['text.usetex'] = False

def draw(X, Y, data, theta_max,output_directory=".", output_filename="plot.png"):
    """
    Generates a contour plot on [-0.5, 0.5]^2 and overlays a circular arc centered at (0, 0) with radius 0.5.
    """

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)

    plt.figure(figsize=(8, 6))

    contour_fill = plt.contourf(X, Y, data, levels=20, cmap='rainbow', alpha=1, 
                                vmin=np.nanmin(data), vmax=np.nanmax(data))

    cbar = plt.colorbar(contour_fill, shrink=1, aspect=10)
    cbar.ax.tick_params(labelsize=16)

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.xticks(np.linspace(-0.5, 0.5, 5))
    plt.yticks(np.linspace(-0.5, 0.5, 5))
    arc = Arc((0, 0), width=1.0, height=1.0, angle=0, theta1=0, theta2=theta_max, 
              color='red', linewidth=3)
    plt.gca().add_patch(arc)
    box_x = [-0.3, 0.3, 0.3, -0.3, -0.3]
    box_y = [-0.3, -0.3, 0.3, 0.3, -0.3]
    plt.plot(box_x, box_y, color='black', linewidth=1)
    plt.xlabel(r'$x_1$', fontsize=30)
    plt.ylabel(r'$x_2$', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor('white')

    plt.tight_layout()

    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {full_output_path}")
    except Exception as e:
        print(f"Error saving plot to {full_output_path}: {e}")

    plt.show()
    plt.close()
