import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os 

import matplotlib as mpl
mpl.rcdefaults()
def draw(X, Y, data, output_directory=".", output_filename="plot.png"):
    """
    Generates a contour plot and saves it to a specified directory and filename.

    Args:
        X (np.ndarray): X-coordinates for the grid.
        Y (np.ndarray): Y-coordinates for the grid.
        data (np.ndarray): Data values for the contour plot.
        output_directory (str): The directory where the plot will be saved.
                                Defaults to the current directory (".").
        output_filename (str): The name of the output file (e.g., "my_plot.png").
                               Defaults to "plot.png".
    """
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)

    plt.figure(figsize=(8, 6))

    contour_fill = plt.contourf(X, Y, data, levels=20, cmap='rainbow', alpha=1, vmin=np.nanmin(data), vmax=np.nanmax(data))

    cbar = plt.colorbar(contour_fill, shrink=1, aspect=10)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel(r'$x_1$', fontsize=38)
    plt.ylabel(r'$x_2$', fontsize=38)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.gca().set_facecolor('white')

    plt.tight_layout()

    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {full_output_path}")
    except Exception as e:
        print(f"Error saving plot to {full_output_path}: {e}")

    plt.show()
    plt.close()