import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Serif",
    "axes.unicode_minus": False
})
def draw(X, Y, data,data_1D, x_label,y_label,xx,output_directory=".", output_filename1="plot.png",output_filename2="plot.png",temp_point=False,temp_v=False):
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

    full_output_path1 = os.path.join(output_directory, output_filename1)
    full_output_path2 = os.path.join(output_directory, output_filename2)
    plt.figure(figsize=(8, 6))

    contour_fill = plt.contourf(X, Y, data, levels=20, cmap='rainbow', alpha=1, vmin=np.nanmin(data), vmax=np.nanmax(data))

    cbar = plt.colorbar(contour_fill, shrink=1, aspect=10)
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.gca().set_facecolor('white')
    if temp_point:
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)

        max_idx = np.nanargmax(data)
        min_idx = np.nanargmin(data)

        x_max = X.flat[max_idx]
        y_max = Y.flat[max_idx]
        x_min = X.flat[min_idx]
        y_min = Y.flat[min_idx]

        dx = 0.05 * (X.max() - X.min())
        dy = 0.05 * (Y.max() - Y.min())

        plt.plot(x_max, y_max, 'ro', markersize=6)
        plt.annotate(f"Max:({x_max:.2f}, {y_max:.2f})",
                    xy=(x_max, y_max), xycoords='data',
                    xytext=(x_max + dx, y_max + dy), textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=12, color='black', weight='bold', ha='left', va='bottom')

        plt.plot(x_min, y_min, 'bo', markersize=6)
        plt.annotate(f"Min:({x_min:.2f}, {y_min:.2f})",
                    xy=(x_min, y_min), xycoords='data',
                    xytext=(x_min - dx, y_min - dy), textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=12, color='black', weight='bold', ha='right', va='top')
    plt.tight_layout()
    plt.savefig(full_output_path1, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {full_output_path1}")
    if temp_v:
        plt.figure(figsize=(5, 3), dpi=120)
        x_devide = np.linspace(0, 1, len(data_1D))
        abs_data = np.abs(data_1D)
        max_abs_val = np.max(abs_data)
        max_abs_idx = np.argmax(abs_data)
        half_max = data_1D[max_abs_idx] / 2
        half_max_indices = np.where((data_1D >= half_max - 10**(-2.2)) & (data_1D <= half_max + 10**(-2.2)))
        half_max_indices_x = half_max_indices[0]
        plt.figure(figsize=(5, 3), dpi=120)
        plt.plot(range(int(data.shape[0])), data_1D, linestyle='-', color='blue',label="Num_S(x1, 0.5, 0.3)")
        plt.axhline(y=half_max, color='r', linestyle='--', label='1/2 Maximum Value')
        for i in range(len(half_max_indices_x)):
            idx=half_max_indices_x[i]
            plt.axvline(x=half_max_indices_x[i], color='black', linestyle='--', linewidth=1)
            x_val = x_devide[idx]
            plt.text(idx+1, data_1D[idx],f"{xx}={x_val:.2f}", color='black', fontsize=7, rotation=0, va='bottom')
        plt.xlabel(xx, fontsize=14)
        n_ticks=7
        x_labels = np.linspace(x_min, x_max, n_ticks)
        y_labels = np.linspace(y_min, y_max, n_ticks)
        x_labels = np.round(x_labels, decimals=2)
        y_labels = np.round(y_labels, decimals=2)
        plt.xticks(ticks=np.linspace(0, data.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)
        plt.savefig(full_output_path2, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {full_output_path2}")

    plt.show()
    plt.close()