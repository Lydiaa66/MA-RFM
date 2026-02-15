import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def improved_plot(data, title, x_min, x_max, y_min, y_max,
                  xlabel="x1", ylabel="x2", cmap="plasma", n_ticks=5,shrink=0.5,ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
        created_fig = True

    sns.heatmap(data.T, cmap=cmap, cbar_kws={'shrink': shrink}, cbar=False, ax=ax)

    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)

    x_labels = np.round(np.linspace(x_min, x_max, n_ticks), 2)
    y_labels = np.round(np.linspace(y_min, y_max, n_ticks), 2)
    ax.set_xticks(np.linspace(0, data.shape[0], n_ticks))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(np.linspace(0, data.shape[1], n_ticks))
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.set_aspect('equal', adjustable='box')

    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, shrink=shrink)
    cbar.set_label("Error Magnitude")



    
    

    
