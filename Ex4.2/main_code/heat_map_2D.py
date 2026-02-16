import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def improved_plot(data, title, x_min, x_max, y_min, y_max, xlabel="x1", ylabel="x2", cmap="plasma", n_ticks=5):
    plt.figure(figsize=(3, 3),dpi=120)
    ax = sns.heatmap(data.T, cmap=cmap, cbar_kws={'shrink': 0.8},cbar=False)
    plt.gca().invert_yaxis()
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    x_labels = np.linspace(x_min, x_max, n_ticks)
    y_labels = np.linspace(y_min, y_max, n_ticks)

    x_labels = np.round(x_labels, decimals=2)
    y_labels = np.round(y_labels, decimals=2)
    plt.xticks(ticks=np.linspace(0, data.shape[0], n_ticks), labels=x_labels, rotation=0, fontsize=10)
    plt.yticks(ticks=np.linspace(0, data.shape[1], n_ticks), labels=y_labels, fontsize=10)
    
    plt.title(title, fontsize=16, pad=15)
    plt.colorbar(ax.collections[0], label="Error Magnitude")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()