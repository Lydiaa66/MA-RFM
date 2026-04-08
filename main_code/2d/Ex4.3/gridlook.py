import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys
import matplotlib as mpl
mpl.rcdefaults()
sys.path.append(r"./main_code2")

def kidney_curve(x, y, h, k, a):
    x_shifted = x - h
    y_shifted = y - k
    if a == 0:
        a = 1e-9
    term1 = (x_shifted**2 + y_shifted**2 - 4 * a**2)**3
    term2 = 108 * a**4 * y_shifted**2
    return term1 - term2

class MeshVisualizer:
    def __init__(self, figsize=(30, 8)):
        """
        网格可视化工具
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.func_params = {'h': 0.6, 'k': 0.25, 'a': 0.05}


    def collect_leaf_cells(self, cells):
        """递归收集所有叶子节点"""
        leaf_cells = []
        for cell in cells:
            if not cell.children:
                leaf_cells.append(cell)
            else:
                leaf_cells.extend(self.collect_leaf_cells(cell.children))
        return leaf_cells
    
    def plot_mesh_comparison(self, initial_cells, refined_cells,
                             S_num=None, points=None, save_path=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        self._plot_single_mesh(fig, ax1, initial_cells)
        
        self._plot_single_mesh(fig, ax2, refined_cells,
                               S_num=S_num, points=points)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_mesh_evolution(self, cells_store, S_num_stores=None, points=None, save_path=None,label=None):
        """
        绘制网格演化图
        参数：
            cell_stores: list，每个元素是某一步的网格单元
            S_num_stores: list 或 None，每一步对应的数值解 (可选)
            points: 可选，数值解的点
            save_path: 保存路径
        """
        n_steps = len(cells_store)
        fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))
        if n_steps == 1:
            axes = [axes]

        for i, cells in enumerate(cells_store):
            S_num = S_num_stores[i] if S_num_stores is not None else None
            if S_num_stores is not None and i < len(S_num_stores):
                S_num = S_num_stores[i]

            self._plot_single_mesh(fig, axes[i], cells, S_num=S_num, points=points,label=label)
            axes[i].set_title(f"Cell {i}",fontsize=20)
            boundary_circle = patches.Circle((0.5, 0.5), 0.2,
                                            linewidth=2,
                                            edgecolor='r',
                                            facecolor='none',
                                            label='Boundary Layer')

            axes[i].add_patch(boundary_circle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()
        return fig



    def _plot_single_mesh(self, fig, ax, cells, S_num=None, points=None,label=None):
        """
        绘制单个网格。如果提供了S_num和points，则将其作为背景云图绘制。
        """
        leaf_cells = self.collect_leaf_cells(cells)
        
        if S_num is not None and points is not None:
            p_np = points.cpu().detach().numpy() if hasattr(points, 'cpu') else np.array(points)
            s_np = S_num.cpu().detach().numpy().flatten() if hasattr(S_num, 'cpu') else np.array(S_num).flatten()
            
            if p_np.shape[1] >= 2:
                contour = ax.tricontourf(p_np[:, 0], p_np[:, 1], s_np,
                                         levels=20, cmap='rainbow', alpha=0.9)
                cbar = fig.colorbar(contour, ax=ax)
                cbar.ax.tick_params(labelsize=16)
            else:
                ax.text(0.5, 0.5, "Not enough points for contour plot", 
                        ha='center', va='center', transform=ax.transAxes)

        for cell in leaf_cells:
            rect = patches.Rectangle(
                (cell.x0, cell.y0), cell.size, cell.size,
                linewidth=0.8,
                edgecolor='black',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
        

        boundary_circle = patches.Circle((0.5, 0.5), 0.2,
                                        linewidth=1,
                                        edgecolor='r',
                                        facecolor='none',
                                        label='Boundary Layer')

        ax.add_patch(boundary_circle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x_1$',fontsize=24)
        ax.set_ylabel('$x_2$',fontsize=24)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        if label is None:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=20)
        xlabel="$x_1$"
        ylabel="$x_2$"
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        n_ticks=5

        return fig


