from tkinter import font
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys
import matplotlib as mpl
mpl.rcdefaults()
sys.path.append(r"./main_code2")
import true_bound,smooth

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
        self.contour_points_implicit = true_bound.get_contour_points_from_implicit(
            kidney_curve,
            func_args=self.func_params,
            x_range=(0, 1),
            y_range=(0, 1)
        )

    def collect_leaf_cells(self, cells):
        """递归收集所有叶子节点"""
        leaf_cells = []
        for cell in cells:
            if not cell.children:
                leaf_cells.append(cell)
            else:
                leaf_cells.extend(self.collect_leaf_cells(cell.children))
        return leaf_cells
    
    def plot_mesh_comparison(self, initial_cells, refined_cells, kidney_bound,
                             S_num=None, points=None, save_path=None):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
        
        self._plot_single_mesh(fig, ax1, initial_cells)
        
        self._plot_single_mesh(fig, ax2, refined_cells,
                               S_num=S_num, points=points)
        self._scale(fig,ax3,kidney_bound)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig
    
    def _scale(self,fig,ax,kidney_bound):
        offset_distances = np.linspace(-0.02,0.02
                                    ,2)


        kidney_boundary=kidney_bound
        alpha=2*10**(-3)
        contour_points_scaled = smooth.simple_offset_batch(kidney_boundary, alpha,offset_distances)

        x = np.linspace(0, 1, 150)
        y = np.linspace(0, 1, 150)
        X, Y = np.meshgrid(x, y)

        Z = true_bound.generate_soft_boundary(X, Y, self.contour_points_implicit, K=2000)

        ax.clear()

        ax.plot(kidney_boundary[:, 0], kidney_boundary[:, 1], c='g', linewidth=1, label='Original detected boundary')
        ax.plot(
                self.contour_points_implicit[:, 0],
                self.contour_points_implicit[:, 1],
                color='red',
                linewidth=3,
                linestyle='--',
                label='True Boundary'
            )
        colors = ["#470eaa", "#fb5817"]


        for i, d in enumerate(offset_distances):
            ax.plot(contour_points_scaled[i,:, 0], contour_points_scaled[i,:, 1], color=colors[i], linewidth=1, label=rf'Offset boundary($\rho={d}$)')
            if abs(d) > 1e-6:
                if d > 0:
                    label_position_index = len(contour_points_scaled[1]) // 2
                else:
                    label_position_index = ( len(contour_points_scaled[1])) // 2
                    
                point_for_label = contour_points_scaled[i,label_position_index,:]
                
                ax.text(
                    x=point_for_label[0],
                    y=point_for_label[1],
                    s=f'{d:.3f}',
                    color=colors[i],
                    fontweight='bold',
                    fontsize=14,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5)
                )
                ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        ax.set_xlabel('$x_1$', fontsize=30)
        ax.set_ylabel('$x_2$', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=16, loc='best')
        plt.tight_layout(rect=[0, 0, 1, 1])
        return fig

    def _plot_single_mesh(self, fig, ax, cells, S_num=None, points=None,save_path=None):
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
                cbar.ax.tick_params(labelsize=18)
            else:

                ax.text(0.5, 0.5, "Not enough points for contour plot", 
                ha='center', va='center', transform=ax.transAxes)
        else:
            import matplotlib.cm as cm
            import matplotlib.colors as colors

            dummy_norm = colors.Normalize(vmin=0, vmax=1)
            dummy_mappable = cm.ScalarMappable(norm=dummy_norm, cmap='rainbow')
            dummy_mappable.set_array([])

            cbar = fig.colorbar(dummy_mappable, ax=ax)
            cbar.ax.set_visible(False)  

        for cell in leaf_cells:
            rect = patches.Rectangle(
                (cell.x0, cell.y0), cell.size, cell.size,
                linewidth=0.8,
                edgecolor='black',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
        
        if 'true_bound' in sys.modules:
            contour_points_implicit = np.array(self.contour_points_implicit)
            ax.plot(
                contour_points_implicit[:, 0],
                contour_points_implicit[:, 1],
                color='red',
                linewidth=3,
                linestyle='--',
                label='True Boundary'
            )
            ax.legend(fontsize=18)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        ax.set_xlabel('$x_1$',fontsize=30)
        ax.set_ylabel('$x_2$',fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout(rect=[0, 0, 1, 1])
        return fig


