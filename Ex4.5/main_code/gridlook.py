import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull
import Bound_detect

import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
def kidney_curve(x, y, h, k, a):
    x_shifted = x - h
    y_shifted = y - k
    if a == 0:
        a = 1e-9
    term1 = (x_shifted**2 + y_shifted**2 - 4 * a**2)**3
    term2 = 108 * a**4 * y_shifted**2
    return term1 - term2

def map_index_to_domain(points,tau_x_min,tau_x_max,tau_y_min,tau_y_max,shape):
    """
    将索引点映射到定义域
    :param points: 索引点数组，形状为 (n, 2)
    :param tau_x_min: x 轴最小值
    :param tau_x_max: x 轴最大值
    :param tau_y_min: y 轴最小值
    :param tau_y_max: y 轴最大值
    :param shape: 数组形状 (height, width)
    :return: 映射后的点数组
    """
    x_scale=(tau_x_max-tau_x_min)/shape[1]
    y_scale=(tau_y_max-tau_y_min)/shape[0]
    
    mapped_points=np.zeros_like(points,dtype=float)
    mapped_points[:,0] = tau_x_min + points[:,0]*x_scale
    mapped_points[:,1] = tau_y_min + points[:,1]*y_scale
    return mapped_points


def detect(grad,X,Y,tau_x_min,tau_x_max,tau_y_min,tau_y_max,para,ax,fig):
    
    shape=grad.shape[0]
    mask=np.where(grad>np.max(grad)/para)
    
    mask1=mask[0]
    mask2=mask[1]
    boolean=(mask[0]>=5) & (mask[0]<=shape-5)
    mask=(mask1[boolean],mask2[boolean])

    values = grad[mask[0], mask[1]] 

    sc = ax.scatter(mask[0], mask[1], c=values, cmap="jet", marker='o', s=5,
                    edgecolor='white', linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=1) 
    cbar.ax.tick_params(labelsize=20)

    mask=np.array(mask).T
    hull = ConvexHull(mask)
    hull_points = mask[hull.vertices]
    
    line=0.5*(np.min(mask[:,0])+np.max(mask[:,0]))
    s1=np.where(mask[:,0]<line)[0]
    s2=np.where(mask[:,0]>line)[0]
    area1=mask[s1,:]
    area2=mask[s2,:]
    
    aa1_hull=ConvexHull(area1)
    aa1=area1[aa1_hull.vertices]
    aa2_hull=ConvexHull(area2)
    aa2=area2[aa2_hull.vertices]
    
    xlabel="$x_1$"
    ylabel="$x_2$"
    n_ticks=5
    x_labels = np.linspace(tau_x_min, tau_x_max, n_ticks)
    y_labels = np.linspace(tau_y_min, tau_y_max, n_ticks)

    x_labels = np.round(x_labels, decimals=2)
    y_labels = np.round(y_labels, decimals=2)
    plt.xticks(ticks=np.linspace(0, grad.shape[0], n_ticks), labels=x_labels, rotation=0, fontsize=20)
    plt.yticks(ticks=np.linspace(0, grad.shape[1], n_ticks), labels=y_labels, fontsize=20)
    ax.set_title("$\mathcal{Q}_{grad}$", fontsize=24)
    plt.legend()

    plt.show()



class MeshVisualizer:
    def __init__(self, figsize=(52, 8)):
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
    
    def plot_mesh_evolution(self, cells_store, S_num_stores=None, points=None, save_path=None,label=None,temp=None,g_S=None):
        """
        绘制网格演化图
        参数：
            cell_stores: list，每个元素是某一步的网格单元
            S_num_stores: list 或 None，每一步对应的数值解 (可选)
            points: 可选，数值解的点
            save_path: 保存路径
        """
        n_steps = len(cells_store)

        if points is not None:
             if hasattr(points, 'cpu'): points_np = points.cpu().detach().numpy()
             else: points_np = np.array(points)
             
             if points_np.shape[0] > 0:
                 x_min, x_max = points_np[:,0].min(), points_np[:,0].max()
                 y_min, y_max = points_np[:,1].min(), points_np[:,1].max()
             else:
                 x_min, x_max, y_min, y_max = -0.3, 0.3, -0.3, 0.3
        else:
             x_min, x_max, y_min, y_max = -0.3, 0.3, -0.3, 0.3

        if temp==True:
            fig, axes = plt.subplots(1, n_steps+1, figsize=(6.5* (n_steps+1), 7))
        else:
            fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))
        if n_steps == 1:
            axes = [axes]

        is_single_solution = False
        if S_num_stores is not None:
             if isinstance(S_num_stores, (list, tuple)) and len(S_num_stores) == n_steps:
                 is_single_solution = False
             elif isinstance(S_num_stores, np.ndarray) and S_num_stores.ndim > 1 and len(S_num_stores) == n_steps:
                 is_single_solution = False
             else:
                 is_single_solution = True

        for i, cells in enumerate(cells_store):
            S_num = None
            if S_num_stores is not None:
                if is_single_solution:
                    S_num = S_num_stores
                elif i < len(S_num_stores):
                    S_num = S_num_stores[i]

            self._plot_single_mesh(fig, axes[i], cells, S_num=None, points=points,label=label, x_range=(x_min, x_max), y_range=(y_min, y_max))
            axes[i].set_title(f"$\mathcal{{C}}_{{{i}}}$",fontsize=40)
            axes[i].tick_params(axis='x', labelsize=28)
            #axes[i].tick_params(axis='y', labelsize=28)
            boundary_circle = patches.Circle((0.71, 0.5), 0.2,
                                            linewidth=2,
                                            edgecolor='r',
                                            facecolor='none',
                                            label='Boundary Layer')

            axes[i].add_patch(boundary_circle)
            boundary_rectangle = patches.Rectangle((0.29, 0.3), 0.2, 0.4,
                                        linewidth=2,
                                        edgecolor='b',
                                        facecolor='none',
                                        label='Defined Rectangle')
            axes[i].add_patch(boundary_rectangle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if temp and g_S is not None:
            tau_x_min,tau_x_max=x_min, x_max
            tau_y_min,tau_y_max=y_min, y_max
            x=np.linspace(tau_x_min,tau_x_max,301)
            y=np.linspace(tau_y_min,tau_y_max,301)
            X,Y=np.meshgrid(x,y)
            eps_cluster=0.01
            mini_samples=20
            para_abs=3
            para_grad=3
            
            S_for_detect = S_num_stores
            if S_num_stores is not None and not is_single_solution:
                 if isinstance(S_num_stores, (list, tuple)):
                      S_for_detect = S_num_stores[-1]
                 elif isinstance(S_num_stores, np.ndarray) and len(S_num_stores) == n_steps:
                      S_for_detect = S_num_stores[-1]

            Bound_detect.detect_shape(g_S,X.T,Y.T,tau_x_min,tau_x_max,tau_y_min,tau_y_max,para_abs,para_grad,eps_cluster,mini_samples,
                                      S_num=S_for_detect,ax=axes[-1], fig=fig,
                                      output_directory="./Noise/noise=10%", output_filename="plot.pdf")
            if axes[-1].collections:
                mappable = axes[-1].collections[0]
            else:
                mappable = axes[-1].images[0]
            cbar = fig.colorbar(mappable, ax=axes, orientation='vertical',
                                fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=24)
            cbar.set_label(r'$S^{(0)}$', fontsize=30)
            axes[-1].tick_params(axis='x', labelsize=28)
            axes[-1].set_xlabel("$x_1$", fontsize=36)
            #axes[-1].set_ylabel("$x_2$", fontsize=36)
            #axes[-1].tick_params(axis='y', labelsize=28)
            axes[-1].set_title("$\mathcal{Q}$", fontsize=40)
        axes[0].set_ylabel("$x_2$", fontsize=36)
        x_range=(x_min, x_max)
        y_range=(y_min, y_max)
        axes[0].set_xlim(x_range[0], x_range[1])
        axes[0].set_ylim(y_range[0], y_range[1])
        n_ticks=5
        x_labels = np.linspace(x_range[0], x_range[1], n_ticks)
        y_labels = np.linspace(y_range[0], y_range[1], n_ticks)

        x_labels = np.round(x_labels, decimals=2)
        y_labels = np.round(y_labels, decimals=2)
        
        axes[0].set_xticks(np.linspace(x_range[0], x_range[1], n_ticks))
        axes[0].set_xticklabels(x_labels, rotation=0, fontsize=20)
        axes[0].set_yticks(np.linspace(y_range[0], y_range[1], n_ticks))
        axes[0].set_yticklabels(y_labels, fontsize=20)
        
        # ax.set_xlabel('$x_1$',fontsize=34)
        #ax.set_ylabel('$x_2$',fontsize=34)
        axes[0].set_aspect('equal')
        axes[0].grid(True, linestyle=':', alpha=0.2)
        # if label is None:
        #     ax.set_xlim(x_range[0], x_range[1])
        #     ax.set_ylim(y_range[0], y_range[1])
        #     ax.set_xlabel("")
        #     ax.set_ylabel("")
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        axes[0].tick_params(axis='both', which='major', labelsize=18)
        axes[0].tick_params(axis='x', labelsize=28)
        axes[0].tick_params(axis='y', labelsize=28)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()
        return fig



    def _plot_single_mesh(self, fig, ax, cells, S_num=None, points=None,label=None, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3)):
        """
        绘制单个网格。如果提供了S_num和points，则将其作为背景云图绘制。
        """
        leaf_cells = self.collect_leaf_cells(cells)
        
        if S_num is not None and points is not None:
            p_np = points.cpu().detach().numpy() if hasattr(points, 'cpu') else np.array(points)
            s_np = S_num.cpu().detach().numpy().flatten() if hasattr(S_num, 'cpu') else np.array(S_num).flatten()
            
            if p_np.shape[1] >= 2 and s_np.size == p_np.shape[0]:
                contour = ax.tricontourf(p_np[:, 0], p_np[:, 1], s_np,
                                         levels=20, cmap='rainbow', alpha=0.9)
                cbar = fig.colorbar(contour, ax=ax)
            else:
                if s_np.size != p_np.shape[0]:
                    print(f"Warning: S_num shape {s_np.shape} mismatch with points {p_np.shape}, skipping contour")
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
            


        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        n_ticks=5
        x_labels = np.linspace(x_range[0], x_range[1], n_ticks)
        y_labels = np.linspace(y_range[0], y_range[1], n_ticks)

        x_labels = np.round(x_labels, decimals=2)
        y_labels = np.round(y_labels, decimals=2)
        
        ax.set_xticks(np.linspace(x_range[0], x_range[1], n_ticks))
        ax.set_xticklabels(x_labels, rotation=0, fontsize=20)
        ax.set_yticks([])
        # ax.set_yticks(np.linspace(y_range[0], y_range[1], n_ticks))
        # ax.set_yticklabels(y_labels, fontsize=20)
        
        ax.set_xlabel('$x_1$',fontsize=34)
        # ax.set_ylabel('$x_2$',fontsize=34)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        if label is None:
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=18)



