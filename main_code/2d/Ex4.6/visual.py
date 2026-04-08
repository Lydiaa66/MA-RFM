import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys

sys.path.append(r"./main_code2")
import true_bound

def kidney_curve(x, y, h, k, a):
    x_shifted = x - h
    y_shifted = y - k
    if a == 0:
        a = 1e-9
    term1 = (x_shifted**2 + y_shifted**2 - 4 * a**2)**3
    term2 = 108 * a**4 * y_shifted**2
    return term1 - term2

class MeshVisualizer:
    def __init__(self, figsize=(18, 8)):
        """
        网格可视化工具
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        
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
        """
        绘制细分前后的网格对比图。
        细分后的网格将与数值解S_num的云图叠加显示。
        
        Args:
            initial_cells: 初始网格列表
            refined_cells: 细分后的网格列表  
            S_num (optional): 在细分网格上计算的数值解
            points (optional): 数值解对应的坐标点
            save_path (optional): 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        self._plot_single_mesh(fig, ax1, initial_cells, "Initial Mesh")
        
        self._plot_single_mesh(fig, ax2, refined_cells, "Refined Mesh with Solution", 
                               S_num=S_num, points=points)
        
        initial_count = len(self.collect_leaf_cells(initial_cells))
        refined_count = len(self.collect_leaf_cells(refined_cells))
        
        fig.suptitle(f'Adaptive Mesh Refinement Comparison\n'
                     f'Initial: {initial_count} cells → Refined: {refined_count} cells '
                     f'(Added {refined_count - initial_count} cells)', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def _plot_single_mesh(self, fig, ax, cells, title, S_num=None, points=None):
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
                fig.colorbar(contour, ax=ax, label='Solution Value ($S_{num}$)')
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
            
        
        if 'true_bound' in sys.modules:
            func_params = {'h': 0.6, 'k': 0.25, 'a': 0.05}
            contour_points_implicit = true_bound.get_contour_points_from_implicit(
                kidney_curve,
                func_args=func_params,
                x_range=(0, 1), y_range=(0, 1)
            )
            contour_points_implicit = np.array(contour_points_implicit)
            ax.plot(
                contour_points_implicit[:, 0],
                contour_points_implicit[:, 1],
                color='red',
                linewidth=2.5,
                linestyle='--',
                label='True Boundary'
            )
            ax.legend()

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def plot_solution_and_mesh(self, cells, all_points, S_num, grad_S_num, 
                               save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if hasattr(all_points, 'cpu'):
            points = all_points.cpu().detach().numpy()
        else:
            points = all_points
        if hasattr(S_num, 'cpu'):
            solution = S_num.cpu().detach().numpy().flatten()
        else:
            solution = S_num.flatten()
        if hasattr(grad_S_num, 'cpu'):
            gradient = grad_S_num.cpu().detach().numpy()
        else:
            gradient = grad_S_num
        grad_norm = np.linalg.norm(gradient, axis=1)
        if points.shape[0] >= 3:
            contour1 = ax1.tricontourf(points[:, 0], points[:, 1], solution,
                                       levels=14, cmap='viridis', alpha=0.9)
            ax1.set_title('Numerical Solution $S_{num}$ Distribution')
            fig.colorbar(contour1, ax=ax1, label='Solution Value')
        else:
            ax1.text(0.5, 0.5, "Not enough points for tricontourf", ha='center', va='center')
            ax1.set_title('Numerical Solution $S_{num}$ (Not Plotted)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect('equal', adjustable='box')
        if points.shape[0] >= 3:
            contour2 = ax2.tricontourf(points[:, 0], points[:, 1], grad_norm,
                                       levels=14, cmap='plasma', alpha=0.9)
            ax2.set_title('Gradient Norm $||\\nabla S||$ Distribution')
            fig.colorbar(contour2, ax=ax2, label='Gradient Norm')
        else:
            ax2.text(0.5, 0.5, "Not enough points for tricontourf", ha='center', va='center')
            ax2.set_title('Gradient Norm $||\\nabla S||$ (Not Plotted)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Solution and mesh analysis plot saved to: {save_path}")
        plt.show()
        return fig

    def plot_refinement_history(self, refinement_stats, save_path=None):
        history = refinement_stats['history']
        iterations = [info['iteration'] for info in history]
        active_cells = [info['active_cells'] for info in history]
        refined_cells = [info['refined_cells'] for info in history]
        max_indicators = [info['max_indicator'] for info in history]
        mean_indicators = [info['mean_indicator'] for info in history]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        ax1.plot(iterations, active_cells, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Active Cell Count Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Number of Cells')
        ax1.grid(True, alpha=0.3)
        ax2.plot(iterations, active_cells, 'go-', linewidth=2, markersize=8)
        ax2.set_title('Refined Cells per Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Refined Cells')
        ax2.grid(True, alpha=0.3)
        ax3.plot(iterations, max_indicators, 'ro-', linewidth=2, markersize=8)
        ax3.set_title('Maximum Indicator Value Evolution')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Max Indicator Value')
        ax3.grid(True, alpha=0.3)
        ax4.plot(iterations, mean_indicators, 'go-', linewidth=2, markersize=8)
        ax4.set_title('Mean Indicator Value Evolution')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Mean Indicator Value')
        ax4.grid(True, alpha=0.3)
        plt.suptitle(f'Adaptive Refinement Statistics ({len(history)} iterations)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Refinement history plot saved to: {save_path}")
        plt.show()
        return fig
def visualize_adaptive_refinement(cells, refined_cells, all_g_p, S_num, grad_S_num, 
                                  refinement_stats, cell_indicators=None):
    """
    完整的自适应细分可视化
    """
    visualizer = MeshVisualizer(figsize=(20, 8))
    
    print("Generating visualizations...")
    
    print("1. Plotting mesh comparison with solution overlay...")
    visualizer.plot_mesh_comparison(cells, refined_cells, 
                                     S_num=S_num, 
                                     points=all_g_p)
    
    
    
    print("All visualizations completed!")
