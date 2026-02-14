import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
                           cell_indicators=None, save_path=None):
        """
        绘制细分前后的网格对比图
        
        Args:
            initial_cells: 初始网格列表
            refined_cells: 细分后的网格列表  
            cell_indicators: 网格指示器值字典（用于颜色映射）
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        self._plot_single_mesh(ax1, initial_cells, "Initial Mesh", 
                              cell_indicators=cell_indicators)
        
        self._plot_single_mesh(ax2, refined_cells, "Refined Mesh")
        
        initial_count = len(self.collect_leaf_cells(initial_cells))
        refined_count = len(self.collect_leaf_cells(refined_cells))
        
        fig.suptitle(f'Adaptive Mesh Refinement Comparison\n'
                    f'Initial: {initial_count} cells → Refined: {refined_count} cells '
                    f'(Added {refined_count - initial_count} cells)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_single_mesh(self, ax, cells, title, cell_indicators=None):
        """绘制单个网格"""
        leaf_cells = self.collect_leaf_cells(cells)
        all_levels = []
        for cell in leaf_cells:
            if hasattr(cell, 'level'):
                all_levels.append(cell.level)
        
        level_to_color = {}
        if all_levels:
            unique_levels = sorted(list(set(all_levels)))
            num_unique_levels = len(unique_levels)
            if num_unique_levels == 1:
                color_palette = [plt.cm.viridis(0.5)]
            elif num_unique_levels <= 10:
                color_palette = plt.cm.get_cmap('viridis', num_unique_levels)
            else:
                color_palette = plt.cm.get_cmap('viridis', num_unique_levels)
            current_colors = []
            if callable(color_palette):
                    for i in range(num_unique_levels):
                        current_colors.append(color_palette(i / max(1, num_unique_levels -1) if num_unique_levels > 1 else 0.0 ))
            else:
                    current_colors = color_palette
                    
            level_to_color = {level: current_colors[i] for i, level in enumerate(unique_levels)}
        else:
            print("Warning: No 'level' attribute found in any cells for level-based coloring.")

        for i, cell in enumerate(leaf_cells):
            rect = patches.Rectangle((cell.x0, cell.y0), cell.size, cell.size,
                                   linewidth=1, edgecolor='black', 
                                   facecolor='lightblue', alpha=0.7) 
            

            current_face_color = 'lightgray'
            if hasattr(cell, 'level'):
                if cell.level in level_to_color:
                    current_face_color = level_to_color[cell.level]
            rect.set_facecolor(current_face_color)
            ax.add_patch(rect)
            
            if hasattr(cell, 'level'):
                ax.text(cell.x0 + cell.size/2, cell.y0 + cell.size/2, 
                       f'{cell.level}', ha='center', va='center', 
                       fontsize=5, fontweight='bold')
            
        boundary_circle = patches.Circle((0.5, 0.5), 0.2,
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none',
                                        label='Boundary Layer')

        ax.add_patch(boundary_circle)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def plot_solution_and_mesh(self, cells, all_points, S_num, grad_S_num, 
                              save_path=None):
        """
        绘制数值解、梯度和网格的组合图
        
        Args:
            cells: 网格列表
            all_points: 所有高斯点坐标
            S_num: 数值解
            grad_S_num: 梯度
            save_path: 保存路径
        """
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
        """
        绘制细分历史统计图
        
        Args:
            refinement_stats: 细分统计信息
            save_path: 保存路径
        """
        if not refinement_stats['history']:
            print("No refinement history data available")
            return
        
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

    def create_detailed_mesh_plot(self, cells, cell_indicators=None, 
                                 gauss_points=None, save_path=None):
        """
        创建详细的网格图，包含高斯点
        
        Args:
            cells: 网格列表
            cell_indicators: 指示器值
            gauss_points: 高斯点坐标
            save_path: 保存路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        leaf_cells = self.collect_leaf_cells(cells)
        
        for i, cell in enumerate(leaf_cells):
            if cell_indicators and i in cell_indicators:
                indicator_val = cell_indicators[i]
                max_indicator = max(cell_indicators.values())
                min_indicator = min(cell_indicators.values())
                if max_indicator > min_indicator:
                    normalized = (indicator_val - min_indicator) / (max_indicator - min_indicator)
                    color = plt.cm.Reds(normalized)
                else:
                    color = 'lightblue'
            else:
                color = 'lightblue'
            
            rect = patches.Rectangle((cell.x0, cell.y0), cell.size, cell.size,
                                   linewidth=1.5, edgecolor='black', 
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            ax.text(cell.x0 + cell.size/2, cell.y0 + cell.size/2, 
                   str(i), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='blue')
        
        if gauss_points is not None:
            if hasattr(gauss_points, 'cpu'):
                points = gauss_points.cpu().detach().numpy()
            else:
                points = gauss_points
            ax.scatter(points[:, 0], points[:, 1], 
                      c='red', s=10, alpha=0.8, marker='o')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Detailed Mesh Distribution (Red dots: Gauss points)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed mesh plot saved to: {save_path}")
        
        plt.show()
        return fig


def visualize_adaptive_refinement(cells, refined_cells, all_g_p, S_num, grad_S_num, 
                                 refinement_stats, cell_indicators=None):
    """
    完整的自适应细分可视化
    
    Args:
        cells: 初始网格
        refined_cells: 细分后网格  
        all_g_p: 高斯点
        S_num: 数值解
        grad_S_num: 梯度
        refinement_stats: 细分统计
        cell_indicators: 指示器值
    """
    visualizer = MeshVisualizer()
    
    print("Generating visualizations...")
    
    print("1. Plotting mesh comparison...")
    visualizer.plot_mesh_comparison(cells, refined_cells, cell_indicators)
    
    
    
    
    print("All visualizations completed!")

"""
# 在你的自适应细分完成后添加：
print("开始可视化...")
visualize_adaptive_refinement(
    cells=cells,                    # 初始网格
    refined_cells=refined_cells,    # 细分后网格
    all_g_p=all_g_p,              # 高斯点
    S_num=S_num,                   # 数值解
    grad_S_num=grad_S_num,         # 梯度
    refinement_stats=refinement_stats,  # 细分统计
    cell_indicators=amr.compute_cell_indicators(...)  # 指示器值
)
"""