import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib import rcParams
import os 


def map_index_to_domain_3d(points, domain_min, domain_max, shape):
    """
    将3D索引点映射到物理定义域。
    :param points: 索引点数组，形状为 (n, 3)
    :param domain_min: 坐标轴最小值 (x_min, y_min, z_min)
    :param domain_max: 坐标轴最大值 (x_max, y_max, z_max)
    :param shape: 3D数组形状 (depth, height, width)
    :return: 映射后的点数组
    """
    scales = (np.array(domain_max) - np.array(domain_min)) / (np.array(shape) - 1)
    
    mapped_points = np.zeros_like(points, dtype=float)
    mapped_points[:, 0] = domain_min[0] + points[:, 0] * scales[0]
    mapped_points[:, 1] = domain_min[1] + points[:, 1] * scales[1]
    mapped_points[:, 2] = domain_min[2] + points[:, 2] * scales[2]

    return mapped_points


import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def set_axes_equal(ax):
    """让3D图的XYZ刻度间距相同"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range]) / 2.0

    mid_x = (x_limits[0] + x_limits[1]) / 2
    mid_y = (y_limits[0] + y_limits[1]) / 2
    mid_z = (z_limits[0] + z_limits[1]) / 2

    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

def detect_3d(grad, S_num, domain_min, domain_max, para_grad, para_abs, elev,azim,center,choose,
              output_directory=".", output_filename="plot_3d.png"):
    
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    full_output_path = os.path.join(output_directory, output_filename)

    shape_3d = grad.shape
    grad_threshold = np.max(grad) / para_grad
    abs_threshold = np.max(np.abs(S_num)) / para_abs
    mask = (grad > grad_threshold) & (np.abs(S_num) > abs_threshold)
    mask_indices = np.array(np.where(mask)).T

    if len(mask_indices) < 4:
        print("Warning: Not enough points to form a 3D convex hull.")
        return

    mapped_mask_points = map_index_to_domain_3d(mask_indices, domain_min, domain_max, shape_3d)
    min_x_point = mapped_mask_points[np.argmin(mapped_mask_points[:, 0])]
    max_x_point = mapped_mask_points[np.argmax(mapped_mask_points[:, 0])]
    min_y_point = mapped_mask_points[np.argmin(mapped_mask_points[:, 1])]
    max_y_point = mapped_mask_points[np.argmax(mapped_mask_points[:, 1])]
    min_z_point = mapped_mask_points[np.argmin(mapped_mask_points[:, -1])]
    max_z_point = mapped_mask_points[np.argmax(mapped_mask_points[:, -1])]

    if choose=="grad":
        grad_values = grad[mask]
        norm = plt.Normalize(vmin=np.min(grad_values), vmax=np.max(grad_values))
        
        donut_cmap = LinearSegmentedColormap.from_list(
            "donut_cmap",
            ["#fa8072", "#e84141", "#f23737"],
            N=256
        )
        colors_grad = donut_cmap(norm(grad_values))
        norm_grad = plt.Normalize(vmin=np.min(grad_values), vmax=np.max(grad_values))
        cmap_grad = plt.get_cmap('rainbow')
        colors_grad = cmap_grad(norm_grad(grad_values))

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.set_xlim([domain_min[0], domain_max[0]])
        ax.set_ylim([domain_min[1], domain_max[1]])
        ax.set_zlim([domain_min[2], domain_max[2]])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='z', which='major', labelsize=16)
        ax.set_xlabel("X", fontsize=14, labelpad=10)
        ax.set_ylabel("Y", fontsize=14, labelpad=10)
        ax.set_zlabel("Z", fontsize=14, labelpad=10)
        
        ax.view_init(elev=elev, azim=azim)
        offset = 0.05 
        ax.scatter(mapped_mask_points[:, 0], 
                mapped_mask_points[:, 1], 
                mapped_mask_points[:, 2],
                c=colors_grad, s=3, alpha=0.6, edgecolors='none')
        cmap = plt.get_cmap("rainbow")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.001)
        cbar.ax.tick_params(labelsize=18)

        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    if choose=="S":
        S_values = S_num[mask]
        norm = plt.Normalize(vmin=np.min(S_values), vmax=np.max(S_values))
        
        norm_S = plt.Normalize(vmin=np.min(S_values), vmax=np.max(S_values))
        cmap_S = plt.get_cmap('rainbow')
        colors_S = cmap_S(norm_S(S_values))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.set_position([0.01, 0.02, 0.75, 0.96])
        ax.set_xlim([domain_min[0], domain_max[0]])
        ax.set_ylim([domain_min[1], domain_max[1]])
        ax.set_zlim([domain_min[2], domain_max[2]])
        ax.tick_params(axis='both', which='major', labelsize=16,pad=6)
        ax.tick_params(axis='x', which='major', labelsize=16,pad=6)

        ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
        ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
        ax.set_zlabel("$x_3$", fontsize=25, labelpad=15)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.dist = 10
        
        offset = 0.05 
        ax.scatter(mapped_mask_points[:, 0], 
                mapped_mask_points[:, 1], 
                mapped_mask_points[:, 2],
                c=colors_S, s=3, alpha=0.6, edgecolors='none')
        
        min_idx_val = np.argmin(S_values)
        max_idx_val = np.argmax(S_values)
        
        min_pos = mapped_mask_points[min_idx_val]
        max_pos = mapped_mask_points[max_idx_val]
        min_val_s = S_values[min_idx_val]
        max_val_s = S_values[max_idx_val]

        range_x = domain_max[0] - domain_min[0]
        range_y = domain_max[1] - domain_min[1]
        range_z = domain_max[2] - domain_min[2]
        
        offset_vec_max = np.array([-range_x*0.1, -range_y*0.4, range_z*0.1])
        offset_vec_min = np.array([range_x*0.1, range_y*0.4, -range_z*0.1])

        for i in range(len(center)):
            center_point = center[i]
            ax.scatter(center_point[0], center_point[1], center_point[2], color='black', s=100, marker='x', edgecolors='black', label='Center Point', zorder=200)
            ax.text(center_point[0]+0.4, center_point[1]+0.4, center_point[2]-0.1, 
                    f"${{\\boldsymbol{{c}}_{{{i+1}}}}}$:({center_point[0]:.2f}, {center_point[1]:.2f}, {center_point[2]:.2f})", 
                    color='black', fontsize=16, fontweight='bold', ha='left', va='center', zorder=200)
            ax.plot([center_point[0]+0.05, center_point[0]], 
                    [center_point[1]+0.05, center_point[1]], 
                    [center_point[2]-0.05, center_point[2]],
                    color='black', linestyle='-', linewidth=2, zorder=200)
        
        
                

        cmap = plt.get_cmap("rainbow")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.50, pad=0.01)
        cbar.ax.tick_params(labelsize=  20)

        ax.grid(True, linestyle='--', alpha=0.3)

    
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()
    return  mapped_mask_points,min_x_point,max_x_point,min_y_point,max_y_point,min_z_point,max_z_point
