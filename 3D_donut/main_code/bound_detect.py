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

def detect_3d(grad, S_num, domain_min, domain_max, para_grad, para_abs, elev,azim,choose,
              output_directory=".", output_filename="plot_3d.png",temp=None):
    
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

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.set_position([0.01, 0.03, 0.96, 0.94])
        plt.subplots_adjust(
            left=0.01,
            right=0.98,
            bottom=0.02,
            top=0.98
        )

        ax.set_xlim([domain_min[0], domain_max[0]])
        ax.set_ylim([domain_min[1], domain_max[1]])
        ax.set_zlim([domain_min[2], domain_max[2]])
        ax.set_xlabel("X", fontsize=14, labelpad=20)
        ax.set_ylabel("Y", fontsize=14, labelpad=20)
        ax.set_zlabel("Z", fontsize=14, labelpad=20)
        
        ax.view_init(elev=elev, azim=azim)
        offset = 0.05 
        ax.scatter(mapped_mask_points[:, 0], 
                mapped_mask_points[:, 1], 
                mapped_mask_points[:, 2],
                c=colors_grad, s=3, alpha=0.6, edgecolors='none', zorder=10)

        cmap = plt.get_cmap("rainbow")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=-0.1)
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

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.set_position([0.15, 0.03, 0.80, 0.94])
        ax.scatter(mapped_mask_points[:, 0], 
                mapped_mask_points[:, 1], 
                mapped_mask_points[:, 2],
                c=colors_S, s=3, alpha=0.6, edgecolors='none', zorder=10)


        ax.set_xlim([domain_min[0], domain_max[0]])
        ax.set_ylim([domain_min[1], domain_max[1]])
        ax.set_zlim([domain_min[2], domain_max[2]])
        ax.set_xlabel("$x_1$", fontsize=25, labelpad=22)
        ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
        ax.set_zlabel("$x_3$", fontsize=25, labelpad=20)
        ax.tick_params(axis='x', labelsize=18, pad=12)
        ax.tick_params(axis='y', labelsize=18, pad=12)
        ax.tick_params(axis='z', labelsize=18, pad=12)
        ax.view_init(elev=elev, azim=azim)
        offset = 0.05 
        if elev==0:
            ax.set_xlabel("$x_1$", fontsize=25, labelpad=27)
            ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
            ax.set_zlabel("$x_3$", fontsize=25, labelpad=20)
            ax.tick_params(axis='x', labelsize=18, pad=8)
            ax.tick_params(axis='y', labelsize=18, pad=8)
            ax.tick_params(axis='z', labelsize=18, pad=8)
            if temp==True:
                ax.scatter(*min_y_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(min_y_point[0]+offset, min_y_point[1]-3*offset, min_y_point[2], 
                        f"{min_y_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_y_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(max_y_point[0]+offset, max_y_point[1]-3*offset, max_y_point[2], 
                        f"{max_y_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_z_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(max_z_point[0]+offset, max_z_point[1]-3*offset, max_z_point[2]+0.5*offset, 
                        f"{max_z_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*min_z_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(min_z_point[0]+offset, min_z_point[1]-3*offset, min_z_point[2]-offset, 
                        f"{min_z_point}", color='blue', fontsize=18, zorder=12)
                
        elif elev==35:
            
            ax.set_position([0.5, 0.03, 0.80, 0.94])
            ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
            ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
            ax.tick_params(axis='x', labelsize=18, pad=8)
            ax.tick_params(axis='y', labelsize=18, pad=6)
            ax.tick_params(axis='z', labelsize=18, pad=8)
            x_label_pos = domain_max[0]+0.02
            y_label_pos = domain_min[1]-0.04
            z_label_pos = domain_max[2]+0.05
            ax.text(x_label_pos, y_label_pos, z_label_pos, "$x_3$", fontsize=25, fontweight='bold', color='black')
            if temp==True:

                ax.scatter(*min_x_point, c='blue', s=50, edgecolors='black', zorder=13)
                ax.text(min_x_point[0]+offset, min_x_point[1]-3*offset, min_x_point[2]-0.15, 
                        f"{min_x_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_x_point, c='blue', s=50, edgecolors='black', zorder=13)
                ax.text(max_x_point[0]+offset, max_x_point[1]-3*offset, max_x_point[2], 
                        f"{max_x_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*min_y_point, c='blue', s=50, edgecolors='black', zorder=13)
                ax.text(min_y_point[0]+offset, min_y_point[1]-3*offset, min_y_point[2], 
                        f"{min_y_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_y_point, c='blue', s=50, edgecolors='black', zorder=13)
                ax.text(max_y_point[0]+offset, max_y_point[1]-3*offset, max_y_point[2], 
                        f"{max_y_point}", color='blue', fontsize=18, zorder=12)
        else:
            if temp==True:
                ax.scatter(*min_x_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(min_x_point[0]+offset, min_x_point[1]-3*offset, 0.3, 
                        f"{min_x_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_x_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(max_x_point[0]+offset, max_x_point[1]-3*offset, max_x_point[2], 
                        f"{max_x_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*min_y_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(min_y_point[0]+offset, min_y_point[1]-3*offset, min_y_point[2], 
                        f"{min_y_point}", color='blue', fontsize=18, zorder=12)
                ax.scatter(*max_y_point, c='blue', s=50, edgecolors='black', zorder=11)
                ax.text(max_y_point[0]+offset, max_y_point[1]-3*offset, max_y_point[2], 
                        f"{max_y_point}", color='blue', fontsize=18, zorder=12)

        cmap = plt.get_cmap("rainbow")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=18)

        ax.grid(True, linestyle='--', alpha=0.3)    
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        plt.show()
    return  mapped_mask_points,min_x_point,max_x_point,min_y_point,max_y_point,min_z_point,max_z_point
