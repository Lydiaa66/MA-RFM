import dis
from unittest import result
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib import rcParams
import os 
import matplotlib as mpl
from sympy import Q
import cv2
mpl.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_erosion

def detect_boundary(points):
    mask=points
    padding = 0.1
    x_min, x_max = mask[:, 0].min()-padding, mask[:, 0].max()+padding
    y_min, y_max = mask[:, 1].min()-padding, mask[:, 1].max()+padding

    img_size = 1000
    scale = img_size / max(x_max - x_min, y_max - y_min)
    points_pixel = ((mask - [x_min, y_min]) * scale).astype(int)

    binary_img = np.zeros((img_size, img_size), dtype=np.uint8)
    for x, y in points_pixel:
        cv2.rectangle(binary_img, (x, y), (x, y), 255, -1)

    kernel = np.ones((4, 4), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated_img = cv2.dilate(closed_img, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    largest_contour = max(contours, key=cv2.contourArea)

    contour_points = largest_contour.squeeze() / scale + [x_min, y_min]

    return contour_points

def dist_to_axis_aligned_rect(points, center, width, height):
    p_local = np.abs(points - center)
    
    a = width / 2
    b = height / 2
    
    if a < 1e-6: a = 1e-6
    if b < 1e-6: b = 1e-6
    
    metric = np.maximum(p_local[:, 0] / a, p_local[:, 1] / b)
    
    residual = np.mean(np.abs(metric - 1))
    
    return residual

def dist_to_axis_aligned_ellipsoid(points, center, width, height):
    p_local = points - center
    
    a = width / 2
    b = height / 2
    
    if a < 1e-6: a = 1e-6
    if b < 1e-6: b = 1e-6
    
    metric = np.sqrt((p_local[:, 0] / a)**2 + (p_local[:, 1] / b)**2)
    
    residual = np.mean(np.abs(metric - 1))
    
    return residual
def compute_boundary_curvature(points):
    """计算边界点的曲率统计，用于区分圆/矩形/复杂形状"""
    if len(points) < 4:
        return {"std": 0.0, "mean": 0.0, "peak_ratio": 0.0}

    pts = np.vstack([points, points[0]])

    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1, keepdims=True)
    lengths[lengths < 1e-6] = 1e-6
    tangents = diffs / lengths

    d_tangents = np.diff(tangents, axis=0)
    curvature = np.linalg.norm(d_tangents, axis=1)

    mean_k = np.mean(curvature)
    std_k = np.std(curvature)
    peak_ratio = np.max(curvature) / (mean_k + 1e-8)

    return {"std": std_k, "mean": mean_k, "peak_ratio": peak_ratio}

    
    
    

    
    
    

def detect_shape(grad, X, Y, tau_x_min, tau_x_max, tau_y_min, tau_y_max,para_abs ,para_grad, eps,mini_samples,
                 S_num=None,
                 output_directory=".", output_filename="plot.png"):
    

    shape = grad.shape[0]
    Tg = np.max(grad) / para_grad
    Ts = np.max(np.abs(S_num)) / para_abs

    mask = (grad > Tg) & (np.abs(S_num) > Ts)
    idx = np.where(mask)
    mask1 = idx[0]
    mask2 = idx[1]
    
    boolean = (mask1 >= 10) & (mask1 <= shape - 10) & (mask2 > 10) & (mask2 <= shape - 10)
    
    mask = (mask1[boolean], mask2[boolean])

    x_phys_selected = X[mask[0], mask[1]] 
    y_phys_selected = Y[mask[0], mask[1]]
    
    if S_num is not None:
        s_vals_selected = S_num[mask[0], mask[1]]
    else:
        s_vals_selected = np.zeros_like(x_phys_selected)

    g_vals_selected = grad[mask[0], mask[1]]

    Q_combined = np.column_stack((x_phys_selected, y_phys_selected, s_vals_selected, g_vals_selected))
    idx_rows = mask[0]
    idx_cols = mask[1]
    
    if len(Q_combined) == 0:
        print("No points detected.")
        return []

    segmented_results = []
    
    points_for_clustering = Q_combined[:, :2]

    if len(Q_combined) < 10:
        segmented_results.append({
            'points': points_for_clustering,
            'S_vals': Q_combined[:, 2],
            'g_vals': Q_combined[:, 3],
            'indices': np.column_stack((idx_rows, idx_cols))
        })
    else:
        clusterer = DBSCAN(eps=eps, min_samples=mini_samples).fit(points_for_clustering)
        labels = clusterer.labels_
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)
        
        for k in unique_labels:
            mask_k = (labels == k)
            subset_data = Q_combined[mask_k]
            subset_indices = np.column_stack((idx_rows, idx_cols))[mask_k]
            
            cluster_data = {
                'points': subset_data[:, :2],
                'S_vals': subset_data[:, 2],
                'g_vals': subset_data[:, 3],
                'indices': subset_indices
            }
            segmented_results.append(cluster_data)

    print("Number of detected clusters:", len(segmented_results))

    plt.figure(figsize=(8, 6))
    results = []
    
    all_s_vals = np.concatenate([cluster_info['S_vals'] for cluster_info in segmented_results])
    s_min, s_max = np.min(all_s_vals), np.max(all_s_vals)
    
    for i, cluster_info in enumerate(segmented_results):
        cluster_pts = cluster_info['points']
        s_vals = cluster_info['S_vals']
        g_vals = cluster_info['g_vals']
        cluster_indices = cluster_info.get('indices', None)
        center = np.mean(cluster_pts, axis=0)
        
        x_coords = cluster_pts[:, 0]
        y_coords = cluster_pts[:, 1]
        
        Lx = np.percentile(x_coords, 99) - np.percentile(x_coords, 1)
        Ly = np.percentile(y_coords, 99) - np.percentile(y_coords, 1)
        
        L_max=max(Lx,Ly)
        L_min=min(Lx,Ly)    
        
        aspect_ratio = L_max / L_min if L_min > 0 else 1.0
        

        dists = np.linalg.norm(cluster_pts - center, axis=1)
        r_mean = np.mean(dists)


        boundary_points=detect_boundary(cluster_pts)
        res_rect = dist_to_axis_aligned_rect(boundary_points, center, Lx, Ly)
        res_ellip = dist_to_axis_aligned_ellipsoid(boundary_points, center, Lx, Ly)
        
        shape_type = "Unknown"
        final_params = {}

        dists = np.linalg.norm(cluster_pts - center, axis=1)
        r_mean = np.mean(dists)
        min_dist_to_center = np.min(dists)
        is_donut = False
        if aspect_ratio < 1.2 and min_dist_to_center > 0.3 * r_mean and min_dist_to_center < 0.6 * r_mean:
             is_donut = True        
        if is_donut:
            shape_type = "Donut"
            r_tube = (np.max(dists) - np.min(dists)) / 2
            R_major = r_mean - r_tube
            final_params = {'center': center, 'R_major': R_major, 'r_minor': r_tube}
            
        else:
            if res_rect < res_ellip:
                shape_type = "Rectangle"
                final_params = {'center': center, 'W': Lx, 'H': Ly}
            else:
                if aspect_ratio <= 1.15:
                    shape_type = "Circle"
                    radius = np.percentile(dists, 95) 

                    final_params = {'center': center, 'radius': radius}
                else:
                    shape_type = "Ellipsoid"
                    final_params = {'center': center, 'a': Lx/2, 'b': Ly/2}
        curvature_stats = compute_boundary_curvature(boundary_points)

        is_complex_shape = (
            curvature_stats["std"] > 0.3
            and curvature_stats["peak_ratio"] < 8.0
        )

        if is_complex_shape and abs(aspect_ratio - 1) > 0.2:
            shape_type = "general"
            print(
                f"CurvStd={curvature_stats['std']:.3f}, "
                f"PeakRatio={curvature_stats['peak_ratio']:.2f} -> general"
            )

        print(f"Cluster {i}: AR={aspect_ratio:.2f}")
        print(f"  Res_Rect={res_rect:.3f} vs Res_Ellip={res_ellip:.3f}")
        print(f"  -> Winner: {shape_type}")

        basis_func_type = "Sigmoid"
        if S_num is not None:
            Std_S_num=np.std(s_vals)
            Mean_S_num=np.mean(np.abs(s_vals))
            CV=Std_S_num/Mean_S_num
            if CV>=0.6:
                basis_func_type="Exp"
            else:
                basis_func_type="Sigmoid"


        print(f"Cluster {i}: Lx={Lx:.3f}, Ly={Ly:.3f}, AR={aspect_ratio:.2f}")
        print(f"  -> Decision: {shape_type} ({basis_func_type})")
        
        results.append({
            'cluster_id': i,
            'shape': shape_type,
            'params': final_params,
            'profile': basis_func_type
        })

        sc = plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], c=s_vals, cmap="jet", 
                    marker='o', s=5, edgecolor='white', linewidth=0.5, vmin=s_min, vmax=s_max)

        if shape_type == "Circle":
            circle = plt.Circle(center, final_params['radius'], color='r', fill=False, linestyle='--',linewidth=1.5)
            plt.gca().add_patch(circle)
        elif shape_type == "Rectangle":
            rect = plt.Rectangle((center[0] - Lx/2, center[1] - Ly/2), Lx, Ly, 
                                 linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            plt.gca().add_patch(rect)
        elif shape_type == "Ellipsoid":
            ellipse = plt.matplotlib.patches.Ellipse(center, 2*final_params['a'], 2*final_params['b'], 
                                                     angle=0, edgecolor='m', facecolor='none', linestyle='--')
            plt.gca().add_patch(ellipse)
        #用×画出中心点,加粗
        plt.scatter(center[0], center[1], c='k', marker='x', s=100, linewidth=2)

        
    cbar = plt.colorbar(sc, shrink=1, aspect=10)
    cbar.ax.tick_params(labelsize=18)

    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    n_ticks = 5
    
    x_ticks_loc = np.linspace(tau_x_min, tau_x_max, n_ticks)
    x_tick_labels = np.round(x_ticks_loc, decimals=3)
    
    plt.xticks(ticks=x_ticks_loc, labels=x_tick_labels, rotation=0, fontsize=18)
    
    y_ticks_loc = np.linspace(tau_y_min, tau_y_max, n_ticks)
    y_tick_labels = np.round(y_ticks_loc, decimals=3)
    
    plt.yticks(ticks=y_ticks_loc, labels=y_tick_labels, fontsize=18)
    
    ax = plt.gca()
    ax.set_xlim(tau_x_min, tau_x_max)
    ax.set_ylim(tau_y_min, tau_y_max)
    ax.set_aspect('equal', adjustable='box')
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
