
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_erosion, binary_closing, binary_dilation
import os
import matplotlib as mpl

mpl.rcdefaults()

def detect_boundary_3d(points):
    """
    3D Voxel-based boundary detection.
    Returns points on the surface of the shape.
    """
    if len(points) < 4:
        return points
    
    # 1. Determine bounds and padding
    padding = 0.1
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding
    z_min, z_max = points[:, 2].min() - padding, points[:, 2].max() + padding
    
    # 2. Map coordinates to voxel positions
    # Resolution (img_size in 2D was 1000, for 3D volume 1000^3 is too big. Use 100-200)
    img_size = 100 
    max_span = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_span == 0: return points
    scale = img_size / max_span
    
    points_pixel = ((points - [x_min, y_min, z_min]) * scale).astype(int)
    
    # 3. Create binary voxel grid
    # Determine grid dimensions
    dx = int((x_max - x_min) * scale) + 5
    dy = int((y_max - y_min) * scale) + 5
    dz = int((z_max - z_min) * scale) + 5
    
    grid = np.zeros((dx, dy, dz), dtype=np.uint8)

    # Clip and fill
    points_pixel[:, 0] = np.clip(points_pixel[:, 0], 0, dx - 1)
    points_pixel[:, 1] = np.clip(points_pixel[:, 1], 0, dy - 1)
    points_pixel[:, 2] = np.clip(points_pixel[:, 2], 0, dz - 1)
    
    grid[points_pixel[:, 0], points_pixel[:, 1], points_pixel[:, 2]] = 1
    
    # 4. Morphological processing to close gaps and find surface
    struct = np.ones((3, 3, 3), dtype=bool)
    
    # Close gaps (dilation then erosion)
    closed_grid = binary_closing(grid, structure=struct, iterations=2)
    # Dilate slightly to ensure continuity
    dilated_grid = binary_dilation(closed_grid, structure=struct, iterations=1)
    
    # 5. Extract Surface (Boundary)
    # Surface = Solid - Eroded(Solid)
    eroded_grid = binary_erosion(dilated_grid, structure=struct, iterations=1)
    boundary_grid = dilated_grid ^ eroded_grid
    
    # Get coordinates of boundary voxels
    boundary_indices = np.argwhere(boundary_grid)
    
    if len(boundary_indices) == 0:
        # Fallback to original points if something fails
        return points

    # Convert back to physical coordinates
    contour_points = boundary_indices / scale + [x_min, y_min, z_min]
    
    return contour_points

def compute_boundary_curvature_3d(points, k=15):
    """
    计算3D边界点的局部曲率统计(基于PCA特征值)，用于区分球体/立方体/复杂形状
    
    Args:
        points: (N, 3) array of surface points
        k: nearest neighbors count for local curvature estimation
        
    Returns:
        dict with mean, std, peak_ratio of curvature estimates
    """
    if len(points) < k + 1:
        return {"std": 0.0, "mean": 0.0, "peak_ratio": 0.0}

    # Use NearestNeighbors to find local patches
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    curvatures = []
    
    for i in range(len(points)):
        # Get neighbors
        neighbor_points = points[indices[i]]
        # Center them
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        # Covariance matrix
        cov = np.dot(centered.T, centered) / (k - 1)
        # Eigenvalues
        eigenvalues, _ = np.linalg.eig(cov)
        # Sort eigenvalues small to large
        eigenvalues = np.sort(np.abs(eigenvalues))
        
        # Surface variation as curvature proxy: lambda_min / sum(lambdas)
        # lambda_0 corresponds to the normal direction variance (thickness of the patch)
        # For a flat surface, lambda_0 -> 0.
        denom = np.sum(eigenvalues)
        if denom == 0:
            c = 0
        else:
            c = eigenvalues[0] / denom
        curvatures.append(c)
        
    curvatures = np.array(curvatures)
    
    mean_k = np.mean(curvatures)
    std_k = np.std(curvatures)
    peak_ratio = np.max(curvatures) / (mean_k + 1e-8)
    
    return {"std": std_k, "mean": mean_k, "peak_ratio": peak_ratio}

def dist_to_axis_aligned_box(points, center, width, height, depth):
    # points: (N, 3)
    p_local = np.abs(points - center)
    a, b, c = width / 2, height / 2, depth / 2
    
    if a < 1e-6: a = 1e-6
    if b < 1e-6: b = 1e-6
    if c < 1e-6: c = 1e-6
    
    # Metric represents rescaling to boundary (1.0 at boundary)
    metric = np.maximum(np.maximum(p_local[:, 0] / a, p_local[:, 1] / b), p_local[:, 2] / c)
    
    residual = np.mean(np.abs(metric - 1))
    return residual

def dist_to_axis_aligned_ellipsoid_3d(points, center, width, height, depth):
    # points: (N, 3)
    p_local = points - center
    a, b, c = width / 2, height / 2, depth / 2
    
    if a < 1e-6: a = 1e-6
    if b < 1e-6: b = 1e-6
    if c < 1e-6: c = 1e-6
    
    metric = np.sqrt((p_local[:, 0] / a)**2 + (p_local[:, 1] / b)**2 + (p_local[:, 2] / c)**2)
    
    residual = np.mean(np.abs(metric - 1))
    return residual

def detect_shape_3d(grad, X, Y, Z, 
                    tau_x_min, tau_x_max, tau_y_min, tau_y_max, tau_z_min, tau_z_max,
                    para_abs, para_grad, eps, mini_samples,
                    S_num=None, elev=None, azim=None, zlim=None,
                    output_directory=".", output_filename="plot3d.png"):
    
    # 1. Filtering
    Tg = np.max(grad) / para_grad
    if S_num is not None:
        Ts = np.max(np.abs(S_num)) / para_abs
        mask = (grad > Tg) & (np.abs(S_num) > Ts)
    else:
        mask = (grad > Tg)
        
    idx = np.where(mask) 
    
    # Boundary protection (remove indices near edges of array)
    shape = grad.shape
    valid_mask = np.ones(len(idx[0]), dtype=bool)
    
    margin = 5 
    for dim in range(3): # Assuming 3D
        valid_mask &= (idx[dim] >= margin) & (idx[dim] < shape[dim] - margin)
    
    # Apply margin filtering
    filtered_idx = tuple(d[valid_mask] for d in idx)
    
    if len(filtered_idx[0]) == 0:
        print("No points detected.")
        return []

    # Extract Physics Coordinates
    x_phys = X[filtered_idx]
    y_phys = Y[filtered_idx]
    z_phys = Z[filtered_idx]
    
    if S_num is not None:
        s_vals = S_num[filtered_idx]
    else:
        s_vals = np.zeros_like(x_phys)
        
    g_vals = grad[filtered_idx]
    
    # Combine (N, 5) -> x, y, z, s, g
    Q_combined = np.column_stack((x_phys, y_phys, z_phys, s_vals, g_vals))
    
    # 2. Clustering (use physical coordinates)
    points_for_clustering = Q_combined[:, :3]
    
    segmented_results = []
    
    if len(Q_combined) < 10:
         segmented_results.append({
            'points': points_for_clustering,
            'S_vals': s_vals,
            'g_vals': g_vals
        })
    else:
        clusterer = DBSCAN(eps=eps, min_samples=mini_samples).fit(points_for_clustering)
        labels = clusterer.labels_
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)
        
        for k in unique_labels:
            mask_k = (labels == k)
            subset_data = Q_combined[mask_k]
            segmented_results.append({
                'points': subset_data[:, :3],
                'S_vals': subset_data[:, 3],
                'g_vals': subset_data[:, 4]
            })
            
    print("Number of detected clusters:", len(segmented_results))

    # 3. Visualization & Analysis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.set_position([0.15, 0.03, 0.80, 0.94])
    results = []
    
    all_s_vals = np.concatenate([c['S_vals'] for c in segmented_results]) if segmented_results else []
    s_min = np.min(all_s_vals) if len(all_s_vals) > 0 else 0
    s_max = np.max(all_s_vals) if len(all_s_vals) > 0 else 1
    
    for i, cluster_info in enumerate(segmented_results):
        cluster_pts = cluster_info['points']
        s_vals = cluster_info['S_vals']
        center = np.mean(cluster_pts, axis=0)
        
        # Axis lengths (Bounding Box dimensions)
        x_c, y_c, z_c = cluster_pts[:, 0], cluster_pts[:, 1], cluster_pts[:, 2]
        # Lx = np.percentile(x_c, 99) - np.percentile(x_c, 1)
        # Ly = np.percentile(y_c, 99) - np.percentile(y_c, 1)
        # Lz = np.percentile(z_c, 99) - np.percentile(z_c, 1)
        Lx = np.max(x_c) - np.min(x_c)
        Ly = np.max(y_c) - np.min(y_c   )
        Lz = np.max(z_c) - np.min(z_c)
        L_dims = np.array([Lx, Ly, Lz])
        L_max = np.max(L_dims)
        L_min = np.min(L_dims) if np.min(L_dims) > 0 else 1e-6
        aspect_ratio = L_max / L_min
        
        # Calculate Boundary for shape fitting
        boundary_points = detect_boundary_3d(cluster_pts)
        curvature_stats = compute_boundary_curvature_3d(boundary_points)
        
        # Residuals
        plt.plot(boundary_points[:,0], boundary_points[:,1], boundary_points[:,2], 'k.', markersize=1, alpha=0.1) # For visualization
        res_rect = dist_to_axis_aligned_box(boundary_points, center, Lx, Ly, Lz)
        res_ellip = dist_to_axis_aligned_ellipsoid_3d(boundary_points, center, Lx, Ly, Lz)
        
        # Shape Decision
        shape_type = "Unknown"
        final_params = {}
        
        dists = np.linalg.norm(cluster_pts - center, axis=1)
        r_mean = np.mean(dists)
        min_dist_to_center = np.min(dists)
        max_dist_to_center = np.max(dists)


        if res_rect < res_ellip:
            shape_type = "Cuboid"
            final_params = {'center': center, 'Lx': Lx, 'Ly': Ly, 'Lz': Lz}
        else:
            # Ellipsoid or Sphere
            # Check aspect ratio of the axes
            mean_L = np.mean(L_dims)
            std_L = np.std(L_dims)
            if max_dist_to_center > 0 and min_dist_to_center < 0.3 * max_dist_to_center and min_dist_to_center > 0.15 * max_dist_to_center : 
                shape_type = "SphericalShell" # or "Torus" but we assume spherical shell for 3D usually
                final_params = {'center': center, 'R_outer': np.percentile(dists, 95), 'R_inner':np.percentile(dists, 5),'r':Lz}
            elif std_L / mean_L < 0.15: # Threshold for "Spherical"
                shape_type = "Sphere"
                radius = np.percentile(dists, 95)
                final_params = {'center': center, 'radius': radius}
            else:
                shape_type = "Ellipsoid"
                final_params = {'center': center, 'a': Lx/2, 'b': Ly/2, 'c': Lz/2}
                
        print(f"Cluster {i}: L=({Lx:.2f}, {Ly:.2f}, {Lz:.2f}), AR={aspect_ratio:.2f}")
        print(f"  Res_Box={res_rect:.3f} vs Res_Ellip={res_ellip:.3f}")
        print(f"  -> Decision: {shape_type}")
        
        # Basis Function Decision
        basis_func_type = "Sigmoid"
        if S_num is not None:
             Mean_S = np.mean(np.abs(s_vals))
             if Mean_S > 1e-9:
                cv = np.std(s_vals) / Mean_S
                print(f"  CV of S_num: {cv:.3f}")
                if cv >= 0.7: basis_func_type = "Exp"
        
        results.append({
            'cluster_id': i,
            'shape': shape_type,
            'params': final_params,
            'profile': basis_func_type
        })
        
        # Plot Scatter
        p = ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], cluster_pts[:, 2], 
                   c=s_vals, cmap="rainbow", s=3, vmin=s_min, vmax=s_max, alpha=0.6, edgecolors='none', zorder=10)
                   
        # Draw wireframes for simple shapes
        if shape_type == "Sphere":
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_s = center[0] + final_params['radius']*np.cos(u)*np.sin(v)
            y_s = center[1] + final_params['radius']*np.sin(u)*np.sin(v)
            z_s = center[2] + final_params['radius']*np.cos(v)
            ax.plot_wireframe(x_s, y_s, z_s, color="r", alpha=0.3)
            
        elif shape_type == "Cuboid":
             # Simplified box drawing
             cx, cy, cz = center
             lx, ly, lz = final_params['Lx'], final_params['Ly'], final_params['Lz']
             # ... 3D Box drawing is verbose, skipping for brevity of display unless critical
             pass
        #标记center
        ax.plot([center[0]], [center[1]], [center[2]], 'kx', markersize=8, markeredgewidth=2, zorder=20)
        if shape_type == "Circle":
            print(f"    Radius: {final_params['radius']:.3f}")
        elif shape_type == "Rectangle":
            print(f"    W: {final_params['W']:.3f}, H: {final_params['H']:.3f}")
        elif shape_type == "Ellipsoid":
            print(f"    a: {final_params['a']:.3f}, b: {final_params['b']:.3f}")
        print(f"    Rect Residual: {res_rect:.4f}, Ellip Residual: {res_ellip:.4f}")
        print(f"    Aspect Ratio: {aspect_ratio:.3f}")
        print(f"    Curvature - Std: {curvature_stats['std']:.4f}, Mean: {curvature_stats['mean']:.4f}, Peak Ratio: {curvature_stats['peak_ratio']:.2f}")
        #写上center
    # Setup Axes
    ax.set_xlim(tau_x_min, tau_x_max)
    ax.set_ylim(tau_y_min, tau_y_max)
    if zlim is not None:
         ax.set_zlim(zlim)
    else:
         ax.set_zlim(tau_z_min, tau_z_max)

    ax.set_xlabel("$x_1$", fontsize=25, labelpad=25)
    ax.set_ylabel("$x_2$", fontsize=25, labelpad=25)
    ax.set_zlabel("$x_3$", fontsize=25, labelpad=20)
    
    # 增加刻度与坐标轴的距离 (pad)
    ax.tick_params(axis='x', which='major', labelsize=18, pad=15)
    ax.tick_params(axis='y', which='major', labelsize=18, pad=10)
    ax.tick_params(axis='z', which='major', labelsize=18, pad=10)
    
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)

    # 简单处理视角布局
    if elev==0:
        ax.set_xlabel("$x_1$", fontsize=25, labelpad=30) # 特殊视角加大距离
        ax.set_ylabel("$x_2$", fontsize=25, labelpad=25)
        ax.set_zlabel("$x_3$", fontsize=25, labelpad=20)
    elif elev==35:
        # ax.set_position([0.5, 0.03, 0.80, 0.94]) # 可能会导致裁剪，先注释
        pass

    
    # Try to make aspect ratio reasonable
    try:
        # 使用 set_box_aspect 保持比例
        ax.set_box_aspect([1, 1, 1]) 
    except:
        pass 
    
    # Colorbar
    norm = plt.Normalize(vmin=s_min, vmax=s_max)
    cmap = plt.get_cmap("rainbow")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # 调整 colorbar 参数以匹配 bound_detect1 样式
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.55, pad=0.001)
    cbar.ax.tick_params(labelsize=20)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 减少留白
    # plt.tight_layout() # tight_layout in 3D is tricky, use pad_inches in savefig instead
    
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_filename = output_filename if output_filename else "plot3d.png"
    # savefig 时增加 pad_inches 减少白边，防止特定视角(如elev=0)标签被切掉
    plt.savefig(os.path.join(output_directory, output_filename), dpi=300, bbox_inches='tight', pad_inches=0.6)
    plt.show()
    plt.close() # Close figure to free memory
    
    return results
