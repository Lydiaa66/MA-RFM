import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN

import cv2


def _resolve_field_array(field, name): 
    if field is None:
        raise ValueError(f"{name} is None")

    if isinstance(field, np.ndarray):
        if field.dtype == object:
            if field.ndim == 0:
                return _resolve_field_array(field.item(), name)
            for item in reversed(field.tolist()):
                try:
                    return _resolve_field_array(item, name)
                except Exception:
                    continue
            raise ValueError(f"{name} object array does not contain a valid 2D field")
        if field.ndim >= 2:
            return field
        raise ValueError(f"{name} must be a 2D array, got shape {field.shape}")

    if isinstance(field, (list, tuple)):
        if not field:
            raise ValueError(f"{name} is empty")
        for item in reversed(field):
            try:
                return _resolve_field_array(item, name)
            except Exception:
                continue
        raise ValueError(f"{name} does not contain a valid 2D field")

    if hasattr(field, "detach") and hasattr(field, "cpu"):
        return _resolve_field_array(field.detach().cpu().numpy(), name)

    return _resolve_field_array(np.asarray(field), name)


def map_index_to_domain(points, tau_x_min, tau_x_max, tau_y_min, tau_y_max, shape): # Map grid indices to the physical domain.
    """Map grid indices to the physical 2D domain."""
    x_scale = (tau_x_max - tau_x_min) / shape[1]
    y_scale = (tau_y_max - tau_y_min) / shape[0]

    mapped_points = np.zeros_like(points, dtype=float)
    mapped_points[:, 0] = tau_x_min + points[:, 0] * x_scale
    mapped_points[:, 1] = tau_y_min + points[:, 1] * y_scale
    return mapped_points


def _estimate_grid_spacing(X, Y): # Estimate the grid spacing.
    x_unique = np.unique(np.asarray(X, dtype=float).ravel())
    y_unique = np.unique(np.asarray(Y, dtype=float).ravel())
    dx_candidates = np.diff(x_unique)
    dy_candidates = np.diff(y_unique)
    dx_candidates = dx_candidates[dx_candidates > 1e-12]
    dy_candidates = dy_candidates[dy_candidates > 1e-12]
    dx = float(np.min(dx_candidates)) if dx_candidates.size > 0 else 0.0
    dy = float(np.min(dy_candidates)) if dy_candidates.size > 0 else 0.0
    return max(dx, dy)


def _resolve_dbscan_eps(eps, X, Y, eps_mode): # Resolve the DBSCAN eps parameter according to the selected mode.
    if eps_mode not in {"auto", "absolute", "grid_scaled"}:
        raise ValueError(f"Unsupported eps_mode: {eps_mode}")

    h = _estimate_grid_spacing(X, Y)
    if eps_mode == "absolute":
        return float(eps), h, "absolute"
    if eps_mode == "grid_scaled":
        return float(eps) * h, h, "grid_scaled"

    # Backward compatible auto mode:
    # eps <= 1 keeps the historical absolute-distance interpretation;
    # eps > 1 is treated as a multiplier of the test-grid spacing h.
    if float(eps) > 1.0:
        return float(eps) * h, h, "grid_scaled"
    return float(eps), h, "absolute"


def get_contour_points_from_implicit(implicit_func, func_args, x_range, y_range, grid_points=300): # Generate contour points from an implicit function.
    x_grid = np.linspace(x_range[0], x_range[1], grid_points)
    y_grid = np.linspace(y_range[0], y_range[1], grid_points)
    X_contour, Y_contour = np.meshgrid(x_grid, y_grid)
    Z_contour = implicit_func(X_contour, Y_contour, **func_args)
    cs = plt.contour(X_contour, Y_contour, Z_contour, levels=[0])
    if not cs.allsegs[0]:
        plt.close()
        raise ValueError("Unable to find level=0 contour.")
    longest_contour_idx = np.argmax([len(seg) for seg in cs.allsegs[0]])
    contour_points = cs.allsegs[0][longest_contour_idx]
    plt.close()
    return contour_points


def generate_soft_boundary(x_grid, y_grid, contour_points, K=500): # Generate a smooth soft boundary using a Sigmoid map.
    boundary_tree = cKDTree(contour_points) # Build a KD tree for nearest-distance queries.
    path = mpath.Path(contour_points) # Path object.
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    distances, _ = boundary_tree.query(grid_points, k=1) # Query nearest distances.
    are_inside = path.contains_points(grid_points) # Test whether points lie inside the contour.
    signed_distances = np.where(are_inside, -distances, distances) # Negative inside, positive outside.
    z_flat = 1 / (1 + np.exp(K * signed_distances)) # Generate smooth boundary values with a Sigmoid map.
    return z_flat.reshape(x_grid.shape)


def resample_closed_curve(points, n_points=1200):
    """Resample a closed boundary by arc length to avoid spline artifacts from dense cv2 pixel contours."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    # Remove consecutive duplicate points because splprep is sensitive to duplicates.
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    points = points[keep]
    if len(points) < 4:
        return points

    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points = np.vstack([points, points[0]])

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid = seg > 1e-12
    points = np.vstack([points[:-1][valid], points[0]])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] < 1e-12:
        return points[:-1]

    n_points = int(min(max(32, n_points), max(32, len(points) - 1)))
    target = np.linspace(0.0, arc[-1], n_points, endpoint=False)
    x_new = np.interp(target, arc, points[:, 0])
    y_new = np.interp(target, arc, points[:, 1])
    return np.column_stack((x_new, y_new))


def simple_offset_batch(points, s, distances, n_resample=1200): # Generate smooth offset point sets from a curve.
#    '''
#    points: original curve points with shape (N, 2)
#    s: spline smoothing parameter controlling curve smoothness
#    distances: offset distances, either a scalar or multiple values
#    '''
    points = resample_closed_curve(points, n_points=n_resample)
    # The splprep parameter s is a global residual bound; with more contour points,
    # a fixed s can overfit pixel-level jaggedness.
    # Interpret the original s as per-point smoothing strength to keep notebook parameter scales unchanged.
    smoothing = float(s) * len(points)
    tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothing, per=True) # Build a closed spline curve.
    u_new = np.linspace(0, 1, len(points), endpoint=False) # Uniform parameters.

    der1 = np.array(splev(u_new, tck, der=1)).T # Compute tangent vectors.
    tangent_norm = np.linalg.norm(der1, axis=1)
    tangent_norm[tangent_norm < 1e-12] = 1e-12 # Avoid division by zero.
    unit_tangent = der1 / tangent_norm[:, np.newaxis] # Unit tangent vectors.
    unit_normal = np.c_[-unit_tangent[:, 1], unit_tangent[:, 0]]  # Unit normal vectors.
    smooth_points = np.array(splev(u_new, tck)).T # Smooth coordinates at the sampled spline parameters.

    distances = np.atleast_1d(distances)
    if len(distances) == 1:
        return smooth_points + distances[0] * unit_normal
    return smooth_points[np.newaxis, :, :] + distances[:, np.newaxis, np.newaxis] * unit_normal[np.newaxis, :, :]


def generate_signed_distance(x_train, boundary_points, alpha, scale_factor, M): # Generate the signed-distance indicator.
    if hasattr(x_train, "detach") and hasattr(x_train, "cpu"):
        grid_points = x_train.detach().cpu().numpy()
    else:
        grid_points = np.asarray(x_train)

    bound_num_scaled = simple_offset_batch(boundary_points, alpha, scale_factor) # Generate offset curves.
    signed_distances = np.zeros((grid_points.shape[0], M), dtype=np.float64)
    for i in range(M):
        path = mpath.Path(bound_num_scaled[i, :, :]) # Build a path object.
        are_inside = path.contains_points(grid_points, radius=-1e-9) # Test whether points lie inside the contour.
        signed_distances[:, i : i + 1] = np.where(are_inside, 1, -1).reshape(-1, 1) # Inside = 1, outside = -1.
    return bound_num_scaled, signed_distances


def detect_boundary(points):
    padding = 0.1  # Boundary padding.
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding

    img_size = 1000 # Binary image size.
    scale = img_size / max(x_max - x_min, y_max - y_min) # Scaling factor.
    points_pixel = ((points - [x_min, y_min]) * scale).astype(int) # Map to pixel coordinates.

    binary_img = np.zeros((img_size, img_size), dtype=np.uint8)  # Initialize an empty binary image.
    for x, y in points_pixel:
        cv2.rectangle(binary_img, (x, y), (x, y), 255, -1) # Draw each point on the image.

    kernel = np.ones((4, 4), np.uint8) # Kernel for morphological processing.
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2) # Closing removes small holes.
    dilated_img = cv2.dilate(closed_img, kernel, iterations=1) # Dilation connects the contour.

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Extract the outer contours.
    largest_contour = max(contours, key=cv2.contourArea) # Select the contour with the largest area.
    return largest_contour.squeeze() / scale + [x_min, y_min] # Map back to physical coordinates.


def dist_to_axis_aligned_rect(points, center, width, height): # Compute the rectangle fitting residual.
    p_local = np.abs(points - center)
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    metric = np.maximum(p_local[:, 0] / a, p_local[:, 1] / b)
    return np.mean(np.abs(metric - 1))


def dist_to_axis_aligned_ellipsoid(points, center, width, height): # Compute the ellipse fitting residual.
    p_local = points - center
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    metric = np.sqrt((p_local[:, 0] / a) ** 2 + (p_local[:, 1] / b) ** 2)
    return np.mean(np.abs(metric - 1))


def compute_boundary_curvature(points): # Estimate boundary curvature.
    if len(points) < 4:
        return {"std": 0.0, "mean": 0.0, "peak_ratio": 0.0}

    pts = np.vstack([points, points[0]])  # Close the curve by connecting the last point to the first.
    diffs = np.diff(pts, axis=0) # Difference adjacent points to approximate tangents.
    lengths = np.linalg.norm(diffs, axis=1, keepdims=True) # Compute the length of each tangent segment.
    lengths[lengths < 1e-6] = 1e-6 
    tangents = diffs / lengths # Unit tangent vectors.

    d_tangents = np.diff(tangents, axis=0) # Tangent variation used to approximate curvature.
    curvature = np.linalg.norm(d_tangents, axis=1) # Curvature magnitude.
 
    mean_k = np.mean(curvature) # Mean curvature.
    std_k = np.std(curvature) # Curvature standard deviation.
    peak_ratio = np.max(curvature) / (mean_k + 1e-8) # Peak-to-mean curvature ratio.
    return {"std": std_k, "mean": mean_k, "peak_ratio": peak_ratio} 


def _cluster_points(grad, X, Y, para_abs, para_grad, eps, mini_samples, S_num, eps_mode="auto"):
    S_num = _resolve_field_array(S_num, "S_num")
    try:
        grad = _resolve_field_array(grad, "grad")
    except ValueError as exc:
        if "empty" not in str(exc):
            raise
        grad_y, grad_x = np.gradient(S_num)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        print("grad is empty; using numerical gradient from S_num instead. Classification may differ from true g_S.")
    shape = grad.shape[0]
    grad_threshold = np.max(grad) / para_grad
    abs_threshold = np.max(np.abs(S_num)) / para_abs

    mask = (grad > grad_threshold) & (np.abs(S_num) > abs_threshold) 
    idx = np.where(mask)
    mask1 = idx[0]
    mask2 = idx[1]
    boolean = (mask1 >= 10) & (mask1 <= shape - 10) & (mask2 > 10) & (mask2 <= shape - 10)
    mask = (mask1[boolean], mask2[boolean])

    x_phys_selected = X[mask[0], mask[1]]
    y_phys_selected = Y[mask[0], mask[1]]
    s_vals_selected = S_num[mask[0], mask[1]] if S_num is not None else np.zeros_like(x_phys_selected)
    g_vals_selected = grad[mask[0], mask[1]]

    combined = np.column_stack((x_phys_selected, y_phys_selected, s_vals_selected, g_vals_selected))
    idx_rows = mask[0]
    idx_cols = mask[1]

    if len(combined) == 0:
        print("No points detected.")
        return []

    if len(combined) < 10:
        return [
            {
                "points": combined[:, :2],
                "S_vals": combined[:, 2],
                "g_vals": combined[:, 3],
                "indices": np.column_stack((idx_rows, idx_cols)),
            }
        ]

    eps_dbscan, h, resolved_mode = _resolve_dbscan_eps(eps, X, Y, eps_mode)
    if resolved_mode == "grid_scaled":
        print(f"DBSCAN eps = {eps_dbscan:.6f} ({float(eps):.6f} * h, h = {h:.6f})")
    clusterer = DBSCAN(eps=eps_dbscan, min_samples=mini_samples).fit(combined[:, :2])
    labels = clusterer.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)

    segmented_results = []
    for label in unique_labels:
        mask_k = labels == label
        subset_data = combined[mask_k]
        subset_indices = np.column_stack((idx_rows, idx_cols))[mask_k]
        segmented_results.append(
            {
                "points": subset_data[:, :2],
                "S_vals": subset_data[:, 2],
                "g_vals": subset_data[:, 3],
                "indices": subset_indices,
            }
        )
    return segmented_results


def _filter_clusters(segmented_results, min_cluster_size=None, min_relative_abs_s=0.0, min_relative_grad=0.0, keep_top_k=None):
    if not segmented_results:
        return []

    metrics = []
    for idx, cluster in enumerate(segmented_results):
        points = cluster["points"]
        s_vals = cluster["S_vals"]
        g_vals = cluster["g_vals"]
        size = len(points)
        mean_abs_s = float(np.mean(np.abs(s_vals))) if size > 0 else 0.0
        mean_grad = float(np.mean(g_vals)) if size > 0 else 0.0
        score = float(size * mean_abs_s * mean_grad) # Compute a combined score for each cluster.

        metrics.append(
            {
                "index": idx,
                "size": size,
                "mean_abs_s": mean_abs_s,
                "mean_grad": mean_grad,
                "score": score,
            }
        )

    max_mean_abs_s = max((item["mean_abs_s"] for item in metrics), default=0.0) # Maximum mean_abs_s for relative thresholding.
    max_mean_grad = max((item["mean_grad"] for item in metrics), default=0.0)

    kept = []
    for item in metrics: # Filter retained clusters.
        if min_cluster_size is not None and item["size"] < min_cluster_size:
            continue
        if max_mean_abs_s > 0 and item["mean_abs_s"] < min_relative_abs_s * max_mean_abs_s:
            continue
        if max_mean_grad > 0 and item["mean_grad"] < min_relative_grad * max_mean_grad:
            continue
        kept.append(item)

    if keep_top_k is not None and len(kept) > keep_top_k: # Keep the top-scoring keep_top_k clusters.
        kept = sorted(kept, key=lambda item: item["score"], reverse=True)[:keep_top_k]

    kept_indices = {item["index"] for item in kept}
    filtered = [cluster for idx, cluster in enumerate(segmented_results) if idx in kept_indices]
    return filtered


def _prepare_axes(fig, ax, tau_x_min, tau_x_max, tau_y_min, tau_y_max, style):
    created_fig = False
    if fig is None or ax is None:
        figsize = (7.5, 7.5) if style == "paper" and tau_x_min < 0 else (7, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        created_fig = True
    return fig, ax, created_fig


def _classify_cluster(cluster_pts, boundary_points):
    center = np.mean(cluster_pts, axis=0)
    x_coords = cluster_pts[:, 0]
    y_coords = cluster_pts[:, 1]

    lx = np.percentile(x_coords, 99) - np.percentile(x_coords, 1)
    ly = np.percentile(y_coords, 99) - np.percentile(y_coords, 1)
    l_max = max(lx, ly)
    l_min = min(lx, ly)
    aspect_ratio = l_max / l_min if l_min > 0 else 1.0

    dists = np.linalg.norm(cluster_pts - center, axis=1)
    r_mean = np.mean(dists)
    dists_sorted = np.sort(dists)
    min_dist_to_center = np.min(dists)
    max_dist_to_center = np.max(dists)
    tori_ratio = min_dist_to_center / max_dist_to_center if max_dist_to_center > 1e-12 else 0.0

    res_rect = dist_to_axis_aligned_rect(boundary_points, center, lx, ly)
    res_ellip = dist_to_axis_aligned_ellipsoid(boundary_points, center, lx, ly)

    if res_rect < res_ellip:
        shape_type = "Rectangle"
        final_params = {"center": center, "W": lx, "H": ly}
    elif 0.2 <= tori_ratio <= 0.5:
        idx_min = max(0, min(len(dists_sorted) - 1, int(np.floor(0.05 * len(dists_sorted))) - 1))
        idx_max = max(0, min(len(dists_sorted) - 1, int(np.floor(0.95 * len(dists_sorted))) - 1))
        r_min = float(dists_sorted[idx_min])
        r_max = float(dists_sorted[idx_max])
        r_tube = 0.5 * (r_max - r_min)
        r_major = 0.5 * (r_max + r_min)
        shape_type = "Donut"
        final_params = {
            "center": center,
            "R_inner": r_min,
            "R_outer": r_max,
            "R_major": r_major,
            "r_minor": r_tube,
            "tori_ratio": tori_ratio,
            # Backward-compatible keys used by older notebooks.
            "r_min": r_min,
            "r_max": r_max,
        }
    elif aspect_ratio <= 1.2:
        idx_circle = max(0, min(len(dists_sorted) - 1, int(np.floor(0.95 * len(dists_sorted))) - 1))
        shape_type = "Circle"
        final_params = {"center": center, "radius": float(dists_sorted[idx_circle])}
    else:
        shape_type = "Ellipsoid"
        final_params = {"center": center, "a": lx / 2, "b": ly / 2}

    curvature_stats = compute_boundary_curvature(boundary_points)
    if curvature_stats["std"] > 0.3 and curvature_stats["peak_ratio"] < 8.0:
        shape_type = "general"

    return center, lx, ly, aspect_ratio, res_rect, res_ellip, shape_type, final_params


def _cluster_annotation(shape_type, final_params):
    center = np.asarray(final_params["center"])
    lines = [f"c=({center[0]:.3f}, {center[1]:.3f})"]

    if shape_type == "Circle":
        lines.append(f"r={final_params['radius']:.3f}")
    elif shape_type == "Rectangle":
        lines.append(f"W={final_params['W']:.3f}, H={final_params['H']:.3f}")
    elif shape_type == "Ellipsoid":
        lines.append(f"a={final_params['a']:.3f}, b={final_params['b']:.3f}")
    elif shape_type == "Donut":
        lines.append(f"Rin={final_params['R_inner']:.3f}, Rout={final_params['R_outer']:.3f}")
    else:
        lines.append(shape_type)

    return "\n".join(lines)


def detect_shape(
    grad,
    X,
    Y,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    para_abs,
    para_grad,
    eps,
    mini_samples,
    S_num=None,
    ax=None,
    fig=None,
    output_directory=".",
    output_filename="plot.png",
    style="standard",
    show_colorbar=True,
    min_cluster_size=None,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
):
    segmented_results = _cluster_points(grad, X, Y, para_abs, para_grad, eps, mini_samples, S_num, eps_mode=eps_mode)
    raw_cluster_count = len(segmented_results)
    segmented_results = _filter_clusters(
        segmented_results,
        min_cluster_size=min_cluster_size,
        min_relative_abs_s=min_relative_abs_s,
        min_relative_grad=min_relative_grad,
        keep_top_k=keep_top_k,
    )
    print("Number of detected clusters:", raw_cluster_count)
    if raw_cluster_count != len(segmented_results):
        print("Number of retained clusters:", len(segmented_results))
    if not segmented_results:
        return []

    fig, ax, created_fig = _prepare_axes(fig, ax, tau_x_min, tau_x_max, tau_y_min, tau_y_max, style)

    all_s_vals = np.concatenate([cluster_info["S_vals"] for cluster_info in segmented_results])
    s_min, s_max = np.min(all_s_vals), np.max(all_s_vals)
    results = []
    sc = None

    for i, cluster_info in enumerate(segmented_results):
        cluster_pts = cluster_info["points"]
        s_vals = cluster_info["S_vals"]
        boundary_points = detect_boundary(cluster_pts)

        center, lx, ly, aspect_ratio, res_rect, res_ellip, shape_type, final_params = _classify_cluster(cluster_pts, boundary_points)

        basis_func_type = "Sigmoid"
        if S_num is not None:
            std_s = np.std(s_vals)
            mean_s = np.mean(np.abs(s_vals))
            cv = std_s / mean_s if mean_s > 1e-12 else 0.0
            print(f"  CV of S_num: {cv:.3f}")
            if cv >= 0.7:
                # Algorithm 3 uses the peak-like branch "Exp (or ReLU)".
                # In the 2D implementation we instantiate that branch with
                # Exp only, to keep the basis family fixed once the geometry
                # has been detected.
                basis_func_type = "Exp"

        print(f"Cluster {i}: AR={aspect_ratio:.2f}")
        print(f"  Res_Rect={res_rect:.6f} vs Res_Ellip={res_ellip:.6f}")
        print(f"  center=({center[0]:.4f}, {center[1]:.4f})")
        if shape_type == "Donut":
            print(f"  Tori ratio={final_params['tori_ratio']:.6f}")
        print(f"  -> Decision: {shape_type} ({basis_func_type})")

        results.append(
            {
                "cluster_id": i,
                "shape": shape_type,
                "params": final_params,
                "profile": basis_func_type,
                "boundary_points": boundary_points,
            }
        )

        sc = ax.scatter(
            cluster_pts[:, 0],
            cluster_pts[:, 1],
            c=s_vals,
            cmap="jet",
            marker="o",
            s=5,
            edgecolor="white",
            linewidth=0.5,
            vmin=s_min,
            vmax=s_max,
            zorder=2,
        )

        if shape_type == "Circle":
            patch = plt.Circle(
                center,
                final_params["radius"],
                color="r",
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            ax.add_patch(patch)
        elif shape_type == "Rectangle":
            patch = plt.Rectangle(
                (center[0] - lx / 2, center[1] - ly / 2),
                lx,
                ly,
                linewidth=1.8,
                edgecolor="g",
                facecolor="none",
                linestyle="--",
                zorder=8,
            )
            ax.add_patch(patch)
        elif shape_type == "Ellipsoid":
            patch = plt.matplotlib.patches.Ellipse(
                center,
                2 * final_params["a"],
                2 * final_params["b"],
                angle=0,
                edgecolor="m",
                facecolor="none",
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            ax.add_patch(patch)
        elif shape_type == "Donut":
            outer_patch = plt.Circle(
                center,
                final_params["R_outer"],
                color="orange",
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            inner_patch = plt.Circle(
                center,
                final_params["R_inner"],
                color="orange",
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            ax.add_patch(outer_patch)
            ax.add_patch(inner_patch)

        marker_size = 110 if style != "paper" else 90
        ax.scatter(center[0], center[1], c="k", marker="x", s=marker_size, linewidths=2.0, zorder=6)

    if style != "paper" and sc is not None and show_colorbar:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label("S values", fontsize=18)

    ax.set_xlabel(r"$x_1$", fontsize=40 if style == "paper" else 24)
    if style != "paper":
        ax.set_ylabel(r"$x_2$", fontsize=24)

    n_ticks = 5
    x_ticks_loc = np.linspace(tau_x_min, tau_x_max, n_ticks)
    y_ticks_loc = np.linspace(tau_y_min, tau_y_max, n_ticks)
    ax.set_xticks(x_ticks_loc)
    ax.set_xticklabels(np.round(x_ticks_loc, decimals=3), rotation=0, fontsize=20 if style == "paper" else 18)
    ax.set_yticks(y_ticks_loc)

    if style == "paper":
        ax.set_yticklabels(np.round(y_ticks_loc, decimals=3), fontsize=18)
        ax.set_yticks([])
    else:
        ax.set_yticklabels(np.round(y_ticks_loc, decimals=3), fontsize=18)

    ax.set_xlim(tau_x_min, tau_x_max)
    ax.set_ylim(tau_y_min, tau_y_max)
    ax.set_aspect("equal", adjustable="box")

    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    fig.tight_layout()

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)
    fig.savefig(full_output_path, dpi=300, bbox_inches="tight")

    if created_fig:
        plt.show()

    return results
