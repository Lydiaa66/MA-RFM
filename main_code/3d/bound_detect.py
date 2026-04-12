import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

mpl.rcdefaults()


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
            raise ValueError(f"{name} object array does not contain a valid 3D field")
        if field.ndim >= 3:
            return field
        raise ValueError(f"{name} must be a 3D array, got shape {field.shape}")

    if isinstance(field, (list, tuple)):
        if not field:
            raise ValueError(f"{name} is empty")
        for item in reversed(field):
            try:
                return _resolve_field_array(item, name)
            except Exception:
                continue
        raise ValueError(f"{name} does not contain a valid 3D field")

    if hasattr(field, "detach") and hasattr(field, "cpu"):
        return _resolve_field_array(field.detach().cpu().numpy(), name)

    return _resolve_field_array(np.asarray(field), name)


def _estimate_grid_spacing(X, Y, Z):
    spacings = []
    for axis in (X, Y, Z):
        unique_vals = np.unique(np.asarray(axis, dtype=float).ravel())
        diffs = np.diff(unique_vals)
        diffs = diffs[diffs > 1e-12]
        if diffs.size > 0:
            spacings.append(float(np.min(diffs)))
    return max(spacings) if spacings else 0.0


def _resolve_dbscan_eps(eps, X, Y, Z, eps_mode):
    if eps_mode not in {"auto", "absolute", "grid_scaled"}:
        raise ValueError(f"Unsupported eps_mode: {eps_mode}")

    h = _estimate_grid_spacing(X, Y, Z)
    if eps_mode == "absolute":
        return float(eps), h, "absolute"
    if eps_mode == "grid_scaled":
        return float(eps) * h, h, "grid_scaled"

    if float(eps) > 1.0:
        return float(eps) * h, h, "grid_scaled"
    return float(eps), h, "absolute"


def _sorted_quantile_index(length, quantile):
    if length <= 0:
        return 0
    idx = int(np.floor(quantile * length)) - 1
    return max(0, min(length - 1, idx))


def _sample_field_to_shape(field, target_shape, name):
    field = np.asarray(field)
    if field.shape == target_shape:
        return field
    if field.ndim != len(target_shape):
        raise ValueError(f"{name} shape {field.shape} cannot be sampled to target shape {target_shape}")
    if any(target > source for target, source in zip(target_shape, field.shape)):
        raise ValueError(f"{name} shape {field.shape} is coarser than target shape {target_shape}; refusing to upsample")

    indexer = tuple(np.rint(np.linspace(0, source - 1, target)).astype(int) for target, source in zip(target_shape, field.shape))
    print(f"{name} shape {field.shape} does not match S_num shape {target_shape}; sampling on the S_num grid without interpolation.")
    return field[np.ix_(*indexer)]


def detect_boundary_3d(points):
    if len(points) < 4:
        return points

    padding = 0.1
    xyz_min = points.min(axis=0) - padding
    xyz_max = points.max(axis=0) + padding
    max_span = float(np.max(xyz_max - xyz_min))
    if max_span <= 1e-12:
        return points

    img_size = 100
    scale = img_size / max_span
    points_pixel = ((points - xyz_min) * scale).astype(int) # Map 3D points to an integer voxel grid for morphological processing.

    grid_shape = np.maximum(((xyz_max - xyz_min) * scale).astype(int) + 5, 5)
    grid = np.zeros(tuple(grid_shape.tolist()), dtype=np.uint8)

    for axis in range(3):
        points_pixel[:, axis] = np.clip(points_pixel[:, axis], 0, grid_shape[axis] - 1)

    grid[points_pixel[:, 0], points_pixel[:, 1], points_pixel[:, 2]] = 1 # Mark discretized points as occupied voxels in the 3D grid.

    structure = np.ones((3, 3, 3), dtype=bool)
    closed_grid = binary_closing(grid, structure=structure, iterations=2) # Closing fills small holes in the voxel grid.
    dilated_grid = binary_dilation(closed_grid, structure=structure, iterations=1) # Dilation expands occupied voxels so the boundary becomes connected.
    eroded_grid = binary_erosion(dilated_grid, structure=structure, iterations=1)  # Erosion shrinks the occupied volume.
    boundary_grid = dilated_grid ^ eroded_grid # Keep voxels that remain after dilation but are removed by erosion, i.e. the grid surface.

    boundary_indices = np.argwhere(boundary_grid)
    if len(boundary_indices) == 0:
        return points
    return boundary_indices / scale + xyz_min


def compute_boundary_curvature_3d(points, k=15):
    if len(points) < 4:
        return {"std": 0.0, "mean": 0.0, "peak_ratio": 0.0}

    neighbor_count = min(max(4, k), len(points)) 
    nbrs = NearestNeighbors(n_neighbors=neighbor_count, algorithm="ball_tree").fit(points) # Find k nearest neighbors for each point.
    _, indices = nbrs.kneighbors(points)

    curvatures = []
    for idx in range(len(points)):
        neighbor_points = points[indices[idx]] # Extract neighboring points for the current point.
        centered = neighbor_points - np.mean(neighbor_points, axis=0) # Center the neighborhood at its mean.
        denom = max(neighbor_count - 1, 1) 
        cov = np.dot(centered.T, centered) / denom
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov)))
        eigen_sum = np.sum(eigenvalues)
        curvature = eigenvalues[0] / eigen_sum if eigen_sum > 1e-12 else 0.0
        curvatures.append(curvature)

    curvatures = np.asarray(curvatures, dtype=float)
    mean_k = float(np.mean(curvatures))
    std_k = float(np.std(curvatures))
    peak_ratio = float(np.max(curvatures) / (mean_k + 1e-8))
    return {"std": std_k, "mean": mean_k, "peak_ratio": peak_ratio}


def dist_to_axis_aligned_box(points, center, width, height, depth):
    p_local = np.abs(points - center)
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    c = max(depth / 2, 1e-6)
    metric = np.maximum(np.maximum(p_local[:, 0] / a, p_local[:, 1] / b), p_local[:, 2] / c)
    return float(np.mean(np.abs(metric - 1)))


def dist_to_axis_aligned_ellipsoid_3d(points, center, width, height, depth):
    p_local = points - center
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    c = max(depth / 2, 1e-6)
    metric = np.sqrt((p_local[:, 0] / a) ** 2 + (p_local[:, 1] / b) ** 2 + (p_local[:, 2] / c) ** 2)
    return float(np.mean(np.abs(metric - 1)))


def _cluster_points_3d(grad, X, Y, Z, para_abs, para_grad, eps, mini_samples, S_num, eps_mode="auto"):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    try:
        grad = _resolve_field_array(grad, "grad")
    except ValueError as exc:
        if "empty" not in str(exc):
            raise
        S_num = _resolve_field_array(S_num, "S_num")
        grad_axes = np.gradient(S_num)
        grad = np.sqrt(sum(axis_grad**2 for axis_grad in grad_axes))
        print("grad is empty; using numerical gradient from S_num instead. Classification may differ from true g_S.")

    S_num = _resolve_field_array(S_num, "S_num")
    if grad.shape != S_num.shape:
        target_shape = S_num.shape
        grad = _sample_field_to_shape(grad, target_shape, "grad")
        X = _sample_field_to_shape(X, target_shape, "X")
        Y = _sample_field_to_shape(Y, target_shape, "Y")
        Z = _sample_field_to_shape(Z, target_shape, "Z")

    grad_threshold = np.max(grad) / para_grad
    abs_threshold = np.max(np.abs(S_num)) / para_abs
    mask = (grad > grad_threshold) & (np.abs(S_num) > abs_threshold)
    idx = np.where(mask)

    if len(idx[0]) == 0:
        print("No points detected.")
        return []

    margin = 5
    valid_mask = np.ones(len(idx[0]), dtype=bool)
    shape = grad.shape
    for dim in range(3):
        valid_mask &= (idx[dim] >= margin) & (idx[dim] < shape[dim] - margin)
    filtered_idx = tuple(axis_idx[valid_mask] for axis_idx in idx)

    if len(filtered_idx[0]) == 0:
        print("No points detected.")
        return []

    x_phys = X[filtered_idx]
    y_phys = Y[filtered_idx]
    z_phys = Z[filtered_idx]
    s_vals = S_num[filtered_idx]
    g_vals = grad[filtered_idx]
    combined = np.column_stack((x_phys, y_phys, z_phys, s_vals, g_vals))
    indices = np.column_stack(filtered_idx)

    if len(combined) < 10:
        return [{"points": combined[:, :3], "S_vals": combined[:, 3], "g_vals": combined[:, 4], "indices": indices}]

    eps_dbscan, h, resolved_mode = _resolve_dbscan_eps(eps, X, Y, Z, eps_mode)
    if resolved_mode == "grid_scaled":
        print(f"DBSCAN eps = {eps_dbscan:.6f} ({float(eps):.6f} * h, h = {h:.6f})")

    clusterer = DBSCAN(eps=eps_dbscan, min_samples=mini_samples).fit(combined[:, :3])
    labels = clusterer.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)

    segmented_results = []
    for label in unique_labels:
        mask_k = labels == label
        subset = combined[mask_k]
        segmented_results.append(
            {
                "points": subset[:, :3],
                "S_vals": subset[:, 3],
                "g_vals": subset[:, 4],
                "indices": indices[mask_k],
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
        score = float(size * mean_abs_s * mean_grad)
        metrics.append(
            {
                "index": idx,
                "size": size,
                "mean_abs_s": mean_abs_s,
                "mean_grad": mean_grad,
                "score": score,
            }
        )

    max_mean_abs_s = max((item["mean_abs_s"] for item in metrics), default=0.0)
    max_mean_grad = max((item["mean_grad"] for item in metrics), default=0.0)

    kept = []
    for item in metrics:
        if min_cluster_size is not None and item["size"] < min_cluster_size:
            continue
        if max_mean_abs_s > 0 and item["mean_abs_s"] < min_relative_abs_s * max_mean_abs_s:
            continue
        if max_mean_grad > 0 and item["mean_grad"] < min_relative_grad * max_mean_grad:
            continue
        kept.append(item)

    if keep_top_k is not None and len(kept) > keep_top_k:
        kept = sorted(kept, key=lambda item: item["score"], reverse=True)[:keep_top_k]

    kept_indices = {item["index"] for item in kept}
    return [cluster for idx, cluster in enumerate(segmented_results) if idx in kept_indices]


def _classify_cluster_3d(cluster_pts, boundary_points, t_aspect_ratio=1.2, t_curvature_std=0.3, t_peak_ratio=8.0):
    center = np.mean(cluster_pts, axis=0)

    span_x = np.percentile(cluster_pts[:, 0], 99) - np.percentile(cluster_pts[:, 0], 1)
    span_y = np.percentile(cluster_pts[:, 1], 99) - np.percentile(cluster_pts[:, 1], 1)
    span_z = np.percentile(cluster_pts[:, 2], 99) - np.percentile(cluster_pts[:, 2], 1)
    lx, ly, lz = span_x, span_y, span_z
    # Algorithm 3 uses half axis-aligned lengths L_{alpha,i}.
    half_lengths = 0.5 * np.asarray([span_x, span_y, span_z], dtype=float)
    l_dims = np.asarray([span_x, span_y, span_z], dtype=float)
    l_max = float(np.max(l_dims))
    l_min = float(max(np.min(l_dims), 1e-12))
    aspect_ratio = l_max / l_min

    dists = np.linalg.norm(cluster_pts - center, axis=1)
    dists_sorted = np.sort(dists)
    idx_min = _sorted_quantile_index(len(dists_sorted), 0.05)
    idx_max = _sorted_quantile_index(len(dists_sorted), 0.95)
    min_dist_to_center = float(np.min(dists_sorted))
    max_dist_to_center = float(np.max(dists_sorted))
    raw_tori_ratio = min_dist_to_center / max_dist_to_center if max_dist_to_center > 1e-12 else 0.0
    r_min = float(dists_sorted[idx_min])
    r_max = float(dists_sorted[idx_max])
    tori_ratio = r_min / r_max if r_max > 1e-12 else 0.0

    # Use raw detected points for fitting residuals. The voxel boundary is
    # useful for curvature, but its axis-aligned grid surface biases E_rect.
    residual_points = cluster_pts
    res_rect = dist_to_axis_aligned_box(residual_points, center, lx, ly, lz)
    res_ellip = dist_to_axis_aligned_ellipsoid_3d(residual_points, center, lx, ly, lz)

    if res_rect < res_ellip:
        shape_type = "Rectangle"
        final_params = {"center": center, "Lx": half_lengths[0], "Ly": half_lengths[1], "Lz": half_lengths[2]}
    elif 0.2 <= tori_ratio <= 0.5:
        r_major = 0.5 * (r_max + r_min)
        r_tube = 0.5 * (r_max - r_min)
        shape_type = "Donut"
        final_params = {
            "center": center,
            "R_inner": r_min,
            "R_outer": r_max,
            "R_major": r_major,
            "r_minor": r_tube,
            "tori_ratio": tori_ratio,
            "tori_ratio_raw": raw_tori_ratio,
            # Backward-compatible keys used by older notebooks.
            "r_min": r_min,
            "r_max": r_max,
            "r_inner": float(half_lengths[2]),
            "r": float(span_z),
        }
    elif aspect_ratio <= t_aspect_ratio:
        shape_type = "Circle"
        final_params = {"center": center, "radius": float(dists_sorted[idx_max])}
    else:
        shape_type = "Ellipsoid"
        final_params = {"center": center, "a": lx / 2, "b": ly / 2, "c": lz / 2}

    curvature_stats = compute_boundary_curvature_3d(boundary_points)
    if curvature_stats["std"] > t_curvature_std and curvature_stats["peak_ratio"] < t_peak_ratio:
        shape_type = "general"

    return center, lx, ly, lz, aspect_ratio, res_rect, res_ellip, shape_type, final_params, curvature_stats


def _draw_box(ax, center, lx, ly, lz, color="g"):
    cx, cy, cz = center
    x0, x1 = cx - lx / 2, cx + lx / 2
    y0, y1 = cy - ly / 2, cy + ly / 2
    z0, z1 = cz - lz / 2, cz + lz / 2
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.8,
            zorder=8,
        )


def _draw_sphere(ax, center, radius, color="r"):
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.25, linewidth=0.8, zorder=8)


def _draw_ellipsoid(ax, center, a, b, c, color="m"):
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    x = center[0] + a * np.cos(u) * np.sin(v)
    y = center[1] + b * np.sin(u) * np.sin(v)
    z = center[2] + c * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.25, linewidth=0.8, zorder=8)


def _draw_torus(ax, center, r_major, r_minor, color="orange"):
    u, v = np.mgrid[0 : 2 * np.pi : 28j, 0 : 2 * np.pi : 16j]
    x = center[0] + (r_major + r_minor * np.cos(v)) * np.cos(u)
    y = center[1] + (r_major + r_minor * np.cos(v)) * np.sin(u)
    z = center[2] + r_minor * np.sin(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.25, linewidth=0.8, zorder=8)


def detect_shape_3d(
    grad,
    X,
    Y,
    Z,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    tau_z_min,
    tau_z_max,
    para_abs,
    para_grad,
    eps,
    mini_samples,
    S_num=None,
    elev=None,
    azim=None,
    zlim=None,
    output_directory=".",
    output_filename="plot3d.png",
    show_colorbar=True,
    min_cluster_size=None,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
):
    segmented_results = _cluster_points_3d(grad, X, Y, Z, para_abs, para_grad, eps, mini_samples, S_num, eps_mode=eps_mode)
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

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_position([0.10, 0.03, 0.78, 0.94])

    all_s_vals = np.concatenate([cluster["S_vals"] for cluster in segmented_results])
    s_min, s_max = np.min(all_s_vals), np.max(all_s_vals)
    results = []
    sc = None

    for cluster_id, cluster_info in enumerate(segmented_results):
        cluster_pts = cluster_info["points"]
        s_vals = cluster_info["S_vals"]
        boundary_points = detect_boundary_3d(cluster_pts)

        center, lx, ly, lz, aspect_ratio, res_rect, res_ellip, shape_type, final_params, curvature_stats = _classify_cluster_3d(
            cluster_pts,
            boundary_points,
        )

        basis_func_type = "Sigmoid"
        if S_num is not None:
            std_s = np.std(s_vals)
            mean_s = np.mean(np.abs(s_vals))
            cv = std_s / mean_s if mean_s > 1e-12 else 0.0
            print(f"  CV of S_num: {cv:.3f}")
            if cv >= 0.7:
                basis_func_type = "Exp"

        print(f"Cluster {cluster_id}: AR={aspect_ratio:.2f}")
        print(f"  Res_Rect={res_rect:.6f} vs Res_Ellip={res_ellip:.6f}")
        print(f"  center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        if shape_type in {"Donut", "tori"}:
            print(
                f"  Tori ratio={final_params['tori_ratio']:.6f} "
                f"(raw min/max={final_params['tori_ratio_raw']:.6f})"
            )
        print(f"  -> Decision: {shape_type} ({basis_func_type})")

        results.append(
            {
                "cluster_id": cluster_id,
                "shape": shape_type,
                "params": final_params,
                "profile": basis_func_type,
                "boundary_points": boundary_points,
            }
        )

        sc = ax.scatter(
            cluster_pts[:, 0],
            cluster_pts[:, 1],
            cluster_pts[:, 2],
            c=s_vals,
            cmap="rainbow",
            s=4,
            vmin=s_min,
            vmax=s_max,
            alpha=0.6,
            edgecolors="none",
            zorder=2,
        )

        if shape_type == "Circle":
            _draw_sphere(ax, center, final_params["radius"], color="r")
        elif shape_type == "Rectangle":
            _draw_box(ax, center, lx, ly, lz, color="g")
        elif shape_type == "Ellipsoid":
            _draw_ellipsoid(ax, center, final_params["a"], final_params["b"], final_params["c"], color="m")
        elif shape_type in {"Donut", "tori"}:
            _draw_torus(ax, center, final_params["R_major"], final_params["r_minor"], color="orange")

        ax.scatter(center[0], center[1], center[2], c="k", marker="x", s=90, linewidths=2.0, zorder=10)

    ax.set_xlim(tau_x_min, tau_x_max)
    ax.set_ylim(tau_y_min, tau_y_max)
    ax.set_zlim(zlim if zlim is not None else (tau_z_min, tau_z_max))

    ax.set_xlabel(r"$x_1$", fontsize=25, labelpad=25)
    ax.set_ylabel(r"$x_2$", fontsize=25, labelpad=25)
    ax.set_zlabel(r"$x_3$", fontsize=25, labelpad=20)
    ax.tick_params(axis="x", which="major", labelsize=18, pad=12)
    ax.tick_params(axis="y", which="major", labelsize=18, pad=10)
    ax.tick_params(axis="z", which="major", labelsize=18, pad=10)

    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)

    try:
        ax.set_box_aspect(
            [
                max(tau_x_max - tau_x_min, 1e-12),
                max(tau_y_max - tau_y_min, 1e-12),
                max((zlim[1] - zlim[0]) if zlim is not None else (tau_z_max - tau_z_min), 1e-12),
            ]
        )
    except Exception:
        pass

    if show_colorbar and sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cbar.ax.tick_params(labelsize=20)

    ax.grid(True, linestyle="--", alpha=0.3)

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif not output_directory:
        output_directory = "."

    fig.savefig(os.path.join(output_directory, output_filename or "plot3d.png"), dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.show()
    plt.close(fig)
    return results


def detect_shape(*args, **kwargs):
    return detect_shape_3d(*args, **kwargs)
