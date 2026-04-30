import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
from matplotlib.ticker import FuncFormatter
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
import cv2


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
plt.rcParams["font.family"] = "serif"

PAPER_SEQUENTIAL_CMAP = plt.get_cmap("viridis")
PAPER_BOUNDARY_RED = "#ff3b30"
PAPER_BOUNDARY_BLUE = "#0000cc"
PAPER_COLOR_LEVELS = 20
VISUAL_MATCH_FIGSIZE = (8, 6)
VISUAL_MATCH_LABEL_SIZE = 36
VISUAL_MATCH_TICK_SIZE = 18
VISUAL_MATCH_COLORBAR_TICK_SIZE = 18
VISUAL_MATCH_TICK_FORMAT = "{:.2f}"
VISUAL_MATCH_COLORBAR_TICK_FORMAT = "{:.2f}"
VISUAL_MATCH_COLORBAR_SHRINK = 1
VISUAL_MATCH_COLORBAR_ASPECT = 12
EXAMPLE_DIR_PATTERN = re.compile(r"^Ex\d+\.\d+$", re.IGNORECASE)


def _build_segmented_colormap_and_norm(vmin, vmax, n_levels=PAPER_COLOR_LEVELS):
    if np.isclose(vmin, vmax):
        boundaries = np.linspace(vmin - 0.5, vmax + 0.5, n_levels + 1)
    else:
        boundaries = np.linspace(vmin, vmax, n_levels + 1)
    discrete_cmap = plt.get_cmap(PAPER_SEQUENTIAL_CMAP.name, n_levels)
    norm = mpl.colors.BoundaryNorm(boundaries, discrete_cmap.N, clip=True)
    return discrete_cmap, norm, boundaries


def _make_tick_formatter(tick_format=VISUAL_MATCH_TICK_FORMAT):
    return FuncFormatter(lambda value, _: tick_format.format(value))


def _infer_example_prefix_from_path(path):
    if not path:
        return None
    normalized = os.path.abspath(os.path.expanduser(str(path)))
    for part in reversed(re.split(r"[\\/]+", normalized)):
        if EXAMPLE_DIR_PATTERN.fullmatch(part):
            return part.lower()
    return None


def _infer_example_prefix(*paths):
    for path in paths:
        prefix = _infer_example_prefix_from_path(path)
        if prefix:
            return prefix
    return _infer_example_prefix_from_path(os.getcwd())


def _prefix_filename_for_example(filename, *paths):
    if not filename:
        return filename
    prefix = _infer_example_prefix(*paths)
    if not prefix:
        return filename
    if filename.lower().startswith(prefix + "_"):
        return filename
    return f"{prefix}_{filename}"


def _resolve_output_path(output_directory, output_filename):
    directory = output_directory or "."
    filename = _prefix_filename_for_example(output_filename, directory)
    return os.path.join(directory, filename)


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


def map_index_to_domain(points, tau_x_min, tau_x_max, tau_y_min, tau_y_max, shape):
    """Map grid indices to the physical 2D domain."""
    x_scale = (tau_x_max - tau_x_min) / shape[1]
    y_scale = (tau_y_max - tau_y_min) / shape[0]

    mapped_points = np.zeros_like(points, dtype=float)
    mapped_points[:, 0] = tau_x_min + points[:, 0] * x_scale
    mapped_points[:, 1] = tau_y_min + points[:, 1] * y_scale
    return mapped_points


def _estimate_grid_spacing(X, Y):
    x_unique = np.unique(np.asarray(X, dtype=float).ravel())
    y_unique = np.unique(np.asarray(Y, dtype=float).ravel())
    dx_candidates = np.diff(x_unique)
    dy_candidates = np.diff(y_unique)
    dx_candidates = dx_candidates[dx_candidates > 1e-12]
    dy_candidates = dy_candidates[dy_candidates > 1e-12]
    dx = float(np.min(dx_candidates)) if dx_candidates.size > 0 else 0.0
    dy = float(np.min(dy_candidates)) if dy_candidates.size > 0 else 0.0
    return max(dx, dy)


def _resolve_dbscan_eps(eps, X, Y, eps_mode):
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


def get_contour_points_from_implicit(implicit_func, func_args, x_range, y_range, grid_points=300):
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


def generate_soft_boundary(x_grid, y_grid, contour_points, K=500):
    boundary_tree = cKDTree(contour_points)
    path = mpath.Path(contour_points)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    distances, _ = boundary_tree.query(grid_points, k=1)
    are_inside = path.contains_points(grid_points)
    signed_distances = np.where(are_inside, -distances, distances)
    z_flat = 1 / (1 + np.exp(K * signed_distances))
    return z_flat.reshape(x_grid.shape)


def resample_closed_curve(points, n_points=1200):
    """Resample a closed boundary by arc length."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    # Remove consecutive duplicates before spline fitting.
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


def simple_offset_batch(points, s, distances, n_resample=1200):
    points = resample_closed_curve(points, n_points=n_resample)
    # Interpret s as a per-point smoothing strength to keep legacy notebook scales stable.
    smoothing = float(s) * len(points)
    tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothing, per=True)
    u_new = np.linspace(0, 1, len(points), endpoint=False)

    der1 = np.array(splev(u_new, tck, der=1)).T
    tangent_norm = np.linalg.norm(der1, axis=1)
    tangent_norm[tangent_norm < 1e-12] = 1e-12
    unit_tangent = der1 / tangent_norm[:, np.newaxis]
    unit_normal = np.c_[-unit_tangent[:, 1], unit_tangent[:, 0]]
    smooth_points = np.array(splev(u_new, tck)).T

    distances = np.atleast_1d(distances)
    if len(distances) == 1:
        return smooth_points + distances[0] * unit_normal
    return smooth_points[np.newaxis, :, :] + distances[:, np.newaxis, np.newaxis] * unit_normal[np.newaxis, :, :]


def smooth_closed_curve(points, window=15):
    """Smooth an ordered closed curve by circular moving average."""
    points = np.asarray(points, dtype=float)
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1

    radius = window // 2
    smoothed = np.zeros_like(points)
    for shift in range(-radius, radius + 1):
        smoothed += np.roll(points, shift, axis=0)
    return smoothed / window


def center_oriented_offset_batch(boundary, center, distances, n_resample=1200, smooth_window=15):
    """Offset an ordered boundary with a fixed inward/outward normal convention.

    cv2.findContours returns ordered contour points, but the contour orientation
    can be clockwise or counter-clockwise.  Therefore the raw normal direction in
    simple_offset_batch is ambiguous.  This helper orients the normal away from an
    interior reference point, so negative distances consistently shrink inward.
    """
    curve = resample_closed_curve(boundary, n_points=n_resample)
    curve = smooth_closed_curve(curve, window=smooth_window)

    tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-12] = 1e-12
    tangent = tangent / tangent_norm

    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    to_center = np.asarray(center, dtype=float) - curve
    if np.nanmean(np.sum(normal * to_center, axis=1)) > 0:
        normal = -normal

    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    offset = curve[np.newaxis, :, :] + distances[:, np.newaxis, np.newaxis] * normal[np.newaxis, :, :]
    return offset[0] if len(distances) == 1 else offset


def generate_signed_distance(
    x_train,
    boundary_points,
    alpha,
    scale_factor,
    M,
    center=None,
    center_oriented=False,
    n_resample=1200,
):
    if hasattr(x_train, "detach") and hasattr(x_train, "cpu"):
        grid_points = x_train.detach().cpu().numpy()
    else:
        grid_points = np.asarray(x_train)

    if center_oriented:
        if center is None:
            center = np.mean(np.asarray(boundary_points, dtype=float), axis=0)
        bound_num_scaled = center_oriented_offset_batch(
            boundary_points,
            center,
            scale_factor,
            n_resample=n_resample,
        )
    else:
        bound_num_scaled = simple_offset_batch(boundary_points, alpha, scale_factor, n_resample=n_resample)

    signed_distances = np.zeros((grid_points.shape[0], M), dtype=np.float64)
    for i in range(M):
        path = mpath.Path(bound_num_scaled[i, :, :])
        are_inside = path.contains_points(grid_points, radius=-1e-9)
        signed_distances[:, i : i + 1] = np.where(are_inside, 1, -1).reshape(-1, 1)
    return bound_num_scaled, signed_distances


def _point_spacing(points):
    if len(points) < 2:
        return 0.0
    distances, _ = cKDTree(points).query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-12)]
    if nearest.size == 0:
        return 0.0
    return float(np.percentile(nearest, 75))


def _fallback_hull_boundary(points):
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def detect_boundary(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    padding = 0.1
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding

    img_size = 1000
    span = max(x_max - x_min, y_max - y_min)
    if span <= 1e-12:
        return points
    scale = img_size / span
    points_pixel = np.rint((points - [x_min, y_min]) * scale).astype(int)
    points_pixel = np.clip(points_pixel, 0, img_size - 1)

    pixel_spacing = max(_point_spacing(points) * scale, 1.0)
    base_radius = max(1, int(np.ceil(0.55 * pixel_spacing)))
    min_contour_points = 3 if len(points) <= 50 else min(80, max(20, int(0.05 * len(points))))

    # Sparse grid points can be 8-10 pixels apart after rasterization.  Drawing
    # one-pixel dots makes cv2.findContours select a tiny local component, so
    # the drawing radius and closing kernel are tied to the measured point gap.
    for growth in (1.0, 1.5, 2.0, 3.0):
        point_radius = max(1, int(np.ceil(base_radius * growth)))
        close_radius = max(1, int(np.ceil(0.35 * pixel_spacing * growth)))
        kernel_size = 2 * close_radius + 1

        binary_img = np.zeros((img_size, img_size), dtype=np.uint8)
        for x, y in points_pixel:
            cv2.circle(binary_img, (int(x), int(y)), point_radius, 255, -1)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated_img = cv2.dilate(closed_img, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.squeeze()
        if contour_points.ndim != 2 or contour_points.shape[0] < 3:
            continue

        boundary_points = contour_points / scale + [x_min, y_min]
        if len(boundary_points) >= min_contour_points or growth == 3.0:
            return boundary_points

    return _fallback_hull_boundary(points)


def dist_to_axis_aligned_rect(points, center, width, height):
    p_local = np.abs(points - center)
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    metric = np.maximum(p_local[:, 0] / a, p_local[:, 1] / b)
    return np.mean(np.abs(metric - 1))


def dist_to_axis_aligned_ellipsoid(points, center, width, height):
    p_local = points - center
    a = max(width / 2, 1e-6)
    b = max(height / 2, 1e-6)
    metric = np.sqrt((p_local[:, 0] / a) ** 2 + (p_local[:, 1] / b) ** 2)
    return np.mean(np.abs(metric - 1))


def _compute_cv_from_values(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0
    mean_abs = np.mean(np.abs(values))
    cv = np.std(values) / mean_abs if mean_abs > 1e-12 else 0.0
    return float(cv), int(values.size)


def _mask_inside_polygon_2d(X, Y, boundary_points):
    boundary_points = np.asarray(boundary_points, dtype=float)
    if boundary_points.ndim != 2 or boundary_points.shape[0] < 3:
        return np.zeros(np.asarray(X).shape, dtype=bool)
    path = mpath.Path(boundary_points, closed=True)
    coords = np.column_stack((np.asarray(X, dtype=float).ravel(), np.asarray(Y, dtype=float).ravel()))
    return path.contains_points(coords, radius=1e-12).reshape(np.asarray(X).shape)


def _mask_inside_shape_2d(X, Y, shape_type, final_params, boundary_points):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    center = np.asarray(final_params.get("center", [np.nan, np.nan]), dtype=float)
    tol = 1e-12

    if shape_type == "Rectangle":
        width = float(final_params.get("W", np.nan))
        height = float(final_params.get("H", np.nan))
        return (np.abs(X - center[0]) <= width / 2 + tol) & (np.abs(Y - center[1]) <= height / 2 + tol)

    if shape_type == "Ellipsoid":
        a = max(float(final_params.get("a", np.nan)), 1e-12)
        b = max(float(final_params.get("b", np.nan)), 1e-12)
        metric = ((X - center[0]) / a) ** 2 + ((Y - center[1]) / b) ** 2
        return metric <= 1.0 + 1e-9

    if shape_type == "Donut":
        r_min = float(final_params.get("r_min", np.nan))
        r_max = float(final_params.get("r_max", np.nan))
        rho = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return (rho >= r_min - tol) & (rho <= r_max + tol)

    if shape_type == "general":
        return _mask_inside_polygon_2d(X, Y, boundary_points)

    return _mask_inside_polygon_2d(X, Y, boundary_points)


def _interior_s_values_2d(X, Y, S_num, shape_type, final_params, boundary_points):
    S_num = _resolve_field_array(S_num, "S_num")
    mask = _mask_inside_shape_2d(X, Y, shape_type, final_params, boundary_points)
    values = np.asarray(S_num, dtype=float)[mask]
    source = "geometry_interior" if shape_type != "general" else "boundary_interior"
    if values.size == 0 and len(boundary_points) >= 3:
        fallback_mask = _mask_inside_polygon_2d(X, Y, boundary_points)
        values = np.asarray(S_num, dtype=float)[fallback_mask]
        if values.size > 0:
            source = "boundary_interior_fallback"
    return values, source


def convex_hull_boundary(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]


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
    filtered = [cluster for idx, cluster in enumerate(segmented_results) if idx in kept_indices]
    return filtered


def _prepare_axes(fig, ax, tau_x_min, tau_x_max, tau_y_min, tau_y_max, style):
    created_fig = False
    if fig is None or ax is None:
        figsize = (7.5, 7.5) if style == "paper" and tau_x_min < 0 else VISUAL_MATCH_FIGSIZE
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        created_fig = True
    return fig, ax, created_fig


def _classify_cluster(cluster_pts, boundary_points, fit_rotation=False, residual_threshold=0.05, t_aspect_ratio=1.2):
    del fit_rotation
    geometry_points = boundary_points if len(boundary_points) >= 3 else cluster_pts
    center = np.mean(geometry_points, axis=0)
    x_coords = cluster_pts[:, 0]
    y_coords = cluster_pts[:, 1]

    lx = np.max(x_coords) - np.min(x_coords)
    ly = np.max(y_coords) - np.min(y_coords)
    res_rect = dist_to_axis_aligned_rect(geometry_points, center, lx, ly)
    res_ellip = dist_to_axis_aligned_ellipsoid(geometry_points, center, lx, ly)

    l_max = max(lx, ly)
    l_min = min(lx, ly)
    aspect_ratio = l_max / l_min if l_min > 0 else 1.0

    dists = np.linalg.norm(cluster_pts - center, axis=1)
    r_mean = np.mean(dists)
    dists_sorted = np.sort(dists)
    min_dist_to_center = np.min(dists)
    max_dist_to_center = np.max(dists)
    tori_ratio = min_dist_to_center / max_dist_to_center if max_dist_to_center > 1e-12 else 0.0
    idx_dist_05 = max(0, min(len(dists_sorted) - 1, int(np.floor(0.05 * len(dists_sorted))) - 1))
    idx_dist_95 = max(0, min(len(dists_sorted) - 1, int(np.floor(0.95 * len(dists_sorted))) - 1))

    if min(res_rect, res_ellip) >= residual_threshold:
        shape_type = "general"
        final_params = {"center": center}
    elif res_rect <= res_ellip:
        shape_type = "Rectangle"
        final_params = {"center": center, "W": lx, "H": ly}
    else:
        if 0.2 <= tori_ratio <= 0.5:
            r_min = float(dists_sorted[idx_dist_05])
            r_max = float(dists_sorted[idx_dist_95])
            r_tube = 0.5 * (r_max - r_min)
            r_major = 0.5 * (r_max + r_min)
            shape_type = "Donut"
            final_params = {
                "center": center,
                "r_min": r_min,
                "r_max": r_max,
                "tori_ratio": tori_ratio,
                "R_major": r_major,
                "r_minor": r_tube,
            }
        else:
            shape_type = "Ellipsoid"
            final_params = {"center": center, "a": lx / 2, "b": ly / 2, "radius": float(dists_sorted[idx_dist_95])}

    diagnostics = {
        "Lx": float(lx),
        "Ly": float(ly),
        "aspect_ratio": float(aspect_ratio),
        "res_rect": float(res_rect),
        "res_ellip": float(res_ellip),
        "residual_threshold": float(residual_threshold),
        "tori_ratio": float(tori_ratio),
        "r_mean": float(r_mean),
        "min_dist_to_center": float(min_dist_to_center),
        "max_dist_to_center": float(max_dist_to_center),
        "dist_05": float(dists_sorted[idx_dist_05]),
        "dist_95": float(dists_sorted[idx_dist_95]),
    }

    return center, lx, ly, aspect_ratio, res_rect, res_ellip, shape_type, final_params, diagnostics


def _cluster_annotation(shape_type, final_params):
    center = np.asarray(final_params["center"])
    lines = [f"c=({center[0]:.3f}, {center[1]:.3f})"]

    if shape_type == "Rectangle":
        lines.append(f"W={final_params['W']:.3f}, H={final_params['H']:.3f}")
    elif shape_type == "Ellipsoid":
        lines.append(f"a={final_params['a']:.3f}, b={final_params['b']:.3f}")
    elif shape_type == "Donut":
        lines.append(f"rmin={final_params['r_min']:.3f}, rmax={final_params['r_max']:.3f}")
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
    fit_rotation=False,
    show_general_boundary=False,
    show_detected_boundary=False,
    debug_plot_boundary_points=False,
    residual_threshold=0.05,
    t_aspect_ratio=1.2,
    t_cv=0.7,
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
    segmented_cmap, segmented_norm, color_boundaries = _build_segmented_colormap_and_norm(s_min, s_max)
    results = []
    sc = None
    detected_boundary_label_used = False

    for i, cluster_info in enumerate(segmented_results):
        cluster_pts = cluster_info["points"]
        s_vals = cluster_info["S_vals"]
        boundary_points = detect_boundary(cluster_pts)
        if debug_plot_boundary_points:
            debug_fig, debug_ax = plt.subplots(figsize=(6, 5))
            debug_ax.scatter(
                cluster_pts[:, 0],
                cluster_pts[:, 1],
                s=6,
                c="lightgray",
                alpha=0.7,
                linewidths=0,
                label="Cluster points",
            )
            debug_ax.plot(
                boundary_points[:, 0],
                boundary_points[:, 1],
                color="black",
                linewidth=1.5,
                label="detect_boundary output",
            )
            debug_ax.scatter(
                boundary_points[:, 0],
                boundary_points[:, 1],
                s=8,
                c="red",
                alpha=0.55,
                linewidths=0,
                label="Boundary points",
            )
            debug_ax.set_title(f"Cluster {i}: boundary_points from detect_boundary")
            debug_ax.set_xlabel(r"$x_1$")
            debug_ax.set_ylabel(r"$x_2$")
            debug_ax.set_xlim(tau_x_min, tau_x_max)
            debug_ax.set_ylim(tau_y_min, tau_y_max)
            debug_ax.set_aspect("equal", adjustable="box")
            debug_ax.grid(True, linestyle=":", alpha=0.2)
            debug_ax.legend(fontsize=9)
            debug_fig.tight_layout()
            plt.show()
        center, lx, ly, aspect_ratio, res_rect, res_ellip, shape_type, final_params, diagnostics = _classify_cluster(
            cluster_pts,
            boundary_points,
            fit_rotation=fit_rotation,
            residual_threshold=residual_threshold,
            t_aspect_ratio=t_aspect_ratio,
        )

        basis_func_type = "Sigmoid"
        cv = np.nan
        cv_point_count = 0
        cv_source = "unavailable"
        if S_num is not None:
            cv_values, cv_source = _interior_s_values_2d(X, Y, S_num, shape_type, final_params, boundary_points)
            cv, cv_point_count = _compute_cv_from_values(cv_values)
            if cv_point_count == 0:
                cv, cv_point_count = _compute_cv_from_values(s_vals)
                cv_source = "cluster_points_fallback"
            print(f"  CV of S_num ({cv_source}, n={cv_point_count}): {cv:.3f}")
            if cv >= t_cv:
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
                "diagnostics": diagnostics,
                "aspect_ratio": diagnostics["aspect_ratio"],
                "res_rect": diagnostics["res_rect"],
                "res_ellip": diagnostics["res_ellip"],
                "Lx": diagnostics["Lx"],
                "Ly": diagnostics["Ly"],
                "tori_ratio": diagnostics["tori_ratio"],
                "dist_05": diagnostics["dist_05"],
                "dist_95": diagnostics["dist_95"],
                "cv": float(cv) if np.isfinite(cv) else np.nan,
                "cv_source": cv_source,
                "cv_point_count": int(cv_point_count),
            }
        )

        sc = ax.scatter(
            cluster_pts[:, 0],
            cluster_pts[:, 1],
            c=s_vals,
            cmap=segmented_cmap,
            norm=segmented_norm,
            marker="s",
            s=9,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            rasterized=True,
            zorder=2,
        )

        if (show_detected_boundary or (show_general_boundary and shape_type == "general")) and len(boundary_points) > 0:
            label = None if detected_boundary_label_used else "Detected outer boundary"
            ax.plot(
                boundary_points[:, 0],
                boundary_points[:, 1],
                color="black",
                linestyle="-",
                linewidth=1.4,
                alpha=0.95,
                label=label,
                zorder=9,
            )
            detected_boundary_label_used = True

        plot_center = np.asarray(final_params.get("center", center), dtype=float)
        if shape_type == "Rectangle":
            width = final_params["W"]
            height = final_params["H"]
            patch = plt.Rectangle(
                (plot_center[0] - width / 2, plot_center[1] - height / 2),
                width,
                height,
                linewidth=2.4,
                edgecolor=PAPER_BOUNDARY_BLUE,
                facecolor="none",
                linestyle="--",
                zorder=8,
            )
            ax.add_patch(patch)
        elif shape_type == "Ellipsoid":
            patch = plt.matplotlib.patches.Ellipse(
                plot_center,
                2 * final_params["a"],
                2 * final_params["b"],
                edgecolor=PAPER_BOUNDARY_RED,
                facecolor="none",
                linestyle="--",
                linewidth=2.8,
                zorder=8,
            )
            ax.add_patch(patch)
        elif shape_type == "Donut":
            outer_patch = plt.Circle(
                plot_center,
                final_params["r_max"],
                color="orange",
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            inner_patch = plt.Circle(
                plot_center,
                final_params["r_min"],
                color="orange",
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=8,
            )
            ax.add_patch(outer_patch)
            ax.add_patch(inner_patch)

        marker_size = 110 if style != "paper" else 90
        ax.scatter(plot_center[0], plot_center[1], c="k", marker="x", s=marker_size, linewidths=2.0, zorder=6)

    if style != "paper" and sc is not None and show_colorbar:
        cbar = fig.colorbar(
            sc,
            ax=ax,
            shrink=VISUAL_MATCH_COLORBAR_SHRINK,
            aspect=VISUAL_MATCH_COLORBAR_ASPECT,
            boundaries=color_boundaries,
            spacing="proportional",
        )
        cbar.ax.tick_params(labelsize=VISUAL_MATCH_COLORBAR_TICK_SIZE)
        cbar.ax.yaxis.set_major_formatter(_make_tick_formatter(VISUAL_MATCH_COLORBAR_TICK_FORMAT))
        cbar.update_ticks()
        colorbar_ticks = cbar.get_ticks()
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels([VISUAL_MATCH_COLORBAR_TICK_FORMAT.format(tick) for tick in colorbar_ticks])
        #cbar.set_label("S values", fontsize=18)

    ax.set_xlabel(r"$x_1$", fontsize=40 if style == "paper" else VISUAL_MATCH_LABEL_SIZE)
    if style != "paper":
        ax.set_ylabel(r"$x_2$", fontsize=VISUAL_MATCH_LABEL_SIZE)

    if style == "paper":
        n_ticks = 5
        x_ticks_loc = np.linspace(tau_x_min, tau_x_max, n_ticks)
        y_ticks_loc = np.linspace(tau_y_min, tau_y_max, n_ticks)
        ax.set_xticks(x_ticks_loc)
        ax.set_xticklabels(np.round(x_ticks_loc, decimals=3), rotation=0, fontsize=20)
        ax.set_yticks(y_ticks_loc)
        ax.set_yticklabels(np.round(y_ticks_loc, decimals=3), fontsize=18)
        ax.set_yticks([])
    else:
        formatter = _make_tick_formatter()
        ax.tick_params(axis="both", which="major", labelsize=VISUAL_MATCH_TICK_SIZE)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    ax.set_xlim(tau_x_min, tau_x_max)
    ax.set_ylim(tau_y_min, tau_y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    fig.tight_layout()

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = _resolve_output_path(output_directory, output_filename)
    fig.savefig(full_output_path, dpi=300, bbox_inches="tight")

    if created_fig:
        plt.show()

    return results


def split_detected_basis_families(
    results,
    min_general_boundary_points=None,
    regular_shape_map=None,
    require_general=True,
    warn_multiple_regular=True,
    verbose=True,
):
    """Split detect_shape results into general and regular basis branches.

    This is a notebook helper for the general-source examples: it keeps all
    general clusters that pass the optional boundary-size filter, chooses the
    first regular cluster as the single regular branch, and returns the boundary
    dictionaries used by the later SDF and basis-initialization cells.
    """
    if regular_shape_map is None:
        regular_shape_map = {"rectangle": "rec", "ellipsoid": "ellipsoid", "donut": "donut"}

    cluster_lookup = {int(item["cluster_id"]): item for item in results}
    general_candidates = [item for item in results if str(item.get("shape", "")).lower() == "general"]
    regular_results = [item for item in results if str(item.get("shape", "")).lower() != "general"]

    general_results = []
    for item in general_candidates:
        cluster_id = int(item["cluster_id"])
        boundary_count = len(np.asarray(item.get("boundary_points", [])))
        if min_general_boundary_points is not None and boundary_count < min_general_boundary_points:
            if verbose:
                print(
                    f"Drop general cluster_id={cluster_id}: boundary_points={boundary_count} "
                    f"< {min_general_boundary_points}"
                )
            continue
        general_results.append(item)

    if require_general and not general_results:
        raise ValueError("No general clusters detected in results.")
    if warn_multiple_regular and len(regular_results) > 1 and verbose:
        print(f"Warning: {len(regular_results)} regular clusters detected; using the first one for the single regular branch.")

    general_cluster_ids = [int(item["cluster_id"]) for item in general_results]
    regular_result = regular_results[0] if regular_results else None
    regular_cluster_id = int(regular_result["cluster_id"]) if regular_result is not None else None
    regular_shape_detected = str(regular_result["shape"]).lower() if regular_result is not None else None
    regular_shape = regular_shape_map.get(regular_shape_detected, regular_shape_detected)

    general_boundaries = {}
    for item in general_results:
        cluster_id = int(item["cluster_id"])
        boundary = np.asarray(item["boundary_points"])
        general_boundaries[cluster_id] = boundary
        if verbose:
            center = np.asarray(item["params"]["center"])
            print(
                f"general cluster_id={cluster_id}, center=({center[0]:.4f}, {center[1]:.4f}), "
                f"boundary_points={boundary.shape[0]}"
            )

    regular_boundary = None
    if regular_result is not None:
        regular_boundary = np.asarray(regular_result["boundary_points"])
        if verbose:
            regular_params = regular_result["params"]
            regular_center = np.asarray(regular_params["center"])
            print(
                f"regular cluster_id={regular_cluster_id}, shape={regular_shape}, "
                f"center=({regular_center[0]:.4f}, {regular_center[1]:.4f}), "
                f"a={regular_params.get('a', np.nan):.4f}, b={regular_params.get('b', np.nan):.4f}"
            )

    return {
        "cluster_lookup": cluster_lookup,
        "general_results": general_results,
        "regular_results": regular_results,
        "general_cluster_ids": general_cluster_ids,
        "regular_result": regular_result,
        "regular_cluster_id": regular_cluster_id,
        "regular_shape_detected": regular_shape_detected,
        "regular_shape": regular_shape,
        "general_boundaries": general_boundaries,
        "regular_boundary": regular_boundary,
    }


def _format_sci(value, precision=2):
    try:
        value = float(value)
    except Exception:
        return "-"
    return f"{value:.{precision}e}"


def _format_fixed(value, precision=3):
    try:
        value = float(value)
    except Exception:
        return "-"
    return f"{value:.{precision}f}"


def _half_value(value):
    try:
        return 0.5 * float(value)
    except Exception:
        return np.nan


def _latex_param_text(result, precision=2):
    shape = str(result.get("shape", "")).lower()
    params = result.get("params", {})
    if shape == "rectangle":
        width = params.get("W", result.get("Lx", np.nan))
        height = params.get("H", result.get("Ly", np.nan))
        return rf"$\boldsymbol{{L}}$=({_format_sci(_half_value(width), precision)}, {_format_sci(_half_value(height), precision)})"
    if shape == "ellipsoid":
        return rf"$\boldsymbol{{L}}$=({_format_sci(params.get('a', np.nan), precision)}, {_format_sci(params.get('b', np.nan), precision)})"
    if shape == "donut":
        r_min = params.get("r_min", params.get("R_inner", np.nan))
        r_max = params.get("r_max", params.get("R_outer", np.nan))
        return rf"$(r_{{min}},r_{{max}})$=({_format_sci(r_min, precision)}, {_format_sci(r_max, precision)})"
    return "-"


def print_latex_result_rows(results, example_name="Ex 4.7", start_cluster=1, precision=2, include_header=False):
    """Print detect_shape results as LaTeX table rows for the paper.

    The parameter column reports half axis-aligned lengths for rectangles and
    ellipsoids. General clusters are
    reported with "-" because their geometry is represented by the detected
    numerical boundary rather than a finite-dimensional shape parameter.
    """
    rows = []
    if include_header:
        header = (
            r"{\emph{Ex}} & Cluster $k$ & $\boldsymbol{c}_k$ & "
            r"$r_k$ or $\boldsymbol{L}_k$ & "
            r"$\mathcal{E}_{\mathrm{rect}}^{(k)}$ & "
            r"$\mathcal{E}_{\mathrm{ellip}}^{(k)}$ & "
            r"$\mathrm{AR}_k$ & $\mathrm{CV}_k$ & $\mathcal{T}_k$ & $\sigma_k$ \\"
        )
        rows.append(header)
    n_rows = len(results)
    for row_idx, result in enumerate(results):
        params = result.get("params", {})
        center = np.asarray(params.get("center", [np.nan, np.nan]), dtype=float)
        cv = result.get("cv", result.get("diagnostics", {}).get("cv", np.nan))
        ex_cell = rf"\multirow{{{n_rows}}}{{*}}{{{example_name}}}" if row_idx == 0 and n_rows > 1 else (example_name if n_rows == 1 else "")
        row = (
            f"{ex_cell} & {start_cluster + row_idx} "
            f"& ({_format_sci(center[0], precision)}, {_format_sci(center[1], precision)}) "
            f"& {_latex_param_text(result, precision)} "
            f"& {_format_fixed(result.get('res_rect', np.nan), 4)} "
            f"& {_format_fixed(result.get('res_ellip', np.nan), 4)} "
            f"& {_format_fixed(result.get('aspect_ratio', np.nan), 2)} "
            f"& {_format_fixed(cv, 3)} "
            f"& \\textbf{{{result.get('shape', '-')}}} "
            f"& {result.get('profile', '-')} \\\\"
        )
        rows.append(row)
    print("\n".join(rows))
    return rows
