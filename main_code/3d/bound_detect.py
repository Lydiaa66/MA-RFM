import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from sklearn.cluster import DBSCAN

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


def _voxelize_points_3d(points, padding=0.1, img_size=100):
    xyz_min = points.min(axis=0) - padding
    xyz_max = points.max(axis=0) + padding
    max_span = float(np.max(xyz_max - xyz_min))
    if max_span <= 1e-12:
        return None, None, None

    scale = img_size / max_span
    points_pixel = ((points - xyz_min) * scale).astype(int)
    grid_shape = np.maximum(((xyz_max - xyz_min) * scale).astype(int) + 5, 5)
    grid = np.zeros(tuple(grid_shape.tolist()), dtype=np.uint8)

    for axis in range(3):
        points_pixel[:, axis] = np.clip(points_pixel[:, axis], 0, grid_shape[axis] - 1)

    grid[points_pixel[:, 0], points_pixel[:, 1], points_pixel[:, 2]] = 1
    return grid, scale, xyz_min


def _extract_boundary_2d(points, padding=0.02, img_size=200):
    if len(points) < 4:
        return points

    mins = points.min(axis=0) - padding
    maxs = points.max(axis=0) + padding
    max_span = float(np.max(maxs - mins))
    if max_span <= 1e-12:
        return points

    scale = img_size / max_span
    pixels = ((points - mins) * scale).astype(int)
    grid_shape = np.maximum(((maxs - mins) * scale).astype(int) + 5, 5)
    grid = np.zeros(tuple(grid_shape.tolist()), dtype=np.uint8)

    for axis in range(2):
        pixels[:, axis] = np.clip(pixels[:, axis], 0, grid_shape[axis] - 1)

    grid[pixels[:, 0], pixels[:, 1]] = 1
    structure = np.ones((3, 3), dtype=bool)
    closed_grid = binary_closing(grid, structure=structure, iterations=1)
    dilated_grid = binary_dilation(closed_grid, structure=structure, iterations=1)
    eroded_grid = binary_erosion(dilated_grid, structure=structure, iterations=1)
    boundary_grid = dilated_grid ^ eroded_grid

    boundary_indices = np.argwhere(boundary_grid)
    if len(boundary_indices) == 0:
        return points
    return boundary_indices / scale + mins


def _extract_annulus_boundaries_2d_cv(points, padding=0.02, img_size=400, dilate_iterations=1, close_iterations=1):
    points = np.asarray(points, dtype=float)
    if len(points) < 8:
        return None, None

    mins = points.min(axis=0) - padding
    maxs = points.max(axis=0) + padding
    max_span = float(np.max(maxs - mins))
    if max_span <= 1e-12:
        return None, None

    scale = img_size / max_span
    pixels = ((points - mins) * scale).astype(int)
    grid_shape = np.maximum(((maxs - mins) * scale).astype(int) + 5, 5)
    width = int(grid_shape[0])
    height = int(grid_shape[1])
    image = np.zeros((height, width), dtype=np.uint8)

    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    image[pixels[:, 1], pixels[:, 0]] = 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    if dilate_iterations > 0:
        image = cv2.dilate(image, kernel, iterations=int(dilate_iterations))
    if close_iterations > 0:
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))

    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return None, None

    hierarchy = hierarchy[0]
    best_outer_idx = None
    best_outer_area = -np.inf
    best_inner_idx = None
    for idx, h in enumerate(hierarchy):
        parent_idx = h[3]
        child_idx = h[2]
        if parent_idx != -1 or child_idx == -1:
            continue
        outer_area = cv2.contourArea(contours[idx])
        if outer_area <= best_outer_area:
            continue
        inner_candidates = []
        current_child = child_idx
        while current_child != -1:
            inner_candidates.append(current_child)
            current_child = hierarchy[current_child][0]
        if not inner_candidates:
            continue
        largest_inner_idx = max(inner_candidates, key=lambda child: cv2.contourArea(contours[child]))
        best_outer_idx = idx
        best_inner_idx = largest_inner_idx
        best_outer_area = outer_area

    if best_outer_idx is None or best_inner_idx is None:
        return None, None

    outer_contour = contours[best_outer_idx].reshape(-1, 2).astype(float)
    inner_contour = contours[best_inner_idx].reshape(-1, 2).astype(float)
    outer_boundary = np.column_stack((outer_contour[:, 0] / scale + mins[0], outer_contour[:, 1] / scale + mins[1]))
    inner_boundary = np.column_stack((inner_contour[:, 0] / scale + mins[0], inner_contour[:, 1] / scale + mins[1]))
    return outer_boundary, inner_boundary


def detect_boundary_voxels(points, padding=0.1, img_size=100, closing_iter=1):
    if len(points) < 4:
        return points

    grid, scale, xyz_min = _voxelize_points_3d(points, padding=padding, img_size=img_size)
    if grid is None:
        return points

    occ = grid.astype(bool)
    structure = np.ones((3, 3, 3), dtype=bool)

    if closing_iter > 0:
        occ = binary_closing(occ, structure=structure, iterations=closing_iter)

    inner = binary_erosion(occ, structure=structure, iterations=1)
    boundary = occ & (~inner)

    boundary_idx = np.argwhere(boundary)
    if len(boundary_idx) == 0:
        return points

    return boundary_idx / scale + xyz_min

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


def detect_raw_shell_3d(points):
    """Extract a shell from the raw occupied voxels without topology-altering closing."""
    if len(points) < 4:
        return points

    grid, scale, xyz_min = _voxelize_points_3d(points)
    if grid is None:
        return points

    structure = np.ones((3, 3, 3), dtype=bool)
    eroded_grid = binary_erosion(grid, structure=structure, iterations=1)
    shell_grid = grid ^ eroded_grid
    shell_indices = np.argwhere(shell_grid)
    if len(shell_indices) == 0:
        return points
    return shell_indices / scale + xyz_min


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


def dist_to_axis_aligned_torus_3d(points, center, r_major, r_minor, axis=2):
    r_major = max(float(r_major), 1e-6)
    r_minor = max(float(r_minor), 1e-6)
    p_local = points - center
    axis = int(axis)
    if axis == 0:
        rho = np.sqrt(p_local[:, 1] ** 2 + p_local[:, 2] ** 2)
        axial = p_local[:, 0]
    elif axis == 1:
        rho = np.sqrt(p_local[:, 0] ** 2 + p_local[:, 2] ** 2)
        axial = p_local[:, 1]
    else:
        rho = np.sqrt(p_local[:, 0] ** 2 + p_local[:, 1] ** 2)
        axial = p_local[:, 2]
    metric = np.sqrt((rho - r_major) ** 2 + axial ** 2) / r_minor
    return float(np.mean(np.abs(metric - 1)))


def dist_to_annulus_projection_2d(points_xy, center_xy, r_inner=None, r_outer=None, return_fit=False):
    points_xy = np.asarray(points_xy, dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    rho = np.sqrt(np.sum((points_xy - center_xy) ** 2, axis=1))
    if r_inner is None:
        r_inner = float(np.median(rho)) if len(rho) > 0 else 0.0
    if r_outer is None:
        r_outer = float(np.median(rho)) if len(rho) > 0 else 0.0
    r_inner = max(float(r_inner), 1e-6)
    r_outer = max(float(r_outer), r_inner + 1e-6)
    r_major = 0.5 * (r_inner + r_outer)
    r_minor = max(0.5 * (r_outer - r_inner), 1e-6)
    metric = np.abs(rho - r_major) / r_minor
    res = float(np.mean(np.abs(metric - 1)))
    if return_fit:
        return res, float(r_inner), float(r_outer)
    return res


def _fit_two_radii_iterative_1d(radii, lower_q=0.00, upper_q=1.0, max_iter=50):
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if len(radii) < 4:
        radius = float(np.median(radii)) if len(radii) > 0 else 0.0
        return radius, radius

    c1 = float(np.quantile(radii, lower_q))
    c2 = float(np.quantile(radii, upper_q))
    for _ in range(max_iter):
        d1 = np.abs(radii - c1)
        d2 = np.abs(radii - c2)
        mask1 = d1 <= d2
        mask2 = ~mask1
        if mask1.sum() == 0 or mask2.sum() == 0:
            break
        new_c1 = float(np.median(radii[mask1]))
        new_c2 = float(np.median(radii[mask2]))
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-10:
            c1, c2 = new_c1, new_c2
            break
        c1, c2 = new_c1, new_c2

    r_inner, r_outer = sorted((float(c1), float(c2)))
    return r_inner, r_outer


def _fit_two_radii_1d(radii, lower_q=0.01, upper_q=0.99, method="quantile", max_iter=50):
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if len(radii) < 4:
        radius = float(np.median(radii)) if len(radii) > 0 else 0.0
        return radius, radius

    method = str(method).lower()
    if method == "quantile":
        radii_sorted = np.sort(radii)
        idx_inner = _sorted_quantile_index(len(radii_sorted), lower_q)
        idx_outer = _sorted_quantile_index(len(radii_sorted), upper_q)
        r_inner = float(radii_sorted[idx_inner])
        r_outer = float(radii_sorted[idx_outer])
        return sorted((r_inner, r_outer))
    if method == "iterative":
        return _fit_two_radii_iterative_1d(radii, lower_q=lower_q, upper_q=upper_q, max_iter=max_iter)
    raise ValueError(f"Unsupported two-radius fitting method: {method}")


def _fit_axis_aligned_torus_details(points, cluster_points=None): # boundary for residual, cluster for parameter fit
    points = np.asarray(points, dtype=float)
    if cluster_points is None:
        cluster_points = points
    cluster_points = np.asarray(cluster_points, dtype=float)
    if len(points) < 4:
        center_ref = cluster_points if len(cluster_points) > 0 else points
        center = np.mean(center_ref, axis=0) if len(center_ref) > 0 else np.zeros(3, dtype=float)
        return {
            "center": center,
            "r_inner": np.nan,
            "r_outer": np.nan,
            "plane_boundary": np.zeros((0, 2), dtype=float),
            "plane_points": np.zeros((0, 2), dtype=float),
            "outer_boundary": None,
            "inner_boundary": None,
            "axis": 2,
            "plane_dims": (0, 1),
            "center_plane": np.zeros(2, dtype=float),
            "res": np.inf,
            "method": "degenerate",
        }

    best = None
    for axis in range(3):
        plane_dims = tuple(dim for dim in range(3) if dim != axis)
        plane_points = points[:, plane_dims] # projected boundary points
        cluster_plane_points = cluster_points[:, plane_dims] # projected cluster points for parameter fit
        center_plane = np.mean(cluster_plane_points, axis=0)
        rho_points = np.sqrt(np.sum((cluster_plane_points - center_plane) ** 2, axis=1))
        r_inner, r_outer = _fit_two_radii_1d(rho_points, lower_q=0.01, upper_q=0.99, method="quantile")
        outer_boundary, inner_boundary = _extract_annulus_boundaries_2d_cv(plane_points)
        if outer_boundary is not None and inner_boundary is not None and len(outer_boundary) >= 8 and len(inner_boundary) >= 8:
            plane_boundary = np.vstack((outer_boundary, inner_boundary))
            rho_outer = np.sqrt(np.sum((outer_boundary - center_plane) ** 2, axis=1))
            rho_inner = np.sqrt(np.sum((inner_boundary - center_plane) ** 2, axis=1))
            r_inner_res = float(np.median(rho_inner))
            r_outer_res = float(np.median(rho_outer))
            res = dist_to_annulus_projection_2d(
                plane_boundary,
                center_plane,
                r_inner=r_inner_res,
                r_outer=r_outer_res,
            )
            method = "cv2_ccomp"
        else:
            plane_boundary = _extract_boundary_2d(plane_points)
            if len(plane_boundary) < 8:
                plane_boundary = plane_points
            if len(plane_boundary) < 8:
                continue
            res = dist_to_annulus_projection_2d(
                plane_boundary,
                center_plane,
                r_inner=r_inner,
                r_outer=r_outer,
            )
            outer_boundary = None
            inner_boundary = None
            method = "fallback_boundary"

        center = np.mean(cluster_points, axis=0)
        center[plane_dims[0]] = center_plane[0]
        center[plane_dims[1]] = center_plane[1]
        candidate = {
            "center": center,
            "r_inner": r_inner,
            "r_outer": r_outer,
            "plane_boundary": plane_boundary,
            "plane_points": plane_points,
            "outer_boundary": outer_boundary,
            "inner_boundary": inner_boundary,
            "axis": axis,
            "plane_dims": plane_dims,
            "center_plane": center_plane,
            "res": res,
            "method": method,
        }
        if best is None or res < best["res"]:
            best = candidate

    if best is None:
        center_ref = cluster_points if len(cluster_points) > 0 else points
        center = np.mean(center_ref, axis=0)
        return {
            "center": center,
            "r_inner": np.nan,
            "r_outer": np.nan,
            "plane_boundary": np.zeros((0, 2), dtype=float),
            "plane_points": np.zeros((0, 2), dtype=float),
            "outer_boundary": None,
            "inner_boundary": None,
            "axis": 2,
            "plane_dims": (0, 1),
            "center_plane": np.zeros(2, dtype=float),
            "res": np.inf,
            "method": "failed",
        }

    return best


def fit_axis_aligned_torus_3d(points, cluster_points=None):
    details = _fit_axis_aligned_torus_details(points, cluster_points=cluster_points)
    return (
        details["center"],
        details["r_inner"],
        details["r_outer"],
        details["plane_boundary"],
        details["axis"],
        details["plane_dims"],
        details["res"],
    )


def plot_torus_projection_boundaries_3d(points, use_raw_shell=False, output_path=None, title=None):
    points = np.asarray(points, dtype=float)
    if use_raw_shell:
        points = detect_boundary_voxels(points)
    details = _fit_axis_aligned_torus_details(points)

    plane_points = np.asarray(details["plane_points"], dtype=float)
    outer_boundary = details["outer_boundary"]
    inner_boundary = details["inner_boundary"]
    plane_boundary = np.asarray(details["plane_boundary"], dtype=float)
    center_plane = np.asarray(details["center_plane"], dtype=float)
    r_inner = float(details["r_inner"])
    r_outer = float(details["r_outer"])
    plane_dims = tuple(details["plane_dims"])
    axis = int(details["axis"])

    fig, ax = plt.subplots(figsize=(7, 6))
    if len(plane_points) > 0:
        ax.scatter(
            plane_points[:, 0],
            plane_points[:, 1],
            s=6,
            c="lightgray",
            alpha=0.35,
            linewidths=0,
            label="Projected points",
        )
    if outer_boundary is not None and len(outer_boundary) > 0:
        ax.plot(
            outer_boundary[:, 0],
            outer_boundary[:, 1],
            color="#1f77b4",
            linewidth=1.6,
            label="Outer boundary",
        )
    if inner_boundary is not None and len(inner_boundary) > 0:
        ax.plot(
            inner_boundary[:, 0],
            inner_boundary[:, 1],
            color="#d62728",
            linewidth=1.6,
            label="Inner boundary",
        )
    if (outer_boundary is None or inner_boundary is None) and len(plane_boundary) > 0:
        ax.scatter(
            plane_boundary[:, 0],
            plane_boundary[:, 1],
            s=8,
            c="black",
            alpha=0.5,
            linewidths=0,
            label="Extracted boundary",
        )

    if np.all(np.isfinite(center_plane)) and np.isfinite(r_inner) and np.isfinite(r_outer):
        inner_circle = plt.Circle(center_plane, r_inner, fill=False, linestyle="--", linewidth=1.4, color="#d62728", alpha=0.9, label="Fitted inner radius")
        outer_circle = plt.Circle(center_plane, r_outer, fill=False, linestyle="--", linewidth=1.4, color="#1f77b4", alpha=0.9, label="Fitted outer radius")
        ax.add_patch(inner_circle)
        ax.add_patch(outer_circle)
        ax.scatter(center_plane[0], center_plane[1], c="black", marker="x", s=60, linewidths=2.0, label="Projected center")

    dim_labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
    ax.set_xlabel(dim_labels[plane_dims[0]], fontsize=16)
    ax.set_ylabel(dim_labels[plane_dims[1]], fontsize=16)
    if title is None:
        title = (
            f"Torus projection boundary extraction "
            f"(axis={axis}, plane={plane_dims}, method={details['method']})"
        )
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return details


def _torus_axis_frame(axis):
    axis = int(axis)
    if axis == 0:
        plane_dims = (1, 2)
        axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        e1 = np.array([0.0, 1.0, 0.0], dtype=float)
        e2 = np.array([0.0, 0.0, 1.0], dtype=float)
    elif axis == 1:
        plane_dims = (0, 2)
        axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e2 = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        plane_dims = (0, 1)
        axis_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e2 = np.array([0.0, 1.0, 0.0], dtype=float)
    return axis_vec, e1, e2, plane_dims


def sample_fitted_torus_surface_3d(points, use_raw_shell=True, n_major=240, n_minor=120):
    points = np.asarray(points, dtype=float)
    fit_points = detect_raw_shell_3d(points) if use_raw_shell else points
    details = _fit_axis_aligned_torus_details(fit_points)

    center = np.asarray(details["center"], dtype=float)
    axis = int(details["axis"])
    r_inner = float(details["r_inner"])
    r_outer = float(details["r_outer"])
    if not np.all(np.isfinite(center)) or not np.isfinite(r_inner) or not np.isfinite(r_outer) or r_outer <= r_inner:
        raise ValueError("Failed to fit a valid torus from the provided points.")

    r_major = 0.5 * (r_inner + r_outer)
    r_minor = 0.5 * (r_outer - r_inner)
    axis_vec, e1, e2, plane_dims = _torus_axis_frame(axis)

    u = np.linspace(0.0, 2.0 * np.pi, int(max(16, n_major)), endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, int(max(16, n_minor)), endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    cos_u = np.cos(U)
    sin_u = np.sin(U)
    cos_v = np.cos(V)
    sin_v = np.sin(V)

    ring_dir = cos_u[..., None] * e1 + sin_u[..., None] * e2
    tube_offset = (r_major + r_minor * cos_v)[..., None] * ring_dir + (r_minor * sin_v)[..., None] * axis_vec
    surface = center + tube_offset

    details = dict(details)
    details["r_major"] = r_major
    details["r_minor"] = r_minor
    details["axis_vec"] = axis_vec
    details["plane_dims"] = plane_dims
    details["surface_points"] = surface.reshape(-1, 3)
    details["surface_grid_x"] = surface[..., 0]
    details["surface_grid_y"] = surface[..., 1]
    details["surface_grid_z"] = surface[..., 2]
    return details


def plot_fitted_torus_surface_3d(points, use_raw_shell=True, n_major=180, n_minor=90, show_input=True, output_path=None, title=None):
    details = sample_fitted_torus_surface_3d(
        points,
        use_raw_shell=use_raw_shell,
        n_major=n_major,
        n_minor=n_minor,
    )
    fit_points = detect_raw_shell_3d(points) if use_raw_shell else np.asarray(points, dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if show_input:
        ax.scatter(
            fit_points[:, 0],
            fit_points[:, 1],
            fit_points[:, 2],
            s=3,
            c="lightgray",
            alpha=0.18,
            linewidths=0,
            label="Input shell points",
        )

    ax.plot_surface(
        details["surface_grid_x"],
        details["surface_grid_y"],
        details["surface_grid_z"],
        rstride=1,
        cstride=1,
        color="#4c72b0",
        alpha=0.85,
        linewidth=0,
        antialiased=True,
    )
    center = np.asarray(details["center"], dtype=float)
    ax.scatter(center[0], center[1], center[2], c="red", marker="x", s=80, linewidths=2.0, label="Fitted center")
    if title is None:
        title = (
            f"Fitted thin torus surface "
            f"(axis={details['axis']}, R={details['r_major']:.4f}, r={details['r_minor']:.4f})"
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"$x_1$", fontsize=15)
    ax.set_ylabel(r"$x_2$", fontsize=15)
    ax.set_zlabel(r"$x_3$", fontsize=15)
    ax.set_box_aspect(
        (
            np.ptp(details["surface_grid_x"]),
            np.ptp(details["surface_grid_y"]),
            np.ptp(details["surface_grid_z"]),
        )
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return details


def _normalize_margin_3d(margin):
    if np.isscalar(margin):
        value = int(margin)
        return (value, value, value)
    if len(margin) != 3:
        raise ValueError("boundary_margin for 3D detection must be an int or a length-3 tuple/list.")
    return tuple(int(v) for v in margin)


def _cluster_points_3d(grad, X, Y, Z, para_abs, para_grad, eps, mini_samples, S_num, eps_mode="auto", boundary_margin=5):
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

    margin_xyz = _normalize_margin_3d(boundary_margin)
    valid_mask = np.ones(len(idx[0]), dtype=bool)
    shape = grad.shape
    for dim in range(3):
        margin = margin_xyz[dim]
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


def _classify_cluster_3d(cluster_pts, boundary_points, residual_threshold=0.05):
    geometry_points = boundary_points if len(boundary_points) >= 4 else cluster_pts
    center = np.mean(cluster_pts, axis=0)

    span_x = np.max(cluster_pts[:, 0]) - np.min(cluster_pts[:, 0])
    span_y = np.max(cluster_pts[:, 1]) - np.min(cluster_pts[:, 1])
    span_z = np.max(cluster_pts[:, 2]) - np.min(cluster_pts[:, 2])
    lx, ly, lz = span_x, span_y, span_z
    half_lengths = 0.5 * np.asarray([span_x, span_y, span_z], dtype=float)
    l_dims = np.asarray([span_x, span_y, span_z], dtype=float)
    l_max = float(np.max(l_dims))
    l_min = float(max(np.min(l_dims), 1e-12))
    aspect_ratio = l_max / l_min

    dists = np.linalg.norm(cluster_pts- center, axis=1)
    dists_sorted = np.sort(dists)
    idx_min = _sorted_quantile_index(len(dists_sorted), 0.01)
    idx_max = _sorted_quantile_index(len(dists_sorted), 0.99)
    min_dist_to_center = float(np.min(dists_sorted))
    max_dist_to_center = float(np.max(dists_sorted))
    r_min = float(dists_sorted[idx_min])
    r_max = float(dists_sorted[idx_max])
    (
        torus_center,
        r_inner_fit,
        r_outer_fit,
        torus_projection_points,
        torus_axis,
        torus_plane_dims,
        torus_projection_res,
    ) = fit_axis_aligned_torus_3d(geometry_points, cluster_points=cluster_pts)
    r_major = 0.5 * (r_inner_fit + r_outer_fit)
    r_tube = 0.5 * (r_outer_fit - r_inner_fit)

    residual_points = geometry_points
    # residual_points_tori = geometry_points
    res_rect = dist_to_axis_aligned_box(residual_points, center, lx, ly, lz)
    res_ellip = dist_to_axis_aligned_ellipsoid_3d(residual_points, center, lx, ly, lz)
    res_tori_3d = np.inf
    res_tori_projection = np.inf
    if np.isfinite(torus_projection_res) and r_tube > 1e-12 and r_outer_fit > r_inner_fit:
        center_plane = torus_center[list(torus_plane_dims)]
        # residual_points_tori = _select_torus_residual_points_3d(
        #     geometry_points,
        #     torus_center,
        #     torus_plane_dims,
        #     r_inner_fit,
        #     r_outer_fit,
        # )
        res_tori_3d = dist_to_axis_aligned_torus_3d(
            geometry_points,
            torus_center,
            r_major,
            r_tube,
            axis=torus_axis,
        )
        res_tori_projection = float(torus_projection_res)
    res_tori = min(res_tori_3d, res_tori_projection)
   

    if min(res_rect, res_ellip, res_tori) > residual_threshold:
        shape_type = "general"
        final_params = {"center": center}
    elif res_rect <= min(res_ellip, res_tori):
        shape_type = "Rectangle"
        final_params = {"center": center, "Lx": half_lengths[0], "Ly": half_lengths[1], "Lz": half_lengths[2]}
    elif res_tori <= min(res_rect, res_ellip):
        shape_type = "Donut"
        final_params = {
            "center": torus_center,
            "R_inner": r_inner_fit,
            "R_outer": r_outer_fit,
            "R_major": r_major,
            "r_minor": r_tube,
            "r_min": r_inner_fit,
            "r_max": r_outer_fit,
            "r_inner": r_tube,
            "r": 2.0 * r_tube,
        }
    else:
        shape_type = "Ellipsoid"
        final_params = {
            "center": center,
            "a": lx / 2,
            "b": ly / 2,
            "c": lz / 2,
            "radius": float(dists_sorted[idx_max]),
        }

    diagnostics = {
        "Lx": float(lx),
        "Ly": float(ly),
        "Lz": float(lz),
        "aspect_ratio": float(aspect_ratio),
        "res_rect": float(res_rect),
        "res_ellip": float(res_ellip),
        "res_tori": float(res_tori),
        "res_tori_3d": float(res_tori_3d),
        "res_tori_projection": float(res_tori_projection),
        "residual_threshold": float(residual_threshold),
        "boundary_point_count": int(len(residual_points)),
        "torus_point_count": int(len(torus_projection_points)),
        "torus_center": np.asarray(torus_center, dtype=float),
        "torus_axis": int(torus_axis),
        "torus_plane_dims": tuple(int(dim) for dim in torus_plane_dims),
        "dist_01": float(r_min),
        "dist_99": float(r_max),
        "min_dist_to_center": float(min_dist_to_center),
        "max_dist_to_center": float(max_dist_to_center),
    }

    return center, lx, ly, lz, aspect_ratio, res_rect, res_ellip, res_tori, shape_type, final_params, diagnostics


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
    boundary_margin=0,
    residual_threshold=0.05,
    t_aspect_ratio=1.2,
    t_cv=0.7,
):
    del t_aspect_ratio  # Kept only for backward-compatible notebook calls.
    segmented_results = _cluster_points_3d(
        grad,
        X,
        Y,
        Z,
        para_abs,
        para_grad,
        eps,
        mini_samples,
        S_num,
        eps_mode=eps_mode,
        boundary_margin=boundary_margin,
    )
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
        center, lx, ly, lz, aspect_ratio, res_rect, res_ellip, res_tori, shape_type, final_params, diagnostics = _classify_cluster_3d(
            cluster_pts,
            boundary_points,
            residual_threshold=residual_threshold,
        )

        basis_func_type = "Sigmoid"
        cv = np.nan
        if S_num is not None:
            std_s = np.std(s_vals)
            mean_s = np.mean(np.abs(s_vals))
            cv = std_s / mean_s if mean_s > 1e-12 else 0.0
            print(f"  CV of S_num: {cv:.3f}")
            if cv >= t_cv:
                basis_func_type = "Exp"

        print(f"Cluster {cluster_id}: AR={aspect_ratio:.2f}")
        print(f"  Res_Rect={res_rect:.6f} vs Res_Ellip={res_ellip:.6f} vs Res_Tori={res_tori:.6f}")
        print(f"  center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  Lx={lx:.6f}, Ly={ly:.6f}, Lz={lz:.6f}")
        print(
            f"  dist_01={diagnostics['dist_01']:.6f}, "
            f"dist_99={diagnostics['dist_99']:.6f}"
        )
        if shape_type == "Rectangle":
            print(
                f"  half-lengths=({final_params['Lx']:.6f}, "
                f"{final_params['Ly']:.6f}, {final_params['Lz']:.6f})"
            )
        elif shape_type == "Ellipsoid":
            print(
                f"  half-axes=({final_params['a']:.6f}, "
                f"{final_params['b']:.6f}, {final_params['c']:.6f})"
            )
        if shape_type in {"Donut", "tori"}:
            print(
                f"  R_inner={final_params['R_inner']:.6f}, "
                f"R_outer={final_params['R_outer']:.6f}"
            )
            print(
                f"  R_major={final_params['R_major']:.6f}, "
                f"r_minor={final_params['r_minor']:.6f}"
            )
            print(
                f"  torus_axis={diagnostics['torus_axis']}, "
                f"torus_plane_dims={diagnostics['torus_plane_dims']}"
            )
        elif shape_type == "general":
            print(
                f"  residual_threshold={diagnostics['residual_threshold']:.6f}, "
                f"boundary_points={diagnostics['boundary_point_count']}"
            )
        print(f"  -> Decision: {shape_type} ({basis_func_type})")

        results.append(
            {
                "cluster_id": cluster_id,
                "shape": shape_type,
                "params": final_params,
                "profile": basis_func_type,
                "boundary_points": boundary_points,
                "diagnostics": diagnostics,
                "aspect_ratio": diagnostics["aspect_ratio"],
                "res_rect": diagnostics["res_rect"],
                "res_ellip": diagnostics["res_ellip"],
                "res_tori": diagnostics["res_tori"],
                "res_tori_3d": diagnostics["res_tori_3d"],
                "res_tori_projection": diagnostics["res_tori_projection"],
                "Lx": diagnostics["Lx"],
                "Ly": diagnostics["Ly"],
                "Lz": diagnostics["Lz"],
                "dist_01": diagnostics["dist_01"],
                "dist_99": diagnostics["dist_99"],
                "cv": float(cv) if np.isfinite(cv) else np.nan,
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

        if shape_type == "Rectangle":
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


def print_detected_results_3d(results):
    if not results:
        print("No detected 3D clusters.")
        return

    for item in results:
        cluster_id = int(item.get("cluster_id", -1))
        shape = str(item.get("shape", "unknown"))
        profile = str(item.get("profile", "unknown"))
        params = item.get("params", {})
        diagnostics = item.get("diagnostics", {})
        center = np.asarray(params.get("center", diagnostics.get("torus_center", [np.nan, np.nan, np.nan])), dtype=float)

        aspect_ratio = float(item.get("aspect_ratio", diagnostics.get("aspect_ratio", np.nan)))
        res_rect = float(item.get("res_rect", diagnostics.get("res_rect", np.nan)))
        res_ellip = float(item.get("res_ellip", diagnostics.get("res_ellip", np.nan)))
        res_tori = float(item.get("res_tori", diagnostics.get("res_tori", np.nan)))
        lx = float(item.get("Lx", diagnostics.get("Lx", np.nan)))
        ly = float(item.get("Ly", diagnostics.get("Ly", np.nan)))
        lz = float(item.get("Lz", diagnostics.get("Lz", np.nan)))
        dist_01 = float(item.get("dist_01", diagnostics.get("dist_01", np.nan)))
        dist_99 = float(item.get("dist_99", diagnostics.get("dist_99", np.nan)))
        cv = item.get("cv", np.nan)

        print(f"Cluster {cluster_id}: AR={aspect_ratio:.2f}")
        print(f"  Res_Rect={res_rect:.6f} vs Res_Ellip={res_ellip:.6f} vs Res_Tori={res_tori:.6f}")
        print(f"  center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  Lx={lx:.6f}, Ly={ly:.6f}, Lz={lz:.6f}")
        print(f"  dist_01={dist_01:.6f}, dist_99={dist_99:.6f}")
        if np.isfinite(cv):
            print(f"  CV of S_num: {cv:.3f}")

        if shape == "Rectangle":
            print(
                f"  half-lengths=({float(params.get('Lx', np.nan)):.6f}, "
                f"{float(params.get('Ly', np.nan)):.6f}, {float(params.get('Lz', np.nan)):.6f})"
            )
        elif shape == "Ellipsoid":
            print(
                f"  half-axes=({float(params.get('a', np.nan)):.6f}, "
                f"{float(params.get('b', np.nan)):.6f}, {float(params.get('c', np.nan)):.6f})"
            )
        elif shape in {"Donut", "tori"}:
            print(
                f"  R_inner={float(params.get('R_inner', np.nan)):.6f}, "
                f"R_outer={float(params.get('R_outer', np.nan)):.6f}"
            )
            print(
                f"  R_major={float(params.get('R_major', np.nan)):.6f}, "
                f"r_minor={float(params.get('r_minor', np.nan)):.6f}"
            )
            print(
                f"  torus_axis={diagnostics.get('torus_axis', None)}, "
                f"torus_plane_dims={diagnostics.get('torus_plane_dims', None)}"
            )
        else:
            print(
                f"  residual_threshold={float(diagnostics.get('residual_threshold', np.nan)):.6f}, "
                f"boundary_points={int(diagnostics.get('boundary_point_count', 0))}"
            )

        print(f"  -> Decision: {shape} ({profile})")


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


def _latex_param_text_3d(result, precision=2):
    shape = str(result.get("shape", "")).lower()
    params = result.get("params", {})
    if shape == "rectangle":
        return (
            rf"$\boldsymbol{{L}}$=("
            rf"{_format_sci(params.get('Lx', np.nan), precision)}, "
            rf"{_format_sci(params.get('Ly', np.nan), precision)}, "
            rf"{_format_sci(params.get('Lz', np.nan), precision)})"
        )
    if shape == "ellipsoid":
        return (
            rf"$\boldsymbol{{L}}$=("
            rf"{_format_sci(params.get('a', np.nan), precision)}, "
            rf"{_format_sci(params.get('b', np.nan), precision)}, "
            rf"{_format_sci(params.get('c', np.nan), precision)})"
        )
    if shape == "donut":
        return (
            rf"$(R,r)$=("
            rf"{_format_sci(params.get('R_major', np.nan), precision)}, "
            rf"{_format_sci(params.get('r_minor', np.nan), precision)})"
        )
    return "-"


def _latex_shape_name_3d(shape):
    shape = str(shape).strip().lower()
    if shape == "donut":
        return r"\emph{Tori}"
    if shape == "rectangle":
        return r"\emph{Rectangle}"
    if shape == "ellipsoid":
        return r"\emph{Ellipsoid}"
    if shape == "general":
        return r"\emph{general}"
    return rf"\emph{{{shape}}}"


def _latex_param_text_3d_detection(result, precision=2):
    shape = str(result.get("shape", "")).lower()
    params = result.get("params", {})
    if shape == "rectangle":
        return (
            rf"$\boldsymbol{{L}}$=("
            rf"{_format_sci(params.get('Lx', np.nan), precision)}, "
            rf"{_format_sci(params.get('Ly', np.nan), precision)}, "
            rf"{_format_sci(params.get('Lz', np.nan), precision)})"
        )
    if shape == "ellipsoid":
        return (
            rf"$\boldsymbol{{L}}$=("
            rf"{_format_sci(params.get('a', np.nan), precision)}, "
            rf"{_format_sci(params.get('b', np.nan), precision)}, "
            rf"{_format_sci(params.get('c', np.nan), precision)})"
        )
    if shape == "donut":
        return (
            r"\begin{tabular}{@{}c@{}} "
            + "\n"
            + rf"$\mathbf{{r_{{\text{{min}}}}={_format_sci(params.get('r_min', params.get('R_inner', np.nan)), precision)}}}$ \\"
            + "\n"
            + rf"$\mathbf{{r_{{\text{{max}}}}={_format_sci(params.get('r_max', params.get('R_outer', np.nan)), precision)}}}$"
            + "\n"
            + r"\end{tabular}"
        )
    return "-"


def print_latex_shape_detection_rows_3d(results, example_name='3-D "donut"', start_cluster=1, precision=2, include_header=True):
    rows = []
    if include_header:
        rows.append("LaTeX table rows:")
        rows.append(
            r"{\emph{Ex}} & Cluster $k$ & $\boldsymbol{c}_k$ & $r_k$ or $\boldsymbol{L}_k$ & "
            r"$\mathcal{E}_{\text{rect}}^{(k)}$ & $\mathcal{E}_{\text{ellip}}^{(k)}$ & "
            r"$\mathcal{E}_{\text{tori}}^{(k)}$ & $\mathcal{T}_k$ & $\sigma_k$ \\"
        )

    n_rows = len(results)
    for row_idx, result in enumerate(results):
        params = result.get("params", {})
        diagnostics = result.get("diagnostics", {})
        center = np.asarray(params.get("center", diagnostics.get("torus_center", [np.nan, np.nan, np.nan])), dtype=float)
        ex_cell = rf"\multirow{{{n_rows}}}{{*}}{{{example_name}}}" if row_idx == 0 and n_rows > 1 else (example_name if n_rows == 1 else "")
        row = (
            f"{ex_cell} & {start_cluster + row_idx} & "
            f"({_format_sci(center[0], precision)}, {_format_sci(center[1], precision)}, {_format_sci(center[2], precision)}) & "
            f"{_latex_param_text_3d_detection(result, precision)} & "
            f"{_format_fixed(result.get('res_rect', np.nan), 3)} & "
            f"{_format_fixed(result.get('res_ellip', np.nan), 3)} & "
            f"{_format_fixed(result.get('res_tori', np.nan), 3)} & "
            f"{_latex_shape_name_3d(result.get('shape', 'unknown'))} & "
            f"{result.get('profile', 'unknown')} \\\\"
        )
        rows.append(row)

    print("\n".join(rows))


def print_latex_result_rows_3d(results, example_name="Ex 4.8", start_cluster=1, precision=2, include_header=False):
    rows = []
    if include_header:
        header = (
            r"{\emph{Ex}} & Cluster $k$ & $\boldsymbol{c}_k$ & "
            r"$r_k$ or $\boldsymbol{L}_k$ & "
            r"$\mathcal{E}_{\mathrm{rect}}^{(k)}$ & "
            r"$\mathcal{E}_{\mathrm{ellip}}^{(k)}$ & "
            r"$\mathcal{E}_{\mathrm{tori}}^{(k)}$ & "
            r"$\mathrm{AR}_k$ & $\mathrm{CV}_k$ & $\mathcal{T}_k$ & $\sigma_k$ \\"
        )
        rows.append("LaTeX table rows:")
        rows.append(header)
    n_rows = len(results)
    for row_idx, result in enumerate(results):
        params = result.get("params", {})
        center = np.asarray(params.get("center", [np.nan, np.nan, np.nan]), dtype=float)
        cv = result.get("cv", result.get("diagnostics", {}).get("cv", np.nan))
        ex_cell = rf"\multirow{{{n_rows}}}{{*}}{{{example_name}}}" if row_idx == 0 and n_rows > 1 else (example_name if n_rows == 1 else "")
        row = (
            f"{ex_cell} & {start_cluster + row_idx} "
            f"& ({_format_sci(center[0], precision)}, {_format_sci(center[1], precision)}, {_format_sci(center[2], precision)}) "
            f"& {_latex_param_text_3d(result, precision)} "
            f"& {_format_fixed(result.get('res_rect', np.nan), 4)} "
            f"& {_format_fixed(result.get('res_ellip', np.nan), 4)} "
            f"& {_format_fixed(result.get('res_tori', np.nan), 4)} "
            f"& {_format_fixed(result.get('aspect_ratio', np.nan), 2)} "
            f"& {_format_fixed(cv, 3)} "
            f"& \\textbf{{{result.get('shape', '-')}}} "
            f"& {result.get('profile', '-')} \\\\"
        )
        rows.append(row)
    print("\n".join(rows))
    return rows
