import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

try:
    import bound_detect
except Exception:  # pragma: no cover - optional example dependency
    bound_detect = None


CJET_CMAP = LinearSegmentedColormap.from_list(
    "cjet",
    plt.get_cmap("jet")(np.linspace(0, 1, 256)),
    N=256,
)
PAPER_SEQUENTIAL_CMAP = CJET_CMAP


def _get_bound_detect_module():
    """Import bound_detect lazily because notebooks often mutate sys.path before plotting."""
    global bound_detect
    if bound_detect is not None:
        return bound_detect
    try:
        import importlib

        bound_detect = importlib.import_module("bound_detect")
    except Exception:
        bound_detect = None
    return bound_detect


def kidney_curve(x, y, h, k, a):
    x_shifted = x - h
    y_shifted = y - k
    if a == 0:
        a = 1e-9
    term1 = (x_shifted**2 + y_shifted**2 - 4 * a**2) ** 3
    term2 = 108 * a**4 * y_shifted**2
    return term1 - term2


def _resolve_sequential_cmap(cmap=None):
    if cmap is None:
        return PAPER_SEQUENTIAL_CMAP
    if isinstance(cmap, str) and cmap.lower() == "cjet":
        return CJET_CMAP
    return cmap


def draw(
    X,
    Y,
    data,
    *args,
    output_directory=None,
    output_filename=None,
    cmap=None,
    label_size=30,
    tick_size=18,
    colorbar_tick_size=18,
    tick_format="{:g}",
):
    """Draw a 2D contour map with legacy positional-call compatibility."""
    if len(args) == 0:
        pass
    elif len(args) == 2:
        output_directory, output_filename = args
    elif len(args) == 3:
        cmap, output_directory, output_filename = args
    else:
        raise TypeError("draw() supports (X,Y,data), (X,Y,data,outdir,file), or (X,Y,data,cmap,outdir,file)")

    if output_directory is None:
        output_directory = "."
    if output_filename is None:
        output_filename = "plot.png"
    cmap = _resolve_sequential_cmap(cmap)

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)

    plt.figure(figsize=(8, 6))
    contour_fill = plt.contourf(
        X,
        Y,
        data,
        levels=20,
        cmap=cmap,
        alpha=1,
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
    )
    ax = plt.gca()

    cbar = plt.colorbar(contour_fill, shrink=1, aspect=12)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)

    plt.xlabel(r"$x_1$", fontsize=label_size)
    plt.ylabel(r"$x_2$", fontsize=label_size)
    plt.tick_params(axis="both", which="major", labelsize=tick_size)
    formatter = FuncFormatter(lambda value, _: tick_format.format(value))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_facecolor("white")
    plt.tight_layout()

    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {full_output_path}")
    except Exception as exc:
        print(f"Error saving plot to {full_output_path}: {exc}")

    plt.show()
    plt.close()


def show_saved_images(image_paths, titles=None, output_directory=".", output_filename=None, figsize_per_image=(7.0, 5.8)):
    """Display one or more saved image files in a single figure."""
    normalized = []
    for i, item in enumerate(image_paths):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            path, title = item
        else:
            path = item
            title = titles[i] if titles is not None and i < len(titles) else os.path.basename(str(path))
        if os.path.exists(path):
            normalized.append((path, title))

    if not normalized:
        print("No saved images were found.")
        return None

    fig, axes = plt.subplots(1, len(normalized), figsize=(figsize_per_image[0] * len(normalized), figsize_per_image[1]))
    axes = np.atleast_1d(axes)
    for ax, (path, title) in zip(axes, normalized):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    if output_filename:
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        elif not output_directory:
            output_directory = "."
        fig.savefig(os.path.join(output_directory, output_filename), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig


class _PickledCell:
    """Minimal stand-in for mesh.Cell when loading saved cells for plotting."""


def load_cells_from_pickle(cells_path):
    """Load saved adaptive cells without requiring the full training stack."""
    import pickle

    class _CellUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "mesh" and name == "Cell":
                return _PickledCell
            return super().find_class(module, name)

    with open(cells_path, "rb") as file_obj:
        return _CellUnpickler(file_obj).load()


def _collect_leaf_cells(cells):
    leaf_cells = []
    for cell in cells:
        children = getattr(cell, "children", [])
        if not children:
            leaf_cells.append(cell)
        else:
            leaf_cells.extend(_collect_leaf_cells(children))
    return leaf_cells


def _format_plain_axis(ax, xlim, ylim, xlabel_size=32, ylabel_size=32, tick_size=18):
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.2)
    ax.set_xlabel(r"$x_1$", fontsize=xlabel_size)
    ax.set_ylabel(r"$x_2$", fontsize=ylabel_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)


def _draw_cell_mesh_on_axis(ax, cells, linewidth=0.8):
    leaf_cells = _collect_leaf_cells(cells)
    for cell in leaf_cells:
        ax.add_patch(
            patches.Rectangle(
                (cell.x0, cell.y0),
                cell.size,
                cell.size,
                linewidth=linewidth,
                edgecolor="black",
                facecolor="none",
                alpha=0.8,
            )
        )
    return leaf_cells


def plot_saved_cell_mesh(
    cells_path,
    output_directory=".",
    output_filename="visual_cells.pdf",
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    figsize=(7.5, 6.5),
    linewidth=0.8,
):
    """Plot the saved adaptive 2D cells."""
    cells = load_cells_from_pickle(cells_path)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_cell_mesh_on_axis(ax, cells, linewidth=linewidth)
    _format_plain_axis(ax, xlim, ylim, xlabel_size=40, ylabel_size=40, tick_size=22)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if output_filename:
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        elif not output_directory:
            output_directory = "."
        fig.savefig(os.path.join(output_directory, output_filename), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig


def _load_noise_model_centers(noise_dir, model_prefix="noise_model_sig_"):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("torch is required to load saved noise models.") from exc

    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"noise_dir does not exist: {noise_dir}")

    model_paths = sorted(
        os.path.join(noise_dir, name)
        for name in os.listdir(noise_dir)
        if name.startswith(model_prefix) and name.endswith(".pth")
    )
    centers_by_model = []
    for path in model_paths:
        state = torch.load(path, map_location="cpu")
        b1 = -state["hidden_layer_1.0.bias"].detach().cpu().numpy()
        b2 = -state["hidden_layer_2.0.bias"].detach().cpu().numpy()
        centers_by_model.append((path, np.column_stack((b1, b2))))
    return centers_by_model


def detect_boundaries_from_solution(
    S_num,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    para_abs=3.0,
    para_grad=3.0,
    eps=5.0,
    mini_samples=20,
    min_cluster_size=200,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
):
    """Extract detected boundary curves from a saved 2D numerical solution."""
    S_num = np.asarray(S_num, dtype=float)
    if S_num.ndim != 2:
        raise ValueError(f"S_num must be a 2D array, got shape {S_num.shape}")

    grad_y, grad_x = np.gradient(S_num)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    x = np.linspace(xlim[0], xlim[1], S_num.shape[0])
    y = np.linspace(ylim[0], ylim[1], S_num.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")

    bd = _get_bound_detect_module()
    if bd is not None and all(hasattr(bd, name) for name in ("_cluster_points", "_filter_clusters", "detect_boundary")):
        segmented_results = bd._cluster_points(  # pylint: disable=protected-access
            grad,
            X,
            Y,
            para_abs,
            para_grad,
            eps,
            mini_samples,
            S_num,
            eps_mode=eps_mode,
        )
        segmented_results = bd._filter_clusters(  # pylint: disable=protected-access
            segmented_results,
            min_cluster_size=min_cluster_size,
            min_relative_abs_s=min_relative_abs_s,
            min_relative_grad=min_relative_grad,
            keep_top_k=keep_top_k,
        )

        detected_results = []
        for cluster_id, cluster in enumerate(segmented_results):
            points = cluster["points"]
            if len(points) == 0:
                continue
            detected_results.append(
                {
                    "cluster_id": cluster_id,
                    "points": points,
                    "boundary_points": bd.detect_boundary(points),
                }
            )
        return detected_results

    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN

    h = max((xlim[1] - xlim[0]) / max(S_num.shape[0] - 1, 1), (ylim[1] - ylim[0]) / max(S_num.shape[1] - 1, 1))
    eps_dbscan = float(eps) * h if eps_mode == "grid_scaled" or (eps_mode == "auto" and float(eps) > 1.0) else float(eps)
    mask = (grad > np.max(grad) / para_grad) & (np.abs(S_num) > np.max(np.abs(S_num)) / para_abs)
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return []

    margin = 10
    valid = (
        (idx[0] >= margin)
        & (idx[0] <= S_num.shape[0] - margin)
        & (idx[1] > margin)
        & (idx[1] <= S_num.shape[1] - margin)
    )
    idx = (idx[0][valid], idx[1][valid])
    points = np.column_stack((X[idx], Y[idx]))
    s_vals = S_num[idx]
    g_vals = grad[idx]
    if len(points) == 0:
        return []

    if len(points) < 10:
        segmented_results = [{"points": points, "S_vals": s_vals, "g_vals": g_vals}]
    else:
        labels = DBSCAN(eps=eps_dbscan, min_samples=mini_samples).fit(points).labels_
        segmented_results = [
            {"points": points[labels == label], "S_vals": s_vals[labels == label], "g_vals": g_vals[labels == label]}
            for label in sorted(set(labels) - {-1})
        ]

    if min_cluster_size is not None:
        segmented_results = [cluster for cluster in segmented_results if len(cluster["points"]) >= min_cluster_size]
    if keep_top_k is not None and len(segmented_results) > keep_top_k:
        segmented_results = sorted(segmented_results, key=lambda cluster: len(cluster["points"]), reverse=True)[:keep_top_k]

    detected_results = []
    for cluster_id, cluster in enumerate(segmented_results):
        points = cluster["points"]
        if len(points) < 3:
            continue
        hull = ConvexHull(points)
        detected_results.append(
            {
                "cluster_id": cluster_id,
                "points": points,
                "boundary_points": points[hull.vertices],
            }
        )
    if detected_results:
        return detected_results

    from scipy.ndimage import binary_closing, binary_erosion, label

    support = np.abs(S_num) > np.max(np.abs(S_num)) / para_abs
    support = binary_closing(support, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labeled, num_features = label(support)
    if num_features == 0:
        return []

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    component_ids = [idx for idx in np.argsort(component_sizes)[::-1] if component_sizes[idx] > 0]
    if keep_top_k is not None:
        component_ids = component_ids[:keep_top_k]

    support_results = []
    for cluster_id, component_id in enumerate(component_ids):
        if min_cluster_size is not None and component_sizes[component_id] < min_cluster_size:
            continue
        component = labeled == component_id
        boundary_mask = component & ~binary_erosion(component, structure=np.ones((3, 3), dtype=bool), iterations=1)
        rows, cols = np.where(boundary_mask)
        if rows.size < 3:
            continue
        boundary = np.column_stack((x[rows], y[cols]))
        center = boundary.mean(axis=0)
        angles = np.arctan2(boundary[:, 1] - center[1], boundary[:, 0] - center[0])
        support_results.append(
            {
                "cluster_id": cluster_id,
                "points": np.column_stack(np.where(component)),
                "boundary_points": boundary[np.argsort(angles)],
            }
        )
    return support_results


def _resolve_plot_boundaries(
    noise_dir="./noise=5%",
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    detected_results=None,
    detected_boundaries=None,
    boundary_filename="detected_boundaries.npz",
    S_num=None,
    detect_boundary_from_solution=False,
    para_abs=3.0,
    para_grad=3.0,
    eps=5.0,
    mini_samples=20,
    min_cluster_size=200,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
):
    boundaries = []
    if detected_results is not None:
        for item in detected_results:
            if isinstance(item, dict) and "boundary_points" in item:
                boundaries.append(np.asarray(item["boundary_points"], dtype=float))
    if detected_results is None and detected_boundaries is None and boundary_filename:
        boundary_path = os.path.join(noise_dir, boundary_filename)
        if os.path.exists(boundary_path):
            with np.load(boundary_path) as boundary_data:
                def _boundary_key(name):
                    try:
                        return int(str(name).split("_")[-1])
                    except Exception:
                        return str(name)

                for name in sorted(boundary_data.files, key=_boundary_key):
                    boundaries.append(np.asarray(boundary_data[name], dtype=float))
    if not boundaries and detect_boundary_from_solution and S_num is not None:
        detected_results_from_solution = detect_boundaries_from_solution(
            S_num,
            xlim=xlim,
            ylim=ylim,
            para_abs=para_abs,
            para_grad=para_grad,
            eps=eps,
            mini_samples=mini_samples,
            min_cluster_size=min_cluster_size,
            min_relative_abs_s=min_relative_abs_s,
            min_relative_grad=min_relative_grad,
            keep_top_k=keep_top_k,
            eps_mode=eps_mode,
        )
        boundaries.extend(np.asarray(item["boundary_points"], dtype=float) for item in detected_results_from_solution)
        if boundary_filename and boundaries:
            np.savez(
                os.path.join(noise_dir, boundary_filename),
                **{f"cluster_{i}": boundary for i, boundary in enumerate(boundaries)},
            )
    if detected_boundaries is not None:
        boundary_iter = detected_boundaries.values() if isinstance(detected_boundaries, dict) else detected_boundaries
        for boundary in boundary_iter:
            boundaries.append(np.asarray(boundary, dtype=float))
    return boundaries


def _draw_noise_centers_on_axis(
    ax,
    centers_by_model,
    boundaries=None,
    boundary_color="g",
    boundary_linewidth=1.6,
    point_size=8,
    legend_fontsize=9,
    show_legend=True,
):
    boundaries = [] if boundaries is None else boundaries
    for i, boundary in enumerate(boundaries):
        if boundary.ndim != 2 or boundary.shape[1] < 2 or len(boundary) == 0:
            continue
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=boundary_color,
            linewidth=boundary_linewidth,
            linestyle="-",
            label="Detected boundary" if i == 0 else None,
            zorder=5,
        )

    colors = ["#15ef70", "#f81313", "#35bfff", "#e34bfa", "#8172b3"]
    for i, (_, centers) in enumerate(centers_by_model):
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=point_size,
            c=colors[i % len(colors)],
            alpha=0.62,
            linewidths=0,
            label=f"circle-basis centers {i + 1}",
            zorder=3,
        )
    if show_legend:
        ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.92)


def plot_noise_centers_from_models(
    noise_dir="./noise=5%",
    output_filename="visual_noise_centers.pdf",
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    model_prefix="noise_model_sig_",
    detected_results=None,
    detected_boundaries=None,
    boundary_filename="detected_boundaries.npz",
    boundary_color="g",
    boundary_linewidth=1.4,
    S_num=None,
    detect_boundary_from_solution=False,
    para_abs=3.0,
    para_grad=3.0,
    eps=5.0,
    mini_samples=20,
    min_cluster_size=200,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
    draw_support=False,
    X=None,
    Y=None,
    true_support=None,
):
    """Plot local circular-basis centers saved in noise-model checkpoints."""
    centers_by_model = _load_noise_model_centers(noise_dir, model_prefix=model_prefix)
    if not centers_by_model:
        print(f"No noise model files were found in {noise_dir}.")
        return None

    fig, ax = plt.subplots(figsize=(8, 7))
    if draw_support and X is not None and Y is not None and true_support is not None:
        ax.contour(X, Y, true_support, levels=[0.5], colors="black", linewidths=1.2, zorder=5)

    boundaries = _resolve_plot_boundaries(
        noise_dir=noise_dir,
        xlim=xlim,
        ylim=ylim,
        detected_results=detected_results,
        detected_boundaries=detected_boundaries,
        boundary_filename=boundary_filename,
        S_num=S_num,
        detect_boundary_from_solution=detect_boundary_from_solution,
        para_abs=para_abs,
        para_grad=para_grad,
        eps=eps,
        mini_samples=mini_samples,
        min_cluster_size=min_cluster_size,
        min_relative_abs_s=min_relative_abs_s,
        min_relative_grad=min_relative_grad,
        keep_top_k=keep_top_k,
        eps_mode=eps_mode,
    )
    _draw_noise_centers_on_axis(
        ax,
        centers_by_model,
        boundaries=boundaries,
        boundary_color=boundary_color,
        boundary_linewidth=boundary_linewidth,
        point_size=8,
        legend_fontsize=12,
    )

    _format_plain_axis(ax, xlim, ylim, xlabel_size=40, ylabel_size=40, tick_size=22)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if output_filename:
        fig.savefig(os.path.join(noise_dir, output_filename), dpi=300, bbox_inches="tight")
        if output_filename.lower().endswith(".pdf"):
            fig.savefig(os.path.join(noise_dir, output_filename[:-4] + ".png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("noise models:", [os.path.basename(path) for path, _ in centers_by_model])
    return fig


def plot_cell_mesh_and_noise_centers(
    cells_path,
    noise_dir="./noise=5%",
    output_filename="irregular_cells_and_enhance_centers.pdf",
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    model_prefix="noise_model_sig_",
    detected_results=None,
    detected_boundaries=None,
    boundary_filename="detected_boundaries.npz",
    S_num=None,
    detect_boundary_from_solution=False,
    para_abs=3.0,
    para_grad=3.0,
    eps=5.0,
    mini_samples=20,
    min_cluster_size=200,
    min_relative_abs_s=0.0,
    min_relative_grad=0.0,
    keep_top_k=None,
    eps_mode="auto",
    figsize=(8.2, 7.2),
    show_legend=True,
    label_size=42,
    tick_size=22,
):
    """Overlay adaptive cells and local circular-basis centers in one axis."""
    cells = load_cells_from_pickle(cells_path)
    centers_by_model = _load_noise_model_centers(noise_dir, model_prefix=model_prefix)
    if not centers_by_model:
        print(f"No noise model files were found in {noise_dir}.")
        return None

    boundaries = _resolve_plot_boundaries(
        noise_dir=noise_dir,
        xlim=xlim,
        ylim=ylim,
        detected_results=detected_results,
        detected_boundaries=detected_boundaries,
        boundary_filename=boundary_filename,
        S_num=S_num,
        detect_boundary_from_solution=detect_boundary_from_solution,
        para_abs=para_abs,
        para_grad=para_grad,
        eps=eps,
        mini_samples=mini_samples,
        min_cluster_size=min_cluster_size,
        min_relative_abs_s=min_relative_abs_s,
        min_relative_grad=min_relative_grad,
        keep_top_k=keep_top_k,
        eps_mode=eps_mode,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_cell_mesh_on_axis(ax, cells, linewidth=0.7)
    _draw_noise_centers_on_axis(
        ax,
        centers_by_model,
        boundaries=boundaries,
        boundary_color="g",
        boundary_linewidth=1.6,
        point_size=7,
        legend_fontsize=12,
        show_legend=show_legend,
    )
    _format_plain_axis(ax, xlim, ylim, xlabel_size=label_size, ylabel_size=label_size, tick_size=tick_size)

    fig.tight_layout(rect=[0, 0, 1, 1])
    if output_filename:
        if noise_dir and not os.path.exists(noise_dir):
            os.makedirs(noise_dir)
        fig.savefig(os.path.join(noise_dir, output_filename), dpi=300, bbox_inches="tight")
        if output_filename.lower().endswith(".pdf"):
            fig.savefig(os.path.join(noise_dir, output_filename[:-4] + ".png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("noise models:", [os.path.basename(path) for path, _ in centers_by_model])
    return fig


def _coerce_cluster_items(cluster_items):
    if isinstance(cluster_items, dict):
        return [cluster_items]
    return list(cluster_items)


def _get_cluster_boundary(item, boundaries, row):
    cluster_id = int(item.get("cluster_id", row))
    if boundaries is None:
        if "boundary_points" not in item:
            raise KeyError(f"cluster_id={cluster_id} has no boundary_points and boundaries is None")
        boundary = item["boundary_points"]
    elif isinstance(boundaries, dict):
        boundary = boundaries[cluster_id]
    else:
        boundary = boundaries[row]
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] < 2 or len(boundary) < 3:
        raise ValueError(f"cluster_id={cluster_id} boundary must have shape (N, 2), got {boundary.shape}")
    return cluster_id, boundary[:, :2]


def _get_cluster_center(item, boundary):
    params = item.get("params", {}) if isinstance(item, dict) else {}
    if isinstance(params, dict) and "center" in params:
        return np.asarray(params["center"], dtype=float)[:2]
    return np.mean(boundary[:, :2], axis=0)


def _random_uniform(low, high, size, rng=None):
    if rng is None:
        return np.random.uniform(low, high, size=size)
    return rng.uniform(low, high, size=size)


def _random_choice(n, size, replace=True, p=None, rng=None):
    if rng is None:
        return np.random.choice(n, size=size, replace=replace, p=p)
    return rng.choice(n, size=size, replace=replace, p=p)


def plot_boundary_shrinking_and_noise_centers(
    cluster_items,
    boundaries=None,
    g_S=None,
    tau_x_min=0.0,
    tau_x_max=1.0,
    tau_y_min=0.0,
    tau_y_max=1.0,
    h=None,
    alpha=5e-3,
    M_sig=800,
    shrink_distances=None,
    noise_scale_range=(-10.0, 10.0),
    n_resample_shrink=1200,
    n_resample_noise=800,
    output_directory=".",
    output_filename="general_cluster_boundary_shrinking_and_noise.png",
    show=True,
    show_titles=True,
    show_candidates=True,
    show_q_grad=True,
    rng=None,
):
    """Plot boundary shrinking and sampled local circular-basis centers.

    The sampling candidates are generated from center-oriented normal offsets of
    the detected numerical boundary.  When gradient indices are available in a
    cluster item, sampling is biased toward nearby high-gradient Q points.
    """
    bd = _get_bound_detect_module()
    if bd is None or not hasattr(bd, "center_oriented_offset_batch"):
        raise RuntimeError("bound_detect.center_oriented_offset_batch is required for this plot.")

    cluster_items = _coerce_cluster_items(cluster_items)
    if len(cluster_items) == 0:
        raise ValueError("cluster_items is empty.")

    if h is None:
        h = max((tau_x_max - tau_x_min) / 300.0, (tau_y_max - tau_y_min) / 300.0)
    if shrink_distances is None:
        shrink_distances = np.array([-6.0 * h, -3.0 * h], dtype=float)
    else:
        shrink_distances = np.asarray(shrink_distances, dtype=float)

    M_sig = int(M_sig)
    offset_curves_by_cluster = {}
    noise_centers_by_cluster = {}
    candidate_centers_by_cluster = {}
    noise_scale_factors_by_cluster = {}

    n_clusters = len(cluster_items)
    fig, axes = plt.subplots(n_clusters, 2, figsize=(14, 6 * n_clusters), squeeze=False)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    from scipy.spatial import cKDTree

    for row, item in enumerate(cluster_items):
        cluster_id, boundary = _get_cluster_boundary(item, boundaries, row)
        center = _get_cluster_center(item, boundary)

        ax_bound = axes[row, 0]
        offset_curves = bd.center_oriented_offset_batch(
            boundary,
            center,
            shrink_distances,
            n_resample=n_resample_shrink,
        )
        offset_curves_by_cluster[cluster_id] = offset_curves

        ax_bound.plot(
            boundary[:, 0],
            boundary[:, 1],
            c="black",
            linestyle=":",
            linewidth=1.3,
            label="Detected boundary",
        )
        for curve_idx, dist in enumerate(shrink_distances):
            curve = offset_curves[curve_idx] if offset_curves.ndim == 3 else offset_curves
            ax_bound.plot(
                curve[:, 0],
                curve[:, 1],
                color=colors[curve_idx % len(colors)],
                linewidth=1.4,
                label=fr"Shrinked boundary ($\rho={dist:.4f}$)",
            )
        ax_bound.scatter(center[0], center[1], c="red", marker="x", s=60, linewidths=2, label="Center")
        if show_titles:
            ax_bound.set_title(f"General cluster {cluster_id}: boundary shrinking", fontsize=15)

        ax_noise = axes[row, 1]
        scale_factor_noise = _random_uniform(noise_scale_range[0] * h, noise_scale_range[1] * h, size=M_sig, rng=rng)
        noise_offset_curves = bd.center_oriented_offset_batch(
            boundary,
            center,
            scale_factor_noise,
            n_resample=n_resample_noise,
        )
        candidate_centers = noise_offset_curves.reshape(-1, 2)
        inside_domain = (
            (candidate_centers[:, 0] >= tau_x_min)
            & (candidate_centers[:, 0] <= tau_x_max)
            & (candidate_centers[:, 1] >= tau_y_min)
            & (candidate_centers[:, 1] <= tau_y_max)
        )
        candidate_centers = candidate_centers[inside_domain]
        if len(candidate_centers) == 0:
            raise ValueError(f"No offset candidate centers remain inside the domain for cluster_id={cluster_id}.")

        q_points = None
        q_grad_vals = None
        weights = None
        q_indices = item.get("indices") if isinstance(item, dict) else None
        if q_indices is not None and g_S is not None and "points" in item:
            q_indices = np.asarray(q_indices, dtype=int)
            q_points = np.asarray(item["points"], dtype=float)
            q_grad_vals = np.asarray(g_S)[q_indices[:, 0], q_indices[:, 1]]
            tree = cKDTree(q_points)
            _, nearest = tree.query(candidate_centers, k=1)
            weights = np.maximum(q_grad_vals[nearest], 0.0)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else None

        noise_choice = _random_choice(len(candidate_centers), size=M_sig, replace=True, p=weights, rng=rng)
        noise_centers = candidate_centers[noise_choice]
        noise_centers_by_cluster[cluster_id] = noise_centers
        candidate_centers_by_cluster[cluster_id] = candidate_centers
        noise_scale_factors_by_cluster[cluster_id] = scale_factor_noise

        ax_noise.plot(boundary[:, 0], boundary[:, 1], c="black", linestyle=":", linewidth=1.2, label="Detected boundary")
        if show_candidates:
            step = max(1, len(candidate_centers) // 500)
            ax_noise.scatter(
                candidate_centers[::step, 0],
                candidate_centers[::step, 1],
                s=4,
                c="lightgray",
                alpha=0.55,
                label="Offset candidates",
            )
        ax_noise.scatter(
            noise_centers[:, 0],
            noise_centers[:, 1],
            s=4,
            c="red",
            alpha=0.35,
            label="Circle-basis centers",
        )
        if show_q_grad and q_points is not None:
            sc = ax_noise.scatter(
                q_points[:, 0],
                q_points[:, 1],
                c=q_grad_vals,
                cmap=PAPER_SEQUENTIAL_CMAP,
                s=5,
                alpha=0.4,
                label=r"$Q_{grad}$",
            )
            cbar = fig.colorbar(sc, ax=ax_noise, shrink=0.75)
            cbar.set_label("gradient")
        if show_titles:
            ax_noise.set_title(f"General cluster {cluster_id}: sampled centers", fontsize=15)

        for ax in (ax_bound, ax_noise):
            ax.set_xlabel(r"$x_1$", fontsize=15)
            ax.set_ylabel(r"$x_2$", fontsize=15)
            ax.set_xlim(tau_x_min, tau_x_max)
            ax.set_ylim(tau_y_min, tau_y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", alpha=0.2)
            ax.legend(fontsize=9)

    plt.tight_layout()
    if output_filename:
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        elif not output_directory:
            output_directory = "."
        fig.savefig(os.path.join(output_directory, output_filename), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "offset_curves": offset_curves_by_cluster,
        "noise_centers_by_cluster": noise_centers_by_cluster,
        "candidate_centers_by_cluster": candidate_centers_by_cluster,
        "noise_scale_factors_by_cluster": noise_scale_factors_by_cluster,
        "fig": fig,
    }


def improved_plot(data, title, x_min, x_max, y_min, y_max, xlabel="x1", ylabel="x2", cmap=PAPER_SEQUENTIAL_CMAP, n_ticks=5):
    plt.figure(figsize=(3, 3), dpi=120)
    ax = sns.heatmap(data.T, cmap=cmap, cbar_kws={"shrink": 0.8}, cbar=False)
    plt.gca().invert_yaxis()

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    x_labels = np.round(np.linspace(x_min, x_max, n_ticks), decimals=2)
    y_labels = np.round(np.linspace(y_min, y_max, n_ticks), decimals=2)
    plt.xticks(ticks=np.linspace(0, data.shape[0], n_ticks), labels=x_labels, rotation=0, fontsize=10)
    plt.yticks(ticks=np.linspace(0, data.shape[1], n_ticks), labels=y_labels, fontsize=10)

    plt.title(title, fontsize=16, pad=15)
    plt.colorbar(ax.collections[0], label="Error Magnitude")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


class MeshVisualizer:
    def __init__(self, figsize=(18, 8)):
        self.figsize = figsize
        self._kidney_params = {"h": 0.6, "k": 0.25, "a": 0.05}
        self.show_kidney_boundary = False

    def collect_leaf_cells(self, cells):
        leaf_cells = []
        for cell in cells:
            if not cell.children:
                leaf_cells.append(cell)
            else:
                leaf_cells.extend(self.collect_leaf_cells(cell.children))
        return leaf_cells

    def _infer_domain(self, cells, points):
        if points is not None:
            p_np = points.cpu().detach().numpy() if hasattr(points, "cpu") else np.asarray(points)
            if p_np.size > 0 and p_np.ndim == 2 and p_np.shape[1] >= 2:
                return (float(np.min(p_np[:, 0])), float(np.max(p_np[:, 0]))), (
                    float(np.min(p_np[:, 1])),
                    float(np.max(p_np[:, 1])),
                )

        leaf_cells = self.collect_leaf_cells(cells)
        if leaf_cells:
            x_min = min(cell.x0 for cell in leaf_cells)
            x_max = max(cell.x0 + cell.size for cell in leaf_cells)
            y_min = min(cell.y0 for cell in leaf_cells)
            y_max = max(cell.y0 + cell.size for cell in leaf_cells)
            return (x_min, x_max), (y_min, y_max)

        return (0.0, 1.0), (0.0, 1.0)

    def _get_kidney_boundary(self):
        if bound_detect is None or not hasattr(bound_detect, "get_contour_points_from_implicit"):
            return None
        return np.asarray(
            bound_detect.get_contour_points_from_implicit(
                kidney_curve,
                func_args=self._kidney_params,
                x_range=(0, 1),
                y_range=(0, 1),
            )
        )

    def _add_boundary_overlay(self, ax, x_range, y_range):
        if not self.show_kidney_boundary:
            return
        kidney_boundary = self._get_kidney_boundary()
        if kidney_boundary is not None and x_range[0] >= 0 and x_range[1] <= 1 and y_range[0] >= 0 and y_range[1] <= 1:
            ax.plot(
                kidney_boundary[:, 0],
                kidney_boundary[:, 1],
                color="red",
                linewidth=2.5,
                linestyle="--",
                label="True Boundary",
            )
            ax.legend()

    def _plot_single_mesh(self, fig, ax, cells, title, S_num=None, points=None):
        leaf_cells = self.collect_leaf_cells(cells)

        if S_num is not None and points is not None:
            p_np = points.cpu().detach().numpy() if hasattr(points, "cpu") else np.asarray(points)
            s_np = S_num.cpu().detach().numpy().flatten() if hasattr(S_num, "cpu") else np.asarray(S_num).flatten()
            if p_np.ndim == 2 and p_np.shape[0] >= 3 and p_np.shape[1] >= 2 and s_np.size == p_np.shape[0]:
                contour = ax.tricontourf(
                    p_np[:, 0],
                    p_np[:, 1],
                    s_np,
                    levels=20,
                    cmap=PAPER_SEQUENTIAL_CMAP,
                    alpha=0.95,
                )
                fig.colorbar(contour, ax=ax, label="Solution Value")
            else:
                ax.text(0.5, 0.5, "Not enough points for contour plot", ha="center", va="center", transform=ax.transAxes)

        for cell in leaf_cells:
            rect = patches.Rectangle(
                (cell.x0, cell.y0),
                cell.size,
                cell.size,
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

        x_range, y_range = self._infer_domain(cells, points)
        self._add_boundary_overlay(ax, x_range, y_range)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def plot_mesh_comparison(self, initial_cells, refined_cells, cell_indicators=None, save_path=None, S_num=None, points=None):
        del cell_indicators

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        self._plot_single_mesh(fig, ax1, initial_cells, "Initial Mesh")
        self._plot_single_mesh(fig, ax2, refined_cells, "Refined Mesh", S_num=S_num, points=points)

        initial_count = len(self.collect_leaf_cells(initial_cells))
        refined_count = len(self.collect_leaf_cells(refined_cells))
        fig.suptitle(
            "Adaptive Mesh Refinement Comparison\n"
            f"Initial: {initial_count} cells -> Refined: {refined_count} cells "
            f"(Added {refined_count - initial_count} cells)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        plt.show()
        return fig

    def plot_solution_and_mesh(self, cells, all_points, S_num, grad_S_num, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        points = all_points.cpu().detach().numpy() if hasattr(all_points, "cpu") else np.asarray(all_points)
        solution = S_num.cpu().detach().numpy().flatten() if hasattr(S_num, "cpu") else np.asarray(S_num).flatten()
        gradient = grad_S_num.cpu().detach().numpy() if hasattr(grad_S_num, "cpu") else np.asarray(grad_S_num)
        grad_norm = np.linalg.norm(gradient, axis=1)

        if points.shape[0] >= 3:
            contour1 = ax1.tricontourf(points[:, 0], points[:, 1], solution, levels=14, cmap=PAPER_SEQUENTIAL_CMAP, alpha=0.95)
            ax1.set_title("Numerical Solution $S_{num}$ Distribution")
            fig.colorbar(contour1, ax=ax1, label="Solution Value")
        else:
            ax1.text(0.5, 0.5, "Not enough points for tricontourf", ha="center", va="center")

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect("equal", adjustable="box")

        if points.shape[0] >= 3:
            contour2 = ax2.tricontourf(points[:, 0], points[:, 1], grad_norm, levels=14, cmap=PAPER_SEQUENTIAL_CMAP, alpha=0.95)
            ax2.set_title("Gradient Norm $||\\nabla S||$ Distribution")
            fig.colorbar(contour2, ax=ax2, label="Gradient Norm")
        else:
            ax2.text(0.5, 0.5, "Not enough points for tricontourf", ha="center", va="center")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Solution and mesh analysis plot saved to: {save_path}")
        plt.show()
        return fig

    def plot_refinement_history(self, refinement_stats, save_path=None):
        history = refinement_stats.get("history", [])
        if not history:
            print("No refinement history data available")
            return None

        iterations = [info["iteration"] for info in history]
        active_cells = [info["active_cells"] for info in history]
        refined_cells = [info["refined_cells"] for info in history]
        max_indicators = [info["max_indicator"] for info in history]
        mean_indicators = [info["mean_indicator"] for info in history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        ax1.plot(iterations, active_cells, "bo-", linewidth=2, markersize=8)
        ax1.set_title("Active Cell Count Evolution")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Number of Cells")
        ax1.grid(True, alpha=0.3)

        ax2.plot(iterations, refined_cells, "go-", linewidth=2, markersize=8)
        ax2.set_title("Refined Cells per Iteration")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Number of Refined Cells")
        ax2.grid(True, alpha=0.3)

        ax3.plot(iterations, max_indicators, "ro-", linewidth=2, markersize=8)
        ax3.set_title("Maximum Indicator Value Evolution")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Max Indicator Value")
        ax3.grid(True, alpha=0.3)

        ax4.plot(iterations, mean_indicators, "go-", linewidth=2, markersize=8)
        ax4.set_title("Mean Indicator Value Evolution")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Mean Indicator Value")
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f"Adaptive Refinement Statistics ({len(history)} iterations)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Refinement history plot saved to: {save_path}")
        plt.show()
        return fig


class GridMeshVisualizer:
    def __init__(self, figsize=(30, 8)):
        self.figsize = figsize
        self.func_params = {"h": 0.6, "k": 0.25, "a": 0.05}
        self.contour_points_implicit = None

    def _load_true_boundary(self):
        if bound_detect is None or not hasattr(bound_detect, "get_contour_points_from_implicit"):
            return None
        try:
            return np.asarray(
                bound_detect.get_contour_points_from_implicit(
                    kidney_curve,
                    func_args=self.func_params,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
            )
        except Exception:
            return None

    def collect_leaf_cells(self, cells):
        leaf_cells = []
        for cell in cells:
            if not cell.children:
                leaf_cells.append(cell)
            else:
                leaf_cells.extend(self.collect_leaf_cells(cell.children))
        return leaf_cells

    def _infer_ranges(self, cells, points, default=None):
        if default is not None:
            return default
        if points is not None:
            p_np = points.cpu().detach().numpy() if hasattr(points, "cpu") else np.asarray(points)
            if p_np.size > 0 and p_np.ndim == 2 and p_np.shape[1] >= 2:
                return (float(np.min(p_np[:, 0])), float(np.max(p_np[:, 0]))), (
                    float(np.min(p_np[:, 1])),
                    float(np.max(p_np[:, 1])),
                )
        leaf_cells = self.collect_leaf_cells(cells)
        if leaf_cells:
            x_min = min(cell.x0 for cell in leaf_cells)
            x_max = max(cell.x0 + cell.size for cell in leaf_cells)
            y_min = min(cell.y0 for cell in leaf_cells)
            y_max = max(cell.y0 + cell.size for cell in leaf_cells)
            return (x_min, x_max), (y_min, y_max)
        return (0.0, 1.0), (0.0, 1.0)

    def _infer_style(self, x_range, y_range, method, kidney_bound=None):
        del y_range
        if kidney_bound is not None:
            return "kidney"
        if method == "evolution":
            return "ex44" if x_range[0] < 0 else "ex45"
        return "ex43"

    def _plot_solution(self, fig, ax, points, S_num, colorbar_size=16):
        if S_num is None or points is None:
            return
        p_np = points.cpu().detach().numpy() if hasattr(points, "cpu") else np.asarray(points)
        s_np = S_num.cpu().detach().numpy().flatten() if hasattr(S_num, "cpu") else np.asarray(S_num).flatten()
        if p_np.ndim == 2 and p_np.shape[0] >= 3 and p_np.shape[1] >= 2 and s_np.size == p_np.shape[0]:
            contour = ax.tricontourf(
                p_np[:, 0],
                p_np[:, 1],
                s_np,
                levels=20,
                cmap=PAPER_SEQUENTIAL_CMAP,
                alpha=0.95,
            )
            cbar = fig.colorbar(contour, ax=ax)
            cbar.ax.tick_params(labelsize=colorbar_size)

    def _draw_boundaries(self, ax, style):
        if style == "ex43":
            ax.add_patch(
                patches.Circle((0.5, 0.5), 0.2, linewidth=1.5, edgecolor="r", facecolor="none", label="Boundary Layer")
            )
        elif style == "ex44":
            ax.add_patch(
                patches.Circle((-0.06, 0.0), 0.06, linewidth=2, edgecolor="r", facecolor="none", label="Boundary Layer")
            )
            ax.add_patch(
                patches.Circle((0.08, 0.0), 0.06, linewidth=2, edgecolor="r", facecolor="none", label="Boundary Layer")
            )
        elif style == "ex45":
            ax.add_patch(
                patches.Circle((0.71, 0.5), 0.2, linewidth=2, edgecolor="r", facecolor="none", label="Boundary Layer")
            )
            ax.add_patch(
                patches.Rectangle((0.29, 0.3), 0.2, 0.4, linewidth=2, edgecolor="b", facecolor="none", label="Defined Rectangle")
            )
        elif style == "kidney":
            self.contour_points_implicit = self._load_true_boundary()
        if style == "kidney" and self.contour_points_implicit is not None:
            ax.plot(
                self.contour_points_implicit[:, 0],
                self.contour_points_implicit[:, 1],
                color="red",
                linewidth=3,
                linestyle="--",
                label="True Boundary",
            )
            ax.legend(fontsize=22 if ax.figure.get_figwidth() > 8 else 12)

    def _format_axes(self, ax, x_range, y_range, style, label):
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)

        if style in {"ex43", "kidney"}:
            ax.set_xlabel("$x_1$", fontsize=40 if style == "kidney" else 24)
            ax.set_ylabel("$x_2$", fontsize=40 if style == "kidney" else 24)
            ax.tick_params(axis="both", which="major", labelsize=22 if style == "kidney" else 20)
            if label is None and style == "ex43":
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            n_ticks = 5
            x_labels = np.round(np.linspace(x_range[0], x_range[1], n_ticks), 2)
            y_labels = np.round(np.linspace(y_range[0], y_range[1], n_ticks), 2)
            ax.set_xticks(np.linspace(x_range[0], x_range[1], n_ticks))
            ax.set_xticklabels(x_labels, rotation=0, fontsize=28 if style == "ex44" else 20)
            if style == "ex44":
                ax.set_yticks(np.linspace(y_range[0], y_range[1], n_ticks))
                ax.set_yticklabels(y_labels, fontsize=28)
                ax.set_xlabel("$x_1$", fontsize=38)
            else:
                ax.set_yticks([])
                ax.set_xlabel("$x_1$", fontsize=34)
            if label:
                ax.set_ylabel("$x_2$", fontsize=38 if style == "ex44" else 36)

    def _safe_layout(self, fig, bottom=0.14):
        """Leave enough bottom margin for large math axis labels in saved PDFs."""
        fig.tight_layout(rect=[0.02, bottom, 0.98, 0.98])

    def _plot_single_mesh(
        self,
        fig,
        ax,
        cells,
        S_num=None,
        points=None,
        label=None,
        x_range=None,
        y_range=None,
        save_path=None,
        style=None,
    ):
        del save_path
        leaf_cells = self.collect_leaf_cells(cells)
        xr, yr = self._infer_ranges(cells, points, default=((x_range, y_range) if x_range is not None and y_range is not None else None))
        if style is None:
            style = self._infer_style(xr, yr, method="single")

        self._plot_solution(fig, ax, points, S_num, colorbar_size=22 if style == "kidney" else 16)

        for cell in leaf_cells:
            ax.add_patch(
                patches.Rectangle(
                    (cell.x0, cell.y0),
                    cell.size,
                    cell.size,
                    linewidth=0.8,
                    edgecolor="black",
                    facecolor="none",
                    alpha=0.8,
                )
            )

        self._draw_boundaries(ax, style)
        self._format_axes(ax, xr, yr, style, label)
        return fig

    def plot_mesh_comparison(self, initial_cells, refined_cells, kidney_bound=None, S_num=None, points=None, save_path=None):
        if kidney_bound is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
            self._plot_single_mesh(fig, ax1, initial_cells)
            self._plot_single_mesh(fig, ax2, refined_cells, S_num=S_num, points=points)
            self._scale(fig, ax3, kidney_bound)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            self._plot_single_mesh(fig, ax1, initial_cells)
            self._plot_single_mesh(fig, ax2, refined_cells, S_num=S_num, points=points)
        self._safe_layout(fig, bottom=0.08)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.18)
            print(f"Figure saved to: {save_path}")
        plt.show()
        return fig

    def _scale(self, fig, ax, kidney_bound):
        if self.contour_points_implicit is None:
            self.contour_points_implicit = self._load_true_boundary()
        if self.contour_points_implicit is None or bound_detect is None:
            raise RuntimeError("Kidney boundary helpers are unavailable.")

        offset_distances = np.linspace(-0.02, 0.02, 2)
        kidney_center = np.mean(np.asarray(kidney_bound, dtype=float), axis=0)
        contour_points_scaled = bound_detect.center_oriented_offset_batch(
            kidney_bound,
            kidney_center,
            offset_distances,
            n_resample=1200,
        )
        ax.clear()
        ax.plot(kidney_bound[:, 0], kidney_bound[:, 1], c="g", linewidth=1, label="Original detected boundary")
        ax.plot(
            self.contour_points_implicit[:, 0],
            self.contour_points_implicit[:, 1],
            color="red",
            linewidth=3,
            linestyle="--",
            label="True Boundary",
        )
        colors = ["#470eaa", "#fb5817"]
        for i, d in enumerate(offset_distances):
            ax.plot(contour_points_scaled[i, :, 0], contour_points_scaled[i, :, 1], color=colors[i], linewidth=1, label=rf"Offset boundary($\rho={d}$)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)
        ax.set_xlabel("$x_1$", fontsize=40)
        ax.set_ylabel("$x_2$", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.legend(fontsize=22, loc="best")
        self._safe_layout(fig, bottom=0.14)
        return fig

    def plot_local_basis_centers(
        self,
        kidney_bound,
        noise_dir="./noise=5%",
        model_noise_filename="model_noise.pth",
        ax=None,
        fig=None,
        output_filename="local_basis_centers.pdf",
        fallback_count=1000,
        seed=2,
    ):
        if bound_detect is None:
            raise RuntimeError("Kidney boundary helpers are unavailable.")

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        kidney_bound = np.asarray(kidney_bound, dtype=float)
        kidney_center = np.mean(kidney_bound, axis=0)
        tau_x_min, tau_x_max = 0.0, 1.0
        tau_y_min, tau_y_max = 0.0, 1.0
        h = max((tau_x_max - tau_x_min) / 300.0, (tau_y_max - tau_y_min) / 300.0)

        model_noise_path = os.path.join(noise_dir, model_noise_filename)
        if os.path.exists(model_noise_path):
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional plotting dependency
                raise RuntimeError("torch is required to load the local noise model.") from exc

            noise_state = torch.load(model_noise_path, map_location="cpu")
            b1_noise = -noise_state["hidden_layer_1.0.bias"].detach().cpu().numpy()
            b2_noise = -noise_state["hidden_layer_2.0.bias"].detach().cpu().numpy()
            noise_centers = np.column_stack((b1_noise, b2_noise))
        else:
            rng = np.random.default_rng(seed)
            candidate_offsets = -np.linspace(3 * h, 10 * h, 8)
            candidate_curves = bound_detect.center_oriented_offset_batch(
                kidney_bound,
                kidney_center,
                candidate_offsets,
                n_resample=700,
            )
            candidate_centers = candidate_curves.reshape(-1, 2)
            inside_domain = (
                (candidate_centers[:, 0] >= tau_x_min)
                & (candidate_centers[:, 0] <= tau_x_max)
                & (candidate_centers[:, 1] >= tau_y_min)
                & (candidate_centers[:, 1] <= tau_y_max)
            )
            candidate_centers = candidate_centers[inside_domain]
            if len(candidate_centers) == 0:
                raise ValueError("No offset candidate centers remain inside the computational domain.")
            noise_choice = rng.choice(len(candidate_centers), size=fallback_count, replace=True)
            noise_centers = candidate_centers[noise_choice]

        ax.clear()
        ax.scatter(
            noise_centers[:, 0],
            noise_centers[:, 1],
            s=8,
            c="red",
            alpha=0.58,
            linewidths=0,
            label="Circle-basis centers",
            zorder=3,
        )
        ax.plot(kidney_bound[:, 0], kidney_bound[:, 1], c="g", linewidth=1.4, label="Detected boundary", zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)
        ax.set_xlabel("$x_1$", fontsize=40)
        ax.set_ylabel("$x_2$", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.legend(fontsize=22, loc="best")
        self._safe_layout(fig, bottom=0.14)

        if output_filename:
            if noise_dir and not os.path.exists(noise_dir):
                os.makedirs(noise_dir)
            fig.savefig(os.path.join(noise_dir, output_filename), dpi=300, bbox_inches="tight", pad_inches=0.18)
            if output_filename.lower().endswith(".pdf"):
                fig.savefig(os.path.join(noise_dir, output_filename[:-4] + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.18)

        return fig

    def plot_scale_and_local_basis(
        self,
        kidney_bound,
        noise_dir="./noise=5%",
        output_filename="scale_and_local_basis.pdf",
    ):
        if self.contour_points_implicit is None or bound_detect is None:
            raise RuntimeError("Kidney boundary helpers are unavailable.")

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        kidney_bound = np.asarray(kidney_bound, dtype=float)
        kidney_center = np.mean(kidney_bound, axis=0)
        offset_distances = np.linspace(-0.02, 0.02, 2)
        contour_points_scaled = bound_detect.center_oriented_offset_batch(
            kidney_bound,
            kidney_center,
            offset_distances,
            n_resample=1200,
        )

        model_noise_path = os.path.join(noise_dir, "model_noise.pth")
        if os.path.exists(model_noise_path):
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional plotting dependency
                raise RuntimeError("torch is required to load the local noise model.") from exc

            noise_state = torch.load(model_noise_path, map_location="cpu")
            b1_noise = -noise_state["hidden_layer_1.0.bias"].detach().cpu().numpy()
            b2_noise = -noise_state["hidden_layer_2.0.bias"].detach().cpu().numpy()
            noise_centers = np.column_stack((b1_noise, b2_noise))
        else:
            tau_x_min, tau_x_max = 0.0, 1.0
            tau_y_min, tau_y_max = 0.0, 1.0
            h = max((tau_x_max - tau_x_min) / 300.0, (tau_y_max - tau_y_min) / 300.0)
            rng = np.random.default_rng(2)
            candidate_offsets = -np.linspace(3 * h, 10 * h, 8)
            candidate_curves = bound_detect.center_oriented_offset_batch(
                kidney_bound,
                kidney_center,
                candidate_offsets,
                n_resample=700,
            )
            candidate_centers = candidate_curves.reshape(-1, 2)
            inside_domain = (
                (candidate_centers[:, 0] >= tau_x_min)
                & (candidate_centers[:, 0] <= tau_x_max)
                & (candidate_centers[:, 1] >= tau_y_min)
                & (candidate_centers[:, 1] <= tau_y_max)
            )
            candidate_centers = candidate_centers[inside_domain]
            if len(candidate_centers) == 0:
                raise ValueError("No offset candidate centers remain inside the computational domain.")
            noise_choice = rng.choice(len(candidate_centers), size=1000, replace=True)
            noise_centers = candidate_centers[noise_choice]

        ax.clear()
        ax.scatter(
            noise_centers[:, 0],
            noise_centers[:, 1],
            s=8,
            c="#fff422",
            alpha=0.68,
            linewidths=0,
            label="Circle-basis centers",
            zorder=2,
        )
        ax.plot(kidney_bound[:, 0], kidney_bound[:, 1], c="g", linewidth=1.4, label="Detected boundary", zorder=4)
        ax.plot(
            self.contour_points_implicit[:, 0],
            self.contour_points_implicit[:, 1],
            color="red",
            linewidth=3,
            linestyle="--",
            label="True Boundary",
            zorder=5,
        )
        colors = ["#470eaa", "#fb5817"]
        for i, d in enumerate(offset_distances):
            ax.plot(
                contour_points_scaled[i, :, 0],
                contour_points_scaled[i, :, 1],
                color=colors[i],
                linewidth=1,
                label=rf"Offset boundary($\rho={d}$)",
                zorder=6,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)
        ax.set_xlabel("$x_1$", fontsize=40)
        ax.set_ylabel("$x_2$", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.legend(fontsize=18, loc="best")
        self._safe_layout(fig, bottom=0.14)

        if output_filename:
            if noise_dir and not os.path.exists(noise_dir):
                os.makedirs(noise_dir)
            fig.savefig(os.path.join(noise_dir, output_filename), dpi=300, bbox_inches="tight", pad_inches=0.18)
            if output_filename.lower().endswith(".pdf"):
                fig.savefig(os.path.join(noise_dir, output_filename[:-4] + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.18)
        return fig

    def plot_mesh_evolution(self, cells_store, S_num_stores=None, points=None, save_path=None, label=None, temp=None, g_S=None):
        n_steps = len(cells_store)
        x_range, y_range = self._infer_ranges(cells_store[0], points, default=None)
        style = self._infer_style(x_range, y_range, method="evolution")

        if temp:
            width = (8.8 if style == "ex44" else 6.5) * (n_steps + 1)
            height = 7.5 if style == "ex44" else 7
            fig, axes = plt.subplots(1, n_steps + 1, figsize=(width, height))
        else:
            fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))
        if n_steps == 1:
            axes = [axes]

        is_single_solution = S_num_stores is not None and not (
            isinstance(S_num_stores, (list, tuple)) and len(S_num_stores) == n_steps
        )

        for i, cells in enumerate(cells_store):
            self._plot_single_mesh(
                fig,
                axes[i],
                cells,
                S_num=None,
                points=points,
                label=(label if i == 0 else False),
                x_range=x_range,
                y_range=y_range,
                style=style,
            )
            axes[i].set_title(f"$\\mathcal{{C}}_{{{i}}}$", fontsize=40 if style in {"ex44", "ex45"} else 20)
            if style == "ex44":
                axes[i].tick_params(axis="both", labelsize=28)
            elif style == "ex45":
                axes[i].tick_params(axis="x", labelsize=28)

        if temp and g_S is not None and bound_detect is not None:
            tau_x_min, tau_x_max = x_range
            tau_y_min, tau_y_max = y_range
            x = np.linspace(tau_x_min, tau_x_max, 301)
            y = np.linspace(tau_y_min, tau_y_max, 301)
            X, Y = np.meshgrid(x, y)
            params = {
                "ex44": {"para_abs": 100, "para_grad": 3, "eps": 5.0, "mini_samples": 20, "title": "$\\mathcal{Q}_{grad}$"},
                "ex45": {"para_abs": 3, "para_grad": 3, "eps": 3.0, "mini_samples": 20, "title": "$\\mathcal{Q}$"},
            }[style]
            s_for_detect = S_num_stores[-1] if not is_single_solution and S_num_stores is not None else S_num_stores
            bound_detect.detect_shape(
                g_S,
                X.T,
                Y.T,
                tau_x_min,
                tau_x_max,
                tau_y_min,
                tau_y_max,
                params["para_abs"],
                params["para_grad"],
                params["eps"],
                params["mini_samples"],
                S_num=s_for_detect,
                ax=axes[-1],
                fig=fig,
                output_directory=".",
                output_filename="plot.pdf",
                show_colorbar=False,
                eps_mode="grid_scaled",
            )
            axes[-1].set_title(params["title"], fontsize=42 if style == "ex44" else 40)
            if axes[-1].collections:
                mappable = axes[-1].collections[0]
                cbar = fig.colorbar(mappable, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
                cbar.ax.tick_params(labelsize=24)
                cbar.set_label(r"$S^{(0)}$", fontsize=30)
            self._format_axes(axes[-1], x_range, y_range, style, True)

            if style == "ex44":
                for ax in axes[1:]:
                    ax.set_ylabel("")
                    ax.set_yticks([])
                axes[0].set_ylabel("$x_2$", fontsize=38)
                n_ticks = 5
                y_labels = np.round(np.linspace(y_range[0], y_range[1], n_ticks), 2)
                axes[0].set_yticks(np.linspace(y_range[0], y_range[1], n_ticks))
                axes[0].set_yticklabels(y_labels, fontsize=28)
                axes[-1].set_ylabel("")
                axes[-1].set_yticks([])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        plt.show()
        return fig


def visualize_adaptive_refinement(cells, refined_cells, all_g_p, S_num, grad_S_num, refinement_stats, cell_indicators=None):
    del grad_S_num, refinement_stats

    visualizer = MeshVisualizer(figsize=(20, 8))
    print("Generating visualizations...")
    print("1. Plotting mesh comparison...")
    visualizer.plot_mesh_comparison(
        cells,
        refined_cells,
        cell_indicators=cell_indicators,
        S_num=S_num,
        points=all_g_p,
    )
    print("All visualizations completed!")
