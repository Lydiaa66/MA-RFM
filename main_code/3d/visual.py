import os
import re
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection

plt.rcParams["mathtext.fontset"] = "cm"

EXAMPLE_DIR_PATTERN = re.compile(r"^Ex\d+\.\d+$", re.IGNORECASE)


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


def _resolve_save_path(save_path):
    if not save_path:
        return save_path
    directory = os.path.dirname(save_path)
    filename = os.path.basename(save_path)
    prefixed = _prefix_filename_for_example(filename, save_path, directory)
    return os.path.join(directory, prefixed) if directory else prefixed


class MeshVisualizer3D:
    def __init__(self, figsize=(18, 8)):
        """
        Mesh visualization utility.
        
        Args:
            figsize: Figure size.
        """
        self.figsize = figsize
        
    def collect_leaf_cells(self, cells):
        """Recursively collect all leaf cells."""
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
        Plot the mesh before and after refinement.
        
        Args:
            initial_cells: Initial mesh-cell list.
            refined_cells: Refined mesh-cell list.
            cell_indicators: Mesh indicator dictionary used for color mapping.
            save_path: Output path.
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
            save_path = _resolve_save_path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_single_mesh(self, ax, cells, title, cell_indicators=None):
        """Plot a single mesh."""
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
            
        boundary_circle1 = patches.Circle((-0.06, 0), 0.06,
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none',
                                        label='Boundary Layer')
        boundary_circle2 = patches.Circle((0.08, 0), 0.06,
                                linewidth=2,
                                edgecolor='r',
                                facecolor='none',
                                label='Boundary Layer')

        ax.add_patch(boundary_circle1)
        ax.add_patch(boundary_circle2)

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def plot_solution_and_mesh(self, cells, all_points, S_num, grad_S_num, 
                              save_path=None):
        """
        Plot the numerical solution, gradient, and mesh together.
        
        Args:
            cells: Mesh-cell list.
            all_points: Coordinates of all Gauss points.
            S_num: Numerical solution.
            grad_S_num: Gradient.
            save_path: Output path.
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
            save_path = _resolve_save_path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Solution and mesh analysis plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_refinement_history(self, refinement_stats, save_path=None):
        """
        Plot refinement-history statistics.
        
        Args:
            refinement_stats: Refinement statistics.
            save_path: Output path.
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
            save_path = _resolve_save_path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Refinement history plot saved to: {save_path}")
        
        plt.show()
        return fig

    def create_detailed_mesh_plot(self, cells, cell_indicators=None, 
                                 gauss_points=None, save_path=None):
        """
        Create a detailed mesh plot including Gauss points.
        
        Args:
            cells: Mesh-cell list.
            cell_indicators: Indicator values.
            gauss_points: Coordinates of Gauss points.
            save_path: Output path.
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
            save_path = _resolve_save_path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed mesh plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_isosurface(
        self,
        volume,
        level=None,
        color="#5AA7D9",
        alpha=0.25,
        show_wireframe=False,
        show_center=True,
        center=None,
        elev=30,
        azim=-60,
        save_path=None,
        ax=None,
        title=None,
        scalar_field=None,
        cmap="viridis",
        value_limits=None,
        show_colorbar=False,
        colorbar_ticks=None,
        colorbar_label=r"$S^{(0)}$",
        rasterized=False,
    ):
        """
        Plot a shaded isosurface.

        If ``scalar_field`` is provided, the surface geometry is extracted from
        ``volume`` while the face colors are interpolated from ``scalar_field``.
        """
        volume = np.asarray(volume, dtype=float)
        volume = np.nan_to_num(volume, copy=False)

        volume_min = float(np.min(volume))
        volume_max = float(np.max(volume))
        if level is None:
            level = 0.5 * (volume_min + volume_max)

        print(
            f"Global volume range: [{volume_min:.3f}, {volume_max:.3f}], using level={level:.3f}"
        )

        if not (volume_min < level < volume_max):
            print("Marching cubes skipped: surface level must lie strictly inside the data range.")
            return None

        try:
            verts, faces, normals, _ = measure.marching_cubes(volume, level=level)
        except Exception as e:
            print(f"Marching cubes failed: {e}")
            return None

        if len(verts) == 0 or len(faces) == 0:
            print("No isosurface found at this level.")
            return None

        if hasattr(self, "domain_min") and hasattr(self, "domain_max"):
            dmin = np.array(self.domain_min, dtype=float)
            dmax = np.array(self.domain_max, dtype=float)
            shape = np.maximum(np.array(volume.shape, dtype=float) - 1.0, 1.0)
            mapped_verts = dmin + (verts / shape) * (dmax - dmin)
        else:
            mapped_verts = verts
            print("Warning: domain_min/domain_max not set. Using raw voxel coordinates.")

        try:
            base_rgba = np.array(plt.get_cmap(color)(0.65))
        except ValueError:
            base_rgba = np.array(to_rgba(color))

        face_normals = normals[faces].mean(axis=1)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12

        light_dir = np.array([0.35, -0.45, 0.82], dtype=float)
        light_dir /= np.linalg.norm(light_dir)
        lighting = np.clip(face_normals @ light_dir, 0.0, 1.0)
        lighting = 0.35 + 0.65 * lighting

        if scalar_field is not None:
            from scipy.interpolate import RegularGridInterpolator

            scalar_field = np.asarray(scalar_field, dtype=float)
            scalar_field = np.nan_to_num(scalar_field, copy=False)
            grid = [np.arange(s) for s in scalar_field.shape]
            interp = RegularGridInterpolator(
                grid,
                scalar_field,
                bounds_error=False,
                fill_value=np.nan,
            )
            vertex_values = interp(verts)
            face_values = np.nanmean(vertex_values[faces], axis=1)

            if value_limits is None:
                value_min = float(np.nanmin(face_values))
                value_max = float(np.nanmax(face_values))
            else:
                value_min, value_max = value_limits

            if np.isclose(value_min, value_max):
                delta = max(1e-8, 1e-6 * max(1.0, abs(value_min)))
                value_min -= delta
                value_max += delta

            norm = plt.Normalize(vmin=value_min, vmax=value_max)
            cmap_obj = plt.get_cmap(cmap)
            face_colors = cmap_obj(norm(face_values))
            face_colors[:, :3] *= lighting[:, None]
            face_colors[:, 3] = alpha
        else:
            face_values = None
            norm = None
            cmap_obj = None
            face_colors = np.repeat(base_rgba[None, :], len(faces), axis=0)
            face_colors[:, :3] *= lighting[:, None]
            face_colors[:, 3] = alpha

        mesh = Poly3DCollection(
            mapped_verts[faces],
            linewidths=0.15,
            edgecolors=(0.1, 0.1, 0.1, 0.15) if show_wireframe else "none",
        )
        mesh.set_rasterized(rasterized)
        mesh.set_facecolors(face_colors)

        created_figure = ax is None
        if created_figure:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.add_collection3d(mesh)

        if hasattr(self, "domain_min") and hasattr(self, "domain_max"):
            ax.set_xlim(self.domain_min[0], self.domain_max[0])
            ax.set_ylim(self.domain_min[1], self.domain_max[1])
            ax.set_zlim(self.domain_min[2], self.domain_max[2])

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)

        if created_figure:
            ax.set_xlabel("$x_1$", fontsize=14)
            ax.set_ylabel("$x_2$", fontsize=14)
            ax.set_zlabel("$x_3$", fontsize=14)

        if show_center and center is not None:
            centers = np.atleast_2d(center)
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                c="red",
                marker="x",
                s=80,
                depthshade=False,
            )

        if show_colorbar and face_values is not None:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array(face_values)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.58, pad=0.04)
            cbar.set_label(colorbar_label, fontsize=20)
            cbar.ax.tick_params(labelsize=16)
            if colorbar_ticks is not None:
                cbar.set_ticks(colorbar_ticks)

        if created_figure:
            ax.set_facecolor("white")
            ax.grid(False)
            if title is None:
                title = f"Isosurface at level = {level:.4f}"
            ax.set_title(title, fontsize=12)
            if save_path:
                plt.savefig(_resolve_save_path(save_path), dpi=300, bbox_inches="tight")
            plt.show()

        return mesh

def visualize_adaptive_refinement(cells, refined_cells, all_g_p, S_num, grad_S_num, 
                                 refinement_stats, cell_indicators=None):
    """
    Complete visualization for adaptive refinement.
    
    Args:
        cells: Initial mesh.
        refined_cells: Refined mesh.
        all_g_p: Gauss points.
        S_num: Numerical solution.
        grad_S_num: Gradient.
        refinement_stats: Refinement statistics.
        cell_indicators: Indicator values.
    """
    visualizer = MeshVisualizer3D()
    
    print("Generating visualizations...")
    
    initial_count = len(visualizer.collect_leaf_cells(cells))
    refined_count = len(visualizer.collect_leaf_cells(refined_cells))

    print("1. Plotting refined mesh (3 views)...")
    print(
        f"   cells: {initial_count} -> {refined_count} "
        f"(added {refined_count - initial_count})"
    )
    visualize_3d_grid(
        refined_cells,
        title="Refined Mesh",
        views=[(0, 90), (90, 180), (0, 180)],
        output_directory=".",
        output_filename=None,
    )
    
    
    
    
    print("All visualizations completed!")


MeshVisualizer = MeshVisualizer3D


def draw(
    X,
    Y,
    data,
    data_1D,
    x_label,
    y_label,
    xx,
    cmap="viridis",
    x_ticks=None,
    y_ticks=None,
    output_directory=".",
    output_filename1="plot.png",
    output_filename2="plot_1d.png",
    temp_point=False,
    temp_v=False,
):
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif not output_directory:
        output_directory = "."

    full_output_path1 = _resolve_output_path(output_directory, output_filename1)
    full_output_path2 = _resolve_output_path(output_directory, output_filename2)

    data = np.asarray(data)
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
    cbar = plt.colorbar(contour_fill, shrink=1, aspect=10)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    if x_ticks is not None:
        plt.xticks(ticks=x_ticks)
    if y_ticks is not None:
        plt.yticks(ticks=y_ticks)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.gca().set_facecolor("white")

    if temp_point:
        max_idx = np.nanargmax(data)
        min_idx = np.nanargmin(data)
        x_max = X.flat[max_idx]
        y_max = Y.flat[max_idx]
        x_min = X.flat[min_idx]
        y_min = Y.flat[min_idx]
        dx = 0.05 * (X.max() - X.min())
        dy = 0.05 * (Y.max() - Y.min())

        plt.plot(x_max, y_max, "ro", markersize=6)
        plt.annotate(
            f"Max:({x_max:.2f}, {y_max:.2f})",
            xy=(x_max, y_max),
            xycoords="data",
            xytext=(x_max + dx, y_max + dy),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=12,
            color="black",
            weight="bold",
            ha="left",
            va="bottom",
        )

        plt.plot(x_min, y_min, "bo", markersize=6)
        plt.annotate(
            f"Min:({x_min:.2f}, {y_min:.2f})",
            xy=(x_min, y_min),
            xycoords="data",
            xytext=(x_min - dx, y_min - dy),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=12,
            color="black",
            weight="bold",
            ha="right",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(full_output_path1, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {full_output_path1}")

    if temp_v and len(np.asarray(data_1D).shape) > 0 and np.asarray(data_1D).size > 0:
        data_1D = np.asarray(data_1D).reshape(-1)
        x_divide = np.linspace(0, 1, len(data_1D))
        max_abs_idx = np.argmax(np.abs(data_1D))
        half_max = data_1D[max_abs_idx] / 2
        half_max_indices = np.where(
            (data_1D >= half_max - 10 ** (-2.2)) & (data_1D <= half_max + 10 ** (-2.2))
        )[0]

        plt.figure(figsize=(5, 3), dpi=120)
        plt.plot(
            range(int(data.shape[0])),
            data_1D,
            linestyle="-",
            color="blue",
            label="1D slice",
        )
        plt.axhline(y=half_max, color="black", linestyle="--", label="1/2 Maximum Value")
        plt.plot(
            [0, -0.015],
            [half_max, half_max],
            color="black",
            linewidth=1,
            transform=plt.gca().get_yaxis_transform(),
            clip_on=False,
        )
        plt.text(
            -0.02,
            half_max,
            f"{half_max:.2f}",
            color="black",
            fontsize=14,
            va="center",
            ha="right",
            transform=plt.gca().get_yaxis_transform(),
        )
        ax = plt.gca()
        for i, idx in enumerate(half_max_indices):
            plt.axvline(x=idx, color="red", linestyle="--", linewidth=1)
            x_val = x_divide[idx]
            y_pos = 0.02 if len(half_max_indices) <= 2 or i % 2 == 0 else 0.08
            ax.text(
                idx,
                y_pos,
                f"{xx}=${x_val:.2f}$",
                color="red",
                fontsize=12,
                rotation=0,
                va="bottom",
                ha="center",
                transform=ax.get_xaxis_transform(),
            )

        plt.xlabel(xx, fontsize=24)
        plt.tick_params(axis="y", labelsize=12)
        n_ticks = 7
        x_labels = np.round(np.linspace(np.min(X), np.max(X), n_ticks), decimals=2)
        plt.xticks(
            ticks=np.linspace(0, data.shape[1] - 1, n_ticks),
            labels=x_labels,
            rotation=0,
            fontsize=12,
        )
        plt.savefig(full_output_path2, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {full_output_path2}")

    plt.show()
    plt.close("all")


def visualize_3d_grid(
    cells,
    title="3D Adaptive Grid",
    views=None,
    alpha=0.9,
    noaxis=None,
    output_directory=".",
    output_filename=None,
):
    if views is None:
        views = [(0, 90), (90, 180), (0, 180)]

    full_output_path = None
    if output_filename:
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        elif not output_directory:
            output_directory = "."
        full_output_path = _resolve_output_path(output_directory, output_filename)

    if not cells:
        print("No cells to visualize.")
        return

    visualizer = MeshVisualizer3D()
    leaf_cells = visualizer.collect_leaf_cells(cells)
    if not leaf_cells:
        print("No leaf cells to visualize.")
        return

    segments = []
    colors = []

    levels = [c.level for c in leaf_cells]
    min_level = min(levels) if levels else 0
    max_level = max(levels) if levels else 1
    cmap = plt.get_cmap("jet", max_level - min_level + 1)

    print(f"Plotting {len(leaf_cells)} cells. Levels: {min_level} to {max_level}")

    for cell in leaf_cells:
        x, y, z = cell.x0, cell.y0, cell.z0
        s = cell.size
        lvl = cell.level
        c = cmap((lvl - min_level) / (max_level - min_level + 1e-6))

        if lvl == min_level:
            current_alpha = 0.05
        elif lvl == min_level + 1:
            current_alpha = 0.2
        else:
            current_alpha = alpha

        cell_segments = [
            [(x, y, z), (x + s, y, z)],
            [(x + s, y, z), (x + s, y + s, z)],
            [(x + s, y + s, z), (x, y + s, z)],
            [(x, y + s, z), (x, y, z)],
            [(x, y, z + s), (x + s, y, z + s)],
            [(x + s, y, z + s), (x + s, y + s, z + s)],
            [(x + s, y + s, z + s), (x, y + s, z + s)],
            [(x, y + s, z + s), (x, y, z + s)],
            [(x, y, z), (x, y, z + s)],
            [(x + s, y, z), (x + s, y, z + s)],
            [(x + s, y + s, z), (x + s, y + s, z + s)],
            [(x, y + s, z), (x, y, z + s)],
        ]
        segments.extend(cell_segments)
        c_rgba = list(c)
        c_rgba[3] = current_alpha
        colors.extend([c_rgba] * 12)

    all_x = [c.x0 for c in leaf_cells] + [c.x0 + c.size for c in leaf_cells]
    all_y = [c.y0 for c in leaf_cells] + [c.y0 + c.size for c in leaf_cells]
    all_z = [c.z0 for c in leaf_cells] + [c.z0 + c.size for c in leaf_cells]
    xlims = (min(all_x), max(all_x))
    ylims = (min(all_y), max(all_y))
    zlims = (min(all_z), max(all_z))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_level, vmax=max_level))
    sm.set_array([])

    if isinstance(noaxis, (list, tuple)):
        axis_modes = list(noaxis)
    elif noaxis is None:
        axis_modes = [None] * len(views)
    else:
        axis_modes = [noaxis] * len(views)

    def _plot_axis(ax, elev, azim, axis_mode=None, single_view=False):
        lc = Line3DCollection(segments, colors=colors, linewidths=0.5)
        ax.add_collection3d(lc)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)

        if single_view:
            ax.tick_params(axis="both", which="major", labelsize=16, pad=6)
            ax.tick_params(axis="x", which="major", labelsize=16, pad=6)
            ax.set_xlabel(r"$x_1$", fontsize=25, labelpad=20)
            ax.set_ylabel(r"$x_2$", fontsize=25, labelpad=20)
            ax.set_zlabel(r"$x_3$", fontsize=25, labelpad=15)
        else:
            ax.tick_params(axis="both", which="major", labelsize=12, pad=3)
            ax.set_xlabel(r"$x_1$", fontsize=17, labelpad=8)
            ax.set_ylabel(r"$x_2$", fontsize=17, labelpad=9)
            ax.set_zlabel(r"$x_3$", fontsize=17, labelpad=6)

        if axis_mode == "x1":
            ax.tick_params(axis="x", which="major", labelsize=0, pad=6)
            ax.set_xlabel("", fontsize=25 if single_view else 17, labelpad=15)
        elif axis_mode == "x2":
            ax.tick_params(axis="y", which="major", labelsize=0, pad=6)
            ax.set_ylabel("", fontsize=25 if single_view else 17, labelpad=15)
        elif axis_mode == "x3":
            ax.tick_params(axis="z", which="major", labelsize=0, pad=6)
            ax.set_zlabel("", fontsize=25 if single_view else 17, labelpad=15)

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, linestyle=":", alpha=0.2)
        try:
            ax.dist = 10
        except Exception:
            pass

    if len(views) == 1:
        fig = plt.figure(figsize=(10, 10))
        elev, azim = views[0] if views else (30, -60)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_position([0.01, 0.02, 0.75, 0.96])
        _plot_axis(ax, elev, azim, axis_modes[0] if axis_modes else None, single_view=True)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    else:
        fig = plt.figure(figsize=(6.5 * len(views), 6))
        axes = []
        for idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, len(views), idx + 1, projection="3d")
            axis_mode = axis_modes[idx] if idx < len(axis_modes) else None
            _plot_axis(ax, elev, azim, axis_mode, single_view=False)
            ax.set_title(f"View {idx + 1}", fontsize=16)
            axes.append(ax)
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.90, wspace=0.02)
        cax = fig.add_axes([0.92, 0.20, 0.012, 0.58])
        cbar = fig.colorbar(sm, cax=cax)

    cbar.ax.tick_params(labelsize=20)
    try:
        cbar.set_ticks(range(min_level, max_level + 1))
    except Exception:
        pass

    if full_output_path:
        plt.savefig(full_output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()

"""
# Add this after adaptive refinement is completed:
print("Starting visualization...")
visualize_adaptive_refinement(
    cells=cells,                    # Initial mesh
    refined_cells=refined_cells,    # Refined mesh
    all_g_p=all_g_p,                # Gauss points
    S_num=S_num,                    # Numerical solution
    grad_S_num=grad_S_num,          # Gradient
    refinement_stats=refinement_stats,  # Refinement statistics
    cell_indicators=amr.compute_cell_indicators(...)  # Indicator values
)
"""

# Q-filter visualizations migrated from q_filter.py.
def map_index_to_domain_3d(points, domain_min, domain_max, shape):
    """
    Map 3D index points to the physical domain.
    :param points: Index-point array with shape (n, 3).
    :param domain_min: Coordinate-wise lower bound (x_min, y_min, z_min).
    :param domain_max: Coordinate-wise upper bound (x_max, y_max, z_max).
    :param shape: 3D array shape (depth, height, width).
    :return: Mapped point array.
    """
    scales = (np.array(domain_max) - np.array(domain_min)) / (np.array(shape) - 1)

    mapped_points = np.zeros_like(points, dtype=float)
    mapped_points[:, 0] = domain_min[0] + points[:, 0] * scales[0]
    mapped_points[:, 1] = domain_min[1] + points[:, 1] * scales[1]
    mapped_points[:, 2] = domain_min[2] + points[:, 2] * scales[2]
    return mapped_points


def set_axes_equal(ax):
    """Use equal tick spacing for the x, y, and z axes in a 3D plot."""
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


def _infer_layout(domain_min, domain_max, layout):
    if layout != "auto":
        return layout
    domain_min = np.asarray(domain_min, dtype=float)
    domain_max = np.asarray(domain_max, dtype=float)
    if np.allclose(domain_min, -domain_max):
        return "donut"
    return "ex47"


def _iter_centers(center):
    center = np.asarray(center)
    if center.size == 0:
        return np.empty((0, 3))
    return np.atleast_2d(center)


def _normalize_axis_key(noaxis):
    if noaxis is None:
        return None
    key = str(noaxis).strip().lower()
    key = key.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    key = key.replace("_", "").replace(" ", "")
    aliases = {
        "x": "x1",
        "x1": "x1",
        "y": "x2",
        "x2": "x2",
        "z": "x3",
        "x3": "x3",
    }
    return aliases.get(key, key)


def _latex_number(value):
    value = float(value)
    if np.isclose(value, 0.0):
        value = 0.0
    return format(value, "g")


def _compact_tick_label(value):
    value = float(value)
    if np.isclose(value, 0.0):
        value = 0.0
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _compact_tick_formatter(value, _pos=None):
    return _compact_tick_label(value)


def _set_compact_axis_ticks(axis_obj, set_ticks, ticks, hide_endpoints=False, hide_first=False, hide_last=False):
    if ticks is None:
        axis_obj.set_major_formatter(FuncFormatter(_compact_tick_formatter))
        return
    set_ticks(ticks)
    labels = [_compact_tick_label(value) for value in ticks]
    if hide_endpoints and len(labels) >= 2:
        labels[0] = ""
        labels[-1] = ""
    if hide_first and labels:
        labels[0] = ""
    if hide_last and labels:
        labels[-1] = ""
    axis_obj.set_ticklabels(labels)


def _latex_fixed_number(value, decimals=2):
    value = float(value)
    if np.isclose(value, 0.0):
        value = 0.0
    return f"{value:.{decimals}f}"


def _latex_tick_formatter(value, _pos=None):
    return rf"${_latex_number(value)}$"


def _latex_center_label(index, center_point):
    return (
        rf"$\boldsymbol{{c}}_{{{index}}}"
        rf"=\left({_latex_fixed_number(center_point[0])},\,"
        rf"{_latex_fixed_number(center_point[1])},\,"
        rf"{_latex_fixed_number(center_point[2])}\right)$"
    )


def _plot_center_markers(ax, center, noaxis, domain_min, domain_max, layout):
    noaxis = _normalize_axis_key(noaxis)
    centers = _iter_centers(center)
    if centers.size == 0:
        return

    if layout == "donut":
        if noaxis == "x1":
            ax.tick_params(axis="x", which="major", labelsize=0, pad=6)
            ax.set_xlabel("", fontsize=25, labelpad=15)
            for i, center_point in enumerate(centers):
                ax.scatter(
                    center_point[0],
                    center_point[1],
                    center_point[2],
                    color="black",
                    s=100,
                    marker="x",
                    zorder=200,
                )
                ax.text(
                    center_point[0] + 0.3,
                    center_point[1] + 0.34,
                    center_point[2] - 0.1,
                    _latex_center_label(i + 1, center_point),
                    color="black",
                    fontsize=16,
                    ha="left",
                    va="center",
                    zorder=200,
                )
                ax.plot(
                    [center_point[0] + 0.1, center_point[0]],
                    [center_point[1], center_point[1]],
                    [center_point[2] - 0.05, center_point[2]],
                    color="black",
                    linestyle="-",
                    linewidth=2,
                    zorder=200,
                )
        elif noaxis == "x2":
            for i, center_point in enumerate(centers):
                ax.scatter(
                    center_point[0],
                    center_point[1],
                    center_point[2],
                    color="black",
                    s=100,
                    marker="x",
                    zorder=200,
                )
                ax.set_zlabel("", fontsize=0, labelpad=0)
                ax.text(
                    domain_max[0] + 0.1,
                    domain_min[1] - 0.01,
                    domain_max[2] + 0.05,
                    "$x_3$",
                    fontsize=25,
                    ha="center",
                )
                ax.text(
                    center_point[0] + 0.3,
                    center_point[1] - 0.2,
                    center_point[2] - 0.1,
                    _latex_center_label(i + 1, center_point),
                    color="black",
                    fontsize=16,
                    ha="left",
                    va="center",
                    zorder=200,
                )
                ax.plot(
                    [center_point[0] + 0.1, center_point[0]],
                    [center_point[1], center_point[1]],
                    [center_point[2] - 0.05, center_point[2]],
                    color="black",
                    linestyle="-",
                    linewidth=2,
                    zorder=200,
                )
        elif noaxis == "x3":
            ax.tick_params(axis="z", which="major", labelsize=0, pad=6)
            ax.set_zlabel("", fontsize=25, labelpad=15)
            ax.set_xlabel("$x_1$", fontsize=25, labelpad=30)
            ax.tick_params(axis="x", which="major", labelsize=18, pad=15)
            ax.tick_params(axis="y", which="major", labelsize=18, pad=12)
            for i, center_point in enumerate(centers):
                ax.scatter(
                    center_point[0],
                    center_point[1],
                    center_point[2],
                    color="black",
                    s=100,
                    marker="x",
                    zorder=200,
                )
                ax.text(
                    center_point[0] + 0.15,
                    center_point[1] - 0.32,
                    center_point[2] - 0.1,
                    _latex_center_label(i + 1, center_point),
                    color="black",
                    fontsize=16,
                    ha="left",
                    va="center",
                    zorder=200,
                )
                ax.plot(
                    [center_point[0] + 0.1, center_point[0]],
                    [center_point[1], center_point[1]],
                    [center_point[2] - 0.05, center_point[2]],
                    color="black",
                    linestyle="-",
                    linewidth=2,
                    zorder=200,
                )
        return

    if noaxis == "x1":
        ax.tick_params(axis="x", which="major", labelsize=0, pad=6)
        ax.set_xlabel("", fontsize=25, labelpad=15)
        ax.set_zticks([tick for tick in ax.get_zticks() if tick != 0])
        for i, center_point in enumerate(centers):
            ax.scatter(
                center_point[0],
                center_point[1],
                center_point[2],
                color="black",
                s=100,
                marker="x",
                zorder=200,
            )
            ax.text(
                center_point[0] + 0.4,
                center_point[1] + 0.4,
                center_point[2] - 0.1,
                _latex_center_label(i + 1, center_point),
                color="black",
                fontsize=16,
                ha="left",
                va="center",
                zorder=200,
            )
            ax.plot(
                [center_point[0] + 0.05, center_point[0]],
                [center_point[1] + 0.05, center_point[1]],
                [center_point[2] - 0.05, center_point[2]],
                color="black",
                linestyle="-",
                linewidth=2,
                zorder=200,
            )
    elif noaxis == "x2":
        ax.tick_params(axis="y", which="major", labelsize=0, pad=6)
        ax.set_ylabel("", fontsize=25, labelpad=20)
        ax.set_zlim([domain_min[2], domain_max[2]])
        ax.set_zticks([tick for tick in ax.get_zticks() if tick != 0])
        ax.tick_params(axis="x", which="major", labelsize=18, pad=3)
        ax.tick_params(axis="z", which="major", labelsize=18, pad=10)
        ax.set_zlabel("$x_3$", fontsize=25, labelpad=20)
        for i, center_point in enumerate(centers):
            ax.scatter(
                center_point[0],
                center_point[1],
                center_point[2],
                color="black",
                s=100,
                marker="x",
                zorder=200,
            )
            ax.text(
                center_point[0] + 0.4,
                center_point[1] + 0.4,
                center_point[2] - 0.1,
                _latex_center_label(i + 1, center_point),
                color="black",
                fontsize=16,
                ha="left",
                va="center",
                zorder=200,
            )
            ax.plot(
                [center_point[0] + 0.05, center_point[0]],
                [center_point[1] + 0.05, center_point[1]],
                [center_point[2] - 0.05, center_point[2]],
                color="black",
                linestyle="-",
                linewidth=2,
                zorder=200,
            )
    elif noaxis == "x3":
        ax.tick_params(axis="z", which="major", labelsize=0, pad=6)
        ax.set_zlabel("", fontsize=25, labelpad=0)
        ax.set_xlabel("$x_1$", fontsize=25, labelpad=25)
        ax.tick_params(axis="x", which="major", labelsize=18, pad=10)
        ax.tick_params(axis="y", which="major", labelsize=18, pad=3)
        ax.set_xticks([tick for tick in ax.get_zticks() if tick != 0])
        for i, center_point in enumerate(centers):
            ax.scatter(
                center_point[0],
                center_point[1],
                center_point[2],
                color="black",
                s=100,
                marker="x",
                zorder=200,
            )
            ax.text(
                center_point[0] + 0.1,
                center_point[1] + 0.3,
                center_point[2] - 0.1,
                _latex_center_label(i + 1, center_point),
                color="black",
                fontsize=16,
                ha="left",
                va="center",
                zorder=200,
            )
            ax.plot(
                [center_point[0] + 0.05, center_point[0]],
                [center_point[1] + 0.05, center_point[1]],
                [center_point[2] - 0.05, center_point[2]],
                color="black",
                linestyle="-",
                linewidth=2,
                zorder=200,
            )


def _apply_3d_axis_style(
    ax,
    noaxis,
    domain_min=None,
    domain_max=None,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    xlabel=r"$x_1$",
    ylabel=r"$x_2$",
    zlabel=r"$x_3$",
    tick_size=18,
    label_size=25,
    x_tick_pad=8,
    y_tick_pad=8,
    z_tick_pad=8,
    x_label_pad=15,
    y_label_pad=20,
    z_label_pad=15,
    zlabel_position="default",
    hide_duplicate_corner_ticks=False,
):
    noaxis = _normalize_axis_key(noaxis)
    hide_x_endpoints = False
    hide_y_endpoints = hide_duplicate_corner_ticks and noaxis is None
    hide_z_endpoints = False
    hide_x_first = hide_duplicate_corner_ticks and noaxis == "x3"
    hide_y_first = False
    hide_z_first = hide_duplicate_corner_ticks and noaxis == "x1"

    if noaxis != "x1":
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=x_label_pad)
        ax.tick_params(axis="x", which="major", labelsize=tick_size, pad=x_tick_pad)
        _set_compact_axis_ticks(ax.xaxis, ax.set_xticks, x_ticks, hide_x_endpoints, hide_first=hide_x_first)
    else:
        ax.tick_params(axis="x", which="major", labelsize=0, pad=6)
        ax.set_xlabel("", fontsize=0, labelpad=0)

    if noaxis != "x2":
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=y_label_pad)
        ax.tick_params(axis="y", which="major", labelsize=tick_size, pad=y_tick_pad)
        _set_compact_axis_ticks(ax.yaxis, ax.set_yticks, y_ticks, hide_y_endpoints, hide_first=hide_y_first)
    else:
        ax.tick_params(axis="y", which="major", labelsize=0, pad=6)
        ax.set_ylabel("", fontsize=0, labelpad=0)

    if noaxis != "x3":
        if zlabel_position == "upper_left":
            ax.set_zlabel("", fontsize=0, labelpad=0)
            ax.text2D(-0.06, 0.72, zlabel, transform=ax.transAxes, fontsize=label_size)
        elif zlabel_position == "axis_top":
            ax.set_zlabel("", fontsize=0, labelpad=0)
            if domain_min is None or domain_max is None:
                ax.set_zlabel(zlabel, fontsize=label_size, labelpad=z_label_pad)
            else:
                domain_min = np.asarray(domain_min, dtype=float)
                domain_max = np.asarray(domain_max, dtype=float)
                ax.text(
                    domain_max[0],
                    domain_min[1],
                    domain_max[2] + 0.05 * (domain_max[2] - domain_min[2]),
                    zlabel,
                    fontsize=label_size,
                    ha="center",
                    va="bottom",
                )
        else:
            ax.set_zlabel(zlabel, fontsize=label_size, labelpad=z_label_pad)
        ax.tick_params(axis="z", which="major", labelsize=tick_size, pad=z_tick_pad)
        _set_compact_axis_ticks(ax.zaxis, ax.set_zticks, z_ticks, hide_z_endpoints, hide_first=hide_z_first)
    else:
        ax.tick_params(axis="z", which="major", labelsize=0, pad=6)
        ax.set_zlabel("", fontsize=0, labelpad=0)


def _plot_grad_view(mapped_mask_points, grad_values, domain_min, domain_max, elev, azim, full_output_path, dpi, layout):
    norm = plt.Normalize(vmin=np.min(grad_values), vmax=np.max(grad_values))

    donut_cmap = LinearSegmentedColormap.from_list(
        "donut_cmap",
        ["#fa8072", "#e84141", "#f23737"],
        N=256,
    )
    colors_grad = donut_cmap(norm(grad_values))
    norm_grad = plt.Normalize(vmin=np.min(grad_values), vmax=np.max(grad_values))
    cmap_grad = plt.get_cmap("rainbow")
    colors_grad = cmap_grad(norm_grad(grad_values))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_xlim([domain_min[0], domain_max[0]])
    ax.set_ylim([domain_min[1], domain_max[1]])
    ax.set_zlim([domain_min[2], domain_max[2]])

    if layout == "donut":
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="z", which="major", labelsize=16)
        ax.set_xlabel("X", fontsize=14, labelpad=10)
        ax.set_ylabel("Y", fontsize=14, labelpad=10)
        ax.set_zlabel("Z", fontsize=14, labelpad=10)
    else:
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlabel("X", fontsize=18, labelpad=10)
        ax.set_ylabel("Y", fontsize=18, labelpad=10)
        ax.set_zlabel("Z", fontsize=18, labelpad=10)

    ax.view_init(elev=elev, azim=azim)
    ax.scatter(
        mapped_mask_points[:, 0],
        mapped_mask_points[:, 1],
        mapped_mask_points[:, 2],
        c=colors_grad,
        s=3,
        alpha=0.6,
        edgecolors="none",
    )

    cmap = plt.get_cmap("rainbow")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.001)
    cbar.ax.tick_params(labelsize=18)

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(full_output_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def _plot_s_view(
    mapped_mask_points,
    s_values,
    domain_min,
    domain_max,
    elev,
    azim,
    center,
    noaxis,
    full_output_path,
    dpi,
    layout,
    surface_volume=None,
    surface_level=None,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    colorbar_ticks=None,
    xlabel=r"$x_1$",
    ylabel=r"$x_2$",
    zlabel=r"$x_3$",
    colorbar_label=r"$S^{(0)}$",
    cmap="viridis",
    show_grid=True,
    x_tick_pad=8,
    y_tick_pad=8,
    z_tick_pad=8,
):
    norm = plt.Normalize(vmin=np.min(s_values), vmax=np.max(s_values))
    norm_s = plt.Normalize(vmin=np.min(s_values), vmax=np.max(s_values))
    cmap_s = plt.get_cmap(cmap)
    colors_s = cmap_s(norm_s(s_values))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_position([0.01, 0.02, 0.75, 0.96])
    ax.set_xlim([domain_min[0], domain_max[0]])
    ax.set_ylim([domain_min[1], domain_max[1]])
    ax.set_zlim([domain_min[2], domain_max[2]])

    if layout == "donut":
        ax.tick_params(axis="both", which="major", labelsize=16, pad=6)
        ax.tick_params(axis="x", which="major", labelsize=16, pad=6)
        ax.set_xlabel(xlabel, fontsize=25, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=25, labelpad=20)
        ax.set_zlabel(zlabel, fontsize=25, labelpad=15)
        set_axes_equal(ax)
    else:
        ax.tick_params(axis="both", which="major", labelsize=18, pad=6)
        ax.set_xlabel(xlabel, fontsize=25, labelpad=15)
        ax.set_ylabel(ylabel, fontsize=25, labelpad=20)
        ax.set_zlabel(zlabel, fontsize=25, labelpad=15)

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    # _plot_center_markers(ax, center, noaxis, domain_min, domain_max, layout)
    _apply_3d_axis_style(
        ax,
        noaxis,
        domain_min=domain_min,
        domain_max=domain_max,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        x_tick_pad=x_tick_pad,
        y_tick_pad=y_tick_pad,
        z_tick_pad=z_tick_pad,
    )
    ax.dist = 10

    if surface_volume is not None and surface_level is not None:
        visualizer = MeshVisualizer3D()
        visualizer.domain_min = domain_min
        visualizer.domain_max = domain_max
        visualizer.plot_isosurface(
            surface_volume,
            level=surface_level,
            color="#6CB6E9",
            alpha=0.22,
            show_wireframe=False,
            show_center=False,
            center=None,
            elev=elev,
            azim=azim,
            save_path=None,
            ax=ax,
        )

    ax.scatter(
        mapped_mask_points[:, 0],
        mapped_mask_points[:, 1],
        mapped_mask_points[:, 2],
        c=colors_s,
        s=3,
        alpha=0.6,
        edgecolors="none",
    )

    cmap_obj = plt.get_cmap(cmap)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label(colorbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)

    ax.grid(show_grid, linestyle="--", alpha=0.3)
    plt.savefig(full_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def _plot_surface_only_view(
    surface_volume,
    surface_level,
    surface_color_values,
    domain_min,
    domain_max,
    elev,
    azim,
    center,
    noaxis,
    full_output_path,
    dpi,
    layout,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    colorbar_ticks=None,
    xlabel=r"$x_1$",
    ylabel=r"$x_2$",
    zlabel=r"$x_3$",
    colorbar_label=r"$S^{(0)}$",
    cmap="viridis",
    show_grid=True,
    x_tick_pad=8,
    y_tick_pad=8,
    z_tick_pad=8,
    x_label_pad=15,
    y_label_pad=20,
    z_label_pad=15,
    surface_stride=1,
    zlabel_position="default",
    show_colorbar=True,
    axis_label_size=30,
    axis_tick_size=20,
    hide_duplicate_corner_ticks=False,
    surface_rasterized=False,
):
    surface_stride = max(1, int(surface_stride))
    if surface_stride > 1:
        surface_volume = surface_volume[::surface_stride, ::surface_stride, ::surface_stride]
        if surface_color_values is not None:
            surface_color_values = surface_color_values[::surface_stride, ::surface_stride, ::surface_stride]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_position([0.02, 0.02, 0.80, 0.96])
    ax.set_xlim([domain_min[0], domain_max[0]])
    ax.set_ylim([domain_min[1], domain_max[1]])
    ax.set_zlim([domain_min[2], domain_max[2]])

    if layout == "donut":
        ax.tick_params(axis="both", which="major", labelsize=16, pad=6)
        ax.tick_params(axis="x", which="major", labelsize=16, pad=6)
        ax.set_xlabel(xlabel, fontsize=25, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=25, labelpad=20)
        ax.set_zlabel(zlabel, fontsize=25, labelpad=15)
        set_axes_equal(ax)
    else:
        ax.tick_params(axis="both", which="major", labelsize=18, pad=6)
        ax.set_xlabel(xlabel, fontsize=25, labelpad=15)
        ax.set_ylabel(ylabel, fontsize=25, labelpad=20)
        ax.set_zlabel(zlabel, fontsize=25, labelpad=15)

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(show_grid, linestyle="--", alpha=0.3)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.45))

    visualizer = MeshVisualizer3D()
    visualizer.domain_min = domain_min
    visualizer.domain_max = domain_max
    visualizer.plot_isosurface(
        surface_volume,
        level=surface_level,
        alpha=0.95,
        show_wireframe=False,
        show_center=False,
        center=None,
        elev=elev,
        azim=azim,
        save_path=None,
        ax=ax,
        scalar_field=surface_color_values,
        cmap=cmap,
        show_colorbar=show_colorbar,
        colorbar_ticks=colorbar_ticks,
        colorbar_label=colorbar_label,
        rasterized=surface_rasterized,
    )

    # _plot_center_markers(ax, center, noaxis, domain_min, domain_max, layout)
    _apply_3d_axis_style(
        ax,
        noaxis,
        domain_min=domain_min,
        domain_max=domain_max,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        tick_size=axis_tick_size,
        label_size=axis_label_size,
        x_tick_pad=x_tick_pad,
        y_tick_pad=y_tick_pad,
        z_tick_pad=z_tick_pad,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        z_label_pad=z_label_pad,
        zlabel_position=zlabel_position,
        hide_duplicate_corner_ticks=hide_duplicate_corner_ticks,
    )

    plt.savefig(full_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.show()


def filter_q_3d(
    grad,
    S_num,
    domain_min,
    domain_max,
    para_grad,
    para_abs,
    elev,
    azim,
    center,
    choose,
    noaxis,
    output_directory=".",
    output_filename="plot_3d.png",
    dpi=300,
    layout="auto",
    render_mode="overlay",
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    colorbar_ticks=None,
    xlabel=r"$x_1$",
    ylabel=r"$x_2$",
    zlabel=r"$x_3$",
    colorbar_label=r"$S^{(0)}$",
    cmap="viridis",
    show_grid=True,
    x_tick_pad=8,
    y_tick_pad=8,
    z_tick_pad=8,
    x_label_pad=15,
    y_label_pad=20,
    z_label_pad=15,
    surface_stride=1,
    zlabel_position="default",
    show_colorbar=True,
    axis_label_size=30,
    axis_tick_size=20,
    hide_duplicate_corner_ticks=False,
    surface_rasterized=False,
):
    if dpi is None:
        dpi = 300

    noaxis = _normalize_axis_key(noaxis)

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    full_output_path = _resolve_output_path(output_directory, output_filename)

    layout = _infer_layout(domain_min, domain_max, layout)
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
    min_z_point = mapped_mask_points[np.argmin(mapped_mask_points[:, 2])]
    max_z_point = mapped_mask_points[np.argmax(mapped_mask_points[:, 2])]

    if choose == "grad":
        grad_values = grad[mask]
        _plot_grad_view(
            mapped_mask_points,
            grad_values,
            domain_min,
            domain_max,
            elev,
            azim,
            full_output_path,
            dpi,
            layout,
        )
    elif choose == "S":
        s_values = S_num[mask]
        if render_mode == "surface_only":
            _plot_surface_only_view(
                surface_volume=mask.astype(float),
                surface_level=0.5,
                surface_color_values=S_num,
                domain_min=domain_min,
                domain_max=domain_max,
                elev=elev,
                azim=azim,
                center=center,
                noaxis=noaxis,
                full_output_path=full_output_path,
                dpi=dpi,
                layout=layout,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                z_ticks=z_ticks,
                colorbar_ticks=colorbar_ticks,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                colorbar_label=colorbar_label,
                cmap=cmap,
                show_grid=show_grid,
                x_tick_pad=x_tick_pad,
                y_tick_pad=y_tick_pad,
                z_tick_pad=z_tick_pad,
                x_label_pad=x_label_pad,
                y_label_pad=y_label_pad,
                z_label_pad=z_label_pad,
                surface_stride=surface_stride,
                zlabel_position=zlabel_position,
                show_colorbar=show_colorbar,
                axis_label_size=axis_label_size,
                axis_tick_size=axis_tick_size,
                hide_duplicate_corner_ticks=hide_duplicate_corner_ticks,
                surface_rasterized=surface_rasterized,
            )
        else:
            _plot_s_view(
                mapped_mask_points,
                s_values,
                domain_min,
                domain_max,
                elev,
                azim,
                center,
                noaxis,
                full_output_path,
                dpi,
                layout,
                surface_volume=mask.astype(float),
                surface_level=0.5,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                z_ticks=z_ticks,
                colorbar_ticks=colorbar_ticks,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                colorbar_label=colorbar_label,
                cmap=cmap,
                show_grid=show_grid,
                x_tick_pad=x_tick_pad,
                y_tick_pad=y_tick_pad,
                z_tick_pad=z_tick_pad,
            )

    return (
        mapped_mask_points,
        min_x_point,
        max_x_point,
        min_y_point,
        max_y_point,
        min_z_point,
        max_z_point,
    )


def detect_3d(*args, **kwargs):
    return filter_q_3d(*args, **kwargs)
