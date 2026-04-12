import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

try:
    import bound_detect
except Exception:  # pragma: no cover - optional example dependency
    bound_detect = None


def kidney_curve(x, y, h, k, a):
    x_shifted = x - h
    y_shifted = y - k
    if a == 0:
        a = 1e-9
    term1 = (x_shifted**2 + y_shifted**2 - 4 * a**2) ** 3
    term2 = 108 * a**4 * y_shifted**2
    return term1 - term2


def draw(X, Y, data, *args, output_directory=None, output_filename=None, cmap=None):
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
    if cmap is None:
        cmap = "rainbow"

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

    cbar = plt.colorbar(contour_fill, shrink=1, aspect=12)
    cbar.ax.tick_params(labelsize=18)

    plt.xlabel(r"$x_1$", fontsize=30)
    plt.ylabel(r"$x_2$", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.gca().set_facecolor("white")
    plt.tight_layout()

    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {full_output_path}")
    except Exception as exc:
        print(f"Error saving plot to {full_output_path}: {exc}")

    plt.show()
    plt.close()


def improved_plot(data, title, x_min, x_max, y_min, y_max, xlabel="x1", ylabel="x2", cmap="plasma", n_ticks=5):
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
                contour = ax.tricontourf(p_np[:, 0], p_np[:, 1], s_np, levels=20, cmap="rainbow", alpha=0.9)
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
            contour1 = ax1.tricontourf(points[:, 0], points[:, 1], solution, levels=14, cmap="viridis", alpha=0.9)
            ax1.set_title("Numerical Solution $S_{num}$ Distribution")
            fig.colorbar(contour1, ax=ax1, label="Solution Value")
        else:
            ax1.text(0.5, 0.5, "Not enough points for tricontourf", ha="center", va="center")

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect("equal", adjustable="box")

        if points.shape[0] >= 3:
            contour2 = ax2.tricontourf(points[:, 0], points[:, 1], grad_norm, levels=14, cmap="plasma", alpha=0.9)
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
        self.contour_points_implicit = self._load_true_boundary()

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
        if kidney_bound is not None or self.contour_points_implicit is not None:
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
            contour = ax.tricontourf(p_np[:, 0], p_np[:, 1], s_np, levels=20, cmap="rainbow", alpha=0.9)
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
        elif style == "kidney" and self.contour_points_implicit is not None:
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        plt.show()
        return fig

    def _scale(self, fig, ax, kidney_bound):
        if self.contour_points_implicit is None or bound_detect is None:
            raise RuntimeError("Kidney boundary helpers are unavailable.")

        offset_distances = np.linspace(-0.02, 0.02, 2)
        contour_points_scaled = bound_detect.simple_offset_batch(kidney_bound, 2e-3, offset_distances)
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
        plt.tight_layout(rect=[0, 0, 1, 1])
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
