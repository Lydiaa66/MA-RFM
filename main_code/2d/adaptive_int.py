import re
import numpy as np
import torch
import mesh
import copy
import sys

class AdaptiveMeshRefinementFix:
    def __init__(self, cells, model_evaluator, nx, device):
        """
        New Adaptive Mesh Refinement Strategy
        
        Args:
            cells: Initial list of cells (will be modified in-place if refined)
            model_evaluator: Function that takes (N, 2) points and returns (N, ) values (the quantity to integrate)
            nx: Number of Gauss points per direction
            device: torch device
        """
        self.cells = cells
        self.model_evaluator = model_evaluator
        self.nx = nx
        self.device = device

    def _compute_multidim_thresholds(
        self,
        cell_metrics,
        total_diff_grad,
        total_abs_fine,
        total_grad_fine,
        threshold_S,
        threshold_grad,
        threshold_noise,
        refine_mode,
    ):
        dynamic_threshold_abs = total_abs_fine * threshold_S
        dynamic_threshold_noise = total_grad_fine * threshold_noise
        if refine_mode == "signed":
            dynamic_threshold_grad = total_grad_fine * threshold_grad
        else:
            dynamic_threshold_grad = total_diff_grad * threshold_grad
        return dynamic_threshold_abs, dynamic_threshold_grad, dynamic_threshold_noise

    def _should_refine_multidim(self, metrics, dynamic_threshold_abs, dynamic_threshold_grad, dynamic_threshold_noise, refine_mode):
        indicator_S_fine = metrics["indicator_S_fine"]
        indicator_grad_fine = metrics["indicator_grad_fine"]
        # Keep dynamic_threshold_grad/refine_mode in the signature for notebook compatibility.
        del dynamic_threshold_grad, refine_mode

        cond1 = indicator_S_fine > dynamic_threshold_abs
        cond4 = indicator_grad_fine > dynamic_threshold_noise
        return cond1 or cond4

    def refine_by_integral_difference(
        self,
        threshold_S,
        threshold_grad,
        threshold_noise,
        current_maxiter,
        max_level=10,
        refine_mode="standard",
    ):
        """
        Refine cells if the difference between coarse and fine integral is larger than threshold.
        Iterative breadth-first-search (BFS) approach.
        1. For each cell, compute coarse integral using its Gauss points.
        2. Virtually refine cell into 4 children, compute fine integral using children's Gauss points.
        3. If |fine - coarse| > threshold, actually refine the cell.
        4. Repeat until no more cells to refine or max_level reached.
        """
        current_level_cells = mesh.collect_leaf_cells(self.cells) 
        queue = current_level_cells 
        iter=0
        while iter<current_maxiter:
            iter+=1
            next_generation = [] 
            cells_to_process = [] 
            points_coarse = []
            info_coarse = []
            idx_counter = 0
            for cell in queue:
                if cell.level >= max_level:
                    continue
                
                if cell.gauss_points is None:
                    cell.pre_calculate(self.nx)
                
                p = cell.gauss_points
                
                cells_to_process.append(cell)
                points_coarse.append(p)
                info_coarse.append((len(cells_to_process)-1, idx_counter, idx_counter + len(p)))
                idx_counter += len(p)
                
            if not cells_to_process:
                break
                
            all_p_coarse = np.vstack(points_coarse)
            vals_coarse_all = self.model_evaluator(all_p_coarse)
            
            points_fine = []
            weights_fine = []
            info_fine = []
            
            idx_counter_fine = 0
            children_store = {}
            
            for i, cell in enumerate(cells_to_process):
                h = cell.size / 2
                temps_children = []
                for dx in [0, h]:
                    for dy in [0, h]:
                        child = mesh.Cell(cell.x0 + dx, cell.y0 + dy, h, cell.x, cell.level + 1)
                        child.pre_calculate(self.nx)
                        temps_children.append(child)
                        points_fine.append(child.gauss_points)
                        weights_fine.append(child.w)
                        
                children_store[i] = temps_children
                
                num_p = 4 * (self.nx * self.nx)
                info_fine.append((i, idx_counter_fine, idx_counter_fine + num_p))
                idx_counter_fine += num_p
                
            all_p_fine = np.vstack(points_fine)
            all_w_fine = np.vstack(weights_fine).flatten()
            vals_fine_all = self.model_evaluator(all_p_fine)
            
            refinement_count = 0
            
            is_multidim = vals_coarse_all.ndim > 1 and vals_coarse_all.shape[-1] > 1

            # --- First pass: compute the L2-norm integral and error for every cell using the new formula. ---
            cell_metrics = []
            total_diff_grad = 0.0
            total_abs_fine=0.0
            total_grad_fine=0.0
            for i, cell in enumerate(cells_to_process):
                c_start, c_end = info_coarse[i][1], info_coarse[i][2]
                v_c = vals_coarse_all[c_start:c_end]
                w_c = cell.w.flatten()
                
                f_start, f_end = info_fine[i][1], info_fine[i][2]
                v_f = vals_fine_all[f_start:f_end]
                w_f = all_w_fine[f_start:f_end]
                
                if is_multidim:
                # Extract S and the gradient.
                    S_c = v_c[:, 0]
                    grad_c = v_c[:, 1:]
                    S_f = v_f[:, 0]
                    grad_f = v_f[:, 1:]
                    
                # Compute the L2-norm integral: square, sum with weights, then take the square root.
                    indicator_S_coarse = np.sqrt(np.sum((S_c**2) * w_c))
                    indicator_grad_coarse = np.sqrt(np.sum(np.sum(grad_c**2, axis=1) * w_c))
                    
                    indicator_S_fine = np.sqrt(np.sum((S_f**2) * w_f))
                    indicator_grad_fine = np.sqrt(np.sum(np.sum(grad_f**2, axis=1) * w_f))
                    
                    diff_S = np.abs(indicator_S_fine - indicator_S_coarse)
                    diff_grad = np.abs(indicator_grad_fine - indicator_grad_coarse)
                    total_diff_grad += diff_grad
                    total_abs_fine+=indicator_S_fine
                    total_grad_fine+=indicator_grad_fine
                    cell_metrics.append({
                        'indicator_S_fine': indicator_S_fine,
                        'indicator_grad_fine': indicator_grad_fine,
                        'diff_S': diff_S,
                        'diff_grad': diff_grad
                    })
                else:
                    integral_coarse = np.sum(v_c.flatten() * w_c)
                    integral_fine = np.sum(v_f.flatten() * w_f)
                    diff = abs(integral_fine - integral_coarse)
                    cell_metrics.append({'diff': diff})

            # --- Compute the dynamic gradient threshold: total error divided by the number of cells. ---
            if is_multidim and len(cells_to_process) > 0:
                dynamic_threshold_abs, dynamic_threshold_grad, dynamic_threshold_noise = self._compute_multidim_thresholds(
                    cell_metrics,
                    total_diff_grad,
                    total_abs_fine,
                    total_grad_fine,
                    threshold_S,
                    threshold_grad,
                    threshold_noise,
                    refine_mode,
                )
            else:
                dynamic_threshold_grad = threshold_grad
                dynamic_threshold_abs = threshold_S

            # --- Second pass: decide refinement using the dynamic thresholds. ---
            for i, cell in enumerate(cells_to_process):
                should_refine = False
                metrics = cell_metrics[i]
                
                if is_multidim:
                    if self._should_refine_multidim(
                        metrics,
                        dynamic_threshold_abs,
                        dynamic_threshold_grad,
                        dynamic_threshold_noise,
                        refine_mode,
                    ):
                        should_refine = True
                else:
                    diff = metrics['diff']
                # Use threshold_S here because the original threshold variable is not defined.
                    if diff > dynamic_threshold_abs:
                        should_refine = True
                
                if should_refine:
                    cell.children = children_store[i]
                    next_generation.extend(cell.children)
                    refinement_count += 1
            
            queue = next_generation
            
        return self.cells

def integrate_adaptive_refinement_fix(
    cells,
    model_evaluator,
    threshold_S,
    threshold_grad,
    threshold_noise,
    nx,
    device,
    current_maxiter,
    max_level=10,
    refine_mode="standard",
):
    amr = AdaptiveMeshRefinementFix(cells, model_evaluator, nx, device)
    refined_cells = amr.refine_by_integral_difference(
        threshold_S,
        threshold_grad,
        threshold_noise,
        current_maxiter,
        max_level,
        refine_mode=refine_mode,
    )

    final_leaves = mesh.collect_leaf_cells(refined_cells)
    
    stats = {
        "final_cell_count": len(final_leaves)
    }
    
    return refined_cells, final_leaves, stats
