"""Shared 2D adaptive solver entrypoints with example-specific wrappers."""

import importlib

import matplotlib
import numpy as np
import torch

import adaptive_int
import matrix_assemble
import error_general_2d
import inverse_solver
import mesh
import source_eval


matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["text.latex.preamble"] = ""


def _detach_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _build_refine_payload(s, g_s):
    s = _detach_numpy(s)
    g_s = _detach_numpy(g_s)
    return np.hstack([np.asarray(s).reshape(-1, 1), np.asarray(g_s)])


def _threshold_at(threshold, i):
    """Accept either a scalar threshold or an iteration-wise threshold list."""
    if isinstance(threshold, np.ndarray):
        if threshold.ndim == 0:
            return threshold.item()
        return threshold[i]
    if isinstance(threshold, (list, tuple)):
        return threshold[i]
    return threshold


def _maybe_visualize(cells, refined_cells, all_g_p, S_num, grad_S_num, refinement_stats):
    try:
        visual = importlib.import_module("visual")
    except Exception:
        return

    try:
        visual.visualize_adaptive_refinement(
            cells=cells,
            refined_cells=refined_cells,
            all_g_p=all_g_p,
            S_num=S_num,
            grad_S_num=grad_S_num,
            refinement_stats=refinement_stats,
            cell_indicators=None,
        )
    except Exception as exc:
        print(f"Visualization failed: {exc}")


def _should_stop(i, S_test_num, S_num_store, delta):
    return i >= 1 and np.linalg.norm(S_test_num - S_num_store[i - 1]) / np.linalg.norm(S_test_num) < delta


def _ada_int_plain(
    iter_int,
    delta,
    cells,
    nx,
    models,
    M,
    af,
    kk,
    F,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    r_low,
    r_upper,
    center_true,
    r_true,
    ana_S,
    points_b,
    lamb_regu,
    Qx,
    Qy,
    device,
    cupy_device,
    refine_threshold_S,
    refine_threshold_grad,
    refine_threshold_noise,
    current_maxiter,
    max_level,
):
    cells_store = []
    Cells = cells
    refinement_stats_store = []
    leaf_cells = mesh.collect_leaf_cells(Cells)
    all_points0, _ = mesh.collect_all_gauss_points(leaf_cells)
    w_ = []
    point_number = []
    S_num_store = []
    S_l2 = []
    g_store = []

    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat, all_g_list, all_g_p, Cells = matrix_assemble.mat_assemble2(
            Cells, models, [], [], af, kk, 0, points_b, device, cupy_device
        )
        res, reg_norm, w_store = inverse_solver.L_curve(M, [], [], af, all_mat, F, lamb_regu, cupy_device)
        best_idx, best_lambda, current_w, _ = inverse_solver.choose_lambda_by_curvature(
            lamb_regu,
            res,
            reg_norm,
            w_store=w_store,
            verbose=len(np.asarray(lamb_regu).reshape(-1)) > 1,
        )

        print("Training error at iteration", i, ":")
        S_num, grad_S_num = source_eval.valgrad(
            models, af, M, all_points0, current_w, ana_S, center_true, r_true, device
        )
        print("Generalization error at iteration", i, ":")
        S_test_num, S_test_true, S_l_inf1, S_l_21, g_S = error_general_2d.test(
            models,
            [],
            [],
            Qx,
            Qy,
            current_w,
            af,
            True,
            True,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            [],
            [],
            [],
            [],
            center_true,
            r_true,
            ana_S,
            1,
            device,
        )
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        w_.append(current_w)

        if _should_stop(i, S_test_num, S_num_store, delta):
            print("Stop at iteration", i)
            break

        print("Adaptive mesh refinement at iteration", i, "...")

        def evaluator_func(pts_np):
            s, g_s = source_eval.valgrad(
                models, af, M, pts_np, current_w, ana_S, center_true, r_true, device
            )
            return _build_refine_payload(s, g_s)

        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
            Cells,
            evaluator_func,
            _threshold_at(refine_threshold_S, i),
            _threshold_at(refine_threshold_grad, i),
            _threshold_at(refine_threshold_noise, i),
            nx,
            device,
            current_maxiter=current_maxiter,
            max_level=max_level,
        )
        refinement_stats_store.append(refinement_stats)
        _maybe_visualize(cells_store[i], Cells, all_points0, S_num, grad_S_num, refinement_stats)
        all_points0, _ = mesh.collect_all_gauss_points(Cells)

    return cells_store, refinement_stats_store, w_, point_number, g_store, g_S, S_l2, S_num_store


def _ada_int_fix_ex43(
    iter_int,
    delta,
    cells,
    nx,
    models,
    M,
    af,
    kk,
    F,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    r_low,
    r_upper,
    center_true,
    r_true,
    ana_S,
    points_b,
    lamb_regu,
    Qx,
    Qy,
    device,
    cupy_device,
    refine_threshold_S,
    refine_threshold_grad,
    refine_threshold_noise,
    current_maxiter,
    max_level,
):
    cells_store = []
    Cells = cells
    refinement_stats_store = []
    leaf_cells = mesh.collect_leaf_cells(Cells)
    all_points0, _ = mesh.collect_all_gauss_points(leaf_cells)
    w_ = []
    point_number = []
    S_num_store = []
    S_l2 = []
    g_store = []

    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)

        all_mat, all_g_list, all_g_p, Cells, grads_y1, grads_y2 = matrix_assemble.mat_assemble(
            Cells, models, [], [], af, [], [], kk, 0, points_b, 0, device, cupy_device
        )
        res, reg_norm, w_store = inverse_solver.L_curve(
            all_mat, F, [], [], [], lamb_regu, "linear", device, cupy_device
        )
        best_idx, best_lambda, current_w, _ = inverse_solver.choose_lambda_by_curvature(
            lamb_regu,
            res,
            reg_norm,
            w_store=w_store,
            verbose=len(np.asarray(lamb_regu).reshape(-1)) > 1,
        )
        w_.append(current_w)

        print("Training error at iteration", i, ":")
        S_num, grad_S_num = source_eval.valgrad(
            models, af, [], M, all_points0, current_w, ana_S, center_true, r_true, device
        )
        print("Generalization error at iteration", i, ":")
        S_test_num, _, _, S_l_21, g_S = error_general_2d.test(
            models,
            [],
            [],
            M,
            Qx,
            Qy,
            current_w,
            af,
            [],
            [],
            True,
            True,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            [],
            [],
            [],
            [],
            center_true,
            r_true,
            ana_S,
            1,
            device,
        )
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)

        if _should_stop(i, S_test_num, S_num_store, delta):
            print("Stop at iteration", i)
            break

        print("Adaptive mesh refinement at iteration", i, "(Diff Criterion)...")

        def evaluator_func(pts_np):
            s, g_s = source_eval.valgrad(
                models, af, [], M, pts_np, current_w, ana_S, center_true, r_true, device
            )
            return _build_refine_payload(s, g_s)

        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
            Cells,
            evaluator_func,
            _threshold_at(refine_threshold_S, i),
            _threshold_at(refine_threshold_grad, i),
            _threshold_at(refine_threshold_noise, i),
            nx,
            device,
            current_maxiter=current_maxiter,
            max_level=max_level,
        )
        refinement_stats_store.append(refinement_stats)
        _maybe_visualize(cells_store[i], Cells, all_points0, S_num, grad_S_num, refinement_stats)
        all_points0, _ = mesh.collect_all_gauss_points(Cells)

    return cells_store, refinement_stats_store, w_, point_number, g_store, g_S, S_l2, S_num_store


def _ada_int_shape(
    iter_int,
    delta,
    cells,
    nx,
    models,
    M,
    af,
    Shape,
    kk,
    F,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    x_left,
    x_right,
    y_below,
    y_upper,
    center_true,
    r_true,
    ana_S,
    points_b,
    lamb_regu,
    Qx,
    Qy,
    device,
    cupy_device,
    refine_threshold_S,
    refine_threshold_grad,
    refine_threshold_noise,
    current_maxiter,
    max_level,
):
    cells_store = []
    Cells = cells
    refinement_stats_store = []
    leaf_cells = mesh.collect_leaf_cells(Cells)
    all_points0, _ = mesh.collect_all_gauss_points(leaf_cells)
    w_ = []
    point_number = []
    S_num_store = []
    S_l2 = []
    g_store = []

    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat, all_g_list, all_g_p, Cells = matrix_assemble.mat_assemble2(
            Cells, models, [], [], af, [], [], kk, 0, points_b, device, cupy_device
        )
        res, reg_norm, w_store = inverse_solver.L_curve(
            all_mat, F, [], [], [], lamb_regu, "linear", device, cupy_device
        )
        best_idx, best_lambda, current_w, _ = inverse_solver.choose_lambda_by_curvature(
            lamb_regu,
            res,
            reg_norm,
            w_store=w_store,
            verbose=len(np.asarray(lamb_regu).reshape(-1)) > 1,
        )

        print("Training error at iteration", i, ":")
        S_num, grad_S_num = source_eval.valgrad(
            models, af, Shape, M, all_points0, current_w, ana_S, center_true, r_true, device
        )
        print("Generalization error at iteration", i, ":")
        S_test_num, S_test_true, S_l_inf1, S_l_21, g_S = error_general_2d.test(
            models,
            [],
            [],
            M,
            Qx,
            Qy,
            current_w,
            af,
            [],
            [],
            True,
            True,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            x_left,
            x_right,
            y_below,
            y_upper,
            center_true,
            r_true,
            ana_S,
            1,
            device,
        )
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        w_.append(current_w)

        if _should_stop(i, S_test_num, S_num_store, delta):
            print("Stop at iteration", i)
            break

        print("Adaptive mesh refinement at iteration", i, "...")

        def evaluator_func(pts_np):
            s, g_s = source_eval.valgrad(
                models, af, [], M, pts_np, current_w, ana_S, center_true, r_true, device
            )
            return _build_refine_payload(s, g_s)

        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
            Cells,
            evaluator_func,
            _threshold_at(refine_threshold_S, i),
            _threshold_at(refine_threshold_grad, i),
            _threshold_at(refine_threshold_noise, i),
            nx,
            device,
            current_maxiter=current_maxiter,
            max_level=max_level,
        )
        refinement_stats_store.append(refinement_stats)
        _maybe_visualize(cells_store[i], Cells, all_points0, S_num, grad_S_num, refinement_stats)
        all_points0, _ = mesh.collect_all_gauss_points(Cells)

    return cells_store, refinement_stats_store, w_, point_number, S_num_store, g_store, g_S, S_l2


def _ada_int_signed(
    iter_int,
    delta,
    cells,
    nx,
    models,
    M,
    af,
    Shape,
    sign_distances,
    kk,
    F,
    tau_x_min,
    tau_x_max,
    tau_y_min,
    tau_y_max,
    x_left,
    x_right,
    y_below,
    y_upper,
    center_true,
    r_true,
    ana_S,
    points_b,
    lamb_regu,
    Qx,
    Qy,
    device,
    cupy_device,
    refine_threshold_S,
    refine_threshold_grad,
    refine_threshold_noise,
    current_maxiter,
    max_level,
):
    cells_store = []
    Cells = cells
    refinement_stats_store = []
    leaf_cells = mesh.collect_leaf_cells(Cells)
    all_points0, _ = mesh.collect_all_gauss_points(leaf_cells)
    w_ = []
    point_number = []
    S_num_store = []
    S_l2 = []
    g_store = []

    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat, all_g_list, all_g_p, Cells = matrix_assemble.mat_assemble2(
            Cells, models, [], [], [], af, [], [], [], kk, 0, points_b, device, cupy_device
        )
        res, reg_norm, w_store = inverse_solver.L_curve(
            all_mat, F, [], [], [], lamb_regu, "linear", device, cupy_device
        )
        best_idx, best_lambda, current_w, _ = inverse_solver.choose_lambda_by_curvature(
            lamb_regu,
            res,
            reg_norm,
            w_store=w_store,
            verbose=len(np.asarray(lamb_regu).reshape(-1)) > 1,
        )

        print("Training error at iteration", i, ":")
        S_num, grad_S_num = source_eval.valgrad(
            models, af, Shape, sign_distances, M, all_points0, current_w, ana_S, center_true, r_true, device
        )
        print("Generalization error at iteration", i, ":")
        sample_s = error_general_2d.test_p(Qx, Qy, tau_x_min, tau_x_max, tau_y_min, tau_y_max)
        S_test_num, S_test_true, S_l_inf1, S_l_21, g_S = error_general_2d.test(
            models,
            [],
            [],
            [],
            sample_s,
            current_w,
            af,
            [],
            [],
            [],
            True,
            True,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            x_left,
            x_right,
            y_below,
            y_upper,
            center_true,
            r_true,
            ana_S,
            0.5,
            1,
            device,
        )
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        w_.append(current_w)

        if _should_stop(i, S_test_num, S_num_store, delta):
            print("Stop at iteration", i)
            break

        print("Adaptive mesh refinement at iteration", i, "...")

        def evaluator_func(pts_np):
            s, g_s = source_eval.valgrad(
                models, af, Shape, sign_distances, M, pts_np, current_w, ana_S, center_true, r_true, device
            )
            return _build_refine_payload(s, g_s)

        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
            Cells,
            evaluator_func,
            _threshold_at(refine_threshold_S, i),
            _threshold_at(refine_threshold_grad, i),
            _threshold_at(refine_threshold_noise, i),
            nx,
            device,
            current_maxiter=current_maxiter,
            max_level=max_level,
        )
        refinement_stats_store.append(refinement_stats)
        _maybe_visualize(cells_store[i], Cells, all_points0, S_num, grad_S_num, refinement_stats)
        all_points0, _ = mesh.collect_all_gauss_points(Cells)

    return cells_store, refinement_stats_store, w_, point_number, g_store, g_S, S_l2, S_num_store


def _reuse_grad_threshold_as_noise(args):
    """New 2D AMR API uses refine_threshold_grad as the gradient-energy threshold.

    The internal implementations still accept the legacy
    (refine_threshold_S, refine_threshold_grad, refine_threshold_noise, current_maxiter, max_level)
    tail to keep old notebooks working. For the new public signature, insert
    refine_threshold_grad once more at the legacy noise-threshold position.
    """
    return args[:-2] + (args[-3],) + args[-2:]


def _looks_like_signed_without_noise(args):
    # New signed call:
    # (..., af, Shape, sign_distances, kk, F, tau_x_min, ...)
    # Old shape call with the same length:
    # (..., af, Shape, kk, F, tau_x_min, ...)
    # Therefore args[10] is F for the new signed call, but tau_x_min for the old shape call.
    return len(args) == 32 and not np.isscalar(args[10])


def ada_int(*args):
    if len(args) == 28:
        return _ada_int_plain(*_reuse_grad_threshold_as_noise(args))
    if len(args) == 29:
        return _ada_int_plain(*args)
    if len(args) == 31:
        return _ada_int_shape(*_reuse_grad_threshold_as_noise(args))
    if len(args) == 32:
        if _looks_like_signed_without_noise(args):
            return _ada_int_signed(*_reuse_grad_threshold_as_noise(args))
        return _ada_int_shape(*args)
    if len(args) == 33:
        return _ada_int_signed(*args)
    raise TypeError("Unsupported main.ada_int signature")


def ada_int_fix(*args):
    if len(args) == 28:
        return _ada_int_fix_ex43(*_reuse_grad_threshold_as_noise(args))
    if len(args) == 29:
        return _ada_int_fix_ex43(*args)
    raise TypeError("Unsupported main.ada_int_fix signature")


__all__ = ["ada_int", "ada_int_fix"]
