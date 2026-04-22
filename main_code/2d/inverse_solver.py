import matplotlib.pyplot as plt
import numpy as np

import cupy as cp


def _flatten_complex_stack(matrix):
    real_part = np.real(matrix).reshape(-1, np.shape(matrix)[-1])
    imag_part = np.imag(matrix).reshape(-1, np.shape(matrix)[-1])
    return real_part, imag_part


def A(*args):
    if len(args) == 2:
        b_matrix, db_matrix = args
        A_b, B_b = _flatten_complex_stack(b_matrix)
        A_db, B_db = _flatten_complex_stack(db_matrix)
        return np.concatenate((A_b, B_b, A_db, B_db), axis=0)

    if len(args) == 3:
        b_matrix, db_x1_matrix, db_x2_matrix = args
        A_b, B_b = _flatten_complex_stack(b_matrix)
        A_db_x1, B_db_x1 = _flatten_complex_stack(db_x1_matrix)
        A_db_x2, B_db_x2 = _flatten_complex_stack(db_x2_matrix)
        return np.concatenate((A_b, B_b, A_db_x1, B_db_x1, A_db_x2, B_db_x2), axis=0)

    raise TypeError("Unsupported inverse_solver.A signature")


def F(*args):
    if len(args) == 2:
        F_mea_b, F_mea_db = args
        F_mea_b = F_mea_b.reshape(-1, F_mea_b.shape[-1])
        F_mea_db = F_mea_db.reshape(-1, F_mea_db.shape[-1])
        return np.concatenate((F_mea_b[:, 0:1], F_mea_b[:, 1:], F_mea_db[:, 0:1], F_mea_db[:, 1:]), axis=0)

    if len(args) == 3:
        F_mea_b, F_mea_db_x1, F_mea_db_x2 = args
        F_mea_b = F_mea_b.reshape(-1, F_mea_b.shape[-1])
        F_mea_db_x1 = F_mea_db_x1.reshape(-1, F_mea_db_x1.shape[-1])
        F_mea_db_x2 = F_mea_db_x2.reshape(-1, F_mea_db_x2.shape[-1])
        return np.concatenate(
            (
                F_mea_b[:, 0:1],
                F_mea_b[:, 1:],
                F_mea_db_x1[:, 0:1],
                F_mea_db_x1[:, 1:],
                F_mea_db_x2[:, 0:1],
                F_mea_db_x2[:, 1:],
            ),
            axis=0,
        )

    raise TypeError("Unsupported inverse_solver.F signature")


def _plot_l_curve(res, reg_norm, lamb_regu, best_idx=None):
    plt.figure(figsize=(6, 4))
    log_res = np.log10(res.reshape(-1))
    log_reg_norm = np.log10(reg_norm.reshape(-1))
    plt.plot(log_res, log_reg_norm, marker="o", linestyle="-", color="b", label="Regularization Path")
    for i, lamb in enumerate(lamb_regu):
        plt.text(log_res[i] + 0.0000001, log_reg_norm[i] - 0.001, f"{i}: {lamb:.2e}", fontsize=12, ha="center", color="red")
    if best_idx is not None:
        plt.scatter(
            [log_res[best_idx]],
            [log_reg_norm[best_idx]],
            s=80,
            color="gold",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label=f"Selected idx={best_idx}",
        )
    plt.xlabel("Residual Norm (||Ax - y||)")
    plt.ylabel("Regularization Norm (||w||)")
    plt.title("Regularization Path (Residual Norm vs Regularization Norm)")
    plt.grid(True)
    if best_idx is not None:
        plt.legend()
    plt.show()


def lcurve_corner_index(lamb_regu, res, reg_norm):
    """Return the discrete L-curve corner index using log-log differential curvature."""
    lamb_regu = np.asarray(lamb_regu, dtype=float).reshape(-1)
    res = np.asarray(res, dtype=float).reshape(-1)
    reg_norm = np.asarray(reg_norm, dtype=float).reshape(-1)

    if not (len(lamb_regu) == len(res) == len(reg_norm)):
        raise ValueError("lamb_regu, res, and reg_norm must have the same length")

    if len(lamb_regu) == 0:
        raise ValueError("lamb_regu is empty")

    if len(lamb_regu) < 3:
        curvature = np.zeros_like(lamb_regu, dtype=float)
        return 0, curvature

    eps = 1e-300
    t = np.log(np.maximum(lamb_regu, eps))
    x = np.log(np.maximum(res, eps))
    y = np.log(np.maximum(reg_norm, eps))

    edge_order = 2 if len(lamb_regu) >= 3 else 1
    x1 = np.gradient(x, t, edge_order=edge_order)
    y1 = np.gradient(y, t, edge_order=edge_order)
    x2 = np.gradient(x1, t, edge_order=edge_order)
    y2 = np.gradient(y1, t, edge_order=edge_order)

    curvature = np.abs(x1 * y2 - y1 * x2) / np.maximum((x1**2 + y1**2) ** 1.5, eps)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    curvature[0] = 0.0
    curvature[-1] = 0.0

    best_idx = int(np.argmax(curvature))
    return best_idx, curvature


def _topk_interior_curvature_indices(curvature, k=3):
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if len(curvature) == 0:
        return np.asarray([], dtype=int)
    if len(curvature) <= 2:
        return np.asarray([0], dtype=int)

    interior = np.arange(1, len(curvature) - 1, dtype=int)
    order = interior[np.argsort(curvature[interior])[::-1]]
    k = max(1, min(int(k), len(order)))
    return order[:k]


def lambda_window_by_curvature(lamb_regu, res, reg_norm, radius=1, w_store=None, verbose=True):
    """Return the lambda window centered at the maximum-curvature index."""
    best_idx, curvature = lcurve_corner_index(lamb_regu, res, reg_norm)
    lamb_regu = np.asarray(lamb_regu, dtype=float).reshape(-1)
    radius = max(0, int(radius))
    left = max(0, best_idx - radius)
    right = min(len(lamb_regu) - 1, best_idx + radius)
    window_idx = np.arange(left, right + 1, dtype=int)
    window_lambda = lamb_regu[window_idx]

    if verbose:
        print(f"L-curve corner index: {best_idx}")
        print("Curvature-centered lambda window:")
        for idx in window_idx:
            mark = " <== center" if idx == best_idx else ""
            print(
                f"  idx={idx}, lambda={lamb_regu[idx]:.6e}, "
                f"curvature={curvature[idx]:.6e}{mark}"
            )

    if w_store is None:
        return window_idx, window_lambda, curvature

    w_store = np.asarray(w_store)
    return window_idx, window_lambda, w_store[:, window_idx], curvature


def topk_lambda_by_curvature(lamb_regu, res, reg_norm, k=3, w_store=None, verbose=True):
    """Return the top-k interior lambda candidates ranked by L-curve curvature."""
    _, curvature = lcurve_corner_index(lamb_regu, res, reg_norm)
    lamb_regu = np.asarray(lamb_regu, dtype=float).reshape(-1)
    top_idx = _topk_interior_curvature_indices(curvature, k=k)
    top_lambda = lamb_regu[top_idx]

    if verbose:
        print(f"Top-{len(top_idx)} curvature candidates:")
        for rank, idx in enumerate(top_idx, start=1):
            print(
                f"  rank={rank}, idx={idx}, lambda={lamb_regu[idx]:.6e}, "
                f"curvature={curvature[idx]:.6e}"
            )

    if w_store is None:
        return top_idx, top_lambda, curvature

    w_store = np.asarray(w_store)
    return top_idx, top_lambda, w_store[:, top_idx], curvature


def choose_lambda_by_curvature(lamb_regu, res, reg_norm, w_store=None, verbose=True):
    """Choose the L-curve corner by maximizing the discrete log-log curvature."""
    best_idx, curvature = lcurve_corner_index(lamb_regu, res, reg_norm)
    lamb_regu = np.asarray(lamb_regu, dtype=float).reshape(-1)
    best_lambda = float(lamb_regu[best_idx])

    if verbose:
        print(f"L-curve corner index: {best_idx}")
        print(f"Selected lambda by curvature: {best_lambda:.6e}")
        for i, (lam, curv) in enumerate(zip(lamb_regu, curvature)):
            print(f"  idx={i}, lambda={lam:.6e}, curvature={curv:.6e}")
        top_idx = _topk_interior_curvature_indices(curvature, k=3)
        print("Top curvature candidates:")
        for rank, idx in enumerate(top_idx, start=1):
            print(
                f"  rank={rank}, idx={idx}, lambda={lamb_regu[idx]:.6e}, "
                f"curvature={curvature[idx]:.6e}"
            )

    if w_store is None:
        return best_idx, best_lambda, curvature

    w_store = np.asarray(w_store)
    return best_idx, best_lambda, w_store[:, best_idx], curvature


def _linear_l_curve(mat, F_rhs, lamb_regu, cupy_device, n):
    iter_count = len(lamb_regu)
    res = np.zeros([1, iter_count])
    reg_norm = np.zeros([1, iter_count])
    w_store = np.zeros([n, iter_count])
    cp.cuda.Device(cupy_device).use()

    for i, lamb in enumerate(lamb_regu):
        regu = lamb * np.eye(n)
        mat_regu = np.concatenate((mat, regu), axis=0)
        F_regu = np.concatenate((F_rhs, lamb * np.zeros([n, 1])), axis=0)
        mat_regu_cuda = cp.asarray(mat_regu)
        F_regu_cuda = cp.asarray(F_regu)
        w_regu = cp.linalg.lstsq(mat_regu_cuda, F_regu_cuda)[0]
        w = w_regu.reshape(-1, 1).get()
        w_store[:, i : i + 1] = w
        res[:, i] = np.linalg.norm(mat @ w - F_rhs)
        reg_norm[:, i] = np.linalg.norm(w)

    best_idx, _ = lcurve_corner_index(lamb_regu, res, reg_norm)
    _plot_l_curve(res, reg_norm, lamb_regu, best_idx=best_idx)
    return res, reg_norm, w_store



def L_curve(*args):
    if len(args) == 8:
        M0, M1, M2, af, mat, F_rhs, lamb_regu, cupy_device = args
        n = M0 + M1 + M2 if af == "mix" else M0
        return _linear_l_curve(mat, F_rhs, lamb_regu, cupy_device, n)

    if len(args) == 9:
        mat, F_rhs, grads_y1, grads_y2, lambda_x_y, lamb_regu, solver, device, cupy_device = args
        n = mat.shape[1]
        if solver == "linear":
            return _linear_l_curve(mat, F_rhs, lamb_regu, cupy_device, n)
        raise ValueError(f"Unsupported solver: {solver}")

    raise TypeError("Unsupported inverse_solver.L_curve signature")


__all__ = [
    "A",
    "F",
    "L_curve",
    "lcurve_corner_index",
    "lambda_window_by_curvature",
    "topk_lambda_by_curvature",
    "choose_lambda_by_curvature",
]
