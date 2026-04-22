from scipy.linalg import lstsq,pinv
import numpy as np
from scipy.linalg import block_diag
import scipy
import matplotlib.pyplot as plt
import cupy as cp


def A(b_matrix,db_x1_matrix,db_x2_matrix,db_x3_matrix,condition="Cauchy"):
    if condition=="Cauchy":
        r_b=np.real(b_matrix)
        i_b=np.imag(b_matrix)
        r_db_x1=np.real(db_x1_matrix)
        i_db_x1=np.imag(db_x1_matrix)
        r_db_x2=np.real(db_x2_matrix)
        i_db_x2=np.imag(db_x2_matrix)
        r_db_x3=np.real(db_x3_matrix)
        i_db_x3=np.imag(db_x3_matrix)
        A_b=r_b.reshape(-1,r_b.shape[-1])
        B_b=i_b.reshape(-1,i_b.shape[-1])
        A_db_x1=r_db_x1.reshape(-1,r_db_x1.shape[-1])
        B_db_x1=i_db_x1.reshape(-1,i_db_x1.shape[-1])
        A_db_x2=r_db_x2.reshape(-1,r_db_x2.shape[-1])
        B_db_x2=i_db_x2.reshape(-1,i_db_x2.shape[-1])
        A_db_x3=r_db_x3.reshape(-1,r_db_x3.shape[-1])
        B_db_x3=i_db_x3.reshape(-1,i_db_x3.shape[-1])
        mat=np.concatenate((A_b,B_b,A_db_x1,B_db_x1,A_db_x2,B_db_x2,A_db_x3,B_db_x3),axis=0)
    if condition=="Dirichlet":
        r_b=np.real(b_matrix)
        i_b=np.imag(b_matrix)
        A_b=r_b.reshape(-1,r_b.shape[-1])
        B_b=i_b.reshape(-1,i_b.shape[-1])
        mat=np.concatenate((A_b,B_b),axis=0)

    return mat

def F(F_mea_b,F_mea_db_x1,F_mea_db_x2,F_mea_db_x3,condition="Cauchy"):
    if condition=="Cauchy":
        F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
        F_mea_db_x1=F_mea_db_x1.reshape(-1,F_mea_db_x1.shape[-1])
        F_mea_db_x2=F_mea_db_x2.reshape(-1,F_mea_db_x2.shape[-1])
        F_mea_db_x3=F_mea_db_x3.reshape(-1,F_mea_db_x3.shape[-1])
        F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:],F_mea_db_x1[:,0:1],F_mea_db_x1[:,1:],F_mea_db_x2[:,0:1],F_mea_db_x2[:,1:],F_mea_db_x3[:,0:1],F_mea_db_x3[:,1:]),axis=0)
    if condition=="Dirichlet":
        F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
        F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:]),axis=0)
    return F


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


def _plot_l_curve(res, reg_norm, lamb_regu, best_idx=None):
    plt.figure(figsize=(6, 4))
    log_res=np.log10(res.reshape(-1))
    log_reg_norm=np.log10(reg_norm.reshape(-1))
    plt.plot(log_res,log_reg_norm , marker='o', linestyle='-', color='b', label="Regularization Path")
    for i in range(len(lamb_regu)):
        plt.text(log_res[i]+0.0000001, log_reg_norm[i] - 0.001,   f"{i}: {lamb_regu[i]:.2e}", fontsize=12, ha='center', color='red')
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

def L_curve(*args):
    if len(args) == 4:
        mat,F,lamb_regu,cupy_device = args
        n = mat.shape[1]
    elif len(args) == 10:
        M0,M_relu1,M_relu2,M_gauss1,M_gauss2,af,mat,F,lamb_regu,cupy_device = args
        n_from_mat = mat.shape[1]
        if af=="mixer":
            n=M0+M_relu1+M_relu2+M_gauss1+M_gauss2
        elif af=="mix":
            n=M0+M_relu1+M_relu2
        elif af=="relu":
            n=M_relu1+M_relu2
        else:
            n=M0
        if n != n_from_mat:
            print(
                f"L_curve: parameter width {n} does not match matrix width {n_from_mat}; "
                "using matrix width."
            )
            n = n_from_mat
    else:
        raise TypeError("L_curve expects either 4 args or 10 args")

    iter=len(lamb_regu)
    res=np.zeros([1,iter])
    reg_norm=np.zeros([1,iter])
    w_store=np.zeros([n,iter])
    cp.cuda.Device(cupy_device).use()

    mat_cuda = cp.asarray(mat)
    F_cuda = cp.asarray(F)
    m = mat.shape[0]

    # Use the smaller Tikhonov system on GPU instead of explicitly stacking
    # [A; lambda I], which creates a much larger temporary matrix and spikes
    # GPU memory in notebooks.
    if m <= n:
        gram = mat_cuda @ mat_cuda.T
        eye = cp.eye(m, dtype=mat_cuda.dtype)
        solve_on_rows = True
    else:
        gram = mat_cuda.T @ mat_cuda
        rhs = mat_cuda.T @ F_cuda
        eye = cp.eye(n, dtype=mat_cuda.dtype)
        solve_on_rows = False

    for i in range(iter):
        lam2 = float(lamb_regu[i]) ** 2
        system = gram + lam2 * eye
        if solve_on_rows:
            alpha = cp.linalg.solve(system, F_cuda)
            w_regu = mat_cuda.T @ alpha
        else:
            w_regu = cp.linalg.solve(system, rhs)

        w = cp.asnumpy(w_regu.reshape(-1,1))
        w_store[:,i:i+1]=w
        res[:,i]= float(cp.linalg.norm(mat_cuda @ w_regu - F_cuda).get())
        reg_norm[:,i]= float(cp.linalg.norm(w_regu).get())

        del system
        if solve_on_rows:
            del alpha
        del w_regu

    del mat_cuda, F_cuda, gram, eye
    if not solve_on_rows:
        del rhs
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    best_idx, _ = lcurve_corner_index(lamb_regu, res, reg_norm)
    _plot_l_curve(res, reg_norm, lamb_regu, best_idx=best_idx)
    return res,reg_norm,w_store


__all__ = [
    "A",
    "F",
    "L_curve",
    "lcurve_corner_index",
    "lambda_window_by_curvature",
    "topk_lambda_by_curvature",
    "choose_lambda_by_curvature",
]
