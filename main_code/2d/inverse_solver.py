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


def _plot_l_curve(res, reg_norm, lamb_regu):
    plt.figure(figsize=(6, 4))
    log_res = np.log10(res.reshape(-1))
    log_reg_norm = np.log10(reg_norm.reshape(-1))
    plt.plot(log_res, log_reg_norm, marker="o", linestyle="-", color="b", label="Regularization Path")
    for i, lamb in enumerate(lamb_regu):
        plt.text(log_res[i] + 0.0000001, log_reg_norm[i] - 0.001, f"{i}: {lamb:.2e}", fontsize=12, ha="center", color="red")
    plt.xlabel("Residual Norm (||Ax - y||)")
    plt.ylabel("Regularization Norm (||w||)")
    plt.title("Regularization Path (Residual Norm vs Regularization Norm)")
    plt.grid(True)
    plt.show()


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

    _plot_l_curve(res, reg_norm, lamb_regu)
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
