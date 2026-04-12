import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(2)
warnings.filterwarnings("ignore", message="findfont:")


def test_p(Qx, Qy, Qz, x_min, x_max, y_min, y_max, z_min, z_max):
    test_Qx = 2 * Qx
    test_Qy = 2 * Qy
    test_Qz = 2 * Qz
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    z_devide = np.linspace(z_min, z_max, test_Qz + 1)
    X, Y, Z = np.meshgrid(x_devide, y_devide, z_devide, indexing="ij")
    return np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T


def _plot_mid_slices_2d(S_num, S_true, x_min, x_max, y_min, y_max, z_min, z_max):
    y = np.linspace(y_min, y_max, S_num.shape[1])
    z = np.linspace(z_min, z_max, S_num.shape[2])
    x = np.linspace(x_min, x_max, S_num.shape[0])

    yy, zz = np.meshgrid(y, z, indexing="ij")
    xx, zz_x = np.meshgrid(x, z, indexing="ij")

    x1_idx = S_num.shape[0] // 2
    x2_idx = S_num.shape[1] // 2

    s_true_x1 = S_true[x1_idx, :, :]
    s_num_x1 = S_num[x1_idx, :, :]
    s_true_x2 = S_true[:, x2_idx, :]
    s_num_x2 = S_num[:, x2_idx, :]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (axes[0, 0], yy, zz, s_true_x1, r"True $x_1$ Mid Slice", r"$x_2$", r"$x_3$"),
        (axes[0, 1], yy, zz, s_num_x1, r"Numerical $x_1$ Mid Slice", r"$x_2$", r"$x_3$"),
        (axes[1, 0], xx, zz_x, s_true_x2, r"True $x_2$ Mid Slice", r"$x_1$", r"$x_3$"),
        (axes[1, 1], xx, zz_x, s_num_x2, r"Numerical $x_2$ Mid Slice", r"$x_1$", r"$x_3$"),
    ]

    for ax, X, Y, data, title, xlabel, ylabel in panels:
        contour = ax.contourf(
            X,
            Y,
            data,
            levels=20,
            cmap="rainbow",
            alpha=1,
            vmin=np.nanmin(data),
            vmax=np.nanmax(data),
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=10)
        fig.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.show()


def _half_max_indices(values, coords, tol=10 ** (-2.2)):
    peak_index = int(np.argmax(np.abs(values)))
    half_max = values[peak_index] / 2.0
    indices = np.where((values >= half_max - tol) & (values <= half_max + tol))[0]
    return half_max, indices, coords[indices]


def _plot_jump_lines_1d(S_num, S_true, x_min, x_max, y_min, y_max, ratio1, ratio2):
    x_coords = np.linspace(x_min, x_max, S_num.shape[0])
    y_coords = np.linspace(y_min, y_max, S_num.shape[1])

    s_num_1d_x1 = S_num[
        :,
        int(ratio1[1] * S_num.shape[1]),
        int(ratio1[2] * S_num.shape[2]),
    ]
    s_true_1d_x1 = S_true[
        :,
        int(ratio1[1] * S_true.shape[1]),
        int(ratio1[2] * S_true.shape[2]),
    ]
    half_max_x1, half_indices_x1, half_coords_x1 = _half_max_indices(s_num_1d_x1, x_coords)

    plt.figure(figsize=(5, 3), dpi=120)
    plt.plot(range(int(S_num.shape[0])), s_num_1d_x1, linestyle="--", color="red", label="Num_S(x1, 0.5, 0.3)")
    plt.plot(range(int(S_true.shape[0])), s_true_1d_x1, linestyle="-", color="blue", label="Exact_S(x1, 0.5, 0.3)")
    plt.axhline(y=half_max_x1, color="r", linestyle="--", label="1/2 Maximum Value")
    for idx, x_val in zip(half_indices_x1, half_coords_x1):
        plt.axvline(x=idx, color="g", linestyle="--")
        plt.text(idx + 1, s_num_1d_x1[idx], f"x={x_val:.2f}", color="green", fontsize=9, rotation=0, va="bottom")
    plt.xlabel("x1", fontsize=14)
    plt.legend(fontsize=6)
    plt.grid()
    plt.tight_layout()
    plt.show()

    s_num_1d_x2 = S_num[
        int(ratio2[0] * S_num.shape[0]),
        :,
        int(ratio2[2] * S_num.shape[2]),
    ]
    s_true_1d_x2 = S_true[
        int(ratio2[0] * S_true.shape[0]),
        :,
        int(ratio2[2] * S_true.shape[2]),
    ]
    half_max_x2, half_indices_x2, half_coords_x2 = _half_max_indices(s_num_1d_x2, y_coords)

    plt.figure(figsize=(5, 3), dpi=120)
    plt.plot(range(int(S_num.shape[1])), s_num_1d_x2, linestyle="--", color="red", label="Num_S(0.5, x2, 0.8)")
    plt.plot(range(int(S_true.shape[1])), s_true_1d_x2, linestyle="-", color="blue", label="Exact_S(0.5, x2, 0.8)")
    plt.axhline(y=half_max_x2, color="r", linestyle="--", label="1/2 Maximum Value")
    for idx, y_val in zip(half_indices_x2, half_coords_x2):
        plt.axvline(x=idx, color="g", linestyle="--")
        plt.text(idx + 1, s_num_1d_x2[idx], f"x={y_val:.2f}", color="green", fontsize=9, rotation=0, va="bottom")
    plt.xlabel("x2", fontsize=14)
    plt.legend(fontsize=6)
    plt.grid()
    plt.tight_layout()
    plt.show()


def test(*args):
    plt.rcParams["font.family"] = "DejaVu Serif"
    models0 = args[0]
    if len(args) == 22:
        (
            _models0,
            model_sigmoid,
            sample_s,
            w,
            af,
            shape,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            center_true,
            R_true,
            r_true,
            S0,
            ana_S,
            grad_temp,
            ratio1,
            device,
        ) = args
        model_relu1 = model_sigmoid
        model_relu2 = None
        model_gauss1 = None
        model_gauss2 = None
        ratio2 = ratio1
        mode = "donut"
    else:
        (
            _models0,
            model_relu1,
            model_relu2,
            model_gauss1,
            model_gauss2,
            sample_s,
            w,
            af,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            _x_left,
            _x_right,
            _y_below,
            _y_upper,
            center_true,
            r_true,
            ana_S,
            grad_temp,
            ratio1,
            ratio2,
            device,
        ) = args
        shape = None
        R_true = None
        S0 = None
        mode = "ex47"

    length = sample_s.shape[0]
    test_Qx = round(length ** (1 / 3)) - 1
    test_Qy = round(length ** (1 / 3)) - 1
    test_Qz = round(length ** (1 / 3)) - 1
    test_point = torch.tensor(sample_s, requires_grad=True).to(device)
    grad_enabled = grad_temp == 1

    S_num_pred = torch.zeros([length, 1], device=device)
    batch_size = int(test_point.shape[0] / (test_Qx + 1))
    num_batches = int(test_point.shape[0] / batch_size)
    w = torch.as_tensor(w.reshape(-1, 1), device=device, dtype=test_point.dtype)
    g_S_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_batches - 1 else test_point.shape[0] + 1
        batch_x = test_point[start_idx:end_idx, :]

        if af == "mixer":
            with torch.set_grad_enabled(grad_enabled):
                basis_0 = models0(batch_x, "Tanh", device).to(device)
                basis_1 = model_relu1(batch_x, "relu", device).to(device)
                basis_2 = model_relu2(batch_x, "relu", device).to(device)
                basis_3 = model_gauss1(batch_x, "Gauss", device).to(device)
                basis_4 = model_gauss2(batch_x, "Gauss", device).to(device)
            af_basis = torch.cat((basis_0, basis_1, basis_2, basis_3, basis_4), dim=1)
        elif af == "mix":
            with torch.set_grad_enabled(grad_enabled):
                if mode == "donut":
                    basis_0 = models0(batch_x, "Tanh", [], device).to(device)
                    basis_1 = model_relu1(batch_x, "sigmoid", shape, device).to(device)
                    af_basis = torch.cat((basis_0, basis_1), dim=1)
                else:
                    basis_0 = models0(batch_x, "Tanh", device).to(device)
                    basis_1 = model_relu1(batch_x, "relu", device).to(device)
                    basis_2 = model_relu2(batch_x, "relu", device).to(device)
                    af_basis = torch.cat((basis_0, basis_1, basis_2), dim=1)
        elif af == "Gauss":
            with torch.set_grad_enabled(grad_enabled):
                basis_1 = model_gauss1(batch_x, "Gauss", device).to(device)
                basis_2 = model_gauss2(batch_x, "Gauss", device).to(device)
            af_basis = torch.cat((basis_1, basis_2), dim=1)
        elif af == "relu":
            with torch.set_grad_enabled(grad_enabled):
                basis_1 = model_relu1(batch_x, "relu", device).to(device)
                basis_2 = model_relu2(batch_x, "relu", device).to(device)
            af_basis = torch.cat((basis_1, basis_2), dim=1)
        elif af == "sigmoid" and shape == "sweet":
            with torch.set_grad_enabled(grad_enabled):
                af_basis = model_relu1(batch_x, "sigmoid", shape, device).to(device)
        else:
            with torch.set_grad_enabled(grad_enabled):
                if mode == "donut":
                    af_basis = models0(batch_x, af, shape, device).to(device)
                else:
                    af_basis = models0(batch_x, af, device).to(device)

        batch_output = torch.mm(af_basis, w)
        if grad_temp == 1:
            grad_S_batch = torch.autograd.grad(
                outputs=batch_output,
                inputs=batch_x,
                grad_outputs=torch.ones_like(batch_output),
                create_graph=False,
            )[0]
            g_S_list.append(grad_S_batch.cpu().detach().numpy())
        S_num_pred[start_idx:end_idx, :] = batch_output.detach()

    if grad_temp == 1:
        g_S_num = np.concatenate(g_S_list, axis=0)
        g_S = np.linalg.norm(g_S_num, axis=-1)
        g_S[g_S > 500] = 0
        g_S = g_S.reshape(test_Qx + 1, test_Qy + 1, test_Qz + 1)
    else:
        g_S = []

    S_num = S_num_pred.cpu().detach().numpy().reshape(test_Qx + 1, test_Qy + 1, test_Qz + 1)
    if mode == "donut":
        S_true = ana_S(sample_s, center_true, R_true, r_true, S0).reshape(test_Qx + 1, test_Qy + 1, test_Qz + 1)
    else:
        S_true = ana_S(sample_s, [], [], [], [], [], [], center_true[0], center_true[1], r_true).reshape(
            test_Qx + 1, test_Qy + 1, test_Qz + 1
        )

    S_epsilon = np.abs(S_true - S_num)
    S_o = math.sqrt(np.sum(S_true**2) / length)
    S_l_inf = S_epsilon.max() / S_o
    S_l_2 = math.sqrt(np.sum(S_epsilon**2) / length) / S_o
    print("S_l_inf=", S_l_inf, "S_L_2=", math.sqrt(np.sum(S_epsilon**2) / length), "S_l_2=", S_l_2)

    if temp:
        _plot_mid_slices_2d(S_num, S_true, x_min, x_max, y_min, y_max, z_min, z_max)
    if jump:
        _plot_jump_lines_1d(S_num, S_true, x_min, x_max, y_min, y_max, ratio1, ratio2)

    return S_num, S_true, S_l_inf, S_l_2, g_S
