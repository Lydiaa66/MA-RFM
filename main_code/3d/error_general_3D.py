import math
import warnings

import generate_data3
import matplotlib.pyplot as plt
import numpy as np
import torch
from importlib import reload
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


def test(*args):
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

    S_num_pred = torch.zeros([length, 1])
    batch_size = int(test_point.shape[0] / (test_Qx + 1))
    num_batches = int(test_point.shape[0] / batch_size)
    w = torch.tensor(w.reshape(-1, 1)).to(device)
    g_S_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_batches - 1 else test_point.shape[0] + 1
        batch_x = test_point[start_idx:end_idx, :]

        if af == "mixer":
            with torch.no_grad():
                basis_0 = models0(batch_x, "Tanh", device).to(device)
                basis_1 = model_relu1(batch_x, "relu", device).to(device)
                basis_2 = model_relu2(batch_x, "relu", device).to(device)
                basis_3 = model_gauss1(batch_x, "Gauss", device).to(device)
                basis_4 = model_gauss2(batch_x, "Gauss", device).to(device)
            af_basis = torch.cat((basis_0, basis_1, basis_2, basis_3, basis_4), dim=1)
        elif af == "mix":
            with torch.no_grad():
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
            with torch.no_grad():
                basis_1 = model_gauss1(batch_x, "Gauss", device).to(device)
                basis_2 = model_gauss2(batch_x, "Gauss", device).to(device)
            af_basis = torch.cat((basis_1, basis_2), dim=1)
        elif af == "relu":
            with torch.no_grad():
                basis_1 = model_relu1(batch_x, "relu", device).to(device)
                basis_2 = model_relu2(batch_x, "relu", device).to(device)
            af_basis = torch.cat((basis_1, basis_2), dim=1)
        elif af == "sigmoid" and shape == "sweet":
            af_basis = model_relu1(batch_x, "sigmoid", shape, device).to(device)
        else:
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
        S_num_pred[start_idx:end_idx, :] = batch_output

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

    return S_num, S_true, S_l_inf, S_l_2, g_S
