import pickle

import generate_data
import numpy as np
import inverse_solver
import torch

torch.set_default_dtype(torch.float64)


def mat_assemble2(cells, models0, models1, models2, *args):
    if len(args) == 9 and callable(args[0]):
        model_gauss1, model_gauss2, S_basis, af, kk, temp, points_b, device, cupy_device = args
        shape = None
        condition = None
        mode = "ex47"
    elif len(args) == 9 and isinstance(args[3], str):
        model_gauss1, model_gauss2, S_basis, af, kk, temp, points_b, device, cupy_device = args
        shape = None
        condition = None
        mode = "ex47"
    elif len(args) == 9:
        model_gauss1 = None
        model_gauss2 = None
        S_basis, af, shape, kk, temp, points_b, condition, device, cupy_device = args
        mode = "donut"
    else:
        raise TypeError("Unsupported mat_assemble2 signature")

    cells_copy = pickle.loads(pickle.dumps(cells))
    all_mat = []
    all_g_t = []
    weight_store = []
    all_g_p = None

    if temp == 0:
        for cell in cells_copy:
            loc_g = cell.gauss_points
            all_g_t.append(torch.tensor(loc_g, requires_grad=True).to(device))
            weight_store.append(cell.w)

        all_g_p = torch.cat(all_g_t, dim=0)

        if af == "mix":
            out_cell00 = torch.tensor(S_basis).to(device)
            if mode == "donut":
                out_cell01 = models1(all_g_p, "sigmoid", shape, device)
                out_cell = torch.cat((out_cell00, out_cell01), dim=1)
            else:
                out_cell01 = models1(all_g_p, "relu", device)
                out_cell02 = models2(all_g_p, "relu", device)
                out_cell = torch.cat((out_cell00, out_cell01, out_cell02), dim=1)
        elif af == "mixer":
            out_cell00 = torch.tensor(S_basis).to(device)
            out_cell01 = models1(all_g_p, "relu", device)
            out_cell02 = models2(all_g_p, "relu", device)
            out_cell03 = model_gauss1(all_g_p, "Gauss", device)
            out_cell04 = model_gauss2(all_g_p, "Gauss", device)
            out_cell = torch.cat((out_cell00, out_cell01, out_cell02, out_cell03, out_cell04), dim=1)
        elif af == "Gauss":
            out_cell01 = models1(all_g_p, "Gauss", device)
            out_cell02 = models2(all_g_p, "Gauss", device)
            out_cell = torch.cat((out_cell01, out_cell02), dim=1)
        elif af == "relu":
            out_cell01 = models1(all_g_p, "relu", device)
            out_cell02 = models2(all_g_p, "relu", device)
            out_cell = torch.cat((out_cell01, out_cell02), dim=1)
        elif af == "sigmoid" and shape == "sweet":
            out_cell = models1(all_g_p, "sigmoid", "sweet", device)
        else:
            if mode == "donut":
                out_cell = models0(all_g_p, af, shape, device)
            else:
                out_cell = models0(all_g_p, af, device)

        S_basis = out_cell.cpu().detach().numpy()
        X = points_b[:, np.newaxis, :]
        Y = all_g_p.cpu().detach().numpy()[np.newaxis, :, :]
        diff = X - Y
        RR = np.linalg.norm(diff, axis=2)
        sub_y1 = diff[:, :, 0][: int(2 / 6 * RR.shape[0]), :]
        sub_y2 = diff[:, :, 1][int(2 / 6 * RR.shape[0]) : int(4 / 6 * RR.shape[0]), :]
        sub_y3 = diff[:, :, 2][int(4 / 6 * RR.shape[0]) : int(RR.shape[0]), :]
        W = np.concatenate(weight_store, axis=0).reshape(-1)

        b_matrix = []
        db_x1_matrix = []
        db_x2_matrix = []
        db_x3_matrix = []
        mea = 1
        batch_number = max(1, int(np.ceil(out_cell.shape[1] / 512)))
        for i in range(len(kk)):
            f = generate_data.forward_operator_rec(
                kk[i], RR, W, sub_y1, sub_y2, sub_y3, mea, device, cupy_device
            )
            b_matrix.append(f.int_f(out_cell, batch_number, device, cupy_device).get())
            if mode == "donut" and condition == "Dirichlet":
                continue
            db_x1_matrix.append(f.int_df_x1(out_cell, batch_number, device, cupy_device).get())
            db_x2_matrix.append(f.int_df_x2(out_cell, batch_number, device, cupy_device).get())
            db_x3_matrix.append(f.int_df_x3(out_cell, batch_number, device, cupy_device).get())

        if mode == "donut":
            all_mat = inverse_solver.A(b_matrix, db_x1_matrix, db_x2_matrix, db_x3_matrix, condition)
        else:
            all_mat = inverse_solver.A(b_matrix, db_x1_matrix, db_x2_matrix, db_x3_matrix)

    return all_mat, all_g_t, all_g_p, cells_copy, S_basis
