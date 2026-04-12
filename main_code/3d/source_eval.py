import math
import numpy as np
import torch


def _reshape_basis(basis, num_points):
    if basis.ndim == 1:
        return basis.reshape(num_points, 1)
    return basis.reshape(num_points, -1)


def valgrad(models, af, M, all_points, w, ana_S, center, *extra, verbose=True):
    device = extra[-1]
    length = all_points.shape[0]
    all_g_p = torch.tensor(all_points, requires_grad=True).to(device)
    w_last = torch.tensor(w.reshape(-1, 1)).to(device)

    if len(extra) == 4:
        R, r, S0, _device = extra
        S_basis_ = _reshape_basis(models(all_g_p, af, [], device), length).to(device)
        S_true = ana_S(all_points, center, R, r, S0).reshape(-1, 1)
    elif len(extra) == 2:
        r, _device = extra
        S_basis_ = _reshape_basis(models(all_g_p, af, device), length).to(device)
        S_true = ana_S(all_points, [], [], [], [], [], [], center[0], center[1], r).reshape(-1, 1)
    else:
        raise TypeError("Unsupported source_eval signature")

    if S_basis_.shape[1] != w_last.shape[0]:
        raise ValueError(
            f"source_eval basis width {S_basis_.shape[1]} does not match coefficient width {w_last.shape[0]}"
        )

    S_num = torch.mm(S_basis_, w_last)
    grad_S_num = torch.autograd.grad(
        outputs=S_num,
        inputs=all_g_p,
        grad_outputs=torch.ones_like(S_num),
        create_graph=True,
    )[0]
    S_num_np = S_num.cpu().detach().numpy()
    S_epsilon = np.abs(S_true - S_num_np)
    S_o = math.sqrt(np.sum(S_true**2) / length)
    if verbose:
        print(
            "S_l_inf=",
            S_epsilon.max() / S_o,
            "S_L_2=",
            math.sqrt(np.sum(S_epsilon**2) / length),
            "S_l_2=",
            math.sqrt(np.sum(S_epsilon**2) / length) / S_o,
        )
    return S_num, grad_S_num
