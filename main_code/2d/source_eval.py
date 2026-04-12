"""Shared 2D source/gradient evaluator."""

import math

import numpy as np
import torch


def _unwrap_model(model):
    current = model
    while isinstance(current, (list, tuple)):
        if not current:
            raise ValueError("Empty model container.")
        current = current[0]
    return current


def _eval_basis(models, af, all_g_p, extra_args):
    model = _unwrap_model(models)
    return model(all_g_p, af, *extra_args)


def valgrad(*args, verbose=False):
    if len(args) == 9:
        models, af, M, all_points, w, ana_S, center, r, device = args
        extra_args = ()
    elif len(args) == 10:
        models, af, shape, M, all_points, w, ana_S, center, r, device = args
        extra_args = (shape,)
    elif len(args) == 11:
        models, af, shape, signed_distances, M, all_points, w, ana_S, center, r, device = args
        extra_args = (shape, signed_distances)
    else:
        raise TypeError("Unsupported source_eval.valgrad signature")

    length = all_points.shape[0]
    all_g_p = torch.as_tensor(all_points, dtype=torch.get_default_dtype(), device=device).requires_grad_(True)
    w_last = torch.as_tensor(np.asarray(w).reshape(-1, 1), dtype=torch.get_default_dtype(), device=device)

    S_basis_ = _eval_basis(models, af, all_g_p, extra_args).reshape(-1, M).to(device)
    S_num = torch.mm(S_basis_, w_last)
    grad_S_num = torch.autograd.grad(
        outputs=S_num,
        inputs=all_g_p,
        grad_outputs=torch.ones_like(S_num),
        create_graph=True,
    )[0]

    S_true = ana_S(
        all_points,
        x_left=0.29,
        x_right=0.49,
        y_below=0.3,
        y_upper=0.7,
        center=center,
        r=r,
    ).reshape(-1, 1)
    S_num_np = S_num.detach().cpu().numpy()
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


__all__ = ["valgrad"]
