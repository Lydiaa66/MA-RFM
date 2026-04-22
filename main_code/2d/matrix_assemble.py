import pickle

import generate_data
import numpy as np
import torch

import inverse_solver
torch.set_default_dtype(torch.float64)


def _unwrap_model(model):
    current = model
    while isinstance(current, (list, tuple)):
        if not current:
            return None
        current = current[0]
    return current


def _call_model(model, all_g_p, af, *extra_args):
    resolved = _unwrap_model(model)
    if resolved is None:
        return None
    args = list(extra_args)
    while True:
        try:
            return resolved(all_g_p, af, *args)
        except TypeError:
            if args and isinstance(args[-1], (list, tuple)) and len(args[-1]) == 0:
                args.pop()
                continue
            raise


def _as_model_list(models):
    if models is None:
        return []
    if isinstance(models, list) and not models:
        return []
    if isinstance(models, tuple):
        if not models:
            return []
        return list(models)
    if isinstance(models, list):
        if not models:
            return []
        if len(models) == 1 and isinstance(models[0], (list, tuple)):
            return list(models[0])
        return models
    return [models]


def _as_shape_list(shapes, n):
    if isinstance(shapes, str):
        return [shapes] * n
    if shapes in (None, []):
        return [[] for _ in range(n)]
    if isinstance(shapes, tuple):
        shapes = list(shapes)
    if isinstance(shapes, list):
        if len(shapes) == n:
            return shapes
        if len(shapes) == 1:
            return shapes * n
    return [shapes for _ in range(n)]


def _as_signed_distance_list(sign_distances):
    if isinstance(sign_distances, tuple):
        return list(sign_distances)
    if isinstance(sign_distances, list):
        return sign_distances
    return [sign_distances]


def _first_signed_distance(sign_distances):
    if isinstance(sign_distances, (list, tuple)):
        if not sign_distances:
            return sign_distances
        return sign_distances[0]
    return sign_distances


def _is_radial_boundary(points_b):
    if len(points_b) == 0:
        return False
    radii = np.linalg.norm(points_b, axis=1)
    return np.max(radii) > 0 and np.std(radii) <= 1e-10 * np.max(radii) + 1e-12


def _prepare_cells(cells, device):
    cells_copy = pickle.loads(pickle.dumps(cells))
    all_g_t = []
    weight_store = []
    for cell in cells_copy:
        all_g_t.append(torch.tensor(cell.gauss_points, requires_grad=True).to(device))
        weight_store.append(cell.w)
    return cells_copy, all_g_t, weight_store


def _legacy_batch(sample, num_batches, model, af, M):
    del M
    n = sample.shape[1]
    out_tensor = None
    batch_size = max(int(n / num_batches), 1)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = n if i == num_batches - 1 else min((i + 1) * batch_size, n)
        if start_idx >= n:
            break
        batch_x = sample[:, start_idx:end_idx, :]
        batch_output = _call_model(model, batch_x, af, [])
        if out_tensor is None:
            out_tensor = torch.zeros(
                (sample.shape[0], n, batch_output.shape[-1]),
                dtype=batch_output.dtype,
                device=batch_output.device,
            )
        out_tensor[:, start_idx:end_idx, :] = batch_output
    if out_tensor is None:
        return torch.zeros((sample.shape[0], 0, 0), dtype=sample.dtype, device=sample.device)
    return out_tensor


def _assemble_plain_basis(models0, models1, models2, all_g_p, af):
    if af == "mix":
        out_cell00 = _call_model(models0, all_g_p, "Tanh")
        out_cell01 = _call_model(models1, all_g_p, "Gauss")
        out_cell02 = _call_model(models2, all_g_p, "Gauss")
        return torch.cat((out_cell00, out_cell01, out_cell02), dim=1)
    if af == "Gauss":
        out_cell01 = _call_model(models1, all_g_p, "Gauss")
        out_cell02 = _call_model(models2, all_g_p, "Gauss")
        return torch.cat((out_cell01, out_cell02), dim=1)
    return _call_model(models0, all_g_p, af)


def _assemble_shape_basis(models0, models1, models2, all_g_p, af, shape1, shape2):
    if af == "mix":
        out_cell00 = _call_model(models0, all_g_p, "Tanh", [])
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2)
        return torch.cat((out_cell00, out_cell01, out_cell02), dim=1)
    if af == "circle+rec":
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2)
        return torch.cat((out_cell01, out_cell02), dim=1)
    return _call_model(models0, all_g_p, af, [])


def _assemble_legacy_basis(models0, models1, models2, all_g_p, af, shape1, shape2):
    if af == "mix":
        out_cell00 = _call_model(models0, all_g_p, "Tanh", [])
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1)
        if models2 in ([], (), None):
            return torch.cat((out_cell00, out_cell01), dim=1)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2)
        return torch.cat((out_cell00, out_cell01, out_cell02), dim=1)
    if af == "circle+rec":
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2)
        return torch.cat((out_cell01, out_cell02), dim=1)
    return _call_model(models0, all_g_p, af, shape1)


def _assemble_signed_basis(models0, models1, models2, models3, all_g_p, af, shape1, shape2, sign_distances, device):
    if isinstance(sign_distances, np.ndarray):
        sign_distances = torch.tensor(sign_distances).to(device)

    if af == "mix_general":
        basis_list = []
        tanh_basis = _call_model(models0, all_g_p, "Tanh", [], [])
        if tanh_basis is not None:
            basis_list.append(tanh_basis)

        extra_models = _as_model_list(models1)
        extra_shapes = _as_shape_list(shape1, len(extra_models))
        if not extra_models:
            extra_models = _as_model_list(models2) + _as_model_list(models3)
            extra_shapes = _as_shape_list(shape2, len(extra_models))
        signed_distance_list = _as_signed_distance_list(sign_distances)
        signed_idx = 0
        for model, shape in zip(extra_models, extra_shapes):
            if shape == "general":
                if signed_idx >= len(signed_distance_list):
                    raise ValueError("Not enough signed-distance arrays for general basis branches.")
                basis = _call_model(model, all_g_p, "sigmoid", "general", signed_distance_list[signed_idx])
                signed_idx += 1
            else:
                basis = _call_model(model, all_g_p, "sigmoid", shape, [])
            if basis is not None:
                basis_list.append(basis)
        return torch.cat(tuple(basis_list), dim=1)

    if af == "mix+gauss":
        basis_list = []
        out_cell00 = _call_model(models0, all_g_p, "Tanh", [], [])
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1, sign_distances)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2, [])
        out_cell03 = _call_model(models3, all_g_p, "continue_gauss", [], [])
        for basis in (out_cell00, out_cell01, out_cell02, out_cell03):
            if basis is not None:
                basis_list.append(basis)
        return torch.cat(tuple(basis_list), dim=1)
    if af == "mix":
        basis_list = []
        sign_distances_single = _first_signed_distance(sign_distances)
        out_cell00 = _call_model(models0, all_g_p, "Tanh", [], [])
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1, sign_distances_single)
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2, [])
        for basis in (out_cell00, out_cell01, out_cell02):
            if basis is not None:
                basis_list.append(basis)
        return torch.cat(tuple(basis_list), dim=1)
    if af == "circle+rec":
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape1, [])
        out_cell02 = _call_model(models2, all_g_p, "sigmoid", shape2, [])
        return torch.cat((out_cell01, out_cell02), dim=1)
    if af == "sigmoid" and shape1 == "general" and shape2 != "noise":
        return _call_model(models0, all_g_p, "sigmoid", shape1, sign_distances)
    if af == "sigmoid" and shape1 == "general" and shape2 == "noise":
        out_cell00 = _call_model(models0, all_g_p, "sigmoid", shape1, sign_distances)
        out_cell01 = _call_model(models1, all_g_p, "sigmoid", shape2, [])
        return torch.cat((out_cell00, out_cell01), dim=1)
    return _call_model(models0, all_g_p, af, [], [])


def _assemble_operator_normal(points_b, all_g_p, out_cell, weight_store, kk, device, cupy_device):
    X = points_b[:, np.newaxis, :]
    Y = all_g_p.cpu().detach().numpy()[np.newaxis, :, :]
    diff = X - Y
    radius = np.sqrt(points_b[0, 0] ** 2 + points_b[0, 1] ** 2)
    normal = np.tile(X, (1, all_g_p.shape[0], 1)) / radius
    RR = np.linalg.norm(diff, axis=2)
    W = np.concatenate(weight_store, axis=0).reshape(-1)

    b_matrix = []
    db_matrix = []
    mea = 1
    batch_number = 1
    normal_op = getattr(generate_data, "forward_operator_rec_normal", None)
    for k in kk:
        if normal_op is not None:
            f = normal_op(k, RR, W, diff, normal, mea, device, cupy_device)
        else:
            f = generate_data.forward_operator_rec(k, RR, W, diff, normal, mea, device, cupy_device)
        b_matrix.append(f.int_f(out_cell, batch_number).get())
        db_matrix.append(f.int_df(out_cell, batch_number).get())

    return inverse_solver.A(b_matrix, db_matrix)


def _assemble_operator_rect(points_b, all_g_p, out_cell, weight_store, kk, device, cupy_device):
    X = points_b[:, np.newaxis, :]
    Y = all_g_p.cpu().detach().numpy()[np.newaxis, :, :]
    diff = X - Y
    RR = np.linalg.norm(diff, axis=2)
    half = int(0.5 * RR.shape[0])
    sub_y1 = diff[:, :, 0][:half, :]
    sub_y2 = diff[:, :, 1][half: RR.shape[0], :]
    W = np.concatenate(weight_store, axis=0).reshape(-1)

    b_matrix = []
    db_x1_matrix = []
    db_x2_matrix = []
    mea = 1
    batch_number = 1
    for k in kk:
        f = generate_data.forward_operator_rec(k, RR, W, sub_y1, sub_y2, mea, device, cupy_device)
        b_matrix.append(f.int_f(out_cell, batch_number).get())
        db_x1_matrix.append(f.int_df_x1(out_cell, batch_number).get())
        db_x2_matrix.append(f.int_df_x2(out_cell, batch_number).get())

    return inverse_solver.A(b_matrix, db_x1_matrix, db_x2_matrix)


def _compute_basis_gradients(out_cell, all_g_p, grad_temp):
    width = out_cell.shape[1]
    grads_y1 = np.zeros([out_cell.shape[0], width])
    grads_y2 = np.zeros([out_cell.shape[0], width])
    if grad_temp != 1:
        return grads_y1, grads_y2

    for i in range(width):
        grad = torch.autograd.grad(
            outputs=out_cell[:, i],
            inputs=all_g_p,
            grad_outputs=torch.ones_like(out_cell[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads_y1[:, i] = grad[:, 0].cpu().detach().numpy()
        grads_y2[:, i] = grad[:, 1].cpu().detach().numpy()
    return grads_y1, grads_y2


def _assemble_active_cells(cells, device, basis_builder, points_b, kk, cupy_device, operator_mode, grad_temp=None):
    cells_copy, all_g_t, weight_store = _prepare_cells(cells, device)
    all_g_p = torch.cat(all_g_t, dim=0)
    all_g_p.requires_grad_(True)
    out_cell = basis_builder(all_g_p)

    grads_y1 = None
    grads_y2 = None
    if grad_temp is not None:
        grads_y1, grads_y2 = _compute_basis_gradients(out_cell, all_g_p, grad_temp)

    if operator_mode == "auto":
        use_normal = _is_radial_boundary(points_b)
    elif operator_mode == "normal":
        use_normal = True
    else:
        use_normal = False

    if use_normal:
        all_mat = _assemble_operator_normal(points_b, all_g_p, out_cell, weight_store, kk, device, cupy_device)
    else:
        all_mat = _assemble_operator_rect(points_b, all_g_p, out_cell, weight_store, kk, device, cupy_device)

    return all_mat, all_g_t, all_g_p, cells_copy, grads_y1, grads_y2


def _parse_mat_assemble2_args(args):
    if len(args) == 6:
        af, kk, temp, points_b, device, cupy_device = args
        return "plain", (), af, kk, temp, points_b, device, cupy_device
    if len(args) == 8:
        af, shape1, shape2, kk, temp, points_b, device, cupy_device = args
        return "shape", (shape1, shape2), af, kk, temp, points_b, device, cupy_device
    if len(args) == 10:
        models3, af, shape1, shape2, sign_distances, kk, temp, points_b, device, cupy_device = args
        return "signed", (models3, shape1, shape2, sign_distances), af, kk, temp, points_b, device, cupy_device
    raise TypeError("Unsupported matrix_assemble.mat_assemble2 signature")


def array_difference(B, A):
    A_tuples = [tuple(row) for row in A]
    B_tuples = [tuple(row) for row in B]
    
    diff_tuples = [b for b in B_tuples if b not in A_tuples]
    
    return np.array(diff_tuples)


def residual_vector(
    models,
    p_b,
    x_min,
    x_max,
    y_min,
    y_max,
    center_temp,
    r_temp,
    M,
    af,
    choose,
    mea,
    kk,
    number,
    num_batches_gauss,
    batch_number_mea,
    grad_temp,
    device,
    cupy_device,
):
    points_b = torch.tensor(p_b).to(device)
    points_b.requires_grad_(True)

    if choose == "rec":
        g = generate_data.GaussLegendre2D_rec(number, x_min, x_max, y_min, y_max, p_b)
        gauss_points = g.points_int
        RR = g.R
        w = g.w
        sub_y1 = g.sub_y1
        sub_y2 = g.sub_y2
    elif choose == "circle":
        g = generate_data.GaussLegendre2D_circle(number, center_temp, r_temp, p_b)
        gauss_points = g.y
        RR = g.RR
        w = g.w
        r_points = g.points_int[:, 0]
        sub_y1 = g.sub_y1
        sub_y2 = g.sub_y2
    else:
        raise ValueError(f"Unsupported choose value: {choose}")

    g_p = torch.tensor(gauss_points, requires_grad=True).to(device)
    g_p_T = g_p[None,]
    out_tensor = _legacy_batch(g_p_T, num_batches_gauss, models, af, M).to(device)
    S_basis = out_tensor.reshape(-1, M).to(device)

    grads_y1 = np.zeros([S_basis.shape[0], M])
    grads_y2 = np.zeros([S_basis.shape[0], M])
    if grad_temp:
        for i in range(M):
            grad = torch.autograd.grad(
                outputs=S_basis[:, i],
                inputs=g_p,
                grad_outputs=torch.ones_like(S_basis[:, i]),
                create_graph=True,
                retain_graph=True,
            )[0]
            grads_y1[:, i] = grad[:, 0].cpu().detach().numpy()
            grads_y2[:, i] = grad[:, 1].cpu().detach().numpy()

    S_basis = S_basis.cpu().detach().numpy()
    b_matrix = []
    db_x1_matrix = []
    db_x2_matrix = []

    for k in kk:
        if choose == "rec":
            f = generate_data.forward_operator_rec(k, RR, w, sub_y1, sub_y2, mea, device, cupy_device)
        else:
            f = generate_data.forward_operator_circle(k, RR, w, r_points, sub_y1, sub_y2, mea, device, cupy_device)
        b_matrix.append(f.int_f(S_basis, batch_number_mea).get())
        db_x1_matrix.append(f.int_df_x1(S_basis, batch_number_mea).get())
        db_x2_matrix.append(f.int_df_x2(S_basis, batch_number_mea).get())

    return S_basis, b_matrix, db_x1_matrix, db_x2_matrix, grads_y1, grads_y2


def mat_assemble(cells, models0, models1, models2, af, shape1, shape2, kk, temp, points_b, grad_temp, device, cupy_device):
    if temp == 0:
        return _assemble_active_cells(
            cells,
            device,
            lambda all_g_p: _assemble_legacy_basis(models0, models1, models2, all_g_p, af, shape1, shape2),
            points_b,
            kk,
            cupy_device,
            operator_mode="rect",
            grad_temp=grad_temp,
        )

    cells_copy = pickle.loads(pickle.dumps(cells))
    all_mat = []
    all_g_t = []
    all_g_p = torch.empty((0, 2), device=device)
    grads_y1 = np.zeros((0, 0))
    grads_y2 = np.zeros((0, 0))
    return all_mat, all_g_t, all_g_p, cells_copy, grads_y1, grads_y2


def mat_assemble2(cells, models0, models1, models2, *args):
    mode, extra, af, kk, temp, points_b, device, cupy_device = _parse_mat_assemble2_args(args)
    if temp == 0:
        if mode == "plain":
            basis_builder = lambda all_g_p: _assemble_plain_basis(models0, models1, models2, all_g_p, af)
            operator_mode = "auto"
        elif mode == "shape":
            basis_builder = lambda all_g_p: _assemble_shape_basis(models0, models1, models2, all_g_p, af, extra[0], extra[1])
            operator_mode = "rect"
        else:
            basis_builder = lambda all_g_p: _assemble_signed_basis(
                models0, models1, models2, extra[0], all_g_p, af, extra[1], extra[2], extra[3], device
            )
            operator_mode = "rect"

        all_mat, all_g_t, all_g_p, cells_copy, _, _ = _assemble_active_cells(
            cells,
            device,
            basis_builder,
            points_b,
            kk,
            cupy_device,
            operator_mode=operator_mode,
        )
        return all_mat, all_g_t, all_g_p, cells_copy

    cells_copy, all_g_t, _ = _prepare_cells(cells, device)
    all_mat = []
    all_g_p = torch.cat(all_g_t, dim=0)
    return all_mat, all_g_t, all_g_p, cells_copy
