import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns
import visual


np.random.seed(2)
plt.rcParams["font.family"] = "DejaVu Serif"


def _heatmap_on_axis(ax, data, title, x_min, x_max, y_min, y_max, xlabel="x1", ylabel="x2", cmap="rainbow", n_ticks=5):
    sns.heatmap(data.T, cmap=cmap, cbar=True, ax=ax)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    x_labels = np.round(np.linspace(x_min, x_max, n_ticks), decimals=2)
    y_labels = np.round(np.linspace(y_min, y_max, n_ticks), decimals=2)
    ax.set_xticks(np.linspace(0, data.shape[0], n_ticks))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax.set_yticks(np.linspace(0, data.shape[1], n_ticks))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_aspect("equal", adjustable="box")


def _jump_plot_on_axis(ax, S_num_0, S_true_0, x_min, x_max, n_ticks=7):
    ax.plot(range(int(len(S_num_0))), S_num_0, linestyle="--", color="red", label="Numerical Solution")
    ax.plot(range(int(len(S_true_0))), S_true_0, linestyle="-", color="blue", label="Exact Solution")
    ax.set_title("1D Slice", fontsize=14, pad=10)
    ax.set_xlabel("x1", fontsize=12)
    ax.legend(fontsize=8)
    x_labels = np.round(np.linspace(x_min, x_max, n_ticks), decimals=2)
    ax.set_xticks(np.linspace(0, len(S_num_0) - 1, n_ticks))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax.grid(True, alpha=0.3)


def _unwrap_model(model):
    while isinstance(model, (list, tuple)):
        if len(model) == 0:
            return None
        model = model[0]
    return model


def _call_model(model, batch_x, af, *extras):
    model = _unwrap_model(model)
    if model is None:
        return None
    if isinstance(af, str):
        af = af.strip()

    extras = list(extras)
    while extras and isinstance(extras[-1], (list, tuple)) and len(extras[-1]) == 0:
        extras.pop()

    last_error = None
    for end in range(len(extras), -1, -1):
        try:
            return model(batch_x, af, *extras[:end])
        except TypeError as exc:
            last_error = exc
    raise last_error


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
        if len(models) == 1 and isinstance(models[0], (list, tuple)):
            return list(models[0])
        return models
    return [models]


def _as_shape_list(shapes, n):
    if isinstance(shapes, str):
        return [shapes] * n
    if shapes is None or shapes == []:
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


def _slice_signed_distance(signed_distance, start_idx, end_idx, dtype, device):
    if isinstance(signed_distance, np.ndarray):
        return torch.tensor(signed_distance[start_idx:end_idx, :], dtype=dtype).to(device)
    return signed_distance


def test_p(Qx, Qy, x_min, x_max, y_min, y_max):
    test_Qx = 2 * Qx
    test_Qy = 2 * Qy
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    grid = np.array(list(itertools.product(x_devide, y_devide))).reshape(test_Qx + 1, test_Qy + 1, 2)
    sample_s = grid.reshape(-1, 2)
    return sample_s


def test(*args):
    mode = None

    if len(args) == 14:
        (
            models0,
            M,
            Qx,
            Qy,
            s_delta,
            s_star,
            af,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            device,
        ) = args
        sample_s = test_p(Qx, Qy, x_min, x_max, y_min, y_max)
        test_Qx = 2 * Qx
        test_Qy = 2 * Qy
        models1 = None
        models2 = None
        models3 = None
        Shape1 = []
        Shape2 = []
        sign_distances = []
        grad_temp = 0
        pos = 0.5
        mode = "ex41"
    elif len(args) == 22:
        (
            models0,
            models1,
            models2,
            Qx,
            Qy,
            w,
            af,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            x_left,
            x_right,
            y_below,
            y_upper,
            center_true,
            r_true,
            ana_S,
            grad_temp,
            device,
        ) = args
        sample_s = test_p(Qx, Qy, x_min, x_max, y_min, y_max)
        test_Qx = 2 * Qx
        test_Qy = 2 * Qy
        Shape1 = []
        Shape2 = []
        sign_distances = []
        models3 = None
        pos = 0.5
        mode = "ex44"
    elif len(args) == 25:
        (
            models0,
            models1,
            models2,
            M,
            Qx,
            Qy,
            w,
            af,
            Shape1,
            Shape2,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            x_left,
            x_right,
            y_below,
            y_upper,
            center_true,
            r_true,
            ana_S,
            grad_temp,
            device,
        ) = args
        sample_s = test_p(Qx, Qy, x_min, x_max, y_min, y_max)
        test_Qx = 2 * Qx
        test_Qy = 2 * Qy
        sign_distances = []
        models3 = None
        pos = 0.5
        mode = "shape25"
    elif len(args) == 26:
        (
            models0,
            models1,
            models2,
            models3,
            sample_s,
            w,
            af,
            Shape1,
            Shape2,
            sign_distances,
            temp,
            jump,
            x_min,
            x_max,
            y_min,
            y_max,
            x_left,
            x_right,
            y_below,
            y_upper,
            center_true,
            r_true,
            ana_S,
            pos,
            grad_temp,
            device,
        ) = args
        length = sample_s.shape[0]
        test_Qx = int(math.sqrt(length)) - 1
        test_Qy = int(math.sqrt(length)) - 1
        mode = "ex46"
    else:
        raise TypeError(f"Unsupported error_general_2d.test signature with {len(args)} arguments")

    length = sample_s.shape[0]
    test_point = torch.tensor(sample_s, requires_grad=True).to(device)

    S_num_pred = torch.zeros([length, 1], device=device, dtype=test_point.dtype)
    S_true_pred = None
    batch_size = max(1, int(test_point.shape[0] / 10))
    num_batches = int(np.ceil(test_point.shape[0] / batch_size))
    if mode == "ex41":
        w = torch.tensor(np.asarray(s_delta).reshape(-1, 1), dtype=test_point.dtype).to(device)
        w_true = torch.tensor(np.asarray(s_star).reshape(-1, 1), dtype=test_point.dtype).to(device)
        S_true_pred = torch.zeros([length, 1], device=device, dtype=test_point.dtype)
    else:
        w = torch.tensor(np.asarray(w).reshape(-1, 1), dtype=test_point.dtype).to(device)
    g_S_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        if i < num_batches - 1:
            end_idx = (i + 1) * batch_size
        else:
            end_idx = test_point.shape[0]

        batch_x = test_point[start_idx:end_idx, :]

        with torch.set_grad_enabled(grad_temp == 1):
            if mode == "ex41":
                af_basis = _call_model(models0, batch_x, af, []).to(device)
                if af_basis.ndim != 2:
                    af_basis = af_basis.reshape(batch_x.shape[0], -1)
                batch_output = torch.mm(af_basis, w)
                batch_output_true = torch.mm(af_basis, w_true)
                del af_basis
            elif mode == "ex44":
                if af == "mix":
                    basis_0 = _call_model(models0, batch_x, "Tanh").to(device)
                    basis_1 = _call_model(models1, batch_x, "Gauss").to(device)
                    basis_2 = _call_model(models2, batch_x, "Gauss").to(device)
                    af_basis = torch.cat((basis_0, basis_1, basis_2), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del basis_0, basis_1, basis_2, af_basis
                elif af == "Gauss":
                    basis_1 = _call_model(models1, batch_x, "Gauss").to(device)
                    basis_2 = _call_model(models2, batch_x, "Gauss").to(device)
                    af_basis = torch.cat((basis_1, basis_2), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del basis_1, basis_2, af_basis
                else:
                    af_basis = _call_model(models0, batch_x, af).to(device)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis

            elif mode == "shape25":
                if af == "mix":
                    basis_list = []
                    if _unwrap_model(models0) is not None:
                        basis_list.append(_call_model(models0, batch_x, "Tanh", []).to(device))
                    if _unwrap_model(models1) is not None:
                        basis_list.append(_call_model(models1, batch_x, "sigmoid", Shape1).to(device))
                    if _unwrap_model(models2) is not None:
                        basis_list.append(_call_model(models2, batch_x, "sigmoid", Shape2).to(device))
                    af_basis = torch.cat(tuple(basis_list), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis, basis_list
                elif af == "circle+rec":
                    basis_list = []
                    if _unwrap_model(models1) is not None:
                        basis_list.append(_call_model(models1, batch_x, "sigmoid", Shape1).to(device))
                    if _unwrap_model(models2) is not None:
                        basis_list.append(_call_model(models2, batch_x, "sigmoid", Shape2).to(device))
                    af_basis = torch.cat(tuple(basis_list), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis, basis_list
                else:
                    af_basis = _call_model(models0, batch_x, af, Shape1).to(device)
                    if af_basis.ndim != 2:
                        af_basis = af_basis.reshape(batch_x.shape[0], -1)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis

            else:
                if af == "mix_general":
                    basis_list = []
                    basis_0 = _call_model(models0, batch_x, "Tanh", [], [])
                    if basis_0 is not None:
                        basis_list.append(basis_0.to(device))

                    extra_models = _as_model_list(models1)
                    extra_shapes = _as_shape_list(Shape1, len(extra_models))
                    if not extra_models:
                        extra_models = _as_model_list(models2) + _as_model_list(models3)
                        extra_shapes = _as_shape_list(Shape2, len(extra_models))
                    signed_distance_list = _as_signed_distance_list(sign_distances)
                    signed_idx = 0
                    for model, shape in zip(extra_models, extra_shapes):
                        if shape == "general":
                            if signed_idx >= len(signed_distance_list):
                                raise ValueError("Not enough signed-distance arrays for general basis branches.")
                            sign_distances_loc = _slice_signed_distance(
                                signed_distance_list[signed_idx], start_idx, end_idx, test_point.dtype, device
                            )
                            basis = _call_model(model, batch_x, "sigmoid", "general", sign_distances_loc)
                            signed_idx += 1
                        else:
                            basis = _call_model(model, batch_x, "sigmoid", shape, [])
                        if basis is not None:
                            basis_list.append(basis.to(device))
                    af_basis = torch.cat(tuple(basis_list), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis, basis_list
                elif af == "mix+gauss":
                    if isinstance(sign_distances, np.ndarray):
                        sign_distances_loc = torch.tensor(sign_distances[start_idx:end_idx, :], dtype=test_point.dtype).to(device)
                    else:
                        sign_distances_loc = sign_distances
                    basis_0 = _call_model(models0, batch_x, "Tanh", [], []).to(device)
                    basis_1 = _call_model(models1, batch_x, "sigmoid", Shape1, sign_distances_loc).to(device)
                    basis_2 = _call_model(models2, batch_x, "sigmoid", Shape2, []).to(device)
                    basis_3 = _call_model(models3, batch_x, "continue_gauss", [], []).to(device)
                    af_basis = torch.cat((basis_0, basis_1, basis_2, basis_3), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del basis_0, basis_1, basis_2, basis_3, af_basis
                elif af == "mix":
                    sign_distances_single = _first_signed_distance(sign_distances)
                    sign_distances_loc = _slice_signed_distance(sign_distances_single, start_idx, end_idx, test_point.dtype, device)
                    basis_list = []
                    if _unwrap_model(models0) is not None:
                        basis_list.append(_call_model(models0, batch_x, "Tanh", [], []).to(device))
                    if _unwrap_model(models1) is not None:
                        basis_list.append(_call_model(models1, batch_x, "sigmoid", Shape1, sign_distances_loc).to(device))
                    if _unwrap_model(models2) is not None:
                        basis_list.append(_call_model(models2, batch_x, "sigmoid", Shape2, []).to(device))
                    af_basis = torch.cat(tuple(basis_list), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis, basis_list
                elif af == "circle+rec":
                    basis_1 = _call_model(models1, batch_x, "sigmoid", Shape1, []).to(device)
                    basis_2 = _call_model(models2, batch_x, "sigmoid", Shape2, []).to(device)
                    af_basis = torch.cat((basis_1, basis_2), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del basis_1, basis_2, af_basis
                elif af == "sigmoid" and Shape1 == "general" and Shape2 != "noise":
                    af_basis = _call_model(models0, batch_x, "sigmoid", Shape1, sign_distances[start_idx:end_idx, :]).to(device)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis
                elif af == "sigmoid" and Shape1 == "general" and Shape2 == "noise":
                    basis_0 = _call_model(models0, batch_x, "sigmoid", Shape1, sign_distances[start_idx:end_idx, :]).to(device)
                    basis_1 = _call_model(models1, batch_x, "sigmoid", Shape2, []).to(device)
                    af_basis = torch.cat((basis_0, basis_1), dim=1)
                    batch_output = torch.mm(af_basis, w)
                    del basis_0, basis_1, af_basis
                else:
                    af_basis = _call_model(models0, batch_x, af, Shape1, []).to(device)
                    batch_output = torch.mm(af_basis, w)
                    del af_basis

        if grad_temp == 1:
            grad_S_batch = torch.autograd.grad(
                outputs=batch_output,
                inputs=batch_x,
                grad_outputs=torch.ones_like(batch_output),
                create_graph=False,
            )[0]
            g_S_list.append(grad_S_batch.cpu().detach().numpy())
            del grad_S_batch

        S_num_pred[start_idx:end_idx, :] = batch_output.detach()
        if mode == "ex41":
            S_true_pred[start_idx:end_idx, :] = batch_output_true.detach()
            del batch_output_true
        del batch_output

    if grad_temp == 1:
        g_S_num = np.concatenate(g_S_list, axis=0)
        g_S = np.linalg.norm(g_S_num, axis=-1)
        g_S[g_S > 100] = 0
        # if mode == "ex46":
        #     g_S[g_S > 80] = 0
        # else:
        #     g_S[g_S > 500] = 0
        g_S = g_S.reshape(test_Qx + 1, test_Qy + 1)
        del w, test_point
    else:
        g_S = []

    S_num = (S_num_pred.cpu().detach().numpy()).reshape(test_Qx + 1, test_Qy + 1)
    S_num_0 = S_num[:, int(pos * S_num.shape[1]) + 1 - 1]
    if mode == "ex41":
        S_true = (S_true_pred.cpu().detach().numpy()).reshape(test_Qx + 1, test_Qy + 1)
    else:
        S_true = ana_S(sample_s, x_left, x_right, y_below, y_upper, center_true, r_true).reshape(test_Qx + 1, test_Qy + 1)
    S_true_0 = S_true[:, int(pos * S_true.shape[1]) + 1 - 1]
    S_epsilon = np.abs((S_true - S_num))
    S_o = math.sqrt(np.sum(S_true**2) / length)
    S_l_inf = S_epsilon.max() / S_o
    S_l_2 = math.sqrt(np.sum(S_epsilon**2) / length) / S_o
    summary_text = (
        "S_l_inf= "
        + str(S_l_inf)
        + " S_L_2= "
        + str(math.sqrt(np.sum(S_epsilon**2) / length))
        + " S_l_2= "
        + str(S_l_2)
    )

    plot_items = []
    if grad_temp == 1:
        plot_items.append("grad")
    if temp:
        plot_items.append("temp")
    if jump:
        plot_items.append("jump")

    if plot_items:
        fig, axes = plt.subplots(1, len(plot_items), figsize=(4.8 * len(plot_items), 3.8), dpi=120)
        if len(plot_items) == 1:
            axes = [axes]

        for ax, item in zip(axes, plot_items):
            if item == "grad":
                _heatmap_on_axis(
                    ax,
                    g_S,
                    title="gradient_S",
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    cmap="rainbow",
                )
            elif item == "temp":
                _heatmap_on_axis(
                    ax,
                    S_num,
                    title="Numerical Solution of S",
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    cmap="rainbow",
                )
            elif item == "jump":
                _jump_plot_on_axis(ax, S_num_0, S_true_0, x_min, x_max)

        plt.tight_layout()
        plt.show()

    print(summary_text)

    if mode == "ex41":
        return S_num, S_true, S_l_inf, S_l_2
    return S_num, S_true, S_l_inf, S_l_2, g_S
