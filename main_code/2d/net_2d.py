import torch
import torch.nn as nn


torch.set_default_dtype(torch.float64)


_ELLIPTIC_GAUSS_SHAPES = {
    "elliptic_gauss",
    "elliptic_gaussian",
    "elliptical_gauss",
    "elliptical_gaussian",
}


def _ordered_bounds(a, b):
    a = float(a)
    b = float(b)
    return (a, b) if a <= b else (b, a)


def weights_init(module, R_m, b_min, b_max):
    if not isinstance(module, nn.Linear):
        return
    with torch.no_grad():
        b_min, b_max = _ordered_bounds(b_min, b_max)
        if module.in_features == 1:
            module.weight[0, 0] = 1.0
            if module.weight.shape[0] > 1:
                module.weight[1:, :] = 1.0
            if module.bias.shape[0] > 1:
                module.bias[:] = torch.empty(module.bias.shape[0]).uniform_(b_min, b_max)
        else:
            nn.init.uniform_(module.weight, a=-R_m, b=R_m)
            nn.init.uniform_(module.bias, a=-R_m, b=R_m)


def _is_empty(value):
    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


def _optional_number(value, default=0.0):
    return default if _is_empty(value) else value


def _optional_int(value, default=0):
    return default if _is_empty(value) else int(value)


def _parse_local_rep_args(args, kwargs):
    cfg = {
        "M_noise": 0,
        "b1_noise": [],
        "b2_noise": [],
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "r_min": [],
        "r_max": [],
        "b1_min": [],
        "b1_max": [],
        "b2_min": [],
        "b2_max": [],
        "width_min": [],
        "width_max": [],
        "height_min": [],
        "height_max": [],
        "peak_min": [],
        "peak_max": [],
        "v_min": [],
        "v_max": [],
        "K_min": [],
        "K_max": [],
        "R_m_for_init": None,
        "af": None,
        "Shape": [],
        "device": None,
    }

    variant = None
    if args:
        if len(args) == 15:
            variant = "ex43"
            keys = [
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "r_min",
                "r_max",
                "b1_min",
                "b1_max",
                "b2_min",
                "b2_max",
                "K_min",
                "K_max",
                "R_m_for_init",
                "af",
                "device",
            ]
        elif len(args) == 19:
            variant = "ex44"
            keys = [
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "r_min",
                "r_max",
                "b1_min",
                "b1_max",
                "b2_min",
                "b2_max",
                "peak_min",
                "peak_max",
                "v_min",
                "v_max",
                "K_min",
                "K_max",
                "R_m_for_init",
                "af",
                "device",
            ]
        elif len(args) == 24:
            variant = "shape"
            keys = [
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "r_min",
                "r_max",
                "b1_min",
                "b1_max",
                "b2_min",
                "b2_max",
                "width_min",
                "width_max",
                "height_min",
                "height_max",
                "peak_min",
                "peak_max",
                "v_min",
                "v_max",
                "K_min",
                "K_max",
                "R_m_for_init",
                "af",
                "Shape",
                "device",
            ]
        elif len(args) == 20:
            variant = "shape_kwargs"
            keys = [
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "r_min",
                "r_max",
                "b1_min",
                "b1_max",
                "b2_min",
                "b2_max",
                "width_min",
                "width_max",
                "height_min",
                "height_max",
                "peak_min",
                "peak_max",
                "v_min",
                "v_max",
                "K_min",
                "K_max",
            ]
        elif len(args) == 27:
            variant = "signed"
            keys = [
                "M_noise",
                "b1_noise",
                "b2_noise",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "r_min",
                "r_max",
                "b1_min",
                "b1_max",
                "b2_min",
                "b2_max",
                "width_min",
                "width_max",
                "height_min",
                "height_max",
                "peak_min",
                "peak_max",
                "v_min",
                "v_max",
                "K_min",
                "K_max",
                "R_m_for_init",
                "af",
                "Shape",
                "device",
            ]
        else:
            raise TypeError("Unsupported net_2d.local_rep signature")

        cfg.update(dict(zip(keys, args)))

    cfg.update(kwargs)

    if _is_empty(cfg["Shape"]) and cfg["af"] == "sigmoid" and variant in {"ex43", "ex44"}:
        cfg["Shape"] = "circle"

    return cfg


def _parse_pre_define_args(args, kwargs):
    if kwargs:
        raise TypeError("net_2d.pre_define only supports positional arguments")

    if len(args) == 18:
        (
            Nx,
            Ny,
            M,
            R_m,
            af,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            r_min,
            r_max,
            b1_min,
            b1_max,
            b2_min,
            b2_max,
            K_min,
            K_max,
            device,
        ) = args
        extra = {
            "width_min": [],
            "width_max": [],
            "height_min": [],
            "height_max": [],
            "peak_min": [],
            "peak_max": [],
            "v_min": [],
            "v_max": [],
            "Shape": "circle" if af == "sigmoid" else [],
        }
    elif len(args) == 22:
        (
            Nx,
            Ny,
            M,
            R_m,
            af,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            r_min,
            r_max,
            b1_min,
            b1_max,
            b2_min,
            b2_max,
            peak_min,
            peak_max,
            v_min,
            v_max,
            K_min,
            K_max,
            device,
        ) = args
        extra = {
            "width_min": [],
            "width_max": [],
            "height_min": [],
            "height_max": [],
            "peak_min": peak_min,
            "peak_max": peak_max,
            "v_min": v_min,
            "v_max": v_max,
            "Shape": "circle" if af == "sigmoid" else [],
        }
    elif len(args) == 27:
        (
            Nx,
            Ny,
            M,
            R_m,
            af,
            tau_x_min,
            tau_x_max,
            tau_y_min,
            tau_y_max,
            r_min,
            r_max,
            b1_min,
            b1_max,
            b2_min,
            b2_max,
            width_min,
            width_max,
            height_min,
            height_max,
            peak_min,
            peak_max,
            v_min,
            v_max,
            K_min,
            K_max,
            Shape,
            device,
        ) = args
        extra = {
            "width_min": width_min,
            "width_max": width_max,
            "height_min": height_min,
            "height_max": height_max,
            "peak_min": peak_min,
            "peak_max": peak_max,
            "v_min": v_min,
            "v_max": v_max,
            "Shape": Shape,
        }
    else:
        raise TypeError("Unsupported net_2d.pre_define signature")

    return {
        "Nx": Nx,
        "Ny": Ny,
        "M": M,
        "R_m": R_m,
        "af": af,
        "tau_x_min": tau_x_min,
        "tau_x_max": tau_x_max,
        "tau_y_min": tau_y_min,
        "tau_y_max": tau_y_max,
        "r_min": r_min,
        "r_max": r_max,
        "b1_min": b1_min,
        "b1_max": b1_max,
        "b2_min": b2_min,
        "b2_max": b2_max,
        "K_min": K_min,
        "K_max": K_max,
        "device": device,
        **extra,
    }


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, *args, **kwargs):
        super().__init__()
        cfg = _parse_local_rep_args(args, kwargs)
        original_af = cfg["af"]
        relu_shape_override = original_af in {"Relu", "relu"} and cfg["Shape"] in {
            "circle",
            "rec",
            "ellipsoid",
            "ellipse",
            "general",
            "noise",
        }
        if relu_shape_override:
            cfg = dict(cfg)
            cfg["af"] = "sigmoid"
        init_af = cfg["af"]

        self.device = cfg["device"]
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers = hidden_layers
        self.x_max = cfg["x_max"]
        self.x_min = cfg["x_min"]
        self.y_max = cfg["y_max"]
        self.y_min = cfg["y_min"]
        self.M = M
        self.M_noise = _optional_int(cfg["M_noise"])
        self.af = original_af
        self.Shape = cfg["Shape"]
        self.a = torch.tensor(
            [2.0 / (self.x_max - self.x_min), 2.0 / (self.y_max - self.y_min)]
        ).to(self.device)
        self.x_0 = torch.tensor(
            [(self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2]
        ).to(self.device)
        self.hidden_layer = nn.Sequential(nn.Linear(in_features, self.hidden_features, bias=True))
        self.hidden_layer_1 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))
        self.hidden_layer_2 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))

        r_min = _optional_number(cfg["r_min"])
        r_max = _optional_number(cfg["r_max"])
        b1_min = _optional_number(cfg["b1_min"])
        b1_max = _optional_number(cfg["b1_max"])
        b2_min = _optional_number(cfg["b2_min"])
        b2_max = _optional_number(cfg["b2_max"])
        width_min = _optional_number(cfg["width_min"])
        width_max = _optional_number(cfg["width_max"])
        height_min = _optional_number(cfg["height_min"])
        height_max = _optional_number(cfg["height_max"])
        peak_min = _optional_number(cfg["peak_min"])
        peak_max = _optional_number(cfg["peak_max"])
        v_min = _optional_number(cfg["v_min"])
        v_max = _optional_number(cfg["v_max"])
        K_min = _optional_number(cfg["K_min"])
        K_max = _optional_number(cfg["K_max"])
        R_m_for_init = _optional_number(cfg["R_m_for_init"])

        r_min, r_max = _ordered_bounds(r_min, r_max)
        b1_min, b1_max = _ordered_bounds(b1_min, b1_max)
        b2_min, b2_max = _ordered_bounds(b2_min, b2_max)
        width_min, width_max = _ordered_bounds(width_min, width_max)
        height_min, height_max = _ordered_bounds(height_min, height_max)
        peak_min, peak_max = _ordered_bounds(peak_min, peak_max)
        v_min, v_max = _ordered_bounds(v_min, v_max)
        K_min, K_max = _ordered_bounds(K_min, K_max)

        if init_af == "mix":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(self.hidden_layer[0], R_m=R_m_for_init, b_min=-R_m_for_init, b_max=R_m_for_init)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af == "Tanh":
            weights_init(self.hidden_layer[0], R_m=R_m_for_init, b_min=-R_m_for_init, b_max=R_m_for_init)
        elif init_af == "sigmoid" and self.Shape == "circle":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af == "sigmoid" and self.Shape == "rec":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.width = torch.empty(self.M, dtype=torch.float64).uniform_(width_min, width_max).to(self.device)
            self.height = torch.empty(self.M, dtype=torch.float64).uniform_(height_min, height_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af == "sigmoid" and self.Shape in {"ellipsoid", "ellipse"}:
            self.center_x = torch.empty(self.M, dtype=torch.float64).uniform_(b1_min, b1_max).to(self.device)
            self.center_y = torch.empty(self.M, dtype=torch.float64).uniform_(b2_min, b2_max).to(self.device)
            self.axis_a = torch.empty(self.M, dtype=torch.float64).uniform_(width_min, width_max).to(self.device)
            self.axis_b = torch.empty(self.M, dtype=torch.float64).uniform_(height_min, height_max).to(self.device)
            self.theta_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af == "sigmoid" and self.Shape == "general":
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af == "sigmoid" and self.Shape == "noise":
            noise_width = max(self.M_noise, 1)
            self.hidden_layer_1 = nn.Sequential(nn.Linear(1, noise_width, bias=True))
            self.hidden_layer_2 = nn.Sequential(nn.Linear(1, noise_width, bias=True))
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            if not _is_empty(cfg["b1_noise"]):
                self.hidden_layer_1[0].bias.data = torch.tensor(cfg["b1_noise"], dtype=torch.float64).to(self.device)
            if not _is_empty(cfg["b2_noise"]):
                self.hidden_layer_2[0].bias.data = torch.tensor(cfg["b2_noise"], dtype=torch.float64).to(self.device)
            self.r_values = torch.empty(noise_width, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(noise_width, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af in {"Gauss", "continue_gauss"} and self.Shape in _ELLIPTIC_GAUSS_SHAPES:
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.axis_a = torch.empty(self.M, dtype=torch.float64).uniform_(width_min, width_max).to(self.device)
            self.axis_b = torch.empty(self.M, dtype=torch.float64).uniform_(height_min, height_max).to(self.device)
            self.theta_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
            self.gauss_amp = torch.empty(self.M, dtype=torch.float64).uniform_(peak_min, peak_max).to(self.device)
            self.gauss_scale_exp = torch.empty(self.M, dtype=torch.float64).uniform_(v_min, v_max).to(self.device)
            self.gauss_scale_sig = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif init_af in {"Gauss", "continue_gauss"}:
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
            self.gauss_amp = torch.empty(self.M, dtype=torch.float64).uniform_(peak_min, peak_max).to(self.device)
            self.gauss_scale_exp = torch.empty(self.M, dtype=torch.float64).uniform_(v_min, v_max).to(self.device)
            self.gauss_scale_sig = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        else:
            weights_init(self.hidden_layer[0], R_m=R_m_for_init, b_min=-R_m_for_init, b_max=R_m_for_init)


    def _resolve_shape(self, af, shape):
        if not _is_empty(shape):
            return shape

        shape = getattr(self, "Shape", None)
        if _is_empty(shape) and af in {"sigmoid", "Relu", "relu"}:
            if hasattr(self, "width") and hasattr(self, "height"):
                shape = "rec"
            elif getattr(self, "M_noise", 0):
                shape = "noise"
            else:
                shape = "circle"
            self.Shape = shape
        return shape

    def _move_tensor_attrs(self, *names):
        for name in names:
            if hasattr(self, name):
                value = getattr(self, name)
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to(self.device))

    def _rectangle_metric(self, x):
        self._move_tensor_attrs("width", "height")
        y1 = self.hidden_layer_1(x[..., 0:1])
        y2 = self.hidden_layer_2(x[..., 1:2])
        width = torch.clamp(self.width.repeat(y1.shape[0], 1), min=1e-12)
        height = torch.clamp(self.height.repeat(y2.shape[0], 1), min=1e-12)
        return torch.maximum(2.0 * torch.abs(y1) / width, 2.0 * torch.abs(y2) / height)

    def _ellipse_metric(self, x):
        self._move_tensor_attrs("center_x", "center_y", "axis_a", "axis_b", "theta_values")
        center_x = self.center_x.repeat(x.shape[0], 1)
        center_y = self.center_y.repeat(x.shape[0], 1)
        axis_a = torch.clamp(self.axis_a.repeat(x.shape[0], 1), min=1e-12)
        axis_b = torch.clamp(self.axis_b.repeat(x.shape[0], 1), min=1e-12)
        theta = self.theta_values.repeat(x.shape[0], 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x1 = x[..., 0:1] - center_x
        x2 = x[..., 1:2] - center_y
        x_rot = x1 * cos_theta + x2 * sin_theta
        y_rot = -x1 * sin_theta + x2 * cos_theta
        metric_sq = (x_rot / axis_a) ** 2 + (y_rot / axis_b) ** 2
        metric = torch.sqrt(metric_sq)
        return metric, metric_sq

    def _elliptic_gauss_metric(self, x):
        self._move_tensor_attrs("axis_a", "axis_b", "theta_values")
        x1 = self.hidden_layer_1(x[..., 0:1])
        x2 = self.hidden_layer_2(x[..., 1:2])
        theta = self.theta_values.repeat(x1.shape[0], 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = x1 * cos_theta + x2 * sin_theta
        y_rot = -x1 * sin_theta + x2 * cos_theta
        axis_a = torch.clamp(self.axis_a.repeat(x1.shape[0], 1), min=1e-12)
        axis_b = torch.clamp(self.axis_b.repeat(x2.shape[0], 1), min=1e-12)
        metric_sq = (x_rot / axis_a) ** 2 + (y_rot / axis_b) ** 2
        metric = torch.sqrt(metric_sq)
        return metric, metric_sq

    def forward(self, x, af, shape=None, signed_distances=None):
        shape = self._resolve_shape(af, shape)
        self.device = x.device
        self._move_tensor_attrs("a", "x_0", "K", "r_values", "gauss_amp", "gauss_scale_exp", "gauss_scale_sig")

        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)
        if af == "Tanh":
            return torch.tanh(y)
        if af == "sigmoid" and shape == "circle":
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            r_hat = torch.sqrt(y1**2 + y2**2)
            r = self.r_values.repeat(y1.shape[0], 1)
            K = self.K.repeat(y1.shape[0], 1)
            return torch.sigmoid(K * (r - r_hat))
        if af == "sigmoid" and shape == "rec":
            metric = self._rectangle_metric(x)
            K = self.K.repeat(metric.shape[0], 1)
            return torch.sigmoid(K * (1.0 - metric))
        if af == "sigmoid" and shape in {"ellipsoid", "ellipse"}:
            metric, _ = self._ellipse_metric(x)
            K = self.K.repeat(metric.shape[0], 1)
            return torch.sigmoid(K * (1.0 - metric))
        if af == "sigmoid" and shape == "general":
            if _is_empty(signed_distances):
                raise ValueError("signed_distances is required for Shape='general'")
            if not isinstance(signed_distances, torch.Tensor):
                signed_distances = torch.tensor(signed_distances, dtype=torch.float64, device=self.device)
            else:
                signed_distances = signed_distances.to(self.device)
            K = self.K.repeat(y.shape[0], 1)
            return torch.sigmoid(K * signed_distances)
        if af == "sigmoid" and shape == "noise":
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            r_hat = torch.sqrt(y1**2 + y2**2)
            r = self.r_values.repeat(y1.shape[0], 1)
            K = self.K.repeat(y1.shape[0], 1)
            return torch.sigmoid(K * (r - r_hat))
        if af in {"Relu", "relu"} and shape == "rec":
            metric = self._rectangle_metric(x)
            K = self.K.repeat(metric.shape[0], 1)
            return torch.relu(K * (1.0 - metric))
        if af in {"Relu", "relu"} and shape in {"ellipsoid", "ellipse"}:
            metric, _ = self._ellipse_metric(x)
            K = self.K.repeat(metric.shape[0], 1)
            return torch.relu(K * (1.0 - metric))
        if af == "Gauss":
            if shape in _ELLIPTIC_GAUSS_SHAPES:
                _, metric_sq = self._elliptic_gauss_metric(x)
                gauss_amp_b = self.gauss_amp.repeat(x.shape[0], 1)
                gauss_scale_exp_b = self.gauss_scale_exp.repeat(x.shape[0], 1)
                gauss_scale_sig_b = self.gauss_scale_sig.repeat(x.shape[0], 1)
                return gauss_amp_b * torch.exp(gauss_scale_exp_b * metric_sq) * torch.sigmoid(
                    gauss_scale_sig_b * (1.0 - metric_sq)
                )
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            r_hat_sq = y1**2 + y2**2
            r = self.r_values.repeat(y1.shape[0], 1)
            gauss_amp_b = self.gauss_amp.repeat(y1.shape[0], 1)
            gauss_scale_exp_b = self.gauss_scale_exp.repeat(y1.shape[0], 1)
            gauss_scale_sig_b = self.gauss_scale_sig.repeat(y1.shape[0], 1)
            return gauss_amp_b * torch.exp(gauss_scale_exp_b * r_hat_sq) * torch.sigmoid(
                gauss_scale_sig_b * (r**2 - r_hat_sq)
            )
        if af == "continue_gauss":
            if shape in _ELLIPTIC_GAUSS_SHAPES:
                _, metric_sq = self._elliptic_gauss_metric(x)
                gauss_amp_b = self.gauss_amp.repeat(x.shape[0], 1)
                gauss_scale_exp_b = self.gauss_scale_exp.repeat(x.shape[0], 1)
                return gauss_amp_b * torch.exp(gauss_scale_exp_b * metric_sq)
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            r_hat_sq = y1**2 + y2**2
            gauss_amp_b = self.gauss_amp.repeat(y1.shape[0], 1)
            gauss_scale_exp_b = self.gauss_scale_exp.repeat(y1.shape[0], 1)
            return gauss_amp_b * torch.exp(gauss_scale_exp_b * r_hat_sq)
        if af == "sin":
            return torch.sin(y)
        if af == "Relu":
            y1 = self.hidden_layer_1(y[..., 0:1])
            y2 = self.hidden_layer_2(y[..., 1:2])
            r_hat = torch.sqrt(y1**2 + y2**2)
            return torch.relu(0.4 - r_hat) ** 3
        else:
            return torch.sin(y)*torch.cos(y)


def pre_define(*args, **kwargs):
    cfg = _parse_pre_define_args(args, kwargs)
    models = []
    for k in range(cfg["Nx"]):
        model_for_x = []
        x_min = (cfg["tau_x_max"] - cfg["tau_x_min"]) / cfg["Nx"] * k + cfg["tau_x_min"]
        x_max = (cfg["tau_x_max"] - cfg["tau_x_min"]) / cfg["Nx"] * (k + 1) + cfg["tau_x_min"]
        for n in range(cfg["Ny"]):
            y_min = (cfg["tau_y_max"] - cfg["tau_y_min"]) / cfg["Ny"] * n + cfg["tau_y_min"]
            y_max = (cfg["tau_y_max"] - cfg["tau_y_min"]) / cfg["Ny"] * (n + 1) + cfg["tau_y_min"]
            model = local_rep(
                in_features=2,
                out_features=2,
                hidden_layers=1,
                M=cfg["M"],
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                r_min=cfg["r_min"],
                r_max=cfg["r_max"],
                b1_min=cfg["b1_min"],
                b1_max=cfg["b1_max"],
                b2_min=cfg["b2_min"],
                b2_max=cfg["b2_max"],
                width_min=cfg["width_min"],
                width_max=cfg["width_max"],
                height_min=cfg["height_min"],
                height_max=cfg["height_max"],
                peak_min=cfg["peak_min"],
                peak_max=cfg["peak_max"],
                v_min=cfg["v_min"],
                v_max=cfg["v_max"],
                K_min=cfg["K_min"],
                K_max=cfg["K_max"],
                R_m_for_init=cfg["R_m"],
                af=cfg["af"],
                Shape=cfg["Shape"],
                device=cfg["device"],
            ).to(cfg["device"])
            model = model.double().to(cfg["device"])
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
        models.append(model_for_x)
    return models


__all__ = ["weights_init", "local_rep", "pre_define"]
