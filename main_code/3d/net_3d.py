import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


def _is_empty(value):
    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


def _ordered_bounds(a, b):
    a = float(a)
    b = float(b)
    return (a, b) if a <= b else (b, a)


def _ordered_bounds_if_set(a, b):
    if _is_empty(a) or _is_empty(b):
        return a, b
    return _ordered_bounds(a, b)


def _is_axis_triplet(value):
    return isinstance(value, (list, tuple)) and len(value) == 3


def _ordered_axis_bounds_if_set(a, b):
    if _is_empty(a) or _is_empty(b):
        return a, b
    if not (_is_axis_triplet(a) and _is_axis_triplet(b)):
        raise ValueError("Ellipsoid axis bounds must be length-3 tuples/lists.")
    mins = []
    maxs = []
    for ai, bi in zip(a, b):
        lo, hi = _ordered_bounds(ai, bi)
        mins.append(lo)
        maxs.append(hi)
    return tuple(mins), tuple(maxs)


def weights_init(m, R_m, b_min, b_max):
    if not isinstance(m, nn.Linear):
        return
    with torch.no_grad():
        b_min, b_max = _ordered_bounds(b_min, b_max)
        if m.in_features == 1:
            m.weight[0, 0] = 1.0
            if m.weight.shape[0] > 1:
                m.weight[1:, :] = 1.0
            if m.bias.shape[0] > 1:
                m.bias[:] = torch.empty(m.bias.shape[0]).uniform_(b_min, b_max)
        else:
            nn.init.uniform_(m.weight, a=-R_m, b=R_m)
            nn.init.uniform_(m.bias, a=-R_m, b=R_m)


class local_rep(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers,
        M,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        r_min,
        r_max,
        R_min=None,
        R_max=None,
        b1_min=None,
        b1_max=None,
        b2_min=None,
        b2_max=None,
        b3_min=None,
        b3_max=None,
        peak_min=None,
        peak_max=None,
        v_min=None,
        v_max=None,
        K_min=None,
        K_max=None,
        R_m_for_init=None,
        af=None,
        Shape=None,
        torus_axis=2,
        device=None,
    ):
        super(local_rep, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_min = z_min
        self.z_max = z_max
        self.M = M
        self.af = af
        self.Shape = Shape
        self.torus_axis = int(torus_axis)
        self.a = torch.tensor(
            [2.0 / (x_max - x_min), 2.0 / (y_max - y_min), 2.0 / (z_max - z_min)]
        )
        self.x_0 = torch.tensor(
            [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features, self.hidden_features, bias=True)
        )
        self.hidden_layer_1 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))
        self.hidden_layer_2 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))
        self.hidden_layer_3 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))

        b1_min = -R_m_for_init if _is_empty(b1_min) else b1_min
        b1_max = R_m_for_init if _is_empty(b1_max) else b1_max
        b2_min = -R_m_for_init if _is_empty(b2_min) else b2_min
        b2_max = R_m_for_init if _is_empty(b2_max) else b2_max
        b3_min = -R_m_for_init if _is_empty(b3_min) else b3_min
        b3_max = R_m_for_init if _is_empty(b3_max) else b3_max
        b1_min, b1_max = _ordered_bounds(b1_min, b1_max)
        b2_min, b2_max = _ordered_bounds(b2_min, b2_max)
        b3_min, b3_max = _ordered_bounds(b3_min, b3_max)
        ellipsoid_shape = self.Shape == "ellipsoid" and self.af in {"sigmoid", "relu", "Gauss"}
        if ellipsoid_shape:
            axis_min, axis_max = _ordered_axis_bounds_if_set(r_min, r_max)
        else:
            r_min, r_max = _ordered_bounds_if_set(r_min, r_max)
            axis_min, axis_max = None, None
        R_min, R_max = _ordered_bounds_if_set(R_min, R_max)
        K_min, K_max = _ordered_bounds_if_set(K_min, K_max)
        v_min, v_max = _ordered_bounds_if_set(v_min, v_max)

        if self.af == "mix":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(
                self.hidden_layer[0],
                R_m=R_m_for_init,
                b_min=-R_m_for_init,
                b_max=R_m_for_init,
            )
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max)
        elif self.af == "sigmoid" and self.Shape == "sweet":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(self.hidden_layer_3[0], R_m=R_m_for_init, b_min=b3_min, b_max=b3_max)
            self.R_values = torch.empty(self.M, dtype=torch.float64).uniform_(R_min, R_max)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max)
        elif self.af == "sigmoid" or self.af == "relu":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(self.hidden_layer_3[0], R_m=R_m_for_init, b_min=b3_min, b_max=b3_max)
            if ellipsoid_shape:
                self.axis1_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[0], axis_max[0])
                self.axis2_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[1], axis_max[1])
                self.axis3_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[2], axis_max[2])
            else:
                self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max)
        elif self.af == "Gauss":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(self.hidden_layer_3[0], R_m=R_m_for_init, b_min=b3_min, b_max=b3_max)
            if ellipsoid_shape:
                self.axis1_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[0], axis_max[0])
                self.axis2_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[1], axis_max[1])
                self.axis3_values = torch.empty(self.M, dtype=torch.float64).uniform_(axis_min[2], axis_max[2])
            else:
                self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max)
            self.gauss_scale_exp = torch.empty(self.M, dtype=torch.float64).uniform_(v_min, v_max)
        else:
            weights_init(
                self.hidden_layer[0],
                R_m=R_m_for_init,
                b_min=-R_m_for_init,
                b_max=R_m_for_init,
            )

    def _ellipsoid_metric(self, x):
        self.axis1_values = self.axis1_values.to(self.device)
        self.axis2_values = self.axis2_values.to(self.device)
        self.axis3_values = self.axis3_values.to(self.device)
        y1 = self.hidden_layer_1(x[..., 0:1])
        y2 = self.hidden_layer_2(x[..., 1:2])
        y3 = self.hidden_layer_3(x[..., 2:3])
        axis1 = torch.clamp(self.axis1_values.repeat(y1.shape[0], 1), min=1e-12)
        axis2 = torch.clamp(self.axis2_values.repeat(y1.shape[0], 1), min=1e-12)
        axis3 = torch.clamp(self.axis3_values.repeat(y1.shape[0], 1), min=1e-12)
        metric_sq = (y1 / axis1) ** 2 + (y2 / axis2) ** 2 + (y3 / axis3) ** 2
        metric = torch.sqrt(metric_sq)
        return metric, metric_sq

    def _sweet_metric(self, x):
        self.r_values = self.r_values.to(self.device)
        self.R_values = self.R_values.to(self.device)
        y1 = self.hidden_layer_1(x[..., 0:1])
        y2 = self.hidden_layer_2(x[..., 1:2])
        y3 = self.hidden_layer_3(x[..., 2:3])
        if self.torus_axis == 0:
            radial_1, radial_2, axial = y2, y3, y1
        elif self.torus_axis == 1:
            radial_1, radial_2, axial = y1, y3, y2
        else:
            radial_1, radial_2, axial = y1, y2, y3
        rho = torch.sqrt(radial_1**2 + radial_2**2)
        r = torch.clamp(self.r_values.repeat(y1.shape[0], 1), min=1e-12)
        R = self.R_values.repeat(y1.shape[0], 1)
        metric_sq = ((rho - R) / r) ** 2 + (axial / r) ** 2
        metric = torch.sqrt(metric_sq)
        return metric, metric_sq

    def forward(self, x, af, shape_or_device=None, device=None):
        if device is None and shape_or_device is not None and not isinstance(
            shape_or_device, (str, list, tuple)
        ):
            Shape = None
            device = shape_or_device
        else:
            Shape = shape_or_device

        if Shape in (None, []):
            Shape = getattr(self, "Shape", None)
        if device is None:
            device = self.device

        self.device = device
        self.a = self.a.to(self.device)
        self.x_0 = self.x_0.to(self.device)
        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)

        if af == 0 or af == "Tanh":
            return torch.tanh(y)

        if af == "sigmoid" and Shape == "sweet":
            self.K = self.K.to(self.device)
            metric, _ = self._sweet_metric(x)
            K = self.K.repeat(metric.shape[0], 1)
            return torch.sigmoid(K * (1.0 - metric))

        if af == "sigmoid" or af == "relu":
            self.K = self.K.to(self.device)
            if Shape == "ellipsoid":
                metric, _ = self._ellipsoid_metric(x)
                K = self.K.repeat(metric.shape[0], 1)
                if af == "sigmoid":
                    return torch.sigmoid(K * (1.0 - metric))
                return torch.relu(K * (1.0 - metric))
            self.r_values = self.r_values.to(self.device)
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            y3 = self.hidden_layer_3(x[..., 2:3])
            r_hat = torch.sqrt(y1**2 + y2**2 + y3**2)
            r = self.r_values.repeat(y1.shape[0], 1)
            K = self.K.repeat(y1.shape[0], 1)
            if af == "sigmoid":
                return torch.sigmoid(K * (r - r_hat))
            return torch.relu(K * (r - r_hat))

        if af == "Gauss":
            self.gauss_scale_exp = self.gauss_scale_exp.to(self.device)
            if Shape == "ellipsoid":
                _, metric_sq = self._ellipsoid_metric(x)
                gauss_scale_exp_b = self.gauss_scale_exp.repeat(metric_sq.shape[0], 1)
                return torch.exp(gauss_scale_exp_b * metric_sq)
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[..., 1:2])
            y3 = self.hidden_layer_3(x[..., 2:3])
            r_hat_sq = y1**2 + y2**2 + y3**2
            gauss_scale_exp_b = self.gauss_scale_exp.repeat(y1.shape[0], 1)
            return torch.exp(gauss_scale_exp_b * r_hat_sq)

        if af == "sin":
            return torch.sin(y)

        return torch.sin(y) * torch.cos(y)
