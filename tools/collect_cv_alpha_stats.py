#!/usr/bin/env python3
"""Collect cluster-wise CV statistics for Examples 4.3--4.9.

This script reproduces the shape-detection stage used in the paper and prints
the cluster-wise coefficient-of-variation values stored in the detection
results. It also emits compact LaTeX-ready rows with an explicit CV column.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ma_rfm_cv_alpha_stats_mpl")

import matplotlib
import numpy as np

matplotlib.use("Agg")


REPO_ROOT = Path("/data1/hxw-b25/MA-RFM")
OUT_DIR = Path("/tmp/ma_rfm_cv_alpha_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BD2 = _load_module("bound_detect_2d_stats", REPO_ROOT / "main_code/2d/bound_detect.py")
BD3 = _load_module("bound_detect_3d_stats", REPO_ROOT / "main_code/3d/bound_detect.py")


def _load_final_field(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    if arr.dtype == object:
        if arr.ndim == 0:
            return _load_final_field_from_object(arr.item())
        return _load_final_field_from_object(arr[-1])
    if arr.ndim >= 3:
        return np.asarray(arr[-1], dtype=float)
    return np.asarray(arr, dtype=float)


def _load_final_field_from_object(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.ndim > 0:
            return _load_final_field_from_object(obj[-1])
        return np.asarray(obj, dtype=float)
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Empty object field.")
        return _load_final_field_from_object(obj[-1])
    return np.asarray(obj, dtype=float)


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


@dataclass(frozen=True)
class Example2D:
    name: str
    data_dir: Path
    domain: tuple[float, float, float, float]
    para_abs: float
    para_grad: float
    eps_cluster: float
    mini_samples: int
    s_num_file: str
    grad_file: str | None = "g_S.npy"
    min_cluster_size: int | None = None


@dataclass(frozen=True)
class Example3D:
    name: str
    data_dir: Path
    domain: tuple[float, float, float, float, float, float]
    para_abs: float
    para_grad: float
    eps_cluster: float
    mini_samples: int
    s_num_file: str
    grad_file: str | None = "g_S.npy"
    boundary_margin: int | tuple[int, int, int] = 0


EXAMPLES_2D: list[Example2D] = [
    Example2D(
        name="Ex 4.3",
        data_dir=REPO_ROOT / "Ex4.3/Noise/noise=5%",
        domain=(0.0, 1.0, 0.0, 1.0),
        para_abs=3.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store.npy",
    ),
    Example2D(
        name="Ex 4.4",
        data_dir=REPO_ROOT / "Ex4.4/Noise/noise=5%",
        domain=(-0.3, 0.3, -0.3, 0.3),
        para_abs=100.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store.npy",
    ),
    Example2D(
        name="Ex 4.5",
        data_dir=REPO_ROOT / "Ex4.5/Noise/noise=5%",
        domain=(0.0, 1.0, 0.0, 1.0),
        para_abs=3.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store.npy",
    ),
    Example2D(
        name="Ex 4.6",
        data_dir=REPO_ROOT / "Ex4.6/noise=5%",
        domain=(0.0, 1.0, 0.0, 1.0),
        para_abs=100.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store_ada.npy",
    ),
    Example2D(
        name="Ex 4.7",
        data_dir=REPO_ROOT / "Ex4.7/noise=5%",
        domain=(0.0, 1.0, 0.0, 1.0),
        para_abs=3.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num.npy",
        grad_file=None,
    ),
]


EXAMPLES_3D: list[Example3D] = [
    Example3D(
        name="Ex 4.8",
        data_dir=REPO_ROOT / "Ex4.8/noise=5%",
        domain=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        para_abs=100.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store.npy",
        boundary_margin=(5, 5, 0),
    ),
    Example3D(
        name="Ex 4.9",
        data_dir=REPO_ROOT / "Ex4.9/noise=5%",
        domain=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        para_abs=3.0,
        para_grad=3.0,
        eps_cluster=5.0,
        mini_samples=20,
        s_num_file="S_num_store.npy",
    ),
]


def _run_2d(cfg: Example2D) -> list[dict[str, Any]]:
    grad = [] if cfg.grad_file is None else np.load(cfg.data_dir / cfg.grad_file, allow_pickle=True)
    s_num = _load_final_field(cfg.data_dir / cfg.s_num_file)
    x_min, x_max, y_min, y_max = cfg.domain
    x = np.linspace(x_min, x_max, s_num.shape[0])
    y = np.linspace(y_min, y_max, s_num.shape[1])
    X, Y = np.meshgrid(x, y)
    return BD2.detect_shape(
        grad,
        X.T,
        Y.T,
        x_min,
        x_max,
        y_min,
        y_max,
        cfg.para_abs,
        cfg.para_grad,
        cfg.eps_cluster,
        cfg.mini_samples,
        s_num,
        output_directory=str(OUT_DIR),
        output_filename=f"{cfg.name.replace(' ', '_').lower()}_detect.png",
        show_colorbar=False,
        min_cluster_size=cfg.min_cluster_size,
        eps_mode="grid_scaled",
    )


def _run_3d(cfg: Example3D) -> list[dict[str, Any]]:
    grad = [] if cfg.grad_file is None else np.load(cfg.data_dir / cfg.grad_file, allow_pickle=True)
    s_num = _load_final_field(cfg.data_dir / cfg.s_num_file)
    x_min, x_max, y_min, y_max, z_min, z_max = cfg.domain
    nx, ny, nz = s_num.shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return BD3.detect_shape_3d(
        grad,
        X,
        Y,
        Z,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        cfg.para_abs,
        cfg.para_grad,
        cfg.eps_cluster,
        cfg.mini_samples,
        s_num,
        output_directory=str(OUT_DIR),
        output_filename=f"{cfg.name.replace(' ', '_').lower()}_detect.png",
        show_colorbar=False,
        eps_mode="grid_scaled",
        boundary_margin=cfg.boundary_margin,
    )


def _print_summary(name: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{name}")
    if not results:
        print("  no clusters detected")
        return
    for item in results:
        shape = item.get("shape", "-")
        profile = item.get("profile", "-")
        cv = _format_float(item.get("cv", np.nan), 3)
        res_rect = _format_float(item.get("res_rect", np.nan), 4)
        res_ellip = _format_float(item.get("res_ellip", np.nan), 4)
        res_tori = _format_float(item.get("res_tori", np.nan), 4)
        print(
            f"  cluster {int(item.get('cluster_id', -1)) + 1}: "
            f"shape={shape}, profile={profile}, cv_alpha={cv}, "
            f"res_rect={res_rect}, res_ellip={res_ellip}, res_tori={res_tori}"
        )


def main() -> None:
    all_rows: list[tuple[str, int, str, str]] = []

    for cfg in EXAMPLES_2D:
        results = _run_2d(cfg)
        _print_summary(cfg.name, results)
        for item in results:
            all_rows.append(
                (
                    cfg.name,
                    int(item.get("cluster_id", -1)) + 1,
                    _format_float(item.get("cv", np.nan), 3),
                    str(item.get("profile", "-")),
                )
            )

    for cfg in EXAMPLES_3D:
        results = _run_3d(cfg)
        _print_summary(cfg.name, results)
        for item in results:
            all_rows.append(
                (
                    cfg.name,
                    int(item.get("cluster_id", -1)) + 1,
                    _format_float(item.get("cv", np.nan), 3),
                    str(item.get("profile", "-")),
                )
            )

    print("\nLaTeX-ready cv_alpha rows:")
    print(r"{\emph{Ex}} & Cluster $k$ & $\mathrm{CV}_k$ & $\sigma_k$ \\")
    for example, cluster, cv, profile in all_rows:
        print(f"{example} & {cluster} & {cv} & {profile} \\\\")


if __name__ == "__main__":
    main()
