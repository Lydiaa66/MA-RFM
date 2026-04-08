import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

def get_contour_points_from_implicit(implicit_func, func_args, x_range, y_range, grid_points=300):
    """
    从一个隐式函数 f(x, y) = 0 中提取边界点。

    参数:
        implicit_func (callable): 接受 x, y 和其他参数的隐式函数。
        func_args (dict): 传递给隐式函数的参数 (例如 h, k, s)。
        x_range (tuple): (min, max) x 范围。
        y_range (tuple): (min, max) y 范围。
        grid_points (int): 用于查找等高线的网格分辨率。

    返回:
        np.ndarray: 边界点的 (N, 2) 数组。
    """
    # 在高分辨率网格上计算隐式函数的值
    x_grid = np.linspace(x_range[0], x_range[1], grid_points)
    y_grid = np.linspace(y_range[0], y_range[1], grid_points)
    X_contour, Y_contour = np.meshgrid(x_grid, y_grid)

    Z_contour = implicit_func(X_contour, Y_contour, **func_args)

    # 使用 contour 查找值为 0 的等高线
    cs = plt.contour(X_contour, Y_contour, Z_contour, levels=[0])
    
    # 从结果中提取最长的轮廓路径
    if not cs.allsegs[0]:
        raise ValueError("无法在 level=0 找到等高线。请尝试调整函数参数或范围。")

    longest_contour_idx = np.argmax([len(seg) for seg in cs.allsegs[0]])
    contour_points = cs.allsegs[0][longest_contour_idx]

    plt.close()  # 关闭 contour 自动生成的图形
    return contour_points

def generate_soft_boundary(x_grid, y_grid, contour_points, K=500):
    """
    为由任意点集定义的边界生成一个平滑的示性函数。(此函数无变化)
    """
    boundary_tree = cKDTree(contour_points)
    path = mpath.Path(contour_points)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    distances, _ = boundary_tree.query(grid_points, k=1)
    are_inside = path.contains_points(grid_points)
    signed_distances = np.where(are_inside, -distances, distances)
    z_flat = 1 / (1 + np.exp(K * signed_distances))
    return z_flat.reshape(x_grid.shape)

# --- 演示：使用您提供的隐式函数 ---




