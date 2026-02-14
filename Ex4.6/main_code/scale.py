import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def offset_boundary(points, distance):
    """
    沿法线方向偏移（放大/缩小）一个闭合轮廓。

    参数:
        points (np.ndarray): 形状为 (N, 2) 的原始轮廓点。
        distance (float): 偏移距离。正值为放大, 负值为缩小。

    返回:
        np.ndarray: 偏移后的新轮廓点。
    """
    points_prev = np.roll(points, 1, axis=0)
    points_next = np.roll(points, -1, axis=0)

    tangents = points_next - points_prev
    
    norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    norms[norms == 0] = 1.0
    unit_tangents = tangents / norms

    normals = np.c_[-unit_tangents[:, 1], unit_tangents[:, 0]]

    new_points = points + distance * normals
    
    return new_points

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

