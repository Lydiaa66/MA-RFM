import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

def detect_sharp_corners(points, angle_threshold=np.pi/3):
    """
    检测尖锐角点
    angle_threshold: 角度阈值，小于此角度认为是尖点
    """
    n = len(points)
    angles = []
    corner_indices = []
    
    for i in range(n):
        p1 = points[(i-1) % n]
        p2 = points[i]
        p3 = points[(i+1) % n]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        angles.append(angle)
        
        if angle < angle_threshold:
            corner_indices.append(i)
    
    return corner_indices, np.array(angles)

def adaptive_smooth_preserving_corners(points, corner_indices, smooth_factor=0.3):
    """
    自适应平滑，保持尖点
    """
    smoothed_points = points.copy()
    n = len(points)
    
    for i in range(n):
        if i not in corner_indices:
            prev_idx = (i-1) % n
            next_idx = (i+1) % n
            
            min_dist_to_corner = float('inf')
            for corner_idx in corner_indices:
                dist = min(abs(i - corner_idx), n - abs(i - corner_idx))
                min_dist_to_corner = min(min_dist_to_corner, dist)
            
            local_smooth = smooth_factor * (min_dist_to_corner / max(min_dist_to_corner, 3))
            
            smoothed_points[i] = (1 - local_smooth) * points[i] + \
                               local_smooth * 0.5 * (points[prev_idx] + points[next_idx])
    
    return smoothed_points

def gaussian_smooth_preserving_corners(points, corner_indices, sigma=1.0):
    """
    使用高斯滤波平滑，但保护尖点
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    weights = np.ones(len(points))
    for corner_idx in corner_indices:
        for i in range(len(points)):
            dist = min(abs(i - corner_idx), len(points) - abs(i - corner_idx))
            if dist <= 2:
                weights[i] = 0.1
    
    x_smooth = gaussian_filter1d(x_coords, sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(y_coords, sigma=sigma, mode='wrap')
    
    smoothed_points = np.column_stack([
        weights * x_coords + (1 - weights) * x_smooth,
        weights * y_coords + (1 - weights) * y_smooth
    ])
    
    return smoothed_points

def preprocess_contour(points, method='adaptive', **kwargs):
    """
    轮廓预处理主函数
    
    Parameters:
    - points: 输入轮廓点
    - method: 'adaptive', 'gaussian', 'simple_average'
    """
    corner_indices, angles = detect_sharp_corners(points, 
                                                kwargs.get('angle_threshold', np.pi/3))
    
    if method == 'adaptive':
        return adaptive_smooth_preserving_corners(points, corner_indices, 
                                                kwargs.get('smooth_factor', 0.3))
    elif method == 'gaussian':
        return gaussian_smooth_preserving_corners(points, corner_indices,
                                                kwargs.get('sigma', 1.0))
    elif method == 'simple_average':
        smoothed = points.copy()
        n = len(points)
        for i in range(n):
            if i not in corner_indices:
                smoothed[i] = 0.5 * (points[(i-1) % n] + points[(i+1) % n])
        return smoothed
    else:
        return points

def enhanced_offset_batch(points, s, distances, preprocess=True, preprocess_method='adaptive'):
    """
    增强版偏移函数，包含预处理
    """
    if preprocess:
        processed_points = preprocess_contour(points, method=preprocess_method)
    else:
        processed_points = points
    
    tck, u = splprep([processed_points[:, 0], processed_points[:, 1]], s=s, per=True)
    u_new = np.linspace(0, 1, len(points), endpoint=False)
    
    der1 = np.array(splev(u_new, tck, der=1)).T
    tangent_norm = np.linalg.norm(der1, axis=1)
    unit_tangent = der1 / tangent_norm[:, np.newaxis]
    unit_normal = np.c_[-unit_tangent[:, 1], unit_tangent[:, 0]]
    
    smooth_points = np.array(splev(u_new, tck)).T
    
    distances = np.atleast_1d(distances)
    if len(distances) == 1:
        return smooth_points + distances[0] * unit_normal
    else:
        return smooth_points[np.newaxis, :, :] + distances[:, np.newaxis, np.newaxis] * unit_normal[np.newaxis, :, :]
