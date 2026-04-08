import numpy as np
from scipy.interpolate import splprep, splev

def simple_offset_batch(points, s,distances):
    """
    直接支持单个distance或distance数组的版本
    """
    from scipy.interpolate import splprep, splev
    
    tck, u = splprep([points[:, 0], points[:, 1]], s=s, per=True)
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
