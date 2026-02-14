import numpy as np
from scipy.spatial import cKDTree
import matplotlib.path as mpath
import torch
import smooth
def generate_signed_distance(x_train,kidney_boundary,alpha,scale_factor, M):
    """
    生成网格点到轮廓的符号距离。
    
    参数:
        x_train (torch.Tensor): 网格点，形状为 (N, 2)。
        kidney_boundary (np.ndarray): 肾脏边界点，形状为 (L, 2)。
        alpha (float): 平滑参数。
        scale_factor (float): 缩放因子，用于平滑边界。
        device: PyTorch 设备。
        M: 重复次数，用于扩展 signed_distances 的维度。

    返回:
        torch.Tensor: 符号距离，形状为 (N, M)。
    """
    if isinstance(x_train, torch.Tensor):
        grid_points = x_train.cpu().detach().numpy()
    else:
        grid_points = x_train
    bound_num_scaled = smooth.simple_offset_batch(kidney_boundary, alpha,scale_factor)
    signed_distances = np.zeros((grid_points.shape[0], M), dtype=np.float64)
    for i in range(M):
        path = mpath.Path(bound_num_scaled[i,:,:]) 
        are_inside = path.contains_points(grid_points,radius=-1e-9)
        signed_distances[:,i:i+1] = (np.where(are_inside, 1, -1)).reshape(-1,1)
    return bound_num_scaled,signed_distances


    
        
    
    
    
    
    
    
