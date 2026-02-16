import numpy as np
def p_b_circle(center_x=0, center_y=0, radius=0.5,alpha=1, n=100,temp=True):
    """
    在以 (center_x, center_y) 为圆心，radius 为半径的圆周上均匀采样 n 个点
    """
    theta = np.linspace(0, 2*alpha*np.pi, n, endpoint=temp)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.stack((x, y), axis=1)
