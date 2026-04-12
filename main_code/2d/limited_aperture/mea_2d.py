import numpy as np
def p_b_circle(center_x=0, center_y=0, radius=0.5,alpha=1, n=100,temp=True):
    """
    Uniformly sample n points on a circle centered at (center_x, center_y) with the given radius.
    """
    theta = np.linspace(0, 2*alpha*np.pi, n, endpoint=temp)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.stack((x, y), axis=1)
